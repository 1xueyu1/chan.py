from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ML.routes.btc_futures_v1.data import SYMBOL, parse_utc
from ML.shared.json_utils import json_safe
from ML.shared.threshold_policy import CapabilityV2ThresholdPolicy, add_policy_columns

from .bundle import V3AlphaModelBundle
from .dataset import DEFAULT_BACKTEST_DIR, DEFAULT_DATASET_PATH, DEFAULT_MODEL_DIR, ROUTE_NAME, build_btc_futures_v3_alpha_dataset


def _profit_factor(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    wins = float(values.loc[values > 0].sum())
    losses = float(-values.loc[values < 0].sum())
    if losses <= 0:
        return float("inf") if wins > 0 else float("nan")
    return float(wins / losses)


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    return float((equity / equity.cummax() - 1.0).min())


def _load_bundle(path: Path) -> V3AlphaModelBundle:
    return V3AlphaModelBundle.load(path)


def _score_dataset(dataset: pd.DataFrame, model_dir: Path) -> pd.DataFrame:
    buy_model = _load_bundle(model_dir / "buy_model.pkl")
    sell_model = _load_bundle(model_dir / "sell_model.pkl")
    out = buy_model.prepare_features(dataset)
    out = sell_model.prepare_features(out)
    out["probability"] = np.nan
    out["ml_primary_probability"] = np.nan
    out["ml_bundle_threshold"] = np.nan
    out["ml_model_key"] = ""

    buy_mask = out["is_buy"].astype(bool)
    if bool(buy_mask.any()):
        rows = out.loc[buy_mask]
        prob = buy_model.predict_proba(rows)
        out.loc[buy_mask, "probability"] = prob
        out.loc[buy_mask, "ml_primary_probability"] = prob
        out.loc[buy_mask, "ml_bundle_threshold"] = buy_model.threshold_for_rows(rows)
        out.loc[buy_mask, "ml_model_key"] = buy_model.predict_model_key(rows)
    sell_mask = ~buy_mask
    if bool(sell_mask.any()):
        rows = out.loc[sell_mask]
        prob = sell_model.predict_proba(rows)
        out.loc[sell_mask, "probability"] = prob
        out.loc[sell_mask, "ml_primary_probability"] = prob
        out.loc[sell_mask, "ml_bundle_threshold"] = sell_model.threshold_for_rows(rows)
        out.loc[sell_mask, "ml_model_key"] = sell_model.predict_model_key(rows)

    out["ml_primary_margin"] = pd.to_numeric(out["probability"], errors="coerce") - pd.to_numeric(
        out["ml_bundle_threshold"], errors="coerce"
    )
    return out


def _write_group_summaries(trades_df: pd.DataFrame, out_dir: Path) -> None:
    if trades_df.empty:
        return
    trades = trades_df.copy()
    trades["exit_time"] = pd.to_datetime(trades["exit_time"], utc=True, errors="coerce")
    trades["month"] = trades["exit_time"].dt.tz_convert(None).dt.to_period("M").astype(str)
    trades["year"] = trades["exit_time"].dt.year
    for cols, name in (
        (["month"], "monthly_summary.csv"),
        (["v3_structure_pool"], "pool_summary.csv"),
        (["v3_market_state"], "market_state_summary.csv"),
        (["v3_structure_pool", "side"], "pool_side_summary.csv"),
        (["year", "v3_structure_pool", "side"], "year_pool_side_summary.csv"),
    ):
        summary = trades.groupby(cols, dropna=False).agg(
            trades=("pnl", "size"),
            wins=("pnl", lambda s: int((s > 0).sum())),
            pnl=("pnl", "sum"),
            avg_return=("net_return", "mean"),
            avg_probability=("probability", "mean"),
        )
        summary["win_rate"] = summary["wins"] / summary["trades"]
        summary["profit_factor"] = trades.groupby(cols, dropna=False)["pnl"].apply(_profit_factor)
        summary.to_csv(out_dir / name)


def run_btc_futures_v3_alpha_backtest(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    output_dir: str | Path = DEFAULT_BACKTEST_DIR,
    begin_time: Any | None = "2026-01-01",
    end_time: Any | None = None,
    initial_cash: float = 100000.0,
    stake_fraction: float = 1.0,
) -> dict[str, Any]:
    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        build_btc_futures_v3_alpha_dataset(output_path=dataset_file)
    dataset = pd.read_parquet(dataset_file)
    for col in ("exec_time", "entry_time", "exit_time", "signal_available_time"):
        if col in dataset.columns:
            dataset[col] = pd.to_datetime(dataset[col], utc=True, errors="coerce")
    dataset = dataset.dropna(subset=["label", "entry_time", "exit_time"]).sort_values("entry_time").reset_index(drop=True)
    begin = parse_utc(begin_time)
    end = parse_utc(end_time, end_of_day=True)
    if begin is not None:
        dataset = dataset.loc[dataset["entry_time"] >= begin]
    if end is not None:
        dataset = dataset.loc[dataset["entry_time"] <= end]
    if dataset.empty:
        raise ValueError("empty backtest window")

    model_path = Path(model_dir)
    scored = _score_dataset(dataset, model_path)
    policy_payload = json.loads((model_path / "threshold_policy.json").read_text(encoding="utf-8"))
    policy = CapabilityV2ThresholdPolicy(policy_payload)
    scored = add_policy_columns(scored)

    qualified_rows: list[dict[str, Any]] = []
    for _, row in scored.iterrows():
        side = "buy" if bool(row["is_buy"]) else "sell"
        decision = policy.decide(SYMBOL, side, float(row["probability"]), row)
        if not bool(decision.qualified):
            continue
        trade = row.to_dict()
        trade["side"] = "long" if bool(row["is_buy"]) else "short"
        trade["threshold"] = float(decision.threshold)
        trade["market_state"] = str(decision.market_state)
        trade["quality_score"] = float(decision.quality_score)
        trade["threshold_reason"] = str(decision.reason)
        trade["confidence_tier"] = "fixed"
        trade["confidence_score"] = 1.0
        trade["stake_multiplier"] = 1.0
        trade["leverage"] = 1.0
        qualified_rows.append(trade)

    qualified = pd.DataFrame(qualified_rows)
    trades: list[dict[str, Any]] = []
    equity = float(initial_cash)
    equity_rows: list[dict[str, Any]] = [{"time": dataset["entry_time"].min(), "equity": equity, "drawdown": 0.0}]
    last_exit = pd.Timestamp.min.tz_localize("UTC")
    if not qualified.empty:
        qualified = qualified.sort_values(["entry_time", "probability"], ascending=[True, False]).reset_index(drop=True)
        for _, row in qualified.iterrows():
            entry_time = pd.to_datetime(row["entry_time"], utc=True)
            exit_time = pd.to_datetime(row["exit_time"], utc=True)
            exec_time = pd.to_datetime(row["exec_time"], utc=True)
            if entry_time < exec_time + pd.Timedelta(minutes=15):
                raise RuntimeError("lookahead audit failed during backtest")
            if entry_time <= last_exit:
                continue
            exposure = float(stake_fraction)
            base_return = float(row["net_return"])
            equity_before = equity
            pnl = equity * exposure * base_return
            equity += pnl
            last_exit = exit_time
            trade = dict(row)
            trade["equity_before"] = float(equity_before)
            trade["exposure"] = float(exposure)
            trade["pnl"] = float(pnl)
            trade["equity_after"] = float(equity)
            trades.append(trade)
            eq = pd.Series([r["equity"] for r in equity_rows] + [equity])
            equity_rows.append({"time": exit_time, "equity": float(equity), "drawdown": float(equity / max(eq.max(), equity) - 1.0)})

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows)
    equity_df["time"] = pd.to_datetime(equity_df["time"], utc=True, errors="coerce")
    equity_df = equity_df.sort_values("time")
    returns = pd.to_numeric(trades_df["net_return"], errors="coerce") if not trades_df.empty else pd.Series(dtype="float64")
    pnl = pd.to_numeric(trades_df["pnl"], errors="coerce") if not trades_df.empty else pd.Series(dtype="float64")
    wins = int((pnl > 0).sum()) if not trades_df.empty else 0
    metrics = {
        "route": ROUTE_NAME,
        "symbol": SYMBOL,
        "begin_time": str(begin) if begin is not None else None,
        "end_time": str(end) if end is not None else None,
        "initial_cash": float(initial_cash),
        "final_equity": float(equity),
        "total_return": float(equity / float(initial_cash) - 1.0),
        "trades": int(len(trades_df)),
        "candidate_signals": int(len(scored)),
        "qualified_signals": int(len(qualified)) if not qualified.empty else 0,
        "wins": wins,
        "losses": int(len(trades_df) - wins),
        "win_rate": float(wins / len(trades_df)) if len(trades_df) else float("nan"),
        "avg_signal_return": float(returns.mean()) if len(returns) else float("nan"),
        "avg_trade_pnl": float(pnl.mean()) if len(pnl) else float("nan"),
        "profit_factor": _profit_factor(pnl) if len(pnl) else float("nan"),
        "max_drawdown": _max_drawdown(equity_df.set_index("time")["equity"]),
        "dynamic_sizing": False,
        "lookahead_check": "entry_time >= exec_time + 15min",
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trades_df.to_csv(out_dir / "executed_trades.csv", index=False)
    scored.to_csv(out_dir / "scored_events.csv", index=False)
    equity_df.to_csv(out_dir / "equity_curve.csv", index=False)
    _write_group_summaries(trades_df, out_dir)
    (out_dir / "backtest_metrics.json").write_text(json.dumps(json_safe(metrics), ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Backtest BTC futures v3 alpha route")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_BACKTEST_DIR))
    parser.add_argument("--begin-time", default="2026-01-01")
    parser.add_argument("--end-time", default="")
    parser.add_argument("--initial-cash", type=float, default=100000.0)
    parser.add_argument("--stake-fraction", type=float, default=1.0)
    args = parser.parse_args()
    result = run_btc_futures_v3_alpha_backtest(
        dataset_path=args.dataset,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        begin_time=args.begin_time,
        end_time=args.end_time or None,
        initial_cash=args.initial_cash,
        stake_fraction=args.stake_fraction,
    )
    print(json.dumps(json_safe(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
