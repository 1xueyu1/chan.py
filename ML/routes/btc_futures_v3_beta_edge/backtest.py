from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ML.routes.btc_futures_v1.data import SYMBOL, parse_utc
from ML.shared.json_utils import json_safe

from .bundle import V3BetaEdgeModelBundle, bsp_family_for_frame
from .dataset import DEFAULT_BACKTEST_DIR, DEFAULT_DATASET_PATH, DEFAULT_MODEL_DIR, ROUTE_NAME, build_btc_futures_v3_beta_edge_dataset


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


def _score_dataset(dataset: pd.DataFrame, model_dir: Path) -> pd.DataFrame:
    buy_model = V3BetaEdgeModelBundle.load(model_dir / "buy_model.pkl")
    sell_model = V3BetaEdgeModelBundle.load(model_dir / "sell_model.pkl")
    out = buy_model.prepare_features(dataset)
    out = sell_model.prepare_features(out)
    score_cols = pd.DataFrame(
        {
            "tp_first_probability": np.nan,
            "probability": np.nan,
            "ml_bundle_threshold": np.nan,
            "ml_model_key": "",
        },
        index=out.index,
    )
    out = pd.concat([out.copy(), score_cols], axis=1)
    buy_mask = out["is_buy"].astype(bool)
    if bool(buy_mask.any()):
        rows = out.loc[buy_mask]
        comp = buy_model.predict_components(rows)
        for col in comp.columns:
            out.loc[buy_mask, col] = comp[col].to_numpy()
        if hasattr(buy_model, "threshold_for_rows"):
            out.loc[buy_mask, "ml_bundle_threshold"] = buy_model.threshold_for_rows(rows)
        else:
            out.loc[buy_mask, "ml_bundle_threshold"] = buy_model.threshold
        if hasattr(buy_model, "predict_model_key"):
            out.loc[buy_mask, "ml_model_key"] = buy_model.predict_model_key(rows)
        else:
            out.loc[buy_mask, "ml_model_key"] = "beta_tp_first_buy"
    sell_mask = ~buy_mask
    if bool(sell_mask.any()):
        rows = out.loc[sell_mask]
        comp = sell_model.predict_components(rows)
        for col in comp.columns:
            out.loc[sell_mask, col] = comp[col].to_numpy()
        if hasattr(sell_model, "threshold_for_rows"):
            out.loc[sell_mask, "ml_bundle_threshold"] = sell_model.threshold_for_rows(rows)
        else:
            out.loc[sell_mask, "ml_bundle_threshold"] = sell_model.threshold
        if hasattr(sell_model, "predict_model_key"):
            out.loc[sell_mask, "ml_model_key"] = sell_model.predict_model_key(rows)
        else:
            out.loc[sell_mask, "ml_model_key"] = "beta_tp_first_sell"
    return out


def _assign_frequency_floor_policy(
    scored: pd.DataFrame,
    *,
    enabled: bool,
    side: str,
    low_threshold: float,
    mid_threshold: float,
    high_threshold: float | None,
    low_exposure: float,
    mid_exposure: float,
    high_exposure: float,
    default_exposure: float,
) -> pd.DataFrame:
    out = scored.copy()
    out["confidence_tier"] = "high"
    out["policy_threshold"] = pd.to_numeric(out["ml_bundle_threshold"], errors="coerce")
    out["policy_exposure"] = float(default_exposure)
    out["frequency_floor_enabled"] = False

    if not enabled or out.empty:
        return out

    prob = pd.to_numeric(out["probability"], errors="coerce")
    is_buy = out["is_buy"].astype(bool)
    side_mask = pd.Series(True, index=out.index)
    if side.lower() in {"sell", "short"}:
        side_mask = ~is_buy
    elif side.lower() in {"buy", "long"}:
        side_mask = is_buy

    high_thr = float(high_threshold) if high_threshold is not None else pd.to_numeric(out["ml_bundle_threshold"], errors="coerce")
    if isinstance(high_thr, float):
        high_mask = side_mask & (prob >= high_thr)
    else:
        high_mask = side_mask & (prob >= high_thr)
    mid_mask = side_mask & ~high_mask & (prob >= float(mid_threshold))
    low_mask = side_mask & ~high_mask & ~mid_mask & (prob >= float(low_threshold))
    floor_mask = low_mask | mid_mask

    out.loc[side_mask, "policy_threshold"] = float(low_threshold)
    out.loc[high_mask, "confidence_tier"] = "high"
    out.loc[high_mask, "policy_exposure"] = float(high_exposure)
    out.loc[mid_mask, "confidence_tier"] = "mid"
    out.loc[mid_mask, "policy_exposure"] = float(mid_exposure)
    out.loc[low_mask, "confidence_tier"] = "low"
    out.loc[low_mask, "policy_exposure"] = float(low_exposure)
    out.loc[floor_mask, "frequency_floor_enabled"] = True
    return out


def _qualified_candidates(scored: pd.DataFrame) -> pd.DataFrame:
    if scored.empty:
        return scored.copy()
    threshold_col = "policy_threshold" if "policy_threshold" in scored.columns else "ml_bundle_threshold"
    selected = scored.loc[pd.to_numeric(scored["probability"], errors="coerce") >= pd.to_numeric(scored[threshold_col], errors="coerce")].copy()
    if selected.empty:
        return selected
    selected = selected.sort_values(["beta_event_id", "entry_time", "probability"], ascending=[True, True, False])
    return selected.drop_duplicates(subset=["beta_event_id"], keep="first").reset_index(drop=True)


def _write_summaries(trades: pd.DataFrame, out_dir: Path) -> None:
    if trades.empty:
        return
    frame = trades.copy()
    frame["exit_time"] = pd.to_datetime(frame["exit_time"], utc=True, errors="coerce")
    frame["month"] = frame["exit_time"].dt.tz_convert(None).dt.to_period("M").astype(str)
    frame["year"] = frame["exit_time"].dt.year
    for keys, name in (
        (["month"], "monthly_summary.csv"),
        (["side"], "side_summary.csv"),
        (["v3_structure_pool", "side"], "pool_side_summary.csv"),
        (["beta_entry_delay_minutes", "side"], "delay_side_summary.csv"),
        (["year", "v3_structure_pool", "side"], "year_pool_side_summary.csv"),
        (["confidence_tier"], "confidence_tier_summary.csv"),
        (["confidence_tier", "side"], "confidence_tier_side_summary.csv"),
    ):
        missing = [key for key in keys if key not in frame.columns]
        if missing:
            continue
        summary = frame.groupby(keys, dropna=False).agg(
            trades=("pnl", "size"),
            wins=("pnl", lambda s: int((s > 0).sum())),
            pnl=("pnl", "sum"),
            avg_return=("net_return", "mean"),
            avg_probability=("probability", "mean"),
            avg_delay=("beta_entry_delay_minutes", "mean"),
        )
        summary["win_rate"] = summary["wins"] / summary["trades"]
        summary["profit_factor"] = frame.groupby(keys, dropna=False)["pnl"].apply(_profit_factor)
        summary.to_csv(out_dir / name)


def run_btc_futures_v3_beta_edge_backtest(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    output_dir: str | Path = DEFAULT_BACKTEST_DIR,
    begin_time: Any | None = "2026-01-01",
    end_time: Any | None = None,
    initial_cash: float = 100000.0,
    stake_fraction: float = 1.0,
    frequency_floor_enabled: bool = False,
    frequency_floor_side: str = "sell",
    frequency_floor_low_threshold: float = 0.50,
    frequency_floor_mid_threshold: float = 0.55,
    frequency_floor_high_threshold: float | None = None,
    frequency_floor_low_exposure: float = 0.05,
    frequency_floor_mid_exposure: float = 0.15,
    frequency_floor_high_exposure: float = 1.0,
    bsp_families: str | None = None,
) -> dict[str, Any]:
    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        build_btc_futures_v3_beta_edge_dataset(output_path=dataset_file)
    data = pd.read_parquet(dataset_file)
    for col in ("exec_time", "entry_time", "exit_time", "beta_signal_available_time"):
        data[col] = pd.to_datetime(data[col], utc=True, errors="coerce")
    data = data.dropna(subset=["label_tp_first", "entry_time", "exit_time"]).sort_values("entry_time").reset_index(drop=True)
    begin = parse_utc(begin_time)
    end = parse_utc(end_time, end_of_day=True)
    if begin is not None:
        data = data.loc[data["entry_time"] >= begin]
    if end is not None:
        data = data.loc[data["entry_time"] <= end]
    allowed_families = {x.strip() for x in str(bsp_families or "").split(",") if x.strip()}
    if allowed_families:
        families = bsp_family_for_frame(data)
        data = data.loc[families.astype(str).isin(allowed_families)].copy()
    if data.empty:
        raise ValueError("empty backtest window")
    if not bool((data["entry_time"] >= data["exec_time"] + pd.Timedelta(minutes=15)).all()):
        raise RuntimeError("lookahead audit failed during beta backtest")

    scored = _score_dataset(data, Path(model_dir))
    scored = _assign_frequency_floor_policy(
        scored,
        enabled=bool(frequency_floor_enabled),
        side=str(frequency_floor_side),
        low_threshold=float(frequency_floor_low_threshold),
        mid_threshold=float(frequency_floor_mid_threshold),
        high_threshold=frequency_floor_high_threshold,
        low_exposure=float(frequency_floor_low_exposure),
        mid_exposure=float(frequency_floor_mid_exposure),
        high_exposure=float(frequency_floor_high_exposure),
        default_exposure=float(stake_fraction),
    )
    qualified = _qualified_candidates(scored)
    trades: list[dict[str, Any]] = []
    equity = float(initial_cash)
    last_exit = pd.Timestamp.min.tz_localize("UTC")
    equity_rows: list[dict[str, Any]] = [{"time": data["entry_time"].min(), "equity": equity, "drawdown": 0.0}]
    if not qualified.empty:
        qualified = qualified.sort_values(["entry_time", "probability"], ascending=[True, False]).reset_index(drop=True)
        for _, row in qualified.iterrows():
            entry_time = pd.to_datetime(row["entry_time"], utc=True)
            exit_time = pd.to_datetime(row["exit_time"], utc=True)
            if entry_time <= last_exit:
                continue
            exposure = float(row.get("policy_exposure", stake_fraction))
            base_return = float(row["net_return"])
            equity_before = equity
            pnl = equity * exposure * base_return
            equity += pnl
            last_exit = exit_time
            trade = row.to_dict()
            trade["side"] = "long" if bool(row["is_buy"]) else "short"
            trade["threshold"] = float(row["ml_bundle_threshold"])
            trade["policy_threshold"] = float(row.get("policy_threshold", row["ml_bundle_threshold"]))
            trade["confidence_tier"] = str(row.get("confidence_tier", "high"))
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
    pnl = pd.to_numeric(trades_df["pnl"], errors="coerce") if not trades_df.empty else pd.Series(dtype="float64")
    returns = pd.to_numeric(trades_df["net_return"], errors="coerce") if not trades_df.empty else pd.Series(dtype="float64")
    wins = int((pnl > 0).sum()) if not trades_df.empty else 0
    days = float((data["entry_time"].max() - data["entry_time"].min()).total_seconds() / 86400.0)
    tier_summary: dict[str, Any] = {}
    if not trades_df.empty and "confidence_tier" in trades_df.columns:
        for tier, group in trades_df.groupby("confidence_tier", dropna=False):
            tier_pnl = pd.to_numeric(group["pnl"], errors="coerce")
            tier_returns = pd.to_numeric(group["net_return"], errors="coerce")
            tier_wins = int((tier_pnl > 0).sum())
            tier_summary[str(tier)] = {
                "trades": int(len(group)),
                "wins": tier_wins,
                "win_rate": float(tier_wins / len(group)) if len(group) else float("nan"),
                "avg_signal_return": float(tier_returns.mean()) if len(tier_returns) else float("nan"),
                "pnl": float(tier_pnl.sum()) if len(tier_pnl) else 0.0,
                "profit_factor": _profit_factor(tier_pnl) if len(tier_pnl) else float("nan"),
                "avg_exposure": float(pd.to_numeric(group["exposure"], errors="coerce").mean()),
            }
    metrics = {
        "route": ROUTE_NAME,
        "symbol": SYMBOL,
        "begin_time": str(begin) if begin is not None else None,
        "end_time": str(end) if end is not None else None,
        "initial_cash": float(initial_cash),
        "final_equity": float(equity),
        "total_return": float(equity / float(initial_cash) - 1.0),
        "candidate_rows": int(len(scored)),
        "candidate_events": int(scored["beta_event_id"].nunique()),
        "bsp_families": sorted(allowed_families) if allowed_families else "all",
        "qualified_candidates": int(len(qualified)),
        "trades": int(len(trades_df)),
        "days_per_trade": float(days / len(trades_df)) if len(trades_df) else float("inf"),
        "wins": wins,
        "losses": int(len(trades_df) - wins),
        "win_rate": float(wins / len(trades_df)) if len(trades_df) else float("nan"),
        "avg_signal_return": float(returns.mean()) if len(returns) else float("nan"),
        "avg_trade_pnl": float(pnl.mean()) if len(pnl) else float("nan"),
        "profit_factor": _profit_factor(pnl) if len(pnl) else float("nan"),
        "max_drawdown": _max_drawdown(equity_df.set_index("time")["equity"]),
        "frequency_floor": {
            "enabled": bool(frequency_floor_enabled),
            "side": str(frequency_floor_side),
            "low_threshold": float(frequency_floor_low_threshold),
            "mid_threshold": float(frequency_floor_mid_threshold),
            "high_threshold": float(frequency_floor_high_threshold) if frequency_floor_high_threshold is not None else None,
            "low_exposure": float(frequency_floor_low_exposure),
            "mid_exposure": float(frequency_floor_mid_exposure),
            "high_exposure": float(frequency_floor_high_exposure),
            "tier_summary": tier_summary,
        },
        "lookahead_check": "one sample per BSP event; entry_time >= exec_time + 15min",
    }
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out_dir / "scored_candidates.csv", index=False)
    qualified.to_csv(out_dir / "qualified_candidates.csv", index=False)
    trades_df.to_csv(out_dir / "executed_trades.csv", index=False)
    equity_df.to_csv(out_dir / "equity_curve.csv", index=False)
    _write_summaries(trades_df, out_dir)
    (out_dir / "backtest_metrics.json").write_text(json.dumps(json_safe(metrics), ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Backtest BTC futures v3 beta edge route")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_BACKTEST_DIR))
    parser.add_argument("--begin-time", default="2026-01-01")
    parser.add_argument("--end-time", default="")
    parser.add_argument("--initial-cash", type=float, default=100000.0)
    parser.add_argument("--stake-fraction", type=float, default=1.0)
    parser.add_argument("--frequency-floor", action="store_true")
    parser.add_argument("--frequency-floor-side", default="sell", choices=["sell", "short", "buy", "long", "all"])
    parser.add_argument("--frequency-floor-low-threshold", type=float, default=0.50)
    parser.add_argument("--frequency-floor-mid-threshold", type=float, default=0.55)
    parser.add_argument("--frequency-floor-high-threshold", type=float, default=None)
    parser.add_argument("--frequency-floor-low-exposure", type=float, default=0.05)
    parser.add_argument("--frequency-floor-mid-exposure", type=float, default=0.15)
    parser.add_argument("--frequency-floor-high-exposure", type=float, default=1.0)
    parser.add_argument("--bsp-families", default="", help="comma separated BSP families to include, e.g. 2")
    args = parser.parse_args()
    result = run_btc_futures_v3_beta_edge_backtest(
        dataset_path=args.dataset,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        begin_time=args.begin_time,
        end_time=args.end_time or None,
        initial_cash=args.initial_cash,
        stake_fraction=args.stake_fraction,
        frequency_floor_enabled=args.frequency_floor,
        frequency_floor_side=args.frequency_floor_side,
        frequency_floor_low_threshold=args.frequency_floor_low_threshold,
        frequency_floor_mid_threshold=args.frequency_floor_mid_threshold,
        frequency_floor_high_threshold=args.frequency_floor_high_threshold,
        frequency_floor_low_exposure=args.frequency_floor_low_exposure,
        frequency_floor_mid_exposure=args.frequency_floor_mid_exposure,
        frequency_floor_high_exposure=args.frequency_floor_high_exposure,
        bsp_families=args.bsp_families or None,
    )
    print(json.dumps(json_safe(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
