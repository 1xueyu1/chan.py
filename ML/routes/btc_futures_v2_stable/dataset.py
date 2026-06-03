from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ML.routes.btc_futures_v1.data import DEFAULT_SOURCE_PATH, load_1m_futures_bars


ROUTE_NAME = "btc_futures_v2_stable"
DEFAULT_BASE_DATASET = Path("data/btc_futures_v1/btc_futures_v1_dataset.parquet")
DEFAULT_DATASET_PATH = Path("data/btc_futures_v2_stable/btc_futures_v2_stable_dataset.parquet")
DEFAULT_MODEL_DIR = Path("result/ml/btc_futures_v2_stable")
DEFAULT_BACKTEST_DIR = Path("result/btc_futures_v2_stable")


def _target_pct(row: pd.Series, mode: str, fixed_pct: float, atr_multiplier: float, min_pct: float, max_pct: float) -> float:
    if mode == "fixed":
        return float(fixed_pct)
    atr_pct = float(pd.to_numeric(row.get("bar_atr_pct_14", np.nan), errors="coerce"))
    if not np.isfinite(atr_pct) or atr_pct <= 0:
        return float(fixed_pct)
    return float(np.clip(float(atr_multiplier) * atr_pct, float(min_pct), float(max_pct)))


def relabel_dataset_with_1m_path(
    base_dataset: str | Path = DEFAULT_BASE_DATASET,
    source_path: str | Path = DEFAULT_SOURCE_PATH,
    output_path: str | Path = DEFAULT_DATASET_PATH,
    target_mode: str = "atr",
    fixed_pct: float = 0.01,
    atr_multiplier: float = 1.50,
    min_pct: float = 0.006,
    max_pct: float = 0.018,
    max_holding_minutes: int = 1440,
    fee_rate: float = 0.0004,
    slippage_rate: float = 0.0001,
    same_bar_policy: str = "stop_first",
    force: bool = False,
) -> pd.DataFrame:
    if target_mode not in {"fixed", "atr"}:
        raise ValueError("target_mode must be fixed or atr")
    if same_bar_policy not in {"stop_first", "take_profit_first"}:
        raise ValueError("same_bar_policy must be stop_first or take_profit_first")
    output = Path(output_path)
    if output.exists() and not force:
        return pd.read_parquet(output)

    dataset = pd.read_parquet(base_dataset)
    for col in ("exec_time", "entry_time", "exit_time", "signal_available_time"):
        if col in dataset.columns:
            dataset[col] = pd.to_datetime(dataset[col], utc=True, errors="coerce")
    dataset = dataset.dropna(subset=["exec_time", "entry_time"]).sort_values("entry_time").reset_index(drop=True)

    begin = dataset["entry_time"].min() - pd.Timedelta(days=2)
    end = dataset["entry_time"].max() + pd.Timedelta(days=2)
    bars = load_1m_futures_bars(source_path, begin_time=begin, end_time=end)
    times = pd.DatetimeIndex(pd.to_datetime(bars.index, utc=True))
    time_ns = times.view("int64")
    open_ = bars["open"].to_numpy(dtype="float64", copy=False)
    high = bars["high"].to_numpy(dtype="float64", copy=False)
    low = bars["low"].to_numpy(dtype="float64", copy=False)
    close = bars["close"].to_numpy(dtype="float64", copy=False)
    n = len(bars)
    cost = 2.0 * (float(fee_rate) + float(slippage_rate))

    label_rows: list[dict[str, Any]] = []
    for _, row in dataset.iterrows():
        entry_time = pd.to_datetime(row["entry_time"], utc=True)
        entry_pos = int(np.searchsorted(time_ns, entry_time.value, side="left"))
        is_buy = bool(row["is_buy"])
        direction = 1.0 if is_buy else -1.0
        tp_sl_pct = _target_pct(row, target_mode, fixed_pct, atr_multiplier, min_pct, max_pct)
        result = {
            "target_mode": target_mode,
            "target_pct": tp_sl_pct,
            "label": np.nan,
            "entry_time": entry_time,
            "entry_price": np.nan,
            "exit_time": pd.NaT,
            "exit_price": np.nan,
            "exit_reason": "no_future",
            "gross_return": np.nan,
            "net_return": np.nan,
            "mfe": np.nan,
            "mae": np.nan,
            "entry_bar_idx": entry_pos,
        }
        if entry_pos < 0 or entry_pos >= n:
            label_rows.append(result)
            continue
        entry_price = float(open_[entry_pos])
        if not np.isfinite(entry_price) or entry_price <= 0:
            label_rows.append(result)
            continue
        if is_buy:
            tp_price = entry_price * (1.0 + tp_sl_pct)
            sl_price = entry_price * (1.0 - tp_sl_pct)
        else:
            tp_price = entry_price * (1.0 - tp_sl_pct)
            sl_price = entry_price * (1.0 + tp_sl_pct)
        future_start = entry_pos
        future_end = min(n, entry_pos + max(1, int(max_holding_minutes)))
        exit_pos = future_end - 1
        exit_price = float(close[exit_pos])
        exit_reason = "timeout"
        for pos in range(future_start, future_end):
            if is_buy:
                hit_tp = bool(high[pos] >= tp_price)
                hit_sl = bool(low[pos] <= sl_price)
            else:
                hit_tp = bool(low[pos] <= tp_price)
                hit_sl = bool(high[pos] >= sl_price)
            if hit_tp and hit_sl:
                exit_pos = pos
                if same_bar_policy == "take_profit_first":
                    exit_price = float(tp_price)
                    exit_reason = "take_profit"
                else:
                    exit_price = float(sl_price)
                    exit_reason = "stop_loss"
                break
            if hit_tp:
                exit_pos = pos
                exit_price = float(tp_price)
                exit_reason = "take_profit"
                break
            if hit_sl:
                exit_pos = pos
                exit_price = float(sl_price)
                exit_reason = "stop_loss"
                break
        window_high = float(np.nanmax(high[future_start:future_end]))
        window_low = float(np.nanmin(low[future_start:future_end]))
        if is_buy:
            mfe = (window_high - entry_price) / entry_price
            mae = (window_low - entry_price) / entry_price
        else:
            mfe = (entry_price - window_low) / entry_price
            mae = (entry_price - window_high) / entry_price
        gross_return = direction * (float(exit_price) - entry_price) / entry_price
        net_return = gross_return - cost
        result.update(
            {
                "label": 1.0 if exit_reason == "take_profit" else 0.0,
                "entry_price": entry_price,
                "exit_time": times[exit_pos],
                "exit_price": float(exit_price),
                "exit_reason": exit_reason,
                "gross_return": float(gross_return),
                "net_return": float(net_return),
                "mfe": float(mfe),
                "mae": float(mae),
            }
        )
        label_rows.append(result)

    drop_cols = [
        "label",
        "entry_time",
        "entry_price",
        "exit_time",
        "exit_price",
        "exit_reason",
        "gross_return",
        "net_return",
        "mfe",
        "mae",
        "entry_bar_idx",
        "target_mode",
        "target_pct",
    ]
    dataset = dataset.drop(columns=[col for col in drop_cols if col in dataset.columns])
    labels = pd.DataFrame(label_rows)
    out = pd.concat([dataset.reset_index(drop=True), labels.reset_index(drop=True)], axis=1)
    out["route"] = ROUTE_NAME
    out["label_take_profit_pct"] = out["target_pct"].astype(float)
    out["label_stop_loss_pct"] = out["target_pct"].astype(float)
    out["label_max_holding_minutes"] = int(max_holding_minutes)
    out = out.dropna(subset=["label", "entry_time", "exit_time"]).reset_index(drop=True)
    if not bool((out["entry_time"] >= out["exec_time"] + pd.Timedelta(minutes=15)).all()):
        raise RuntimeError("lookahead audit failed: entry_time is earlier than closed signal bar")

    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output, index=False)
    metadata = {
        "route": ROUTE_NAME,
        "base_dataset": str(base_dataset),
        "source_path": str(source_path),
        "output_path": str(output),
        "rows": int(len(out)),
        "target_mode": target_mode,
        "fixed_pct": float(fixed_pct),
        "atr_multiplier": float(atr_multiplier),
        "min_pct": float(min_pct),
        "max_pct": float(max_pct),
        "max_holding_minutes": int(max_holding_minutes),
        "fee_rate": float(fee_rate),
        "slippage_rate": float(slippage_rate),
        "same_bar_policy": same_bar_policy,
        "label_rate": float(out["label"].mean()),
        "avg_net_return": float(out["net_return"].mean()),
        "lookahead_check": "entry_time >= exec_time + 15min",
    }
    output.with_name("btc_futures_v2_stable_dataset_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build BTC futures v2 stable relabelled dataset")
    parser.add_argument("--base-dataset", default=str(DEFAULT_BASE_DATASET))
    parser.add_argument("--source", default=str(DEFAULT_SOURCE_PATH))
    parser.add_argument("--output", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--target-mode", default="atr", choices=["fixed", "atr"])
    parser.add_argument("--fixed-pct", type=float, default=0.01)
    parser.add_argument("--atr-multiplier", type=float, default=1.50)
    parser.add_argument("--min-pct", type=float, default=0.006)
    parser.add_argument("--max-pct", type=float, default=0.018)
    parser.add_argument("--max-holding-minutes", type=int, default=1440)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    frame = relabel_dataset_with_1m_path(
        base_dataset=args.base_dataset,
        source_path=args.source,
        output_path=args.output,
        target_mode=args.target_mode,
        fixed_pct=args.fixed_pct,
        atr_multiplier=args.atr_multiplier,
        min_pct=args.min_pct,
        max_pct=args.max_pct,
        max_holding_minutes=args.max_holding_minutes,
        force=args.force,
    )
    print(f"built {ROUTE_NAME} dataset rows={len(frame)} output={args.output}")


if __name__ == "__main__":
    main()
