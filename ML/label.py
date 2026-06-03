from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from Backtest.types import RawBSPEvent
from .features import datetime_index_to_ns, normalize_bars


@dataclass(frozen=True)
class LabelConfig:
    horizon_bars: int = 96
    atr_period: int = 14
    take_profit_atr: float = 1.5
    stop_loss_atr: float = 1.0
    min_return: float = 0.0
    fee: float = 0.0004
    slippage: float = 0.0001
    execution_mode: str = "next_bar_open"
    same_bar_policy: str = "stop_first"


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=max(3, period // 3)).mean()


def _event_ns(events: Sequence[RawBSPEvent]) -> np.ndarray:
    return datetime_index_to_ns([event.exec_time for event in events])


def label_events(bars: pd.DataFrame, events: Sequence[RawBSPEvent], config: LabelConfig | None = None) -> pd.DataFrame:
    """Create first-hit TP/SL labels for Chan buy/sell events.

    Label equals 1 when the event reaches take-profit first or has a net return
    above ``min_return`` by horizon end. It equals 0 for stop/horizon failures.
    """
    cfg = config or LabelConfig()
    if cfg.execution_mode not in {"next_bar_open", "close"}:
        raise ValueError("execution_mode must be next_bar_open or close")
    if cfg.same_bar_policy not in {"stop_first", "take_profit_first"}:
        raise ValueError("same_bar_policy must be stop_first or take_profit_first")

    if not events:
        return pd.DataFrame()

    df = normalize_bars(bars)
    open_ = df["open"].to_numpy(dtype="float64", copy=False)
    high = df["high"].to_numpy(dtype="float64", copy=False)
    low = df["low"].to_numpy(dtype="float64", copy=False)
    close = df["close"].to_numpy(dtype="float64", copy=False)
    atr = _atr(df["high"], df["low"], df["close"], cfg.atr_period).to_numpy(dtype="float64", copy=False)
    times = datetime_index_to_ns(df.index)

    loc = np.searchsorted(times, _event_ns(events), side="right") - 1
    if cfg.execution_mode == "next_bar_open":
        entry_idx = loc + 1
    else:
        entry_idx = loc

    rows = []
    cost = 2.0 * (float(cfg.fee) + float(cfg.slippage))
    n = len(df)

    for event, idx in zip(events, entry_idx):
        direction = 1.0 if event.is_buy else -1.0
        row = {
            "label": np.nan,
            "net_return": np.nan,
            "mfe": np.nan,
            "mae": np.nan,
            "entry_time": pd.NaT,
            "entry_price": np.nan,
            "exit_time": pd.NaT,
            "exit_price": np.nan,
            "exit_reason": "no_future",
            "entry_bar_idx": int(idx),
        }

        if idx < 0 or idx >= n or not np.isfinite(atr[idx]) or atr[idx] <= 0:
            rows.append(row)
            continue

        entry_price = open_[idx] if cfg.execution_mode == "next_bar_open" else close[idx]
        future_start = idx if cfg.execution_mode == "next_bar_open" else idx + 1
        future_end = min(n, future_start + int(cfg.horizon_bars))
        if future_start >= future_end or not np.isfinite(entry_price) or entry_price <= 0:
            rows.append(row)
            continue

        tp_price = entry_price + direction * float(cfg.take_profit_atr) * atr[idx]
        sl_price = entry_price - direction * float(cfg.stop_loss_atr) * atr[idx]
        exit_idx = future_end - 1
        exit_price = close[exit_idx]
        exit_reason = "horizon"

        for j in range(future_start, future_end):
            if direction > 0:
                hit_tp = high[j] >= tp_price
                hit_sl = low[j] <= sl_price
            else:
                hit_tp = low[j] <= tp_price
                hit_sl = high[j] >= sl_price

            if hit_tp and hit_sl:
                exit_idx = j
                if cfg.same_bar_policy == "take_profit_first":
                    exit_price = tp_price
                    exit_reason = "take_profit"
                else:
                    exit_price = sl_price
                    exit_reason = "stop_loss"
                break
            if hit_tp:
                exit_idx = j
                exit_price = tp_price
                exit_reason = "take_profit"
                break
            if hit_sl:
                exit_idx = j
                exit_price = sl_price
                exit_reason = "stop_loss"
                break

        window_high = np.nanmax(high[future_start:future_end])
        window_low = np.nanmin(low[future_start:future_end])
        if direction > 0:
            mfe = (window_high - entry_price) / entry_price
            mae = (window_low - entry_price) / entry_price
        else:
            mfe = (entry_price - window_low) / entry_price
            mae = (entry_price - window_high) / entry_price

        net_return = direction * (exit_price - entry_price) / entry_price - cost
        label = 1.0 if exit_reason == "take_profit" or net_return > float(cfg.min_return) else 0.0
        row.update(
            {
                "label": label,
                "net_return": net_return,
                "mfe": mfe,
                "mae": mae,
                "entry_time": df.index[idx],
                "entry_price": entry_price,
                "exit_time": df.index[exit_idx],
                "exit_price": exit_price,
                "exit_reason": exit_reason,
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)
