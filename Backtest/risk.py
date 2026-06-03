from __future__ import annotations

from math import sqrt

import numpy as np
import pandas as pd

from Common.CEnum import KL_TYPE

from .config import BacktestConfig
from .types import SignalMatrix


def _annualization_factor(kl_type: KL_TYPE) -> float:
    mapping = {
        KL_TYPE.K_1M: 365.0 * 24.0 * 60.0,
        KL_TYPE.K_3M: 365.0 * 24.0 * 20.0,
        KL_TYPE.K_5M: 365.0 * 24.0 * 12.0,
        KL_TYPE.K_10M: 365.0 * 24.0 * 6.0,
        KL_TYPE.K_15M: 365.0 * 24.0 * 4.0,
        KL_TYPE.K_30M: 365.0 * 24.0 * 2.0,
        KL_TYPE.K_60M: 365.0 * 24.0,
        KL_TYPE.K_DAY: 365.0,
        KL_TYPE.K_WEEK: 52.0,
        KL_TYPE.K_MON: 12.0,
    }
    return mapping.get(kl_type, 365.0 * 24.0 * 4.0)


def _atr_pct(bars: pd.DataFrame, period: int) -> pd.Series:
    high = bars["high"].astype(float)
    low = bars["low"].astype(float)
    close = bars["close"].astype(float)
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(period, min_periods=max(3, period // 3)).mean()
    return (atr / close).replace([np.inf, -np.inf], np.nan)


def build_risk_size_series(config: BacktestConfig, bars: pd.DataFrame) -> pd.Series:
    index = bars.index
    base_size = float(config.risk_base_position_size if config.risk_enabled else config.position_size)
    size = pd.Series(base_size, index=index, dtype="float64")

    if config.risk_enabled and config.risk_use_volatility_target:
        close = bars["close"].astype(float)
        realized_vol = close.pct_change().rolling(
            int(config.risk_vol_window),
            min_periods=max(8, int(config.risk_vol_window) // 4),
        ).std() * sqrt(_annualization_factor(config.kl_type))
        vol_scaled = base_size * float(config.risk_target_annual_vol) / (realized_vol + 1e-12)
        size = vol_scaled.clip(float(config.risk_min_position_size), float(config.risk_max_position_size))
        size = size.fillna(base_size)
    elif config.risk_enabled:
        size = size.clip(float(config.risk_min_position_size), float(config.risk_max_position_size))

    if config.risk_enabled and config.risk_max_atr_pct > 0:
        atr_pct = _atr_pct(bars, int(config.risk_atr_period))
        size = size.mask(atr_pct > float(config.risk_max_atr_pct), 0.0)

    return size.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _empty_risk_series(index: pd.Index) -> tuple[pd.Series, pd.Series]:
    return (
        pd.Series(np.zeros(len(index), dtype=bool), index=index),
        pd.Series(np.zeros(len(index), dtype=bool), index=index),
    )


def apply_risk_layer(config: BacktestConfig, bars: pd.DataFrame, signal_matrix: SignalMatrix) -> SignalMatrix:
    index = pd.DatetimeIndex(bars.index)
    size = build_risk_size_series(config, bars).reindex(index).fillna(0.0)
    if signal_matrix.size_multiplier is not None:
        multiplier = signal_matrix.size_multiplier.reindex(index).fillna(1.0).astype(float)
        size = (size * multiplier).clip(0.0, float(config.risk_max_position_size if config.risk_enabled else 1.0))
    if not config.risk_enabled:
        signal_matrix.size = size
        blocked, forced = _empty_risk_series(index)
        signal_matrix.risk_blocked_entries = blocked
        signal_matrix.risk_forced_exits = forced
        return signal_matrix

    n = len(index)
    price = (bars["open"] if config.execution_mode == "next_bar_open" else bars["close"]).astype(float).to_numpy()
    size_arr = size.to_numpy(dtype="float64", copy=True)

    long_entries = signal_matrix.long_entries.reindex(index).fillna(False).to_numpy(dtype=bool, copy=True)
    long_exits = signal_matrix.long_exits.reindex(index).fillna(False).to_numpy(dtype=bool, copy=True)
    short_entries = signal_matrix.short_entries.reindex(index).fillna(False).to_numpy(dtype=bool, copy=True)
    short_exits = signal_matrix.short_exits.reindex(index).fillna(False).to_numpy(dtype=bool, copy=True)

    signal_arr = np.zeros(n, dtype=np.int32)
    position_arr = np.zeros(n, dtype=np.int32)
    risk_blocked = np.zeros(n, dtype=bool)
    risk_forced_exit = np.zeros(n, dtype=bool)

    current_pos = 0
    entry_idx = -1
    entry_size = 0.0
    equity = float(config.initial_cash)
    peak_equity = equity
    cooldown = 0

    max_dd = float(config.risk_symbol_drawdown_stop_pct)
    dd_cooldown_bars = int(config.risk_cooldown_bars_after_drawdown)
    max_holding_bars = int(config.risk_max_holding_bars)

    for i in range(n):
        action = 0
        if i > 0 and current_pos != 0 and price[i - 1] > 0 and np.isfinite(price[i]):
            bar_ret = price[i] / price[i - 1] - 1.0
            equity *= max(0.0, 1.0 + current_pos * entry_size * bar_ret)
            peak_equity = max(peak_equity, equity)

        force_exit = False
        if current_pos != 0:
            drawdown = equity / peak_equity - 1.0 if peak_equity > 0 else 0.0
            if max_dd > 0 and drawdown <= -max_dd:
                force_exit = True
                cooldown = dd_cooldown_bars
            if max_holding_bars > 0 and entry_idx >= 0 and i - entry_idx >= max_holding_bars:
                force_exit = True

        if force_exit:
            risk_forced_exit[i] = True
            long_entries[i] = False
            short_entries[i] = False
            if current_pos == 1:
                long_exits[i] = True
                action = -1
            else:
                short_exits[i] = True
                action = 1
            current_pos = 0
            entry_idx = -1
            entry_size = 0.0
            peak_equity = max(peak_equity, equity)
            position_arr[i] = current_pos
            signal_arr[i] = action
            if cooldown > 0:
                risk_blocked[i] = True
                cooldown -= 1
            continue

        if cooldown > 0:
            if long_entries[i] or short_entries[i]:
                risk_blocked[i] = True
            long_entries[i] = False
            short_entries[i] = False
            cooldown -= 1

        if size_arr[i] <= 0 or not np.isfinite(size_arr[i]):
            if long_entries[i] or short_entries[i]:
                risk_blocked[i] = True
            long_entries[i] = False
            short_entries[i] = False

        exited = False
        if current_pos == 1 and long_exits[i]:
            current_pos = 0
            entry_idx = -1
            entry_size = 0.0
            action = -1
            exited = True
        elif current_pos == -1 and short_exits[i]:
            current_pos = 0
            entry_idx = -1
            entry_size = 0.0
            action = 1
            exited = True

        if not exited and current_pos == 0:
            if long_entries[i]:
                current_pos = 1
                entry_idx = i
                entry_size = float(size_arr[i])
                action = 1
            elif short_entries[i]:
                current_pos = -1
                entry_idx = i
                entry_size = float(size_arr[i])
                action = -1

        signal_arr[i] = action
        position_arr[i] = current_pos

    return SignalMatrix(
        long_entries=pd.Series(long_entries, index=index),
        long_exits=pd.Series(long_exits, index=index),
        short_entries=pd.Series(short_entries, index=index),
        short_exits=pd.Series(short_exits, index=index),
        signal=pd.Series(signal_arr, index=index, dtype="int32"),
        position=pd.Series(position_arr, index=index, dtype="int32"),
        size=size,
        size_multiplier=signal_matrix.size_multiplier,
        leverage=signal_matrix.leverage,
        risk_blocked_entries=pd.Series(risk_blocked, index=index),
        risk_forced_exits=pd.Series(risk_forced_exit, index=index),
        execution_pairs=signal_matrix.execution_pairs,
    )
