from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EntryTimingConfig:
    delays_minutes: tuple[int, ...] = (0, 5, 15, 30, 60)
    max_holding_minutes: int = 1440
    signal_timeframe_minutes: int = 15
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    same_bar_policy: str = "stop_first"
    fast_success_minutes: int = 240


def _target_pct(row: pd.Series, default_pct: float = 0.01) -> float:
    for col in ("target_pct", "label_take_profit_pct", "label_stop_loss_pct"):
        if col in row.index:
            value = pd.to_numeric(row.get(col), errors="coerce")
            if np.isfinite(value) and float(value) > 0:
                return float(value)
    return float(default_pct)


def _safe_div(num: float, den: float) -> float:
    return float(num / (den + 1e-12))


def _series_return(close: pd.Series, window: int) -> float:
    if len(close) < 2:
        return 0.0
    tail = close.tail(max(2, int(window)))
    if len(tail) < 2:
        return 0.0
    return _safe_div(float(tail.iloc[-1]) - float(tail.iloc[0]), float(tail.iloc[0]))


def _realized_vol(close: pd.Series, window: int) -> float:
    ret = close.tail(max(2, int(window))).pct_change().dropna()
    return float(ret.std()) if len(ret) else 0.0


def _return_skew(close: pd.Series, window: int) -> float:
    ret = close.tail(max(3, int(window))).pct_change().dropna()
    return float(ret.skew()) if len(ret) >= 3 else 0.0


def _trend_efficiency(close: pd.Series, window: int) -> float:
    tail = close.tail(max(2, int(window)))
    if len(tail) < 2:
        return 0.0
    net = abs(float(tail.iloc[-1]) - float(tail.iloc[0]))
    path = float(tail.diff().abs().sum())
    return _safe_div(net, path)


def _linear_slope(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < 3:
        return 0.0
    y = clean.to_numpy(dtype="float64")
    x = np.arange(len(y), dtype="float64")
    try:
        return float(np.polyfit(x, y, 1)[0])
    except Exception:
        return 0.0


def _safe_mean(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    return float(clean.mean()) if len(clean) else 0.0


def _tail_close_return(close: pd.Series, start_offset: int, end_offset: int) -> float:
    clean = pd.to_numeric(close, errors="coerce").dropna()
    if len(clean) < 2:
        return 0.0
    start = max(0, min(len(clean) - 1, int(start_offset)))
    end = max(start + 1, min(len(clean) - 1, int(end_offset)))
    return _safe_div(float(clean.iloc[end]) - float(clean.iloc[start]), float(clean.iloc[start]))


def _slice_between(times_ns: np.ndarray, start: pd.Timestamp, end: pd.Timestamp) -> tuple[int, int]:
    start_pos = int(np.searchsorted(times_ns, start.value, side="left"))
    end_pos = int(np.searchsorted(times_ns, end.value, side="left"))
    return max(0, start_pos), max(0, end_pos)


def _pre_entry_features(
    bars: pd.DataFrame,
    times_ns: np.ndarray,
    row: pd.Series,
    available_time: pd.Timestamp,
    entry_time: pd.Timestamp,
    entry_pos: int,
) -> dict[str, float]:
    is_buy = bool(row["is_buy"])
    direction = 1.0 if is_buy else -1.0
    trade_price = float(pd.to_numeric(row.get("trade_price", np.nan), errors="coerce"))
    if not np.isfinite(trade_price) or trade_price <= 0:
        trade_price = float(bars["open"].iloc[entry_pos])

    start_pos, end_pos = _slice_between(times_ns, available_time, entry_time)
    history = bars.iloc[max(0, entry_pos - 240) : entry_pos]
    if end_pos <= start_pos:
        window = history.tail(1)
    else:
        window = bars.iloc[start_pos:min(end_pos, entry_pos)]
    if window.empty:
        window = history.tail(1)
    if window.empty:
        window = bars.iloc[max(0, entry_pos - 1) : entry_pos]

    entry_price = float(bars["open"].iloc[entry_pos])
    first_price = float(window["open"].iloc[0]) if not window.empty else entry_price
    last_price = float(window["close"].iloc[-1]) if not window.empty else entry_price
    high = float(window["high"].max())
    low = float(window["low"].min())
    volume = float(window["volume"].sum())
    count = max(1, int(len(window)))
    gross_move = _safe_div(last_price - first_price, first_price)
    entry_move = _safe_div(entry_price - trade_price, trade_price)
    if is_buy:
        pre_runup = _safe_div(high - trade_price, trade_price)
        pre_drawdown = _safe_div(low - trade_price, trade_price)
        close_location = _safe_div(last_price - low, high - low)
    else:
        pre_runup = _safe_div(trade_price - low, trade_price)
        pre_drawdown = _safe_div(trade_price - high, trade_price)
        close_location = _safe_div(high - last_price, high - low)

    ret = window["close"].pct_change().dropna()
    volatility = float(ret.std()) if len(ret) else 0.0
    avg_volume = float(history["volume"].tail(60).mean()) if not history.empty else 0.0

    close_hist = history["close"] if not history.empty else window["close"]
    high_hist = history["high"] if not history.empty else window["high"]
    low_hist = history["low"] if not history.empty else window["low"]
    vol_hist = history["volume"] if not history.empty else window["volume"]
    signed_returns = {w: direction * _series_return(close_hist, w) for w in (5, 15, 30, 60, 120)}
    realized_vol = {w: _realized_vol(close_hist, w) for w in (5, 15, 30, 60)}
    vol_mean_5 = float(vol_hist.tail(5).mean()) if len(vol_hist) else 0.0
    vol_mean_20 = float(vol_hist.tail(20).mean()) if len(vol_hist) else 0.0
    vol_mean_60 = float(vol_hist.tail(60).mean()) if len(vol_hist) else 0.0
    range_5 = float((high_hist.tail(5).max() - low_hist.tail(5).min()) / (trade_price + 1e-12)) if len(high_hist) else 0.0
    range_30 = float((high_hist.tail(30).max() - low_hist.tail(30).min()) / (trade_price + 1e-12)) if len(high_hist) else 0.0
    range_120 = float((high_hist.tail(120).max() - low_hist.tail(120).min()) / (trade_price + 1e-12)) if len(high_hist) else 0.0
    prev_high_30 = float(high_hist.tail(30).max()) if len(high_hist) else trade_price
    prev_low_30 = float(low_hist.tail(30).min()) if len(low_hist) else trade_price
    prev_high_120 = float(high_hist.tail(120).max()) if len(high_hist) else trade_price
    prev_low_120 = float(low_hist.tail(120).min()) if len(low_hist) else trade_price
    if is_buy:
        breakout_30 = _safe_div(entry_price - prev_high_30, entry_price)
        breakout_120 = _safe_div(entry_price - prev_high_120, entry_price)
        pullback_to_30 = _safe_div(entry_price - prev_low_30, entry_price)
        pullback_to_120 = _safe_div(entry_price - prev_low_120, entry_price)
    else:
        breakout_30 = _safe_div(prev_low_30 - entry_price, entry_price)
        breakout_120 = _safe_div(prev_low_120 - entry_price, entry_price)
        pullback_to_30 = _safe_div(prev_high_30 - entry_price, entry_price)
        pullback_to_120 = _safe_div(prev_high_120 - entry_price, entry_price)

    candle_body = (window["close"] - window["open"]) if not window.empty else pd.Series(dtype="float64")
    candle_range = (window["high"] - window["low"]).replace(0, np.nan) if not window.empty else pd.Series(dtype="float64")
    body_direction_ratio = float(((direction * candle_body) > 0).mean()) if len(candle_body) else 0.0
    body_strength = float(((direction * candle_body) / (candle_range + 1e-12)).replace([np.inf, -np.inf], np.nan).mean()) if len(candle_body) else 0.0
    if is_buy and not window.empty:
        support_shadow = ((window[["open", "close"]].min(axis=1) - window["low"]) / (candle_range + 1e-12)).replace([np.inf, -np.inf], np.nan)
        pressure_shadow = ((window["high"] - window[["open", "close"]].max(axis=1)) / (candle_range + 1e-12)).replace([np.inf, -np.inf], np.nan)
    elif not window.empty:
        support_shadow = ((window["high"] - window[["open", "close"]].max(axis=1)) / (candle_range + 1e-12)).replace([np.inf, -np.inf], np.nan)
        pressure_shadow = ((window[["open", "close"]].min(axis=1) - window["low"]) / (candle_range + 1e-12)).replace([np.inf, -np.inf], np.nan)
    else:
        support_shadow = pd.Series(dtype="float64")
        pressure_shadow = pd.Series(dtype="float64")

    ret_tail_5 = direction * _series_return(close_hist, 5)
    ret_tail_30 = direction * _series_return(close_hist, 30)
    ret_tail_60 = direction * _series_return(close_hist, 60)
    momentum_accel = ret_tail_5 - ret_tail_30
    momentum_decay = ret_tail_30 - ret_tail_60
    vol_expansion = _safe_div(realized_vol[5], realized_vol[30])
    volume_impulse = _safe_div(vol_mean_5, vol_mean_60)
    volume_trend = _safe_div(vol_mean_5 - vol_mean_20, vol_mean_20)
    price_volume_confirm = float((ret_tail_5 > 0.0) and (volume_impulse >= 1.05))
    compression_5_30 = _safe_div(range_5, range_30)
    compression_30_120 = _safe_div(range_30, range_120)
    trend_eff_30 = _trend_efficiency(close_hist, 30)
    trend_eff_120 = _trend_efficiency(close_hist, 120)
    ret_skew_30 = direction * _return_skew(close_hist, 30)
    ret_skew_120 = direction * _return_skew(close_hist, 120)
    vol_slope_20 = _linear_slope(vol_hist.tail(20)) / (vol_mean_20 + 1e-12)

    signal_window = window.copy()
    signal_close = pd.to_numeric(signal_window["close"], errors="coerce") if not signal_window.empty else pd.Series(dtype="float64")
    signal_open = pd.to_numeric(signal_window["open"], errors="coerce") if not signal_window.empty else pd.Series(dtype="float64")
    signal_high = pd.to_numeric(signal_window["high"], errors="coerce") if not signal_window.empty else pd.Series(dtype="float64")
    signal_low = pd.to_numeric(signal_window["low"], errors="coerce") if not signal_window.empty else pd.Series(dtype="float64")
    signal_volume = pd.to_numeric(signal_window["volume"], errors="coerce") if not signal_window.empty else pd.Series(dtype="float64")
    signal_ret_1 = direction * _tail_close_return(signal_close, 0, 1)
    signal_ret_3 = direction * _tail_close_return(signal_close, 0, 3)
    signal_ret_5 = direction * _tail_close_return(signal_close, 0, 5)
    signal_ret_10 = direction * _tail_close_return(signal_close, 0, 10)
    signal_ret_all = direction * _safe_div(last_price - first_price, first_price)
    signal_ret_mid = direction * _tail_close_return(signal_close, 0, max(1, len(signal_close) // 2)) if len(signal_close) else 0.0
    signal_ret_late = direction * _tail_close_return(signal_close, max(0, len(signal_close) // 2), len(signal_close) - 1) if len(signal_close) else 0.0
    signal_accel = signal_ret_late - signal_ret_mid
    signal_body = signal_close - signal_open if len(signal_close) else pd.Series(dtype="float64")
    signal_range = (signal_high - signal_low).replace(0, np.nan) if len(signal_close) else pd.Series(dtype="float64")
    directional_body_ratio = float(((direction * signal_body) > 0.0).mean()) if len(signal_body) else 0.0
    directional_close_ratio = float((direction * signal_close.diff() > 0.0).mean()) if len(signal_close) > 1 else 0.0
    directional_body_strength = _safe_mean(direction * signal_body / (signal_range + 1e-12)) if len(signal_body) else 0.0
    close_to_extreme = close_location
    if is_buy and len(signal_high):
        signal_favorable = _safe_div(float(signal_high.max()) - trade_price, trade_price)
        signal_adverse = _safe_div(trade_price - float(signal_low.min()), trade_price)
        prior_break_level = prev_high_30
        breakout_hold = float((entry_price > prior_break_level) and (last_price > prior_break_level))
        failed_breakout = float((signal_high.max() > prior_break_level) and (last_price <= prior_break_level))
        reclaim_after_pullback = float((signal_low.min() <= trade_price) and (last_price > trade_price))
    elif len(signal_low):
        signal_favorable = _safe_div(trade_price - float(signal_low.min()), trade_price)
        signal_adverse = _safe_div(float(signal_high.max()) - trade_price, trade_price)
        prior_break_level = prev_low_30
        breakout_hold = float((entry_price < prior_break_level) and (last_price < prior_break_level))
        failed_breakout = float((signal_low.min() < prior_break_level) and (last_price >= prior_break_level))
        reclaim_after_pullback = float((signal_high.max() >= trade_price) and (last_price < trade_price))
    else:
        signal_favorable = 0.0
        signal_adverse = 0.0
        prior_break_level = trade_price
        breakout_hold = 0.0
        failed_breakout = 0.0
        reclaim_after_pullback = 0.0
    adverse_control = _safe_div(signal_favorable, signal_adverse)
    favorable_efficiency = _safe_div(signal_ret_all, signal_favorable)
    early_follow_through = float(signal_ret_3 > 0.0 and signal_ret_5 >= signal_ret_3 * 0.5)
    late_follow_through = float(signal_ret_late >= 0.0 and signal_accel >= -0.001)
    confirmation_persistence = (
        0.25 * float(signal_ret_3 > 0.0)
        + 0.25 * float(signal_ret_5 > 0.0)
        + 0.20 * float(signal_ret_10 > 0.0)
        + 0.15 * directional_close_ratio
        + 0.15 * directional_body_ratio
    )
    volume_up = signal_volume.where(direction * signal_body > 0.0).mean() if len(signal_volume) else np.nan
    volume_down = signal_volume.where(direction * signal_body <= 0.0).mean() if len(signal_volume) else np.nan
    volume_confirmation = _safe_div(float(volume_up) if np.isfinite(volume_up) else 0.0, float(volume_down) if np.isfinite(volume_down) else 0.0)
    counter_impulse = float((signal_adverse >= max(0.002, signal_favorable * 0.85)) and (signal_ret_all <= 0.0))
    clean_second_confirm = (
        0.22 * float(signal_ret_all > 0.0)
        + 0.18 * early_follow_through
        + 0.16 * late_follow_through
        + 0.15 * float(signal_adverse <= max(0.0025, signal_favorable * 0.75 + 1e-12))
        + 0.12 * breakout_hold
        + 0.10 * float(volume_confirmation >= 1.05)
        + 0.07 * float(close_to_extreme >= 0.58)
    )
    second_failure_risk = (
        0.25 * counter_impulse
        + 0.20 * failed_breakout
        + 0.18 * float(signal_accel < -0.001)
        + 0.15 * float(signal_adverse > max(0.003, signal_favorable))
        + 0.12 * float(directional_body_ratio < 0.45)
        + 0.10 * float(volume_confirmation < 0.80)
    )
    return {
        "beta_pre_entry_minutes": float(max(0.0, (entry_time - available_time).total_seconds() / 60.0)),
        "beta_pre_entry_bar_count": float(count),
        "beta_pre_entry_signal_return": float(direction * gross_move),
        "beta_pre_entry_entry_move": float(direction * entry_move),
        "beta_pre_entry_runup": float(pre_runup),
        "beta_pre_entry_drawdown": float(pre_drawdown),
        "beta_pre_entry_range_pct": float(_safe_div(high - low, trade_price)),
        "beta_pre_entry_close_location": float(np.clip(close_location, 0.0, 1.0)),
        "beta_pre_entry_volatility": float(volatility),
        "beta_pre_entry_volume_ratio_60": float(_safe_div(volume / count, avg_volume)),
        "beta_confirm_momentum": float(direction * gross_move > 0.0),
        "beta_confirm_breakout": float(pre_runup >= max(0.0015, abs(pre_drawdown) * 1.2)),
        "beta_confirm_pullback_control": float(abs(pre_drawdown) <= max(0.0025, pre_runup * 0.8 + 1e-12)),
        "beta_confirm_volume": float(_safe_div(volume / count, avg_volume) >= 1.05),
        "beta_micro_signed_ret_5": float(signed_returns[5]),
        "beta_micro_signed_ret_15": float(signed_returns[15]),
        "beta_micro_signed_ret_30": float(signed_returns[30]),
        "beta_micro_signed_ret_60": float(signed_returns[60]),
        "beta_micro_signed_ret_120": float(signed_returns[120]),
        "beta_micro_momentum_accel_5_30": float(momentum_accel),
        "beta_micro_momentum_decay_30_60": float(momentum_decay),
        "beta_micro_realized_vol_5": float(realized_vol[5]),
        "beta_micro_realized_vol_15": float(realized_vol[15]),
        "beta_micro_realized_vol_30": float(realized_vol[30]),
        "beta_micro_realized_vol_60": float(realized_vol[60]),
        "beta_micro_vol_expansion_5_30": float(vol_expansion),
        "beta_micro_range_5": float(range_5),
        "beta_micro_range_30": float(range_30),
        "beta_micro_range_120": float(range_120),
        "beta_micro_compression_5_30": float(compression_5_30),
        "beta_micro_compression_30_120": float(compression_30_120),
        "beta_micro_volume_impulse_5_60": float(volume_impulse),
        "beta_micro_volume_trend_5_20": float(volume_trend),
        "beta_micro_volume_slope_20": float(vol_slope_20),
        "beta_micro_price_volume_confirm": float(price_volume_confirm),
        "beta_micro_breakout_dist_30": float(breakout_30),
        "beta_micro_breakout_dist_120": float(breakout_120),
        "beta_micro_pullback_dist_30": float(pullback_to_30),
        "beta_micro_pullback_dist_120": float(pullback_to_120),
        "beta_micro_body_direction_ratio": float(body_direction_ratio),
        "beta_micro_body_strength": float(body_strength),
        "beta_micro_support_shadow": float(support_shadow.mean()) if len(support_shadow) else 0.0,
        "beta_micro_pressure_shadow": float(pressure_shadow.mean()) if len(pressure_shadow) else 0.0,
        "beta_micro_shadow_balance": float((support_shadow.mean() if len(support_shadow) else 0.0) - (pressure_shadow.mean() if len(pressure_shadow) else 0.0)),
        "beta_micro_trend_efficiency_30": float(trend_eff_30),
        "beta_micro_trend_efficiency_120": float(trend_eff_120),
        "beta_micro_return_skew_30": float(ret_skew_30),
        "beta_micro_return_skew_120": float(ret_skew_120),
        "beta_second_signal_ret_1": float(signal_ret_1),
        "beta_second_signal_ret_3": float(signal_ret_3),
        "beta_second_signal_ret_5": float(signal_ret_5),
        "beta_second_signal_ret_10": float(signal_ret_10),
        "beta_second_signal_ret_all": float(signal_ret_all),
        "beta_second_signal_accel": float(signal_accel),
        "beta_second_directional_close_ratio": float(directional_close_ratio),
        "beta_second_directional_body_ratio": float(directional_body_ratio),
        "beta_second_directional_body_strength": float(directional_body_strength),
        "beta_second_favorable_excursion": float(signal_favorable),
        "beta_second_adverse_excursion": float(signal_adverse),
        "beta_second_adverse_control": float(adverse_control),
        "beta_second_favorable_efficiency": float(favorable_efficiency),
        "beta_second_breakout_hold": float(breakout_hold),
        "beta_second_failed_breakout": float(failed_breakout),
        "beta_second_reclaim_after_pullback": float(reclaim_after_pullback),
        "beta_second_early_follow_through": float(early_follow_through),
        "beta_second_late_follow_through": float(late_follow_through),
        "beta_second_confirmation_persistence": float(confirmation_persistence),
        "beta_second_volume_confirmation": float(volume_confirmation),
        "beta_second_counter_impulse": float(counter_impulse),
        "beta_second_clean_confirm_score": float(clean_second_confirm),
        "beta_second_failure_risk_score": float(second_failure_risk),
    }


def _simulate_candidate(
    bars: pd.DataFrame,
    times: pd.DatetimeIndex,
    entry_pos: int,
    row: pd.Series,
    target_pct: float,
    cfg: EntryTimingConfig,
) -> dict[str, object]:
    open_ = bars["open"].to_numpy(dtype="float64", copy=False)
    high = bars["high"].to_numpy(dtype="float64", copy=False)
    low = bars["low"].to_numpy(dtype="float64", copy=False)
    close = bars["close"].to_numpy(dtype="float64", copy=False)
    n = len(bars)
    is_buy = bool(row["is_buy"])
    direction = 1.0 if is_buy else -1.0
    cost = 2.0 * (float(cfg.fee_rate) + float(cfg.slippage_rate))
    result = {
        "entry_time": pd.NaT,
        "entry_price": np.nan,
        "exit_time": pd.NaT,
        "exit_price": np.nan,
        "exit_reason": "no_future",
        "gross_return": np.nan,
        "net_return": np.nan,
        "mfe": np.nan,
        "mae": np.nan,
        "holding_minutes": np.nan,
        "label_tp_first": np.nan,
    }
    if entry_pos < 0 or entry_pos >= n:
        return result
    entry_price = float(open_[entry_pos])
    if not np.isfinite(entry_price) or entry_price <= 0:
        return result
    if is_buy:
        tp_price = entry_price * (1.0 + target_pct)
        sl_price = entry_price * (1.0 - target_pct)
    else:
        tp_price = entry_price * (1.0 - target_pct)
        sl_price = entry_price * (1.0 + target_pct)

    future_end = min(n, entry_pos + max(1, int(cfg.max_holding_minutes)))
    exit_pos = future_end - 1
    exit_price = float(close[exit_pos])
    exit_reason = "timeout"
    for pos in range(entry_pos, future_end):
        if is_buy:
            hit_tp = bool(high[pos] >= tp_price)
            hit_sl = bool(low[pos] <= sl_price)
        else:
            hit_tp = bool(low[pos] <= tp_price)
            hit_sl = bool(high[pos] >= sl_price)
        if hit_tp and hit_sl:
            exit_pos = pos
            if cfg.same_bar_policy == "take_profit_first":
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

    window_high = float(np.nanmax(high[entry_pos:future_end]))
    window_low = float(np.nanmin(low[entry_pos:future_end]))
    if is_buy:
        mfe = (window_high - entry_price) / entry_price
        mae = (window_low - entry_price) / entry_price
    else:
        mfe = (entry_price - window_low) / entry_price
        mae = (entry_price - window_high) / entry_price
    gross_return = direction * (float(exit_price) - entry_price) / entry_price
    net_return = gross_return - cost
    holding_minutes = max(0.0, (times[exit_pos] - times[entry_pos]).total_seconds() / 60.0)
    result.update(
        {
            "entry_time": times[entry_pos],
            "entry_price": entry_price,
            "exit_time": times[exit_pos],
            "exit_price": float(exit_price),
            "exit_reason": exit_reason,
            "gross_return": float(gross_return),
            "net_return": float(net_return),
            "mfe": float(mfe),
            "mae": float(mae),
            "holding_minutes": float(holding_minutes),
            "label_tp_first": float(exit_reason == "take_profit"),
        }
    )
    return result


def build_entry_timing_candidates(
    events: pd.DataFrame,
    bars_1m: pd.DataFrame,
    delays_minutes: Iterable[int] | None = None,
    config: EntryTimingConfig | None = None,
) -> pd.DataFrame:
    cfg = config or EntryTimingConfig()
    if cfg.same_bar_policy not in {"stop_first", "take_profit_first"}:
        raise ValueError("same_bar_policy must be stop_first or take_profit_first")
    delays = tuple(int(x) for x in (delays_minutes if delays_minutes is not None else cfg.delays_minutes))
    bars = bars_1m.sort_index().copy()
    times = pd.DatetimeIndex(pd.to_datetime(bars.index, utc=True))
    times_ns = times.view("int64")
    rows: list[dict[str, object]] = []
    for event_pos, (_, event) in enumerate(events.iterrows()):
        exec_time = pd.to_datetime(event["exec_time"], utc=True)
        available_time = exec_time + pd.Timedelta(minutes=int(cfg.signal_timeframe_minutes))
        target_pct = _target_pct(event)
        for delay in delays:
            candidate_time = available_time + pd.Timedelta(minutes=int(delay))
            entry_pos = int(np.searchsorted(times_ns, candidate_time.value, side="left"))
            base = event.to_dict()
            base["beta_event_id"] = int(event_pos)
            base["beta_entry_delay_minutes"] = int(delay)
            base["beta_signal_available_time"] = available_time
            base["target_pct"] = float(target_pct)
            if 0 <= entry_pos < len(bars):
                base.update(_pre_entry_features(bars, times_ns, event, available_time, times[entry_pos], entry_pos))
            else:
                base.update(
                    {
                        "beta_pre_entry_minutes": float(delay),
                        "beta_pre_entry_bar_count": 0.0,
                        "beta_pre_entry_signal_return": np.nan,
                        "beta_pre_entry_entry_move": np.nan,
                        "beta_pre_entry_runup": np.nan,
                        "beta_pre_entry_drawdown": np.nan,
                        "beta_pre_entry_range_pct": np.nan,
                        "beta_pre_entry_close_location": np.nan,
                        "beta_pre_entry_volatility": np.nan,
                        "beta_pre_entry_volume_ratio_60": np.nan,
                        "beta_confirm_momentum": np.nan,
                        "beta_confirm_breakout": np.nan,
                        "beta_confirm_pullback_control": np.nan,
                        "beta_confirm_volume": np.nan,
                    }
                )
            base.update(_simulate_candidate(bars, times, entry_pos, event, target_pct, cfg))
            rows.append(base)
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["label"] = pd.to_numeric(out["label_tp_first"], errors="coerce")
    for col in ("exec_time", "entry_time", "exit_time", "beta_signal_available_time"):
        out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)


def build_entry_signal_samples(
    events: pd.DataFrame,
    bars_1m: pd.DataFrame,
    config: EntryTimingConfig | None = None,
) -> pd.DataFrame:
    """Build exactly one executable sample for each BSP event.

    The previous beta route expanded each BSP into multiple delayed candidate
    entries. This route keeps the decision surface aligned with live trading:
    the signal is available only after the 15m bar closes, and the sample enters
    at the first 1m bar at or after that availability time.
    """
    cfg = config or EntryTimingConfig(delays_minutes=(0,))
    if cfg.same_bar_policy not in {"stop_first", "take_profit_first"}:
        raise ValueError("same_bar_policy must be stop_first or take_profit_first")
    bars = bars_1m.sort_index().copy()
    times = pd.DatetimeIndex(pd.to_datetime(bars.index, utc=True))
    times_ns = times.view("int64")
    rows: list[dict[str, object]] = []
    for event_pos, (_, event) in enumerate(events.iterrows()):
        exec_time = pd.to_datetime(event["exec_time"], utc=True)
        available_time = exec_time + pd.Timedelta(minutes=int(cfg.signal_timeframe_minutes))
        entry_pos = int(np.searchsorted(times_ns, available_time.value, side="left"))
        target_pct = _target_pct(event)
        base = event.to_dict()
        base["beta_event_id"] = int(event_pos)
        base["beta_entry_delay_minutes"] = 0
        base["beta_signal_available_time"] = available_time
        base["target_pct"] = float(target_pct)
        if 0 <= entry_pos < len(bars):
            base.update(_pre_entry_features(bars, times_ns, event, available_time, times[entry_pos], entry_pos))
        else:
            base.update(
                {
                    "beta_pre_entry_minutes": 0.0,
                    "beta_pre_entry_bar_count": 0.0,
                    "beta_pre_entry_signal_return": np.nan,
                    "beta_pre_entry_entry_move": np.nan,
                    "beta_pre_entry_runup": np.nan,
                    "beta_pre_entry_drawdown": np.nan,
                    "beta_pre_entry_range_pct": np.nan,
                    "beta_pre_entry_close_location": np.nan,
                    "beta_pre_entry_volatility": np.nan,
                    "beta_pre_entry_volume_ratio_60": np.nan,
                    "beta_confirm_momentum": np.nan,
                    "beta_confirm_breakout": np.nan,
                    "beta_confirm_pullback_control": np.nan,
                    "beta_confirm_volume": np.nan,
                }
            )
        base.update(_simulate_candidate(bars, times, entry_pos, event, target_pct, cfg))
        rows.append(base)

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["label"] = pd.to_numeric(out["label_tp_first"], errors="coerce")
    for col in ("exec_time", "entry_time", "exit_time", "beta_signal_available_time"):
        out[col] = pd.to_datetime(out[col], utc=True, errors="coerce")
    return out.replace([np.inf, -np.inf], np.nan)
