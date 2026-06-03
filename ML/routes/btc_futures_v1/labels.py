from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FixedTpSlLabelConfig:
    take_profit_pct: float = 0.01
    stop_loss_pct: float = 0.01
    max_holding_minutes: int = 1440
    signal_timeframe_minutes: int = 0
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    same_bar_policy: str = "stop_first"


@dataclass(frozen=True)
class Bsp2StructureLabelConfig:
    max_observe_minutes: int = 1440
    signal_timeframe_minutes: int = 0
    atr_window_minutes: int = 14 * 15
    invalid_lookback_minutes: int = 32 * 15
    first_rebound_lookback_minutes: int = 16 * 15
    atr_buffer_mult: float = 0.20
    fallback_invalid_pct: float = 0.006
    fee_rate: float = 0.0004
    slippage_rate: float = 0.0001
    confirm_buffer_atr_mult: float = 0.20
    third_buy_hold_bars: int = 12
    third_move_buffer_atr_mult: float = 0.05


def _event_zs_prices(event: pd.Series, price: float) -> tuple[float, float, float]:
    direct_low = pd.to_numeric(pd.Series([event.get("chan_bsp2_origin_zs_low", np.nan)]), errors="coerce").iloc[0]
    direct_mid = pd.to_numeric(pd.Series([event.get("chan_bsp2_origin_zs_mid", np.nan)]), errors="coerce").iloc[0]
    direct_high = pd.to_numeric(pd.Series([event.get("chan_bsp2_origin_zs_high", np.nan)]), errors="coerce").iloc[0]
    if np.isfinite(direct_low) and np.isfinite(direct_high):
        low = float(min(direct_low, direct_high))
        high = float(max(direct_low, direct_high))
        mid = float(direct_mid) if np.isfinite(direct_mid) else (low + high) / 2.0
        return low, mid, high

    def price_from_pct(col: str) -> float:
        value = pd.to_numeric(pd.Series([event.get(col, np.nan)]), errors="coerce").iloc[0]
        if not np.isfinite(value):
            return np.nan
        return float(price / (1.0 + float(value)))

    high = price_from_pct("chan_mtf_15m_event_vs_last_zs_high_pct")
    low = price_from_pct("chan_mtf_15m_event_vs_last_zs_low_pct")
    mid = price_from_pct("chan_mtf_15m_event_vs_last_zs_mid_pct")
    if not np.isfinite(high):
        high = price_from_pct("chan_ctx_close_vs_last_zs_high_pct")
    if not np.isfinite(low):
        low = price_from_pct("chan_ctx_close_vs_last_zs_low_pct")
    if not np.isfinite(mid):
        mid = price_from_pct("chan_ctx_close_vs_last_zs_mid_pct")
    if np.isfinite(high) and np.isfinite(low) and high < low:
        high, low = low, high
    if not np.isfinite(mid) and np.isfinite(high) and np.isfinite(low):
        mid = (high + low) / 2.0
    return low, mid, high


def _event_has_direct_bsp2_origin_zs(event: pd.Series) -> bool:
    low = pd.to_numeric(pd.Series([event.get("chan_bsp2_origin_zs_low", np.nan)]), errors="coerce").iloc[0]
    high = pd.to_numeric(pd.Series([event.get("chan_bsp2_origin_zs_high", np.nan)]), errors="coerce").iloc[0]
    return bool(np.isfinite(low) and np.isfinite(high))


def _event_number(event: pd.Series, col: str) -> float:
    return pd.to_numeric(pd.Series([event.get(col, np.nan)]), errors="coerce").iloc[0]


def _bsp2_position_vs_origin_zs(is_buy: bool, price: float, zs_low: float, zs_high: float) -> str:
    if not (np.isfinite(price) and np.isfinite(zs_low) and np.isfinite(zs_high)):
        return "unknown"
    low = min(float(zs_low), float(zs_high))
    high = max(float(zs_low), float(zs_high))
    if price < low:
        return "below_zs"
    if price > high:
        return "above_zs"
    return "inside_zs"


def _third_move_ref_price(event: pd.Series, is_buy: bool, fallback: float) -> tuple[float, str]:
    if is_buy:
        break_price = _event_number(event, "chan_bsp2_break_bi_high")
    else:
        break_price = _event_number(event, "chan_bsp2_break_bi_low")
    if np.isfinite(break_price):
        return float(break_price), "chan_bsp2_break_bi"
    return float(fallback), "fallback_rebound_window"


def _rolling_atr_pct(bars: pd.DataFrame, window_minutes: int) -> pd.Series:
    close = pd.to_numeric(bars["close"], errors="coerce")
    high = pd.to_numeric(bars["high"], errors="coerce")
    low = pd.to_numeric(bars["low"], errors="coerce")
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(max(5, int(window_minutes)), min_periods=max(3, int(window_minutes) // 4)).mean()
    return atr / (close + 1e-12)


def label_bsp2_structure_with_1m_path(
    events: pd.DataFrame,
    bars_1m: pd.DataFrame,
    config: Bsp2StructureLabelConfig | None = None,
) -> pd.DataFrame:
    """Label second-class BSP events by Chan-structure fulfilment.

    The features used for invalidation and targets are available at the
    signal close. Future bars are used only for supervised label outcomes.
    """
    cfg = config or Bsp2StructureLabelConfig()
    if events.empty:
        return pd.DataFrame(index=events.index)

    bars = bars_1m.sort_index()
    times = pd.DatetimeIndex(pd.to_datetime(bars.index, utc=True))
    time_ns = times.view("int64")
    open_ = bars["open"].to_numpy(dtype="float64", copy=False)
    high = bars["high"].to_numpy(dtype="float64", copy=False)
    low = bars["low"].to_numpy(dtype="float64", copy=False)
    close = bars["close"].to_numpy(dtype="float64", copy=False)
    atr_pct = _rolling_atr_pct(bars, cfg.atr_window_minutes).to_numpy(dtype="float64", copy=False)
    max_hold = max(1, int(cfg.max_observe_minutes))
    invalid_lookback = max(1, int(cfg.invalid_lookback_minutes))
    rebound_lookback = max(1, int(cfg.first_rebound_lookback_minutes))
    cost = 2.0 * (float(cfg.fee_rate) + float(cfg.slippage_rate))

    rows: list[dict[str, object]] = []
    for _, event in events.iterrows():
        exec_time = pd.to_datetime(event["exec_time"], utc=True)
        signal_available_time = exec_time + pd.Timedelta(minutes=int(cfg.signal_timeframe_minutes))
        entry_pos = int(np.searchsorted(time_ns, signal_available_time.value, side="left"))
        is_buy = bool(event["is_buy"])
        direction = 1.0 if is_buy else -1.0
        row: dict[str, object] = {
            "signal_available_time": signal_available_time,
            "entry_time": pd.NaT,
            "entry_price": np.nan,
            "exit_time": pd.NaT,
            "exit_price": np.nan,
            "exit_reason": "no_future",
            "gross_return": np.nan,
            "net_return": np.nan,
            "mfe": np.nan,
            "mae": np.nan,
            "entry_bar_idx": entry_pos,
            "label_bsp2_valid": np.nan,
            "label_bsp2_return_prev_zs": np.nan,
            "label_bsp2_return_prev_zs_level": np.nan,
            "label_bsp2_break_first_rebound": np.nan,
            "label_bsp2_third_move": np.nan,
            "label_bsp2_third_move_ref_price": np.nan,
            "label_bsp2_third_move_ref_source": "",
            "label_bsp2_to_bsp3": np.nan,
            "label_bsp2_entry_quality": np.nan,
            "label_bsp2_chan_entry_quality": np.nan,
            "label_bsp2_origin_zs_available": np.nan,
            "label_bsp2_constructive_overlap": np.nan,
            "label_bsp2_strong_2_3_combo": np.nan,
            "label_bsp2_weak_or_invalid": np.nan,
            "label_bsp2_strength_class": "",
            "label_bsp2_confirm_time": np.nan,
            "label_bsp2_confirm_price": np.nan,
            "label_bsp2_confirm_reason": "",
            "label_bsp2_struct_target_time": np.nan,
            "label_bsp2_struct_invalid_time": np.nan,
            "label_bsp2_struct_mfe": np.nan,
            "label_bsp2_struct_mae": np.nan,
            "label_bsp2_invalid_price": np.nan,
            "label_bsp2_invalid_bi_idx": np.nan,
            "label_bsp2_invalid_source": "",
            "label_bsp2_prev_zs_low": np.nan,
            "label_bsp2_prev_zs_mid": np.nan,
            "label_bsp2_prev_zs_high": np.nan,
            "label_bsp2_prev_zs_source": "",
            "label_bsp2_rebound_ref_price": np.nan,
            "label_bsp2_position_vs_origin_zs": "",
            "label_bsp2_follow_path": "",
        }
        if entry_pos < 0 or entry_pos >= len(bars):
            rows.append(row)
            continue

        entry_price = float(open_[entry_pos])
        if not np.isfinite(entry_price) or entry_price <= 0:
            rows.append(row)
            continue

        future_start = entry_pos
        future_end = min(len(bars), entry_pos + max_hold)
        if future_start >= future_end:
            rows.append(row)
            continue

        lookback_start = max(0, entry_pos - invalid_lookback)
        rebound_start = max(0, entry_pos - rebound_lookback)
        atr_buffer = max(
            entry_price * float(cfg.fallback_invalid_pct),
            entry_price * float(np.nan_to_num(atr_pct[entry_pos], nan=0.0)) * float(cfg.atr_buffer_mult),
        )
        bsp_types = str(event.get("bsp_types_str", event.get("bsp_type", ""))).lower()
        actual_invalid = pd.to_numeric(
            pd.Series(
                [
                    event.get("chan_bsp2s_invalid_price", np.nan)
                    if "2s" in bsp_types
                    else event.get("chan_bsp2_invalid_price", np.nan)
                ]
            ),
            errors="coerce",
        ).iloc[0]
        invalid_bi_idx = pd.to_numeric(
            pd.Series(
                [
                    event.get("chan_bsp2s_invalid_bi_idx", np.nan)
                    if "2s" in bsp_types
                    else event.get("chan_bsp2_invalid_bi_idx", np.nan)
                ]
            ),
            errors="coerce",
        ).iloc[0]
        invalid_source = "chan_actual"
        if np.isfinite(actual_invalid):
            invalid_price = float(actual_invalid)
        elif is_buy:
            invalid_price = float(np.nanmin(low[lookback_start : entry_pos + 1]) - atr_buffer)
            invalid_source = "fallback_window_atr"
        else:
            invalid_price = float(np.nanmax(high[lookback_start : entry_pos + 1]) + atr_buffer)
            invalid_source = "fallback_window_atr"
        if is_buy:
            rebound_ref = float(np.nanmax(high[rebound_start : entry_pos + 1]))
        else:
            rebound_ref = float(np.nanmin(low[rebound_start : entry_pos + 1]))

        trade_price = float(event.get("trade_price", entry_price) or entry_price)
        has_direct_origin_zs = _event_has_direct_bsp2_origin_zs(event)
        zs_low, zs_mid, zs_high = _event_zs_prices(event, trade_price)
        position_vs_origin_zs = _bsp2_position_vs_origin_zs(is_buy, trade_price, zs_low, zs_high)
        third_move_ref, third_move_ref_source = _third_move_ref_price(event, is_buy, rebound_ref)
        third_move_buffer = float(atr_buffer) * float(cfg.third_move_buffer_atr_mult)
        if not np.isfinite(third_move_buffer) or third_move_buffer < 0:
            third_move_buffer = 0.0
        first_target_pos: int | None = None
        first_target_price = np.nan
        invalid_pos: int | None = None
        confirm_pos: int | None = None
        confirm_price = np.nan
        confirm_reason = ""
        strong_2_3_combo = 0.0
        return_prev_zs = 0.0
        return_level = 0.0
        break_first_rebound = 0.0
        third_move = 0.0
        third_move_pos: int | None = None
        to_bsp3 = 0.0
        confirm_buffer = float(atr_buffer) * float(cfg.confirm_buffer_atr_mult)
        if not np.isfinite(confirm_buffer) or confirm_buffer < 0:
            confirm_buffer = 0.0
        third_buy_hold_bars = max(1, int(cfg.third_buy_hold_bars))
        for pos in range(future_start, future_end):
            if is_buy:
                if third_move <= 0.0 and low[pos] <= invalid_price:
                    invalid_pos = pos
                    break
                if third_move <= 0.0 and np.isfinite(third_move_ref) and high[pos] >= third_move_ref + third_move_buffer:
                    third_move = 1.0
                    third_move_pos = pos
                    if confirm_pos is None:
                        confirm_pos = pos
                        confirm_price = float(third_move_ref)
                        confirm_reason = "third_same_direction_move"
                    if first_target_pos is None:
                        first_target_pos = pos
                        first_target_price = float(third_move_ref)
                if np.isfinite(zs_low) and high[pos] >= zs_low:
                    return_prev_zs = 1.0
                    return_level = max(return_level, 1.0)
                    if has_direct_origin_zs and confirm_pos is None and third_move <= 0.0:
                        confirm_pos = pos
                        confirm_price = float(zs_low)
                        confirm_reason = "return_origin_zs"
                if np.isfinite(zs_mid) and high[pos] >= zs_mid:
                    return_level = max(return_level, 2.0)
                if np.isfinite(zs_high) and high[pos] >= zs_high:
                    return_level = max(return_level, 3.0)
                if np.isfinite(rebound_ref) and high[pos] >= rebound_ref:
                    break_first_rebound = 1.0
                strong_target = zs_high + atr_buffer if np.isfinite(zs_high) else np.nan
                if np.isfinite(strong_target) and high[pos] >= strong_target:
                    after_low = np.nanmin(low[pos : min(future_end, pos + third_buy_hold_bars)])
                    if np.isfinite(after_low) and after_low >= zs_high - confirm_buffer:
                        to_bsp3 = 1.0
                        strong_2_3_combo = 1.0
                        if confirm_pos is None or confirm_reason != "strong_2_3_combo":
                            confirm_pos = pos
                            confirm_price = float(zs_high + atr_buffer)
                            confirm_reason = "strong_2_3_combo"
                        if third_move <= 0.0:
                            third_move = 1.0
                            third_move_pos = pos
                        if first_target_pos is None:
                            first_target_pos = pos
                            first_target_price = float(zs_high + atr_buffer)
            else:
                if third_move <= 0.0 and high[pos] >= invalid_price:
                    invalid_pos = pos
                    break
                if third_move <= 0.0 and np.isfinite(third_move_ref) and low[pos] <= third_move_ref - third_move_buffer:
                    third_move = 1.0
                    third_move_pos = pos
                    if confirm_pos is None:
                        confirm_pos = pos
                        confirm_price = float(third_move_ref)
                        confirm_reason = "third_same_direction_move"
                    if first_target_pos is None:
                        first_target_pos = pos
                        first_target_price = float(third_move_ref)
                if np.isfinite(zs_high) and low[pos] <= zs_high:
                    return_prev_zs = 1.0
                    return_level = max(return_level, 1.0)
                    if has_direct_origin_zs and confirm_pos is None and third_move <= 0.0:
                        confirm_pos = pos
                        confirm_price = float(zs_high)
                        confirm_reason = "return_origin_zs"
                if np.isfinite(zs_mid) and low[pos] <= zs_mid:
                    return_level = max(return_level, 2.0)
                if np.isfinite(zs_low) and low[pos] <= zs_low:
                    return_level = max(return_level, 3.0)
                if np.isfinite(rebound_ref) and low[pos] <= rebound_ref:
                    break_first_rebound = 1.0
                strong_target = zs_low - atr_buffer if np.isfinite(zs_low) else np.nan
                if np.isfinite(strong_target) and low[pos] <= strong_target:
                    after_high = np.nanmax(high[pos : min(future_end, pos + third_buy_hold_bars)])
                    if np.isfinite(after_high) and after_high <= zs_low + confirm_buffer:
                        to_bsp3 = 1.0
                        strong_2_3_combo = 1.0
                        if confirm_pos is None or confirm_reason != "strong_2_3_combo":
                            confirm_pos = pos
                            confirm_price = float(zs_low - atr_buffer)
                            confirm_reason = "strong_2_3_combo"
                        if third_move <= 0.0:
                            third_move = 1.0
                            third_move_pos = pos
                        if first_target_pos is None:
                            first_target_pos = pos
                            first_target_price = float(zs_low - atr_buffer)

        if third_move > 0.0 and strong_2_3_combo <= 0.0 and third_move_pos is not None:
            confirm_pos = third_move_pos
            confirm_price = float(third_move_ref)
            confirm_reason = "third_same_direction_move"

        terminal_pos = invalid_pos if invalid_pos is not None else (first_target_pos if first_target_pos is not None else future_end - 1)
        if invalid_pos is not None:
            exit_price = float(invalid_price)
        elif first_target_pos is not None and np.isfinite(first_target_price):
            exit_price = float(first_target_price)
        else:
            exit_price = float(close[terminal_pos])
        window = slice(future_start, future_end)
        if is_buy:
            mfe = float((np.nanmax(high[window]) - entry_price) / (entry_price + 1e-12))
            mae = float((np.nanmin(low[window]) - entry_price) / (entry_price + 1e-12))
        else:
            mfe = float((entry_price - np.nanmin(low[window])) / (entry_price + 1e-12))
            mae = float((entry_price - np.nanmax(high[window])) / (entry_price + 1e-12))
        valid = 1.0 if invalid_pos is None else 0.0
        constructive_overlap = 1.0 if valid and has_direct_origin_zs and (return_prev_zs or to_bsp3) else 0.0
        chan_quality = 1.0 if valid and third_move > 0.0 else 0.0
        quality = chan_quality
        weak_or_invalid = 1.0 if chan_quality <= 0.0 else 0.0
        expansion_positions = {"inside_zs", "below_zs"} if is_buy else {"inside_zs", "above_zs"}
        if third_move > 0.0 and return_prev_zs > 0.0:
            constructive_overlap = 1.0
        elif third_move > 0.0 and position_vs_origin_zs in expansion_positions:
            constructive_overlap = 1.0

        if invalid_pos is not None:
            follow_path = "invalid"
        elif strong_2_3_combo > 0.0:
            follow_path = "bsp2_t3_overlap"
        elif return_level >= 3.0:
            follow_path = "return_origin_zs_full"
        elif return_prev_zs > 0.0:
            follow_path = "return_origin_zs"
        elif third_move > 0.0 and position_vs_origin_zs in expansion_positions:
            follow_path = "higher_level_expansion_candidate"
        elif third_move > 0.0:
            follow_path = "third_move_only"
        else:
            follow_path = "weak_no_third_move"
        if invalid_pos is not None:
            strength_class = "invalid"
        elif strong_2_3_combo > 0.0:
            strength_class = "strong_2_3_combo"
        elif third_move > 0.0 and has_direct_origin_zs and return_level >= 2.0:
            strength_class = "third_move_return_origin_zs"
        elif third_move > 0.0:
            strength_class = "third_same_direction_move"
        elif has_direct_origin_zs and return_level >= 2.0:
            strength_class = "normal_return_origin_zs"
        elif has_direct_origin_zs and (return_prev_zs or break_first_rebound):
            strength_class = "weak_confirmed_overlap"
        elif has_direct_origin_zs:
            strength_class = "weak_no_confirm"
        else:
            strength_class = "no_origin_zs"
        row.update(
            {
                "entry_time": times[entry_pos],
                "entry_price": entry_price,
                "exit_time": times[terminal_pos],
                "exit_price": exit_price,
                "exit_reason": "structure_invalid" if invalid_pos is not None else ("structure_target" if first_target_pos is not None else "structure_timeout"),
                "gross_return": float(direction * (exit_price - entry_price) / (entry_price + 1e-12)),
                "net_return": float(direction * (exit_price - entry_price) / (entry_price + 1e-12) - cost),
                "mfe": mfe,
                "mae": mae,
                "label_bsp2_valid": valid,
                "label_bsp2_return_prev_zs": float(return_prev_zs),
                "label_bsp2_return_prev_zs_level": float(return_level),
                "label_bsp2_break_first_rebound": float(break_first_rebound),
                "label_bsp2_third_move": float(third_move),
                "label_bsp2_third_move_ref_price": float(third_move_ref) if np.isfinite(third_move_ref) else np.nan,
                "label_bsp2_third_move_ref_source": third_move_ref_source,
                "label_bsp2_to_bsp3": float(to_bsp3),
                "label_bsp2_entry_quality": float(quality),
                "label_bsp2_chan_entry_quality": float(chan_quality),
                "label_bsp2_origin_zs_available": float(1.0 if has_direct_origin_zs else 0.0),
                "label_bsp2_constructive_overlap": float(constructive_overlap),
                "label_bsp2_strong_2_3_combo": float(strong_2_3_combo),
                "label_bsp2_weak_or_invalid": float(weak_or_invalid),
                "label_bsp2_strength_class": strength_class,
                "label_bsp2_confirm_time": (
                    float((times[confirm_pos] - times[entry_pos]).total_seconds() / 60.0)
                    if confirm_pos is not None
                    else np.nan
                ),
                "label_bsp2_confirm_price": float(confirm_price) if np.isfinite(confirm_price) else np.nan,
                "label_bsp2_confirm_reason": confirm_reason,
                "label_bsp2_struct_target_time": (
                    float((times[first_target_pos] - times[entry_pos]).total_seconds() / 60.0)
                    if first_target_pos is not None
                    else np.nan
                ),
                "label_bsp2_struct_invalid_time": (
                    float((times[invalid_pos] - times[entry_pos]).total_seconds() / 60.0)
                    if invalid_pos is not None
                    else np.nan
                ),
                "label_bsp2_struct_mfe": mfe,
                "label_bsp2_struct_mae": mae,
                "label_bsp2_invalid_price": invalid_price,
                "label_bsp2_invalid_bi_idx": float(invalid_bi_idx) if np.isfinite(invalid_bi_idx) else np.nan,
                "label_bsp2_invalid_source": invalid_source,
                "label_bsp2_prev_zs_low": zs_low,
                "label_bsp2_prev_zs_mid": zs_mid,
                "label_bsp2_prev_zs_high": zs_high,
                "label_bsp2_prev_zs_source": "chan_bsp2_origin_zs" if has_direct_origin_zs else "last_zs_fallback",
                "label_bsp2_rebound_ref_price": rebound_ref,
                "label_bsp2_position_vs_origin_zs": position_vs_origin_zs,
                "label_bsp2_follow_path": follow_path,
            }
        )
        rows.append(row)

    return pd.DataFrame(rows, index=events.index)


def label_events_with_1m_path(
    events: pd.DataFrame,
    bars_1m: pd.DataFrame,
    config: FixedTpSlLabelConfig | None = None,
) -> pd.DataFrame:
    cfg = config or FixedTpSlLabelConfig()
    if cfg.same_bar_policy not in {"stop_first", "take_profit_first"}:
        raise ValueError("same_bar_policy must be stop_first or take_profit_first")
    if events.empty:
        return pd.DataFrame(index=events.index)

    bars = bars_1m.sort_index()
    times = pd.DatetimeIndex(pd.to_datetime(bars.index, utc=True))
    time_ns = times.view("int64")
    open_ = bars["open"].to_numpy(dtype="float64", copy=False)
    high = bars["high"].to_numpy(dtype="float64", copy=False)
    low = bars["low"].to_numpy(dtype="float64", copy=False)
    close = bars["close"].to_numpy(dtype="float64", copy=False)
    n = len(bars)
    max_hold = int(cfg.max_holding_minutes)
    cost = 2.0 * (float(cfg.fee_rate) + float(cfg.slippage_rate))

    rows: list[dict[str, object]] = []
    for _, event in events.iterrows():
        exec_time = pd.to_datetime(event["exec_time"], utc=True)
        signal_available_time = exec_time + pd.Timedelta(minutes=int(cfg.signal_timeframe_minutes))
        entry_pos = int(np.searchsorted(time_ns, signal_available_time.value, side="left"))
        is_buy = bool(event["is_buy"])
        direction = 1.0 if is_buy else -1.0

        row = {
            "signal_available_time": signal_available_time,
            "entry_time": pd.NaT,
            "entry_price": np.nan,
            "exit_time": pd.NaT,
            "exit_price": np.nan,
            "exit_reason": "no_future",
            "gross_return": np.nan,
            "net_return": np.nan,
            "mfe": np.nan,
            "mae": np.nan,
            "label": np.nan,
            "entry_bar_idx": entry_pos,
        }
        if entry_pos < 0 or entry_pos >= n:
            rows.append(row)
            continue

        entry_price = float(open_[entry_pos])
        if not np.isfinite(entry_price) or entry_price <= 0:
            rows.append(row)
            continue

        future_start = entry_pos
        future_end = min(n, entry_pos + max(1, max_hold))
        if future_start >= future_end:
            rows.append(row)
            continue

        if is_buy:
            tp_price = entry_price * (1.0 + float(cfg.take_profit_pct))
            sl_price = entry_price * (1.0 - float(cfg.stop_loss_pct))
        else:
            tp_price = entry_price * (1.0 - float(cfg.take_profit_pct))
            sl_price = entry_price * (1.0 + float(cfg.stop_loss_pct))

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
                if cfg.same_bar_policy == "take_profit_first":
                    exit_price = tp_price
                    exit_reason = "take_profit"
                else:
                    exit_price = sl_price
                    exit_reason = "stop_loss"
                break
            if hit_tp:
                exit_pos = pos
                exit_price = tp_price
                exit_reason = "take_profit"
                break
            if hit_sl:
                exit_pos = pos
                exit_price = sl_price
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
        row.update(
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
                "label": 1.0 if exit_reason == "take_profit" else 0.0,
            }
        )
        rows.append(row)

    return pd.DataFrame(rows, index=events.index)
