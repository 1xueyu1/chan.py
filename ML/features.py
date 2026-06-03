from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Set

import numpy as np
import pandas as pd

from Backtest.types import RawBSPEvent

try:
    from RustCore import RustChanEngine, RustCoreUnavailable
except Exception:  # pragma: no cover - Rust extension is optional
    RustChanEngine = None

    class RustCoreUnavailable(RuntimeError):
        pass


META_COLUMNS = [
    "symbol",
    "exec_time",
    "bsp_time",
    "is_buy",
    "bsp_type",
    "bsp_types_str",
    "trade_price",
    "klu_idx",
]

FEATURE_PREFIXES = ("bar_", "chan_", "bsp_", "mtf_")

MTF_RULES = (
    ("1h", "1h"),
    ("4h", "4h"),
    ("1d", "1D"),
)

CAPABILITY_CHAN_CONTEXT_DEPENDENCIES = (
    "chan_ctx_last_bi_is_up",
    "chan_ctx_last_bi_is_sure",
    "chan_ctx_last_bi_klu_cnt",
    "chan_ctx_last_seg_is_up",
    "chan_ctx_last_seg_is_sure",
    "chan_ctx_last_seg_bi_cnt",
    "chan_ctx_last_seg_klu_cnt",
    "chan_ctx_last_zs_bi_cnt",
    "chan_ctx_close_vs_last_zs_mid_pct",
)

CAPABILITY_CHAN_MTF_DEPENDENCY_SUFFIXES = (
    "event_pos_last_zs",
    "event_inside_last_zs",
    "last_zs_exists",
    "last_zs_is_sure",
    "last_zs_width_pct",
    "last_zs_bi_cnt",
    "signal_align_last_bi",
    "signal_align_last_seg",
    "signal_align_last_zs_mid",
    "last_bi_is_sure",
    "last_bi_klu_cnt",
    "last_seg_is_sure",
    "last_seg_bi_cnt",
    "last_seg_klu_cnt",
    "last_seg_zs_cnt",
    "last_seg_multi_zs_cnt",
)


def _capability_chan_dependency_columns() -> list[str]:
    columns = list(CAPABILITY_CHAN_CONTEXT_DEPENDENCIES)
    for label, _ in MTF_RULES:
        columns.extend(f"chan_mtf_{label}_{suffix}" for suffix in CAPABILITY_CHAN_MTF_DEPENDENCY_SUFFIXES)
    return columns


def normalize_bars(bars: pd.DataFrame) -> pd.DataFrame:
    if "open_time" in bars.columns:
        index = pd.to_datetime(bars["open_time"], utc=True)
        out = bars.drop(columns=["open_time"]).copy()
        out.index = index
    else:
        out = bars.copy(deep=False)
        out.index = pd.to_datetime(out.index, utc=True)

    out = out.sort_index()
    out = out[~out.index.duplicated(keep="last")]
    for col in ("open", "high", "low", "close", "volume"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out[["open", "high", "low", "close", "volume"]].dropna(subset=["open", "high", "low", "close"])


def datetime_index_to_ns(index) -> np.ndarray:
    return pd.DatetimeIndex(pd.to_datetime(index, utc=True)).astype("datetime64[ns, UTC]").view("int64")


def _safe_div(num: pd.Series, den: pd.Series | float) -> pd.Series:
    return num / (den + 1e-12)


def build_bar_feature_frame(
    bars: pd.DataFrame,
    required_columns: Optional[Iterable[str]] = None,
    normalized_bars: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Compute leakage-free bar features once per symbol."""
    required: Optional[Set[str]] = set(required_columns) if required_columns is not None else None

    def need(name: str) -> bool:
        return required is None or name in required

    df = normalized_bars if normalized_bars is not None else normalize_bars(bars)
    close = df["close"]
    open_ = df["open"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"].fillna(0.0)

    feat = pd.DataFrame(index=df.index)
    prev_close = close.shift(1)
    true_range = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr14 = true_range.rolling(14, min_periods=3).mean()

    for window in (1, 3, 4, 8, 12, 16, 24, 32):
        name = f"bar_ret_{window}"
        if need(name):
            feat[name] = close.pct_change(window)

    candle_range = (high - low).replace(0, np.nan)
    if need("bar_range_pct"):
        feat["bar_range_pct"] = _safe_div(high - low, close)
    if need("bar_body_direction"):
        feat["bar_body_direction"] = _safe_div(close - open_, close)
    if need("bar_body_pct"):
        feat["bar_body_pct"] = _safe_div((close - open_).abs(), close)
    if need("bar_upper_shadow_pct"):
        feat["bar_upper_shadow_pct"] = _safe_div(high - np.maximum(open_, close), candle_range)
    if need("bar_lower_shadow_pct"):
        feat["bar_lower_shadow_pct"] = _safe_div(np.minimum(open_, close) - low, candle_range)
    if need("bar_close_pos"):
        feat["bar_close_pos"] = _safe_div(close - low, candle_range)

    for window in (8, 21, 55):
        name = f"bar_ma_dist_{window}"
        if need(name):
            ma = close.rolling(window, min_periods=max(3, window // 3)).mean()
            feat[name] = _safe_div(close - ma, ma)

    for ret_window, z_window in ((8, 64), (16, 96)):
        name = f"bar_ret_z_{ret_window}_{z_window}"
        if need(name):
            ret = close.pct_change(ret_window)
            ret_mean = ret.rolling(z_window, min_periods=max(16, z_window // 4)).mean()
            ret_std = ret.rolling(z_window, min_periods=max(16, z_window // 4)).std()
            feat[name] = _safe_div(ret - ret_mean, ret_std)

    for window in (32, 96):
        name = f"bar_trend_efficiency_{window}"
        if need(name):
            net_move = (close - close.shift(window)).abs()
            path_move = close.diff().abs().rolling(window, min_periods=max(8, window // 4)).sum()
            feat[name] = _safe_div(net_move, path_move)

    if need("bar_ema12_dist") or need("bar_ema26_dist") or need("bar_macd_fast"):
        ema12 = close.ewm(span=12, adjust=False, min_periods=3).mean()
        ema26 = close.ewm(span=26, adjust=False, min_periods=5).mean()
        if need("bar_ema12_dist"):
            feat["bar_ema12_dist"] = _safe_div(close - ema12, ema12)
        if need("bar_ema26_dist"):
            feat["bar_ema26_dist"] = _safe_div(close - ema26, ema26)
        if need("bar_macd_fast"):
            feat["bar_macd_fast"] = _safe_div(ema12 - ema26, close)

    if need("bar_atr_pct_14") or need("bar_atr_z_64"):
        atr_pct_14 = _safe_div(atr14, close)
        if need("bar_atr_pct_14"):
            feat["bar_atr_pct_14"] = atr_pct_14
        if need("bar_atr_z_64"):
            atr_mean = atr_pct_14.rolling(64, min_periods=16).mean()
            atr_std = atr_pct_14.rolling(64, min_periods=16).std()
            feat["bar_atr_z_64"] = _safe_div(atr_pct_14 - atr_mean, atr_std)

    rv_needed = need("bar_rv_16") or need("bar_rv_64") or need("bar_rv_ratio_16_64")
    if rv_needed:
        ret1 = close.pct_change()
        rv16 = ret1.rolling(16, min_periods=4).std()
        rv64 = ret1.rolling(64, min_periods=16).std()
        if need("bar_rv_16"):
            feat["bar_rv_16"] = rv16
        if need("bar_rv_64"):
            feat["bar_rv_64"] = rv64
        if need("bar_rv_ratio_16_64"):
            feat["bar_rv_ratio_16_64"] = _safe_div(rv16, rv64)

    if need("bar_volume_ratio_20") or need("bar_volume_z_20") or need("bar_volume_ratio_5_20") or need("bar_volume_slope_5_20"):
        vol_mean5 = volume.rolling(5, min_periods=2).mean()
        vol_mean20 = volume.rolling(20, min_periods=5).mean()
        vol_std20 = volume.rolling(20, min_periods=5).std()
        if need("bar_volume_ratio_20"):
            feat["bar_volume_ratio_20"] = _safe_div(volume, vol_mean20)
        if need("bar_volume_z_20"):
            feat["bar_volume_z_20"] = _safe_div(volume - vol_mean20, vol_std20)
        if need("bar_volume_ratio_5_20"):
            feat["bar_volume_ratio_5_20"] = _safe_div(vol_mean5, vol_mean20)
        if need("bar_volume_slope_5_20"):
            feat["bar_volume_slope_5_20"] = _safe_div(vol_mean5 - vol_mean20, vol_mean20)

    for window in (20, 55):
        names = [f"bar_donchian_pos_{window}", f"bar_drawdown_{window}", f"bar_runup_{window}"]
        extra_names = [f"bar_breakout_up_{window}", f"bar_breakdown_down_{window}"]
        if any(need(name) for name in names + extra_names):
            roll_high = high.rolling(window, min_periods=max(5, window // 4)).max()
            roll_low = low.rolling(window, min_periods=max(5, window // 4)).min()
            if need(names[0]):
                feat[names[0]] = _safe_div(close - roll_low, roll_high - roll_low)
            if need(names[1]):
                feat[names[1]] = _safe_div(close - roll_high, roll_high)
            if need(names[2]):
                feat[names[2]] = _safe_div(close - roll_low, roll_low)
            if need(extra_names[0]):
                prev_high = high.shift(1).rolling(window, min_periods=max(5, window // 4)).max()
                feat[extra_names[0]] = _safe_div(close - prev_high, close)
            if need(extra_names[1]):
                prev_low = low.shift(1).rolling(window, min_periods=max(5, window // 4)).min()
                feat[extra_names[1]] = _safe_div(prev_low - close, close)

    if need("bar_range_compression_8_32"):
        range_pct = _safe_div(high - low, close)
        feat["bar_range_compression_8_32"] = _safe_div(
            range_pct.rolling(8, min_periods=3).mean(),
            range_pct.rolling(32, min_periods=8).mean(),
        )

    return feat.replace([np.inf, -np.inf], np.nan)


def build_confirmed_fractal_feature_frame(
    bars: pd.DataFrame,
    events: Sequence[RawBSPEvent],
    required_columns: Optional[Iterable[str]] = None,
    normalized_bars: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Build no-lookahead confirmed fractal features for events.

    A fractal at bar t-1 is only marked after bar t closes, so event sampling at
    t can safely use the latest confirmed top/bottom fractal.
    """
    if not events:
        return pd.DataFrame()
    required: Optional[Set[str]] = set(required_columns) if required_columns is not None else None
    if required is not None and not any(str(col).startswith("chan_fx_") for col in required):
        return pd.DataFrame(index=range(len(events)))

    df = normalized_bars if normalized_bars is not None else normalize_bars(bars)
    if df.empty:
        return pd.DataFrame(index=range(len(events)))
    high = df["high"]
    low = df["low"]
    close = df["close"]
    true_range = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = true_range.rolling(14, min_periods=3).mean()
    top_confirm = (high.shift(1) > high.shift(2)) & (high.shift(1) > high)
    bottom_confirm = (low.shift(1) < low.shift(2)) & (low.shift(1) < low)
    prev_ns = pd.Series(df.index.view("int64"), index=df.index).shift(1)

    state = pd.DataFrame(index=df.index)
    state["last_top_price"] = high.shift(1).where(top_confirm).ffill()
    state["last_bottom_price"] = low.shift(1).where(bottom_confirm).ffill()
    state["last_top_strength"] = (
        (high.shift(1) - pd.concat([high.shift(2), high], axis=1).max(axis=1)) / (atr + 1e-12)
    ).where(top_confirm).ffill()
    state["last_bottom_strength"] = (
        (pd.concat([low.shift(2), low], axis=1).min(axis=1) - low.shift(1)) / (atr + 1e-12)
    ).where(bottom_confirm).ffill()
    state["last_top_time_ns"] = prev_ns.where(top_confirm).ffill()
    state["last_bottom_time_ns"] = prev_ns.where(bottom_confirm).ffill()
    state["top_confirm_now"] = top_confirm.astype("float64")
    state["bottom_confirm_now"] = bottom_confirm.astype("float64")
    state["close"] = close
    state["atr"] = atr

    event_ns = _event_time_ns(events)
    sampled = _sample_feature_frame(state, event_ns)
    close_s = sampled["close"]
    atr_s = sampled["atr"].abs() + 1e-12
    is_buy = pd.Series([bool(event.is_buy) for event in events])
    top_dist = (sampled["last_top_price"] - close_s) / (close_s + 1e-12)
    bottom_dist = (close_s - sampled["last_bottom_price"]) / (close_s + 1e-12)
    top_age = (event_ns - sampled["last_top_time_ns"].to_numpy(dtype="float64")) / (15.0 * 60.0 * 1e9)
    bottom_age = (event_ns - sampled["last_bottom_time_ns"].to_numpy(dtype="float64")) / (15.0 * 60.0 * 1e9)
    nearest_is_bottom = bottom_dist.abs() <= top_dist.abs()

    out = pd.DataFrame(index=range(len(events)))
    out["chan_fx_last_top_dist_pct"] = top_dist.to_numpy(dtype="float64")
    out["chan_fx_last_bottom_dist_pct"] = bottom_dist.to_numpy(dtype="float64")
    out["chan_fx_last_top_dist_atr"] = ((sampled["last_top_price"] - close_s) / atr_s).to_numpy(dtype="float64")
    out["chan_fx_last_bottom_dist_atr"] = ((close_s - sampled["last_bottom_price"]) / atr_s).to_numpy(dtype="float64")
    out["chan_fx_last_top_age_bars"] = top_age
    out["chan_fx_last_bottom_age_bars"] = bottom_age
    out["chan_fx_last_top_age_norm"] = np.tanh(top_age / 32.0)
    out["chan_fx_last_bottom_age_norm"] = np.tanh(bottom_age / 32.0)
    out["chan_fx_last_top_strength"] = sampled["last_top_strength"].to_numpy(dtype="float64")
    out["chan_fx_last_bottom_strength"] = sampled["last_bottom_strength"].to_numpy(dtype="float64")
    out["chan_fx_confirm_top_now"] = sampled["top_confirm_now"].to_numpy(dtype="float64")
    out["chan_fx_confirm_bottom_now"] = sampled["bottom_confirm_now"].to_numpy(dtype="float64")
    out["chan_fx_nearest_is_bottom"] = nearest_is_bottom.astype("float64")
    out["chan_fx_signal_nearest_match"] = np.where(is_buy, nearest_is_bottom, ~nearest_is_bottom).astype("float64")
    out["chan_fx_signal_support_dist_atr"] = np.where(is_buy, out["chan_fx_last_bottom_dist_atr"], out["chan_fx_last_top_dist_atr"])
    out["chan_fx_signal_pressure_dist_atr"] = np.where(is_buy, out["chan_fx_last_top_dist_atr"], out["chan_fx_last_bottom_dist_atr"])
    out["chan_fx_signal_fractal_strength"] = np.where(is_buy, out["chan_fx_last_bottom_strength"], out["chan_fx_last_top_strength"])
    out["chan_fx_opposite_fractal_strength"] = np.where(is_buy, out["chan_fx_last_top_strength"], out["chan_fx_last_bottom_strength"])
    out["chan_fx_strength_balance"] = out["chan_fx_signal_fractal_strength"] - out["chan_fx_opposite_fractal_strength"]
    out["chan_fx_reversal_support_score"] = (
        np.exp(-out["chan_fx_signal_support_dist_atr"].abs() / 3.0)
        * out["chan_fx_signal_fractal_strength"].fillna(0.0).clip(lower=0.0, upper=2.0)
        / 2.0
    ).clip(lower=0.0, upper=1.0)
    out["chan_fx_break_pressure_score"] = (
        np.exp(-out["chan_fx_signal_pressure_dist_atr"].abs() / 3.0)
        * out["chan_fx_opposite_fractal_strength"].fillna(0.0).clip(lower=0.0, upper=2.0)
        / 2.0
    ).clip(lower=0.0, upper=1.0)
    out["chan_fx_signal_score"] = (
        0.35 * out["chan_fx_signal_nearest_match"]
        + 0.25 * out["chan_fx_reversal_support_score"]
        + 0.20 * (out["chan_fx_strength_balance"] + 0.5).clip(lower=0.0, upper=1.0)
        + 0.20 * (1.0 - (out["chan_fx_signal_support_dist_atr"].abs() / 6.0).clip(lower=0.0, upper=1.0))
    ).clip(lower=0.0, upper=1.0)
    if required is not None:
        keep = [col for col in out.columns if col in required]
        out = out.reindex(columns=keep)
    return out.replace([np.inf, -np.inf], np.nan)


def _resample_completed_bars(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }
    return (
        df.resample(rule, label="right", closed="left", origin="epoch")
        .agg(agg)
        .dropna(subset=["open", "high", "low", "close"])
    )


def _sample_feature_frame(frame: pd.DataFrame, event_ns: np.ndarray) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(index=range(len(event_ns)))
    frame_ns = datetime_index_to_ns(frame.index)
    loc = np.searchsorted(frame_ns, event_ns, side="right") - 1
    valid = loc >= 0
    sampled = frame.iloc[np.clip(loc, 0, max(len(frame) - 1, 0))].reset_index(drop=True)
    sampled.loc[~valid, :] = np.nan
    return sampled


def _safe_div_array(num, den) -> np.ndarray:
    return np.asarray(num, dtype="float64") / (np.asarray(den, dtype="float64") + 1e-12)


def build_mtf_feature_frame(
    bars: pd.DataFrame,
    events: Sequence[RawBSPEvent],
    required_columns: Optional[Iterable[str]] = None,
    normalized_bars: Optional[pd.DataFrame] = None,
    resampled_bars: Optional[dict[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    """Build completed higher-period alignment and nesting features.

    Higher-period candles are labelled at their close time, so an event inside
    an unfinished 1h/4h/1d candle only sees the last completed higher candle.
    """
    if not events:
        return pd.DataFrame()

    required: Optional[Set[str]] = set(required_columns) if required_columns is not None else None
    if required is not None and not any(str(col).startswith("mtf_") for col in required):
        return pd.DataFrame(index=range(len(events)))

    def need(name: str) -> bool:
        return required is None or name in required

    df = normalized_bars if normalized_bars is not None else normalize_bars(bars)
    resampled = resampled_bars if resampled_bars is not None else {}
    event_ns = _event_time_ns(events)
    event_price = np.asarray([float(event.trade_price) for event in events], dtype="float64")
    signal_dir = np.asarray([1.0 if bool(event.is_buy) else -1.0 for event in events], dtype="float64")
    frames: List[pd.DataFrame] = []

    for label, rule in MTF_RULES:
        htf = resampled.get(label)
        if htf is None:
            htf = _resample_completed_bars(df, rule)
            resampled[label] = htf
        prefix = f"mtf_{label}"
        if htf.empty:
            continue

        close = htf["close"]
        high = htf["high"]
        low = htf["low"]
        volume = htf["volume"].fillna(0.0)
        state = pd.DataFrame(index=htf.index)
        state[f"__{prefix}_close"] = close

        for window in (1, 3, 8):
            name = f"{prefix}_ret_{window}"
            if need(name) or need(f"{prefix}_signal_align_ret_3"):
                state[name] = close.pct_change(window)

        for window in (8, 21):
            ma = close.rolling(window, min_periods=max(3, window // 3)).mean()
            state[f"__{prefix}_ma_{window}"] = ma
            name = f"{prefix}_ma_dist_{window}"
            if need(name):
                state[name] = _safe_div(close - ma, ma)

        prev_close = close.shift(1)
        true_range = pd.concat(
            [
                high - low,
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr14 = true_range.rolling(14, min_periods=3).mean()
        if need(f"{prefix}_atr_pct_14"):
            state[f"{prefix}_atr_pct_14"] = _safe_div(atr14, close)
        if need(f"{prefix}_rv_16"):
            state[f"{prefix}_rv_16"] = close.pct_change().rolling(16, min_periods=4).std()
        if need(f"{prefix}_volume_ratio_20"):
            vol_mean20 = volume.rolling(20, min_periods=5).mean()
            state[f"{prefix}_volume_ratio_20"] = _safe_div(volume, vol_mean20)

        for window in (20, 55):
            roll_high = high.rolling(window, min_periods=max(5, window // 4)).max()
            roll_low = low.rolling(window, min_periods=max(5, window // 4)).min()
            roll_mid = (roll_high + roll_low) / 2.0
            width = roll_high - roll_low
            state[f"__{prefix}_range_high_{window}"] = roll_high
            state[f"__{prefix}_range_low_{window}"] = roll_low
            state[f"__{prefix}_range_mid_{window}"] = roll_mid
            if need(f"{prefix}_range_width_{window}"):
                state[f"{prefix}_range_width_{window}"] = _safe_div(width, close)
            if need(f"{prefix}_close_pos_{window}"):
                state[f"{prefix}_close_pos_{window}"] = _safe_div(close - roll_low, width)

        sampled = _sample_feature_frame(state, event_ns)
        out = pd.DataFrame(index=range(len(events)))
        for col in sampled.columns:
            if not col.startswith("__") and need(col):
                out[col] = sampled[col]

        ret3_name = f"{prefix}_ret_3"
        align_ret_name = f"{prefix}_signal_align_ret_3"
        if need(align_ret_name):
            ret3 = sampled.get(ret3_name, pd.Series(np.nan, index=out.index)).to_numpy(dtype="float64")
            out[align_ret_name] = np.where(np.isfinite(ret3), (signal_dir * ret3 > 0).astype("float64"), np.nan)

        align_ma_name = f"{prefix}_signal_align_ma_21"
        if need(align_ma_name):
            sampled_close = sampled[f"__{prefix}_close"].to_numpy(dtype="float64")
            ma21 = sampled[f"__{prefix}_ma_21"].to_numpy(dtype="float64")
            aligned = ((signal_dir > 0) & (sampled_close >= ma21)) | ((signal_dir < 0) & (sampled_close <= ma21))
            out[align_ma_name] = np.where(np.isfinite(ma21), aligned.astype("float64"), np.nan)

        for window in (20, 55):
            h = sampled[f"__{prefix}_range_high_{window}"].to_numpy(dtype="float64")
            l = sampled[f"__{prefix}_range_low_{window}"].to_numpy(dtype="float64")
            m = sampled[f"__{prefix}_range_mid_{window}"].to_numpy(dtype="float64")
            width = h - l
            if need(f"{prefix}_event_pos_{window}"):
                out[f"{prefix}_event_pos_{window}"] = _safe_div_array(event_price - l, width)
            if need(f"{prefix}_event_dist_mid_{window}"):
                out[f"{prefix}_event_dist_mid_{window}"] = _safe_div_array(event_price - m, m)
            if need(f"{prefix}_event_dist_upper_{window}"):
                out[f"{prefix}_event_dist_upper_{window}"] = _safe_div_array(h - event_price, event_price)
            if need(f"{prefix}_event_dist_lower_{window}"):
                out[f"{prefix}_event_dist_lower_{window}"] = _safe_div_array(event_price - l, event_price)
            align_range_name = f"{prefix}_signal_align_range_mid_{window}"
            if need(align_range_name):
                aligned = ((signal_dir > 0) & (event_price >= m)) | ((signal_dir < 0) & (event_price <= m))
                out[align_range_name] = np.where(np.isfinite(m), aligned.astype("float64"), np.nan)

        if not out.empty:
            frames.append(out)

    if not frames:
        return pd.DataFrame(index=range(len(events)))
    return pd.concat(frames, axis=1).replace([np.inf, -np.inf], np.nan)


def _to_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _direction_is_up(value) -> bool:
    return str(value).lower().endswith("up")


def _direction_is_down(value) -> bool:
    return str(value).lower().endswith("down")


def _range_pos_array(value, low, high) -> np.ndarray:
    return _safe_div_array(np.asarray(value, dtype="float64") - np.asarray(low, dtype="float64"), np.asarray(high, dtype="float64") - np.asarray(low, dtype="float64"))


def _chan_mtf_state_row(prefix: str, snapshot: dict, close_price: float) -> dict[str, float]:
    row: dict[str, float] = {}
    counts = snapshot.get("counts") or {}
    for src, dst in (
        ("klcs", "klc_cnt"),
        ("bis", "bi_cnt"),
        ("segs", "seg_cnt"),
        ("segs_seg", "segseg_cnt"),
        ("bzs", "zs_cnt"),
        ("szs", "seg_zs_cnt"),
        ("bi_bsp", "bi_bsp_cnt"),
        ("seg_bsp", "seg_bsp_cnt"),
    ):
        row[f"{prefix}_{dst}"] = float(counts.get(src, 0) or 0)

    last_klc = snapshot.get("last_klc") or {}
    klc_high = _to_float(last_klc.get("high"))
    klc_low = _to_float(last_klc.get("low"))
    row[f"__{prefix}_last_klc_high"] = klc_high
    row[f"__{prefix}_last_klc_low"] = klc_low
    row[f"{prefix}_last_klc_unit_cnt"] = float(last_klc.get("unit_count", 0) or 0)
    row[f"{prefix}_last_klc_range_pct"] = _to_float((klc_high - klc_low) / (float(close_price) + 1e-12))

    bis = snapshot.get("bis") or []
    last_bi = snapshot.get("last_bi") or (bis[-1] if bis else None)
    if last_bi:
        begin_val = _to_float(last_bi.get("begin_val"))
        end_val = _to_float(last_bi.get("end_val"))
        high = max(begin_val, end_val)
        low = min(begin_val, end_val)
        is_up = _direction_is_up(last_bi.get("direction"))
        is_down = _direction_is_down(last_bi.get("direction"))
        row[f"{prefix}_last_bi_exists"] = 1.0
        row[f"{prefix}_last_bi_is_sure"] = 1.0 if bool(last_bi.get("is_sure")) else 0.0
        row[f"{prefix}_last_bi_is_up"] = 1.0 if is_up else 0.0
        row[f"{prefix}_last_bi_amp_pct"] = _to_float(float(last_bi.get("amp", high - low) or 0.0) / (begin_val + 1e-12))
        row[f"{prefix}_last_bi_klu_cnt"] = float(last_bi.get("klu_cnt", 0) or 0)
        row[f"{prefix}_last_bi_klc_cnt"] = float(last_bi.get("klc_cnt", 0) or 0)
        row[f"__{prefix}_last_bi_high"] = high
        row[f"__{prefix}_last_bi_low"] = low
        row[f"__{prefix}_last_bi_end"] = end_val
        row[f"__{prefix}_last_bi_is_up"] = 1.0 if is_up else 0.0
        row[f"__{prefix}_last_bi_is_down"] = 1.0 if is_down else 0.0
    else:
        row[f"{prefix}_last_bi_exists"] = 0.0

    segs = snapshot.get("segs") or []
    last_seg = snapshot.get("last_seg") or (segs[-1] if segs else None)
    if last_seg:
        begin_val = _to_float(last_seg.get("begin_val"))
        end_val = _to_float(last_seg.get("end_val"))
        high = max(begin_val, end_val)
        low = min(begin_val, end_val)
        is_up = _direction_is_up(last_seg.get("direction"))
        is_down = _direction_is_down(last_seg.get("direction"))
        row[f"{prefix}_last_seg_exists"] = 1.0
        row[f"{prefix}_last_seg_is_sure"] = 1.0 if bool(last_seg.get("is_sure")) else 0.0
        row[f"{prefix}_last_seg_is_up"] = 1.0 if is_up else 0.0
        row[f"{prefix}_last_seg_amp_pct"] = _to_float(float(last_seg.get("amp", high - low) or 0.0) / (begin_val + 1e-12))
        row[f"{prefix}_last_seg_klu_cnt"] = float(last_seg.get("klu_cnt", 0) or 0)
        row[f"{prefix}_last_seg_bi_cnt"] = float(last_seg.get("element_cnt", 0) or 0)
        row[f"{prefix}_last_seg_zs_cnt"] = float(len(last_seg.get("zs_lst") or []))
        row[f"{prefix}_last_seg_multi_zs_cnt"] = float(last_seg.get("multi_bi_zs_cnt", 0) or 0)
        row[f"__{prefix}_last_seg_high"] = high
        row[f"__{prefix}_last_seg_low"] = low
        row[f"__{prefix}_last_seg_end"] = end_val
        row[f"__{prefix}_last_seg_is_up"] = 1.0 if is_up else 0.0
        row[f"__{prefix}_last_seg_is_down"] = 1.0 if is_down else 0.0
    else:
        row[f"{prefix}_last_seg_exists"] = 0.0

    zss = snapshot.get("bzs") or []
    last_zs = snapshot.get("last_zs") or (zss[-1] if zss else None)
    if last_zs:
        low = _to_float(last_zs.get("low"))
        high = _to_float(last_zs.get("high"))
        mid = _to_float(last_zs.get("mid"))
        peak_high = _to_float(last_zs.get("peak_high", high))
        peak_low = _to_float(last_zs.get("peak_low", low))
        row[f"{prefix}_last_zs_exists"] = 1.0
        row[f"{prefix}_last_zs_is_sure"] = 1.0 if bool(last_zs.get("is_sure")) else 0.0
        row[f"{prefix}_last_zs_width_pct"] = _to_float((high - low) / (mid + 1e-12))
        row[f"{prefix}_last_zs_peak_width_pct"] = _to_float((peak_high - peak_low) / (mid + 1e-12))
        row[f"{prefix}_last_zs_bi_cnt"] = float(len(last_zs.get("element_list") or []))
        row[f"__{prefix}_last_zs_low"] = low
        row[f"__{prefix}_last_zs_high"] = high
        row[f"__{prefix}_last_zs_mid"] = mid
    else:
        row[f"{prefix}_last_zs_exists"] = 0.0

    return row


def _build_chan_mtf_state_frame(htf: pd.DataFrame, label: str, rust_config_path: Optional[str]) -> pd.DataFrame:
    if htf.empty or RustChanEngine is None:
        return pd.DataFrame(index=htf.index)

    engine = RustChanEngine(freq=label, config_path=rust_config_path)
    rows: list[dict[str, float]] = []
    prefix = f"chan_mtf_{label}"
    for ts, row in htf.iterrows():
        engine.push_bar(
            ts.to_pydatetime(),
            float(row["open"]),
            float(row["high"]),
            float(row["low"]),
            float(row["close"]),
            float(row.get("volume", 0.0) or 0.0),
        )
        engine.cal_seg_and_zs()
        snapshot = engine.tail_snapshot() if hasattr(engine, "tail_snapshot") else engine.snapshot()
        rows.append(_chan_mtf_state_row(prefix, snapshot, float(row["close"])))
    return pd.DataFrame(rows, index=htf.index)


def build_chan_mtf_feature_frame(
    bars: pd.DataFrame,
    events: Sequence[RawBSPEvent],
    required_columns: Optional[Iterable[str]] = None,
    rust_config_path: Optional[str] = None,
    normalized_bars: Optional[pd.DataFrame] = None,
    resampled_bars: Optional[dict[str, pd.DataFrame]] = None,
) -> pd.DataFrame:
    """Build event features from completed higher-period Rust Chan states."""
    if not events:
        return pd.DataFrame()

    required: Optional[Set[str]] = set(required_columns) if required_columns is not None else None
    if required is not None and not any(str(col).startswith("chan_mtf_") for col in required):
        return pd.DataFrame(index=range(len(events)))
    if RustChanEngine is None:
        return pd.DataFrame(index=range(len(events)))

    def need(name: str) -> bool:
        return required is None or name in required

    df = normalized_bars if normalized_bars is not None else normalize_bars(bars)
    resampled = resampled_bars if resampled_bars is not None else {}
    event_ns = _event_time_ns(events)
    event_price = np.asarray([float(event.trade_price) for event in events], dtype="float64")
    signal_dir = np.asarray([1.0 if bool(event.is_buy) else -1.0 for event in events], dtype="float64")
    frames: List[pd.DataFrame] = []

    for label, rule in MTF_RULES:
        prefix = f"chan_mtf_{label}"
        if required is not None and not any(str(col).startswith(f"{prefix}_") for col in required):
            continue
        htf = resampled.get(label)
        if htf is None:
            htf = _resample_completed_bars(df, rule)
            resampled[label] = htf
        if htf.empty:
            continue
        try:
            state = _build_chan_mtf_state_frame(htf, label, rust_config_path)
        except RustCoreUnavailable:
            continue
        if state.empty:
            continue

        sampled = _sample_feature_frame(state, event_ns)
        out = pd.DataFrame(index=range(len(events)))
        for col in sampled.columns:
            if not col.startswith("__") and need(col):
                out[col] = sampled[col]

        if need(f"{prefix}_event_pos_last_klc"):
            out[f"{prefix}_event_pos_last_klc"] = _range_pos_array(
                event_price,
                sampled.get(f"__{prefix}_last_klc_low", np.nan),
                sampled.get(f"__{prefix}_last_klc_high", np.nan),
            )

        if need(f"{prefix}_event_pos_last_bi"):
            out[f"{prefix}_event_pos_last_bi"] = _range_pos_array(
                event_price,
                sampled.get(f"__{prefix}_last_bi_low", np.nan),
                sampled.get(f"__{prefix}_last_bi_high", np.nan),
            )
        if need(f"{prefix}_event_vs_last_bi_end_pct"):
            end = np.asarray(sampled.get(f"__{prefix}_last_bi_end", np.nan), dtype="float64")
            out[f"{prefix}_event_vs_last_bi_end_pct"] = _safe_div_array(event_price - end, end)
        if need(f"{prefix}_signal_align_last_bi"):
            is_up = np.asarray(sampled.get(f"__{prefix}_last_bi_is_up", np.nan), dtype="float64")
            is_down = np.asarray(sampled.get(f"__{prefix}_last_bi_is_down", np.nan), dtype="float64")
            aligned = ((signal_dir > 0) & (is_up > 0.5)) | ((signal_dir < 0) & (is_down > 0.5))
            out[f"{prefix}_signal_align_last_bi"] = np.where(np.isfinite(is_up + is_down), aligned.astype("float64"), np.nan)

        if need(f"{prefix}_event_pos_last_seg"):
            out[f"{prefix}_event_pos_last_seg"] = _range_pos_array(
                event_price,
                sampled.get(f"__{prefix}_last_seg_low", np.nan),
                sampled.get(f"__{prefix}_last_seg_high", np.nan),
            )
        if need(f"{prefix}_event_vs_last_seg_end_pct"):
            end = np.asarray(sampled.get(f"__{prefix}_last_seg_end", np.nan), dtype="float64")
            out[f"{prefix}_event_vs_last_seg_end_pct"] = _safe_div_array(event_price - end, end)
        if need(f"{prefix}_signal_align_last_seg"):
            is_up = np.asarray(sampled.get(f"__{prefix}_last_seg_is_up", np.nan), dtype="float64")
            is_down = np.asarray(sampled.get(f"__{prefix}_last_seg_is_down", np.nan), dtype="float64")
            aligned = ((signal_dir > 0) & (is_up > 0.5)) | ((signal_dir < 0) & (is_down > 0.5))
            out[f"{prefix}_signal_align_last_seg"] = np.where(np.isfinite(is_up + is_down), aligned.astype("float64"), np.nan)

        zs_low = np.asarray(sampled.get(f"__{prefix}_last_zs_low", np.nan), dtype="float64")
        zs_high = np.asarray(sampled.get(f"__{prefix}_last_zs_high", np.nan), dtype="float64")
        zs_mid = np.asarray(sampled.get(f"__{prefix}_last_zs_mid", np.nan), dtype="float64")
        if need(f"{prefix}_event_inside_last_zs"):
            inside = (event_price >= zs_low) & (event_price <= zs_high)
            out[f"{prefix}_event_inside_last_zs"] = np.where(np.isfinite(zs_low + zs_high), inside.astype("float64"), np.nan)
        if need(f"{prefix}_event_pos_last_zs"):
            out[f"{prefix}_event_pos_last_zs"] = _range_pos_array(event_price, zs_low, zs_high)
        if need(f"{prefix}_event_vs_last_zs_mid_pct"):
            out[f"{prefix}_event_vs_last_zs_mid_pct"] = _safe_div_array(event_price - zs_mid, zs_mid)
        if need(f"{prefix}_event_vs_last_zs_high_pct"):
            out[f"{prefix}_event_vs_last_zs_high_pct"] = _safe_div_array(event_price - zs_high, zs_high)
        if need(f"{prefix}_event_vs_last_zs_low_pct"):
            out[f"{prefix}_event_vs_last_zs_low_pct"] = _safe_div_array(event_price - zs_low, zs_low)
        if need(f"{prefix}_signal_align_last_zs_mid"):
            aligned = ((signal_dir > 0) & (event_price >= zs_mid)) | ((signal_dir < 0) & (event_price <= zs_mid))
            out[f"{prefix}_signal_align_last_zs_mid"] = np.where(np.isfinite(zs_mid), aligned.astype("float64"), np.nan)

        if not out.empty:
            frames.append(out)

    if not frames:
        return pd.DataFrame(index=range(len(events)))
    return pd.concat(frames, axis=1).replace([np.inf, -np.inf], np.nan)


def _event_time_ns(events: Sequence[RawBSPEvent]) -> np.ndarray:
    return datetime_index_to_ns([event.exec_time for event in events])


def _bsp_flags(event: RawBSPEvent) -> dict[str, float]:
    text = str(event.bsp_types_str or event.bsp_type).lower()
    return {
        "bsp_is_buy": 1.0 if event.is_buy else 0.0,
        "bsp_is_1": 1.0 if "1" in text else 0.0,
        "bsp_is_1p": 1.0 if "1p" in text else 0.0,
        "bsp_is_2": 1.0 if "2" in text else 0.0,
        "bsp_is_2s": 1.0 if "2s" in text else 0.0,
        "bsp_is_3a": 1.0 if "3a" in text else 0.0,
        "bsp_is_3b": 1.0 if "3b" in text else 0.0,
    }


def build_capability_context_feature_frame(
    events: Sequence[RawBSPEvent],
    required_columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Build leakage-free event-history features for capability-v2 models."""
    if not events:
        return pd.DataFrame()

    required: Optional[Set[str]] = set(required_columns) if required_columns is not None else None
    context_prefixes = (
        "cap_event_",
        "cap_same_side_",
        "cap_opposite_side_",
        "cap_side_",
        "cap_minutes_",
        "cap_events_",
    )

    def is_context_col(name: object) -> bool:
        return any(str(name).startswith(prefix) for prefix in context_prefixes)

    if required is not None and not any(is_context_col(col) for col in required):
        return pd.DataFrame(index=range(len(events)))

    def need(name: str) -> bool:
        return required is None or name in required

    rows = pd.DataFrame(
        {
            "__pos": np.arange(len(events), dtype="int64"),
            "__symbol": [str(event.symbol) for event in events],
            "__time_ns": _event_time_ns(events),
            "__side": np.asarray([1 if bool(event.is_buy) else -1 for event in events], dtype="int8"),
        }
    ).sort_values(["__symbol", "__time_ns", "__pos"])

    out = pd.DataFrame(index=range(len(events)))
    windows = (8, 32, 96, 192)
    requested_windows = [
        window
        for window in windows
        if any(
            need(f"cap_{name}_{window}")
            for name in (
                "event_count",
                "same_side_count",
                "opposite_side_count",
                "same_side_ratio",
                "side_imbalance",
                "events_per_day",
            )
        )
    ]

    for _, group in rows.groupby("__symbol", sort=False):
        positions = group["__pos"].to_numpy(dtype="int64", copy=False)
        times = group["__time_ns"].to_numpy(dtype="int64", copy=False)
        sides = group["__side"].to_numpy(dtype="int8", copy=False)
        n = len(group)

        if need("cap_event_seq"):
            out.loc[positions, "cap_event_seq"] = np.arange(n, dtype="float64")

        if need("cap_minutes_since_prev_event"):
            delta = np.full(n, np.nan, dtype="float64")
            if n > 1:
                delta[1:] = (times[1:] - times[:-1]) / 60_000_000_000.0
            out.loc[positions, "cap_minutes_since_prev_event"] = delta

        if need("cap_minutes_since_prev_same_side") or need("cap_minutes_since_prev_opposite_side"):
            last_time_by_side: dict[int, int] = {}
            same_delta = np.full(n, np.nan, dtype="float64")
            opp_delta = np.full(n, np.nan, dtype="float64")
            for i, (ts, side) in enumerate(zip(times, sides)):
                same_ts = last_time_by_side.get(int(side))
                opp_ts = last_time_by_side.get(int(-side))
                if same_ts is not None:
                    same_delta[i] = (ts - same_ts) / 60_000_000_000.0
                if opp_ts is not None:
                    opp_delta[i] = (ts - opp_ts) / 60_000_000_000.0
                last_time_by_side[int(side)] = int(ts)
            if need("cap_minutes_since_prev_same_side"):
                out.loc[positions, "cap_minutes_since_prev_same_side"] = same_delta
            if need("cap_minutes_since_prev_opposite_side"):
                out.loc[positions, "cap_minutes_since_prev_opposite_side"] = opp_delta

        buy = (sides > 0).astype("float64")
        sell = (sides < 0).astype("float64")
        buy_cum = np.concatenate([[0.0], np.cumsum(buy)])
        sell_cum = np.concatenate([[0.0], np.cumsum(sell)])
        idx = np.arange(n)

        if need("cap_same_side_streak") or need("cap_opposite_side_streak"):
            same_streak = np.zeros(n, dtype="float64")
            opposite_streak = np.zeros(n, dtype="float64")
            last_side = 0
            streak = 0
            for i, side in enumerate(sides):
                if int(side) == int(last_side):
                    streak += 1
                else:
                    streak = 1
                same_streak[i] = streak
                opposite_streak[i] = 0.0 if i == 0 or int(side) == int(sides[i - 1]) else 1.0
                last_side = int(side)
            if need("cap_same_side_streak"):
                out.loc[positions, "cap_same_side_streak"] = same_streak
            if need("cap_opposite_side_streak"):
                out.loc[positions, "cap_opposite_side_streak"] = opposite_streak

        for window in requested_windows:
            start = np.maximum(0, idx - window)
            buy_count = buy_cum[idx] - buy_cum[start]
            sell_count = sell_cum[idx] - sell_cum[start]
            event_count = buy_count + sell_count
            same = np.where(sides > 0, buy_count, sell_count)
            opposite = np.where(sides > 0, sell_count, buy_count)

            if need(f"cap_event_count_{window}"):
                out.loc[positions, f"cap_event_count_{window}"] = event_count
            if need(f"cap_same_side_count_{window}"):
                out.loc[positions, f"cap_same_side_count_{window}"] = same
            if need(f"cap_opposite_side_count_{window}"):
                out.loc[positions, f"cap_opposite_side_count_{window}"] = opposite
            if need(f"cap_same_side_ratio_{window}"):
                out.loc[positions, f"cap_same_side_ratio_{window}"] = same / (event_count + 1e-12)
            if need(f"cap_side_imbalance_{window}"):
                out.loc[positions, f"cap_side_imbalance_{window}"] = (same - opposite) / (event_count + 1e-12)
            if need(f"cap_events_per_day_{window}"):
                elapsed_days = np.full(n, np.nan, dtype="float64")
                has_history = idx > start
                elapsed_days[has_history] = (times[has_history] - times[start[has_history]]) / 86_400_000_000_000.0
                out.loc[positions, f"cap_events_per_day_{window}"] = event_count / (elapsed_days + 1e-12)

    if required is not None:
        for col in required:
            if is_context_col(col) and col not in out.columns:
                out[col] = np.nan
    return out.replace([np.inf, -np.inf], np.nan)


def _series_or_nan(frame: pd.DataFrame, name: str) -> pd.Series:
    if name in frame.columns:
        return pd.to_numeric(frame[name], errors="coerce")
    return pd.Series(np.nan, index=frame.index, dtype="float64")


def _finite_ratio_sum(values: list[pd.Series]) -> tuple[pd.Series, pd.Series]:
    if not values:
        empty = pd.Series(np.nan)
        return empty, empty
    matrix = pd.concat(values, axis=1)
    available = matrix.notna().sum(axis=1).astype("float64")
    total = matrix.fillna(0.0).sum(axis=1)
    ratio = total / available.replace(0.0, np.nan)
    return ratio, available


def _bsp_divergence_like(frame: pd.DataFrame) -> pd.Series:
    text = (
        frame.get("bsp_types_str", pd.Series("", index=frame.index))
        .fillna("")
        .astype(str)
        .str.lower()
    )
    return (
        text.str.contains("1p")
        | text.str.contains("2")
        | text.str.contains("2s")
        | text.str.contains("3a")
        | text.str.contains("3b")
    ).astype("float64")


def build_capability_chan_feature_frame(
    frame: pd.DataFrame,
    required_columns: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Build capability-v2 Chan structure features from base Chan columns."""
    if frame.empty:
        return pd.DataFrame(index=frame.index)

    required: Optional[Set[str]] = set(required_columns) if required_columns is not None else None
    if required is not None and not any(str(col).startswith("cap_chan_") for col in required):
        return pd.DataFrame(index=frame.index)

    def need(name: str) -> bool:
        return required is None or name in required

    out = pd.DataFrame(index=frame.index)
    is_buy = frame["is_buy"].astype(bool) if "is_buy" in frame.columns else pd.Series(False, index=frame.index)
    signal_dir = pd.Series(np.where(is_buy, 1.0, -1.0), index=frame.index)
    divergence_like = _bsp_divergence_like(frame)

    body_dir = _series_or_nan(frame, "bar_body_direction")
    close_pos = _series_or_nan(frame, "bar_close_pos")
    ma21_dist = _series_or_nan(frame, "bar_ma_dist_21")
    volume_ratio = _series_or_nan(frame, "bar_volume_ratio_20")
    breakout20 = _series_or_nan(frame, "bar_breakout_up_20")
    breakdown20 = _series_or_nan(frame, "bar_breakdown_down_20")
    breakout55 = _series_or_nan(frame, "bar_breakout_up_55")
    breakdown55 = _series_or_nan(frame, "bar_breakdown_down_55")
    ret3 = _series_or_nan(frame, "bar_ret_3")
    ret8 = _series_or_nan(frame, "bar_ret_8")
    lower_shadow = _series_or_nan(frame, "bar_lower_shadow_pct")
    upper_shadow = _series_or_nan(frame, "bar_upper_shadow_pct")

    micro_parts: list[pd.Series] = []
    if body_dir.notna().any():
        body_align = (signal_dir * body_dir > 0.0).astype("float64")
        micro_parts.append(body_align)
        if need("cap_micro_signal_body_align"):
            out["cap_micro_signal_body_align"] = body_align.where(body_dir.notna())
    if ret3.notna().any():
        ret3_align = signal_dir * ret3
        if need("cap_micro_signal_ret3_signed"):
            out["cap_micro_signal_ret3_signed"] = ret3_align
    if ret8.notna().any():
        ret8_align = signal_dir * ret8
        if need("cap_micro_signal_ret8_signed"):
            out["cap_micro_signal_ret8_signed"] = ret8_align
    if ma21_dist.notna().any():
        ma21_align = (signal_dir * ma21_dist >= 0.0).astype("float64")
        micro_parts.append(ma21_align)
        if need("cap_micro_signal_ma21_align"):
            out["cap_micro_signal_ma21_align"] = ma21_align.where(ma21_dist.notna())
    if volume_ratio.notna().any():
        volume_confirm = (volume_ratio >= 1.1).astype("float64")
        micro_parts.append(volume_confirm)
        if need("cap_micro_volume_confirm"):
            out["cap_micro_volume_confirm"] = volume_confirm.where(volume_ratio.notna())
    if close_pos.notna().any():
        close_pos_signal = pd.Series(np.where(is_buy, close_pos, 1.0 - close_pos), index=frame.index)
        if need("cap_micro_signal_close_pos"):
            out["cap_micro_signal_close_pos"] = close_pos_signal
    if lower_shadow.notna().any() or upper_shadow.notna().any():
        shadow_support = pd.Series(np.where(is_buy, lower_shadow, upper_shadow), index=frame.index)
        if need("cap_micro_signal_shadow_support"):
            out["cap_micro_signal_shadow_support"] = shadow_support
    if breakout20.notna().any() or breakdown20.notna().any():
        signal_breakout20 = pd.Series(np.where(is_buy, breakout20, breakdown20), index=frame.index)
        if need("cap_micro_signal_breakout_20"):
            out["cap_micro_signal_breakout_20"] = signal_breakout20
    if breakout55.notna().any() or breakdown55.notna().any():
        signal_breakout55 = pd.Series(np.where(is_buy, breakout55, breakdown55), index=frame.index)
        if need("cap_micro_signal_breakout_55"):
            out["cap_micro_signal_breakout_55"] = signal_breakout55
    if need("cap_micro_confirm_score"):
        out["cap_micro_confirm_score"], _ = _finite_ratio_sum(micro_parts)

    base_alignments: list[pd.Series] = []
    base_bi_up = _series_or_nan(frame, "chan_ctx_last_bi_is_up")
    if base_bi_up.notna().any():
        base_bi_align = pd.Series(np.where(is_buy, base_bi_up > 0.5, base_bi_up <= 0.5), index=frame.index).astype("float64")
        base_alignments.append(base_bi_align)
        if need("cap_chan_15m_signal_align_last_bi"):
            out["cap_chan_15m_signal_align_last_bi"] = base_bi_align

    base_seg_up = _series_or_nan(frame, "chan_ctx_last_seg_is_up")
    if base_seg_up.notna().any():
        base_seg_align = pd.Series(np.where(is_buy, base_seg_up > 0.5, base_seg_up <= 0.5), index=frame.index).astype("float64")
        base_alignments.append(base_seg_align)
        if need("cap_chan_15m_signal_align_last_seg"):
            out["cap_chan_15m_signal_align_last_seg"] = base_seg_align

    base_mid_dist = _series_or_nan(frame, "chan_ctx_close_vs_last_zs_mid_pct")
    if base_mid_dist.notna().any():
        base_zs_mid_align = pd.Series(np.where(is_buy, base_mid_dist >= 0.0, base_mid_dist <= 0.0), index=frame.index).astype("float64")
        base_alignments.append(base_zs_mid_align)
        if need("cap_chan_15m_signal_align_last_zs_mid"):
            out["cap_chan_15m_signal_align_last_zs_mid"] = base_zs_mid_align

    if need("cap_chan_15m_structure_maturity_score"):
        base_bi_sure = _series_or_nan(frame, "chan_ctx_last_bi_is_sure")
        base_bi_klu = _series_or_nan(frame, "chan_ctx_last_bi_klu_cnt")
        base_seg_sure = _series_or_nan(frame, "chan_ctx_last_seg_is_sure")
        base_seg_bi = _series_or_nan(frame, "chan_ctx_last_seg_bi_cnt")
        base_zs_bi = _series_or_nan(frame, "chan_ctx_last_zs_bi_cnt")
        maturity_parts = [
            np.tanh(base_bi_klu / 32.0) * base_bi_sure,
            np.tanh(base_seg_bi / 5.0) * base_seg_sure,
            np.tanh(base_zs_bi / 5.0),
        ]
        out["cap_chan_15m_structure_maturity_score"], _ = _finite_ratio_sum(maturity_parts)

    htf_consistency: list[pd.Series] = []
    htf_bi_align: list[pd.Series] = []
    htf_seg_align: list[pd.Series] = []
    htf_zs_align: list[pd.Series] = []
    htf_structure_maturity: list[pd.Series] = []
    htf_boundary_alignment: list[pd.Series] = []
    htf_regime_continuation: list[pd.Series] = []
    htf_regime_rebound: list[pd.Series] = []
    htf_regime_departure: list[pd.Series] = []
    htf_nested_divergence: list[pd.Series] = []

    for label in ("1h", "4h", "1d"):
        src = f"chan_mtf_{label}"
        dst = f"cap_chan_{label}"

        pos = _series_or_nan(frame, f"{src}_event_pos_last_zs")
        inside = _series_or_nan(frame, f"{src}_event_inside_last_zs")
        zs_exists = _series_or_nan(frame, f"{src}_last_zs_exists")
        zs_sure = _series_or_nan(frame, f"{src}_last_zs_is_sure")
        zs_width = _series_or_nan(frame, f"{src}_last_zs_width_pct")
        zs_bi_cnt = _series_or_nan(frame, f"{src}_last_zs_bi_cnt")

        signal_boundary_score = pd.Series(np.where(is_buy, 1.0 - pos, pos), index=frame.index)
        nearest_boundary_dist = pd.concat([pos, 1.0 - pos], axis=1).min(axis=1)
        outside_dist = pd.concat([(pos - 1.0).clip(lower=0.0), (-pos).clip(lower=0.0)], axis=1).max(axis=1)
        near_signal_boundary = (signal_boundary_score >= 0.8).astype("float64")
        near_opposite_boundary = (signal_boundary_score <= 0.2).astype("float64")

        if need(f"{dst}_zs_pos_centered"):
            out[f"{dst}_zs_pos_centered"] = pos - 0.5
        if need(f"{dst}_zs_nearest_boundary_dist"):
            out[f"{dst}_zs_nearest_boundary_dist"] = nearest_boundary_dist
        if need(f"{dst}_zs_signal_boundary_score"):
            out[f"{dst}_zs_signal_boundary_score"] = signal_boundary_score
        if need(f"{dst}_near_signal_boundary"):
            out[f"{dst}_near_signal_boundary"] = near_signal_boundary.where(pos.notna())
        if need(f"{dst}_near_opposite_boundary"):
            out[f"{dst}_near_opposite_boundary"] = near_opposite_boundary.where(pos.notna())
        if need(f"{dst}_inside_sure_zs"):
            out[f"{dst}_inside_sure_zs"] = inside * zs_sure
        if need(f"{dst}_zs_leave_distance"):
            out[f"{dst}_zs_leave_distance"] = outside_dist

        bi_align = _series_or_nan(frame, f"{src}_signal_align_last_bi")
        seg_align = _series_or_nan(frame, f"{src}_signal_align_last_seg")
        zs_mid_align = _series_or_nan(frame, f"{src}_signal_align_last_zs_mid")
        consistency, available = _finite_ratio_sum([bi_align, seg_align, zs_mid_align])
        htf_consistency.append(consistency)
        htf_bi_align.append(bi_align)
        htf_seg_align.append(seg_align)
        htf_zs_align.append(zs_mid_align)

        if need(f"{dst}_bi_seg_resonance"):
            out[f"{dst}_bi_seg_resonance"], _ = _finite_ratio_sum([bi_align, seg_align])
        if need(f"{dst}_direction_consistency"):
            out[f"{dst}_direction_consistency"] = consistency
        if need(f"{dst}_direction_conflict"):
            out[f"{dst}_direction_conflict"] = (1.0 - consistency).where(available > 0.0)

        bi_sure = _series_or_nan(frame, f"{src}_last_bi_is_sure")
        bi_klu = _series_or_nan(frame, f"{src}_last_bi_klu_cnt")
        seg_sure = _series_or_nan(frame, f"{src}_last_seg_is_sure")
        seg_bi_cnt = _series_or_nan(frame, f"{src}_last_seg_bi_cnt")
        seg_klu = _series_or_nan(frame, f"{src}_last_seg_klu_cnt")
        seg_zs_cnt = _series_or_nan(frame, f"{src}_last_seg_zs_cnt")
        multi_zs_cnt = _series_or_nan(frame, f"{src}_last_seg_multi_zs_cnt")

        bi_maturity = np.tanh(bi_klu / 32.0) * bi_sure
        seg_maturity = ((np.tanh(seg_bi_cnt / 5.0) + np.tanh(seg_klu / 128.0)) / 2.0) * seg_sure
        zs_maturity = np.tanh(zs_bi_cnt / 5.0) * zs_sure
        structure_maturity, _ = _finite_ratio_sum([bi_maturity, seg_maturity, zs_maturity])
        if need(f"{dst}_bi_maturity"):
            out[f"{dst}_bi_maturity"] = bi_maturity
        if need(f"{dst}_seg_maturity"):
            out[f"{dst}_seg_maturity"] = seg_maturity
        if need(f"{dst}_zs_maturity"):
            out[f"{dst}_zs_maturity"] = zs_maturity
        if need(f"{dst}_structure_maturity_score"):
            out[f"{dst}_structure_maturity_score"] = structure_maturity
        if need(f"{dst}_seg_zs_density"):
            out[f"{dst}_seg_zs_density"] = (seg_zs_cnt + multi_zs_cnt) / (seg_bi_cnt + 1e-12)

        divergence_boundary = divergence_like * near_signal_boundary * inside
        if need(f"{dst}_divergence_near_signal_boundary"):
            out[f"{dst}_divergence_near_signal_boundary"] = divergence_boundary.where(pos.notna())
        if need(f"{dst}_divergence_inside_zs"):
            out[f"{dst}_divergence_inside_zs"] = (divergence_like * inside).where(zs_exists > 0.0)
        if need(f"{dst}_divergence_boundary_score"):
            out[f"{dst}_divergence_boundary_score"] = divergence_like * signal_boundary_score

        trend_continuation = consistency * (1.0 - inside.fillna(0.0) * near_opposite_boundary.fillna(0.0)) * structure_maturity
        range_rebound = inside * near_signal_boundary * (1.0 - outside_dist.fillna(0.0)) * structure_maturity
        zs_departure = outside_dist * zs_mid_align * structure_maturity
        nested_divergence = pd.concat(
            [
                divergence_boundary,
                (divergence_like * inside).where(zs_exists > 0.0),
            ],
            axis=1,
        ).max(axis=1)

        htf_structure_maturity.append(structure_maturity)
        htf_boundary_alignment.append(signal_boundary_score)
        htf_regime_continuation.append(trend_continuation)
        htf_regime_rebound.append(range_rebound)
        htf_regime_departure.append(zs_departure)
        htf_nested_divergence.append(nested_divergence)
        if need(f"{dst}_trend_continuation_score"):
            out[f"{dst}_trend_continuation_score"] = trend_continuation
        if need(f"{dst}_range_rebound_score"):
            out[f"{dst}_range_rebound_score"] = range_rebound.where(zs_exists > 0.0)
        if need(f"{dst}_zs_departure_score"):
            out[f"{dst}_zs_departure_score"] = zs_departure.where(zs_exists > 0.0)

        if need(f"{dst}_wide_zs_boundary_pressure"):
            out[f"{dst}_wide_zs_boundary_pressure"] = signal_boundary_score * zs_width

    align_sources = base_alignments + htf_consistency
    all_consistency, all_available = _finite_ratio_sum(align_sources)
    if need("cap_chan_structure_consistency_15m_1h_4h_1d"):
        out["cap_chan_structure_consistency_15m_1h_4h_1d"] = all_consistency
    if need("cap_chan_structure_available_levels"):
        out["cap_chan_structure_available_levels"] = all_available

    htf_consistency_ratio, htf_available = _finite_ratio_sum(htf_consistency)
    structure_maturity_mean, maturity_available = _finite_ratio_sum(htf_structure_maturity)
    boundary_alignment_mean, boundary_available = _finite_ratio_sum(htf_boundary_alignment)
    regime_continuation_strength, continuation_available = _finite_ratio_sum(htf_regime_continuation)
    regime_rebound_strength, rebound_available = _finite_ratio_sum(htf_regime_rebound)
    regime_departure_strength, departure_available = _finite_ratio_sum(htf_regime_departure)
    nested_divergence_strength, nested_available = _finite_ratio_sum(htf_nested_divergence)

    if need("cap_chan_mtf_consensus_strength"):
        out["cap_chan_mtf_consensus_strength"] = htf_consistency_ratio
    if need("cap_chan_structure_maturity_mean"):
        out["cap_chan_structure_maturity_mean"] = structure_maturity_mean
    if need("cap_chan_structure_boundary_alignment"):
        out["cap_chan_structure_boundary_alignment"] = boundary_alignment_mean
    if need("cap_chan_nested_divergence_score"):
        out["cap_chan_nested_divergence_score"] = nested_divergence_strength
    if need("cap_chan_regime_continuation_strength"):
        out["cap_chan_regime_continuation_strength"] = regime_continuation_strength
    if need("cap_chan_regime_rebound_strength"):
        out["cap_chan_regime_rebound_strength"] = regime_rebound_strength
    if need("cap_chan_regime_departure_strength"):
        out["cap_chan_regime_departure_strength"] = regime_departure_strength
    if need("cap_chan_entry_style_strength"):
        style_strength = pd.concat(
            [
                regime_continuation_strength,
                regime_rebound_strength,
                regime_departure_strength,
            ],
            axis=1,
        ).max(axis=1)
        out["cap_chan_entry_style_strength"] = style_strength
    if need("cap_chan_structure_purity_score"):
        purity_score, _ = _finite_ratio_sum(
            [
                all_consistency,
                htf_consistency_ratio,
                structure_maturity_mean,
                boundary_alignment_mean,
                nested_divergence_strength,
            ]
        )
        out["cap_chan_structure_purity_score"] = purity_score
    if need("cap_chan_consensus_x_maturity"):
        out["cap_chan_consensus_x_maturity"] = htf_consistency_ratio * structure_maturity_mean
    if need("cap_chan_boundary_x_maturity"):
        out["cap_chan_boundary_x_maturity"] = boundary_alignment_mean * structure_maturity_mean
    if need("cap_chan_divergence_boundary_x_maturity"):
        out["cap_chan_divergence_boundary_x_maturity"] = nested_divergence_strength * boundary_alignment_mean * structure_maturity_mean
    if need("cap_chan_consensus_minus_conflict"):
        out["cap_chan_consensus_minus_conflict"] = htf_consistency_ratio - (1.0 - htf_consistency_ratio)
    if need("cap_chan_rebound_vs_departure"):
        out["cap_chan_rebound_vs_departure"] = regime_rebound_strength - regime_departure_strength
    if need("cap_chan_mtf_direction_consensus_score"):
        out["cap_chan_mtf_direction_consensus_score"] = htf_consistency_ratio
    if need("cap_chan_mtf_direction_conflict_score"):
        out["cap_chan_mtf_direction_conflict_score"] = (1.0 - htf_consistency_ratio).where(htf_available > 0.0)
    if need("cap_chan_mtf_all_direction_resonance"):
        out["cap_chan_mtf_all_direction_resonance"] = (htf_consistency_ratio >= 0.999).astype("float64").where(htf_available > 0.0)
    if need("cap_chan_mtf_any_direction_conflict"):
        out["cap_chan_mtf_any_direction_conflict"] = (htf_consistency_ratio < 0.999).astype("float64").where(htf_available > 0.0)

    for name, sources in (
        ("bi", htf_bi_align),
        ("seg", htf_seg_align),
        ("zs_mid", htf_zs_align),
    ):
        matrix = pd.concat(sources, axis=1) if sources else pd.DataFrame(index=frame.index)
        available = matrix.notna().sum(axis=1).astype("float64")
        align_count = matrix.fillna(0.0).sum(axis=1)
        if need(f"cap_chan_mtf_{name}_align_count"):
            out[f"cap_chan_mtf_{name}_align_count"] = align_count.where(available > 0.0)
        if need(f"cap_chan_mtf_{name}_conflict_count"):
            out[f"cap_chan_mtf_{name}_conflict_count"] = (available - align_count).where(available > 0.0)

    if required is not None:
        for col in required:
            if str(col).startswith("cap_chan_") and col not in out.columns:
                out[col] = np.nan
    return out.replace([np.inf, -np.inf], np.nan)


def build_event_features(
    bars: pd.DataFrame,
    events: Sequence[RawBSPEvent],
    required_columns: Optional[Iterable[str]] = None,
    rust_config_path: Optional[str] = None,
    normalized_bars: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if not events:
        return pd.DataFrame(columns=META_COLUMNS)

    requested = set(required_columns) if required_columns is not None else None
    required = set(requested) if requested is not None else None
    cap_chan_requested = required is not None and any(str(col).startswith("cap_chan_") for col in required)
    if required is not None and cap_chan_requested:
        required.update(_capability_chan_dependency_columns())

    df = normalized_bars if normalized_bars is not None else normalize_bars(bars)
    resampled_bars: dict[str, pd.DataFrame] = {}

    required_bar_cols = [col for col in required or [] if str(col).startswith("bar_")] if required is not None else None
    required_mtf_cols = [col for col in required or [] if str(col).startswith("mtf_")] if required is not None else None
    required_chan_mtf_cols = [col for col in required or [] if str(col).startswith("chan_mtf_")] if required is not None else None
    required_chan_fx_cols = [col for col in required or [] if str(col).startswith("chan_fx_")] if required is not None else None
    required_cap_cols = [col for col in requested or [] if str(col).startswith("cap_")] if requested is not None else None
    bar_features = build_bar_feature_frame(df, required_columns=required_bar_cols, normalized_bars=df)
    bar_ns = datetime_index_to_ns(bar_features.index)
    event_ns = _event_time_ns(events)
    loc = np.searchsorted(bar_ns, event_ns, side="right") - 1
    valid = loc >= 0

    sampled = bar_features.iloc[np.clip(loc, 0, max(len(bar_features) - 1, 0))].reset_index(drop=True)
    sampled.loc[~valid, :] = np.nan
    if required is None or required_chan_fx_cols:
        fractals = build_confirmed_fractal_feature_frame(
            df,
            events,
            required_columns=required_chan_fx_cols,
            normalized_bars=df,
        )
        sampled = pd.concat([sampled.reset_index(drop=True), fractals.reset_index(drop=True)], axis=1)

    meta_rows: List[dict[str, object]] = []
    if required is None:
        chan_keys = sorted({str(key) for event in events for key in event.feature_map.keys() if not str(key).startswith("mtf_")})
    else:
        chan_keys = sorted(str(col)[5:] for col in required if str(col).startswith("chan_") and not str(col).startswith("chan_mtf_"))
    chan_rows: List[dict[str, float]] = []
    bsp_rows: List[dict[str, float]] = []

    for event in events:
        meta_rows.append(
            {
                "symbol": event.symbol,
                "exec_time": pd.to_datetime(event.exec_time, utc=True),
                "bsp_time": event.bsp_time,
                "is_buy": bool(event.is_buy),
                "bsp_type": event.bsp_type,
                "bsp_types_str": event.bsp_types_str,
                "trade_price": float(event.trade_price),
                "klu_idx": int(event.klu_idx),
            }
        )
        chan_rows.append({f"chan_{key}": float(event.feature_map.get(key, np.nan)) for key in chan_keys})
        bsp_rows.append(_bsp_flags(event))

    meta = pd.DataFrame(meta_rows)
    chan = pd.DataFrame(chan_rows)
    bsp = pd.DataFrame(bsp_rows)
    mtf = build_mtf_feature_frame(
        df,
        events,
        required_columns=required_mtf_cols,
        normalized_bars=df,
        resampled_bars=resampled_bars,
    )
    chan_mtf = build_chan_mtf_feature_frame(
        df,
        events,
        required_columns=required_chan_mtf_cols,
        rust_config_path=rust_config_path,
        normalized_bars=df,
        resampled_bars=resampled_bars,
    )
    if required_cap_cols is not None:
        capability = build_capability_context_feature_frame(events, required_columns=required_cap_cols)
    else:
        capability = pd.DataFrame(index=range(len(events)))
    out = pd.concat(
        [
            meta,
            sampled.reset_index(drop=True),
            bsp,
            chan,
            mtf.reset_index(drop=True),
            chan_mtf.reset_index(drop=True),
            capability.reset_index(drop=True),
        ],
        axis=1,
    )
    out = out.loc[:, ~out.columns.duplicated(keep="first")]
    if requested is not None:
        chan_capability = build_capability_chan_feature_frame(out, required_columns=required_cap_cols)
        if not chan_capability.empty:
            out = pd.concat([out.reset_index(drop=True), chan_capability.reset_index(drop=True)], axis=1)
            out = out.loc[:, ~out.columns.duplicated(keep="first")]
        missing = [col for col in requested if col not in out.columns]
        for col in missing:
            out[col] = np.nan
    return out.replace([np.inf, -np.inf], np.nan)


def infer_feature_columns(frame: pd.DataFrame, extra_prefixes: Iterable[str] = ()) -> List[str]:
    prefixes = tuple(FEATURE_PREFIXES + tuple(extra_prefixes))
    cols = [
        col
        for col in frame.columns
        if any(str(col).startswith(prefix) for prefix in prefixes) and pd.api.types.is_numeric_dtype(frame[col])
    ]
    return sorted(cols)
