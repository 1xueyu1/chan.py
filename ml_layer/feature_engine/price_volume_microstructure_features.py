from __future__ import annotations

# flake8: noqa: E501

from typing import Dict

import numpy as np
import pandas as pd


def _ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    diff = close.diff().fillna(0.0)
    gain = diff.clip(lower=0.0)
    loss = (-diff).clip(lower=0.0)
    avg_gain = gain.rolling(period, min_periods=1).mean()
    avg_loss = loss.rolling(period, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100.0 - (100.0 / (1.0 + rs))


def _compute_kdj_k(df: pd.DataFrame, period: int = 9, smooth: int = 3) -> pd.Series:
    low_n = df["low"].rolling(period, min_periods=1).min()
    high_n = df["high"].rolling(period, min_periods=1).max()
    rsv = (df["close"] - low_n) / (high_n - low_n + 1e-9) * 100.0
    k = np.full(len(rsv), 50.0, dtype=float)
    vals = rsv.fillna(50.0).to_numpy()
    for i in range(1, len(vals)):
        k[i] = (k[i - 1] * (smooth - 1) + vals[i]) / smooth
    return pd.Series(k, index=df.index)


def _compute_cci(df: pd.DataFrame, period: int = 14) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3.0
    ma = tp.rolling(period, min_periods=1).mean()
    md = (tp - ma).abs().rolling(period, min_periods=1).mean()
    return (tp - ma) / (0.015 * (md + 1e-9))


def build_price_volume_microstructure_indicators(
    bar_df: pd.DataFrame,
) -> pd.DataFrame:
    df = bar_df.copy()
    df["ret_1"] = df["close"].pct_change().fillna(0.0)
    df["tr"] = (df["high"] - df["low"]).abs()
    df["atr_14"] = df["tr"].rolling(14, min_periods=1).mean()
    df["vol_ratio_5"] = df["volume"] / (df["volume"].rolling(5, min_periods=1).mean() + 1e-9)
    df["vol_ratio_20"] = df["volume"] / (df["volume"].rolling(20, min_periods=1).mean() + 1e-9)
    roll20_min = df["low"].rolling(20, min_periods=1).min()
    roll20_max = df["high"].rolling(20, min_periods=1).max()
    df["price_position_20"] = (df["close"] - roll20_min) / (roll20_max - roll20_min + 1e-9)
    df["atr_14_normalized"] = df["atr_14"] / (df["close"].abs() + 1e-9)
    df["close_high_ratio"] = df["close"] / (df["high"] + 1e-9)
    df["close_low_ratio"] = df["low"] / (df["close"] + 1e-9)
    df["momentum_5"] = df["close"] - df["close"].shift(5)
    df["price_acceleration"] = (
        df["close"] - 2.0 * df["close"].shift(1) + df["close"].shift(2)
    )
    df["price_pos_20"] = df["price_position_20"]

    ma20 = df["close"].rolling(20, min_periods=1).mean()
    ma60 = df["close"].rolling(60, min_periods=1).mean()
    ma5 = df["close"].rolling(5, min_periods=1).mean()
    ma10 = df["close"].rolling(10, min_periods=1).mean()
    df["price_ma20_dist"] = (df["close"] - ma20) / (ma20 + 1e-9)
    df["price_ma60_dist"] = (df["close"] - ma60) / (ma60 + 1e-9)
    df["ma20_ma60_cross"] = np.where(ma20 > ma60, 1.0, np.where(ma20 < ma60, -1.0, 0.0))
    all_up = (ma5 > ma10) & (ma10 > ma20) & (ma20 > ma60)
    all_down = (ma5 < ma10) & (ma10 < ma20) & (ma20 < ma60)
    df["ma_trend_aligned"] = np.where(all_up, 1.0, np.where(all_down, -1.0, 0.0))
    df["vol_change"] = df["volume"] / (df["volume"].shift(1).abs() + 1e-9) - 1.0
    df["vol_amount_std_20"] = df["volume"].rolling(20, min_periods=1).std(ddof=0)

    body = (df["close"] - df["open"]).abs()
    full_range = (df["high"] - df["low"]).abs()
    upper = df["high"] - np.maximum(df["close"], df["open"])
    lower = np.minimum(df["close"], df["open"]) - df["low"]
    df["upper_shadow_ratio"] = upper / (body + 1e-9)
    df["lower_shadow_ratio"] = lower / (body + 1e-9)
    df["bar_range"] = full_range / (df["open"].abs() + 1e-9)
    df["bar_body_position"] = np.where(
        full_range > 0,
        (np.minimum(df["close"], df["open"]) - df["low"]) / (full_range + 1e-9),
        0.5,
    )
    df["candle_strength"] = body / (full_range + 1e-9)

    df["rsi_14"] = _compute_rsi(df["close"], period=14)
    macd_line = _ema(df["close"], 12) - _ema(df["close"], 26)
    signal_line = _ema(macd_line, 9)
    df["macd_hist"] = macd_line - signal_line
    df["kdj_k"] = _compute_kdj_k(df, period=9, smooth=3)
    std20 = df["close"].rolling(20, min_periods=1).std(ddof=0)
    upper_band = ma20 + 2.0 * std20
    lower_band = ma20 - 2.0 * std20
    df["boll_bandwidth"] = (upper_band - lower_band) / (ma20.abs() + 1e-9)
    df["atr_expanding"] = np.where(
        df["atr_14"] > df["atr_14"].shift(14),
        1.0,
        np.where(df["atr_14"] < df["atr_14"].shift(14), -1.0, 0.0),
    )
    df["cci_value"] = _compute_cci(df, period=14)

    signed = np.where(df["close"] >= df["open"], df["volume"], -df["volume"])
    signed_series = pd.Series(signed, index=df.index)
    df["vol_direction_bias"] = signed_series.rolling(10, min_periods=1).sum() / (df["volume"].rolling(10, min_periods=1).sum() + 1e-9)

    prev_high = df["high"].shift(1)
    prev_low = df["low"].shift(1)
    gap_up = (df["low"] - prev_high).clip(lower=0.0)
    gap_down = (prev_low - df["high"]).clip(lower=0.0)
    df["gap_strength"] = (gap_up + gap_down) / (df["atr_14"] + 1e-9)

    direction = np.sign(df["close"] - df["open"]).replace(0.0, np.nan).ffill().fillna(0.0)
    streak = []
    prev = 0.0
    cur = 0
    for val in direction.to_numpy():
        if val == prev and val != 0:
            cur += 1
        elif val != 0:
            cur = 1
        else:
            cur = 0
        prev = val
        streak.append(float(cur))
    df["consecutive_same_dir"] = streak

    slope = (df["close"] - df["close"].shift(3)) / 3.0
    prev_slope = slope.shift(3)
    df["price_acceleration_ratio"] = slope / (prev_slope.abs() + 1e-9)
    df = df.replace([np.inf, -np.inf], np.nan)
    return df


def compute_price_volume_microstructure_features(
    event_bar_row: Dict,
) -> Dict[str, float]:
    return {
        "vol_ratio_5": float(event_bar_row.get("vol_ratio_5", 1.0)),
        "vol_ratio_20": float(event_bar_row.get("vol_ratio_20", 1.0)),
        "price_position_20": float(event_bar_row.get("price_position_20", 0.5)),
        "atr_14_normalized": float(event_bar_row.get("atr_14_normalized", 0.0)),
        "close_high_ratio": float(event_bar_row.get("close_high_ratio", 1.0)),
        "close_low_ratio": float(event_bar_row.get("close_low_ratio", 1.0)),
        "vol_direction_bias": float(event_bar_row.get("vol_direction_bias", 0.0)),
        "gap_strength": float(event_bar_row.get("gap_strength", 0.0)),
        "consecutive_same_dir": float(event_bar_row.get("consecutive_same_dir", 0.0)),
        "price_acceleration_c": float(event_bar_row.get("price_acceleration_ratio", 0.0)),
    }
