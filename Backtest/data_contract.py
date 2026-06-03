from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

BAR_COLUMNS = ["open", "high", "low", "close", "volume"]
OHLC_COLUMNS = ["open", "high", "low", "close"]


def coerce_open_time_to_utc(values) -> pd.Series:
    """Coerce mixed epoch/string timestamps to UTC timestamps.

    Historical parquet files in this project may contain seconds, milliseconds,
    microseconds, nanoseconds, or a legacy shortened millisecond value. The
    internal backtest contract always uses UTC pandas timestamps.
    """
    if isinstance(values, pd.Series):
        series = values.copy()
        index = series.index
    else:
        series = pd.Series(values)
        index = series.index

    if pd.api.types.is_datetime64_any_dtype(series):
        return pd.to_datetime(series, utc=True, errors="coerce")

    numeric = pd.to_numeric(series, errors="coerce")
    parsed = pd.Series(pd.NaT, index=index, dtype="datetime64[ns, UTC]")
    numeric_mask = numeric.notna()

    if numeric_mask.any():
        epoch_ms = numeric.copy()
        tiny_ms_mask = epoch_ms.between(1e6, 1e8, inclusive="left")
        ms_mask = epoch_ms.between(1e12, 1e14, inclusive="left")
        sec_mask = epoch_ms.between(1e9, 1e11, inclusive="left")
        us_mask = epoch_ms.between(1e14, 1e17, inclusive="left")
        ns_mask = epoch_ms >= 1e17

        if bool(ms_mask.any()) and bool(tiny_ms_mask.any()):
            epoch_ms = epoch_ms.where(~tiny_ms_mask)
        else:
            epoch_ms = epoch_ms.where(~tiny_ms_mask, epoch_ms * 1_000_000.0)
        epoch_ms = epoch_ms.where(~sec_mask, epoch_ms * 1000.0)
        epoch_ms = epoch_ms.where(~us_mask, epoch_ms / 1000.0)
        epoch_ms = epoch_ms.where(~ns_mask, epoch_ms / 1_000_000.0)
        parsed.loc[numeric_mask] = pd.to_datetime(epoch_ms.loc[numeric_mask], unit="ms", utc=True, errors="coerce")

    non_numeric_mask = ~numeric_mask
    if non_numeric_mask.any():
        parsed.loc[non_numeric_mask] = pd.to_datetime(series.loc[non_numeric_mask], utc=True, errors="coerce")
    return parsed


def normalize_bars(frame: pd.DataFrame, time_column: str = "open_time") -> pd.DataFrame:
    """Return a standard BarFrame indexed by UTC time.

    Standard BarFrame contract:
    - UTC DatetimeIndex named ``time``
    - columns ``open/high/low/close/volume``
    - sorted ascending and de-duplicated by timestamp
    - OHLC numeric and non-null; volume numeric with missing values filled by 0
    """
    if frame is None:
        raise ValueError("bars frame is required")
    if isinstance(frame.index, pd.DatetimeIndex) and all(col in frame.columns for col in BAR_COLUMNS):
        df = frame.copy()
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        df.index.name = "time"
    else:
        if time_column not in frame.columns:
            raise ValueError(f"bars frame must contain {time_column!r} or use a DatetimeIndex")
        missing = [col for col in BAR_COLUMNS if col not in frame.columns]
        if missing:
            raise ValueError(f"bars frame missing required columns: {missing}")
        df = frame[[time_column, *BAR_COLUMNS]].copy()
        df[time_column] = coerce_open_time_to_utc(df[time_column])
        df = df.rename(columns={time_column: "time"}).set_index("time")

    for col in BAR_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["volume"] = df["volume"].fillna(0.0)
    df = df.dropna(subset=OHLC_COLUMNS)
    df = df[BAR_COLUMNS].sort_index()
    df = df[~df.index.duplicated(keep="first")]
    df.index.name = "time"
    validate_bars(df)
    return df


def validate_bars(frame: pd.DataFrame, required_columns: Iterable[str] = BAR_COLUMNS) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("bars must be a pandas DataFrame")
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("bars must use a DatetimeIndex")
    if frame.index.tz is None:
        raise ValueError("bars index must be timezone-aware UTC")
    if str(frame.index.tz) != "UTC":
        raise ValueError("bars index must be UTC")
    missing = [col for col in required_columns if col not in frame.columns]
    if missing:
        raise ValueError(f"bars missing required columns: {missing}")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("bars index must be sorted ascending")
    if not frame.index.is_unique:
        raise ValueError("bars index must be unique")
    for col in required_columns:
        if not pd.api.types.is_numeric_dtype(frame[col]):
            raise ValueError(f"bars column {col!r} must be numeric")
    if frame[list(OHLC_COLUMNS)].isna().any().any():
        raise ValueError("bars OHLC columns must not contain NaN")
    if np.isinf(frame[list(required_columns)].to_numpy(dtype="float64", copy=False)).any():
        raise ValueError("bars must not contain infinite values")
