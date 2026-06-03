from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd


def parse_time_bound(value: Any, is_end: bool) -> pd.Timestamp | None:
    if value in (None, ""):
        return None

    text = str(value).replace("/", "-").strip()
    dt = None
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(text, fmt)
            break
        except Exception:
            continue

    if dt is None:
        dt = pd.to_datetime(text, utc=True, errors="coerce")
        if pd.isna(dt):
            raise ValueError(f"unable to parse datetime bound: {value!r}")
        return pd.Timestamp(dt).tz_convert("UTC") if dt.tzinfo is not None else pd.Timestamp(dt).tz_localize("UTC")

    if len(text) == 10 and is_end:
        dt = dt.replace(hour=23, minute=59, second=59)
    return pd.Timestamp(dt, tz="UTC")


def to_utc_timestamp(value: Any) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        raise ValueError(f"invalid timestamp: {value!r}")
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def event_signal_time(event: Any) -> pd.Timestamp:
    actual_exec_time = getattr(event, "actual_exec_time", None)
    base_time = actual_exec_time if actual_exec_time is not None else getattr(event, "exec_time", None)
    if base_time is None:
        raise ValueError("event is missing exec_time")
    return to_utc_timestamp(base_time)
