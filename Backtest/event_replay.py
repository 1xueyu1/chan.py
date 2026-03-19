from __future__ import annotations

# flake8: noqa: E501

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from .config import BacktestConfig
from .types import ScoredSignalEvent


REQUIRED_EVENT_COLUMNS = [
    "exec_time",
    "bsp_time",
    "is_buy",
    "bsp_type",
    "bsp_types_str",
    "probability",
    "qualified",
    "signal",
    "trade_price",
    "symbol",
]

_TRUE_SET = {"1", "true", "t", "yes", "y"}
_FALSE_SET = {"0", "false", "f", "no", "n"}


def _parse_bound(value: Optional[str], is_end: bool) -> Optional[pd.Timestamp]:
    if not value:
        return None

    text = str(value).replace("/", "-").strip()
    dt = None
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(text, fmt)
            break
        except ValueError:
            continue

    if dt is None:
        raise ValueError(f"Unable to parse datetime bound: {value}")

    if len(text) == 10 and is_end:
        dt = dt.replace(hour=23, minute=59, second=59)

    return pd.Timestamp(dt, tz="UTC")


def _normalize_symbol(symbol: str) -> str:
    s = str(symbol).upper().replace("/", "")
    if not s.endswith("USDT"):
        s = f"{s}USDT"
    return s


def _parse_bool(value, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default

    text = str(value).strip().lower()
    if text in _TRUE_SET:
        return True
    if text in _FALSE_SET:
        return False
    return default


def _safe_float(value, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _signal_from_probability(is_buy: bool, probability: float, threshold: float) -> tuple[bool, int]:
    qualified = probability >= threshold
    if not qualified:
        return False, 0
    return True, (1 if is_buy else -1)


def load_scored_events_by_symbol(config: BacktestConfig) -> Dict[str, List[ScoredSignalEvent]]:
    csv_path = Path(config.event_replay_csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"event replay csv not found: {csv_path}")

    df = pd.read_csv(csv_path)

    missing = [col for col in REQUIRED_EVENT_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"event replay csv missing columns {missing}: {csv_path}")

    normalized_symbols = config.normalized_symbols()
    symbol_set = set(normalized_symbols)

    df = df.copy()
    df["symbol"] = df["symbol"].map(_normalize_symbol)
    df = df[df["symbol"].isin(symbol_set)]

    df["exec_time"] = pd.to_datetime(df["exec_time"], utc=True, errors="coerce")
    df = df.dropna(subset=["exec_time"])

    begin_ts = _parse_bound(config.begin_time, is_end=False)
    end_ts = _parse_bound(config.end_time, is_end=True)
    if begin_ts is not None:
        df = df[df["exec_time"] >= begin_ts]
    if end_ts is not None:
        df = df[df["exec_time"] <= end_ts]

    df = df.sort_values(["symbol", "exec_time"], kind="mergesort")

    grouped: Dict[str, List[ScoredSignalEvent]] = {
        symbol: [] for symbol in normalized_symbols
    }

    for row in df.itertuples(index=False):
        symbol = _normalize_symbol(getattr(row, "symbol"))
        is_buy = _parse_bool(getattr(row, "is_buy"), default=False)
        probability = _safe_float(getattr(row, "probability"))

        if config.replay_reapply_threshold and not pd.isna(probability):
            qualified, signal = _signal_from_probability(
                is_buy=is_buy,
                probability=probability,
                threshold=config.signal_threshold,
            )
        else:
            signal = _safe_int(getattr(row, "signal"), default=0)
            if signal > 0:
                signal = 1
            elif signal < 0:
                signal = -1
            else:
                signal = 0
            qualified = _parse_bool(getattr(row, "qualified"), default=(signal != 0))

        grouped.setdefault(symbol, []).append(
            ScoredSignalEvent(
                symbol=symbol,
                exec_time=getattr(row, "exec_time"),
                bsp_time=str(getattr(row, "bsp_time")),
                is_buy=is_buy,
                bsp_type=str(getattr(row, "bsp_type")),
                bsp_types_str=str(getattr(row, "bsp_types_str")),
                trade_price=_safe_float(getattr(row, "trade_price"), default=0.0),
                klu_idx=_safe_int(getattr(row, "klu_idx", -1), default=-1),
                probability=probability,
                qualified=qualified,
                signal=signal,
            )
        )

    return grouped
