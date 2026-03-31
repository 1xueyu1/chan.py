from __future__ import annotations

from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Dict, Optional, Tuple

import pandas as pd

from Common.CEnum import KL_TYPE
from .config import BacktestConfig


REQUIRED_COLUMNS = ["open_time", "open", "high", "low", "close", "volume"]

KL_TYPE_TO_INTERVAL = {
    KL_TYPE.K_1M: "1m",
    KL_TYPE.K_3M: "3m",
    KL_TYPE.K_5M: "5m",
    KL_TYPE.K_10M: "10m",
    KL_TYPE.K_15M: "15m",
    KL_TYPE.K_30M: "30m",
    KL_TYPE.K_60M: "1h",
    KL_TYPE.K_DAY: "1d",
    KL_TYPE.K_WEEK: "1w",
    KL_TYPE.K_MON: "1mo",
}

RESAMPLE_RULE_MAP = {
    KL_TYPE.K_10M: "10min",
    KL_TYPE.K_15M: "15min",
    KL_TYPE.K_30M: "30min",
    KL_TYPE.K_60M: "1h",
    KL_TYPE.K_DAY: "1D",
    KL_TYPE.K_WEEK: "1W",
    KL_TYPE.K_MON: "1MS",
}

FALLBACK_INTERVAL = "5m"


_BARS_CACHE_LOCK = Lock()
_BARS_CACHE: "OrderedDict[tuple, pd.DataFrame]" = OrderedDict()


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
    symbol = symbol.upper().replace("/", "")
    if symbol.endswith("USDT"):
        symbol = symbol[:-4]
    return symbol


def _resolve_source_path(data_dir: Path, symbol: str, kl_type: KL_TYPE) -> Tuple[Path, bool]:
    if kl_type not in KL_TYPE_TO_INTERVAL:
        raise ValueError(f"Unsupported KL_TYPE for parquet loading: {kl_type}")

    base = _normalize_symbol(symbol)
    interval = KL_TYPE_TO_INTERVAL[kl_type]
    target_path = data_dir / f"{base}_{interval}.parquet"
    if target_path.exists():
        return target_path, False

    if kl_type in RESAMPLE_RULE_MAP:
        fallback = data_dir / f"{base}_{FALLBACK_INTERVAL}.parquet"
        if fallback.exists():
            return fallback, True

    raise FileNotFoundError(f"Parquet file not found for {symbol} ({interval})")


def _cache_key(config: BacktestConfig, symbol: str, source_path: Path, need_resample: bool) -> tuple:
    return (
        _normalize_symbol(symbol),
        str(config.kl_type),
        str(config.begin_time),
        str(config.end_time),
        str(source_path),
        bool(need_resample),
    )


def _cache_get(key: tuple, max_size: int) -> Optional[pd.DataFrame]:
    if max_size <= 0:
        return None

    with _BARS_CACHE_LOCK:
        cached = _BARS_CACHE.get(key)
        if cached is None:
            return None
        _BARS_CACHE.move_to_end(key)
        # Return a lightweight copy to avoid accidental cross-task mutation.
        return cached.copy(deep=False)


def _cache_put(key: tuple, frame: pd.DataFrame, max_size: int) -> None:
    if max_size <= 0:
        return

    with _BARS_CACHE_LOCK:
        _BARS_CACHE[key] = frame
        _BARS_CACHE.move_to_end(key)
        while len(_BARS_CACHE) > max_size:
            _BARS_CACHE.popitem(last=False)


def _resample_if_needed(df: pd.DataFrame, kl_type: KL_TYPE, need_resample: bool) -> pd.DataFrame:
    if not need_resample:
        return df

    rule = RESAMPLE_RULE_MAP.get(kl_type)
    if not rule:
        raise ValueError(f"No resample rule for KL_TYPE={kl_type}")

    agg = {
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }

    return (
        df.set_index("open_time")
        .resample(rule, label="left", closed="left", origin="epoch")
        .agg(agg)
        .dropna(subset=["open", "high", "low", "close"])
        .reset_index()
    )


def load_symbol_bars(config: BacktestConfig, symbol: str) -> pd.DataFrame:
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data"

    source_path, need_resample = _resolve_source_path(data_dir, symbol, config.kl_type)
    max_cache_size = int(getattr(config, "data_cache_size", 0) or 0)
    key = _cache_key(config, symbol, source_path, need_resample)
    cached = _cache_get(key, max_cache_size)
    if cached is not None:
        return cached

    df = pd.read_parquet(source_path, columns=REQUIRED_COLUMNS, use_threads=True)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Parquet missing required columns {missing}: {source_path}")

    df = df[REQUIRED_COLUMNS].copy()
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in REQUIRED_COLUMNS[1:]:
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["open_time", "open", "high", "low", "close"])

    open_time = df["open_time"]
    if not open_time.is_monotonic_increasing:
        df = df.sort_values("open_time")
        open_time = df["open_time"]
    if not open_time.is_unique:
        df = df.drop_duplicates(subset=["open_time"])

    df = df.reset_index(drop=True)

    df = _resample_if_needed(df, config.kl_type, need_resample)

    begin_ts = _parse_bound(config.begin_time, is_end=False)
    end_ts = _parse_bound(config.end_time, is_end=True)
    if begin_ts is not None:
        df = df[df["open_time"] >= begin_ts]
    if end_ts is not None:
        df = df[df["open_time"] <= end_ts]

    out = df.set_index("open_time")[["open", "high", "low", "close", "volume"]].copy()
    out.index.name = "time"
    _cache_put(key, out, max_cache_size)
    return out


def load_multi_symbol_bars(config: BacktestConfig) -> Dict[str, pd.DataFrame]:
    bars = {}
    for symbol in config.normalized_symbols():
        bars[symbol] = load_symbol_bars(config, symbol)
    return bars


def preload_symbol_bars(config: BacktestConfig, symbols: list[str], workers: int = 1) -> Dict[str, pd.DataFrame]:
    loaded: Dict[str, pd.DataFrame] = {}
    if workers <= 1:
        for symbol in symbols:
            loaded[symbol] = load_symbol_bars(config, symbol)
        return loaded

    worker_count = max(1, min(int(workers), len(symbols)))
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        frame_list = list(pool.map(lambda sym: load_symbol_bars(config, sym), symbols))
    for symbol, frame in zip(symbols, frame_list):
        loaded[symbol] = frame
    return loaded
