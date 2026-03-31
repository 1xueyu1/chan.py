from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import pandas as pd

from .config import BacktestConfig
from .types import RawBSPEvent, ScoredSignalEvent


_EVENT_CACHE_COLUMNS = [
    "exec_time",
    "bsp_time",
    "is_buy",
    "bsp_type",
    "bsp_types_str",
    "trade_price",
    "klu_idx",
    "probability",
    "qualified",
    "signal",
    "symbol",
]

_EVENT_CACHE_SCHEMA_VERSION = 3


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _fingerprint_file(path_text: str) -> Dict[str, object]:
    path_raw = str(path_text or "").strip()
    if not path_raw:
        return {"path": "", "exists": False}

    path = Path(path_raw)
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _resolved_meta_model_path(config: BacktestConfig) -> str:
    explicit = str(getattr(config, "meta_model_path", "") or "").strip()
    if explicit:
        return explicit

    guessed = Path(str(config.model_buy_path)).resolve().parent / "meta_model.pkl"
    if guessed.exists():
        return str(guessed)
    return ""


def _cache_key_payload(config: BacktestConfig, symbol: str) -> Dict[str, object]:
    return {
        "cache_schema_version": _EVENT_CACHE_SCHEMA_VERSION,
        "symbol": symbol,
        "begin_time": str(config.begin_time),
        "end_time": str(config.end_time),
        "kl_type": str(config.kl_type),
        "signal_threshold": float(config.signal_threshold),
        "signal_margin": float(config.signal_margin),
        "meta_threshold_by_direction": dict(getattr(config, "meta_threshold_by_direction", {}) or {}),
        "meta_threshold_by_bsp": dict(getattr(config, "meta_threshold_by_bsp", {}) or {}),
        "empty_signal_fallback": bool(getattr(config, "empty_signal_fallback", False)),
        "empty_signal_target_rate": float(getattr(config, "empty_signal_target_rate", 0.0)),
        "chan_config": dict(getattr(config, "chan_config", {}) or {}),
        "model_buy": _fingerprint_file(str(config.model_buy_path)),
        "model_sell": _fingerprint_file(str(config.model_sell_path)),
        "meta_buy": _fingerprint_file(str(config.meta_buy_path)),
        "meta_sell": _fingerprint_file(str(config.meta_sell_path)),
        "meta_model": _fingerprint_file(_resolved_meta_model_path(config)),
    }


def _cache_digest(config: BacktestConfig, symbol: str) -> str:
    payload = _cache_key_payload(config, symbol)
    text = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:24]


def _cache_base_dir(config: BacktestConfig) -> Path:
    raw_dir = str(getattr(config, "event_cache_dir", "") or "data/cache/backtest_events").strip()
    cache_dir = Path(raw_dir)
    if not cache_dir.is_absolute():
        cache_dir = _project_root() / cache_dir

    namespace = str(getattr(config, "event_cache_namespace", "default") or "default").strip() or "default"
    return cache_dir / namespace


def _cache_path(config: BacktestConfig, symbol: str, ext: str) -> Path:
    digest = _cache_digest(config, symbol)
    safe_symbol = symbol.upper().replace("/", "")
    return _cache_base_dir(config) / safe_symbol / f"{digest}.{ext}"


def _events_to_frame(events: Sequence[ScoredSignalEvent]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for ev in events:
        rows.append(
            {
                "exec_time": pd.Timestamp(ev.exec_time).strftime("%Y-%m-%d %H:%M:%S"),
                "bsp_time": ev.bsp_time,
                "is_buy": bool(ev.is_buy),
                "bsp_type": str(ev.bsp_type),
                "bsp_types_str": str(ev.bsp_types_str),
                "trade_price": float(ev.trade_price),
                "klu_idx": int(ev.klu_idx),
                "probability": float(ev.probability),
                "qualified": bool(ev.qualified),
                "signal": int(ev.signal),
                "symbol": str(ev.symbol),
            }
        )
    return pd.DataFrame(rows, columns=_EVENT_CACHE_COLUMNS)


def _frame_to_events(df: pd.DataFrame, symbol: str) -> List[ScoredSignalEvent]:
    for col in _EVENT_CACHE_COLUMNS:
        if col not in df.columns:
            raise ValueError(f"cache missing column {col}")

    data = df.copy()
    data["exec_time"] = pd.to_datetime(data["exec_time"], utc=True, errors="coerce")
    data = data.dropna(subset=["exec_time"]).sort_values("exec_time", kind="mergesort")

    out: List[ScoredSignalEvent] = []
    for row in data.itertuples(index=False):
        out.append(
            ScoredSignalEvent(
                symbol=str(getattr(row, "symbol") or symbol),
                exec_time=getattr(row, "exec_time"),
                bsp_time=str(getattr(row, "bsp_time")),
                is_buy=bool(getattr(row, "is_buy")),
                bsp_type=str(getattr(row, "bsp_type")),
                bsp_types_str=str(getattr(row, "bsp_types_str")),
                trade_price=float(getattr(row, "trade_price")),
                klu_idx=int(getattr(row, "klu_idx")),
                probability=float(getattr(row, "probability")),
                qualified=bool(getattr(row, "qualified")),
                signal=int(getattr(row, "signal")),
            )
        )
    return out


def load_scored_events_cache(config: BacktestConfig, symbol: str) -> Optional[List[ScoredSignalEvent]]:
    if not bool(getattr(config, "event_cache_enabled", False)):
        return None

    parquet_path = _cache_path(config, symbol, "parquet")
    csv_path = _cache_path(config, symbol, "csv")

    try:
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
            return _frame_to_events(df, symbol)
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            return _frame_to_events(df, symbol)
    except Exception:
        return None

    return None


def save_scored_events_cache(config: BacktestConfig, symbol: str, events: Sequence[ScoredSignalEvent]) -> Optional[Path]:
    if not bool(getattr(config, "event_cache_enabled", False)):
        return None

    frame = _events_to_frame(events)
    parquet_path = _cache_path(config, symbol, "parquet")
    parquet_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        frame.to_parquet(parquet_path, index=False)
        return parquet_path
    except Exception:
        csv_path = _cache_path(config, symbol, "csv")
        frame.to_csv(csv_path, index=False, encoding="utf-8")
        return csv_path


def cache_key_preview(config: BacktestConfig, symbol: str) -> str:
    return _cache_digest(config, symbol)


def build_signal_event_pairs(events: Sequence[RawBSPEvent]) -> List[tuple[pd.Timestamp, int]]:
    return [(pd.Timestamp(ev.exec_time), int(1 if ev.is_buy else -1)) for ev in events]
