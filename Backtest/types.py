from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


@dataclass
class RawBSPEvent:
    symbol: str
    exec_time: pd.Timestamp
    bsp_time: str
    is_buy: bool
    bsp_type: str
    bsp_types_str: str
    trade_price: float
    klu_idx: int
    feature_map: Dict[str, float] = field(default_factory=dict)
    actual_exec_time: Optional[pd.Timestamp] = None


@dataclass
class ScoredSignalEvent:
    symbol: str
    exec_time: pd.Timestamp
    bsp_time: str
    is_buy: bool
    bsp_type: str
    bsp_types_str: str
    trade_price: float
    klu_idx: int
    probability: float
    qualified: bool
    signal: int
    actual_exec_time: Optional[pd.Timestamp] = None
    threshold: float = 0.0
    market_state: str = ""
    quality_score: float = 0.0
    gate_probability: float = float("nan")
    threshold_reason: str = ""
    structure_confidence_tier: str = ""
    structure_quality_score: float = float("nan")
    structure_risk_pct: float = float("nan")
    risk_size_multiplier: float = 1.0
    quality_size_multiplier: float = 1.0
    stake_multiplier: float = 1.0


@dataclass
class SignalMatrix:
    long_entries: pd.Series
    long_exits: pd.Series
    short_entries: pd.Series
    short_exits: pd.Series
    signal: pd.Series
    position: pd.Series
    size: Optional[pd.Series] = None
    size_multiplier: Optional[pd.Series] = None
    leverage: Optional[pd.Series] = None
    risk_blocked_entries: Optional[pd.Series] = None
    risk_forced_exits: Optional[pd.Series] = None
    execution_pairs: List[Tuple[pd.Timestamp, pd.Timestamp]] = field(
        default_factory=list
    )


@dataclass
class SymbolBacktestResult:
    symbol: str
    metrics: Dict[str, float]
    signal_events: List[ScoredSignalEvent]
    bars: pd.DataFrame
    signal_matrix: SignalMatrix
    equity_curve: pd.Series
    drawdown_curve: pd.Series
    closed_trades: List[Dict[str, object]] = field(default_factory=list)


@dataclass
class BacktestArtifacts:
    events_csv: Optional[Path] = None
    bars_csv: Optional[Path] = None
    metrics_json: Optional[Path] = None
    report_html: Optional[Path] = None
    trades_csv: Optional[Path] = None
    equity_csv: Optional[Path] = None
    per_symbol_metrics_csv: Optional[Path] = None
    period_returns_csvs: Dict[str, Path] = field(default_factory=dict)


@dataclass
class BacktestRunResult:
    aggregate_metrics: Dict[str, float]
    per_symbol: List[SymbolBacktestResult]
    artifacts: Optional[BacktestArtifacts] = None


def _to_utc_timestamp(value, field_name: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        raise ValueError(f"{field_name} is invalid")
    if ts.tzinfo is None:
        return ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def validate_raw_bsp_event(event: RawBSPEvent) -> None:
    if not event.symbol:
        raise ValueError("RawBSPEvent.symbol is required")
    _to_utc_timestamp(event.exec_time, "RawBSPEvent.exec_time")
    if event.actual_exec_time is not None:
        actual = _to_utc_timestamp(
            event.actual_exec_time,
            "RawBSPEvent.actual_exec_time",
        )
        exec_time = _to_utc_timestamp(event.exec_time, "RawBSPEvent.exec_time")
        if actual < exec_time:
            raise ValueError(
                "RawBSPEvent.actual_exec_time must be >= exec_time"
            )
    if not event.bsp_type:
        raise ValueError("RawBSPEvent.bsp_type is required")
    if float(event.trade_price) <= 0:
        raise ValueError("RawBSPEvent.trade_price must be > 0")
    if int(event.klu_idx) < 0:
        raise ValueError("RawBSPEvent.klu_idx must be >= 0")


def validate_scored_signal_event(event: ScoredSignalEvent) -> None:
    raw = RawBSPEvent(
        symbol=event.symbol,
        exec_time=event.exec_time,
        bsp_time=event.bsp_time,
        is_buy=event.is_buy,
        bsp_type=event.bsp_type,
        bsp_types_str=event.bsp_types_str,
        trade_price=event.trade_price,
        klu_idx=event.klu_idx,
        actual_exec_time=event.actual_exec_time,
    )
    validate_raw_bsp_event(raw)
    if event.signal not in {-1, 0, 1}:
        raise ValueError("ScoredSignalEvent.signal must be -1, 0, or 1")
    probability = float(event.probability)
    if probability < 0 or probability > 1:
        raise ValueError("ScoredSignalEvent.probability must be in [0, 1]")
    threshold = float(event.threshold)
    if threshold < 0 or threshold > 1:
        raise ValueError("ScoredSignalEvent.threshold must be in [0, 1]")


def validate_signal_matrix(matrix: SignalMatrix) -> None:
    series_fields = [
        matrix.long_entries,
        matrix.long_exits,
        matrix.short_entries,
        matrix.short_exits,
        matrix.signal,
        matrix.position,
    ]
    index = series_fields[0].index
    if not isinstance(index, pd.DatetimeIndex):
        raise ValueError("SignalMatrix index must be a DatetimeIndex")
    for series in series_fields[1:]:
        if not series.index.equals(index):
            raise ValueError("SignalMatrix series indexes must match")
    optional_fields = (
        matrix.size,
        matrix.size_multiplier,
        matrix.leverage,
        matrix.risk_blocked_entries,
        matrix.risk_forced_exits,
    )
    for optional in optional_fields:
        if optional is not None and not optional.index.equals(index):
            raise ValueError("SignalMatrix optional series indexes must match")
    for entry_time, fill_time in matrix.execution_pairs:
        fill_ts = _to_utc_timestamp(fill_time, "SignalMatrix.fill_time")
        entry_ts = _to_utc_timestamp(entry_time, "SignalMatrix.entry_time")
        if fill_ts < entry_ts:
            raise ValueError(
                "SignalMatrix execution fill_time must be >= entry_time"
            )
