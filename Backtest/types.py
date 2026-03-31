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
    feature_map: Dict[str, float]


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


@dataclass
class SignalMatrix:
    long_entries: pd.Series
    long_exits: pd.Series
    short_entries: pd.Series
    short_exits: pd.Series
    signal: pd.Series
    position: pd.Series
    execution_pairs: List[Tuple[pd.Timestamp, pd.Timestamp]] = field(default_factory=list)


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
    events_csv: Path
    bars_csv: Path
    metrics_json: Path
    report_html: Path
    report_detail_html: Optional[Path] = None
    trades_csv: Optional[Path] = None
    equity_csv: Optional[Path] = None


@dataclass
class BacktestRunResult:
    aggregate_metrics: Dict[str, float]
    per_symbol: List[SymbolBacktestResult]
    artifacts: Optional[BacktestArtifacts] = None
