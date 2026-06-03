from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Protocol, Tuple

import pandas as pd

from .config import BacktestConfig
from .types import BacktestArtifacts, RawBSPEvent, ScoredSignalEvent, SignalMatrix, SymbolBacktestResult


class SignalExtractor(Protocol):
    def extract(self, config: BacktestConfig, symbol: str, bars: pd.DataFrame | None = None) -> List[RawBSPEvent]: ...


class MLScorer(Protocol):
    def score(self, bars: pd.DataFrame, raw_events: List[RawBSPEvent]) -> List[ScoredSignalEvent]: ...


class RiskLayer(Protocol):
    def apply(self, config: BacktestConfig, bars: pd.DataFrame, signal_matrix: SignalMatrix) -> SignalMatrix: ...


class ExecutionEngine(Protocol):
    def run(
        self,
        config: BacktestConfig,
        bars: pd.DataFrame,
        signal_matrix: SignalMatrix,
    ) -> Tuple[object, Dict[str, float], pd.Series, pd.Series, List[Dict[str, object]]]: ...


class Reporter(Protocol):
    def write(
        self,
        output_dir: str | Path,
        aggregate_metrics: Dict[str, float],
        per_symbol: List[SymbolBacktestResult],
    ) -> BacktestArtifacts: ...
