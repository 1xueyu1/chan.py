from __future__ import annotations

from .config import BacktestConfig
from .engine import run_vectorbt_backtest
from .types import BacktestRunResult


def run_backtest(config: BacktestConfig) -> BacktestRunResult:
    """Stable public entrypoint for the backtest pipeline."""
    return run_vectorbt_backtest(config)
