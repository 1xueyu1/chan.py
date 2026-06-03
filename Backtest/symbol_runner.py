from __future__ import annotations

from .engine import _empty_symbol_result, _run_single_symbol_backtest

__all__ = ["run_single_symbol_backtest", "empty_symbol_result"]

run_single_symbol_backtest = _run_single_symbol_backtest
empty_symbol_result = _empty_symbol_result
