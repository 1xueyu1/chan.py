from __future__ import annotations

__all__ = [
    "BacktestConfig",
    "build_arg_parser",
    "config_from_args",
    "run_backtest",
    "run_vectorbt_backtest",
    "run_chan_backtest_no_vnpy",
]


def __getattr__(name: str):
    if name in {"BacktestConfig", "build_arg_parser", "config_from_args"}:
        from . import config

        return getattr(config, name)
    if name == "run_backtest":
        from .facade import run_backtest

        return run_backtest
    if name in {"run_vectorbt_backtest", "run_chan_backtest_no_vnpy"}:
        from . import engine

        return getattr(engine, name)
    raise AttributeError(name)
