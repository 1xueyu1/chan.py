import pandas as pd
import pytest

from Backtest.config import BacktestConfig
from Backtest.signal_builder import build_signal_matrix
from Backtest.types import ScoredSignalEvent
from Backtest.vectorbt_engine import run_vectorbt_for_symbol


@pytest.mark.skipif(pytest.importorskip("vectorbt") is None, reason="vectorbt not installed")
def test_vectorbt_engine_runs_minimal_case():
    idx = pd.date_range("2025-01-01", periods=10, freq="15min", tz="UTC")
    bars = pd.DataFrame(
        {
            "open": [100 + i for i in range(10)],
            "high": [101 + i for i in range(10)],
            "low": [99 + i for i in range(10)],
            "close": [100.5 + i for i in range(10)],
            "volume": [10.0 for _ in range(10)],
        },
        index=idx,
    )

    events = [
        ScoredSignalEvent(
            symbol="BTCUSDT",
            exec_time=idx[1],
            bsp_time=str(idx[1]),
            is_buy=True,
            bsp_type="1",
            bsp_types_str="1",
            trade_price=101.0,
            klu_idx=1,
            probability=0.9,
            qualified=True,
            signal=1,
        ),
        ScoredSignalEvent(
            symbol="BTCUSDT",
            exec_time=idx[6],
            bsp_time=str(idx[6]),
            is_buy=False,
            bsp_type="1",
            bsp_types_str="1",
            trade_price=106.0,
            klu_idx=2,
            probability=0.9,
            qualified=True,
            signal=-1,
        ),
    ]

    cfg = BacktestConfig(symbols=["BTCUSDT"], begin_time="2025-01-01", end_time="2025-01-02")
    matrix = build_signal_matrix(
        bars_index=bars.index,
        scored_events=events,
        allow_short=False,
        execution_mode="next_bar_open",
    )

    _pf, metrics, equity, dd, trades = run_vectorbt_for_symbol(cfg, bars, matrix)

    assert isinstance(metrics, dict)
    assert len(equity) == len(bars)
    assert len(dd) == len(bars)
