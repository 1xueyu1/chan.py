import pandas as pd

from Backtest.signal_builder import build_signal_matrix
from Backtest.types import ScoredSignalEvent


def _event(ts: str, signal: int) -> ScoredSignalEvent:
    t = pd.Timestamp(ts, tz="UTC")
    return ScoredSignalEvent(
        symbol="BTCUSDT",
        exec_time=t,
        bsp_time=str(t),
        is_buy=signal > 0,
        bsp_type="1",
        bsp_types_str="1",
        trade_price=100.0,
        klu_idx=1,
        probability=0.9,
        qualified=True,
        signal=signal,
    )


def test_next_bar_open_shift_and_position_long_only():
    idx = pd.date_range("2025-01-01", periods=6, freq="15min", tz="UTC")
    events = [_event(str(idx[0]), 1), _event(str(idx[3]), -1)]

    matrix = build_signal_matrix(
        bars_index=idx,
        scored_events=events,
        allow_short=False,
        execution_mode="next_bar_open",
    )

    assert matrix.long_entries.iloc[1]
    assert matrix.long_exits.iloc[4]
    assert matrix.position.iloc[1] == 1
    assert matrix.position.iloc[4] == 0


def test_long_short_reversal():
    idx = pd.date_range("2025-01-01", periods=8, freq="15min", tz="UTC")
    events = [
        _event(str(idx[0]), -1),
        _event(str(idx[2]), 1),
    ]

    matrix = build_signal_matrix(
        bars_index=idx,
        scored_events=events,
        allow_short=True,
        execution_mode="next_bar_open",
    )

    assert matrix.short_entries.iloc[1]
    assert matrix.short_exits.iloc[3]
    assert matrix.long_entries.iloc[3]
