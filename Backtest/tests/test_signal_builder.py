import pandas as pd

from Backtest.signal_builder import build_signal_matrix
from Backtest.types import ScoredSignalEvent


def _event(ts: str, signal: int, actual_ts: str | None = None) -> ScoredSignalEvent:
    t = pd.Timestamp(ts, tz="UTC")
    actual_exec_time = None
    if actual_ts is not None:
        actual_exec_time = pd.Timestamp(actual_ts)
        if actual_exec_time.tzinfo is None:
            actual_exec_time = actual_exec_time.tz_localize("UTC")
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
        actual_exec_time=actual_exec_time,
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


def test_long_short_exit_first_no_same_bar_reversal():
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
    assert not matrix.long_entries.iloc[3]
    assert matrix.position.iloc[3] == 0


def test_fixed_tp_sl_builds_entries_only():
    idx = pd.date_range("2025-01-01", periods=8, freq="15min", tz="UTC")
    events = [
        _event(str(idx[0]), 1),
        _event(str(idx[2]), -1),
    ]

    matrix = build_signal_matrix(
        bars_index=idx,
        scored_events=events,
        allow_short=True,
        execution_mode="next_bar_open",
        trade_exit_mode="fixed_tp_sl",
    )

    assert matrix.long_entries.iloc[1]
    assert matrix.short_entries.iloc[3]
    assert not matrix.long_exits.any()
    assert not matrix.short_exits.any()


def test_actual_exec_time_takes_precedence_for_alignment():
    idx = pd.date_range("2025-01-01", periods=8, freq="15min", tz="UTC")
    events = [
        _event("2025-01-01 00:00:00", 1, actual_ts=str(idx[2])),
    ]

    matrix = build_signal_matrix(
        bars_index=idx,
        scored_events=events,
        allow_short=False,
        execution_mode="next_bar_open",
    )

    assert matrix.long_entries.iloc[3]
    assert not matrix.long_entries.iloc[1]
