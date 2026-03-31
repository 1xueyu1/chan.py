from __future__ import annotations

from typing import List

import numpy as np
import pandas as pd

from .types import ScoredSignalEvent, SignalMatrix


def build_signal_matrix(
    bars_index: pd.DatetimeIndex,
    scored_events: List[ScoredSignalEvent],
    allow_short: bool,
    execution_mode: str,
    conflict_policy: str = "exit_first",
    cooldown_bars: int = 0,
) -> SignalMatrix:
    if conflict_policy != "exit_first":
        raise ValueError("Only conflict_policy=exit_first is supported")

    n = len(bars_index)
    long_entries_arr = np.zeros(n, dtype=bool)
    long_exits_arr = np.zeros(n, dtype=bool)
    short_entries_arr = np.zeros(n, dtype=bool)
    short_exits_arr = np.zeros(n, dtype=bool)
    signal_arr = np.zeros(n, dtype=np.int32)
    position_arr = np.zeros(n, dtype=np.int32)

    has_buy = np.zeros(n, dtype=bool)
    has_sell = np.zeros(n, dtype=bool)
    execution_pairs = []

    active_events = [event for event in scored_events if event.signal != 0]
    if not active_events:
        long_entries = pd.Series(long_entries_arr, index=bars_index)
        long_exits = pd.Series(long_exits_arr, index=bars_index)
        short_entries = pd.Series(short_entries_arr, index=bars_index)
        short_exits = pd.Series(short_exits_arr, index=bars_index)
        signal = pd.Series(signal_arr, index=bars_index, dtype="int32")
        position = pd.Series(position_arr, index=bars_index, dtype="int32")
        return SignalMatrix(
            long_entries=long_entries,
            long_exits=long_exits,
            short_entries=short_entries,
            short_exits=short_exits,
            signal=signal,
            position=position,
            execution_pairs=execution_pairs,
        )

    index_ns = bars_index.view("int64")
    event_times_ns = np.fromiter(
        (int(pd.Timestamp(event.exec_time).value) for event in active_events),
        dtype=np.int64,
        count=len(active_events),
    )
    locs = np.searchsorted(index_ns, event_times_ns, side="left")
    if execution_mode == "next_bar_open":
        locs = locs + 1

    valid_mask = (locs >= 0) & (locs < n)
    if np.any(valid_mask):
        valid_locs = locs[valid_mask].astype(np.intp, copy=False)
        valid_signals = np.fromiter(
            (event.signal for event, is_valid in zip(active_events, valid_mask) if bool(is_valid)),
            dtype=np.int8,
            count=int(valid_mask.sum()),
        )
        if np.any(valid_signals == 1):
            np.logical_or.at(has_buy, valid_locs[valid_signals == 1], True)
        if np.any(valid_signals == -1):
            np.logical_or.at(has_sell, valid_locs[valid_signals == -1], True)

    execution_pairs = [
        (event.exec_time, bars_index[int(loc)])
        for event, loc, is_valid in zip(active_events, locs, valid_mask)
        if bool(is_valid)
    ]

    if not has_buy.any() and not has_sell.any():
        long_entries = pd.Series(long_entries_arr, index=bars_index)
        long_exits = pd.Series(long_exits_arr, index=bars_index)
        short_entries = pd.Series(short_entries_arr, index=bars_index)
        short_exits = pd.Series(short_exits_arr, index=bars_index)
        signal = pd.Series(signal_arr, index=bars_index, dtype="int32")
        position = pd.Series(position_arr, index=bars_index, dtype="int32")
        return SignalMatrix(
            long_entries=long_entries,
            long_exits=long_exits,
            short_entries=short_entries,
            short_exits=short_exits,
            signal=signal,
            position=position,
            execution_pairs=execution_pairs,
        )

    current_pos = 0
    cooldown_remaining = 0
    for i in range(n):

        if cooldown_remaining > 0:
            position_arr[i] = current_pos
            signal_arr[i] = 0
            cooldown_remaining -= 1
            continue

        buy_at_i = bool(has_buy[i])
        sell_at_i = bool(has_sell[i])

        action = 0

        if not allow_short:
            if buy_at_i and sell_at_i:
                if current_pos == 1:
                    long_exits_arr[i] = True
                    current_pos = 0
                    action = -1
                elif current_pos == 0:
                    long_entries_arr[i] = True
                    current_pos = 1
                    action = 1
            elif sell_at_i and current_pos == 1:
                long_exits_arr[i] = True
                current_pos = 0
                action = -1
            elif buy_at_i and current_pos == 0:
                long_entries_arr[i] = True
                current_pos = 1
                action = 1
        else:
            if buy_at_i and sell_at_i:
                if current_pos == 1:
                    long_exits_arr[i] = True
                    current_pos = 0
                    action = -1
                elif current_pos == -1:
                    short_exits_arr[i] = True
                    current_pos = 0
                    action = 1
            elif buy_at_i:
                if current_pos == -1:
                    short_exits_arr[i] = True
                    current_pos = 0
                    action = 1
                if current_pos == 0:
                    long_entries_arr[i] = True
                    current_pos = 1
                    action = 1
            elif sell_at_i:
                if current_pos == 1:
                    long_exits_arr[i] = True
                    current_pos = 0
                    action = -1
                if current_pos == 0:
                    short_entries_arr[i] = True
                    current_pos = -1
                    action = -1

        signal_arr[i] = action
        position_arr[i] = current_pos
        if action != 0 and cooldown_bars > 0:
            cooldown_remaining = cooldown_bars

    long_entries = pd.Series(long_entries_arr, index=bars_index)
    long_exits = pd.Series(long_exits_arr, index=bars_index)
    short_entries = pd.Series(short_entries_arr, index=bars_index)
    short_exits = pd.Series(short_exits_arr, index=bars_index)
    signal = pd.Series(signal_arr, index=bars_index, dtype="int32")
    position = pd.Series(position_arr, index=bars_index, dtype="int32")

    return SignalMatrix(
        long_entries=long_entries,
        long_exits=long_exits,
        short_entries=short_entries,
        short_exits=short_exits,
        signal=signal,
        position=position,
        execution_pairs=execution_pairs,
    )


def validate_no_lookahead(signal_matrix: SignalMatrix, execution_mode: str) -> None:
    if execution_mode != "next_bar_open":
        return
    for signal_ts, fill_ts in signal_matrix.execution_pairs:
        if not (fill_ts > signal_ts):
            raise AssertionError(
                f"Lookahead detected: fill_ts={fill_ts}, signal_ts={signal_ts}"
            )
