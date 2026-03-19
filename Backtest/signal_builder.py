from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional

import pandas as pd

from .types import ScoredSignalEvent, SignalMatrix


def _resolve_target_loc(index: pd.DatetimeIndex, ts: pd.Timestamp, execution_mode: str) -> Optional[int]:
    loc = int(index.searchsorted(ts, side="left"))
    if loc >= len(index):
        return None

    if execution_mode == "next_bar_open":
        loc += 1
        if loc >= len(index):
            return None

    return loc


def build_signal_matrix(
    bars_index: pd.DatetimeIndex,
    scored_events: List[ScoredSignalEvent],
    allow_short: bool,
    execution_mode: str,
    conflict_policy: str = "exit_first",
) -> SignalMatrix:
    if conflict_policy != "exit_first":
        raise ValueError("Only conflict_policy=exit_first is supported")

    long_entries = pd.Series(False, index=bars_index)
    long_exits = pd.Series(False, index=bars_index)
    short_entries = pd.Series(False, index=bars_index)
    short_exits = pd.Series(False, index=bars_index)
    signal = pd.Series(0, index=bars_index, dtype="int32")
    position = pd.Series(0, index=bars_index, dtype="int32")

    by_loc: Dict[int, List[ScoredSignalEvent]] = defaultdict(list)
    execution_pairs = []

    for event in scored_events:
        if event.signal == 0:
            continue
        loc = _resolve_target_loc(bars_index, event.exec_time, execution_mode)
        if loc is None:
            continue
        by_loc[loc].append(event)
        execution_pairs.append((event.exec_time, bars_index[loc]))

    current_pos = 0
    for i, ts in enumerate(bars_index):
        events = by_loc.get(i, [])
        has_buy = any(ev.signal == 1 for ev in events)
        has_sell = any(ev.signal == -1 for ev in events)

        action = 0

        if not allow_short:
            if has_buy and has_sell:
                if current_pos == 1:
                    long_exits.iat[i] = True
                    current_pos = 0
                    action = -1
                elif current_pos == 0:
                    long_entries.iat[i] = True
                    current_pos = 1
                    action = 1
            elif has_sell and current_pos == 1:
                long_exits.iat[i] = True
                current_pos = 0
                action = -1
            elif has_buy and current_pos == 0:
                long_entries.iat[i] = True
                current_pos = 1
                action = 1
        else:
            if has_buy and has_sell:
                if current_pos == 1:
                    long_exits.iat[i] = True
                    current_pos = 0
                    action = -1
                elif current_pos == -1:
                    short_exits.iat[i] = True
                    current_pos = 0
                    action = 1
            elif has_buy:
                if current_pos == -1:
                    short_exits.iat[i] = True
                    current_pos = 0
                    action = 1
                if current_pos == 0:
                    long_entries.iat[i] = True
                    current_pos = 1
                    action = 1
            elif has_sell:
                if current_pos == 1:
                    long_exits.iat[i] = True
                    current_pos = 0
                    action = -1
                if current_pos == 0:
                    short_entries.iat[i] = True
                    current_pos = -1
                    action = -1

        signal.iat[i] = action
        position.iat[i] = current_pos

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
