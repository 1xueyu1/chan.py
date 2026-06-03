import pandas as pd

from Backtest.time_utils import parse_time_bound, to_utc_timestamp


def test_parse_time_bound_date_end_expands_to_end_of_day():
    ts = parse_time_bound("2025-01-01", is_end=True)

    assert ts == pd.Timestamp("2025-01-01 23:59:59", tz="UTC")


def test_to_utc_timestamp_localizes_naive_and_converts_aware():
    assert to_utc_timestamp("2025-01-01 00:00:00") == pd.Timestamp(
        "2025-01-01 00:00:00",
        tz="UTC",
    )
    assert to_utc_timestamp("2025-01-01 08:00:00+08:00") == pd.Timestamp(
        "2025-01-01 00:00:00",
        tz="UTC",
    )
