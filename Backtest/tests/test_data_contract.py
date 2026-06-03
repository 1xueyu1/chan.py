import pandas as pd

from Backtest.data_contract import coerce_open_time_to_utc, normalize_bars


def test_coerce_open_time_to_utc_mixed_epoch_units():
    values = pd.Series(
        [
            1735689600,
            1735689600000,
            1735689600000000,
            1735689600000000000,
            "2025-01-01 00:00:00+00:00",
        ]
    )

    parsed = coerce_open_time_to_utc(values)

    assert parsed.notna().all()
    assert parsed.iloc[0] == pd.Timestamp("2025-01-01", tz="UTC")
    assert parsed.iloc[1] == parsed.iloc[0]
    assert parsed.iloc[2] == parsed.iloc[0]
    assert parsed.iloc[3] == parsed.iloc[0]
    assert parsed.iloc[4] == parsed.iloc[0]


def test_normalize_bars_sorts_dedupes_and_sets_contract():
    src = pd.DataFrame(
        {
            "open_time": [1735690500000, 1735689600000, 1735689600000],
            "open": [101.0, 100.0, 999.0],
            "high": [102.0, 101.0, 999.0],
            "low": [100.0, 99.0, 999.0],
            "close": [101.5, 100.5, 999.0],
            "volume": [11.0, 10.0, None],
        }
    )

    bars = normalize_bars(src)

    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    assert bars.index.name == "time"
    assert str(bars.index.tz) == "UTC"
    assert bars.index.is_monotonic_increasing
    assert bars.index.is_unique
    assert len(bars) == 2
    assert bars.iloc[0]["open"] == 100.0
