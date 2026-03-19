from pathlib import Path

# flake8: noqa: E501

import pandas as pd
import pytest

from Backtest.config import BacktestConfig
from Backtest.event_replay import load_scored_events_by_symbol


@pytest.fixture
def replay_csv(tmp_path: Path) -> Path:
    path = tmp_path / "events.csv"
    df = pd.DataFrame(
        [
            {
                "exec_time": "2025-01-01 00:15:00",
                "bsp_time": "2025-01-01 00:00:00",
                "is_buy": True,
                "bsp_type": "1",
                "bsp_types_str": "1",
                "probability": 0.62,
                "qualified": False,
                "signal": 0,
                "trade_price": 100.0,
                "symbol": "BTCUSDT",
            },
            {
                "exec_time": "2025-01-01 00:30:00",
                "bsp_time": "2025-01-01 00:15:00",
                "is_buy": False,
                "bsp_type": "2",
                "bsp_types_str": "2",
                "probability": 0.70,
                "qualified": True,
                "signal": -1,
                "trade_price": 101.0,
                "symbol": "BTCUSDT",
            },
            {
                "exec_time": "2025-01-01 00:45:00",
                "bsp_time": "2025-01-01 00:30:00",
                "is_buy": True,
                "bsp_type": "1",
                "bsp_types_str": "1",
                "probability": 0.90,
                "qualified": True,
                "signal": 1,
                "trade_price": 200.0,
                "symbol": "ETHUSDT",
            },
        ]
    )
    df.to_csv(path, index=False, encoding="utf-8")
    return path


def test_event_replay_loads_symbol_subset(replay_csv: Path):
    cfg = BacktestConfig(
        symbols=["BTCUSDT"],
        begin_time="2025-01-01",
        end_time="2025-01-02",
        event_replay_mode=True,
        event_replay_csv_path=str(replay_csv),
        replay_reapply_threshold=False,
    )

    grouped = load_scored_events_by_symbol(cfg)

    assert list(grouped.keys()) == ["BTCUSDT"]
    assert len(grouped["BTCUSDT"]) == 2
    assert grouped["BTCUSDT"][0].signal == 0
    assert grouped["BTCUSDT"][1].signal == -1


def test_event_replay_reapply_threshold(replay_csv: Path):
    cfg = BacktestConfig(
        symbols=["BTCUSDT"],
        begin_time="2025-01-01",
        end_time="2025-01-02",
        signal_threshold=0.60,
        event_replay_mode=True,
        event_replay_csv_path=str(replay_csv),
        replay_reapply_threshold=True,
    )

    grouped = load_scored_events_by_symbol(cfg)

    first = grouped["BTCUSDT"][0]
    second = grouped["BTCUSDT"][1]
    assert first.signal == 1
    assert first.qualified is True
    assert second.signal == -1
    assert second.qualified is True


def test_event_replay_missing_columns(tmp_path: Path):
    path = tmp_path / "bad_events.csv"
    pd.DataFrame([{"exec_time": "2025-01-01 00:15:00", "symbol": "BTCUSDT"}]).to_csv(path, index=False)

    cfg = BacktestConfig(
        symbols=["BTCUSDT"],
        begin_time="2025-01-01",
        end_time="2025-01-02",
        event_replay_mode=True,
        event_replay_csv_path=str(path),
    )

    with pytest.raises(ValueError):
        load_scored_events_by_symbol(cfg)
