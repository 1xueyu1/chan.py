from pathlib import Path

import pandas as pd

from Backtest.config import BacktestConfig
from Backtest.data_loader import load_symbol_bars


def test_load_symbol_bars_basic(monkeypatch, tmp_path: Path):
    parquet_path = tmp_path / "BTC_15m.parquet"
    src = pd.DataFrame(
        {
            "open_time": [
                1735689600000,
                1735690500000,
                1735691400000,
            ],
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [10.0, 11.0, 12.0],
        }
    )
    src.to_parquet(parquet_path, index=False)

    def fake_resolve_source_path(data_dir, symbol, kl_type):
        return parquet_path, False

    monkeypatch.setattr("Backtest.data_loader._resolve_source_path", fake_resolve_source_path)

    cfg = BacktestConfig(symbols=["BTCUSDT"], begin_time="2025-01-01", end_time="2025-01-02")
    bars = load_symbol_bars(cfg, "BTCUSDT")

    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    assert len(bars) == 3
    assert bars.index.tz is not None
