import pytest

from Backtest.config import BacktestConfig
from Backtest.engine import run_vectorbt_backtest


@pytest.mark.skip(reason="Heavy smoke test; run manually when data/model files are ready")
def test_smoke_e2e():
    cfg = BacktestConfig(
        symbols=["BTCUSDT"],
        begin_time="2025-01-01",
        end_time="2025-01-31",
        output_dir="result",
    )
    result = run_vectorbt_backtest(cfg)
    assert result.artifacts is not None
