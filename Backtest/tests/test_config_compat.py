from Backtest.config import BacktestConfig
from Backtest.settings import BacktestSettings


def test_backtest_config_to_settings_preserves_key_fields():
    cfg = BacktestConfig(
        symbols=["BTCUSDT", "ETHUSDT"],
        begin_time="2025-01-01",
        end_time="2025-02-01",
        allow_short=True,
        ml_enabled=True,
        ml_buy_model_path="buy.pkl",
        risk_enabled=True,
        execution_engine="crypto",
    )
    cfg.chan_config["use_rust_core"] = True

    settings = cfg.to_settings()

    assert isinstance(settings, BacktestSettings)
    assert settings.universe.symbols == ["BTCUSDT", "ETHUSDT"]
    assert settings.execution.allow_short is True
    assert settings.ml.enabled is True
    assert settings.ml.buy_model_path == "buy.pkl"
    assert settings.risk.enabled is True
    assert settings.chan.use_rust_core is True
