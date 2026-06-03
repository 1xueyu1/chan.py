import pandas as pd

from Backtest.config import BacktestConfig
from Backtest.risk import apply_risk_layer, build_risk_size_series
from Backtest.signal_builder import build_signal_matrix
from Backtest.types import ScoredSignalEvent


def _bars(periods=20):
    idx = pd.date_range("2025-01-01", periods=periods, freq="15min", tz="UTC")
    close = pd.Series(100.0, index=idx)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1.0,
        },
        index=idx,
    )


def _event(ts, signal):
    t = pd.Timestamp(ts)
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


def test_risk_size_series_clips_base_size():
    cfg = BacktestConfig(
        risk_enabled=True,
        risk_base_position_size=0.5,
        risk_min_position_size=0.1,
        risk_max_position_size=0.3,
        risk_use_volatility_target=False,
    )
    size = build_risk_size_series(cfg, _bars())
    assert float(size.iloc[0]) == 0.3


def test_risk_max_holding_forces_exit():
    bars = _bars(10)
    cfg = BacktestConfig(
        allow_short=True,
        risk_enabled=True,
        risk_base_position_size=0.25,
        risk_use_volatility_target=False,
        risk_max_holding_bars=3,
    )
    matrix = build_signal_matrix(
        bars_index=bars.index,
        scored_events=[_event(bars.index[0], 1)],
        allow_short=True,
        execution_mode="next_bar_open",
    )
    risked = apply_risk_layer(cfg, bars, matrix)
    assert risked.long_entries.iloc[1]
    assert risked.long_exits.iloc[4]
    assert risked.risk_forced_exits.iloc[4]
