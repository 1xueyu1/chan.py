import pandas as pd

from Backtest.config import BacktestConfig
from Backtest.crypto_engine import run_crypto_for_symbol
from Backtest.signal_builder import build_signal_matrix
from Backtest.types import ScoredSignalEvent


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


def test_crypto_engine_fixed_tp_sl_uses_ohlc_stops():
    idx = pd.date_range("2025-01-01", periods=6, freq="15min", tz="UTC")
    bars = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.0, 100.0, 100.0],
            "high": [101.0, 101.0, 106.0, 101.0, 101.0, 101.0],
            "low": [99.0, 99.0, 99.0, 99.0, 99.0, 99.0],
            "close": [100.0, 100.0, 105.0, 100.0, 100.0, 100.0],
            "volume": [1.0] * 6,
        },
        index=idx,
    )
    cfg = BacktestConfig(
        symbols=["BTCUSDT"],
        execution_engine="crypto",
        trade_exit_mode="fixed_tp_sl",
        fixed_stop_loss_pct=0.02,
        fixed_take_profit_pct=0.05,
        fee=0.0,
        slippage=0.0,
        position_size=1.0,
    )
    matrix = build_signal_matrix(
        bars_index=idx,
        scored_events=[_event(idx[0], 1)],
        allow_short=False,
        execution_mode="next_bar_open",
        trade_exit_mode="fixed_tp_sl",
    )

    _pf, metrics, equity, _dd, trades = run_crypto_for_symbol(cfg, bars, matrix)

    assert metrics["total_trades"] == 1
    assert trades[0]["exit_reason"] == "take_profit"
    assert trades[0]["exit_time"] == idx[2].strftime("%Y-%m-%d %H:%M:%S")
    assert equity.iloc[-1] > cfg.initial_cash


def test_crypto_engine_stop_first_when_both_hit():
    idx = pd.date_range("2025-01-01", periods=4, freq="15min", tz="UTC")
    bars = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.0],
            "high": [101.0, 101.0, 106.0, 101.0],
            "low": [99.0, 99.0, 97.0, 99.0],
            "close": [100.0, 100.0, 100.0, 100.0],
            "volume": [1.0] * 4,
        },
        index=idx,
    )
    cfg = BacktestConfig(
        symbols=["BTCUSDT"],
        execution_engine="crypto",
        trade_exit_mode="fixed_tp_sl",
        fixed_stop_loss_pct=0.02,
        fixed_take_profit_pct=0.05,
        fee=0.0,
        slippage=0.0,
        position_size=1.0,
    )
    matrix = build_signal_matrix(
        bars_index=idx,
        scored_events=[_event(idx[0], 1)],
        allow_short=False,
        execution_mode="next_bar_open",
        trade_exit_mode="fixed_tp_sl",
    )

    _pf, metrics, _equity, _dd, trades = run_crypto_for_symbol(cfg, bars, matrix)

    assert metrics["total_trades"] == 1
    assert trades[0]["exit_reason"] == "stop_loss"


def test_crypto_engine_does_not_reenter_after_intrabar_exit_on_same_bar():
    idx = pd.date_range("2025-01-01", periods=5, freq="15min", tz="UTC")
    bars = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0, 100.0, 100.0],
            "high": [101.0, 101.0, 106.0, 101.0, 101.0],
            "low": [99.0, 99.0, 99.0, 99.0, 99.0],
            "close": [100.0, 100.0, 105.0, 100.0, 100.0],
            "volume": [1.0] * 5,
        },
        index=idx,
    )
    cfg = BacktestConfig(
        symbols=["BTCUSDT"],
        execution_engine="crypto",
        trade_exit_mode="fixed_tp_sl",
        fixed_stop_loss_pct=0.02,
        fixed_take_profit_pct=0.05,
        fee=0.0,
        slippage=0.0,
        position_size=1.0,
    )
    matrix = build_signal_matrix(
        bars_index=idx,
        scored_events=[
            _event(idx[0], 1),  # enters on idx[1]
            _event(idx[1], 1),  # would enter on idx[2], the same bar as the TP exit
        ],
        allow_short=False,
        execution_mode="next_bar_open",
        trade_exit_mode="fixed_tp_sl",
    )

    _pf, metrics, _equity, _dd, trades = run_crypto_for_symbol(cfg, bars, matrix)

    assert metrics["total_trades"] == 1
    assert trades[0]["exit_reason"] == "take_profit"
    assert trades[0]["exit_time"] == idx[2].strftime("%Y-%m-%d %H:%M:%S")


def test_crypto_engine_fixed_tp_sl_respects_max_holding_bars():
    idx = pd.date_range("2025-01-01", periods=6, freq="15min", tz="UTC")
    bars = pd.DataFrame(
        {
            "open": [100.0, 100.0, 101.0, 102.0, 103.0, 104.0],
            "high": [101.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            "low": [99.0, 99.0, 100.0, 101.0, 102.0, 103.0],
            "close": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
            "volume": [1.0] * 6,
        },
        index=idx,
    )
    cfg = BacktestConfig(
        symbols=["BTCUSDT"],
        execution_engine="crypto",
        trade_exit_mode="fixed_tp_sl",
        fixed_stop_loss_pct=0.10,
        fixed_take_profit_pct=0.10,
        risk_max_holding_bars=3,
        fee=0.0,
        slippage=0.0,
        position_size=1.0,
    )
    matrix = build_signal_matrix(
        bars_index=idx,
        scored_events=[_event(idx[0], 1)],
        allow_short=False,
        execution_mode="next_bar_open",
        trade_exit_mode="fixed_tp_sl",
    )

    _pf, metrics, _equity, _dd, trades = run_crypto_for_symbol(cfg, bars, matrix)

    assert metrics["total_trades"] == 1
    assert trades[0]["exit_reason"] == "time_stop"
    assert trades[0]["exit_time"] == idx[3].strftime("%Y-%m-%d %H:%M:%S")
