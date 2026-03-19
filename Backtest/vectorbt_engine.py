from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .config import BacktestConfig
from .types import SignalMatrix


def _safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def run_vectorbt_for_symbol(
    config: BacktestConfig,
    bars: pd.DataFrame,
    signal_matrix: SignalMatrix,
):
    try:
        import vectorbt as vbt
    except ImportError as exc:
        raise ImportError("vectorbt is required. Install it with: pip install vectorbt") from exc

    price = bars["open"] if config.execution_mode == "next_bar_open" else bars["close"]

    freq = pd.infer_freq(price.index)
    kwargs = dict(
        close=price,
        init_cash=config.initial_cash,
        fees=config.fee,
        slippage=config.slippage,
        size=1.0,
        size_type="percent",
        freq=freq,
    )

    if config.allow_short:
        pf = vbt.Portfolio.from_signals(
            entries=signal_matrix.long_entries,
            exits=signal_matrix.long_exits,
            short_entries=signal_matrix.short_entries,
            short_exits=signal_matrix.short_exits,
            **kwargs,
        )
    else:
        pf = vbt.Portfolio.from_signals(
            entries=signal_matrix.long_entries,
            exits=signal_matrix.long_exits,
            **kwargs,
        )

    metrics = extract_metrics(config, pf)

    equity_curve = pf.value()
    drawdown_curve = equity_curve / equity_curve.cummax() - 1.0

    return pf, metrics, equity_curve, drawdown_curve


def extract_metrics(config: BacktestConfig, pf) -> Dict[str, float]:
    trades = pf.trades

    total_return_pct = _safe_float(pf.total_return()) * 100.0
    annualized_return_pct = _safe_float(pf.annualized_return()) * 100.0
    max_drawdown_pct = _safe_float(pf.max_drawdown()) * 100.0

    sharpe = _safe_float(pf.sharpe_ratio())
    sortino = _safe_float(pf.sortino_ratio())
    calmar = _safe_float(pf.calmar_ratio())

    total_trades = int(_safe_float(trades.count()))
    win_rate_pct = _safe_float(trades.win_rate()) * 100.0
    profit_factor = _safe_float(trades.profit_factor())
    avg_trade_return_pct = _safe_float(trades.returns.mean()) * 100.0

    try:
        exposure_time_pct = float(np.mean(pf.position_mask().to_numpy())) * 100.0
    except Exception:
        exposure_time_pct = float("nan")

    value = pf.value()
    end_equity = _safe_float(value.iloc[-1]) if len(value) else config.initial_cash

    return {
        "total_return_pct": total_return_pct,
        "annualized_return_pct": annualized_return_pct,
        "max_drawdown_pct": max_drawdown_pct,
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "win_rate_pct": win_rate_pct,
        "profit_factor": profit_factor,
        "total_trades": total_trades,
        "avg_trade_return_pct": avg_trade_return_pct,
        "exposure_time_pct": exposure_time_pct,
        "start_cash": float(config.initial_cash),
        "end_equity": end_equity,
    }


def metrics_from_equity_curve(equity_curve: pd.Series, initial_cash: float) -> Dict[str, float]:
    if len(equity_curve) < 2:
        return {
            "total_return_pct": 0.0,
            "annualized_return_pct": 0.0,
            "max_drawdown_pct": 0.0,
            "sharpe": 0.0,
            "sortino": 0.0,
            "calmar": 0.0,
            "start_cash": float(initial_cash),
            "end_equity": float(initial_cash),
        }

    returns = equity_curve.pct_change().fillna(0.0)
    total_return = (equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1.0

    time_span_days = max((equity_curve.index[-1] - equity_curve.index[0]).days, 1)
    years = time_span_days / 365.0
    annualized = (1.0 + total_return) ** (1.0 / years) - 1.0 if years > 0 else total_return

    drawdown = equity_curve / equity_curve.cummax() - 1.0
    max_drawdown = float(drawdown.min())

    std = float(returns.std())
    downside_std = float(returns[returns < 0].std())

    sharpe = (returns.mean() / std * np.sqrt(252.0)) if std > 0 else 0.0
    sortino = (returns.mean() / downside_std * np.sqrt(252.0)) if downside_std > 0 else 0.0
    calmar = (annualized / abs(max_drawdown)) if max_drawdown < 0 else 0.0

    return {
        "total_return_pct": float(total_return * 100.0),
        "annualized_return_pct": float(annualized * 100.0),
        "max_drawdown_pct": float(max_drawdown * 100.0),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "calmar": float(calmar),
        "start_cash": float(initial_cash),
        "end_equity": float(equity_curve.iloc[-1]),
    }
