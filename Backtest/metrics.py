from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from .config import BacktestConfig
from .types import SymbolBacktestResult
from .vectorbt_engine import metrics_from_equity_curve


def empty_aggregate_metrics(initial_cash: float) -> Dict[str, float]:
    return {
        "total_return_pct": 0.0,
        "annualized_return_pct": 0.0,
        "max_drawdown_pct": 0.0,
        "sharpe": 0.0,
        "sortino": 0.0,
        "calmar": 0.0,
        "win_rate_pct": 0.0,
        "profit_factor": 0.0,
        "total_trades": 0.0,
        "avg_trade_return_pct": 0.0,
        "exposure_time_pct": 0.0,
        "start_cash": float(initial_cash),
        "end_equity": float(initial_cash),
    }


def aggregate_metrics(
    config: BacktestConfig,
    per_symbol: List[SymbolBacktestResult],
) -> Dict[str, float]:
    if len(per_symbol) == 1:
        return dict(per_symbol[0].metrics)

    curves = [item.equity_curve for item in per_symbol if len(item.equity_curve) > 0]
    if not curves:
        return empty_aggregate_metrics(config.initial_cash)

    first_index = curves[0].index
    same_index = all(curve.index.equals(first_index) for curve in curves[1:])
    if same_index:
        stacked = np.vstack(
            [curve.to_numpy(dtype=np.float64, copy=False) for curve in curves]
        )
        agg_equity = pd.Series(np.nanmean(stacked, axis=0), index=first_index)
    else:
        items = [item for item in per_symbol if len(item.equity_curve) > 0]
        eq_df = pd.concat(
            [item.equity_curve.rename(item.symbol) for item in items],
            axis=1,
        ).sort_index()
        eq_df = eq_df.ffill().dropna(how="all")
        if eq_df.empty:
            return empty_aggregate_metrics(config.initial_cash)
        agg_equity = eq_df.mean(axis=1)

    metrics = metrics_from_equity_curve(agg_equity, config.initial_cash)

    def metric_mean(metric_name: str) -> float:
        vals = []
        for item in per_symbol:
            value = item.metrics.get(metric_name)
            if isinstance(value, (int, float)) and np.isfinite(float(value)):
                vals.append(float(value))
        return float(np.mean(vals)) if vals else 0.0

    all_trades = [trade for item in per_symbol for trade in item.closed_trades]
    metrics["total_trades"] = float(len(all_trades))
    if all_trades:
        pnl = np.asarray(
            [float(trade.get("pnl", 0.0)) for trade in all_trades],
            dtype=np.float64,
        )
        returns = np.asarray(
            [float(trade.get("return_pct", 0.0)) for trade in all_trades],
            dtype=np.float64,
        )
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        metrics["win_rate_pct"] = float((pnl > 0).mean() * 100.0)
        metrics["profit_factor"] = (
            float(wins.sum() / abs(losses.sum())) if len(losses) else float("inf")
        )
        metrics["avg_trade_return_pct"] = float(np.nanmean(returns))
    else:
        metrics["win_rate_pct"] = 0.0
        metrics["profit_factor"] = 0.0
        metrics["avg_trade_return_pct"] = 0.0
    metrics["exposure_time_pct"] = metric_mean("exposure_time_pct")
    return metrics
