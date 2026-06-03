from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd
from Common.CEnum import KL_TYPE

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
    price = bars["open"] if config.execution_mode == "next_bar_open" else bars["close"]

    has_any_signal = bool(signal_matrix.long_entries.any()) or bool(signal_matrix.long_exits.any())
    if config.allow_short:
        has_any_signal = (
            has_any_signal
            or bool(signal_matrix.short_entries.any())
            or bool(signal_matrix.short_exits.any())
        )

    # Fast path: skip vectorbt portfolio construction when there is no effective signal.
    if not has_any_signal:
        if len(price.index) > 0:
            equity_curve = pd.Series(float(config.initial_cash), index=price.index, dtype="float64")
            drawdown_curve = pd.Series(0.0, index=price.index, dtype="float64")
        else:
            empty_index = pd.DatetimeIndex([], name="time")
            equity_curve = pd.Series(dtype="float64", index=empty_index)
            drawdown_curve = pd.Series(dtype="float64", index=empty_index)

        metrics = metrics_from_equity_curve(equity_curve, config.initial_cash)
        metrics.update(
            {
                "win_rate_pct": 0.0,
                "profit_factor": 0.0,
                "total_trades": 0,
                "avg_trade_return_pct": 0.0,
                "exposure_time_pct": 0.0,
            }
        )
        return None, metrics, equity_curve, drawdown_curve, []

    try:
        import vectorbt as vbt
    except ImportError as exc:
        raise ImportError("vectorbt is required. Install it with: pip install vectorbt") from exc

    freq_map = {
        KL_TYPE.K_1M: "1min",
        KL_TYPE.K_3M: "3min",
        KL_TYPE.K_5M: "5min",
        KL_TYPE.K_10M: "10min",
        KL_TYPE.K_15M: "15min",
        KL_TYPE.K_30M: "30min",
        KL_TYPE.K_60M: "1h",
        KL_TYPE.K_DAY: "1d",
        KL_TYPE.K_WEEK: "1w",
        KL_TYPE.K_MON: "1mo",
    }
    freq = freq_map.get(config.kl_type)
    if freq is None:
        freq = pd.infer_freq(price.index) or "15min"

    kwargs = dict(
        close=price,
        init_cash=config.initial_cash,
        fees=config.fee,
        slippage=config.slippage,
        size=signal_matrix.size if signal_matrix.size is not None else config.position_size,
        size_type="percent",
        freq=freq,
    )
    if config.trade_exit_mode == "fixed_tp_sl":
        kwargs.update(
            sl_stop=float(config.fixed_stop_loss_pct),
            tp_stop=float(config.fixed_take_profit_pct),
        )

    if config.allow_short:
        call_kwargs = dict(
            entries=signal_matrix.long_entries,
            exits=signal_matrix.long_exits,
            short_entries=signal_matrix.short_entries,
            short_exits=signal_matrix.short_exits,
            **kwargs,
        )
        if config.trade_exit_mode == "fixed_tp_sl":
            call_kwargs["upon_opposite_entry"] = "ignore"
        pf = vbt.Portfolio.from_signals(**call_kwargs)
    else:
        pf = vbt.Portfolio.from_signals(
            entries=signal_matrix.long_entries,
            exits=signal_matrix.long_exits,
            **kwargs,
        )

    metrics = extract_metrics(config, pf)

    equity_curve = pf.value()
    drawdown_curve = equity_curve / equity_curve.cummax() - 1.0

    closed_trades = extract_closed_trades(config, pf, bars, signal_matrix, equity_curve)
    # Keep aggregate/stat table semantics aligned with detail table semantics.
    metrics["total_trades"] = int(len(closed_trades))

    return pf, metrics, equity_curve, drawdown_curve, closed_trades


def _first_non_null(row: pd.Series, keys: list[str], default=None):
    for key in keys:
        if key not in row:
            continue
        value = row.get(key)
        if value is None:
            continue
        if isinstance(value, float) and np.isnan(value):
            continue
        return value
    return default


def _resolve_bar_index(raw_value, price_index: pd.Index) -> int:
    if raw_value is None:
        return -1

    try:
        value = int(raw_value)
        return value
    except Exception:
        pass

    try:
        ts = pd.Timestamp(raw_value)
    except Exception:
        return -1

    if ts in price_index:
        loc = price_index.get_loc(ts)
        if isinstance(loc, slice):
            return int(loc.start)
        if isinstance(loc, np.ndarray):
            return int(loc[0]) if len(loc) else -1
        return int(loc)

    pos = int(price_index.searchsorted(ts))
    if 0 <= pos < len(price_index):
        return pos
    return -1


def extract_closed_trades(config: BacktestConfig, pf, bars: pd.DataFrame, signal_matrix: SignalMatrix, equity_curve: pd.Series) -> list[dict]:
    try:
        records = pf.trades.records
    except Exception:
        records = None

    if records is None or len(records) == 0:
        try:
            records = pf.trades.records_readable
        except Exception:
            return []

    if records is None or len(records) == 0:
        return []

    out = []
    price_series = bars["open"] if config.execution_mode == "next_bar_open" else bars["close"]

    for _, row in records.iterrows():
        try:
            entry_idx_raw = _first_non_null(
                row,
                ["entry_idx", "Entry Index", "Entry Idx", "Entry Timestamp", "Entry Time"],
                None,
            )
            exit_idx_raw = _first_non_null(
                row,
                ["exit_idx", "Exit Index", "Exit Idx", "Exit Timestamp", "Exit Time"],
                None,
            )
            entry_idx = _resolve_bar_index(entry_idx_raw, price_series.index)
            exit_idx = _resolve_bar_index(exit_idx_raw, price_series.index)
            if entry_idx < 0 or exit_idx < 0:
                continue

            if entry_idx >= len(price_series.index) or exit_idx >= len(price_series.index):
                continue

            entry_ts = price_series.index[entry_idx]
            exit_ts = price_series.index[exit_idx]

            entry_price = float(_first_non_null(row, ["Avg Entry Price", "Entry Price", "entry_price"], float("nan")))
            exit_price = float(_first_non_null(row, ["Avg Exit Price", "Exit Price", "exit_price"], float("nan")))
            pnl = float(_first_non_null(row, ["PnL", "pnl"], float("nan")))

            raw_return = _first_non_null(row, ["Return", "return", "return_pct"], float("nan"))
            ret_pct = float(raw_return)
            if np.isfinite(ret_pct) and abs(ret_pct) <= 2.0:
                ret_pct = ret_pct * 100.0

            fees = float(_first_non_null(row, ["fees", "Fees"], float("nan")))
            if (isinstance(fees, float) and np.isnan(fees)) and ("entry_fees" in row or "exit_fees" in row):
                entry_fees = float(_first_non_null(row, ["entry_fees"], 0.0) or 0.0)
                exit_fees = float(_first_non_null(row, ["exit_fees"], 0.0) or 0.0)
                fees = entry_fees + exit_fees

            eq_before = float(equity_curve.iloc[entry_idx - 1]) if entry_idx > 0 else float(config.initial_cash)
            eq_after = float(equity_curve.iloc[exit_idx])
            eq_delta = eq_after - eq_before
            eq_delta_pct = (eq_delta / eq_before * 100.0) if eq_before != 0 else float("nan")

            direction = "long"
            direction_raw = _first_non_null(row, ["Direction", "direction"], "")
            if direction_raw:
                direction_text = str(direction_raw).lower()
                if "short" in direction_text:
                    direction = "short"

            out.append(
                {
                    "symbol": "",
                    "direction": direction,
                    "entry_time": pd.Timestamp(entry_ts).strftime("%Y-%m-%d %H:%M:%S"),
                    "exit_time": pd.Timestamp(exit_ts).strftime("%Y-%m-%d %H:%M:%S"),
                    "holding_bars": max(0, exit_idx - entry_idx),
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "return_pct": ret_pct,
                    "fees": fees,
                    "equity_before": eq_before,
                    "equity_after": eq_after,
                    "equity_delta": eq_delta,
                    "equity_delta_pct": eq_delta_pct,
                }
            )
        except Exception:
            continue

    return out


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
