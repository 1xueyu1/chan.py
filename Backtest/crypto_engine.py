from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import pandas as pd

from .config import BacktestConfig
from .types import SignalMatrix
from .vectorbt_engine import metrics_from_equity_curve


@dataclass
class _Position:
    direction: int
    qty: float
    entry_price: float
    entry_idx: int
    entry_time: pd.Timestamp
    entry_fee: float
    notional: float
    equity_before: float
    leverage: float
    size_pct: float


def _apply_entry_slippage(price: float, direction: int, slippage: float) -> float:
    return price * (1.0 + slippage) if direction > 0 else price * (1.0 - slippage)


def _apply_exit_slippage(price: float, direction: int, slippage: float) -> float:
    return price * (1.0 - slippage) if direction > 0 else price * (1.0 + slippage)


def _pnl(position: _Position, price: float) -> float:
    return position.direction * position.qty * (price - position.entry_price)


def _notional(position: _Position, price: float) -> float:
    return abs(position.qty * price)


def _liquidation_price(config: BacktestConfig, position: _Position) -> Optional[float]:
    leverage = float(position.leverage)
    mmr = float(config.crypto_maintenance_margin_rate)
    if leverage <= 0:
        return None
    if position.direction > 0:
        return max(0.0, position.entry_price * (1.0 - 1.0 / leverage + mmr))
    return position.entry_price * (1.0 + 1.0 / leverage - mmr)


def _round_qty(qty: float, step: float) -> float:
    if step <= 0:
        return qty
    return np.floor(qty / step) * step


def _is_funding_bar(ts: pd.Timestamp) -> bool:
    ts = pd.Timestamp(ts)
    return ts.minute == 0 and ts.hour % 8 == 0


def _extract_signal(series: pd.Series, i: int) -> bool:
    return bool(series.iloc[i]) if i < len(series) else False


def _open_position(
    config: BacktestConfig,
    ts: pd.Timestamp,
    i: int,
    raw_price: float,
    direction: int,
    size_pct: float,
    leverage: float,
    equity: float,
) -> tuple[Optional[_Position], float, Optional[dict]]:
    if equity <= 0 or size_pct <= 0 or raw_price <= 0 or not np.isfinite(raw_price):
        return None, 0.0, None

    fill_price = _apply_entry_slippage(raw_price, direction, float(config.slippage))
    entry_leverage = max(1e-9, float(leverage))
    notional = equity * float(size_pct) * entry_leverage
    if notional < float(config.crypto_min_notional):
        return None, 0.0, None

    qty = _round_qty(notional / fill_price, float(config.crypto_qty_step))
    if qty <= 0 or qty < float(config.crypto_min_qty):
        return None, 0.0, None

    notional = qty * fill_price
    fee = notional * float(config.fee)
    position = _Position(
        direction=direction,
        qty=qty,
        entry_price=fill_price,
        entry_idx=i,
        entry_time=ts,
        entry_fee=fee,
        notional=notional,
        equity_before=equity,
        leverage=float(entry_leverage),
        size_pct=float(size_pct),
    )
    return position, fee, None


def _close_position(
    config: BacktestConfig,
    position: _Position,
    ts: pd.Timestamp,
    i: int,
    raw_price: float,
    reason: str,
    cash: float,
    funding_paid: float,
) -> tuple[float, dict]:
    exit_price = _apply_exit_slippage(raw_price, position.direction, float(config.slippage))
    pnl = _pnl(position, exit_price)
    exit_fee = _notional(position, exit_price) * float(config.fee)
    cash_after = cash + pnl - exit_fee
    total_fees = position.entry_fee + exit_fee
    equity_delta = cash_after - position.equity_before
    equity_delta_pct = equity_delta / position.equity_before * 100.0 if position.equity_before else 0.0
    price_return_pct = position.direction * (exit_price - position.entry_price) / position.entry_price * 100.0
    trade = {
        "symbol": "",
        "direction": "long" if position.direction > 0 else "short",
        "entry_time": position.entry_time.strftime("%Y-%m-%d %H:%M:%S"),
        "exit_time": pd.Timestamp(ts).strftime("%Y-%m-%d %H:%M:%S"),
        "holding_bars": max(0, i - position.entry_idx),
        "entry_price": float(position.entry_price),
        "exit_price": float(exit_price),
        "qty": float(position.qty),
        "notional": float(position.notional),
        "leverage": float(position.leverage),
        "size_pct": float(position.size_pct),
        "pnl": float(pnl),
        "return_pct": float(equity_delta_pct),
        "price_return_pct": float(price_return_pct),
        "fees": float(total_fees),
        "funding": float(funding_paid),
        "exit_reason": reason,
        "equity_before": float(position.equity_before),
        "equity_after": float(cash_after),
        "equity_delta": float(equity_delta),
        "equity_delta_pct": float(equity_delta_pct),
    }
    return cash_after, trade


def _fixed_exit_price(config: BacktestConfig, position: _Position, high: float, low: float) -> tuple[Optional[float], Optional[str]]:
    sl = float(config.fixed_stop_loss_pct)
    tp = float(config.fixed_take_profit_pct)
    if position.direction > 0:
        stop_price = position.entry_price * (1.0 - sl)
        take_price = position.entry_price * (1.0 + tp)
        hit_stop = low <= stop_price
        hit_take = high >= take_price
    else:
        stop_price = position.entry_price * (1.0 + sl)
        take_price = position.entry_price * (1.0 - tp)
        hit_stop = high >= stop_price
        hit_take = low <= take_price

    if hit_stop and hit_take:
        return (stop_price, "stop_loss") if config.crypto_stop_first else (take_price, "take_profit")
    if hit_stop:
        return stop_price, "stop_loss"
    if hit_take:
        return take_price, "take_profit"
    return None, None


def run_crypto_for_symbol(config: BacktestConfig, bars: pd.DataFrame, signal_matrix: SignalMatrix):
    index = pd.DatetimeIndex(bars.index)
    if len(index) == 0:
        equity_curve = pd.Series(dtype="float64", index=index)
        drawdown_curve = pd.Series(dtype="float64", index=index)
        return None, metrics_from_equity_curve(equity_curve, config.initial_cash), equity_curve, drawdown_curve, []

    open_ = bars["open"].astype(float).to_numpy()
    high = bars["high"].astype(float).to_numpy()
    low = bars["low"].astype(float).to_numpy()
    close = bars["close"].astype(float).to_numpy()
    exec_price = open_ if config.execution_mode == "next_bar_open" else close
    size = (
        signal_matrix.size.reindex(index).fillna(0.0).to_numpy(dtype="float64")
        if signal_matrix.size is not None
        else np.full(len(index), float(config.position_size), dtype="float64")
    )
    leverage = (
        signal_matrix.leverage.reindex(index).fillna(float(config.crypto_leverage)).to_numpy(dtype="float64")
        if signal_matrix.leverage is not None
        else np.full(len(index), float(config.crypto_leverage), dtype="float64")
    )
    leverage = np.where(np.isfinite(leverage) & (leverage > 0), leverage, float(config.crypto_leverage))

    cash = float(config.initial_cash)
    position: Optional[_Position] = None
    funding_for_position = 0.0
    entry_cooldown = 0
    loss_streak = 0
    closed_equity_peak = cash
    equity_values = []
    exposure = np.zeros(len(index), dtype=bool)
    trades = []

    def _register_closed_trade(trade: dict) -> None:
        nonlocal entry_cooldown, loss_streak, closed_equity_peak

        return_pct = float(trade.get("return_pct", 0.0) or 0.0)
        loss_streak = loss_streak + 1 if return_pct < 0.0 else 0
        closed_equity_peak = max(float(closed_equity_peak), float(cash))

        drawdown_stop = float(config.risk_symbol_drawdown_stop_pct)
        if drawdown_stop > 0 and closed_equity_peak > 0:
            closed_equity_drawdown = float(cash) / float(closed_equity_peak) - 1.0
            if closed_equity_drawdown <= -drawdown_stop:
                entry_cooldown = max(entry_cooldown, int(config.risk_cooldown_bars_after_drawdown))

        streak_stop = int(getattr(config, "risk_loss_streak_stop", 0) or 0)
        if streak_stop > 0 and loss_streak >= streak_stop:
            entry_cooldown = max(entry_cooldown, int(getattr(config, "risk_cooldown_bars_after_loss_streak", 0) or 0))
            loss_streak = 0

    for i, ts in enumerate(index):
        raw_price = float(exec_price[i])
        closed_this_bar = False

        if position is not None and float(config.crypto_funding_rate_8h) != 0.0 and _is_funding_bar(ts):
            mark_price = float(open_[i])
            funding = -position.direction * _notional(position, mark_price) * float(config.crypto_funding_rate_8h)
            cash += funding
            funding_for_position += funding

        if position is not None:
            liq_price = _liquidation_price(config, position)
            liquidated = False
            if liq_price is not None:
                if position.direction > 0 and low[i] <= liq_price:
                    cash, trade = _close_position(config, position, ts, i, liq_price, "liquidation", cash, funding_for_position)
                    trades.append(trade)
                    _register_closed_trade(trade)
                    position = None
                    funding_for_position = 0.0
                    liquidated = True
                    closed_this_bar = True
                elif position.direction < 0 and high[i] >= liq_price:
                    cash, trade = _close_position(config, position, ts, i, liq_price, "liquidation", cash, funding_for_position)
                    trades.append(trade)
                    _register_closed_trade(trade)
                    position = None
                    funding_for_position = 0.0
                    liquidated = True
                    closed_this_bar = True
            if liquidated:
                equity_values.append(max(cash, 0.0))
                continue

        if position is not None and config.trade_exit_mode == "fixed_tp_sl":
            exit_raw, reason = _fixed_exit_price(config, position, high[i], low[i])
            if exit_raw is not None and reason is not None:
                cash, trade = _close_position(config, position, ts, i, exit_raw, reason, cash, funding_for_position)
                trades.append(trade)
                _register_closed_trade(trade)
                position = None
                funding_for_position = 0.0
                closed_this_bar = True

        if (
            position is not None
            and config.trade_exit_mode == "fixed_tp_sl"
            and int(config.risk_max_holding_bars) > 0
            and i - position.entry_idx + 1 >= int(config.risk_max_holding_bars)
        ):
            cash, trade = _close_position(config, position, ts, i, float(close[i]), "time_stop", cash, funding_for_position)
            trades.append(trade)
            _register_closed_trade(trade)
            position = None
            funding_for_position = 0.0
            closed_this_bar = True

        if position is not None and config.trade_exit_mode == "opposite_signal":
            exit_signal = (
                _extract_signal(signal_matrix.long_exits, i)
                if position.direction > 0
                else _extract_signal(signal_matrix.short_exits, i)
            )
            if exit_signal:
                cash, trade = _close_position(config, position, ts, i, raw_price, "opposite_signal", cash, funding_for_position)
                trades.append(trade)
                _register_closed_trade(trade)
                position = None
                funding_for_position = 0.0
                closed_this_bar = True

        if position is None and entry_cooldown > 0:
            entry_cooldown -= 1
        elif position is None and cash > 0 and not closed_this_bar:
            direction = 0
            if _extract_signal(signal_matrix.long_entries, i):
                direction = 1
            elif config.allow_short and _extract_signal(signal_matrix.short_entries, i):
                direction = -1
            if direction != 0:
                pos, fee, _ = _open_position(
                    config,
                    ts,
                    i,
                    raw_price,
                    direction,
                    float(size[i]),
                    float(leverage[i]),
                    cash,
                )
                if pos is not None:
                    cash -= fee
                    pos.equity_before = cash + fee
                    position = pos
                    funding_for_position = 0.0

        if position is not None:
            exposure[i] = True
            mark_price = float(close[i])
            equity = cash + _pnl(position, mark_price)
        else:
            equity = cash
        equity_values.append(max(float(equity), 0.0))

        if equity <= 0:
            position = None
            cash = 0.0

    if position is not None:
        ts = index[-1]
        cash, trade = _close_position(config, position, ts, len(index) - 1, float(close[-1]), "end_of_data", cash, funding_for_position)
        trades.append(trade)
        _register_closed_trade(trade)
        equity_values[-1] = max(cash, 0.0)

    equity_curve = pd.Series(equity_values, index=index, dtype="float64")
    drawdown_curve = equity_curve / equity_curve.cummax() - 1.0
    metrics = metrics_from_equity_curve(equity_curve, config.initial_cash)

    if trades:
        trade_df = pd.DataFrame(trades)
        pnl = trade_df["pnl"].astype(float) + trade_df["funding"].astype(float) - trade_df["fees"].astype(float)
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        metrics.update(
            {
                "win_rate_pct": float((pnl > 0).mean() * 100.0),
                "profit_factor": float(wins.sum() / abs(losses.sum())) if abs(losses.sum()) > 0 else float("inf"),
                "total_trades": int(len(trades)),
                "avg_trade_return_pct": float(trade_df["return_pct"].astype(float).mean()),
                "exposure_time_pct": float(exposure.mean() * 100.0),
            }
        )
    else:
        metrics.update(
            {
                "win_rate_pct": 0.0,
                "profit_factor": 0.0,
                "total_trades": 0,
                "avg_trade_return_pct": 0.0,
                "exposure_time_pct": float(exposure.mean() * 100.0),
            }
        )

    return None, metrics, equity_curve, drawdown_curve, trades
