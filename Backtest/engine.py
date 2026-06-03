from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ChanConfig import CChanConfig
from Common.CEnum import DATA_SRC, KL_TYPE
from RustCore import rust_config_path_from_chan_config
from ML.shared.position_sizing import PositionSizingConfig

from .chan_signal_extractor import extract_raw_bsp_events, extract_raw_bsp_events_from_bars
from .config import BacktestConfig
from .crypto_engine import run_crypto_for_symbol
from .data_loader import load_symbol_bars, preload_symbol_bars
from .metrics import aggregate_metrics as aggregate_symbol_metrics
from .ml_filter import score_raw_events_with_ml
from .time_utils import event_signal_time, parse_time_bound
from .reporter import write_outputs
from .risk import apply_risk_layer
from .signal_builder import build_signal_matrix, validate_no_lookahead
from .strategy import ChanStrategyBase
from .types import BacktestRunResult, RawBSPEvent, ScoredSignalEvent, SignalMatrix, SymbolBacktestResult
from .vectorbt_engine import metrics_from_equity_curve, run_vectorbt_for_symbol


def _empty_signal_matrix(index: pd.DatetimeIndex) -> SignalMatrix:
    n = len(index)
    return SignalMatrix(
        long_entries=pd.Series(np.zeros(n, dtype=bool), index=index),
        long_exits=pd.Series(np.zeros(n, dtype=bool), index=index),
        short_entries=pd.Series(np.zeros(n, dtype=bool), index=index),
        short_exits=pd.Series(np.zeros(n, dtype=bool), index=index),
        signal=pd.Series(np.zeros(n, dtype=np.int32), index=index, dtype="int32"),
        position=pd.Series(np.zeros(n, dtype=np.int32), index=index, dtype="int32"),
        execution_pairs=[],
    )


def _empty_symbol_result(symbol: str, initial_cash: float, bars: Optional[pd.DataFrame] = None) -> SymbolBacktestResult:
    if bars is None:
        bars = pd.DataFrame(columns=["open", "high", "low", "close", "volume"], index=pd.DatetimeIndex([], name="time"))

    index = pd.DatetimeIndex(bars.index)
    if len(index):
        equity_curve = pd.Series(float(initial_cash), index=index, dtype="float64")
        drawdown_curve = pd.Series(0.0, index=index, dtype="float64")
    else:
        equity_curve = pd.Series(dtype="float64", index=index)
        drawdown_curve = pd.Series(dtype="float64", index=index)

    metrics = {
        "total_return_pct": 0.0,
        "annualized_return_pct": 0.0,
        "max_drawdown_pct": 0.0,
        "sharpe": 0.0,
        "sortino": 0.0,
        "calmar": 0.0,
        "win_rate_pct": 0.0,
        "profit_factor": 0.0,
        "total_trades": 0,
        "avg_trade_return_pct": 0.0,
        "exposure_time_pct": 0.0,
        "start_cash": float(initial_cash),
        "end_equity": float(initial_cash),
    }

    return SymbolBacktestResult(
        symbol=symbol,
        metrics=metrics,
        signal_events=[],
        bars=bars,
        signal_matrix=_empty_signal_matrix(index),
        equity_curve=equity_curve,
        drawdown_curve=drawdown_curve,
        closed_trades=[],
    )


def _raw_events_to_scored(raw_events: List[RawBSPEvent]) -> List[ScoredSignalEvent]:
    scored: List[ScoredSignalEvent] = []
    for ev in raw_events:
        signal = 1 if bool(ev.is_buy) else -1
        scored.append(
            ScoredSignalEvent(
                symbol=ev.symbol,
                exec_time=event_signal_time(ev),
                bsp_time=ev.bsp_time,
                is_buy=ev.is_buy,
                bsp_type=ev.bsp_type,
                bsp_types_str=ev.bsp_types_str,
                trade_price=ev.trade_price,
                klu_idx=ev.klu_idx,
                probability=1.0,
                qualified=True,
                signal=signal,
                actual_exec_time=getattr(ev, "actual_exec_time", None),
            )
        )
    return scored


def _rust_feature_config_path(config: BacktestConfig) -> str | None:
    return rust_config_path_from_chan_config(CChanConfig(dict(config.chan_config)))


def _position_sizing_config(config: BacktestConfig) -> PositionSizingConfig | None:
    if not bool(getattr(config, "dynamic_position_sizing_enabled", False)):
        return None
    return PositionSizingConfig(
        enabled=True,
        low_stake_multiplier=float(config.dynamic_low_stake_multiplier),
        medium_stake_multiplier=float(config.dynamic_medium_stake_multiplier),
        high_stake_multiplier=float(config.dynamic_high_stake_multiplier),
        low_leverage=float(config.dynamic_low_leverage),
        medium_leverage=float(config.dynamic_medium_leverage),
        high_leverage=float(config.dynamic_high_leverage),
        max_leverage=float(config.dynamic_max_leverage),
    )


def _run_single_symbol_backtest(config: BacktestConfig, symbol: str) -> SymbolBacktestResult:
    bars: Optional[pd.DataFrame] = None
    try:
        bars = load_symbol_bars(config, symbol)
        active_begin_ts = parse_time_bound(config.begin_time, is_end=False)
        active_end_ts = parse_time_bound(config.end_time, is_end=True)
        use_preloaded_bars = bool(getattr(config, "preload_bars", False)) and int(
            getattr(config, "signal_warmup_bars", 0) or 0
        ) > 0
        if use_preloaded_bars:
            raw_events = extract_raw_bsp_events_from_bars(config, symbol, bars)
            if active_begin_ts is not None or active_end_ts is not None:
                raw_events = [
                    event
                    for event in raw_events
                    if (active_begin_ts is None or event_signal_time(event) >= active_begin_ts)
                    and (active_end_ts is None or event_signal_time(event) <= active_end_ts)
                ]
        else:
            raw_events = extract_raw_bsp_events(config, symbol)
        if config.ml_enabled:
            scored_events = score_raw_events_with_ml(
                bars=bars,
                raw_events=raw_events,
                buy_model_path=config.ml_buy_model_path or None,
                sell_model_path=config.ml_sell_model_path or None,
                both_model_path=config.ml_model_path or None,
                buy_threshold=config.ml_buy_threshold,
                sell_threshold=config.ml_sell_threshold,
                rust_config_path=_rust_feature_config_path(config),
                threshold_policy_path=config.ml_threshold_policy_path or None,
            )
        else:
            scored_events = _raw_events_to_scored(raw_events)

        if active_begin_ts is not None or active_end_ts is not None:
            scored_events = [
                event
                for event in scored_events
                if (active_begin_ts is None or event_signal_time(event) >= active_begin_ts)
                and (active_end_ts is None or event_signal_time(event) <= active_end_ts)
            ]

        if use_preloaded_bars:
            if active_begin_ts is not None:
                bars = bars[bars.index >= active_begin_ts]
            if active_end_ts is not None:
                bars = bars[bars.index <= active_end_ts]

        signal_matrix = build_signal_matrix(
            bars_index=pd.DatetimeIndex(bars.index),
            scored_events=scored_events,
            allow_short=config.allow_short,
            execution_mode=config.execution_mode,
            conflict_policy=config.conflict_policy,
            cooldown_bars=config.cooldown_bars,
            trade_exit_mode=config.trade_exit_mode,
            position_sizing_config=_position_sizing_config(config),
        )
        signal_matrix = apply_risk_layer(config, bars, signal_matrix)
        validate_no_lookahead(signal_matrix, config.execution_mode)

        if config.execution_engine == "crypto":
            _pf, metrics, equity_curve, drawdown_curve, closed_trades = run_crypto_for_symbol(config, bars, signal_matrix)
        else:
            _pf, metrics, equity_curve, drawdown_curve, closed_trades = run_vectorbt_for_symbol(config, bars, signal_matrix)

        for trade in closed_trades:
            trade["symbol"] = symbol

        return SymbolBacktestResult(
            symbol=symbol,
            metrics=metrics,
            signal_events=scored_events,
            bars=bars,
            signal_matrix=signal_matrix,
            equity_curve=equity_curve,
            drawdown_curve=drawdown_curve,
            closed_trades=closed_trades,
        )
    except Exception as ex:
        if bool(getattr(config, "skip_symbol_errors", True)):
            print(f"[BACKTEST][WARN] {symbol} failed and will be skipped: {ex}")
            return _empty_symbol_result(symbol=symbol, initial_cash=config.initial_cash, bars=bars)
        raise


def _aggregate_metrics(config: BacktestConfig, per_symbol: List[SymbolBacktestResult]) -> Dict[str, float]:
    if len(per_symbol) == 1:
        return dict(per_symbol[0].metrics)

    non_empty_curves = [item.equity_curve for item in per_symbol if len(item.equity_curve) > 0]
    if not non_empty_curves:
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
            "start_cash": float(config.initial_cash),
            "end_equity": float(config.initial_cash),
        }

    first_index = non_empty_curves[0].index
    same_index = all(curve.index.equals(first_index) for curve in non_empty_curves[1:])

    if same_index:
        stacked = np.vstack([curve.to_numpy(dtype=np.float64, copy=False) for curve in non_empty_curves])
        agg_equity = pd.Series(np.nanmean(stacked, axis=0), index=first_index)
    else:
        non_empty_items = [item for item in per_symbol if len(item.equity_curve) > 0]
        eq_df = pd.concat([item.equity_curve.rename(item.symbol) for item in non_empty_items], axis=1).sort_index()
        eq_df = eq_df.ffill().dropna(how="all")
        if eq_df.empty:
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
                "start_cash": float(config.initial_cash),
                "end_equity": float(config.initial_cash),
            }
        agg_equity = eq_df.mean(axis=1)

    metrics = metrics_from_equity_curve(agg_equity, config.initial_cash)

    def _metric_mean(metric_name: str) -> float:
        vals = []
        for item in per_symbol:
            value = item.metrics.get(metric_name)
            if isinstance(value, (int, float)) and np.isfinite(float(value)):
                vals.append(float(value))
        return float(np.mean(vals)) if vals else 0.0

    all_trades = [trade for item in per_symbol for trade in item.closed_trades]
    metrics["total_trades"] = float(len(all_trades))
    if all_trades:
        pnl = np.asarray([float(trade.get("pnl", 0.0)) for trade in all_trades], dtype=np.float64)
        returns = np.asarray([float(trade.get("return_pct", 0.0)) for trade in all_trades], dtype=np.float64)
        wins = pnl[pnl > 0]
        losses = pnl[pnl < 0]
        metrics["win_rate_pct"] = float((pnl > 0).mean() * 100.0)
        metrics["profit_factor"] = float(wins.sum() / abs(losses.sum())) if len(losses) else float("inf")
        metrics["avg_trade_return_pct"] = float(np.nanmean(returns))
    else:
        metrics["win_rate_pct"] = 0.0
        metrics["profit_factor"] = 0.0
        metrics["avg_trade_return_pct"] = 0.0
    metrics["exposure_time_pct"] = _metric_mean("exposure_time_pct")
    return metrics


def _sync_strategy_signal_stats(strategy_instance: Optional[ChanStrategyBase], result: BacktestRunResult) -> None:
    if strategy_instance is None:
        return

    events: List[ScoredSignalEvent] = []
    for item in result.per_symbol:
        events.extend(item.signal_events)

    if hasattr(strategy_instance, "n_buy_signals"):
        setattr(strategy_instance, "n_buy_signals", sum(1 for ev in events if ev.is_buy))
    if hasattr(strategy_instance, "n_sell_signals"):
        setattr(strategy_instance, "n_sell_signals", sum(1 for ev in events if not ev.is_buy))
    if hasattr(strategy_instance, "n_buy_traded"):
        setattr(strategy_instance, "n_buy_traded", sum(1 for ev in events if ev.signal == 1))
    if hasattr(strategy_instance, "n_sell_traded"):
        setattr(strategy_instance, "n_sell_traded", sum(1 for ev in events if ev.signal == -1))


def run_vectorbt_backtest(config: BacktestConfig) -> BacktestRunResult:
    config.validate()
    if config.data_src != DATA_SRC.PARQUET:
        raise NotImplementedError("Current implementation only supports DATA_SRC.PARQUET")

    per_symbol: List[SymbolBacktestResult] = []
    symbols = config.normalized_symbols()
    max_workers = min(config.symbol_workers, len(symbols))

    if bool(getattr(config, "preload_bars", False)) and symbols:
        preload_workers = 1 if config.parallel_mode == "process" else max_workers
        preload_symbol_bars(config, symbols, workers=preload_workers)

    if max_workers <= 1:
        for symbol in symbols:
            per_symbol.append(_run_single_symbol_backtest(config, symbol))
    else:
        result_by_symbol: Dict[str, SymbolBacktestResult] = {}
        executor_cls = ProcessPoolExecutor if config.parallel_mode == "process" else ThreadPoolExecutor
        with executor_cls(max_workers=max_workers) as pool:
            future_map = {pool.submit(_run_single_symbol_backtest, config, symbol): symbol for symbol in symbols}
            for future in as_completed(future_map):
                symbol = future_map[future]
                try:
                    result_by_symbol[symbol] = future.result()
                except Exception as ex:
                    if bool(getattr(config, "skip_symbol_errors", True)):
                        print(f"[BACKTEST][WARN] {symbol} failed in pool and will be skipped: {ex}")
                        result_by_symbol[symbol] = _empty_symbol_result(symbol, config.initial_cash)
                    else:
                        raise
        per_symbol = [result_by_symbol[symbol] for symbol in symbols]

    if not per_symbol:
        raise RuntimeError("No symbol backtest result produced.")

    aggregate_metrics = aggregate_symbol_metrics(config, per_symbol)
    artifacts = write_outputs(
        output_dir=config.output_dir,
        aggregate_metrics=aggregate_metrics,
        per_symbol=per_symbol,
        save_events_csv=config.save_events_csv,
        save_bars_csv=config.save_bars_csv,
        save_metrics_json_flag=config.save_metrics_json,
        save_html_report_flag=config.save_html_report,
        save_trades_csv_flag=config.save_trades_csv,
        save_equity_csv_flag=config.save_equity_csv,
        report_params={
            "symbols": symbols,
            "begin_time": config.begin_time,
            "end_time": config.end_time,
            "kl_type": str(config.kl_type.name),
            "execution_engine": config.execution_engine,
            "execution_mode": config.execution_mode,
            "trade_exit_mode": config.trade_exit_mode,
            "fixed_stop_loss_pct": config.fixed_stop_loss_pct,
            "fixed_take_profit_pct": config.fixed_take_profit_pct,
            "allow_short": config.allow_short,
            "initial_cash": config.initial_cash,
            "fee": config.fee,
            "slippage": config.slippage,
            "position_size": config.position_size,
            "ml_enabled": config.ml_enabled,
            "ml_model_path": config.ml_model_path,
            "ml_buy_model_path": config.ml_buy_model_path,
            "ml_sell_model_path": config.ml_sell_model_path,
            "ml_buy_threshold": config.ml_buy_threshold,
            "ml_sell_threshold": config.ml_sell_threshold,
            "ml_threshold_policy_path": config.ml_threshold_policy_path,
            "risk_enabled": config.risk_enabled,
            "risk_base_position_size": config.risk_base_position_size,
            "risk_min_position_size": config.risk_min_position_size,
            "risk_max_position_size": config.risk_max_position_size,
            "risk_use_volatility_target": config.risk_use_volatility_target,
            "risk_target_annual_vol": config.risk_target_annual_vol,
            "risk_vol_window": config.risk_vol_window,
            "risk_max_atr_pct": config.risk_max_atr_pct,
            "risk_symbol_drawdown_stop_pct": config.risk_symbol_drawdown_stop_pct,
            "risk_cooldown_bars_after_drawdown": config.risk_cooldown_bars_after_drawdown,
            "risk_max_holding_bars": config.risk_max_holding_bars,
            "risk_loss_streak_stop": config.risk_loss_streak_stop,
            "risk_cooldown_bars_after_loss_streak": config.risk_cooldown_bars_after_loss_streak,
            "crypto_leverage": config.crypto_leverage,
            "crypto_maintenance_margin_rate": config.crypto_maintenance_margin_rate,
            "crypto_funding_rate_8h": config.crypto_funding_rate_8h,
            "crypto_min_notional": config.crypto_min_notional,
            "crypto_qty_step": config.crypto_qty_step,
            "crypto_min_qty": config.crypto_min_qty,
            "crypto_stop_first": config.crypto_stop_first,
            "output_dir": config.output_dir,
        },
    )

    return BacktestRunResult(aggregate_metrics=aggregate_metrics, per_symbol=per_symbol, artifacts=artifacts)


def run_chan_backtest_no_vnpy(
    strategy_instance: Optional[ChanStrategyBase] = None,
    code: str = "BTCUSDT",
    begin_time: str = "2025-01-01",
    end_time: str = "2026-01-01",
    kl_type: KL_TYPE = KL_TYPE.K_15M,
    initial_cash: float = 100000.0,
    data_src: DATA_SRC = DATA_SRC.PARQUET,
    autype=None,
    chan_config_override: Optional[Dict[str, object]] = None,
    **kwargs,
) -> BacktestRunResult:
    cfg = BacktestConfig(
        symbols=[code],
        begin_time=begin_time,
        end_time=end_time,
        kl_type=kl_type,
        data_src=data_src,
        initial_cash=initial_cash,
    )

    if strategy_instance is not None and hasattr(strategy_instance, "on_chan_init"):
        strategy_instance.on_chan_init()

    if chan_config_override:
        cfg.chan_config.update(chan_config_override)

    if "allow_short" in kwargs:
        cfg.allow_short = bool(kwargs["allow_short"])
    if "execution_mode" in kwargs:
        cfg.execution_mode = str(kwargs["execution_mode"])
    if "cooldown_bars" in kwargs:
        cfg.cooldown_bars = int(kwargs["cooldown_bars"])
    if "output_dir" in kwargs:
        cfg.output_dir = str(kwargs["output_dir"])

    result = run_vectorbt_backtest(cfg)

    if strategy_instance is not None and hasattr(strategy_instance, "set_backtest_result"):
        strategy_instance.set_backtest_result(result)

    _sync_strategy_signal_stats(strategy_instance, result)

    if strategy_instance is not None and hasattr(strategy_instance, "on_backtest_end"):
        strategy_instance.on_backtest_end()

    return result
