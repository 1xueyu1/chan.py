from __future__ import annotations

# flake8: noqa: E501

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from Common.CEnum import DATA_SRC, KL_TYPE

from .chan_signal_extractor import extract_raw_bsp_events
from .config import BacktestConfig
from .data_loader import load_symbol_bars, preload_symbol_bars
from .event_cache import load_scored_events_cache, save_scored_events_cache
from .event_replay import load_scored_events_by_symbol
from .feature_adapter import enrich_raw_events_with_feature_engine
from .model_gate import get_dual_model_gate
from .reporter import write_outputs
from .signal_builder import build_signal_matrix, validate_no_lookahead
from .strategy import ChanStrategyBase
from .types import BacktestRunResult, SignalMatrix, SymbolBacktestResult
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
        bars = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            index=pd.DatetimeIndex([], name="time"),
        )

    index = pd.DatetimeIndex(bars.index)
    equity_curve = pd.Series(float(initial_cash), index=index, dtype="float64") if len(index) else pd.Series(dtype="float64", index=index)
    drawdown_curve = pd.Series(0.0, index=index, dtype="float64") if len(index) else pd.Series(dtype="float64", index=index)

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


def _run_single_symbol_backtest(
    config: BacktestConfig,
    symbol: str,
    replay_events_for_symbol: Optional[list] = None,
) -> SymbolBacktestResult:
    bars: Optional[pd.DataFrame] = None
    try:
        bars = load_symbol_bars(config, symbol)
        if config.event_replay_mode:
            scored_events = list(replay_events_for_symbol or [])
        else:
            cached_events = load_scored_events_cache(config, symbol)
            if cached_events is not None:
                scored_events = cached_events
            else:
                gate = get_dual_model_gate(config)
                raw_events = extract_raw_bsp_events(config, symbol)
                try:
                    raw_events = enrich_raw_events_with_feature_engine(
                        symbol=symbol,
                        bars=bars,
                        raw_events=raw_events,
                    )
                except Exception as ex:
                    print(
                        f"[WARN] feature adapter failed for {symbol}: {ex}. "
                        "Fallback to raw feature_map."
                    )
                scored_events = gate.score_events(raw_events)
                save_scored_events_cache(config, symbol, scored_events)

        signal_matrix = build_signal_matrix(
            bars_index=pd.DatetimeIndex(bars.index),
            scored_events=scored_events,
            allow_short=config.allow_short,
            execution_mode=config.execution_mode,
            conflict_policy=config.conflict_policy,
            cooldown_bars=config.cooldown_bars,
        )
        validate_no_lookahead(signal_matrix, config.execution_mode)

        _pf, metrics, equity_curve, drawdown_curve, closed_trades = run_vectorbt_for_symbol(
            config,
            bars,
            signal_matrix,
        )

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
        agg_values = np.nanmean(stacked, axis=0)
        agg_equity = pd.Series(agg_values, index=first_index)
    else:
        non_empty_items = [item for item in per_symbol if len(item.equity_curve) > 0]
        eq_df = pd.concat(
            [item.equity_curve.rename(item.symbol) for item in non_empty_items],
            axis=1,
        ).sort_index()
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

    metrics["win_rate_pct"] = _metric_mean("win_rate_pct")
    metrics["profit_factor"] = _metric_mean("profit_factor")
    metrics["total_trades"] = float(sum(len(item.closed_trades) for item in per_symbol))
    metrics["avg_trade_return_pct"] = _metric_mean("avg_trade_return_pct")
    metrics["exposure_time_pct"] = _metric_mean("exposure_time_pct")
    return metrics


def _sync_strategy_signal_stats(
    strategy_instance: Optional[ChanStrategyBase],
    result: BacktestRunResult,
) -> None:
    if strategy_instance is None:
        return

    events = []
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


def run_vectorbt_backtest(
    config: BacktestConfig,
) -> BacktestRunResult:
    config.validate()
    if config.data_src != DATA_SRC.PARQUET:
        raise NotImplementedError("Current implementation only supports DATA_SRC.PARQUET")

    replay_events_by_symbol = None
    if config.event_replay_mode:
        replay_events_by_symbol = load_scored_events_by_symbol(config)

    per_symbol: List[SymbolBacktestResult] = []
    symbols = config.normalized_symbols()
    max_workers = min(config.symbol_workers, len(symbols))

    if bool(getattr(config, "preload_bars", False)) and symbols:
        preload_workers = 1 if config.parallel_mode == "process" else max_workers
        preload_symbol_bars(config, symbols, workers=preload_workers)
        print(
            f"[BACKTEST] preloaded bars for {len(symbols)} symbols "
            f"(parallel_mode={config.parallel_mode}, preload_workers={preload_workers})"
        )

    if max_workers == 1:
        for symbol in symbols:
            per_symbol.append(
                _run_single_symbol_backtest(
                    config,
                    symbol,
                    replay_events_for_symbol=(replay_events_by_symbol or {}).get(symbol),
                )
            )
    else:
        result_by_symbol: Dict[str, SymbolBacktestResult] = {}
        executor_cls = (
            ProcessPoolExecutor
            if config.parallel_mode == "process"
            else ThreadPoolExecutor
        )
        with executor_cls(max_workers=max_workers) as pool:
            future_map = {
                pool.submit(
                    _run_single_symbol_backtest,
                    config,
                    symbol,
                    (replay_events_by_symbol or {}).get(symbol),
                ): symbol
                for symbol in symbols
            }
            for future in as_completed(future_map):
                symbol = future_map[future]
                try:
                    result_by_symbol[symbol] = future.result()
                except Exception as ex:
                    if bool(getattr(config, "skip_symbol_errors", True)):
                        print(f"[BACKTEST][WARN] {symbol} failed in pool and will be skipped: {ex}")
                        result_by_symbol[symbol] = _empty_symbol_result(
                            symbol=symbol,
                            initial_cash=config.initial_cash,
                            bars=None,
                        )
                    else:
                        raise

        per_symbol = [result_by_symbol[symbol] for symbol in symbols]

    if not per_symbol:
        raise RuntimeError("No symbol backtest result produced.")

    aggregate_metrics = _aggregate_metrics(config, per_symbol)
    artifacts = write_outputs(
        output_dir=config.output_dir,
        aggregate_metrics=aggregate_metrics,
        per_symbol=per_symbol,
        save_events_csv=config.save_events_csv,
        save_bars_csv=config.save_bars_csv,
        save_metrics_json_flag=config.save_metrics_json,
        save_html_report_flag=config.save_html_report,
        save_html_detail_report_flag=config.save_html_detail_report,
        save_trades_csv_flag=bool(getattr(config, "save_trades_csv", True)),
        save_equity_csv_flag=bool(getattr(config, "save_equity_csv", True)),
        report_params={
            "symbols": symbols,
            "begin_time": config.begin_time,
            "end_time": config.end_time,
            "kl_type": str(config.kl_type.name),
            "execution_mode": config.execution_mode,
            "event_source": "复用事件CSV" if config.event_replay_mode else "实时提取+模型打分",
            "event_replay_csv_path": config.event_replay_csv_path if config.event_replay_mode else "--",
            "replay_reapply_threshold": config.replay_reapply_threshold if config.event_replay_mode else False,
            "allow_short": config.allow_short,
            "signal_threshold": config.signal_threshold,
            "initial_cash": config.initial_cash,
            "fee": config.fee,
            "slippage": config.slippage,
            "conflict_policy": config.conflict_policy,
            "output_dir": config.output_dir,
            "model_buy_path": config.model_buy_path,
            "model_sell_path": config.model_sell_path,
            "meta_buy_path": config.meta_buy_path,
            "meta_sell_path": config.meta_sell_path,
            "meta_model_path": config.meta_model_path or "auto_detect",
        },
    )

    return BacktestRunResult(
        aggregate_metrics=aggregate_metrics,
        per_symbol=per_symbol,
        artifacts=artifacts,
    )


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

    if strategy_instance is not None:
        if hasattr(strategy_instance, "signal_threshold"):
            cfg.signal_threshold = float(getattr(strategy_instance, "signal_threshold"))
        if hasattr(strategy_instance, "on_chan_init"):
            strategy_instance.on_chan_init()

    if "signal_threshold" in kwargs:
        cfg.signal_threshold = float(kwargs["signal_threshold"])

    if chan_config_override:
        cfg.chan_config.update(chan_config_override)

    if "allow_short" in kwargs:
        cfg.allow_short = bool(kwargs["allow_short"])
    if "model_buy_path" in kwargs:
        cfg.model_buy_path = str(kwargs["model_buy_path"])
    if "model_sell_path" in kwargs:
        cfg.model_sell_path = str(kwargs["model_sell_path"])
    if "meta_buy_path" in kwargs:
        cfg.meta_buy_path = str(kwargs["meta_buy_path"])
    if "meta_sell_path" in kwargs:
        cfg.meta_sell_path = str(kwargs["meta_sell_path"])
    if "output_dir" in kwargs:
        cfg.output_dir = str(kwargs["output_dir"])
    if "event_replay_mode" in kwargs:
        cfg.event_replay_mode = bool(kwargs["event_replay_mode"])
    if "event_replay_csv_path" in kwargs:
        cfg.event_replay_csv_path = str(kwargs["event_replay_csv_path"])
    if "replay_reapply_threshold" in kwargs:
        cfg.replay_reapply_threshold = bool(kwargs["replay_reapply_threshold"])
    if "save_html_detail_report" in kwargs:
        cfg.save_html_detail_report = bool(kwargs["save_html_detail_report"])

    result = run_vectorbt_backtest(cfg)

    if strategy_instance is not None and hasattr(strategy_instance, "set_backtest_result"):
        strategy_instance.set_backtest_result(result)

    _sync_strategy_signal_stats(strategy_instance, result)

    if strategy_instance is not None and hasattr(strategy_instance, "on_backtest_end"):
        strategy_instance.on_backtest_end()

    return result
