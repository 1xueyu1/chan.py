from __future__ import annotations

# flake8: noqa: E501

import argparse
import os
from dataclasses import dataclass, field
from typing import Dict, List

from Common.CEnum import DATA_SRC, KL_TYPE


KL_TYPE_TEXT_TO_ENUM = {
    "1m": KL_TYPE.K_1M,
    "3m": KL_TYPE.K_3M,
    "5m": KL_TYPE.K_5M,
    "10m": KL_TYPE.K_10M,
    "15m": KL_TYPE.K_15M,
    "30m": KL_TYPE.K_30M,
    "1h": KL_TYPE.K_60M,
    "1d": KL_TYPE.K_DAY,
    "1w": KL_TYPE.K_WEEK,
    "1mo": KL_TYPE.K_MON,
}


DEFAULT_CHAN_CONFIG: Dict[str, object] = {
    "trigger_step": True,
    "bi_strict": True,
    "skip_step": 0,
    "divergence_rate": float("inf"),
    "bsp2_follow_1": False,
    "bsp3_follow_1": False,
    "min_zs_cnt": 0,
    "bs1_peak": False,
    "macd_algo": "peak",
    "bs_type": "1,2,3a,1p,2s,3b",
    "print_warning": False,
    "zs_algo": "normal",
}

_CPU_COUNT = os.cpu_count() or 4
_DEFAULT_SYMBOL_WORKERS = max(1, min(8, _CPU_COUNT // 2))


@dataclass
class BacktestConfig:
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT"])
    begin_time: str = "2025-01-01"
    end_time: str = "2026-01-01"
    kl_type: KL_TYPE = KL_TYPE.K_15M
    data_src: DATA_SRC = DATA_SRC.PARQUET

    initial_cash: float = 100000.0
    fee: float = 0.0004
    slippage: float = 0.0001

    signal_threshold: float = 0.55
    allow_short: bool = False
    execution_mode: str = "next_bar_open"
    conflict_policy: str = "exit_first"
    symbol_workers: int = field(default_factory=lambda: _DEFAULT_SYMBOL_WORKERS)
    parallel_mode: str = "process"
    fast_mode: bool = False

    model_buy_path: str = "Debug/model_buy.json"
    model_sell_path: str = "Debug/model_sell.json"
    meta_buy_path: str = "Debug/meta_buy.json"
    meta_sell_path: str = "Debug/meta_sell.json"

    event_replay_mode: bool = False
    event_replay_csv_path: str = "result/model_signal_events.csv"
    replay_reapply_threshold: bool = False

    output_dir: str = "result"
    save_events_csv: bool = True
    save_bars_csv: bool = True
    save_metrics_json: bool = True
    save_html_report: bool = True
    save_html_detail_report: bool = False

    random_seed: int = 42
    chan_config: Dict[str, object] = field(default_factory=lambda: dict(DEFAULT_CHAN_CONFIG))

    def normalized_symbols(self) -> List[str]:
        uniq = []
        seen = set()
        for symbol in self.symbols:
            s = symbol.upper()
            if not s.endswith("USDT"):
                s = f"{s}USDT"
            if s not in seen:
                uniq.append(s)
                seen.add(s)
        return uniq

    def validate(self) -> None:
        if self.execution_mode not in {"next_bar_open", "close"}:
            raise ValueError("execution_mode must be next_bar_open or close")
        if self.conflict_policy not in {"exit_first"}:
            raise ValueError("unsupported conflict_policy")
        if not (0 <= self.signal_threshold <= 1):
            raise ValueError("signal_threshold must be in [0, 1]")
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be > 0")
        if self.kl_type not in KL_TYPE_TEXT_TO_ENUM.values():
            raise ValueError(f"unsupported kl_type: {self.kl_type}")
        if self.event_replay_mode and not self.event_replay_csv_path:
            raise ValueError("event_replay_csv_path is required in event_replay_mode")
        if self.symbol_workers < 1:
            raise ValueError("symbol_workers must be >= 1")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run vectorbt backtest for chan signals")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT"], help="symbols, e.g. BTCUSDT ETHUSDT")
    parser.add_argument("--begin-time", default="2025-01-01")
    parser.add_argument("--end-time", default="2026-01-01")
    parser.add_argument("--kl-type", default="15m", choices=list(KL_TYPE_TEXT_TO_ENUM.keys()))

    parser.add_argument("--initial-cash", type=float, default=100000.0)
    parser.add_argument("--fee", type=float, default=0.0004)
    parser.add_argument("--slippage", type=float, default=0.0001)
    parser.add_argument("--signal-threshold", type=float, default=0.55)

    parser.add_argument("--allow-short", action="store_true")
    parser.add_argument("--execution-mode", default="next_bar_open", choices=["next_bar_open", "close"])
    parser.add_argument(
        "--symbol-workers",
        type=int,
        default=_DEFAULT_SYMBOL_WORKERS,
        help="parallel workers for symbol-level backtest execution (auto by CPU cores)",
    )
    parser.add_argument(
        "--parallel-mode",
        choices=["process", "thread"],
        default="process",
        help="symbol并行模式，process通常更快（CPU密集）",
    )
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="快速回测模式：关闭大体积输出与HTML渲染，优先速度",
    )

    parser.add_argument("--model-buy-path", default="Debug/model_buy.json")
    parser.add_argument("--model-sell-path", default="Debug/model_sell.json")
    parser.add_argument("--meta-buy-path", default="Debug/meta_buy.json")
    parser.add_argument("--meta-sell-path", default="Debug/meta_sell.json")

    parser.add_argument(
        "--event-replay",
        action="store_true",
        help="reuse existing scored events CSV and skip chan/model inference",
    )
    parser.add_argument(
        "--event-replay-csv",
        default="result/model_signal_events.csv",
        help="path to reused scored events CSV",
    )
    parser.add_argument(
        "--replay-reapply-threshold",
        action="store_true",
        help="reapply current --signal-threshold to CSV probability in replay mode",
    )

    parser.add_argument("--output-dir", default="result")
    parser.add_argument("--no-events-csv", action="store_true")
    parser.add_argument("--no-bars-csv", action="store_true")
    parser.add_argument("--no-metrics-json", action="store_true")
    parser.add_argument("--no-html-report", action="store_true")
    parser.add_argument(
        "--save-html-detail-report",
        action="store_true",
        help="save detail html report with full charts/trade table (summary report is still saved by default)",
    )

    return parser


def config_from_args(args: argparse.Namespace) -> BacktestConfig:
    cfg = BacktestConfig(
        symbols=args.symbols,
        begin_time=args.begin_time,
        end_time=args.end_time,
        kl_type=KL_TYPE_TEXT_TO_ENUM[args.kl_type],
        initial_cash=args.initial_cash,
        fee=args.fee,
        slippage=args.slippage,
        signal_threshold=args.signal_threshold,
        allow_short=bool(args.allow_short),
        execution_mode=args.execution_mode,
        symbol_workers=args.symbol_workers,
        parallel_mode=args.parallel_mode,
        fast_mode=bool(args.fast_mode),
        model_buy_path=args.model_buy_path,
        model_sell_path=args.model_sell_path,
        meta_buy_path=args.meta_buy_path,
        meta_sell_path=args.meta_sell_path,
        event_replay_mode=bool(args.event_replay),
        event_replay_csv_path=args.event_replay_csv,
        replay_reapply_threshold=bool(args.replay_reapply_threshold),
        output_dir=args.output_dir,
        save_events_csv=(not args.no_events_csv) and (not args.fast_mode),
        save_bars_csv=(not args.no_bars_csv) and (not args.fast_mode),
        save_metrics_json=not args.no_metrics_json,
        save_html_report=(not args.no_html_report) and (not args.fast_mode),
        save_html_detail_report=bool(args.save_html_detail_report) and (not args.no_html_report),
    )
    if cfg.fast_mode:
        cfg.save_html_detail_report = False
    cfg.validate()
    return cfg
