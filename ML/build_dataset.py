from __future__ import annotations

import argparse

from Backtest.config import BacktestConfig, KL_TYPE_TEXT_TO_ENUM
from .dataset import build_dataset
from .label import LabelConfig


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build Chan ML dataset from parquet bars")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--begin-time", default="2022-01-01")
    parser.add_argument("--end-time", default="2026-05-05")
    parser.add_argument("--kl-type", default="15m", choices=list(KL_TYPE_TEXT_TO_ENUM.keys()))
    parser.add_argument("--output", default="data/btc_futures_v1/generic_dataset.parquet")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--parallel-mode", default="process", choices=["process", "thread"])
    parser.add_argument("--horizon-bars", type=int, default=96)
    parser.add_argument("--take-profit-atr", type=float, default=1.5)
    parser.add_argument("--stop-loss-atr", type=float, default=1.0)
    parser.add_argument("--min-return", type=float, default=0.0)
    parser.add_argument("--use-rust-core", action="store_true", help="use Rust chan-core acceleration for event extraction")
    parser.add_argument("--rust-core-config-path", default=None, help="optional chan-core Rust config file")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = BacktestConfig(
        symbols=args.symbols or ["BTCUSDT"],
        begin_time=args.begin_time,
        end_time=args.end_time,
        kl_type=KL_TYPE_TEXT_TO_ENUM[args.kl_type],
        symbol_workers=args.workers or 1,
        parallel_mode=args.parallel_mode,
    )
    cfg.chan_config["use_rust_core"] = bool(args.use_rust_core)
    if args.rust_core_config_path:
        cfg.chan_config["rust_core_config_path"] = str(args.rust_core_config_path)
    label_cfg = LabelConfig(
        horizon_bars=args.horizon_bars,
        take_profit_atr=args.take_profit_atr,
        stop_loss_atr=args.stop_loss_atr,
        min_return=args.min_return,
        execution_mode=cfg.execution_mode,
        fee=cfg.fee,
        slippage=cfg.slippage,
    )
    symbols = args.symbols or cfg.normalized_symbols()
    build_dataset(
        config=cfg,
        symbols=symbols,
        label_config=label_cfg,
        output_path=args.output,
        workers=args.workers,
        parallel_mode=args.parallel_mode,
    )


if __name__ == "__main__":
    main()
