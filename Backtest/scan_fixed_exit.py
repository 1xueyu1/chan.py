from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import BacktestConfig, KL_TYPE_TEXT_TO_ENUM
from .engine import run_vectorbt_backtest


DEFAULT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "XRPUSDT", "BNBUSDT", "SOLUSDT",
    "TRXUSDT", "DOGEUSDT", "HYPEUSDT", "ADAUSDT", "BCHUSDT",
    "XMRUSDT", "ZECUSDT", "LINKUSDT", "CCUSDT", "XLMUSDT",
    "LTCUSDT", "AVAXUSDT", "HBARUSDT", "TONUSDT", "SUIUSDT",
]


def _parse_pair(text: str) -> tuple[float, float]:
    left, right = text.replace("/", ":").split(":", 1)
    return float(left), float(right)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Scan fixed TP/SL Chan ML backtests")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--begin-time", default="2025-01-01")
    parser.add_argument("--end-time", default="2026-05-05")
    parser.add_argument("--kl-type", default="15m", choices=list(KL_TYPE_TEXT_TO_ENUM.keys()))
    parser.add_argument("--pairs", nargs="+", default=["0.02:0.05", "0.03:0.06", "0.04:0.08", "0.05:0.10"])
    parser.add_argument("--max-drawdown-pct", type=float, default=10.0)
    parser.add_argument("--output-dir", default="result/fixed_exit_scan")
    parser.add_argument("--summary-output", default="result/fixed_exit_scan/summary.csv")

    parser.add_argument("--buy-model-path", required=True)
    parser.add_argument("--sell-model-path", required=True)
    parser.add_argument("--buy-threshold", type=float, default=0.74)
    parser.add_argument("--sell-threshold", type=float, default=0.74)
    parser.add_argument("--threshold-policy-path", default="", help="optional capability-v2 threshold policy json")

    parser.add_argument("--risk-base-position-size", type=float, default=0.85)
    parser.add_argument("--risk-min-position-size", type=float, default=0.05)
    parser.add_argument("--risk-max-position-size", type=float, default=1.0)
    parser.add_argument("--risk-target-annual-vol", type=float, default=1.2)
    parser.add_argument("--risk-symbol-drawdown-stop-pct", type=float, default=0.25)
    parser.add_argument("--risk-cooldown-bars-after-drawdown", type=int, default=672)
    parser.add_argument("--risk-loss-streak-stop", type=int, default=0)
    parser.add_argument("--risk-cooldown-bars-after-loss-streak", type=int, default=0)
    parser.add_argument(
        "--risk-max-holding-bars",
        type=int,
        default=96,
        help="maximum bars to hold a fixed TP/SL trade; use 0 for unlimited",
    )
    parser.add_argument("--dynamic-position-sizing", action="store_true")
    parser.add_argument("--dynamic-low-stake-multiplier", type=float, default=0.50)
    parser.add_argument("--dynamic-medium-stake-multiplier", type=float, default=1.00)
    parser.add_argument("--dynamic-high-stake-multiplier", type=float, default=1.50)
    parser.add_argument("--dynamic-low-leverage", type=float, default=1.00)
    parser.add_argument("--dynamic-medium-leverage", type=float, default=2.00)
    parser.add_argument("--dynamic-high-leverage", type=float, default=3.00)
    parser.add_argument("--dynamic-max-leverage", type=float, default=3.00)
    parser.add_argument("--symbol-workers", type=int, default=4)
    parser.add_argument("--preload-bars", action="store_true")
    parser.add_argument("--signal-warmup-bars", type=int, default=1500)
    parser.add_argument("--use-rust-core", action="store_true", help="use Rust chan-core acceleration for signal extraction")
    parser.add_argument("--rust-core-config-path", default=None, help="optional chan-core Rust config file")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []

    for stop_loss, take_profit in [_parse_pair(pair) for pair in args.pairs]:
        run_name = f"sl_{stop_loss:.3f}_tp_{take_profit:.3f}".replace(".", "p")
        cfg = BacktestConfig(
            symbols=list(args.symbols),
            begin_time=args.begin_time,
            end_time=args.end_time,
            kl_type=KL_TYPE_TEXT_TO_ENUM[args.kl_type],
            allow_short=True,
            execution_mode="next_bar_open",
            trade_exit_mode="fixed_tp_sl",
            fixed_stop_loss_pct=stop_loss,
            fixed_take_profit_pct=take_profit,
            ml_enabled=True,
            ml_buy_model_path=args.buy_model_path,
            ml_sell_model_path=args.sell_model_path,
            ml_buy_threshold=args.buy_threshold,
            ml_sell_threshold=args.sell_threshold,
            ml_threshold_policy_path=str(args.threshold_policy_path),
            risk_enabled=True,
            risk_base_position_size=args.risk_base_position_size,
            risk_min_position_size=args.risk_min_position_size,
            risk_max_position_size=args.risk_max_position_size,
            risk_target_annual_vol=args.risk_target_annual_vol,
            risk_symbol_drawdown_stop_pct=args.risk_symbol_drawdown_stop_pct,
            risk_cooldown_bars_after_drawdown=args.risk_cooldown_bars_after_drawdown,
            risk_loss_streak_stop=int(args.risk_loss_streak_stop),
            risk_cooldown_bars_after_loss_streak=int(args.risk_cooldown_bars_after_loss_streak),
            risk_max_holding_bars=int(args.risk_max_holding_bars),
            dynamic_position_sizing_enabled=bool(args.dynamic_position_sizing),
            dynamic_low_stake_multiplier=float(args.dynamic_low_stake_multiplier),
            dynamic_medium_stake_multiplier=float(args.dynamic_medium_stake_multiplier),
            dynamic_high_stake_multiplier=float(args.dynamic_high_stake_multiplier),
            dynamic_low_leverage=float(args.dynamic_low_leverage),
            dynamic_medium_leverage=float(args.dynamic_medium_leverage),
            dynamic_high_leverage=float(args.dynamic_high_leverage),
            dynamic_max_leverage=float(args.dynamic_max_leverage),
            crypto_leverage=float(args.dynamic_max_leverage if args.dynamic_position_sizing else 1.0),
            symbol_workers=args.symbol_workers,
            preload_bars=bool(args.preload_bars),
            signal_warmup_bars=int(args.signal_warmup_bars),
            parallel_mode="process",
            output_dir=str(out_dir / run_name),
            save_bars_csv=False,
        )
        cfg.chan_config["use_rust_core"] = bool(args.use_rust_core)
        if args.rust_core_config_path:
            cfg.chan_config["rust_core_config_path"] = str(args.rust_core_config_path)
        result = run_vectorbt_backtest(cfg)
        row = {
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "output_dir": str(out_dir / run_name),
            **result.aggregate_metrics,
        }
        row["within_drawdown_limit"] = abs(float(row["max_drawdown_pct"])) <= float(args.max_drawdown_pct)
        rows.append(row)
        pd.DataFrame(rows).to_csv(args.summary_output, index=False)
        print(row)

    summary = pd.DataFrame(rows).sort_values(["within_drawdown_limit", "total_return_pct"], ascending=[False, False])
    Path(args.summary_output).parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary_output, index=False)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
