from __future__ import annotations

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
    "use_rust_core": False,
    "rust_core_config_path": None,
    "mtf_chan_features": False,
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
    execution_engine: str = "crypto"

    allow_short: bool = False
    execution_mode: str = "next_bar_open"
    conflict_policy: str = "exit_first"
    trade_exit_mode: str = "opposite_signal"
    fixed_stop_loss_pct: float = 0.04
    fixed_take_profit_pct: float = 0.08
    cooldown_bars: int = 0
    position_size: float = 1.0

    ml_enabled: bool = False
    ml_model_path: str = ""
    ml_buy_model_path: str = ""
    ml_sell_model_path: str = ""
    ml_buy_threshold: float = 0.5
    ml_sell_threshold: float = 0.5
    ml_threshold_policy_path: str = ""

    risk_enabled: bool = False
    risk_base_position_size: float = 0.25
    risk_min_position_size: float = 0.02
    risk_max_position_size: float = 0.35
    risk_use_volatility_target: bool = True
    risk_target_annual_vol: float = 0.35
    risk_vol_window: int = 96
    risk_max_atr_pct: float = 0.12
    risk_atr_period: int = 14
    risk_symbol_drawdown_stop_pct: float = 0.25
    risk_cooldown_bars_after_drawdown: int = 672
    risk_max_holding_bars: int = 0
    risk_loss_streak_stop: int = 0
    risk_cooldown_bars_after_loss_streak: int = 0

    dynamic_position_sizing_enabled: bool = False
    dynamic_low_stake_multiplier: float = 0.50
    dynamic_medium_stake_multiplier: float = 1.00
    dynamic_high_stake_multiplier: float = 1.50
    dynamic_low_leverage: float = 1.00
    dynamic_medium_leverage: float = 2.00
    dynamic_high_leverage: float = 3.00
    dynamic_max_leverage: float = 3.00

    crypto_leverage: float = 1.0
    crypto_maintenance_margin_rate: float = 0.005
    crypto_funding_rate_8h: float = 0.0
    crypto_min_notional: float = 5.0
    crypto_qty_step: float = 0.0
    crypto_min_qty: float = 0.0
    crypto_stop_first: bool = True

    symbol_workers: int = field(default_factory=lambda: _DEFAULT_SYMBOL_WORKERS)
    parallel_mode: str = "process"
    preload_bars: bool = False
    signal_warmup_bars: int = 0
    data_cache_size: int = 8
    skip_symbol_errors: bool = True

    output_dir: str = "result"
    save_events_csv: bool = True
    save_bars_csv: bool = True
    save_metrics_json: bool = True
    save_html_report: bool = True
    save_trades_csv: bool = True
    save_equity_csv: bool = True

    chan_config: Dict[str, object] = field(default_factory=lambda: dict(DEFAULT_CHAN_CONFIG))

    def normalized_symbols(self) -> List[str]:
        uniq: List[str] = []
        seen = set()
        for symbol in self.symbols:
            s = symbol.upper().replace("/", "")
            if not s.endswith("USDT"):
                s = f"{s}USDT"
            if s not in seen:
                seen.add(s)
                uniq.append(s)
        return uniq

    def validate(self) -> None:
        if self.execution_engine not in {"crypto", "vectorbt"}:
            raise ValueError("execution_engine must be crypto or vectorbt")
        if self.execution_mode not in {"next_bar_open", "close"}:
            raise ValueError("execution_mode must be next_bar_open or close")
        if self.conflict_policy != "exit_first":
            raise ValueError("unsupported conflict_policy")
        if self.trade_exit_mode not in {"opposite_signal", "fixed_tp_sl"}:
            raise ValueError("trade_exit_mode must be opposite_signal or fixed_tp_sl")
        if self.fixed_stop_loss_pct <= 0:
            raise ValueError("fixed_stop_loss_pct must be > 0")
        if self.fixed_take_profit_pct <= 0:
            raise ValueError("fixed_take_profit_pct must be > 0")
        if self.cooldown_bars < 0:
            raise ValueError("cooldown_bars must be >= 0")
        if self.position_size <= 0 or self.position_size > 1:
            raise ValueError("position_size must be in (0, 1]")
        if self.ml_buy_threshold < 0 or self.ml_buy_threshold > 1:
            raise ValueError("ml_buy_threshold must be between 0 and 1")
        if self.ml_sell_threshold < 0 or self.ml_sell_threshold > 1:
            raise ValueError("ml_sell_threshold must be between 0 and 1")
        if self.ml_enabled and not (self.ml_model_path or self.ml_buy_model_path or self.ml_sell_model_path):
            raise ValueError("ml_enabled requires at least one model path")
        if self.risk_base_position_size <= 0 or self.risk_base_position_size > 1:
            raise ValueError("risk_base_position_size must be in (0, 1]")
        if self.risk_min_position_size < 0 or self.risk_min_position_size > 1:
            raise ValueError("risk_min_position_size must be in [0, 1]")
        if self.risk_max_position_size <= 0 or self.risk_max_position_size > 1:
            raise ValueError("risk_max_position_size must be in (0, 1]")
        if self.risk_min_position_size > self.risk_max_position_size:
            raise ValueError("risk_min_position_size must be <= risk_max_position_size")
        if self.risk_target_annual_vol <= 0:
            raise ValueError("risk_target_annual_vol must be > 0")
        if self.risk_vol_window < 2:
            raise ValueError("risk_vol_window must be >= 2")
        if self.risk_atr_period < 2:
            raise ValueError("risk_atr_period must be >= 2")
        if self.risk_max_atr_pct < 0:
            raise ValueError("risk_max_atr_pct must be >= 0")
        if self.risk_symbol_drawdown_stop_pct < 0 or self.risk_symbol_drawdown_stop_pct >= 1:
            raise ValueError("risk_symbol_drawdown_stop_pct must be in [0, 1)")
        if self.risk_cooldown_bars_after_drawdown < 0:
            raise ValueError("risk_cooldown_bars_after_drawdown must be >= 0")
        if self.risk_max_holding_bars < 0:
            raise ValueError("risk_max_holding_bars must be >= 0")
        if self.risk_loss_streak_stop < 0:
            raise ValueError("risk_loss_streak_stop must be >= 0")
        if self.risk_cooldown_bars_after_loss_streak < 0:
            raise ValueError("risk_cooldown_bars_after_loss_streak must be >= 0")
        for name in (
            "dynamic_low_stake_multiplier",
            "dynamic_medium_stake_multiplier",
            "dynamic_high_stake_multiplier",
        ):
            if float(getattr(self, name)) < 0:
                raise ValueError(f"{name} must be >= 0")
        for name in (
            "dynamic_low_leverage",
            "dynamic_medium_leverage",
            "dynamic_high_leverage",
            "dynamic_max_leverage",
        ):
            if float(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be > 0")
        if self.crypto_leverage <= 0:
            raise ValueError("crypto_leverage must be > 0")
        if self.crypto_maintenance_margin_rate < 0 or self.crypto_maintenance_margin_rate >= 1:
            raise ValueError("crypto_maintenance_margin_rate must be in [0, 1)")
        if self.crypto_min_notional < 0:
            raise ValueError("crypto_min_notional must be >= 0")
        if self.crypto_qty_step < 0:
            raise ValueError("crypto_qty_step must be >= 0")
        if self.crypto_min_qty < 0:
            raise ValueError("crypto_min_qty must be >= 0")
        if self.initial_cash <= 0:
            raise ValueError("initial_cash must be > 0")
        if self.symbol_workers < 1:
            raise ValueError("symbol_workers must be >= 1")
        if self.signal_warmup_bars < 0:
            raise ValueError("signal_warmup_bars must be >= 0")
        if self.parallel_mode not in {"process", "thread"}:
            raise ValueError("parallel_mode must be process or thread")
        if self.data_cache_size < 0:
            raise ValueError("data_cache_size must be >= 0")
        if self.kl_type not in KL_TYPE_TEXT_TO_ENUM.values():
            raise ValueError(f"unsupported kl_type: {self.kl_type}")

    def to_settings(self):
        from .settings import (
            BacktestSettings,
            ChanRuntimeConfig,
            DataConfig,
            ExecutionConfig,
            MLConfig,
            OutputConfig,
            RiskConfig,
            RuntimeConfig,
            UniverseConfig,
        )

        return BacktestSettings(
            universe=UniverseConfig(
                symbols=list(self.symbols),
                begin_time=self.begin_time,
                end_time=self.end_time,
                kl_type=self.kl_type,
                data_src=self.data_src,
            ),
            data=DataConfig(
                cache_size=self.data_cache_size,
                preload_bars=self.preload_bars,
                signal_warmup_bars=self.signal_warmup_bars,
            ),
            chan=ChanRuntimeConfig(
                chan_config=dict(self.chan_config),
                use_rust_core=bool(self.chan_config.get("use_rust_core", False)),
                rust_core_config_path=self.chan_config.get("rust_core_config_path"),
                mtf_chan_features=bool(
                    self.chan_config.get("mtf_chan_features", False)
                ),
            ),
            ml=MLConfig(
                enabled=self.ml_enabled,
                model_path=self.ml_model_path,
                buy_model_path=self.ml_buy_model_path,
                sell_model_path=self.ml_sell_model_path,
                buy_threshold=self.ml_buy_threshold,
                sell_threshold=self.ml_sell_threshold,
                threshold_policy_path=self.ml_threshold_policy_path,
            ),
            risk=RiskConfig(
                enabled=self.risk_enabled,
                base_position_size=self.risk_base_position_size,
                min_position_size=self.risk_min_position_size,
                max_position_size=self.risk_max_position_size,
                use_volatility_target=self.risk_use_volatility_target,
                target_annual_vol=self.risk_target_annual_vol,
                vol_window=self.risk_vol_window,
                max_atr_pct=self.risk_max_atr_pct,
                atr_period=self.risk_atr_period,
                symbol_drawdown_stop_pct=self.risk_symbol_drawdown_stop_pct,
                cooldown_bars_after_drawdown=(
                    self.risk_cooldown_bars_after_drawdown
                ),
                max_holding_bars=self.risk_max_holding_bars,
                loss_streak_stop=self.risk_loss_streak_stop,
                cooldown_bars_after_loss_streak=(
                    self.risk_cooldown_bars_after_loss_streak
                ),
            ),
            execution=ExecutionConfig(
                engine=self.execution_engine,
                initial_cash=self.initial_cash,
                fee=self.fee,
                slippage=self.slippage,
                allow_short=self.allow_short,
                execution_mode=self.execution_mode,
                conflict_policy=self.conflict_policy,
                trade_exit_mode=self.trade_exit_mode,
                fixed_stop_loss_pct=self.fixed_stop_loss_pct,
                fixed_take_profit_pct=self.fixed_take_profit_pct,
                cooldown_bars=self.cooldown_bars,
                position_size=self.position_size,
                crypto_leverage=self.crypto_leverage,
                crypto_maintenance_margin_rate=(
                    self.crypto_maintenance_margin_rate
                ),
                crypto_funding_rate_8h=self.crypto_funding_rate_8h,
                crypto_min_notional=self.crypto_min_notional,
                crypto_qty_step=self.crypto_qty_step,
                crypto_min_qty=self.crypto_min_qty,
                crypto_stop_first=self.crypto_stop_first,
            ),
            output=OutputConfig(
                output_dir=self.output_dir,
                save_events_csv=self.save_events_csv,
                save_bars_csv=self.save_bars_csv,
                save_metrics_json=self.save_metrics_json,
                save_html_report=self.save_html_report,
                save_trades_csv=self.save_trades_csv,
                save_equity_csv=self.save_equity_csv,
            ),
            runtime=RuntimeConfig(
                symbol_workers=self.symbol_workers,
                parallel_mode=self.parallel_mode,
                skip_symbol_errors=self.skip_symbol_errors,
            ),
        )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run rule-based chan vectorbt backtest")
    parser.add_argument("--symbols", nargs="+", default=["BTCUSDT"], help="symbols, e.g. BTCUSDT ETHUSDT")
    parser.add_argument("--begin-time", default="2025-01-01")
    parser.add_argument("--end-time", default="2026-01-01")
    parser.add_argument("--kl-type", default="15m", choices=list(KL_TYPE_TEXT_TO_ENUM.keys()))

    parser.add_argument("--allow-short", action="store_true")
    parser.add_argument("--trade-exit-mode", default="opposite_signal", choices=["opposite_signal", "fixed_tp_sl"])
    parser.add_argument("--fixed-stop-loss-pct", type=float, default=0.04)
    parser.add_argument("--fixed-take-profit-pct", type=float, default=0.08)

    parser.add_argument("--ml-enabled", action="store_true")
    parser.add_argument("--ml-buy-model-path", default="", help="model used for buy events")
    parser.add_argument("--ml-sell-model-path", default="", help="model used for sell events")
    parser.add_argument("--ml-buy-threshold", type=float, default=0.5)
    parser.add_argument("--ml-sell-threshold", type=float, default=0.5)
    parser.add_argument("--ml-threshold-policy-path", default="", help="optional capability-v2 dynamic threshold policy json")

    parser.add_argument("--risk-enabled", action="store_true")
    parser.add_argument("--risk-base-position-size", type=float, default=0.25)
    parser.add_argument("--risk-max-position-size", type=float, default=0.35)
    parser.add_argument("--risk-target-annual-vol", type=float, default=0.35)

    parser.add_argument("--symbol-workers", type=int, default=_DEFAULT_SYMBOL_WORKERS)
    parser.add_argument("--preload-bars", action="store_true")
    parser.add_argument("--signal-warmup-bars", type=int, default=0)
    parser.add_argument("--output-dir", default="result")
    parser.add_argument("--no-bars-csv", action="store_true")
    parser.add_argument("--use-rust-core", action="store_true", help="use Rust chan-core acceleration for signal extraction")
    parser.add_argument("--rust-core-config-path", default=None, help="optional chan-core Rust config file")
    return parser


def config_from_args(args: argparse.Namespace) -> BacktestConfig:
    cfg = BacktestConfig(
        symbols=list(args.symbols),
        begin_time=args.begin_time,
        end_time=args.end_time,
        kl_type=KL_TYPE_TEXT_TO_ENUM[args.kl_type],
        allow_short=bool(args.allow_short),
        trade_exit_mode=str(args.trade_exit_mode),
        fixed_stop_loss_pct=float(args.fixed_stop_loss_pct),
        fixed_take_profit_pct=float(args.fixed_take_profit_pct),
        ml_enabled=bool(args.ml_enabled),
        ml_buy_model_path=str(args.ml_buy_model_path),
        ml_sell_model_path=str(args.ml_sell_model_path),
        ml_buy_threshold=float(args.ml_buy_threshold),
        ml_sell_threshold=float(args.ml_sell_threshold),
        ml_threshold_policy_path=str(args.ml_threshold_policy_path),
        risk_enabled=bool(args.risk_enabled),
        risk_base_position_size=float(args.risk_base_position_size),
        risk_max_position_size=float(args.risk_max_position_size),
        risk_target_annual_vol=float(args.risk_target_annual_vol),
        symbol_workers=int(args.symbol_workers),
        preload_bars=bool(args.preload_bars),
        signal_warmup_bars=int(args.signal_warmup_bars),
        output_dir=str(args.output_dir),
        save_bars_csv=(not bool(args.no_bars_csv)),
    )
    cfg.chan_config["use_rust_core"] = bool(args.use_rust_core)
    if args.rust_core_config_path:
        cfg.chan_config["rust_core_config_path"] = str(args.rust_core_config_path)
    cfg.validate()
    return cfg
