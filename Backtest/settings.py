from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from Common.CEnum import DATA_SRC, KL_TYPE


@dataclass
class UniverseConfig:
    symbols: List[str] = field(default_factory=lambda: ["BTCUSDT"])
    begin_time: str = "2025-01-01"
    end_time: str = "2026-01-01"
    kl_type: KL_TYPE = KL_TYPE.K_15M
    data_src: DATA_SRC = DATA_SRC.PARQUET


@dataclass
class DataConfig:
    data_root: str = "data"
    cache_size: int = 8
    preload_bars: bool = False
    signal_warmup_bars: int = 0


@dataclass
class ChanRuntimeConfig:
    chan_config: Dict[str, object] = field(default_factory=dict)
    use_rust_core: bool = False
    rust_core_config_path: str | None = None
    mtf_chan_features: bool = False


@dataclass
class MLConfig:
    enabled: bool = False
    model_path: str = ""
    buy_model_path: str = ""
    sell_model_path: str = ""
    buy_threshold: float = 0.5
    sell_threshold: float = 0.5
    threshold_policy_path: str = ""


@dataclass
class RiskConfig:
    enabled: bool = False
    base_position_size: float = 0.25
    min_position_size: float = 0.02
    max_position_size: float = 0.35
    use_volatility_target: bool = True
    target_annual_vol: float = 0.35
    vol_window: int = 96
    max_atr_pct: float = 0.12
    atr_period: int = 14
    symbol_drawdown_stop_pct: float = 0.25
    cooldown_bars_after_drawdown: int = 672
    max_holding_bars: int = 0
    loss_streak_stop: int = 0
    cooldown_bars_after_loss_streak: int = 0


@dataclass
class ExecutionConfig:
    engine: str = "crypto"
    initial_cash: float = 100000.0
    fee: float = 0.0004
    slippage: float = 0.0001
    allow_short: bool = False
    execution_mode: str = "next_bar_open"
    conflict_policy: str = "exit_first"
    trade_exit_mode: str = "opposite_signal"
    fixed_stop_loss_pct: float = 0.04
    fixed_take_profit_pct: float = 0.08
    cooldown_bars: int = 0
    position_size: float = 1.0
    crypto_leverage: float = 1.0
    crypto_maintenance_margin_rate: float = 0.005
    crypto_funding_rate_8h: float = 0.0
    crypto_min_notional: float = 5.0
    crypto_qty_step: float = 0.0
    crypto_min_qty: float = 0.0
    crypto_stop_first: bool = True


@dataclass
class OutputConfig:
    output_dir: str = "result"
    save_events_csv: bool = True
    save_bars_csv: bool = True
    save_metrics_json: bool = True
    save_html_report: bool = True
    save_trades_csv: bool = True
    save_equity_csv: bool = True


@dataclass
class RuntimeConfig:
    symbol_workers: int = 1
    parallel_mode: str = "process"
    skip_symbol_errors: bool = True


@dataclass
class BacktestSettings:
    universe: UniverseConfig = field(default_factory=UniverseConfig)
    data: DataConfig = field(default_factory=DataConfig)
    chan: ChanRuntimeConfig = field(default_factory=ChanRuntimeConfig)
    ml: MLConfig = field(default_factory=MLConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)
    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
