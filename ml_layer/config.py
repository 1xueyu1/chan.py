from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class LabelConfig:
    pt_multiplier: float = 2.0
    dynamic_pt_enabled: bool = False
    pt_low_vol_multiplier: float = 1.8
    pt_mid_vol_multiplier: float = 2.0
    pt_high_vol_multiplier: float = 2.2
    vol_window: int = 96
    vol_quantile_low: float = 0.33
    vol_quantile_high: float = 0.67
    timeout_bars: int = 20
    weak_timeout_bars: int = 10
    weak_bsp_types: str = "3"
    min_ret_threshold: float = 0.003


@dataclass
class FeatureConfig:
    min_history_bars: int = 50
    normalize_method: str = "expanding"
    vol_regime_window: int = 252
    symbol_workers: int = max(1, min(8, (os.cpu_count() or 2) // 2))


@dataclass
class ModelConfig:
    primary_model_type: str = "xgboost"
    meta_model_type: str = "logistic"
    meta_threshold: float = 0.55
    meta_calibration: str = "none"
    meta_calibration_ratio: float = 0.2
    xgb_num_rounds: int = 280
    xgb_early_stop: int = 30
    xgb_params: Dict[str, object] = field(
        default_factory=lambda: {
            "max_depth": 4,
            "eta": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "reg_alpha": 0.2,
            "reg_lambda": 2.0,
            "seed": 42,
        }
    )


@dataclass
class TrainValidatorConfig:
    n_splits: int = 5
    embargo_bars: int = 10
    optuna_trials: int = 0
    pass_precision: float = 0.55
    pass_sharpe: float = 0.0
    pass_positive_month_ratio: float = 0.70
    mda_max_samples: int = 6000
    mda_n_jobs: int = max(1, min(8, (os.cpu_count() or 2) // 2))


@dataclass
class OptimizationObjectiveConfig:
    version: str = "v1_backtest_composite"
    ret_weight: float = 0.45
    mdd_weight: float = 0.25
    sharpe_weight: float = 0.20
    trade_weight: float = 0.10
    mdd_worsen_tolerance_pct: float = 1.0
    sharpe_drop_tolerance: float = 0.05
    trade_min_ratio: float = 0.60
    trade_max_ratio: float = 1.40
    trade_abs_min: float = 30.0
    trade_abs_max: float = 5000.0
