from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict


@dataclass
class LabelConfig:
    pt_multiplier: float = 2.0
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
    xgb_num_rounds: int = 280
    xgb_early_stop: int = 30
    xgb_params: Dict[str, object] = field(
        default_factory=lambda: {
            "max_depth": 4,
            "eta": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "objective": "multi:softprob",
            "eval_metric": "mlogloss",
            "num_class": 3,
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
    mda_max_samples: int = 6000
    mda_n_jobs: int = max(1, min(8, (os.cpu_count() or 2) // 2))
