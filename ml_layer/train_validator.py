from __future__ import annotations

# flake8: noqa: E501

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .config import ModelConfig, TrainValidatorConfig
from .models import MetaModel, PrimaryModel
from .utils.metrics import annualized_sharpe
from .validation import PurgedKFold, compute_mdi_mda


@dataclass
class FoldMetrics:
    fold: int
    train_size: int
    test_size: int
    signal_count: int
    precision: float
    recall: float
    macro_f1: float
    sharpe: float


class TrainValidator:
    def __init__(self, model_config: ModelConfig, validator_config: TrainValidatorConfig):
        self.model_config = model_config
        self.validator_config = validator_config
        self.primary_model: PrimaryModel | None = None
        self.meta_model: MetaModel | None = None

    @staticmethod
    def _signal_from_class(classes: np.ndarray) -> np.ndarray:
        # Binary class map: 0=SL(-1), 1=PT(+1)
        return np.where(classes == 1, 1, -1).astype(np.int32)

    @staticmethod
    def _build_meta_features(X: np.ndarray, p_primary: np.ndarray) -> np.ndarray:
        # 叠加第一层概率，形成第二层输入
        return np.hstack([np.nan_to_num(X, nan=0.0), p_primary])

    def _fit_one_fold(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        w_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        test_rets: np.ndarray,
        train_mode: str,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, FoldMetrics]:
        primary = PrimaryModel(
            params=self.model_config.xgb_params,
            num_rounds=self.model_config.xgb_num_rounds,
            early_stop=self.model_config.xgb_early_stop,
        ).fit(X_train, y_train, w_train, X_test, y_test, train_mode=train_mode)

        p_train = primary.predict_proba(X_train)
        c_train = np.argmax(p_train, axis=1)

        meta = MetaModel()
        meta_y = (c_train == y_train).astype(int)
        meta_X = self._build_meta_features(X_train, p_train)
        meta.fit(meta_X, meta_y)

        p_test = primary.predict_proba(X_test)
        c_test = np.argmax(p_test, axis=1)
        keep_mask = np.zeros(len(y_test), dtype=bool)
        exec_prob = np.zeros(len(y_test), dtype=np.float32)

        if len(y_test) > 0:
            meta_test_X = self._build_meta_features(X_test, p_test)
            p_exec = meta.predict_proba(meta_test_X)
            keep_mask = p_exec >= self.model_config.meta_threshold
            exec_prob = p_exec.astype(np.float32)

        precision = float(np.mean((c_test[keep_mask] == y_test[keep_mask]).astype(float))) if keep_mask.any() else 0.0
        recall = float(np.sum((c_test[keep_mask] == y_test[keep_mask]).astype(float)) / max(1, len(y_test)))
        macro_f1 = float(f1_score(y_test, c_test, average="macro", zero_division=0))
        sharpe = annualized_sharpe(test_rets[keep_mask]) if keep_mask.any() else 0.0

        metrics = FoldMetrics(
            fold=0,
            train_size=int(len(y_train)),
            test_size=int(len(y_test)),
            signal_count=int(keep_mask.sum()),
            precision=precision,
            recall=recall,
            macro_f1=macro_f1,
            sharpe=sharpe,
        )
        return c_test, exec_prob, keep_mask, metrics

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray,
        t0_pos: np.ndarray,
        t1_pos: np.ndarray,
        returns: np.ndarray,
        feature_names: List[str],
        train_mode: str = "auto",
    ) -> Tuple[Dict[str, object], pd.DataFrame, pd.DataFrame, List[str], np.ndarray, np.ndarray]:
        splitter = PurgedKFold(
            n_splits=self.validator_config.n_splits,
            embargo_bars=self.validator_config.embargo_bars,
        )

        oos_cls = np.zeros(len(y), dtype=np.int32)
        oos_exec = np.zeros(len(y), dtype=np.float32)
        oos_keep = np.zeros(len(y), dtype=bool)
        folds: List[FoldMetrics] = []

        for fold_id, (train_idx, test_idx) in enumerate(splitter.split(t0_pos=t0_pos, t1_pos=t1_pos), start=1):
            if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
                continue

            cls, exec_prob, keep_mask, fold_metric = self._fit_one_fold(
                X_train=X[train_idx],
                y_train=y[train_idx],
                w_train=sample_weight[train_idx],
                X_test=X[test_idx],
                y_test=y[test_idx],
                test_rets=returns[test_idx],
                train_mode=train_mode,
            )
            fold_metric.fold = fold_id
            folds.append(fold_metric)
            oos_cls[test_idx] = cls
            oos_exec[test_idx] = exec_prob
            oos_keep[test_idx] = keep_mask

        if not folds:
            raise RuntimeError("PurgedKFold 未生成有效折，请扩大样本或降低切分数")

        # 全量模型
        self.primary_model = PrimaryModel(
            params=self.model_config.xgb_params,
            num_rounds=self.model_config.xgb_num_rounds,
            early_stop=self.model_config.xgb_early_stop,
        ).fit(X, y, sample_weight, X, y, train_mode=train_mode)

        p_all = self.primary_model.predict_proba(X)
        c_all = np.argmax(p_all, axis=1)

        self.meta_model = MetaModel()
        meta_y = (c_all == y).astype(int)
        meta_X = self._build_meta_features(X, p_all)
        self.meta_model.fit(meta_X, meta_y)

        mdi_df, mda_df = compute_mdi_mda(
            self.primary_model.booster,
            feature_names,
            X,
            y,
            mda_max_samples=max(1000, int(self.validator_config.mda_max_samples)),
            mda_n_jobs=max(1, int(self.validator_config.mda_n_jobs)),
        )
        mdi_keep = set(mdi_df.head(max(1, len(mdi_df) // 2))["feature"].tolist())
        mda_keep = set(mda_df[mda_df["mda_drop"] > 0]["feature"].tolist())
        selected = sorted(mdi_keep.intersection(mda_keep))

        oos_signal = self._signal_from_class(oos_cls)
        oos_signal[~oos_keep] = 0
        oos_mask = oos_keep
        oos_precision = float(np.mean((oos_cls[oos_mask] == y[oos_mask]).astype(float))) if oos_mask.any() else 0.0
        oos_recall = float(np.sum((oos_cls[oos_mask] == y[oos_mask]).astype(float)) / max(1, len(y)))
        oos_macro_f1 = float(f1_score(y, oos_cls, average="macro", zero_division=0))
        oos_sharpe = annualized_sharpe(returns[oos_mask]) if oos_mask.any() else 0.0

        fold_df = pd.DataFrame([asdict(f) for f in folds])
        report = {
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "engine": "TrainValidator",
            "architecture": {
                "primary_model": self.model_config.primary_model_type,
                "meta_model": self.model_config.meta_model_type,
                "meta_threshold": self.model_config.meta_threshold,
                "cv": {
                    "splits": self.validator_config.n_splits,
                    "embargo_bars": self.validator_config.embargo_bars,
                    "purged": True,
                },
                "feature_importance": {
                    "mda_max_samples": int(self.validator_config.mda_max_samples),
                    "mda_n_jobs": int(self.validator_config.mda_n_jobs),
                },
            },
            "oos_metrics": {
                "sharpe": oos_sharpe,
                "precision": oos_precision,
                "recall": oos_recall,
                "macro_f1": oos_macro_f1,
                "signal_count": int(oos_mask.sum()),
            },
            "fold_metrics": fold_df.to_dict(orient="records"),
            "fold_summary": {
                "sharpe_mean": float(fold_df["sharpe"].mean()),
                "sharpe_std": float(fold_df["sharpe"].std(ddof=0)),
                "precision_mean": float(fold_df["precision"].mean()),
                "macro_f1_mean": float(fold_df["macro_f1"].mean()),
            },
            "pass_criteria": {
                "passed": bool(oos_precision >= self.validator_config.pass_precision and oos_sharpe > self.validator_config.pass_sharpe),
                "rule": f"oos_precision>={self.validator_config.pass_precision} and oos_sharpe>{self.validator_config.pass_sharpe}",
            },
        }

        return report, mdi_df, mda_df, selected, oos_cls, oos_exec
