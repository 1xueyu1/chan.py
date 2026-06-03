from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .features import infer_feature_columns
from .registry import load_manifest, save_manifest
from .schema import ModelManifest


@dataclass
class ModelBundle:
    estimator: object
    feature_columns: List[str]
    fill_values: Dict[str, float]
    threshold: float = 0.5
    side: str = "both"
    model_kind: str = "unknown"
    manifest: ModelManifest | None = None

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        x = frame.reindex(columns=self.feature_columns)
        x = x.replace([np.inf, -np.inf], np.nan)
        return x.fillna(self.fill_values).fillna(0.0)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        x = self.transform(frame)
        if hasattr(self.estimator, "predict_proba"):
            return np.asarray(self.estimator.predict_proba(x)[:, 1], dtype="float64")
        score = np.asarray(self.estimator.decision_function(x), dtype="float64")
        return 1.0 / (1.0 + np.exp(-score))

    def score_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out["ml_probability"] = self.predict_proba(out)
        out["ml_qualified"] = out["ml_probability"] >= float(self.threshold)
        return out

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if getattr(self, "manifest", None) is None:
            self.manifest = ModelManifest(
                side=self.side,
                model_kind=self.model_kind,
                threshold=float(self.threshold),
                feature_columns=list(self.feature_columns),
            )
        with path.open("wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)
        save_manifest(path, self.manifest)

    @staticmethod
    def load(path: str | Path) -> "ModelBundle":
        path = Path(path)
        with path.open("rb") as fh:
            bundle = pickle.load(fh)
        if getattr(bundle, "manifest", None) is None:
            bundle.manifest = load_manifest(path)
        return bundle


def _make_estimator(kind: str = "auto", random_state: int = 42):
    if kind in {"chan_xgb", "xgboost_regularized", "regularized_xgboost"}:
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=260,
            max_depth=3,
            learning_rate=0.025,
            min_child_weight=8,
            gamma=0.05,
            subsample=0.75,
            colsample_bytree=0.70,
            reg_alpha=0.10,
            reg_lambda=3.00,
            eval_metric="logloss",
            tree_method="hist",
            random_state=random_state,
            n_jobs=-1,
        )

    if kind in {"chan_hgb", "hist_regularized", "regularized_hist"}:
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            max_iter=220,
            learning_rate=0.025,
            max_leaf_nodes=15,
            min_samples_leaf=80,
            l2_regularization=1.0,
            random_state=random_state,
        )

    if kind in {"lightgbm_regularized", "regularized_lightgbm", "chan_lgbm"}:
        from lightgbm import LGBMClassifier

        return LGBMClassifier(
            n_estimators=260,
            learning_rate=0.025,
            num_leaves=15,
            max_depth=4,
            min_child_samples=80,
            min_split_gain=0.01,
            subsample=0.75,
            colsample_bytree=0.70,
            reg_alpha=0.20,
            reg_lambda=2.00,
            objective="binary",
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        )

    if kind in {"auto", "lightgbm"}:
        try:
            from lightgbm import LGBMClassifier

            return LGBMClassifier(
                n_estimators=500,
                learning_rate=0.03,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="binary",
                random_state=random_state,
                n_jobs=-1,
                verbose=-1,
            )
        except Exception:
            if kind == "lightgbm":
                raise

    if kind in {"auto", "xgboost"}:
        try:
            from xgboost import XGBClassifier

            return XGBClassifier(
                n_estimators=500,
                max_depth=5,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                eval_metric="logloss",
                tree_method="hist",
                random_state=random_state,
                n_jobs=-1,
            )
        except Exception:
            if kind == "xgboost":
                raise

    from sklearn.ensemble import HistGradientBoostingClassifier

    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.04,
        max_leaf_nodes=31,
        l2_regularization=0.01,
        random_state=random_state,
    )


def train_classifier(
    train_frame: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
    model_kind: str = "auto",
    threshold: float = 0.5,
    side: str = "both",
    random_state: int = 42,
) -> ModelBundle:
    frame = train_frame.dropna(subset=["label"]).copy()
    if side == "buy":
        frame = frame[frame["is_buy"].astype(bool)]
    elif side == "sell":
        frame = frame[~frame["is_buy"].astype(bool)]
    elif side != "both":
        raise ValueError("side must be buy, sell, or both")

    if frame.empty:
        raise ValueError("empty training frame")

    cols = feature_columns or infer_feature_columns(frame)
    if not cols:
        raise ValueError("no feature columns found")

    x = frame[cols].replace([np.inf, -np.inf], np.nan)
    fill_values = x.median(numeric_only=True).fillna(0.0).to_dict()
    x = x.fillna(fill_values).fillna(0.0)
    y = frame["label"].astype(int)
    sample_weight = None
    if "sample_weight" in frame.columns:
        sample_weight = pd.to_numeric(frame["sample_weight"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        sample_weight = sample_weight.fillna(1.0).clip(lower=0.05).to_numpy(dtype="float64")

    estimator = _make_estimator(model_kind, random_state=random_state)
    if sample_weight is not None:
        estimator.fit(x, y, sample_weight=sample_weight)
    else:
        estimator.fit(x, y)
    return ModelBundle(
        estimator=estimator,
        feature_columns=list(cols),
        fill_values={str(k): float(v) for k, v in fill_values.items()},
        threshold=float(threshold),
        side=side,
        model_kind=model_kind,
        manifest=ModelManifest(
            side=side,
            model_kind=model_kind,
            threshold=float(threshold),
            feature_columns=list(cols),
        ),
    )
