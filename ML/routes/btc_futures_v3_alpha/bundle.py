from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .dataset import enhance_btc_futures_v3_features, required_base_columns_for_v3


@dataclass
class PoolModelSpec:
    estimator: object
    feature_columns: list[str]
    fill_values: dict[str, float]
    threshold: float
    enabled: bool = True
    validation_metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class V3AlphaModelBundle:
    side: str
    global_spec: PoolModelSpec
    pool_specs: dict[str, PoolModelSpec] = field(default_factory=dict)
    route: str = "btc_futures_v3_alpha"
    dependency_columns: list[str] = field(default_factory=required_base_columns_for_v3)

    @property
    def threshold(self) -> float:
        return float(self.global_spec.threshold)

    @threshold.setter
    def threshold(self, value: float) -> None:
        self.global_spec.threshold = float(value)

    @property
    def feature_columns(self) -> list[str]:
        cols: set[str] = set(self.dependency_columns)
        cols.update(self.global_spec.feature_columns)
        for spec in self.pool_specs.values():
            cols.update(spec.feature_columns)
        return sorted(cols)

    def prepare_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        return enhance_btc_futures_v3_features(frame)

    def _transform(self, frame: pd.DataFrame, spec: PoolModelSpec) -> pd.DataFrame:
        x = frame.reindex(columns=spec.feature_columns)
        x = x.replace([np.inf, -np.inf], np.nan)
        return x.fillna(spec.fill_values).fillna(0.0)

    def _predict_spec(self, frame: pd.DataFrame, spec: PoolModelSpec) -> np.ndarray:
        if frame.empty:
            return np.empty(0, dtype="float64")
        x = self._transform(frame, spec)
        estimator = spec.estimator
        if hasattr(estimator, "predict_proba"):
            return np.asarray(estimator.predict_proba(x)[:, 1], dtype="float64")
        score = np.asarray(estimator.decision_function(x), dtype="float64")
        return 1.0 / (1.0 + np.exp(-score))

    def _pool_series(self, frame: pd.DataFrame) -> pd.Series:
        if "v3_structure_pool" not in frame.columns:
            prepared = self.prepare_features(frame)
            return prepared["v3_structure_pool"].astype(str)
        return frame["v3_structure_pool"].astype(str)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        prepared = self.prepare_features(frame)
        probability = self._predict_spec(prepared, self.global_spec)
        pools = prepared["v3_structure_pool"].astype(str)
        for pool, spec in self.pool_specs.items():
            if not bool(spec.enabled):
                continue
            mask = pools == str(pool)
            if bool(mask.any()):
                probability[mask.to_numpy()] = self._predict_spec(prepared.loc[mask], spec)
        return probability

    def predict_model_key(self, frame: pd.DataFrame) -> np.ndarray:
        prepared = self.prepare_features(frame)
        keys = np.full(len(prepared), "global", dtype=object)
        pools = prepared["v3_structure_pool"].astype(str)
        for pool, spec in self.pool_specs.items():
            if not bool(spec.enabled):
                continue
            keys[(pools == str(pool)).to_numpy()] = f"pool:{pool}"
        return keys

    def threshold_for_rows(self, frame: pd.DataFrame) -> np.ndarray:
        prepared = self.prepare_features(frame)
        thresholds = np.full(len(prepared), float(self.global_spec.threshold), dtype="float64")
        pools = prepared["v3_structure_pool"].astype(str)
        for pool, spec in self.pool_specs.items():
            if not bool(spec.enabled):
                continue
            thresholds[(pools == str(pool)).to_numpy()] = float(spec.threshold)
        return thresholds

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str | Path) -> "V3AlphaModelBundle":
        with Path(path).open("rb") as fh:
            return pickle.load(fh)

