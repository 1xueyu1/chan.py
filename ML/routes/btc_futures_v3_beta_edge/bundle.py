from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .dataset import enhance_btc_futures_v3_beta_features, required_base_columns_for_beta


@dataclass
class BetaModelSpec:
    estimator: object
    feature_columns: list[str]
    fill_values: dict[str, float]
    target_column: str
    validation_metrics: dict[str, Any] = field(default_factory=dict)


@dataclass
class StableRankClassifier:
    """Small no-tree ranker built from temporally stable signed features."""

    feature_stats: list[dict[str, float | str]]
    clip_z: float = 6.0

    @staticmethod
    def _sigmoid(values: np.ndarray) -> np.ndarray:
        clipped = np.clip(values, -6.0, 6.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray:
        n = len(x)
        if n == 0:
            return np.empty((0, 2), dtype="float64")
        if not self.feature_stats:
            prob = np.full(n, 0.5, dtype="float64")
            return np.column_stack([1.0 - prob, prob])

        scores: list[np.ndarray] = []
        weights: list[float] = []
        for stat in self.feature_stats:
            feature = str(stat["feature"])
            sign = float(stat.get("sign", 1.0))
            mean = float(stat.get("mean", 0.0))
            std = max(float(stat.get("std", 1.0)), 1e-9)
            fill_value = float(stat.get("fill_value", 0.0))
            weight = max(float(stat.get("weight", 1.0)), 1e-6)
            if feature in x.columns:
                raw = pd.to_numeric(x[feature], errors="coerce").fillna(fill_value).to_numpy(dtype="float64")
            else:
                raw = np.full(n, fill_value, dtype="float64")
            z = (raw * sign - mean) / std
            scores.append(self._sigmoid(np.clip(z, -float(self.clip_z), float(self.clip_z))))
            weights.append(weight)

        score_matrix = np.vstack(scores).T
        weight_array = np.asarray(weights, dtype="float64")
        weight_array = weight_array / max(float(weight_array.sum()), 1e-12)
        prob = np.clip(score_matrix @ weight_array, 1e-6, 1.0 - 1e-6)
        return np.column_stack([1.0 - prob, prob])


@dataclass
class V3BetaEdgeModelBundle:
    side: str
    primary_spec: BetaModelSpec
    threshold: float = 0.99
    route: str = "btc_futures_v3_beta_edge"
    dependency_columns: list[str] = field(default_factory=required_base_columns_for_beta)

    @property
    def feature_columns(self) -> list[str]:
        cols: set[str] = set(self.dependency_columns)
        cols.update(self.primary_spec.feature_columns)
        return sorted(cols)

    def prepare_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        return enhance_btc_futures_v3_beta_features(frame)

    def _transform(self, frame: pd.DataFrame, spec: BetaModelSpec) -> pd.DataFrame:
        x = frame.reindex(columns=spec.feature_columns)
        x = x.replace([np.inf, -np.inf], np.nan)
        return x.fillna(spec.fill_values).fillna(0.0)

    def _predict_spec(self, frame: pd.DataFrame, spec: BetaModelSpec) -> np.ndarray:
        if frame.empty:
            return np.empty(0, dtype="float64")
        x = self._transform(frame, spec)
        estimator = spec.estimator
        if hasattr(estimator, "predict_proba"):
            return np.asarray(estimator.predict_proba(x)[:, 1], dtype="float64")
        score = np.asarray(estimator.decision_function(x), dtype="float64")
        return 1.0 / (1.0 + np.exp(-score))

    def predict_components(self, frame: pd.DataFrame) -> pd.DataFrame:
        prepared = self.prepare_features(frame)
        out = pd.DataFrame(index=prepared.index)
        out["tp_first_probability"] = self._predict_spec(prepared, self.primary_spec)
        out["probability"] = out["tp_first_probability"]
        return out

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return self.predict_components(frame)["probability"].to_numpy(dtype="float64")

    def threshold_for_rows(self, frame: pd.DataFrame) -> np.ndarray:
        return np.full(len(frame), float(self.threshold), dtype="float64")

    def predict_model_key(self, frame: pd.DataFrame) -> np.ndarray:
        return np.full(len(frame), "beta_tp_first", dtype=object)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str | Path) -> "V3BetaEdgeModelBundle":
        with Path(path).open("rb") as fh:
            return pickle.load(fh)


def normalize_bsp_family(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text.startswith("1"):
        return "1"
    if text.startswith("2"):
        return "2"
    if text.startswith("3"):
        return "3"
    return "unknown"


def bsp_family_for_frame(frame: pd.DataFrame) -> pd.Series:
    if "bsp_type" in frame.columns:
        return frame["bsp_type"].map(normalize_bsp_family).astype("object")
    if "bsp_types_str" in frame.columns:
        return frame["bsp_types_str"].map(normalize_bsp_family).astype("object")
    out = pd.Series("unknown", index=frame.index, dtype="object")
    for family in ("1", "2", "3"):
        col = f"v3_bsp_has_{family}"
        if col in frame.columns:
            out.loc[pd.to_numeric(frame[col], errors="coerce").fillna(0.0) > 0.0] = family
    return out


@dataclass
class SideBspSplitBetaEdgeModelBundle:
    side: str
    family_bundles: dict[str, V3BetaEdgeModelBundle]
    disabled_threshold: float = 0.99
    route: str = "btc_futures_v3_beta_edge_bsp_split"
    dependency_columns: list[str] = field(default_factory=required_base_columns_for_beta)

    @property
    def threshold(self) -> float:
        thresholds = [float(bundle.threshold) for bundle in self.family_bundles.values()]
        return float(min(thresholds)) if thresholds else float(self.disabled_threshold)

    @property
    def feature_columns(self) -> list[str]:
        cols: set[str] = set(self.dependency_columns)
        cols.update({"bsp_type", "bsp_types_str"})
        for bundle in self.family_bundles.values():
            cols.update(bundle.feature_columns)
        return sorted(cols)

    def prepare_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        return enhance_btc_futures_v3_beta_features(frame)

    def predict_components(self, frame: pd.DataFrame) -> pd.DataFrame:
        prepared = self.prepare_features(frame)
        out = pd.DataFrame(
            {
                "tp_first_probability": np.zeros(len(prepared), dtype="float64"),
                "probability": np.zeros(len(prepared), dtype="float64"),
            },
            index=prepared.index,
        )
        if prepared.empty:
            return out
        families = bsp_family_for_frame(prepared)
        for family, bundle in self.family_bundles.items():
            mask = families == str(family)
            if not bool(mask.any()):
                continue
            comp = bundle.predict_components(prepared.loc[mask])
            for col in out.columns:
                out.loc[mask, col] = comp[col].to_numpy(dtype="float64")
        return out

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return self.predict_components(frame)["probability"].to_numpy(dtype="float64")

    def threshold_for_rows(self, frame: pd.DataFrame) -> np.ndarray:
        prepared = self.prepare_features(frame)
        out = np.full(len(prepared), float(self.disabled_threshold), dtype="float64")
        families = bsp_family_for_frame(prepared)
        for family, bundle in self.family_bundles.items():
            out[(families == str(family)).to_numpy()] = float(bundle.threshold)
        return out

    def predict_qualified(self, frame: pd.DataFrame, default_threshold: float = 0.99) -> np.ndarray:
        prob = self.predict_proba(frame)
        thresholds = self.threshold_for_rows(frame)
        fallback = np.full(len(frame), float(default_threshold), dtype="float64")
        thresholds = np.where(np.isfinite(thresholds), thresholds, fallback)
        return prob >= thresholds

    def predict_model_key(self, frame: pd.DataFrame) -> np.ndarray:
        families = bsp_family_for_frame(frame)
        return np.asarray([f"beta_bsp_{family}_{self.side}" for family in families], dtype=object)

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str | Path) -> "SideBspSplitBetaEdgeModelBundle":
        with Path(path).open("rb") as fh:
            return pickle.load(fh)
