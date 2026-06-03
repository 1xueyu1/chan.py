from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ML.shared.threshold_policy import market_state_from_row


@dataclass
class StableGateModelBundle:
    estimator: object
    primary_feature_columns: list[str]
    primary_fill_values: dict[str, float]
    threshold: float
    side: str
    gate_estimator: object | None = None
    gate_feature_columns: list[str] | None = None
    gate_fill_values: dict[str, float] | None = None
    gate_threshold: float = 1.0
    quality_threshold: float = 0.0
    market_state_categories: list[str] | None = None

    @property
    def feature_columns(self) -> list[str]:
        cols = list(self.primary_feature_columns)
        for col in self.gate_feature_columns or []:
            if col.startswith("gate_") or col in {"primary_probability", "primary_margin"}:
                continue
            if col not in cols:
                cols.append(col)
        return cols

    def _transform_primary(self, frame: pd.DataFrame) -> pd.DataFrame:
        x = frame.reindex(columns=self.primary_feature_columns)
        x = x.replace([np.inf, -np.inf], np.nan)
        return x.fillna(self.primary_fill_values).fillna(0.0)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        x = self._transform_primary(frame)
        if hasattr(self.estimator, "predict_proba"):
            return np.asarray(self.estimator.predict_proba(x)[:, 1], dtype="float64")
        score = np.asarray(self.estimator.decision_function(x), dtype="float64")
        return 1.0 / (1.0 + np.exp(-score))

    def _gate_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=frame.index)
        primary_probability = self.predict_proba(frame)
        out["primary_probability"] = primary_probability
        out["primary_margin"] = primary_probability - float(self.threshold)
        for col in self.gate_feature_columns or []:
            if col in out.columns:
                continue
            if col.startswith("gate_state_"):
                state_name = col[len("gate_state_") :]
                states = frame.apply(market_state_from_row, axis=1)
                out[col] = (states == state_name).astype("float64")
            elif col in frame.columns:
                out[col] = pd.to_numeric(frame[col], errors="coerce")
            else:
                out[col] = np.nan
        x = out.reindex(columns=self.gate_feature_columns or [])
        x = x.replace([np.inf, -np.inf], np.nan)
        return x.fillna(self.gate_fill_values or {}).fillna(0.0)

    def predict_gate_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if self.gate_estimator is None or not self.gate_feature_columns:
            return np.ones(len(frame), dtype="float64")
        x = self._gate_frame(frame)
        if hasattr(self.gate_estimator, "predict_proba"):
            return np.asarray(self.gate_estimator.predict_proba(x)[:, 1], dtype="float64")
        score = np.asarray(self.gate_estimator.decision_function(x), dtype="float64")
        return 1.0 / (1.0 + np.exp(-score))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str | Path) -> "StableGateModelBundle":
        with Path(path).open("rb") as fh:
            return pickle.load(fh)


def make_gate_features(frame: pd.DataFrame, primary_probability: np.ndarray, primary_threshold: float) -> pd.DataFrame:
    out = pd.DataFrame(index=frame.index)
    out["primary_probability"] = np.asarray(primary_probability, dtype="float64")
    out["primary_margin"] = out["primary_probability"] - float(primary_threshold)
    numeric_candidates = [
        "bar_atr_pct_14",
        "bar_atr_z_64",
        "bar_rv_ratio_16_64",
        "bar_trend_efficiency_32",
        "bar_trend_efficiency_96",
        "bar_ma_dist_55",
        "bar_donchian_pos_55",
        "cap_chan_structure_consistency_15m_1h_4h_1d",
        "cap_chan_mtf_direction_consensus_score",
        "cap_chan_mtf_direction_conflict_score",
        "cap_chan_structure_maturity_mean",
        "cap_chan_structure_boundary_alignment",
        "cap_chan_nested_divergence_score",
        "cap_chan_entry_style_strength",
        "cap_micro_confirm_score",
    ]
    for col in numeric_candidates:
        if col in frame.columns:
            out[col] = pd.to_numeric(frame[col], errors="coerce")
    states = frame.apply(market_state_from_row, axis=1)
    for state in sorted(set(states.dropna().astype(str))):
        out[f"gate_state_{state}"] = (states == state).astype("float64")
    return out

