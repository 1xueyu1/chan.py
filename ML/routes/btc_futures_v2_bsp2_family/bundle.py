from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from ML.model import ModelBundle


def normalize_bsp_family(value: Any) -> str:
    text = str(value or "").strip().lower()
    if text in {
        "standard_bsp2",
        "subclass_bsp2s",
        "bsp2_t3_overlap",
        "bsp2_after_bsp1",
        "bsp2_near_origin_zs_boundary",
    }:
        return text
    if text.startswith("1"):
        return "1"
    if text.startswith("2"):
        return "2"
    if text.startswith("3"):
        return "3"
    return "unknown"


def bsp_family_for_frame(frame: pd.DataFrame) -> pd.Series:
    if "bsp2_candidate_family" in frame.columns:
        return frame["bsp2_candidate_family"].map(normalize_bsp_family).astype("object")
    if "bsp2_family" in frame.columns:
        return frame["bsp2_family"].map(normalize_bsp_family).astype("object")
    if "bsp_type" in frame.columns:
        return frame["bsp_type"].map(normalize_bsp_family).astype("object")
    if "bsp_types_str" in frame.columns:
        return frame["bsp_types_str"].map(normalize_bsp_family).astype("object")
    return pd.Series("unknown", index=frame.index, dtype="object")


@dataclass
class SideBspFamilyBundle:
    side: str
    family_bundles: dict[str, ModelBundle]
    disabled_threshold: float = 0.99
    route: str = "btc_futures_v2_bsp2_family"
    extra_feature_columns: list[str] = field(default_factory=lambda: ["bsp_type", "bsp_types_str", "bsp2_candidate_family", "bsp2_family"])

    @property
    def threshold(self) -> float:
        thresholds = [float(bundle.threshold) for bundle in self.family_bundles.values()]
        return float(min(thresholds)) if thresholds else float(self.disabled_threshold)

    @property
    def feature_columns(self) -> list[str]:
        cols: set[str] = set(self.extra_feature_columns)
        for bundle in self.family_bundles.values():
            cols.update(bundle.feature_columns)
        return sorted(cols)

    def prepare_features(self, frame: pd.DataFrame) -> pd.DataFrame:
        try:
            from ML.routes.btc_futures_v3_alpha.dataset import enhance_btc_futures_v3_features
            from ML.routes.btc_futures_v2_bsp2_family.features import enhance_chan_bi_features, enhance_chan_zs_bsp_features

            return enhance_chan_zs_bsp_features(enhance_chan_bi_features(enhance_btc_futures_v3_features(frame)))
        except Exception:
            return frame

    def _predict_for_family(self, frame: pd.DataFrame, family: str) -> np.ndarray:
        bundle = self.family_bundles.get(str(family))
        if bundle is None:
            return np.zeros(len(frame), dtype="float64")
        return bundle.predict_proba(frame)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        out = np.zeros(len(frame), dtype="float64")
        if frame.empty:
            return out
        families = bsp_family_for_frame(frame)
        for family in sorted(set(families.astype(str))):
            mask = (families == str(family)).to_numpy()
            if not bool(mask.any()):
                continue
            out[mask] = self._predict_for_family(frame.loc[mask], family)
        return out

    def threshold_for_rows(self, frame: pd.DataFrame) -> np.ndarray:
        out = np.full(len(frame), float(self.disabled_threshold), dtype="float64")
        if frame.empty:
            return out
        families = bsp_family_for_frame(frame)
        for family, bundle in self.family_bundles.items():
            out[(families == str(family)).to_numpy()] = float(bundle.threshold)
        return out

    def predict_qualified(self, frame: pd.DataFrame, default_threshold: float = 0.99) -> np.ndarray:
        probability = self.predict_proba(frame)
        thresholds = self.threshold_for_rows(frame)
        fallback = np.full(len(frame), float(default_threshold), dtype="float64")
        thresholds = np.where(np.isfinite(thresholds), thresholds, fallback)
        return probability >= thresholds

    def predict_model_key(self, frame: pd.DataFrame) -> np.ndarray:
        families = bsp_family_for_frame(frame)
        return np.asarray([f"v2_{family}_{self.side}" for family in families], dtype=object)

    def save(self, path) -> None:
        import pickle
        from pathlib import Path

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path) -> "SideBspFamilyBundle":
        import pickle
        from pathlib import Path

        with Path(path).open("rb") as fh:
            return pickle.load(fh)

