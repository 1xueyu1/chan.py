from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np

from .feature_engine import FeatureEngine
from .models import MetaModel, PrimaryModel


@dataclass
class SignalOutput:
    ts: str
    direction: int
    confidence: float
    primary_proba: Dict[str, float]
    meta_proba: float
    bsp_type: str
    sl_price: float
    features_snapshot: Dict[str, float]


class InferenceEngine:
    def __init__(self, feature_engine: FeatureEngine, primary: PrimaryModel, meta: MetaModel, meta_threshold: float = 0.55):
        self.feature_engine = feature_engine
        self.primary = primary
        self.meta = meta
        self.meta_threshold = float(meta_threshold)

    def infer(self, ts: str, bsp_type: str, sl_price: float, feature_row: Dict[str, float]) -> SignalOutput:
        import pandas as pd

        row = pd.Series(feature_row, dtype=float)
        row_norm = self.feature_engine.transform_realtime(row)
        X = np.array(row_norm.values, dtype=np.float32).reshape(1, -1)

        p_primary = self.primary.predict_proba(X)[0]
        primary_class = int(np.argmax(p_primary))
        primary_direction = -1 if primary_class == 0 else 1

        p_meta = 1.0
        if primary_direction != 0:
            meta_input = np.hstack([np.nan_to_num(X, nan=0.0), p_primary.reshape(1, -1)])
            p_meta = float(self.meta.predict_proba(meta_input)[0])

        final_direction = primary_direction if p_meta >= self.meta_threshold else 0
        p_short = float(p_primary[0])
        p_long = float(p_primary[-1])
        return SignalOutput(
            ts=ts,
            direction=final_direction,
            confidence=float(p_meta),
            primary_proba={"short": p_short, "long": p_long},
            meta_proba=float(p_meta),
            bsp_type=bsp_type,
            sl_price=float(sl_price),
            features_snapshot={k: float(v) for k, v in row_norm.to_dict().items()},
        )
