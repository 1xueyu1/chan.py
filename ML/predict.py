from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd

from Backtest.types import RawBSPEvent
from .features import build_event_features
from .model import ModelBundle


def score_events(
    model_path: str | Path,
    bars: pd.DataFrame,
    events: Sequence[RawBSPEvent],
) -> pd.DataFrame:
    bundle = ModelBundle.load(model_path)
    features = build_event_features(bars, events, required_columns=bundle.feature_columns)
    if features.empty:
        return features

    if bundle.side == "buy":
        features = features[features["is_buy"].astype(bool)].copy()
    elif bundle.side == "sell":
        features = features[~features["is_buy"].astype(bool)].copy()
    return bundle.score_frame(features)
