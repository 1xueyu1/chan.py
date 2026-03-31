from __future__ import annotations

# flake8: noqa: E501

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
import pandas as pd


@dataclass
class NormalizerState:
    mu: Dict[str, float]
    sigma: Dict[str, float]


class ExpandingWindowNormalizer:
    def __init__(self, categorical_features: Iterable[str] | None = None):
        self.categorical_features = set(categorical_features or [])
        self.state = NormalizerState(mu={}, sigma={})

    def fit_transform(self, df: pd.DataFrame, ordered_index: pd.Index | None = None) -> pd.DataFrame:
        out = df.copy()
        if ordered_index is None:
            order_pos = np.arange(len(out), dtype=np.int64)
        else:
            if len(ordered_index) != len(out):
                raise ValueError(
                    "ordered_index length mismatch: "
                    f"{len(ordered_index)} != {len(out)}"
                )
            # 使用位置索引而非标签索引，避免重复标签导致loc扩容。
            order_pos = np.argsort(np.asarray(ordered_index), kind="mergesort")

        ordered = out.iloc[order_pos]

        for col in out.columns:
            if col in self.categorical_features:
                continue
            vals = ordered[col].astype(float)
            exp_mean = vals.expanding(min_periods=2).mean().shift(1)
            exp_std = vals.expanding(min_periods=2).std(ddof=0).shift(1)
            exp_mean = exp_mean.fillna(vals.expanding(min_periods=1).mean())
            exp_std = exp_std.fillna(vals.expanding(min_periods=1).std(ddof=0)).replace(0.0, np.nan)
            transformed = (vals - exp_mean) / (exp_std.fillna(1.0) + 1e-8)
            out.iloc[order_pos, out.columns.get_loc(col)] = transformed.to_numpy()

            self.state.mu[col] = float(vals.mean())
            std = float(vals.std(ddof=0))
            self.state.sigma[col] = std if std > 1e-8 else 1.0

        return out

    def transform_realtime(self, row: pd.Series) -> pd.Series:
        out = row.copy()
        for col, value in out.items():
            if col in self.categorical_features:
                continue
            mu = self.state.mu.get(col, 0.0)
            sigma = self.state.sigma.get(col, 1.0)
            out[col] = (float(value) - mu) / (sigma + 1e-8)
        return out

    def to_dict(self) -> Dict[str, Dict[str, float]]:
        return {"mu": self.state.mu, "sigma": self.state.sigma}


# 向后兼容旧命名。
ExpandingNormalizer = ExpandingWindowNormalizer
