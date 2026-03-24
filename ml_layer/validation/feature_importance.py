from __future__ import annotations

# flake8: noqa: E501

from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import f1_score


def compute_mdi_mda(
    primary_booster,
    feature_names: List[str],
    X: np.ndarray,
    y: np.ndarray,
    mda_max_samples: int = 6000,
    mda_n_jobs: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    gain = primary_booster.get_score(importance_type="gain")
    mdi_rows = []
    for i, name in enumerate(feature_names):
        mdi_rows.append({"feature": name, "mdi_gain": float(gain.get(f"f{i}", gain.get(name, 0.0)))})
    mdi_df = pd.DataFrame(mdi_rows).sort_values("mdi_gain", ascending=False).reset_index(drop=True)

    if X.shape[0] > int(mda_max_samples):
        rng = np.random.default_rng(42)
        pick = rng.choice(X.shape[0], size=int(mda_max_samples), replace=False)
        X_eval = X[pick]
        y_eval = y[pick]
    else:
        X_eval = X
        y_eval = y

    base_prob = primary_booster.predict(xgb.DMatrix(X_eval, missing=np.nan))
    base_prob = np.asarray(base_prob)
    if base_prob.ndim == 1:
        base_prob = np.stack([1.0 - base_prob, np.zeros_like(base_prob), base_prob], axis=1)
    base_pred = np.argmax(base_prob, axis=1)
    base_score = f1_score(y_eval, base_pred, average="macro", zero_division=0)

    def _one_feature_mda(idx_name: Tuple[int, str]):
        i, name = idx_name
        local_rng = np.random.default_rng(42 + i)
        X_perm = np.array(X_eval, copy=True)
        shuffled = np.array(X_perm[:, i], copy=True)
        local_rng.shuffle(shuffled)
        X_perm[:, i] = shuffled
        p = primary_booster.predict(xgb.DMatrix(X_perm, missing=np.nan))
        p = np.asarray(p)
        if p.ndim == 1:
            p = np.stack([1.0 - p, np.zeros_like(p), p], axis=1)
        pred = np.argmax(p, axis=1)
        score = f1_score(y_eval, pred, average="macro", zero_division=0)
        return {"feature": name, "mda_drop": float(base_score - score)}

    jobs = max(1, int(mda_n_jobs))
    idx_names = list(enumerate(feature_names))
    if jobs <= 1 or len(idx_names) <= 1:
        mda_rows = [_one_feature_mda(item) for item in idx_names]
    else:
        with ThreadPoolExecutor(max_workers=min(jobs, len(idx_names))) as ex:
            mda_rows = list(ex.map(_one_feature_mda, idx_names))

    mda_df = pd.DataFrame(mda_rows).sort_values("mda_drop", ascending=False).reset_index(drop=True)
    return mdi_df, mda_df
