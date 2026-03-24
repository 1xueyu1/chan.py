from __future__ import annotations

from typing import Dict

import numpy as np
import xgboost as xgb


class PrimaryModel:
    def __init__(self, params: Dict[str, object], num_rounds: int, early_stop: int):
        self.params = dict(params)
        self.num_rounds = int(num_rounds)
        self.early_stop = int(early_stop)
        self.model = None

    def fit(self, X_train, y_train, w_train, X_valid, y_valid, train_mode: str = "auto"):
        params = dict(self.params)
        params["tree_method"] = "hist"
        if train_mode == "gpu":
            params["device"] = "cuda"
        elif train_mode == "cpu":
            params["device"] = "cpu"
        else:
            params["device"] = "cuda"

        dtr = xgb.DMatrix(X_train, label=y_train, weight=w_train, missing=np.nan)
        dva = xgb.DMatrix(X_valid, label=y_valid, missing=np.nan)

        try:
            self.model = xgb.train(
                params,
                dtr,
                num_boost_round=self.num_rounds,
                evals=[(dva, "valid")],
                early_stopping_rounds=self.early_stop,
                verbose_eval=False,
            )
        except xgb.core.XGBoostError:
            params["device"] = "cpu"
            self.model = xgb.train(
                params,
                dtr,
                num_boost_round=self.num_rounds,
                evals=[(dva, "valid")],
                early_stopping_rounds=self.early_stop,
                verbose_eval=False,
            )
        return self

    def predict_proba(self, X):
        if self.model is None:
            raise RuntimeError("PrimaryModel is not fitted")
        p = self.model.predict(xgb.DMatrix(X, missing=np.nan))
        arr = np.asarray(p)
        if arr.ndim == 1:
            arr = np.stack([1.0 - arr, np.zeros_like(arr), arr], axis=1)
        return arr

    def predict_class(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def save(self, path: str):
        if self.model is None:
            raise RuntimeError("PrimaryModel is not fitted")
        self.model.save_model(path)

    @property
    def booster(self):
        return self.model
