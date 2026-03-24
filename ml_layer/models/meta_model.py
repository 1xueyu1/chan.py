from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


class MetaModel:
    def __init__(self):
        self.model = LogisticRegression(max_iter=500, class_weight="balanced")
        self.enabled = False

    def fit(self, X, y):
        if len(y) < 10 or len(np.unique(y)) < 2:
            self.enabled = False
            return self
        self.model.fit(X, y)
        self.enabled = True
        return self

    def predict_proba(self, X):
        if not self.enabled:
            return np.ones(X.shape[0], dtype=np.float32)
        return self.model.predict_proba(X)[:, 1].astype(np.float32)
