from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class MetaModel:
    def __init__(self):
        self.model = LogisticRegression(max_iter=500, class_weight="balanced")
        self.enabled = False
        self.calibration_method = "none"
        self.calibrator = None

    def fit(
        self,
        X,
        y,
        calibration_method: str = "none",
        calibration_ratio: float = 0.2,
    ):
        method = str(calibration_method or "none").strip().lower()
        if method not in {"none", "platt", "isotonic"}:
            method = "none"
        self.calibration_method = method
        self.calibrator = None

        if len(y) < 10 or len(np.unique(y)) < 2:
            self.enabled = False
            return self

        if method == "none" or len(y) < 30:
            self.model.fit(X, y)
            self.enabled = True
            return self

        ratio = float(np.clip(float(calibration_ratio), 0.05, 0.5))
        split = int(round(len(y) * (1.0 - ratio)))
        split = max(10, min(len(y) - 10, split))

        X_fit = X[:split]
        y_fit = y[:split]
        X_cal = X[split:]
        y_cal = y[split:]

        if len(np.unique(y_fit)) < 2 or len(np.unique(y_cal)) < 2:
            self.model.fit(X, y)
            self.calibration_method = "none"
            self.enabled = True
            return self

        self.model.fit(X_fit, y_fit)
        cal_prob = self.model.predict_proba(X_cal)[:, 1].astype(np.float64)

        try:
            if method == "platt":
                calibrator = LogisticRegression(
                    max_iter=300,
                    class_weight="balanced",
                )
                calibrator.fit(cal_prob.reshape(-1, 1), y_cal)
                self.calibrator = calibrator
            elif method == "isotonic":
                calibrator = IsotonicRegression(out_of_bounds="clip")
                calibrator.fit(cal_prob, y_cal)
                self.calibrator = calibrator
        except Exception:
            self.calibrator = None
            self.calibration_method = "none"

        self.enabled = True
        return self

    def predict_proba(self, X):
        if not self.enabled:
            return np.ones(X.shape[0], dtype=np.float32)
        prob = self.model.predict_proba(X)[:, 1].astype(np.float64)

        # Backward compatibility for old pickled MetaModel objects.
        calibrator = getattr(self, "calibrator", None)
        calibration_method = str(
            getattr(self, "calibration_method", "none") or "none"
        ).lower()
        if calibrator is not None and calibration_method == "platt":
            prob = calibrator.predict_proba(prob.reshape(-1, 1))[:, 1]
        elif calibrator is not None and calibration_method == "isotonic":
            prob = calibrator.predict(prob)

        return np.clip(prob, 1e-6, 1.0 - 1e-6).astype(np.float32)
