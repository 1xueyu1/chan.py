from __future__ import annotations

from typing import Dict

import numpy as np
import xgboost as xgb


class _BasePrimaryModel:
    def __init__(self, params: Dict[str, object], num_rounds: int, early_stop: int):
        self.params = dict(params)
        self.num_rounds = int(num_rounds)
        self.early_stop = int(early_stop)
        self.model = None
        self.backend = "unknown"

    def _build_train_params(self) -> Dict[str, object]:
        raise NotImplementedError

    def _fit_once(self, params: Dict[str, object], dtr: xgb.DMatrix, dva: xgb.DMatrix):
        self.model = xgb.train(
            params,
            dtr,
            num_boost_round=self.num_rounds,
            evals=[(dva, "valid")],
            early_stopping_rounds=self.early_stop,
            verbose_eval=False,
        )

    def predict_proba(self, X):
        if self.model is None:
            raise RuntimeError("PrimaryModel is not fitted")
        p = self.model.predict(xgb.DMatrix(X, missing=np.nan))
        arr = np.asarray(p)
        if arr.ndim == 1:
            arr = np.stack([1.0 - arr, arr], axis=1)
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


class CPUPrimaryModel(_BasePrimaryModel):
    def _build_train_params(self) -> Dict[str, object]:
        params = dict(self.params)
        params["tree_method"] = "hist"
        params["device"] = "cpu"
        params["predictor"] = "cpu_predictor"
        self.backend = "cpu"
        return params

    def fit(self, X_train, y_train, w_train, X_valid, y_valid, train_mode: str = "auto"):
        _ = train_mode  # 保持签名兼容
        params = self._build_train_params()

        dtr = xgb.DMatrix(X_train, label=y_train, weight=w_train, missing=np.nan)
        dva = xgb.DMatrix(X_valid, label=y_valid, missing=np.nan)
        self._fit_once(params, dtr, dva)
        return self


class GPUPrimaryModel(_BasePrimaryModel):
    def _build_gpu_hist_params(self) -> Dict[str, object]:
        params = dict(self.params)
        # 关键：gpu_hist 才是传统明确的 GPU 训练开关。
        params["tree_method"] = "gpu_hist"
        params["predictor"] = "gpu_predictor"
        params["device"] = "cuda"
        return params

    def _build_cuda_hist_params(self) -> Dict[str, object]:
        params = dict(self.params)
        # XGBoost 2.x+ 推荐写法：hist + device=cuda。
        params["tree_method"] = "hist"
        params["predictor"] = "gpu_predictor"
        params["device"] = "cuda"
        return params

    def fit(self, X_train, y_train, w_train, X_valid, y_valid, train_mode: str = "gpu"):
        _ = train_mode  # 保持签名兼容
        dtr = xgb.DMatrix(X_train, label=y_train, weight=w_train, missing=np.nan)
        dva = xgb.DMatrix(X_valid, label=y_valid, missing=np.nan)

        try:
            self._fit_once(self._build_gpu_hist_params(), dtr, dva)
            self.backend = "gpu:gpu_hist"
        except xgb.core.XGBoostError:
            self._fit_once(self._build_cuda_hist_params(), dtr, dva)
            self.backend = "gpu:hist+cuda"
        return self


class AutoPrimaryModel(_BasePrimaryModel):
    """自动模式：优先GPU，失败后自动回退CPU。"""

    def fit(self, X_train, y_train, w_train, X_valid, y_valid, train_mode: str = "auto"):
        _ = train_mode  # 保持签名兼容
        gpu_model = GPUPrimaryModel(self.params, self.num_rounds, self.early_stop)
        try:
            gpu_model.fit(X_train, y_train, w_train, X_valid, y_valid, train_mode="gpu")
            self.model = gpu_model.model
            self.backend = gpu_model.backend
            return self
        except xgb.core.XGBoostError:
            cpu_model = CPUPrimaryModel(self.params, self.num_rounds, self.early_stop)
            cpu_model.fit(X_train, y_train, w_train, X_valid, y_valid, train_mode="cpu")
            self.model = cpu_model.model
            self.backend = "auto->cpu"
        return self


def create_primary_model(
    params: Dict[str, object],
    num_rounds: int,
    early_stop: int,
    train_mode: str,
) -> _BasePrimaryModel:
    mode = str(train_mode or "auto").strip().lower()
    if mode == "cpu":
        return CPUPrimaryModel(params, num_rounds, early_stop)
    if mode == "gpu":
        return GPUPrimaryModel(params, num_rounds, early_stop)
    return AutoPrimaryModel(params, num_rounds, early_stop)


# 向后兼容旧引用：PrimaryModel 默认等价自动模式。
PrimaryModel = AutoPrimaryModel
