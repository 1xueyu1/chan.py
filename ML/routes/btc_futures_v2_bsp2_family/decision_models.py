from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ML.model import ModelBundle, train_classifier
from ML.features import infer_feature_columns

from .candidate_family import attach_bsp2_candidate_family
from .labels import BSP2_REALTIME_PATH_CLASSES


DECISION_MODEL_FILENAME = "bsp2_decision_models.pkl"
DROP_FEATURE_PREFIXES = ("label_",)
DROP_FEATURE_EXACT = {"entry_time", "exit_time", "signal_available_time", "exec_time"}


def _candidate_features(frame: pd.DataFrame, min_feature_coverage: float) -> list[str]:
    cols = infer_feature_columns(
        frame,
        extra_prefixes=("cap_", "v3_", "peer_", "tech_", "asset_", "bsp2_family_is_"),
    )
    out: list[str] = []
    for col in cols:
        if col in DROP_FEATURE_EXACT or any(str(col).startswith(prefix) for prefix in DROP_FEATURE_PREFIXES):
            continue
        series = pd.to_numeric(frame[col], errors="coerce")
        if float(series.notna().mean()) < float(min_feature_coverage):
            continue
        if int(series.nunique(dropna=True)) <= 1:
            continue
        out.append(col)
    return out


def _classification_metrics(frame: pd.DataFrame, probability: np.ndarray) -> dict[str, Any]:
    y = pd.to_numeric(frame["label"], errors="coerce")
    prob = np.asarray(probability, dtype="float64")
    out: dict[str, Any] = {"rows": int(len(frame)), "positive_rate": float(y.mean()) if len(y) else float("nan")}
    if len(frame) and y.nunique(dropna=True) > 1:
        try:
            from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

            out.update(
                {
                    "auc": float(roc_auc_score(y.astype(int), prob)),
                    "ap": float(average_precision_score(y.astype(int), prob)),
                    "brier": float(brier_score_loss(y.astype(int), prob)),
                }
            )
        except Exception:
            pass
    return out


@dataclass
class Bsp2DecisionModelBundle:
    identity_models: dict[str, ModelBundle]
    invalid_risk_models: dict[str, ModelBundle]
    path_models: dict[str, dict[str, ModelBundle]]
    path_classes: tuple[str, ...] = BSP2_REALTIME_PATH_CLASSES

    def _side_key(self, frame: pd.DataFrame) -> pd.Series:
        return pd.Series(np.where(frame["is_buy"].astype(bool), "buy", "sell"), index=frame.index)

    def score_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = frame.copy()
        out["bsp2_identity_prob"] = np.nan
        out["bsp2_pre_invalid_prob"] = np.nan
        for path in self.path_classes:
            out[f"bsp2_path_prob_{path}"] = np.nan
        side_key = self._side_key(out)
        for side in ("buy", "sell"):
            mask = side_key.eq(side)
            if not bool(mask.any()):
                continue
            rows = out.loc[mask]
            identity = self.identity_models.get(side)
            invalid = self.invalid_risk_models.get(side)
            if identity is not None:
                out.loc[mask, "bsp2_identity_prob"] = identity.predict_proba(rows)
            if invalid is not None:
                out.loc[mask, "bsp2_pre_invalid_prob"] = invalid.predict_proba(rows)
            for path, model in self.path_models.get(side, {}).items():
                out.loc[mask, f"bsp2_path_prob_{path}"] = model.predict_proba(rows)
        path_cols = [f"bsp2_path_prob_{path}" for path in self.path_classes]
        values = out[path_cols].replace([np.inf, -np.inf], np.nan).fillna(-1.0).to_numpy(dtype="float64")
        best_idx = np.argmax(values, axis=1) if len(values) else np.array([], dtype=int)
        out["bsp2_path_pred"] = [self.path_classes[int(idx)] for idx in best_idx] if len(best_idx) else []
        out["bsp2_path_confidence"] = np.nanmax(values, axis=1) if len(values) else np.nan
        return out

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str | Path) -> "Bsp2DecisionModelBundle":
        with Path(path).open("rb") as fh:
            return pickle.load(fh)


def _train_binary(
    frame: pd.DataFrame,
    *,
    label_column: str,
    feature_columns: list[str],
    model_kind: str,
    side: str,
    random_state: int,
) -> ModelBundle | None:
    data = frame.dropna(subset=[label_column, "is_buy"]).copy()
    data["label"] = pd.to_numeric(data[label_column], errors="coerce")
    data = data.dropna(subset=["label"])
    data["label"] = data["label"].astype(int)
    if data["label"].nunique(dropna=True) < 2 or len(data) < 200:
        return None
    return train_classifier(
        data,
        feature_columns=feature_columns,
        model_kind=model_kind,
        threshold=0.5,
        side=side,
        random_state=random_state,
    )


def train_bsp2_decision_models(
    dataset: pd.DataFrame,
    *,
    model_dir: str | Path,
    train_start: Any | None = "2021-01-01",
    valid_start: Any | None = "2025-01-01",
    model_kind: str = "auto",
    min_feature_coverage: float = 0.05,
) -> dict[str, Any]:
    frame = attach_bsp2_candidate_family(dataset)
    for col in ("exec_time", "entry_time"):
        frame[col] = pd.to_datetime(frame[col], utc=True, errors="coerce")
    valid_start_ts = pd.to_datetime(valid_start, utc=True) if valid_start is not None else None
    train_start_ts = pd.to_datetime(train_start, utc=True) if train_start is not None else None
    train_mask = frame["exec_time"].notna()
    if train_start_ts is not None:
        train_mask &= frame["exec_time"] >= train_start_ts
    if valid_start_ts is not None:
        train_mask &= frame["exec_time"] < valid_start_ts
    train = frame.loc[train_mask].dropna(subset=["entry_time"]).copy()
    if train.empty:
        raise ValueError("empty train split for BSP2 decision models")
    feature_columns = _candidate_features(train, min_feature_coverage)
    if not feature_columns:
        raise ValueError("no usable features for BSP2 decision models")

    identity_models: dict[str, ModelBundle] = {}
    invalid_models: dict[str, ModelBundle] = {}
    path_models: dict[str, dict[str, ModelBundle]] = {"buy": {}, "sell": {}}
    metrics: dict[str, Any] = {
        "route": "btc_futures_v2_bsp2_family",
        "model": "bsp2_decision_models",
        "rows": int(len(train)),
        "feature_count": int(len(feature_columns)),
        "sides": {},
        "path_classes": list(BSP2_REALTIME_PATH_CLASSES),
    }
    for side, is_buy in (("buy", True), ("sell", False)):
        side_train = train.loc[train["is_buy"].astype(bool).eq(is_buy)].copy()
        side_metrics: dict[str, Any] = {"rows": int(len(side_train)), "heads": {}}
        identity = _train_binary(
            side_train,
            label_column="label_bsp2_identity_strict",
            feature_columns=feature_columns,
            model_kind=model_kind,
            side="both",
            random_state=3101 if is_buy else 3102,
        )
        if identity is not None:
            identity_models[side] = identity
            side_metrics["heads"]["identity"] = _classification_metrics(side_train.assign(label=side_train["label_bsp2_identity_strict"]), identity.predict_proba(side_train))
        invalid = _train_binary(
            side_train,
            label_column="label_pre_confirm_invalid",
            feature_columns=feature_columns,
            model_kind=model_kind,
            side="both",
            random_state=3201 if is_buy else 3202,
        )
        if invalid is not None:
            invalid_models[side] = invalid
            side_metrics["heads"]["pre_confirm_invalid"] = _classification_metrics(side_train.assign(label=side_train["label_pre_confirm_invalid"]), invalid.predict_proba(side_train))
        for idx, path in enumerate(BSP2_REALTIME_PATH_CLASSES):
            label_col = f"label_path_is_{path}"
            side_train[label_col] = (side_train["label_bsp2_realtime_path"].astype(str) == path).astype(int)
            model = _train_binary(
                side_train,
                label_column=label_col,
                feature_columns=feature_columns,
                model_kind=model_kind,
                side="both",
                random_state=3300 + idx * 10 + (1 if is_buy else 2),
            )
            if model is not None:
                path_models[side][path] = model
                side_metrics["heads"][f"path_{path}"] = _classification_metrics(side_train.assign(label=side_train[label_col]), model.predict_proba(side_train))
        metrics["sides"][side] = side_metrics

    bundle = Bsp2DecisionModelBundle(
        identity_models=identity_models,
        invalid_risk_models=invalid_models,
        path_models=path_models,
    )
    model_root = Path(model_dir)
    bundle.save(model_root / DECISION_MODEL_FILENAME)
    (model_root / "bsp2_decision_models_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def load_bsp2_decision_models(model_dir: str | Path) -> Bsp2DecisionModelBundle | None:
    path = Path(model_dir) / DECISION_MODEL_FILENAME
    if not path.exists():
        return None
    return Bsp2DecisionModelBundle.load(path)
