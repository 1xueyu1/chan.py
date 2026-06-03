from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd


def time_split(
    frame: pd.DataFrame,
    valid_start: Optional[str] = None,
    valid_ratio: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if frame.empty:
        return frame, frame

    data = frame.sort_values("exec_time").reset_index(drop=True)
    if valid_start:
        split_ts = pd.Timestamp(valid_start, tz="UTC")
        return data[data["exec_time"] < split_ts].copy(), data[data["exec_time"] >= split_ts].copy()

    split_idx = int(len(data) * (1.0 - float(valid_ratio)))
    split_idx = min(max(split_idx, 1), len(data) - 1)
    return data.iloc[:split_idx].copy(), data.iloc[split_idx:].copy()


def classification_report_dict(y_true, y_score, threshold: float = 0.5) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, average_precision_score, precision_score, recall_score, roc_auc_score

    y_pred = (y_score >= threshold).astype(int)
    out = {
        "rows": float(len(y_pred)),
        "positive_rate": float(pd.Series(y_true).mean()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "average_precision": float(average_precision_score(y_true, y_score)),
    }
    if len(set(y_true)) > 1:
        out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    return out
