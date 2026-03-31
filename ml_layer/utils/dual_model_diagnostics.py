from __future__ import annotations

# flake8: noqa: E501

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss

from .metrics import annualized_sharpe


@dataclass
class BucketRow:
    bucket: str
    lower: float
    upper: float
    sample_count: int
    mean_return_pct: float
    win_rate: float
    accuracy: float


def _safe_float(v: object, default: float = 0.0) -> float:
    try:
        x = float(v)
        if np.isfinite(x):
            return x
    except Exception:
        pass
    return float(default)


def _max_drawdown_pct(returns: np.ndarray) -> float:
    if returns.size == 0:
        return 0.0
    equity = np.cumprod(1.0 + returns.astype(np.float64))
    running_max = np.maximum.accumulate(equity)
    drawdown = equity / np.maximum(running_max, 1e-12) - 1.0
    return float(np.min(drawdown) * 100.0)


def _threshold_scan(
    returns: np.ndarray,
    primary_correct: np.ndarray,
    exec_prob: np.ndarray,
    start: float = 0.45,
    stop: float = 0.70,
    step: float = 0.01,
) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    th = start
    while th <= stop + 1e-12:
        keep = exec_prob >= th
        picked = returns[keep]
        picked_correct = primary_correct[keep]
        rows.append(
            {
                "threshold": float(round(th, 4)),
                "trade_count": int(keep.sum()),
                "keep_rate": float(np.mean(keep.astype(np.float32))),
                "mean_return_pct": (
                    float(np.mean(picked) * 100.0) if picked.size else 0.0
                ),
                "total_return_pct": float((np.prod(1.0 + picked) - 1.0) * 100.0) if picked.size else 0.0,
                "max_drawdown_pct": _max_drawdown_pct(picked),
                "sharpe": float(annualized_sharpe(picked)) if picked.size else 0.0,
                "primary_accuracy_on_kept": float(np.mean(picked_correct.astype(np.float32))) if picked_correct.size else 0.0,
            }
        )
        th += step
    return rows


def _primary_confidence_buckets(
    conf: np.ndarray,
    returns: np.ndarray,
    correct_mask: np.ndarray,
) -> List[Dict[str, float]]:
    bins = [(0.50, 0.60), (0.60, 0.70), (0.70, 0.80), (0.80, 0.90), (0.90, 1.01)]
    out: List[Dict[str, float]] = []
    for lo, hi in bins:
        mask = (conf >= lo) & (conf < hi)
        sub_ret = returns[mask]
        sub_corr = correct_mask[mask]
        row = BucketRow(
            bucket=f"{lo:.2f}-{min(hi, 1.0):.2f}",
            lower=float(lo),
            upper=float(min(hi, 1.0)),
            sample_count=int(mask.sum()),
            mean_return_pct=float(np.mean(sub_ret) * 100.0) if sub_ret.size else 0.0,
            win_rate=float(np.mean((sub_ret > 0).astype(np.float32))) if sub_ret.size else 0.0,
            accuracy=float(np.mean(sub_corr.astype(np.float32))) if sub_corr.size else 0.0,
        )
        out.append(row.__dict__)
    return out


def _segment_stats(
    df: pd.DataFrame,
    period: str,
) -> List[Dict[str, float | str | int]]:
    if df.empty:
        return []
    grp = df.groupby(df["open_time"].dt.to_period(period), sort=True)
    rows: List[Dict[str, float | str | int]] = []
    for key, part in grp:
        ret_arr = part["realized_return"].to_numpy(dtype=np.float32)
        corr_arr = part["primary_correct"].to_numpy(dtype=np.float32)
        keep_arr = part["meta_keep"].to_numpy(dtype=np.float32)
        rows.append(
            {
                "segment": str(key),
                "sample_count": int(len(part)),
                "trade_count": int(np.sum(keep_arr)),
                "mean_return_pct": float(np.mean(ret_arr) * 100.0) if ret_arr.size else 0.0,
                "total_return_pct": float((np.prod(1.0 + ret_arr) - 1.0) * 100.0) if ret_arr.size else 0.0,
                "primary_accuracy": float(np.mean(corr_arr)) if corr_arr.size else 0.0,
                "meta_keep_rate": float(np.mean(keep_arr)) if keep_arr.size else 0.0,
            }
        )
    return rows


def _meta_calibration(meta_y: np.ndarray, probs: np.ndarray) -> Dict[str, float]:
    if meta_y.size == 0:
        return {"brier": 0.0, "logloss": 0.0}
    probs = np.clip(probs.astype(np.float64), 1e-8, 1.0 - 1e-8)
    brier = float(np.mean((probs - meta_y.astype(np.float64)) ** 2))
    ll = 0.0
    try:
        ll = float(log_loss(meta_y, probs, labels=[0, 1]))
    except Exception:
        ll = 0.0
    return {"brier": brier, "logloss": ll}


def _meta_feature_dependency(meta_model, meta_feature_names: List[str]) -> Dict[str, object]:
    model = getattr(meta_model, "model", None)
    enabled = bool(getattr(meta_model, "enabled", False))
    if not enabled or model is None or not hasattr(model, "coef_"):
        return {
            "enabled": False,
            "top_feature_abs_share": 0.0,
            "top5_abs_share": 0.0,
            "top10_abs_share": 0.0,
            "top_features": [],
        }

    coef = np.asarray(model.coef_[0], dtype=np.float64)
    abs_coef = np.abs(coef)
    total = float(np.sum(abs_coef))
    if total <= 1e-12:
        return {
            "enabled": True,
            "top_feature_abs_share": 0.0,
            "top5_abs_share": 0.0,
            "top10_abs_share": 0.0,
            "top_features": [],
        }

    pairs = []
    for idx, weight in enumerate(coef):
        name = meta_feature_names[idx] if idx < len(meta_feature_names) else f"f_{idx}"
        share = float(abs(weight) / total)
        pairs.append(
            {
                "feature": name,
                "coef": float(weight),
                "abs_coef": float(abs(weight)),
                "abs_share": share,
            }
        )
    pairs.sort(key=lambda x: x["abs_coef"], reverse=True)

    top5 = float(sum(item["abs_share"] for item in pairs[:5]))
    top10 = float(sum(item["abs_share"] for item in pairs[:10]))
    return {
        "enabled": True,
        "top_feature_abs_share": float(pairs[0]["abs_share"] if pairs else 0.0),
        "top5_abs_share": top5,
        "top10_abs_share": top10,
        "top_features": pairs[:20],
    }


def build_dual_model_diagnostics(
    samples_df: pd.DataFrame,
    y: np.ndarray,
    primary_probs: np.ndarray,
    primary_pred: np.ndarray,
    meta_exec_prob: np.ndarray,
    meta_threshold: float,
    meta_model,
    base_feature_names: List[str],
) -> Dict[str, object]:
    if samples_df.empty:
        return {"error": "empty_samples"}

    df = samples_df.copy()
    df["open_time"] = pd.to_datetime(df["open_time"], errors="coerce", utc=True)
    df = df.dropna(subset=["open_time"]).copy()

    rets = df["realized_return"].to_numpy(dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.int32)
    pred_arr = np.asarray(primary_pred, dtype=np.int32)
    probs = np.asarray(primary_probs, dtype=np.float32)
    exec_prob = np.asarray(meta_exec_prob, dtype=np.float32)

    if probs.ndim == 1:
        probs = np.stack([1.0 - probs, probs], axis=1)

    n = int(min(len(df), len(y_arr), len(pred_arr), len(exec_prob), probs.shape[0]))
    if n <= 0:
        return {"error": "no_aligned_samples"}

    df = df.iloc[:n].copy()
    rets = rets[:n]
    y_arr = y_arr[:n]
    pred_arr = pred_arr[:n]
    probs = probs[:n]
    exec_prob = exec_prob[:n]

    # 对二分类约定: 1=PT(long), 0=SL(short)
    p_pt = probs[:, -1]
    p_sl = probs[:, 0]
    conf = np.maximum(p_pt, p_sl)

    primary_correct = (pred_arr == y_arr)
    meta_keep = exec_prob >= float(meta_threshold)

    df["y"] = y_arr
    df["primary_pred"] = pred_arr
    df["primary_correct"] = primary_correct.astype(np.int32)
    df["meta_exec_prob"] = exec_prob
    df["meta_keep"] = meta_keep.astype(np.int32)

    label_pt_ratio = float(np.mean((y_arr == 1).astype(np.float32))) if y_arr.size else 0.0
    pred_pt_ratio = float(np.mean((pred_arr == 1).astype(np.float32))) if pred_arr.size else 0.0

    bsp_rows: List[Dict[str, object]] = []
    if "bsp_main_type" in df.columns:
        for bsp_type, part in df.groupby("bsp_main_type", sort=True):
            corr = part["primary_correct"].to_numpy(dtype=np.float32)
            r = part["realized_return"].to_numpy(dtype=np.float32)
            bsp_rows.append(
                {
                    "bsp_type": str(bsp_type),
                    "sample_count": int(len(part)),
                    "primary_accuracy": float(np.mean(corr)) if corr.size else 0.0,
                    "mean_return_pct": float(np.mean(r) * 100.0) if r.size else 0.0,
                    "meta_keep_rate": float(np.mean(part["meta_keep"].to_numpy(dtype=np.float32))) if len(part) else 0.0,
                }
            )

    threshold_scan_rows = _threshold_scan(
        returns=rets,
        primary_correct=primary_correct,
        exec_prob=exec_prob,
        start=0.45,
        stop=0.70,
        step=0.01,
    )

    meta_y = primary_correct.astype(np.int32)
    calibration = _meta_calibration(meta_y, exec_prob)

    high_quality = (primary_correct == 1) & (rets > 0)
    low_quality = (primary_correct == 0) | (rets <= 0)

    false_reject = (~meta_keep) & high_quality
    false_accept = meta_keep & low_quality
    high_quality_count = int(np.sum(high_quality))
    low_quality_count = int(np.sum(low_quality))

    meta_feature_names = list(base_feature_names)
    if probs.shape[1] == 2:
        meta_feature_names.extend(["primary_p_sl", "primary_p_pt"])
    else:
        meta_feature_names.extend([f"primary_p_cls_{i}" for i in range(probs.shape[1])])

    dependency = _meta_feature_dependency(meta_model, meta_feature_names)

    month_stats = _segment_stats(df, "M")
    quarter_stats = _segment_stats(df, "Q")

    positive_month_ratio = 0.0
    if month_stats:
        positive_month_ratio = float(
            np.mean(np.array([row["total_return_pct"] > 0 for row in month_stats], dtype=np.float32))
        )

    return {
        "primary": {
            "class_distribution": {
                "label_pt_ratio": label_pt_ratio,
                "label_sl_ratio": float(1.0 - label_pt_ratio),
                "pred_pt_ratio": pred_pt_ratio,
                "pred_sl_ratio": float(1.0 - pred_pt_ratio),
                "direction_bias": float(pred_pt_ratio - label_pt_ratio),
                "mean_p_pt": float(np.mean(p_pt)) if p_pt.size else 0.0,
                "mean_p_sl": float(np.mean(p_sl)) if p_sl.size else 0.0,
            },
            "bsp_type_accuracy": bsp_rows,
            "confidence_bucket_returns": _primary_confidence_buckets(
                conf=conf,
                returns=rets,
                correct_mask=primary_correct,
            ),
            "time_stability": {
                "monthly": month_stats,
                "quarterly": quarter_stats,
                "positive_month_ratio": positive_month_ratio,
            },
        },
        "meta": {
            "threshold_scan": threshold_scan_rows,
            "calibration": calibration,
            "false_reject_false_accept": {
                "high_quality_count": high_quality_count,
                "low_quality_count": low_quality_count,
                "false_reject_count": int(np.sum(false_reject)),
                "false_accept_count": int(np.sum(false_accept)),
                "false_reject_ratio": float(np.sum(false_reject) / max(1, high_quality_count)),
                "false_accept_ratio": float(np.sum(false_accept) / max(1, low_quality_count)),
            },
            "feature_dependency": dependency,
        },
        "summary": {
            "sample_count": int(len(df)),
            "meta_threshold": float(meta_threshold),
            "meta_keep_rate": float(np.mean(meta_keep.astype(np.float32))) if meta_keep.size else 0.0,
            "primary_accuracy": float(np.mean(primary_correct.astype(np.float32))) if primary_correct.size else 0.0,
            "primary_mean_return_pct": float(np.mean(rets) * 100.0) if rets.size else 0.0,
            "meta_kept_mean_return_pct": float(np.mean(rets[meta_keep]) * 100.0) if np.any(meta_keep) else 0.0,
        },
    }
