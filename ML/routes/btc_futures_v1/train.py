from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ML.features import infer_feature_columns
from ML.model import ModelBundle, train_classifier
from ML.shared.json_utils import json_safe

from .bundle import SideBspFamilyBundle, bsp_family_for_frame
from .data import DEFAULT_DATASET_PATH, DEFAULT_MODEL_DIR, parse_utc


PRIMARY_LABEL_COLUMN = "label_bsp2_valid"
BSP_FAMILIES = ("2",)

DROP_FEATURE_EXACT = {
    "label",
    "gross_return",
    "net_return",
    "mfe",
    "mae",
    "entry_price",
    "exit_price",
    "holding_minutes",
    "entry_bar_idx",
    "label_take_profit_pct",
    "label_stop_loss_pct",
    "label_max_holding_minutes",
}
DROP_FEATURE_PREFIXES = ("cap_event_seq",)


def _safe_metric(func, y_true: np.ndarray, y_score: np.ndarray) -> float:
    try:
        return float(func(y_true, y_score))
    except Exception:
        return float("nan")


def _classification_metrics(frame: pd.DataFrame, probability: np.ndarray) -> dict[str, Any]:
    y = frame["label"].astype(int).to_numpy()
    out: dict[str, Any] = {"rows": int(len(frame)), "positive_rate": float(np.mean(y)) if len(y) else float("nan")}
    if len(np.unique(y)) < 2:
        out.update({"auc": float("nan"), "ap": float("nan"), "brier": float("nan")})
        return out
    try:
        from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

        out["auc"] = _safe_metric(roc_auc_score, y, probability)
        out["ap"] = _safe_metric(average_precision_score, y, probability)
        out["brier"] = _safe_metric(brier_score_loss, y, probability)
    except Exception:
        out.update({"auc": float("nan"), "ap": float("nan"), "brier": float("nan")})
    return out


def _threshold_scan(
    frame: pd.DataFrame,
    probability: np.ndarray,
    *,
    min_trades: int,
    min_daily_trades: float,
    min_precision: float,
    min_trade_win_rate: float,
    min_avg_net_return: float,
    frequency_floor_threshold: float | None = None,
) -> tuple[float, pd.DataFrame, dict[str, Any]]:
    if frame.empty:
        return 0.99, pd.DataFrame(), {"enabled": False, "reason": "empty_valid"}
    y = frame["label"].astype(int).to_numpy()
    times = pd.to_datetime(frame["entry_time"], utc=True, errors="coerce")
    days = max(1.0, (times.max() - times.min()).total_seconds() / 86400.0)
    rows = []
    for threshold in np.round(np.arange(0.45, 0.951, 0.01), 2):
        mask = probability >= threshold
        trades = int(mask.sum())
        precision = float(np.mean(y[mask])) if trades else float("nan")
        returns = (
            pd.to_numeric(frame.loc[mask, "net_return"], errors="coerce").dropna()
            if trades and "net_return" in frame.columns
            else pd.Series(dtype="float64")
        )
        avg_return = float(returns.mean()) if len(returns) else float("nan")
        median_return = float(returns.median()) if len(returns) else float("nan")
        trade_win_rate = float((returns > 0).mean()) if len(returns) else float("nan")
        gross_profit = float(returns.loc[returns > 0].sum()) if len(returns) else 0.0
        gross_loss = float(-returns.loc[returns < 0].sum()) if len(returns) else 0.0
        profit_factor = float(gross_profit / gross_loss) if gross_loss > 0 else (float("inf") if gross_profit > 0 else float("nan"))
        total_net_return = float(returns.sum()) if len(returns) else float("nan")
        rows.append(
            {
                "threshold": float(threshold),
                "trades": trades,
                "daily_trades": float(trades / days),
                "precision": precision,
                "avg_net_return": avg_return,
                "median_net_return": median_return,
                "trade_win_rate": trade_win_rate,
                "profit_factor": profit_factor,
                "total_net_return": total_net_return,
            }
        )
    scan = pd.DataFrame(rows)
    liquid = scan.loc[
        (scan["trades"] >= int(min_trades))
        & (scan["daily_trades"] >= float(min_daily_trades))
    ].copy()
    if liquid.empty:
        return 0.99, scan, {"enabled": False, "reason": "no_frequency_threshold"}

    ok = liquid.loc[
        (liquid["precision"] >= float(min_precision))
        & (liquid["trade_win_rate"] >= float(min_trade_win_rate))
        & (liquid["avg_net_return"] >= float(min_avg_net_return))
    ].copy()
    if ok.empty:
        fallback = liquid.copy()
        if frequency_floor_threshold is not None:
            capped = fallback.loc[fallback["threshold"] <= float(frequency_floor_threshold)].copy()
            if not capped.empty:
                fallback = capped
        fallback["edge_score"] = (
            fallback["avg_net_return"].fillna(-1.0) * 100.0
            + fallback["trade_win_rate"].fillna(0.0) * 2.0
            + np.log1p(fallback["trades"].astype(float)) * 0.02
            + fallback["precision"].fillna(0.0) * 0.25
        )
        best = fallback.sort_values(["edge_score", "avg_net_return", "trade_win_rate", "threshold"], ascending=[False, False, False, False]).iloc[0]
        decision = {
            "enabled": True,
            "threshold": float(best["threshold"]),
            "valid_trades": int(best["trades"]),
            "valid_daily_trades": float(best["daily_trades"]),
            "valid_precision": float(best["precision"]) if pd.notna(best["precision"]) else float("nan"),
            "valid_avg_net_return": float(best["avg_net_return"]) if pd.notna(best["avg_net_return"]) else float("nan"),
            "valid_trade_win_rate": float(best["trade_win_rate"]) if pd.notna(best["trade_win_rate"]) else float("nan"),
            "valid_profit_factor": float(best["profit_factor"]) if pd.notna(best["profit_factor"]) else float("nan"),
            "valid_total_net_return": float(best["total_net_return"]) if pd.notna(best["total_net_return"]) else float("nan"),
            "reason": "best_available_edge_no_positive_pass",
            "strict_edge_pass": False,
        }
        return float(best["threshold"]), scan, decision
    ok["edge_score"] = (
        ok["avg_net_return"].fillna(-1.0) * 100.0
        + ok["trade_win_rate"].fillna(0.0) * 2.0
        + np.log1p(ok["trades"].astype(float)) * 0.02
        + ok["precision"].fillna(0.0) * 0.25
    )
    best = ok.sort_values(["edge_score", "avg_net_return", "trade_win_rate", "trades"], ascending=[False, False, False, False]).iloc[0]
    decision = {
        "enabled": True,
        "threshold": float(best["threshold"]),
        "valid_trades": int(best["trades"]),
        "valid_daily_trades": float(best["daily_trades"]),
        "valid_precision": float(best["precision"]),
        "valid_avg_net_return": float(best["avg_net_return"]),
        "valid_trade_win_rate": float(best["trade_win_rate"]),
        "valid_profit_factor": float(best["profit_factor"]) if pd.notna(best["profit_factor"]) else float("nan"),
        "valid_total_net_return": float(best["total_net_return"]) if pd.notna(best["total_net_return"]) else float("nan"),
        "reason": "edge_frequency_pass",
        "strict_edge_pass": True,
    }
    return float(best["threshold"]), scan, decision


def _feature_importance(bundle: ModelBundle) -> pd.DataFrame:
    estimator = bundle.estimator
    values = None
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype="float64")
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.asarray(estimator.coef_, dtype="float64")).ravel()
    if values is None or len(values) != len(bundle.feature_columns):
        return pd.DataFrame({"feature": bundle.feature_columns, "importance": np.ones(len(bundle.feature_columns))})
    return pd.DataFrame({"feature": bundle.feature_columns, "importance": values}).sort_values("importance", ascending=False)


def _candidate_features(train: pd.DataFrame, min_feature_coverage: float) -> list[str]:
    all_features = infer_feature_columns(train, extra_prefixes=("cap_", "v3_", "peer_", "tech_", "asset_"))
    usable: list[str] = []
    for col in all_features:
        if col in DROP_FEATURE_EXACT:
            continue
        if any(str(col).startswith(prefix) for prefix in DROP_FEATURE_PREFIXES):
            continue
        series = pd.to_numeric(train[col], errors="coerce")
        coverage = float(series.notna().mean())
        if str(col).startswith("peer_") and coverage < 0.95:
            continue
        if coverage < float(min_feature_coverage):
            continue
        if int(series.nunique(dropna=True)) <= 1:
            continue
        usable.append(col)
    return usable


def _select_top_features(
    train: pd.DataFrame,
    *,
    side: str,
    candidates: list[str],
    model_kind: str,
    top_k_features: int,
    random_state: int,
) -> tuple[list[str], pd.DataFrame]:
    probe = train_classifier(
        train,
        feature_columns=candidates,
        model_kind=model_kind,
        threshold=0.5,
        side=side,
        random_state=random_state,
    )
    report = _feature_importance(probe)
    report = report.sort_values("importance", ascending=False).reset_index(drop=True)
    selected = report.head(int(top_k_features))["feature"].astype(str).tolist()
    report["selected"] = report["feature"].isin(selected)
    return selected, report


def _family_mask(frame: pd.DataFrame, family: str) -> pd.Series:
    return bsp_family_for_frame(frame).astype(str) == str(family)


def _pure_bsp2_mask(frame: pd.DataFrame) -> pd.Series:
    if "bsp_types_str" not in frame.columns:
        return pd.Series(False, index=frame.index)
    return frame["bsp_types_str"].astype(str).str.lower().eq("2")


def _attach_bsp2_quality_weights(
    frame: pd.DataFrame,
    *,
    hard_negative_weight: float,
    severe_negative_weight: float,
    efficient_positive_weight: float,
) -> pd.DataFrame:
    out = frame.copy()
    label = pd.to_numeric(out.get("label", np.nan), errors="coerce").fillna(0).astype(int)
    weights = pd.Series(1.0, index=out.index, dtype="float64")

    invalid_time = pd.to_numeric(out.get("label_bsp2_struct_invalid_time", np.nan), errors="coerce")
    third_time = pd.to_numeric(out.get("label_bsp2_confirm_time", np.nan), errors="coerce")
    mae = pd.to_numeric(out.get("label_bsp2_struct_mae", out.get("mae", np.nan)), errors="coerce")
    net_return = pd.to_numeric(out.get("net_return", np.nan), errors="coerce")

    hard_negative = (label <= 0) & invalid_time.notna()
    severe_negative = hard_negative & (
        (mae <= -0.02)
        | (net_return <= -0.01)
        | (invalid_time <= 8 * 15)
    )
    efficient_positive = (label > 0) & third_time.notna() & (third_time <= 8 * 15)

    weights.loc[hard_negative] = np.maximum(weights.loc[hard_negative], float(hard_negative_weight))
    weights.loc[severe_negative] = np.maximum(weights.loc[severe_negative], float(severe_negative_weight))
    weights.loc[efficient_positive] = np.maximum(weights.loc[efficient_positive], float(efficient_positive_weight))
    out["sample_weight"] = weights.clip(lower=0.05, upper=max(float(severe_negative_weight), 1.0)).astype("float64")
    return out


def train_btc_futures_models(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    train_start: Any | None = "2021-01-01",
    valid_start: Any | None = "2025-01-01",
    test_start: Any | None = "2026-01-01",
    test_end: Any | None = None,
    model_kind: str = "auto",
    min_feature_coverage: float = 0.05,
    top_k_features: int = 80,
    min_threshold_trades: int = 20,
    min_daily_trades: float = 0.08,
    min_precision: float = 0.52,
    min_trade_win_rate: float = 0.50,
    min_avg_net_return: float = 0.0,
    frequency_floor_threshold: float | None = None,
    min_family_train_rows: int = 200,
    min_family_valid_rows: int = 40,
    pure_bsp2_only: bool = True,
    require_origin_zs: bool = True,
    hard_negative_weight: float = 3.0,
    severe_negative_weight: float = 5.0,
    efficient_positive_weight: float = 1.2,
    primary_label_column: str = PRIMARY_LABEL_COLUMN,
) -> dict[str, Any]:
    dataset = pd.read_parquet(dataset_path)
    for col in ("exec_time", "entry_time", "exit_time"):
        dataset[col] = pd.to_datetime(dataset[col], utc=True, errors="coerce")
    primary_label_column = str(primary_label_column or PRIMARY_LABEL_COLUMN)
    if primary_label_column not in dataset.columns:
        raise ValueError(f"dataset missing primary label column: {primary_label_column}")
    dataset = dataset.dropna(subset=[primary_label_column, "exec_time", "entry_time"]).sort_values("exec_time").reset_index(drop=True)
    dataset["bsp_family"] = bsp_family_for_frame(dataset).astype(str)
    dataset = dataset.loc[dataset["bsp_family"].isin(BSP_FAMILIES)].copy()
    if pure_bsp2_only:
        dataset = dataset.loc[_pure_bsp2_mask(dataset)].copy()
    if require_origin_zs:
        origin_cols = {"chan_bsp2_origin_zs_low", "chan_bsp2_origin_zs_high"}
        missing = sorted(origin_cols - set(dataset.columns))
        if missing:
            raise ValueError(f"strict BSP2 training requires origin Zhongshu columns, missing: {missing}")
        origin_mask = (
            pd.to_numeric(dataset["chan_bsp2_origin_zs_low"], errors="coerce").notna()
            & pd.to_numeric(dataset["chan_bsp2_origin_zs_high"], errors="coerce").notna()
        )
        dataset = dataset.loc[origin_mask].copy()
    dataset["label"] = pd.to_numeric(dataset[primary_label_column], errors="coerce").astype(int)

    train_start_ts = parse_utc(train_start)
    valid_start_ts = parse_utc(valid_start)
    test_start_ts = parse_utc(test_start)
    test_end_ts = parse_utc(test_end)
    if valid_start_ts is None or test_start_ts is None:
        raise ValueError("valid_start and test_start are required")

    train_mask = dataset["exec_time"] < valid_start_ts
    if train_start_ts is not None:
        train_mask &= dataset["exec_time"] >= train_start_ts
    valid_mask = (dataset["exec_time"] >= valid_start_ts) & (dataset["exec_time"] < test_start_ts)
    test_mask = dataset["exec_time"] >= test_start_ts
    if test_end_ts is not None:
        test_mask &= dataset["exec_time"] < test_end_ts
    train = dataset.loc[train_mask].copy()
    valid = dataset.loc[valid_mask].copy()
    test = dataset.loc[test_mask].copy()
    if train.empty or valid.empty:
        raise ValueError(f"empty split: train={len(train)} valid={len(valid)} test={len(test)}")
    train = _attach_bsp2_quality_weights(
        train,
        hard_negative_weight=hard_negative_weight,
        severe_negative_weight=severe_negative_weight,
        efficient_positive_weight=efficient_positive_weight,
    )

    model_output = Path(model_dir)
    model_output.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = {
        "route": "btc_futures_v1",
        "dataset_path": str(dataset_path),
        "model_dir": str(model_output),
        "model_design": "BSP2-only split: second-class buy/sell models; label is third same-direction move before structural invalidation; no TP-first label",
        "bsp_families": list(BSP_FAMILIES),
        "primary_label": primary_label_column,
        "pure_bsp2_only": bool(pure_bsp2_only),
        "require_origin_zs": bool(require_origin_zs),
        "edge_threshold": {
            "enabled": True,
            "min_precision": float(min_precision),
            "min_trade_win_rate": float(min_trade_win_rate),
            "min_avg_net_return": float(min_avg_net_return),
            "frequency_floor_threshold": float(frequency_floor_threshold) if frequency_floor_threshold is not None else None,
            "per_submodel_min_daily_trades": float(min_daily_trades),
        },
        "sample_weighting": {
            "enabled": True,
            "hard_negative_definition": "label=0 and structural invalidation happened before third move",
            "severe_negative_definition": "hard negative with MAE <= -2%, net_return <= -1%, or invalidation within 8 bars",
            "efficient_positive_definition": "label=1 and third move confirms within 8 bars",
            "hard_negative_weight": float(hard_negative_weight),
            "severe_negative_weight": float(severe_negative_weight),
            "efficient_positive_weight": float(efficient_positive_weight),
        },
        "splits": {
            "train": {"start": str(train["exec_time"].min()), "end": str(train["exec_time"].max()), "rows": int(len(train))},
            "valid": {"start": str(valid["exec_time"].min()), "end": str(valid["exec_time"].max()), "rows": int(len(valid))},
            "test": {"start": str(test["exec_time"].min()), "end": str(test["exec_time"].max()), "rows": int(len(test))},
        },
        "models": {},
    }

    manifest_rows: list[dict[str, Any]] = []
    side_thresholds: dict[str, float] = {}
    for side, is_buy_value in (("buy", True), ("sell", False)):
        side_bundles: dict[str, ModelBundle] = {}
        metrics["models"][side] = {}
        side_valid_thresholds: list[float] = []
        for family in BSP_FAMILIES:
            key = f"{side}_bsp{family}"
            side_train = train.loc[(train["is_buy"].astype(bool) == is_buy_value) & _family_mask(train, family)].copy()
            side_valid = valid.loc[(valid["is_buy"].astype(bool) == is_buy_value) & _family_mask(valid, family)].copy()
            side_test = test.loc[(test["is_buy"].astype(bool) == is_buy_value) & _family_mask(test, family)].copy()
            info: dict[str, Any] = {
                "side": side,
                "bsp_family": family,
                "rows": {"train": int(len(side_train)), "valid": int(len(side_valid)), "test": int(len(side_test))},
                "enabled": False,
            }
            if len(side_train) < int(min_family_train_rows) or len(side_valid) < int(min_family_valid_rows):
                info["reason"] = "not_enough_rows"
                metrics["models"][side][family] = info
                continue
            if side_train["label"].nunique(dropna=True) < 2:
                info["reason"] = "one_class_target"
                metrics["models"][side][family] = info
                continue
            candidates = _candidate_features(side_train, min_feature_coverage)
            if not candidates:
                info["reason"] = "no_usable_features"
                metrics["models"][side][family] = info
                continue
            selected, report = _select_top_features(
                side_train,
                side=side,
                candidates=candidates,
                model_kind=model_kind,
                top_k_features=top_k_features,
                random_state=1000 + int(family) * 10 + (0 if side == "buy" else 1),
            )
            report.to_csv(model_output / f"{key}_feature_importance.csv", index=False)
            bundle = train_classifier(
                side_train,
                feature_columns=selected,
                model_kind=model_kind,
                threshold=0.99,
                side=side,
                random_state=2000 + int(family) * 10 + (0 if side == "buy" else 1),
            )
            valid_probability = bundle.predict_proba(side_valid)
            threshold, scan, decision = _threshold_scan(
                side_valid,
                valid_probability,
                min_trades=min_threshold_trades,
                min_daily_trades=min_daily_trades,
                min_precision=min_precision,
                min_trade_win_rate=min_trade_win_rate,
                min_avg_net_return=min_avg_net_return,
                frequency_floor_threshold=frequency_floor_threshold,
            )
            scan.to_csv(model_output / f"{key}_threshold_scan.csv", index=False)
            bundle.threshold = float(threshold)
            bundle.save(model_output / f"{key}_model.pkl")
            if decision.get("enabled"):
                side_bundles[family] = bundle
                side_valid_thresholds.append(float(threshold))
                info["enabled"] = True
            else:
                info["reason"] = str(decision.get("reason", "disabled"))
            test_probability = bundle.predict_proba(side_test) if not side_test.empty else np.empty(0, dtype="float64")
            info.update(
                {
                    "threshold": float(threshold),
                    "decision": decision,
                    "feature_count": int(len(selected)),
                    "selected_features": list(selected),
                    "train": _classification_metrics(side_train, bundle.predict_proba(side_train)),
                    "valid": _classification_metrics(side_valid, valid_probability),
                    "test": _classification_metrics(side_test, test_probability) if not side_test.empty else {},
                }
            )
            for rank, feature in enumerate(selected, start=1):
                manifest_rows.append({"side": side, "bsp_family": family, "rank": rank, "feature": feature})
            metrics["models"][side][family] = info

        aggregate = SideBspFamilyBundle(side=side, family_bundles=side_bundles, route="btc_futures_v1")
        aggregate.save(model_output / f"{side}_model.pkl")
        side_thresholds[side] = float(min(side_valid_thresholds)) if side_valid_thresholds else float(aggregate.disabled_threshold)

    pd.DataFrame(manifest_rows).to_csv(model_output / "selected_feature_manifest.csv", index=False)
    policy = {
        "route": "btc_futures_v1",
        "model_design": "BSP2-only long/short split",
        "default_thresholds": {"buy": float(side_thresholds.get("buy", 0.99)), "sell": float(side_thresholds.get("sell", 0.99))},
        "symbol_thresholds": {"BTCUSDT": {"buy": float(side_thresholds.get("buy", 0.99)), "sell": float(side_thresholds.get("sell", 0.99))}},
        "bsp_family_thresholds": {
            side: {
                family: float(metrics["models"][side].get(family, {}).get("threshold", 0.99))
                for family in BSP_FAMILIES
            }
            for side in ("buy", "sell")
        },
        "quality_filter": {"enabled": False},
        "gate_filter": {"enabled": False},
        "market_state_filter": {"enabled": False},
        "cost_aware_threshold": {"enabled": False},
        "edge_threshold": {
            "enabled": True,
            "min_precision": float(min_precision),
            "min_trade_win_rate": float(min_trade_win_rate),
            "min_avg_net_return": float(min_avg_net_return),
            "per_submodel_min_daily_trades": float(min_daily_trades),
            "frequency_floor_threshold": float(frequency_floor_threshold) if frequency_floor_threshold is not None else None,
            "note": "阈值优先按验证集真实净收益、交易胜率和 PF 选择；precision 只作为辅助约束。",
        },
        "sample_weighting": {
            "enabled": True,
            "hard_negative_weight": float(hard_negative_weight),
            "severe_negative_weight": float(severe_negative_weight),
            "efficient_positive_weight": float(efficient_positive_weight),
            "note": "仍然是同一个二买/二卖入场模型；只是让第三段前结构失败这类高成本负样本在训练中更重要。",
        },
        "label": {
            "primary": primary_label_column,
            "legacy_tp_first_removed": True,
            "max_holding_minutes": int(dataset["label_max_holding_minutes"].dropna().iloc[0]) if "label_max_holding_minutes" in dataset else 1440,
        },
    }
    (model_output / "threshold_policy.json").write_text(json.dumps(json_safe(policy), ensure_ascii=False, indent=2), encoding="utf-8")
    metrics["policy"] = policy
    (model_output / "metrics.json").write_text(json.dumps(json_safe(metrics), ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train BTC futures v1 BSP-family split models")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--train-start", default="2021-01-01")
    parser.add_argument("--valid-start", default="2025-01-01")
    parser.add_argument("--test-start", default="2026-01-01")
    parser.add_argument("--test-end", default="")
    parser.add_argument("--model-kind", default="auto")
    parser.add_argument("--min-feature-coverage", type=float, default=0.05)
    parser.add_argument("--top-k-features", type=int, default=80)
    parser.add_argument("--min-threshold-trades", type=int, default=20)
    parser.add_argument("--min-daily-trades", type=float, default=0.08)
    parser.add_argument("--min-precision", type=float, default=0.52)
    parser.add_argument("--min-trade-win-rate", type=float, default=0.50)
    parser.add_argument("--min-avg-net-return", type=float, default=0.0)
    parser.add_argument("--frequency-floor-threshold", type=float, default=None)
    parser.add_argument("--disable-frequency-floor", action="store_true")
    parser.add_argument("--min-family-train-rows", type=int, default=200)
    parser.add_argument("--min-family-valid-rows", type=int, default=40)
    parser.add_argument("--include-bsp2s-family", action="store_true")
    parser.add_argument("--allow-missing-origin-zs", action="store_true")
    parser.add_argument("--hard-negative-weight", type=float, default=3.0)
    parser.add_argument("--severe-negative-weight", type=float, default=5.0)
    parser.add_argument("--efficient-positive-weight", type=float, default=1.2)
    parser.add_argument("--primary-label-column", default=PRIMARY_LABEL_COLUMN)
    args = parser.parse_args()

    result = train_btc_futures_models(
        dataset_path=args.dataset,
        model_dir=args.model_dir,
        train_start=args.train_start,
        valid_start=args.valid_start,
        test_start=args.test_start,
        test_end=args.test_end or None,
        model_kind=args.model_kind,
        min_feature_coverage=args.min_feature_coverage,
        top_k_features=args.top_k_features,
        min_threshold_trades=args.min_threshold_trades,
        min_daily_trades=args.min_daily_trades,
        min_precision=args.min_precision,
        min_trade_win_rate=args.min_trade_win_rate,
        min_avg_net_return=args.min_avg_net_return,
        frequency_floor_threshold=None if args.disable_frequency_floor else args.frequency_floor_threshold,
        min_family_train_rows=args.min_family_train_rows,
        min_family_valid_rows=args.min_family_valid_rows,
        pure_bsp2_only=not args.include_bsp2s_family,
        require_origin_zs=not args.allow_missing_origin_zs,
        hard_negative_weight=args.hard_negative_weight,
        severe_negative_weight=args.severe_negative_weight,
        efficient_positive_weight=args.efficient_positive_weight,
        primary_label_column=args.primary_label_column,
    )
    print(json.dumps(json_safe(result["models"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
