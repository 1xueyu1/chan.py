from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ML.features import infer_feature_columns
from ML.model import _make_estimator
from ML.routes.btc_futures_v1.data import parse_utc
from ML.shared.json_utils import json_safe

from .bundle import PoolModelSpec, V3AlphaModelBundle
from .dataset import DEFAULT_DATASET_PATH, DEFAULT_MODEL_DIR, ROUTE_NAME, STRUCTURE_POOLS, build_btc_futures_v3_alpha_dataset


DROP_FEATURE_PREFIXES = ("cap_event_seq",)
DROP_FEATURE_COLUMNS = {
    "v3_structure_pool_id",
}


def _side_mask(frame: pd.DataFrame, side: str) -> pd.Series:
    mask = frame["is_buy"].astype(bool)
    return mask if side == "buy" else ~mask


def _profit_factor(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    wins = float(values.loc[values > 0].sum())
    losses = float(-values.loc[values < 0].sum())
    if losses <= 0:
        return float("inf") if wins > 0 else float("nan")
    return float(wins / losses)


def _classification_metrics(frame: pd.DataFrame, probability: np.ndarray) -> dict[str, Any]:
    y = frame["label"].astype(int).to_numpy()
    out: dict[str, Any] = {"rows": int(len(frame)), "positive_rate": float(np.mean(y)) if len(y) else float("nan")}
    if len(np.unique(y)) < 2:
        out.update({"auc": float("nan"), "ap": float("nan"), "brier": float("nan")})
        return out
    try:
        from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

        out["auc"] = float(roc_auc_score(y, probability))
        out["ap"] = float(average_precision_score(y, probability))
        out["brier"] = float(brier_score_loss(y, probability))
    except Exception:
        out.update({"auc": float("nan"), "ap": float("nan"), "brier": float("nan")})
    return out


def _metrics_for_selection(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {
            "trades": 0,
            "precision": float("nan"),
            "avg_net_return": float("nan"),
            "profit_factor": float("nan"),
        }
    returns = pd.to_numeric(frame["net_return"], errors="coerce")
    return {
        "trades": int(len(frame)),
        "precision": float(pd.to_numeric(frame["label"], errors="coerce").mean()),
        "avg_net_return": float(returns.mean()),
        "profit_factor": _profit_factor(returns),
    }


def _candidate_features(train: pd.DataFrame, min_feature_coverage: float) -> list[str]:
    all_features = infer_feature_columns(train, extra_prefixes=("cap_", "v3_"))
    cols: list[str] = []
    for col in all_features:
        if col in DROP_FEATURE_COLUMNS:
            continue
        if any(str(col).startswith(prefix) for prefix in DROP_FEATURE_PREFIXES):
            continue
        series = pd.to_numeric(train[col], errors="coerce")
        if float(series.notna().mean()) < float(min_feature_coverage):
            continue
        if int(series.nunique(dropna=True)) <= 1:
            continue
        cols.append(col)
    return cols


def _drift_score(train: pd.DataFrame, valid: pd.DataFrame, col: str) -> float:
    a = pd.to_numeric(train[col], errors="coerce")
    b = pd.to_numeric(valid[col], errors="coerce")
    if a.notna().sum() < 20 or b.notna().sum() < 20:
        return float("inf")
    return float(abs(b.mean() - a.mean()) / (a.std() + 1e-9))


def _feature_importance(estimator: object, features: list[str]) -> np.ndarray:
    if hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_, dtype="float64")
    elif hasattr(estimator, "coef_"):
        values = np.abs(np.asarray(estimator.coef_, dtype="float64")).ravel()
    else:
        values = np.ones(len(features), dtype="float64")
    if len(values) < len(features):
        values = np.pad(values, (0, len(features) - len(values)), constant_values=0.0)
    return values[: len(features)]


def _select_stable_features(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    *,
    model_kind: str,
    min_feature_coverage: float,
    max_drift_score: float,
    top_k: int,
) -> tuple[list[str], pd.DataFrame]:
    candidates = _candidate_features(train, min_feature_coverage)
    if len(candidates) < 10:
        raise ValueError("not enough candidate features")
    stable = [col for col in candidates if _drift_score(train, valid, col) <= float(max_drift_score)]
    if len(stable) < min(40, len(candidates)):
        stable = candidates
    probe = _fit_spec(train, stable, model_kind=model_kind, threshold=0.5)
    importance = _feature_importance(probe.estimator, stable)
    report = pd.DataFrame(
        {
            "feature": stable,
            "importance": importance,
            "drift_score": [_drift_score(train, valid, col) for col in stable],
            "coverage": [float(pd.to_numeric(train[col], errors="coerce").notna().mean()) for col in stable],
        }
    ).sort_values(["importance", "coverage"], ascending=[False, False])
    selected = report.head(int(top_k))["feature"].astype(str).tolist()
    return selected, report


def _fit_spec(frame: pd.DataFrame, features: list[str], *, model_kind: str, threshold: float, random_state: int = 42) -> PoolModelSpec:
    train = frame.dropna(subset=["label"]).copy()
    if train.empty:
        raise ValueError("empty training frame")
    if train["label"].nunique() < 2:
        raise ValueError("training frame has one class")
    x = train.reindex(columns=features).replace([np.inf, -np.inf], np.nan)
    fill_values = x.median(numeric_only=True).fillna(0.0).to_dict()
    x = x.fillna(fill_values).fillna(0.0)
    y = train["label"].astype(int)
    estimator = _make_estimator(model_kind, random_state=random_state)
    estimator.fit(x, y)
    return PoolModelSpec(
        estimator=estimator,
        feature_columns=list(features),
        fill_values={str(k): float(v) for k, v in fill_values.items()},
        threshold=float(threshold),
    )


def _predict_spec(frame: pd.DataFrame, spec: PoolModelSpec) -> np.ndarray:
    if frame.empty:
        return np.empty(0, dtype="float64")
    x = frame.reindex(columns=spec.feature_columns).replace([np.inf, -np.inf], np.nan)
    x = x.fillna(spec.fill_values).fillna(0.0)
    if hasattr(spec.estimator, "predict_proba"):
        return np.asarray(spec.estimator.predict_proba(x)[:, 1], dtype="float64")
    score = np.asarray(spec.estimator.decision_function(x), dtype="float64")
    return 1.0 / (1.0 + np.exp(-score))


def _scan_thresholds(
    frame: pd.DataFrame,
    *,
    probability_col: str = "probability",
    min_trades: int,
    min_daily_trades: float,
    min_year_trades: int,
    min_precision: float,
    min_avg_net_return: float,
    min_profit_factor: float,
    threshold_start: float = 0.45,
    threshold_end: float = 0.92,
    threshold_step: float = 0.01,
    holdout_last_year: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if frame.empty or probability_col not in frame.columns:
        return {"enabled": False, "reason": "empty"}, pd.DataFrame()
    frame_year = pd.to_datetime(frame["entry_time"], utc=True, errors="coerce").dt.year
    years = sorted(int(year) for year in frame_year.dropna().unique())
    scan_source = frame
    confirm_source = pd.DataFrame()
    confirm_year = None
    if bool(holdout_last_year) and len(years) >= 2:
        confirm_year = years[-1]
        scan_source = frame.loc[frame_year != confirm_year].copy()
        confirm_source = frame.loc[frame_year == confirm_year].copy()
        if scan_source.empty or confirm_source.empty:
            scan_source = frame
            confirm_source = pd.DataFrame()
            confirm_year = None

    times = pd.to_datetime(scan_source["entry_time"], utc=True, errors="coerce")
    days = max(1.0, (times.max() - times.min()).total_seconds() / 86400.0)
    effective_min_trades = max(int(min_year_trades), int(np.ceil(float(min_trades) / 2.0))) if confirm_year is not None else int(min_trades)
    rows: list[dict[str, Any]] = []
    for threshold in np.round(np.arange(float(threshold_start), float(threshold_end) + 1e-9, float(threshold_step)), 2):
        selected = pd.to_numeric(scan_source[probability_col], errors="coerce") >= float(threshold)
        sub = scan_source.loc[selected]
        metrics = _metrics_for_selection(sub)
        rows.append(
            {
                "threshold": float(threshold),
                "trades": int(metrics["trades"]),
                "daily_trades": float(metrics["trades"] / days),
                "precision": metrics["precision"],
                "avg_net_return": metrics["avg_net_return"],
                "profit_factor": metrics["profit_factor"],
            }
        )
    scan = pd.DataFrame(rows)
    ok = scan.loc[
        (scan["trades"] >= int(effective_min_trades))
        & (scan["daily_trades"] >= float(min_daily_trades))
        & (scan["precision"] >= float(min_precision))
        & (scan["avg_net_return"] > float(min_avg_net_return))
        & (scan["profit_factor"] >= float(min_profit_factor))
    ].copy()

    if confirm_year is not None and not ok.empty:
        confirmed_rows = []
        for _, candidate in ok.iterrows():
            selected = pd.to_numeric(confirm_source[probability_col], errors="coerce") >= float(candidate["threshold"])
            confirm_metrics = _metrics_for_selection(confirm_source.loc[selected])
            if int(confirm_metrics["trades"]) < int(min_year_trades):
                continue
            if float(confirm_metrics["precision"]) < float(min_precision):
                continue
            if float(confirm_metrics["avg_net_return"]) <= float(min_avg_net_return):
                continue
            if float(confirm_metrics["profit_factor"]) < float(min_profit_factor):
                continue
            rec = candidate.to_dict()
            rec["selection_trades"] = int(candidate["trades"])
            rec["selection_precision"] = float(candidate["precision"])
            rec["selection_avg_net_return"] = float(candidate["avg_net_return"])
            rec["selection_profit_factor"] = float(candidate["profit_factor"])
            rec["confirmation_year"] = int(confirm_year)
            rec["trades"] = int(confirm_metrics["trades"])
            rec["daily_trades"] = float(confirm_metrics["trades"] / max(1.0, (confirm_source["entry_time"].max() - confirm_source["entry_time"].min()).total_seconds() / 86400.0))
            rec["precision"] = float(confirm_metrics["precision"])
            rec["avg_net_return"] = float(confirm_metrics["avg_net_return"])
            rec["profit_factor"] = float(confirm_metrics["profit_factor"])
            confirmed_rows.append(rec)
        ok = pd.DataFrame(confirmed_rows)

    if len(years) >= 2 and not ok.empty and int(min_year_trades) > 0:
        stable_rows = []
        for _, candidate in ok.iterrows():
            selected = pd.to_numeric(frame[probability_col], errors="coerce") >= float(candidate["threshold"])
            yearly_records = []
            stable = True
            for year in years:
                sub = frame.loc[selected & (frame_year == year)]
                metrics = _metrics_for_selection(sub)
                yearly_records.append({"year": int(year), **metrics})
                if int(metrics["trades"]) < int(min_year_trades):
                    stable = False
                    break
                if float(metrics["precision"]) < float(min_precision):
                    stable = False
                    break
                if float(metrics["avg_net_return"]) <= float(min_avg_net_return):
                    stable = False
                    break
                if float(metrics["profit_factor"]) < float(min_profit_factor):
                    stable = False
                    break
            if stable:
                rec = candidate.to_dict()
                rec["yearly_records"] = json.dumps(yearly_records, ensure_ascii=False)
                stable_rows.append(rec)
        ok = pd.DataFrame(stable_rows)

    if ok.empty:
        return {"enabled": False, "reason": "no_cost_aware_threshold"}, scan
    ok["score"] = (
        ok["avg_net_return"].fillna(-1.0) * 1500.0
        + ok["precision"].fillna(0.0) * 2.0
        + np.log1p(ok["trades"].astype(float)) * 0.04
        - ok["threshold"].astype(float) * 0.01
    )
    best = ok.sort_values(["score", "precision", "trades"], ascending=[False, False, False]).iloc[0]
    return {
        "enabled": True,
        "threshold": float(best["threshold"]),
        "trades": int(best["trades"]),
        "daily_trades": float(best["daily_trades"]),
        "precision": float(best["precision"]),
        "avg_net_return": float(best["avg_net_return"]),
        "profit_factor": float(best["profit_factor"]),
        "reason": "cost_aware_pass",
    }, scan


def _build_policy(
    side_decisions: dict[str, dict[str, Any]],
    pool_decisions: dict[str, dict[str, dict[str, Any]]],
    *,
    min_precision: float,
    min_avg_net_return: float,
    min_profit_factor: float,
    min_side_auc: float,
    min_pool_auc: float,
) -> dict[str, Any]:
    default_thresholds = {}
    for side in ("buy", "sell"):
        decision = side_decisions.get(side, {})
        default_thresholds[side] = float(decision.get("threshold", 0.99)) if decision.get("enabled") else 0.99

    pool_groups: dict[str, dict[str, Any]] = {}
    for side, decisions in pool_decisions.items():
        for pool in STRUCTURE_POOLS:
            decision = decisions.get(pool) or {"enabled": False, "reason": "not_seen"}
            key = f"{pool}|{side}"
            if not decision.get("enabled", False):
                pool_groups[key] = {"enabled": False, "reason": decision.get("reason", "disabled")}
                continue
            pool_groups[key] = {
                "enabled": True,
                "threshold": float(decision["threshold"]),
                "validation_trades": int(decision.get("trades", 0)),
                "validation_precision": float(decision.get("precision", np.nan)),
                "validation_avg_net_return": float(decision.get("avg_net_return", np.nan)),
                "validation_profit_factor": float(decision.get("profit_factor", np.nan)),
                "model_source": str(decision.get("model_source", "global")),
            }

    return {
        "route": ROUTE_NAME,
        "default_thresholds": default_thresholds,
        "symbol_thresholds": {"BTCUSDT": default_thresholds},
        "structure_pool_filter": {
            "enabled": True,
            "require_configured_pool": True,
            "threshold_mode": "override",
            "fallback_order": ["pool_side"],
            "groups": {"pool_side": pool_groups},
        },
        "quality_filter": {"enabled": False},
        "gate_filter": {"enabled": False},
        "position_sizing": {"enabled": False},
        "cost_aware_constraints": {
            "min_precision": float(min_precision),
            "min_avg_net_return": float(min_avg_net_return),
            "min_profit_factor": float(min_profit_factor),
            "min_side_auc": float(min_side_auc),
            "min_pool_auc": float(min_pool_auc),
        },
    }


def train_btc_futures_v3_alpha_models(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    train_start: Any | None = "2021-01-01",
    valid_start: Any | None = "2024-01-01",
    test_start: Any | None = "2026-01-01",
    test_end: Any | None = None,
    model_kind: str = "auto",
    min_feature_coverage: float = 0.08,
    max_drift_score: float = 2.8,
    top_k_features: int = 180,
    top_k_pool_features: int = 90,
    min_precision: float = 0.60,
    min_avg_net_return: float = 0.0,
    min_profit_factor: float = 1.20,
    min_side_auc: float = 0.54,
    min_pool_auc: float = 0.54,
    min_side_trades: int = 60,
    min_side_daily_trades: float = 0.06,
    min_pool_train_rows: int = 160,
    min_pool_valid_rows: int = 80,
    min_pool_trades: int = 20,
    min_pool_daily_trades: float = 0.015,
    min_year_trades: int = 6,
) -> dict[str, Any]:
    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        build_btc_futures_v3_alpha_dataset(output_path=dataset_file)
    dataset = pd.read_parquet(dataset_file)
    for col in ("exec_time", "entry_time", "exit_time"):
        dataset[col] = pd.to_datetime(dataset[col], utc=True, errors="coerce")
    dataset = dataset.dropna(subset=["label", "exec_time", "entry_time", "exit_time"]).sort_values("exec_time").reset_index(drop=True)
    if not bool((dataset["entry_time"] >= dataset["exec_time"] + pd.Timedelta(minutes=15)).all()):
        raise RuntimeError("lookahead audit failed before training")

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

    out_dir = Path(model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = {
        "route": ROUTE_NAME,
        "dataset_path": str(dataset_file),
        "model_dir": str(out_dir),
        "splits": {
            "train": {"start": str(train["exec_time"].min()), "end": str(train["exec_time"].max()), "rows": int(len(train))},
            "valid": {"start": str(valid["exec_time"].min()), "end": str(valid["exec_time"].max()), "rows": int(len(valid))},
            "test": {"start": str(test["exec_time"].min()), "end": str(test["exec_time"].max()), "rows": int(len(test))},
        },
        "models": {},
        "pools": {},
    }

    side_decisions: dict[str, dict[str, Any]] = {}
    pool_decisions: dict[str, dict[str, dict[str, Any]]] = {"buy": {}, "sell": {}}

    for side in ("buy", "sell"):
        side_train = train.loc[_side_mask(train, side)].copy()
        side_valid = valid.loc[_side_mask(valid, side)].copy()
        side_test = test.loc[_side_mask(test, side)].copy()
        features, feature_report = _select_stable_features(
            side_train,
            side_valid,
            model_kind=model_kind,
            min_feature_coverage=min_feature_coverage,
            max_drift_score=max_drift_score,
            top_k=top_k_features,
        )
        global_spec = _fit_spec(side_train, features, model_kind=model_kind, threshold=0.5, random_state=47 if side == "buy" else 53)
        valid_global_prob = _predict_spec(side_valid, global_spec)
        test_global_prob = _predict_spec(side_test, global_spec) if not side_test.empty else np.empty(0)
        global_train_metrics = _classification_metrics(side_train, _predict_spec(side_train, global_spec))
        global_valid_metrics = _classification_metrics(side_valid, valid_global_prob)
        global_test_metrics = _classification_metrics(side_test, test_global_prob) if not side_test.empty else {}
        global_scan_frame = side_valid.copy()
        global_scan_frame["probability"] = valid_global_prob
        side_decision, side_scan = _scan_thresholds(
            global_scan_frame,
            min_trades=min_side_trades,
            min_daily_trades=min_side_daily_trades,
            min_year_trades=max(min_year_trades, int(min_side_trades // 6)),
            min_precision=min_precision,
            min_avg_net_return=min_avg_net_return,
            min_profit_factor=min_profit_factor,
        )
        if side_decision.get("enabled") and float(global_valid_metrics.get("auc", np.nan)) < float(min_side_auc):
            side_decision = {"enabled": False, "reason": "valid_auc_below_min", "valid_auc": global_valid_metrics.get("auc")}
        if side_decision.get("enabled"):
            global_spec.threshold = float(side_decision["threshold"])
        else:
            global_spec.threshold = 0.99
        side_decisions[side] = side_decision

        pool_specs: dict[str, PoolModelSpec] = {}
        pool_records: list[dict[str, Any]] = []
        for pool in STRUCTURE_POOLS:
            pool_train = side_train.loc[side_train["v3_structure_pool"].astype(str) == pool].copy()
            pool_valid = side_valid.loc[side_valid["v3_structure_pool"].astype(str) == pool].copy()
            if len(pool_valid):
                pool_valid = pool_valid.copy()
                pool_valid["probability"] = valid_global_prob[(side_valid["v3_structure_pool"].astype(str) == pool).to_numpy()]
            global_pool_decision, global_pool_scan = _scan_thresholds(
                pool_valid,
                min_trades=min_pool_trades,
                min_daily_trades=min_pool_daily_trades,
                min_year_trades=min_year_trades,
                min_precision=min_precision,
                min_avg_net_return=min_avg_net_return,
                min_profit_factor=min_profit_factor,
            )
            global_pool_decision["model_source"] = "global"
            if global_pool_decision.get("enabled") and float(global_valid_metrics.get("auc", np.nan)) < float(min_side_auc):
                global_pool_decision = {
                    "enabled": False,
                    "reason": "global_valid_auc_below_min",
                    "valid_auc": global_valid_metrics.get("auc"),
                    "model_source": "global",
                }
            best_decision = global_pool_decision
            best_scan = global_pool_scan
            pool_model_metrics: dict[str, Any] = {}

            can_train_pool = (
                len(pool_train) >= int(min_pool_train_rows)
                and len(pool_valid) >= int(min_pool_valid_rows)
                and pool_train["label"].nunique() >= 2
                and pool_valid["label"].nunique() >= 2
            )
            if can_train_pool:
                try:
                    pool_features, pool_feature_report = _select_stable_features(
                        pool_train,
                        pool_valid,
                        model_kind=model_kind,
                        min_feature_coverage=max(0.12, min_feature_coverage),
                        max_drift_score=max_drift_score,
                        top_k=top_k_pool_features,
                    )
                    pool_spec = _fit_spec(pool_train, pool_features, model_kind=model_kind, threshold=0.5, random_state=101)
                    pool_prob = _predict_spec(pool_valid, pool_spec)
                    pool_scan_frame = pool_valid.copy()
                    pool_scan_frame["probability"] = pool_prob
                    pool_decision, pool_scan = _scan_thresholds(
                        pool_scan_frame,
                        min_trades=min_pool_trades,
                        min_daily_trades=min_pool_daily_trades,
                        min_year_trades=min_year_trades,
                        min_precision=min_precision,
                        min_avg_net_return=min_avg_net_return,
                        min_profit_factor=min_profit_factor,
                    )
                    pool_decision["model_source"] = "pool"
                    pool_model_metrics = {
                        "train": _classification_metrics(pool_train, _predict_spec(pool_train, pool_spec)),
                        "valid": _classification_metrics(pool_valid, pool_prob),
                        "feature_count": int(len(pool_features)),
                    }
                    if pool_decision.get("enabled") and float((pool_model_metrics.get("valid") or {}).get("auc", np.nan)) < float(min_pool_auc):
                        pool_decision = {
                            "enabled": False,
                            "reason": "pool_valid_auc_below_min",
                            "valid_auc": (pool_model_metrics.get("valid") or {}).get("auc"),
                            "model_source": "pool",
                        }
                    pool_feature_report.to_csv(out_dir / f"{side}_{pool}_feature_report.csv", index=False)
                    pool_scan.to_csv(out_dir / f"{side}_{pool}_pool_model_threshold_scan.csv", index=False)
                    if pool_decision.get("enabled"):
                        global_score = float(global_pool_decision.get("avg_net_return", -1.0)) if global_pool_decision.get("enabled") else -1.0
                        pool_score = float(pool_decision.get("avg_net_return", -1.0))
                        if (not global_pool_decision.get("enabled")) or pool_score >= global_score:
                            pool_spec.threshold = float(pool_decision["threshold"])
                            pool_spec.validation_metrics = dict(pool_decision)
                            pool_specs[pool] = pool_spec
                            best_decision = pool_decision
                            best_scan = pool_scan
                except Exception as exc:
                    pool_model_metrics = {"error": str(exc)}

            if best_decision.get("enabled"):
                pool_decisions[side][pool] = best_decision
            else:
                pool_decisions[side][pool] = {"enabled": False, "reason": best_decision.get("reason", "disabled")}
            global_pool_scan.to_csv(out_dir / f"{side}_{pool}_global_pool_threshold_scan.csv", index=False)
            if best_scan is not global_pool_scan:
                best_scan.to_csv(out_dir / f"{side}_{pool}_selected_threshold_scan.csv", index=False)
            pool_records.append(
                {
                    "side": side,
                    "pool": pool,
                    "train_rows": int(len(pool_train)),
                    "valid_rows": int(len(pool_valid)),
                    "selected_enabled": bool(best_decision.get("enabled", False)),
                    "selected_model_source": str(best_decision.get("model_source", "global")),
                    "selected_threshold": float(best_decision.get("threshold", np.nan)),
                    "selected_trades": int(best_decision.get("trades", 0)),
                    "selected_precision": float(best_decision.get("precision", np.nan)),
                    "selected_avg_net_return": float(best_decision.get("avg_net_return", np.nan)),
                    "selected_profit_factor": float(best_decision.get("profit_factor", np.nan)),
                    "pool_model_metrics": json.dumps(pool_model_metrics, ensure_ascii=False),
                }
            )

        bundle = V3AlphaModelBundle(side=side, global_spec=global_spec, pool_specs=pool_specs)
        bundle.save(out_dir / f"{side}_model.pkl")
        feature_report.to_csv(out_dir / f"{side}_global_feature_report.csv", index=False)
        side_scan.to_csv(out_dir / f"{side}_global_threshold_scan.csv", index=False)
        pd.DataFrame(pool_records).to_csv(out_dir / f"{side}_pool_decisions.csv", index=False)

        metrics["models"][side] = {
            "enabled": bool(side_decision.get("enabled", False)),
            "global_threshold": float(global_spec.threshold),
            "global_decision": side_decision,
            "feature_count": int(len(features)),
            "pool_model_count": int(len(pool_specs)),
            "min_side_auc": float(min_side_auc),
            "train": global_train_metrics,
            "valid": global_valid_metrics,
            "test": global_test_metrics,
        }
        metrics["pools"][side] = pool_records

    policy = _build_policy(
        side_decisions,
        pool_decisions,
        min_precision=min_precision,
        min_avg_net_return=min_avg_net_return,
        min_profit_factor=min_profit_factor,
        min_side_auc=min_side_auc,
        min_pool_auc=min_pool_auc,
    )
    (out_dir / "threshold_policy.json").write_text(json.dumps(json_safe(policy), ensure_ascii=False, indent=2), encoding="utf-8")
    metrics["policy"] = policy
    (out_dir / "metrics.json").write_text(json.dumps(json_safe(metrics), ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train BTC futures v3 alpha models")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--train-start", default="2021-01-01")
    parser.add_argument("--valid-start", default="2024-01-01")
    parser.add_argument("--test-start", default="2026-01-01")
    parser.add_argument("--test-end", default="")
    parser.add_argument("--model-kind", default="auto")
    parser.add_argument("--min-precision", type=float, default=0.60)
    parser.add_argument("--min-profit-factor", type=float, default=1.20)
    parser.add_argument("--min-side-auc", type=float, default=0.54)
    parser.add_argument("--min-pool-auc", type=float, default=0.54)
    parser.add_argument("--top-k-features", type=int, default=180)
    parser.add_argument("--top-k-pool-features", type=int, default=90)
    args = parser.parse_args()
    result = train_btc_futures_v3_alpha_models(
        dataset_path=args.dataset,
        model_dir=args.model_dir,
        train_start=args.train_start,
        valid_start=args.valid_start,
        test_start=args.test_start,
        test_end=args.test_end or None,
        model_kind=args.model_kind,
        min_precision=args.min_precision,
        min_profit_factor=args.min_profit_factor,
        min_side_auc=args.min_side_auc,
        min_pool_auc=args.min_pool_auc,
        top_k_features=args.top_k_features,
        top_k_pool_features=args.top_k_pool_features,
    )
    print(json.dumps(result["models"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
