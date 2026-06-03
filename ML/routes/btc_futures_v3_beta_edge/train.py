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

from .bundle import BetaModelSpec, StableRankClassifier, V3BetaEdgeModelBundle
from .dataset import DEFAULT_DATASET_PATH, DEFAULT_MODEL_DIR, ROUTE_NAME, build_btc_futures_v3_beta_edge_dataset


TARGET_COLUMNS = {
    "tp_first": "label_tp_first",
}

DROP_FEATURE_EXACT = {
    "label",
    "label_tp_first",
    "gross_return",
    "net_return",
    "mfe",
    "mae",
    "entry_price",
    "exit_price",
    "holding_minutes",
    "entry_bar_idx",
}
DROP_FEATURE_PREFIXES = ("cap_event_seq",)


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


def _classification_metrics(frame: pd.DataFrame, target: str, probability: np.ndarray) -> dict[str, Any]:
    y = pd.to_numeric(frame[target], errors="coerce").fillna(0).astype(int).to_numpy()
    out: dict[str, Any] = {
        "rows": int(len(frame)),
        "positive_rate": float(np.mean(y)) if len(y) else float("nan"),
    }
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


def _feature_auc(frame: pd.DataFrame, col: str, target: str, sign: float = 1.0) -> float:
    if frame.empty or col not in frame.columns or target not in frame.columns:
        return float("nan")
    x = pd.to_numeric(frame[col], errors="coerce") * float(sign)
    y = pd.to_numeric(frame[target], errors="coerce")
    mask = x.notna() & y.notna()
    if int(mask.sum()) < 100:
        return float("nan")
    if int(y.loc[mask].nunique()) < 2 or int(x.loc[mask].nunique()) < 2:
        return float("nan")
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(y.loc[mask].astype(int), x.loc[mask]))
    except Exception:
        return float("nan")


def _candidate_features(train: pd.DataFrame, min_feature_coverage: float) -> list[str]:
    all_features = infer_feature_columns(train, extra_prefixes=("cap_", "v3_", "beta_"))
    cols: list[str] = []
    for col in all_features:
        if col in DROP_FEATURE_EXACT:
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
    if col not in train.columns or col not in valid.columns:
        return float("inf")
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


def _fit_stable_rank_spec(
    frame: pd.DataFrame,
    features: list[str],
    target: str,
    *,
    feature_report: pd.DataFrame,
) -> BetaModelSpec:
    data = frame.dropna(subset=[target]).copy()
    if data.empty:
        raise ValueError(f"empty training frame for target={target}")
    if data[target].nunique() < 2:
        raise ValueError(f"target has one class: {target}")
    x = data.reindex(columns=features).replace([np.inf, -np.inf], np.nan)
    fill_values = x.median(numeric_only=True).fillna(0.0).to_dict()
    x = x.fillna(fill_values).fillna(0.0)

    report = feature_report.copy()
    if "feature" not in report.columns:
        report = pd.DataFrame({"feature": features})
    report = report.drop_duplicates(subset=["feature"], keep="first").set_index("feature", drop=False)
    stats: list[dict[str, float | str]] = []
    for feature in features:
        row = report.loc[feature] if feature in report.index else pd.Series(dtype="object")
        sign = float(row.get("sign", row.get("direction", 1.0)))
        if not np.isfinite(sign) or sign == 0:
            sign = 1.0
        raw_observed = pd.to_numeric(data[feature], errors="coerce") if feature in data.columns else pd.Series(dtype="float64")
        signed = raw_observed * sign
        weight = row.get("temporal_score", row.get("score", row.get("importance", 1.0)))
        try:
            weight_value = float(weight)
        except Exception:
            weight_value = 1.0
        signed_clean = signed.replace([np.inf, -np.inf], np.nan).dropna()
        signed_mean = float(signed_clean.mean()) if not signed_clean.empty else 0.0
        signed_std = float(signed_clean.std()) if len(signed_clean) > 1 else 1.0
        if not np.isfinite(signed_mean):
            signed_mean = 0.0
        if not np.isfinite(signed_std) or signed_std <= 0:
            signed_std = 1.0
        stats.append(
            {
                "feature": str(feature),
                "sign": float(sign),
                "fill_value": float(fill_values.get(feature, 0.0)),
                "mean": float(signed_mean),
                "std": float(signed_std + 1e-9),
                "weight": float(max(weight_value, 1e-6)),
            }
        )
    return BetaModelSpec(
        estimator=StableRankClassifier(feature_stats=stats),
        feature_columns=list(features),
        fill_values={str(k): float(v) for k, v in fill_values.items()},
        target_column=target,
    )


def _fit_spec(
    frame: pd.DataFrame,
    features: list[str],
    target: str,
    *,
    model_kind: str,
    random_state: int,
    feature_report: pd.DataFrame | None = None,
) -> BetaModelSpec:
    if model_kind in {"stable_rank", "rank"}:
        return _fit_stable_rank_spec(
            frame,
            features,
            target,
            feature_report=feature_report if feature_report is not None else pd.DataFrame(),
        )
    data = frame.dropna(subset=[target]).copy()
    if data.empty:
        raise ValueError(f"empty training frame for target={target}")
    if data[target].nunique() < 2:
        raise ValueError(f"target has one class: {target}")
    x = data.reindex(columns=features).replace([np.inf, -np.inf], np.nan)
    fill_values = x.median(numeric_only=True).fillna(0.0).to_dict()
    x = x.fillna(fill_values).fillna(0.0)
    y = pd.to_numeric(data[target], errors="coerce").fillna(0).astype(int)
    estimator = _make_estimator(model_kind, random_state=random_state)
    estimator.fit(x, y)
    return BetaModelSpec(
        estimator=estimator,
        feature_columns=list(features),
        fill_values={str(k): float(v) for k, v in fill_values.items()},
        target_column=target,
    )


def _predict_spec(frame: pd.DataFrame, spec: BetaModelSpec) -> np.ndarray:
    x = frame.reindex(columns=spec.feature_columns).replace([np.inf, -np.inf], np.nan)
    x = x.fillna(spec.fill_values).fillna(0.0)
    if hasattr(spec.estimator, "predict_proba"):
        return np.asarray(spec.estimator.predict_proba(x)[:, 1], dtype="float64")
    score = np.asarray(spec.estimator.decision_function(x), dtype="float64")
    return 1.0 / (1.0 + np.exp(-score))


def _select_features(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    confirm: pd.DataFrame | None = None,
    *,
    target: str,
    model_kind: str,
    min_feature_coverage: float,
    max_drift_score: float,
    top_k: int,
    selection_mode: str = "model_importance",
    min_feature_release_auc: float = 0.515,
) -> tuple[list[str], pd.DataFrame]:
    candidates = _candidate_features(train, min_feature_coverage)
    if not candidates:
        raise ValueError(f"no feature candidates for target={target}")

    if selection_mode in {"temporal_stability", "stable"} or model_kind in {"stable_rank", "rank"}:
        rows: list[dict[str, Any]] = []
        for col in candidates:
            raw_train_auc = _feature_auc(train, col, target, sign=1.0)
            if not np.isfinite(raw_train_auc):
                continue
            sign = 1.0 if raw_train_auc >= 0.5 else -1.0
            train_auc = raw_train_auc if sign > 0 else 1.0 - raw_train_auc
            valid_auc = _feature_auc(valid, col, target, sign=sign)
            confirm_auc = _feature_auc(confirm, col, target, sign=sign) if confirm is not None else float("nan")
            release_values = [value for value in (valid_auc, confirm_auc) if np.isfinite(value)]
            if not release_values:
                continue
            release_min_auc = float(min(release_values))
            release_mean_auc = float(np.mean(release_values))
            drift_values = [_drift_score(train, valid, col)]
            if confirm is not None:
                drift_values.append(_drift_score(train, confirm, col))
            drift = float(max(drift_values))
            coverage = float(pd.to_numeric(train[col], errors="coerce").notna().mean())
            temporal_score = (
                max(0.0, release_min_auc - 0.5) * 4.0
                + max(0.0, release_mean_auc - 0.5) * 2.0
                + max(0.0, train_auc - 0.5) * 0.5
                - min(drift, 20.0) * 0.003
                + min(coverage, 1.0) * 0.01
            )
            rows.append(
                {
                    "feature": col,
                    "sign": sign,
                    "train_auc": train_auc,
                    "valid_auc": valid_auc,
                    "confirm_auc": confirm_auc,
                    "release_min_auc": release_min_auc,
                    "release_mean_auc": release_mean_auc,
                    "drift_score": drift,
                    "coverage": coverage,
                    "temporal_score": temporal_score,
                }
            )
        report = pd.DataFrame(rows)
        if report.empty:
            raise ValueError(f"no temporal feature report rows for target={target}")
        filtered = report.loc[
            (report["train_auc"] >= 0.501)
            & (report["release_min_auc"] >= float(min_feature_release_auc))
            & (report["drift_score"] <= max(float(max_drift_score), 6.0))
        ].copy()
        if filtered.empty:
            filtered = report.copy()
        filtered = filtered.sort_values(["temporal_score", "release_min_auc", "coverage"], ascending=[False, False, False])
        selected = filtered.head(int(top_k))["feature"].astype(str).tolist()
        report["selected"] = report["feature"].isin(selected)
        return selected, report.sort_values(["selected", "temporal_score"], ascending=[False, False])

    stable = [col for col in candidates if _drift_score(train, valid, col) <= max_drift_score]
    if len(stable) < min(50, len(candidates)):
        stable = candidates
    probe = _fit_spec(train, stable, target, model_kind=model_kind, random_state=31)
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
    report["selected"] = report["feature"].isin(selected)
    return selected, report


def _score_bundle(bundle: V3BetaEdgeModelBundle, frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    components = bundle.predict_components(out)
    out["tp_first_probability"] = components["tp_first_probability"].to_numpy()
    out["probability"] = components["probability"].to_numpy()
    return out


def _selected_candidates(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    selected = frame.loc[pd.to_numeric(frame["probability"], errors="coerce") >= float(threshold)].copy()
    if selected.empty:
        return selected
    selected = selected.sort_values(["beta_event_id", "entry_time", "probability"], ascending=[True, True, False])
    return selected.drop_duplicates(subset=["beta_event_id"], keep="first").reset_index(drop=True)


def _selection_metrics(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {"trades": 0, "precision": float("nan"), "avg_net_return": float("nan"), "profit_factor": float("nan")}
    returns = pd.to_numeric(frame["net_return"], errors="coerce")
    return {
        "trades": int(len(frame)),
        "precision": float(pd.to_numeric(frame["label_tp_first"], errors="coerce").mean()),
        "avg_net_return": float(returns.mean()),
        "profit_factor": _profit_factor(returns),
    }


def _scan_thresholds(
    valid_scored: pd.DataFrame,
    confirm_scored: pd.DataFrame,
    *,
    min_trades: int,
    min_daily_trades: float,
    min_precision: float,
    min_avg_net_return: float,
    min_profit_factor: float,
) -> tuple[dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    valid_days = max(
        1.0,
        (pd.to_datetime(valid_scored["entry_time"], utc=True).max() - pd.to_datetime(valid_scored["entry_time"], utc=True).min()).total_seconds()
        / 86400.0,
    )
    confirm_days = max(
        1.0,
        (
            pd.to_datetime(confirm_scored["entry_time"], utc=True).max()
            - pd.to_datetime(confirm_scored["entry_time"], utc=True).min()
        ).total_seconds()
        / 86400.0,
    )
    for threshold in np.round(np.arange(0.45, 0.951, 0.01), 2):
        v = _selected_candidates(valid_scored, float(threshold))
        c = _selected_candidates(confirm_scored, float(threshold))
        vm = _selection_metrics(v)
        cm = _selection_metrics(c)
        rows.append(
            {
                "threshold": float(threshold),
                "valid_trades": int(vm["trades"]),
                "valid_daily_trades": float(vm["trades"] / valid_days),
                "valid_precision": vm["precision"],
                "valid_avg_net_return": vm["avg_net_return"],
                "valid_profit_factor": vm["profit_factor"],
                "confirm_trades": int(cm["trades"]),
                "confirm_daily_trades": float(cm["trades"] / confirm_days),
                "confirm_precision": cm["precision"],
                "confirm_avg_net_return": cm["avg_net_return"],
                "confirm_profit_factor": cm["profit_factor"],
            }
        )
    scan = pd.DataFrame(rows)
    ok = scan.loc[
        (scan["valid_trades"] >= int(min_trades))
        & (scan["confirm_trades"] >= max(5, int(min_trades // 2)))
        & (scan["valid_daily_trades"] >= float(min_daily_trades))
        & (scan["valid_precision"] >= float(min_precision))
        & (scan["confirm_precision"] >= float(min_precision))
        & (scan["valid_avg_net_return"] > float(min_avg_net_return))
        & (scan["confirm_avg_net_return"] > float(min_avg_net_return))
        & (scan["valid_profit_factor"] >= float(min_profit_factor))
        & (scan["confirm_profit_factor"] >= float(min_profit_factor))
    ].copy()
    if ok.empty:
        return {"enabled": False, "reason": "no_valid_confirm_threshold"}, scan
    ok["score"] = (
        ok["confirm_avg_net_return"].fillna(-1.0) * 1600.0
        + ok["confirm_precision"].fillna(0.0) * 2.0
        + np.log1p(ok["confirm_trades"].astype(float)) * 0.05
        - ok["threshold"].astype(float) * 0.01
    )
    best = ok.sort_values(["score", "confirm_precision", "confirm_trades"], ascending=[False, False, False]).iloc[0]
    return {
        "enabled": True,
        "threshold": float(best["threshold"]),
        "valid_trades": int(best["valid_trades"]),
        "valid_precision": float(best["valid_precision"]),
        "valid_avg_net_return": float(best["valid_avg_net_return"]),
        "valid_profit_factor": float(best["valid_profit_factor"]),
        "confirm_trades": int(best["confirm_trades"]),
        "confirm_precision": float(best["confirm_precision"]),
        "confirm_avg_net_return": float(best["confirm_avg_net_return"]),
        "confirm_profit_factor": float(best["confirm_profit_factor"]),
        "reason": "valid_confirm_pass",
    }, scan


def train_btc_futures_v3_beta_edge_models(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    train_start: Any | None = "2021-01-01",
    valid_start: Any | None = "2024-01-01",
    confirm_start: Any | None = "2025-01-01",
    test_start: Any | None = "2026-01-01",
    test_end: Any | None = None,
    model_kind: str = "stable_rank",
    min_feature_coverage: float = 0.08,
    max_drift_score: float = 3.0,
    top_k_features: int = 3,
    selection_mode: str = "temporal_stability",
    min_feature_release_auc: float = 0.515,
    min_final_auc: float = 0.55,
    min_precision: float = 0.55,
    min_profit_factor: float = 1.05,
    min_trades: int = 40,
    min_daily_trades: float = 0.05,
) -> dict[str, Any]:
    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        build_btc_futures_v3_beta_edge_dataset(output_path=dataset_file)
    data = pd.read_parquet(dataset_file)
    for col in ("exec_time", "entry_time", "exit_time", "beta_signal_available_time"):
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], utc=True, errors="coerce")
    data = data.dropna(subset=["label_tp_first", "exec_time", "entry_time", "exit_time"]).sort_values("entry_time").reset_index(drop=True)
    if not bool((data["entry_time"] >= data["exec_time"] + pd.Timedelta(minutes=15)).all()):
        raise RuntimeError("lookahead audit failed before beta training")

    train_start_ts = parse_utc(train_start)
    valid_start_ts = parse_utc(valid_start)
    confirm_start_ts = parse_utc(confirm_start)
    test_start_ts = parse_utc(test_start)
    test_end_ts = parse_utc(test_end)
    if valid_start_ts is None or confirm_start_ts is None or test_start_ts is None:
        raise ValueError("valid_start, confirm_start and test_start are required")

    train_mask = data["exec_time"] < valid_start_ts
    if train_start_ts is not None:
        train_mask &= data["exec_time"] >= train_start_ts
    valid_mask = (data["exec_time"] >= valid_start_ts) & (data["exec_time"] < confirm_start_ts)
    confirm_mask = (data["exec_time"] >= confirm_start_ts) & (data["exec_time"] < test_start_ts)
    test_mask = data["exec_time"] >= test_start_ts
    if test_end_ts is not None:
        test_mask &= data["exec_time"] < test_end_ts
    train = data.loc[train_mask].copy()
    valid = data.loc[valid_mask].copy()
    confirm = data.loc[confirm_mask].copy()
    test = data.loc[test_mask].copy()
    if train.empty or valid.empty or confirm.empty:
        raise ValueError(f"empty split: train={len(train)} valid={len(valid)} confirm={len(confirm)} test={len(test)}")

    out_dir = Path(model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = {
        "route": ROUTE_NAME,
        "dataset_path": str(dataset_file),
        "model_dir": str(out_dir),
        "splits": {
            "train": {"rows": int(len(train)), "start": str(train["exec_time"].min()), "end": str(train["exec_time"].max())},
            "valid": {"rows": int(len(valid)), "start": str(valid["exec_time"].min()), "end": str(valid["exec_time"].max())},
            "confirm": {"rows": int(len(confirm)), "start": str(confirm["exec_time"].min()), "end": str(confirm["exec_time"].max())},
            "test": {"rows": int(len(test)), "start": str(test["exec_time"].min()), "end": str(test["exec_time"].max())},
        },
        "models": {},
    }
    thresholds: dict[str, float] = {}
    for side in ("buy", "sell"):
        side_train = train.loc[_side_mask(train, side)].copy()
        side_valid = valid.loc[_side_mask(valid, side)].copy()
        side_confirm = confirm.loc[_side_mask(confirm, side)].copy()
        side_test = test.loc[_side_mask(test, side)].copy()
        target = TARGET_COLUMNS["tp_first"]
        features, report = _select_features(
            side_train,
            side_valid,
            side_confirm,
            target=target,
            model_kind=model_kind,
            min_feature_coverage=min_feature_coverage,
            max_drift_score=max_drift_score,
            top_k=top_k_features,
            selection_mode=selection_mode,
            min_feature_release_auc=min_feature_release_auc,
        )
        fit_frame = (
            pd.concat([side_train, side_valid], axis=0, ignore_index=True)
            if model_kind in {"stable_rank", "rank"}
            else side_train
        )
        primary_spec = _fit_spec(
            fit_frame,
            features,
            target,
            model_kind=model_kind,
            random_state=73,
            feature_report=report,
        )
        report.to_csv(out_dir / f"{side}_tp_first_feature_report.csv", index=False)

        bundle = V3BetaEdgeModelBundle(
            side=side,
            primary_spec=primary_spec,
            threshold=0.99,
        )
        valid_scored = _score_bundle(bundle, side_valid)
        confirm_scored = _score_bundle(bundle, side_confirm)
        test_scored = _score_bundle(bundle, side_test) if not side_test.empty else pd.DataFrame()
        final_valid = _classification_metrics(valid_scored, "label_tp_first", valid_scored["probability"].to_numpy(dtype="float64"))
        final_confirm = _classification_metrics(confirm_scored, "label_tp_first", confirm_scored["probability"].to_numpy(dtype="float64"))
        decision: dict[str, Any]
        if float(final_valid.get("auc", np.nan)) < float(min_final_auc) or float(final_confirm.get("auc", np.nan)) < float(min_final_auc):
            decision = {
                "enabled": False,
                "reason": "final_auc_below_min",
                "valid_auc": final_valid.get("auc"),
                "confirm_auc": final_confirm.get("auc"),
            }
            scan = pd.DataFrame()
        else:
            decision, scan = _scan_thresholds(
                valid_scored,
                confirm_scored,
                min_trades=min_trades,
                min_daily_trades=min_daily_trades,
                min_precision=min_precision,
                min_avg_net_return=0.0,
                min_profit_factor=min_profit_factor,
            )
        if decision.get("enabled"):
            bundle.threshold = float(decision["threshold"])
        thresholds[side] = float(bundle.threshold)
        bundle.save(out_dir / f"{side}_model.pkl")
        scan.to_csv(out_dir / f"{side}_threshold_scan.csv", index=False)
        metrics["models"][side] = {
            "enabled": bool(decision.get("enabled", False)),
            "threshold": float(bundle.threshold),
            "decision": decision,
            "feature_counts": {"tp_first": int(len(primary_spec.feature_columns))},
            "selected_features": {"tp_first": list(primary_spec.feature_columns)},
            "tp_first": {
                "train": _classification_metrics(side_train, target, _predict_spec(side_train, primary_spec)),
                "valid": _classification_metrics(side_valid, target, _predict_spec(side_valid, primary_spec)),
                "confirm": _classification_metrics(side_confirm, target, _predict_spec(side_confirm, primary_spec)),
            },
            "final": {
                "valid": final_valid,
                "confirm": final_confirm,
                "test": _classification_metrics(test_scored, "label_tp_first", test_scored["probability"].to_numpy(dtype="float64"))
                if not test_scored.empty
                else {},
            },
        }

    policy = {
        "route": ROUTE_NAME,
        "default_thresholds": thresholds,
        "symbol_thresholds": {"BTCUSDT": thresholds},
        "quality_filter": {"enabled": False},
        "gate_filter": {"enabled": False},
        "position_sizing": {"enabled": False},
        "validation": {
            "train_start": str(train_start),
            "valid_start": str(valid_start),
            "confirm_start": str(confirm_start),
            "test_start": str(test_start),
            "model_kind": str(model_kind),
            "selection_mode": str(selection_mode),
            "top_k_features": int(top_k_features),
            "min_feature_release_auc": float(min_feature_release_auc),
            "min_final_auc": float(min_final_auc),
            "min_precision": float(min_precision),
            "min_profit_factor": float(min_profit_factor),
        },
    }
    (out_dir / "threshold_policy.json").write_text(json.dumps(json_safe(policy), ensure_ascii=False, indent=2), encoding="utf-8")
    metrics["policy"] = policy
    (out_dir / "metrics.json").write_text(json.dumps(json_safe(metrics), ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train BTC futures v3 beta edge three-layer models")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--train-start", default="2021-01-01")
    parser.add_argument("--valid-start", default="2024-01-01")
    parser.add_argument("--confirm-start", default="2025-01-01")
    parser.add_argument("--test-start", default="2026-01-01")
    parser.add_argument("--test-end", default="")
    parser.add_argument("--model-kind", default="stable_rank")
    parser.add_argument("--selection-mode", default="temporal_stability")
    parser.add_argument("--min-final-auc", type=float, default=0.55)
    parser.add_argument("--min-precision", type=float, default=0.55)
    parser.add_argument("--min-profit-factor", type=float, default=1.05)
    parser.add_argument("--min-feature-release-auc", type=float, default=0.515)
    parser.add_argument("--top-k-features", type=int, default=3)
    args = parser.parse_args()
    result = train_btc_futures_v3_beta_edge_models(
        dataset_path=args.dataset,
        model_dir=args.model_dir,
        train_start=args.train_start,
        valid_start=args.valid_start,
        confirm_start=args.confirm_start,
        test_start=args.test_start,
        test_end=args.test_end or None,
        model_kind=args.model_kind,
        selection_mode=args.selection_mode,
        min_final_auc=args.min_final_auc,
        min_precision=args.min_precision,
        min_profit_factor=args.min_profit_factor,
        min_feature_release_auc=args.min_feature_release_auc,
        top_k_features=args.top_k_features,
    )
    print(json.dumps(json_safe(result["models"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
