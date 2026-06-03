from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ML.features import infer_feature_columns
from ML.model import _make_estimator, train_classifier
from ML.routes.btc_futures_v1.data import parse_utc
from ML.shared.threshold_policy import market_state_from_row

from .bundle import StableGateModelBundle, make_gate_features
from .dataset import DEFAULT_DATASET_PATH, DEFAULT_MODEL_DIR, ROUTE_NAME


DROP_FEATURE_PREFIXES = (
    "cap_event_seq",
)


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


def _auc_or_nan(frame: pd.DataFrame, probability: np.ndarray) -> float:
    try:
        from sklearn.metrics import roc_auc_score

        y = frame["label"].astype(int).to_numpy()
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(roc_auc_score(y, probability))
    except Exception:
        return float("nan")


def _candidate_features(train: pd.DataFrame, min_feature_coverage: float) -> list[str]:
    all_features = infer_feature_columns(train, extra_prefixes=("cap_",))
    cols: list[str] = []
    for col in all_features:
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


def _select_stable_features(
    train: pd.DataFrame,
    valid: pd.DataFrame,
    side: str,
    model_kind: str,
    min_feature_coverage: float,
    max_drift_score: float,
    top_k: int,
) -> tuple[list[str], pd.DataFrame]:
    side_train = train.loc[_side_mask(train, side)].copy()
    side_valid = valid.loc[_side_mask(valid, side)].copy()
    candidates = _candidate_features(side_train, min_feature_coverage)
    stable = [col for col in candidates if _drift_score(side_train, side_valid, col) <= float(max_drift_score)]
    if len(stable) < 40:
        stable = candidates
    probe = train_classifier(side_train, feature_columns=stable, model_kind=model_kind, side="both")
    estimator = probe.estimator
    if hasattr(estimator, "feature_importances_"):
        importance = np.asarray(estimator.feature_importances_, dtype="float64")
    elif hasattr(estimator, "coef_"):
        importance = np.abs(np.asarray(estimator.coef_, dtype="float64")).ravel()
    else:
        importance = np.ones(len(stable), dtype="float64")
    report = pd.DataFrame(
        {
            "feature": stable,
            "importance": importance[: len(stable)],
            "drift_score": [_drift_score(side_train, side_valid, col) for col in stable],
            "coverage": [float(pd.to_numeric(side_train[col], errors="coerce").notna().mean()) for col in stable],
        }
    ).sort_values(["importance", "coverage"], ascending=[False, False])
    selected = report.head(int(top_k))["feature"].astype(str).tolist()
    return selected, report


def _fit_gate(
    train: pd.DataFrame,
    side: str,
    primary_features: list[str],
    model_kind: str,
    primary_threshold_hint: float = 0.5,
) -> tuple[object | None, list[str], dict[str, float]]:
    side_frame = train.loc[_side_mask(train, side)].sort_values("exec_time").copy()
    if len(side_frame) < 400 or side_frame["label"].nunique() < 2:
        return None, [], {}
    cut = max(50, int(len(side_frame) * 0.70))
    inner_train = side_frame.iloc[:cut].copy()
    gate_train = side_frame.iloc[cut:].copy()
    if gate_train["label"].nunique() < 2:
        return None, [], {}
    inner_bundle = train_classifier(
        inner_train,
        feature_columns=primary_features,
        model_kind=model_kind,
        threshold=primary_threshold_hint,
        side="both",
    )
    primary_probability = inner_bundle.predict_proba(gate_train)
    gate_x = make_gate_features(gate_train, primary_probability, primary_threshold_hint)
    gate_cols = list(gate_x.columns)
    gate_fill = gate_x.replace([np.inf, -np.inf], np.nan).median(numeric_only=True).fillna(0.0).to_dict()
    gate_x = gate_x.replace([np.inf, -np.inf], np.nan).fillna(gate_fill).fillna(0.0)
    gate_y = gate_train["label"].astype(int)
    estimator = _make_estimator("sklearn", random_state=73)
    estimator.fit(gate_x, gate_y)
    return estimator, gate_cols, {str(k): float(v) for k, v in gate_fill.items()}


def _scan_thresholds(
    frame: pd.DataFrame,
    *,
    min_trades: int,
    min_daily_trades: float,
    min_year_trades: int = 10,
    min_precision: float,
    min_avg_net_return: float,
    min_profit_factor: float,
    primary_grid: np.ndarray | None = None,
    gate_grid: np.ndarray | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    if frame.empty:
        return {"enabled": False, "reason": "empty"}, pd.DataFrame()
    times = pd.to_datetime(frame["entry_time"], utc=True, errors="coerce")
    days = max(1.0, (times.max() - times.min()).total_seconds() / 86400.0)
    p_grid = primary_grid if primary_grid is not None else np.round(np.arange(0.45, 0.901, 0.01), 2)
    g_grid = gate_grid if gate_grid is not None else np.round(np.arange(0.45, 0.901, 0.02), 2)
    rows: list[dict[str, Any]] = []
    has_gate = "gate_probability" in frame.columns and frame["gate_probability"].notna().any()
    for p_th in p_grid:
        for g_th in (g_grid if has_gate else np.asarray([0.0])):
            selected = frame["primary_probability"].astype(float) >= float(p_th)
            if has_gate:
                selected &= frame["gate_probability"].astype(float) >= float(g_th)
            sub = frame.loc[selected]
            metrics = _metrics_for_selection(sub)
            daily = float(metrics["trades"] / days)
            rows.append(
                {
                    "primary_threshold": float(p_th),
                    "gate_threshold": float(g_th) if has_gate else float("nan"),
                    "trades": int(metrics["trades"]),
                    "daily_trades": daily,
                    "precision": metrics["precision"],
                    "avg_net_return": metrics["avg_net_return"],
                    "profit_factor": metrics["profit_factor"],
                }
            )
    scan = pd.DataFrame(rows)
    ok = scan.loc[
        (scan["trades"] >= int(min_trades))
        & (scan["daily_trades"] >= float(min_daily_trades))
        & (scan["precision"] >= float(min_precision))
        & (scan["avg_net_return"] > float(min_avg_net_return))
        & (scan["profit_factor"] >= float(min_profit_factor))
    ].copy()
    frame_year = pd.to_datetime(frame["entry_time"], utc=True, errors="coerce").dt.year
    validation_years = sorted(int(year) for year in frame_year.dropna().unique())
    if len(validation_years) >= 2 and not ok.empty:
        stable_rows = []
        for _, candidate in ok.iterrows():
            selected = frame["primary_probability"].astype(float) >= float(candidate["primary_threshold"])
            if "gate_probability" in frame.columns and np.isfinite(float(candidate["gate_threshold"])):
                selected &= frame["gate_probability"].astype(float) >= float(candidate["gate_threshold"])
            stable = True
            yearly_records = []
            for year in validation_years:
                sub = frame.loc[selected & (frame_year == year)]
                metrics = _metrics_for_selection(sub)
                yearly_records.append(
                    {
                        "year": int(year),
                        "trades": int(metrics["trades"]),
                        "precision": metrics["precision"],
                        "avg_net_return": metrics["avg_net_return"],
                        "profit_factor": metrics["profit_factor"],
                    }
                )
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
        ok["avg_net_return"].fillna(-1.0) * 1000.0
        + ok["precision"].fillna(0.0) * 2.0
        + np.log1p(ok["trades"].astype(float)) * 0.02
        - ok["primary_threshold"].astype(float) * 0.01
    )
    best = ok.sort_values(["score", "precision", "trades"], ascending=[False, False, False]).iloc[0]
    return {
        "enabled": True,
        "primary_threshold": float(best["primary_threshold"]),
        "gate_threshold": float(best["gate_threshold"]) if np.isfinite(best["gate_threshold"]) else 0.0,
        "trades": int(best["trades"]),
        "daily_trades": float(best["daily_trades"]),
        "precision": float(best["precision"]),
        "avg_net_return": float(best["avg_net_return"]),
        "profit_factor": float(best["profit_factor"]),
        "reason": "cost_aware_pass",
    }, scan


def _build_policy(
    valid_scored: pd.DataFrame,
    side_decisions: dict[str, dict[str, Any]],
    group_decisions: dict[str, dict[str, dict[str, Any]]],
    min_precision: float,
    min_avg_net_return: float,
    min_profit_factor: float,
) -> dict[str, Any]:
    allowed_states = {"buy": [], "sell": []}
    market_state_thresholds: dict[str, dict[str, float]] = {}
    groups: dict[str, dict[str, Any]] = {}
    enabled_thresholds = {"buy": [], "sell": []}
    for side, state_map in group_decisions.items():
        for state, decision in state_map.items():
            key = f"{state}|{side}"
            if not decision.get("enabled", False):
                groups[key] = {"enabled": False, "reason": decision.get("reason", "disabled")}
                continue
            allowed_states[side].append(state)
            enabled_thresholds[side].append(float(decision["primary_threshold"]))
            market_state_thresholds.setdefault(state, {})[side] = float(decision["primary_threshold"])
            groups[key] = {
                "enabled": True,
                "gate_min": float(decision.get("gate_threshold", 0.0)),
                "validation_trades": int(decision.get("trades", 0)),
                "validation_precision": float(decision.get("precision", np.nan)),
                "validation_avg_net_return": float(decision.get("avg_net_return", np.nan)),
                "validation_profit_factor": float(decision.get("profit_factor", np.nan)),
            }
    default_buy = side_decisions.get("buy", {}).get("primary_threshold")
    default_sell = side_decisions.get("sell", {}).get("primary_threshold")
    if default_buy is None or not side_decisions.get("buy", {}).get("enabled", False):
        default_buy = min(enabled_thresholds["buy"]) if enabled_thresholds["buy"] else 0.99
    if default_sell is None or not side_decisions.get("sell", {}).get("enabled", False):
        default_sell = min(enabled_thresholds["sell"]) if enabled_thresholds["sell"] else 0.99
    return {
        "route": ROUTE_NAME,
        "default_thresholds": {"buy": float(default_buy), "sell": float(default_sell)},
        "symbol_thresholds": {"BTCUSDT": {"buy": float(default_buy), "sell": float(default_sell)}},
        "market_state_thresholds": market_state_thresholds,
        "validation_filter": {
            "enabled": True,
            "allowed_market_states": {
                "buy": sorted(set(allowed_states["buy"])),
                "sell": sorted(set(allowed_states["sell"])),
            },
        },
        "group_frequency_filter": {
            "enabled": True,
            "fallback_order": ["state_side"],
            "groups": {"state_side": groups},
        },
        "gate_filter": {
            "enabled": True,
            "model_threshold_mode": "policy",
            "buy": {"min_score": 1.0},
            "sell": {"min_score": 1.0},
        },
        "quality_filter": {"enabled": False},
        "position_sizing": {"enabled": False},
        "cost_aware_constraints": {
            "min_precision": float(min_precision),
            "min_avg_net_return": float(min_avg_net_return),
            "min_profit_factor": float(min_profit_factor),
        },
    }


def train_btc_futures_v2_stable_models(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    train_start: Any | None = "2021-01-01",
    valid_start: Any | None = "2025-01-01",
    test_start: Any | None = "2026-01-01",
    test_end: Any | None = None,
    model_kind: str = "auto",
    min_feature_coverage: float = 0.08,
    max_drift_score: float = 2.5,
    top_k_features: int = 140,
    min_precision: float = 0.65,
    min_avg_net_return: float = 0.0,
    min_profit_factor: float = 1.50,
    min_gate_auc: float = 0.55,
    min_side_trades: int = 50,
    min_side_daily_trades: float = 0.10,
    min_group_trades: int = 20,
    min_group_daily_trades: float = 0.03,
) -> dict[str, Any]:
    dataset = pd.read_parquet(dataset_path)
    for col in ("exec_time", "entry_time", "exit_time"):
        dataset[col] = pd.to_datetime(dataset[col], utc=True, errors="coerce")
    dataset = dataset.dropna(subset=["label", "exec_time", "entry_time"]).sort_values("exec_time").reset_index(drop=True)
    dataset["market_state"] = dataset.apply(market_state_from_row, axis=1)

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
        "dataset_path": str(dataset_path),
        "model_dir": str(out_dir),
        "splits": {
            "train": {"start": str(train["exec_time"].min()), "end": str(train["exec_time"].max()), "rows": int(len(train))},
            "valid": {"start": str(valid["exec_time"].min()), "end": str(valid["exec_time"].max()), "rows": int(len(valid))},
            "test": {"start": str(test["exec_time"].min()), "end": str(test["exec_time"].max()), "rows": int(len(test))},
        },
        "models": {},
        "groups": {},
    }

    valid_scored_parts: list[pd.DataFrame] = []
    side_decisions: dict[str, dict[str, Any]] = {}
    group_decisions: dict[str, dict[str, dict[str, Any]]] = {"buy": {}, "sell": {}}

    for side in ("buy", "sell"):
        side_train = train.loc[_side_mask(train, side)].copy()
        side_valid = valid.loc[_side_mask(valid, side)].copy()
        side_test = test.loc[_side_mask(test, side)].copy()
        features, feature_report = _select_stable_features(
            train,
            valid,
            side,
            model_kind,
            min_feature_coverage,
            max_drift_score,
            top_k_features,
        )
        primary = train_classifier(train, feature_columns=features, model_kind=model_kind, threshold=0.5, side=side)
        gate_estimator, gate_cols, gate_fill = _fit_gate(train, side, features, model_kind)
        valid_primary = primary.predict_proba(side_valid)
        test_primary = primary.predict_proba(side_test) if not side_test.empty else np.empty(0)
        if gate_estimator is not None:
            temp_bundle = StableGateModelBundle(
                estimator=primary.estimator,
                primary_feature_columns=features,
                primary_fill_values=primary.fill_values,
                threshold=0.5,
                side=side,
                gate_estimator=gate_estimator,
                gate_feature_columns=gate_cols,
                gate_fill_values=gate_fill,
                gate_threshold=1.0,
            )
            valid_gate = temp_bundle.predict_gate_proba(side_valid)
            test_gate = temp_bundle.predict_gate_proba(side_test) if not side_test.empty else np.empty(0)
            gate_auc = _auc_or_nan(side_valid, valid_gate)
            primary_auc = _auc_or_nan(side_valid, valid_primary)
            if (not np.isfinite(gate_auc)) or gate_auc < float(min_gate_auc) or gate_auc <= primary_auc:
                gate_estimator = None
                gate_cols = []
                gate_fill = {}
                valid_gate = np.ones(len(side_valid), dtype="float64")
                test_gate = np.ones(len(side_test), dtype="float64")
        else:
            valid_gate = np.ones(len(side_valid), dtype="float64")
            test_gate = np.ones(len(side_test), dtype="float64")

        scan_frame = side_valid.copy()
        scan_frame["primary_probability"] = valid_primary
        scan_frame["gate_probability"] = valid_gate
        side_decision, side_scan = _scan_thresholds(
            scan_frame,
            min_trades=min_side_trades,
            min_daily_trades=min_side_daily_trades,
            min_year_trades=max(10, int(min_side_trades // 3)),
            min_precision=min_precision,
            min_avg_net_return=min_avg_net_return,
            min_profit_factor=min_profit_factor,
        )
        side_decisions[side] = side_decision
        primary.threshold = float(side_decision.get("primary_threshold", 0.99))
        gate_threshold_default = float(side_decision.get("gate_threshold", 1.0)) if side_decision.get("enabled") else 1.0

        group_records = []
        for state, state_frame in scan_frame.groupby("market_state", sort=False):
            decision, state_scan = _scan_thresholds(
                state_frame,
                min_trades=min_group_trades,
                min_daily_trades=min_group_daily_trades,
                min_year_trades=max(5, int(min_group_trades // 3)),
                min_precision=min_precision,
                min_avg_net_return=min_avg_net_return,
                min_profit_factor=min_profit_factor,
            )
            group_decisions[side][str(state)] = decision
            rec = {"side": side, "market_state": str(state), **decision}
            group_records.append(rec)
            state_scan.to_csv(out_dir / f"{side}_{state}_threshold_scan.csv", index=False)

        bundle = StableGateModelBundle(
            estimator=primary.estimator,
            primary_feature_columns=features,
            primary_fill_values=primary.fill_values,
            threshold=float(primary.threshold),
            side=side,
            gate_estimator=gate_estimator,
            gate_feature_columns=gate_cols,
            gate_fill_values=gate_fill,
            gate_threshold=gate_threshold_default,
        )
        bundle.save(out_dir / f"{side}_model.pkl")
        feature_report.to_csv(out_dir / f"{side}_stable_feature_report.csv", index=False)
        side_scan.to_csv(out_dir / f"{side}_global_threshold_scan.csv", index=False)
        pd.DataFrame(group_records).to_csv(out_dir / f"{side}_group_decisions.csv", index=False)

        valid_scored = side_valid.copy()
        valid_scored["primary_probability"] = valid_primary
        valid_scored["gate_probability"] = valid_gate
        valid_scored_parts.append(valid_scored)

        metrics["models"][side] = {
            "primary_threshold": float(bundle.threshold),
            "gate_threshold": float(bundle.gate_threshold),
            "enabled": bool(side_decision.get("enabled", False)),
            "feature_count": int(len(features)),
            "gate_feature_count": int(len(gate_cols)),
            "side_decision": side_decision,
            "train": _classification_metrics(side_train, primary.predict_proba(side_train)),
            "valid": _classification_metrics(side_valid, valid_primary),
            "test": _classification_metrics(side_test, test_primary) if not side_test.empty else {},
            "gate_valid": _classification_metrics(side_valid, valid_gate) if gate_estimator is not None else {},
            "gate_test": _classification_metrics(side_test, test_gate) if gate_estimator is not None and not side_test.empty else {},
        }

    valid_scored_all = pd.concat(valid_scored_parts, ignore_index=True) if valid_scored_parts else pd.DataFrame()
    policy = _build_policy(
        valid_scored_all,
        side_decisions,
        group_decisions,
        min_precision=min_precision,
        min_avg_net_return=min_avg_net_return,
        min_profit_factor=min_profit_factor,
    )
    (out_dir / "threshold_policy.json").write_text(json.dumps(policy, ensure_ascii=False, indent=2), encoding="utf-8")
    metrics["policy"] = policy
    (out_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Train BTC futures v2 stable models")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--train-start", default="2021-01-01")
    parser.add_argument("--valid-start", default="2025-01-01")
    parser.add_argument("--test-start", default="2026-01-01")
    parser.add_argument("--test-end", default="")
    parser.add_argument("--model-kind", default="auto")
    parser.add_argument("--min-precision", type=float, default=0.65)
    parser.add_argument("--min-profit-factor", type=float, default=1.50)
    parser.add_argument("--min-gate-auc", type=float, default=0.55)
    parser.add_argument("--top-k-features", type=int, default=140)
    args = parser.parse_args()
    result = train_btc_futures_v2_stable_models(
        dataset_path=args.dataset,
        model_dir=args.model_dir,
        train_start=args.train_start,
        valid_start=args.valid_start,
        test_start=args.test_start,
        test_end=args.test_end or None,
        model_kind=args.model_kind,
        min_precision=args.min_precision,
        min_profit_factor=args.min_profit_factor,
        min_gate_auc=args.min_gate_auc,
        top_k_features=args.top_k_features,
    )
    print(json.dumps(result["models"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
