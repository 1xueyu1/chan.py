from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ML.routes.btc_futures_v1.data import parse_utc
from ML.shared.json_utils import json_safe

from .bundle import SideBspSplitBetaEdgeModelBundle, V3BetaEdgeModelBundle, bsp_family_for_frame
from .dataset import DEFAULT_DATASET_PATH, ROUTE_NAME, build_btc_futures_v3_beta_edge_dataset
from .train import (
    TARGET_COLUMNS,
    _classification_metrics,
    _fit_spec,
    _predict_spec,
    _scan_thresholds,
    _score_bundle,
    _select_features,
    _side_mask,
)


SPLIT_ROUTE_NAME = f"{ROUTE_NAME}_bsp2_only"
DEFAULT_SPLIT_MODEL_DIR = Path("result/ml/btc_futures_v3_beta_edge_bsp2_only")
DEFAULT_SPLIT_BACKTEST_DIR = Path("result/btc_futures_v3_beta_edge_bsp2_only")
BSP_FAMILIES = ("2",)


def _family_mask(frame: pd.DataFrame, family: str) -> pd.Series:
    return bsp_family_for_frame(frame).astype(str) == str(family)


def _load_dataset(dataset_path: str | Path) -> pd.DataFrame:
    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        build_btc_futures_v3_beta_edge_dataset(output_path=dataset_file)
    data = pd.read_parquet(dataset_file)
    for col in ("exec_time", "entry_time", "exit_time", "beta_signal_available_time"):
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], utc=True, errors="coerce")
    data = data.dropna(subset=["label_tp_first", "exec_time", "entry_time", "exit_time"]).sort_values("entry_time").reset_index(drop=True)
    if not bool((data["entry_time"] >= data["exec_time"] + pd.Timedelta(minutes=15)).all()):
        raise RuntimeError("lookahead audit failed before bsp split training")
    data["bsp_family"] = bsp_family_for_frame(data).astype(str)
    return data


def _split_data(
    data: pd.DataFrame,
    *,
    train_start: Any | None,
    valid_start: Any,
    confirm_start: Any,
    test_start: Any,
    test_end: Any | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
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
    return data.loc[train_mask].copy(), data.loc[valid_mask].copy(), data.loc[confirm_mask].copy(), data.loc[test_mask].copy()


def _train_one_family_bundle(
    *,
    side: str,
    family: str,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    confirm: pd.DataFrame,
    test: pd.DataFrame,
    out_dir: Path,
    model_kind: str,
    selection_mode: str,
    min_feature_coverage: float,
    max_drift_score: float,
    top_k_features: int,
    min_feature_release_auc: float,
    min_final_auc: float,
    min_precision: float,
    min_profit_factor: float,
    min_trades: int,
    min_daily_trades: float,
    min_family_train_rows: int,
    min_family_valid_rows: int,
    min_family_confirm_rows: int,
) -> tuple[V3BetaEdgeModelBundle | None, dict[str, Any]]:
    key = f"{side}_bsp{family}"
    side_train = train.loc[_side_mask(train, side) & _family_mask(train, family)].copy()
    side_valid = valid.loc[_side_mask(valid, side) & _family_mask(valid, family)].copy()
    side_confirm = confirm.loc[_side_mask(confirm, side) & _family_mask(confirm, family)].copy()
    side_test = test.loc[_side_mask(test, side) & _family_mask(test, family)].copy()
    info: dict[str, Any] = {
        "side": side,
        "bsp_family": family,
        "rows": {
            "train": int(len(side_train)),
            "valid": int(len(side_valid)),
            "confirm": int(len(side_confirm)),
            "test": int(len(side_test)),
        },
        "enabled": False,
    }
    if (
        len(side_train) < int(min_family_train_rows)
        or len(side_valid) < int(min_family_valid_rows)
        or len(side_confirm) < int(min_family_confirm_rows)
    ):
        info["reason"] = "not_enough_rows"
        return None, info
    if side_train["label_tp_first"].nunique(dropna=True) < 2:
        info["reason"] = "one_class_target_in_train"
        return None, info

    try:
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
            random_state=730 + int(family) * 10,
            feature_report=report,
        )
        report.to_csv(out_dir / f"{key}_tp_first_feature_report.csv", index=False)
    except Exception as exc:
        info["reason"] = f"feature_or_fit_failed: {exc}"
        return None, info

    bundle = V3BetaEdgeModelBundle(
        side=side,
        primary_spec=primary_spec,
        threshold=0.99,
        route=SPLIT_ROUTE_NAME,
    )
    valid_scored = _score_bundle(bundle, side_valid)
    confirm_scored = _score_bundle(bundle, side_confirm)
    test_scored = _score_bundle(bundle, side_test) if not side_test.empty else pd.DataFrame()
    final_valid = _classification_metrics(valid_scored, "label_tp_first", valid_scored["probability"].to_numpy(dtype="float64"))
    final_confirm = _classification_metrics(confirm_scored, "label_tp_first", confirm_scored["probability"].to_numpy(dtype="float64"))
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
        if decision.get("enabled") and int(decision.get("confirm_trades", 0)) < int(min_trades):
            decision = {
                **decision,
                "enabled": False,
                "reason": "confirm_trades_below_full_min",
                "required_confirm_trades": int(min_trades),
            }
    scan.to_csv(out_dir / f"{key}_threshold_scan.csv", index=False)
    if decision.get("enabled"):
        bundle.threshold = float(decision["threshold"])
        info["enabled"] = True
    else:
        bundle.threshold = 0.99
    info.update(
        {
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
    )
    return bundle, info


def train_bsp_split_models(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    model_dir: str | Path = DEFAULT_SPLIT_MODEL_DIR,
    train_start: Any | None = "2021-01-01",
    valid_start: Any = "2024-01-01",
    confirm_start: Any = "2025-01-01",
    test_start: Any = "2026-01-01",
    test_end: Any | None = None,
    model_kind: str = "stable_rank",
    selection_mode: str = "temporal_stability",
    min_feature_coverage: float = 0.08,
    max_drift_score: float = 3.0,
    top_k_features: int = 20,
    min_feature_release_auc: float = 0.505,
    min_final_auc: float = 0.52,
    min_precision: float = 0.52,
    min_profit_factor: float = 1.01,
    min_trades: int = 12,
    min_daily_trades: float = 0.01,
    min_family_train_rows: int = 1500,
    min_family_valid_rows: int = 500,
    min_family_confirm_rows: int = 500,
) -> dict[str, Any]:
    data = _load_dataset(dataset_path)
    train, valid, confirm, test = _split_data(
        data,
        train_start=train_start,
        valid_start=valid_start,
        confirm_start=confirm_start,
        test_start=test_start,
        test_end=test_end,
    )
    if train.empty or valid.empty or confirm.empty:
        raise ValueError(f"empty split: train={len(train)} valid={len(valid)} confirm={len(confirm)} test={len(test)}")
    out_dir = Path(model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics: dict[str, Any] = {
        "route": SPLIT_ROUTE_NAME,
        "dataset_path": str(dataset_path),
        "model_dir": str(out_dir),
        "bsp_families": list(BSP_FAMILIES),
        "feature_policy": {
            "shared_candidate_pool": "cap_, v3_, beta_ numeric features",
            "selection": "bsp family 2 only; per side tp_first top_k only; low impact features are not saved into model feature_columns",
            "top_k_features": int(top_k_features),
            "selection_mode": str(selection_mode),
            "min_family_train_rows": int(min_family_train_rows),
            "min_family_valid_rows": int(min_family_valid_rows),
            "min_family_confirm_rows": int(min_family_confirm_rows),
        },
        "splits": {
            "train": {"rows": int(len(train)), "start": str(train["exec_time"].min()), "end": str(train["exec_time"].max())},
            "valid": {"rows": int(len(valid)), "start": str(valid["exec_time"].min()), "end": str(valid["exec_time"].max())},
            "confirm": {"rows": int(len(confirm)), "start": str(confirm["exec_time"].min()), "end": str(confirm["exec_time"].max())},
            "test": {"rows": int(len(test)), "start": str(test["exec_time"].min()), "end": str(test["exec_time"].max())},
        },
        "models": {},
    }

    manifest_rows: list[dict[str, Any]] = []
    side_thresholds: dict[str, float] = {}
    for side in ("buy", "sell"):
        family_bundles: dict[str, V3BetaEdgeModelBundle] = {}
        metrics["models"][side] = {}
        for family in BSP_FAMILIES:
            bundle, info = _train_one_family_bundle(
                side=side,
                family=family,
                train=train,
                valid=valid,
                confirm=confirm,
                test=test,
                out_dir=out_dir,
                model_kind=model_kind,
                selection_mode=selection_mode,
                min_feature_coverage=min_feature_coverage,
                max_drift_score=max_drift_score,
                top_k_features=top_k_features,
                min_feature_release_auc=min_feature_release_auc,
                min_final_auc=min_final_auc,
                min_precision=min_precision,
                min_profit_factor=min_profit_factor,
                min_trades=min_trades,
                min_daily_trades=min_daily_trades,
                min_family_train_rows=min_family_train_rows,
                min_family_valid_rows=min_family_valid_rows,
                min_family_confirm_rows=min_family_confirm_rows,
            )
            metrics["models"][side][family] = info
            if bundle is not None:
                family_bundles[family] = bundle
            for layer, selected in info.get("selected_features", {}).items():
                for rank, feature in enumerate(selected, start=1):
                    manifest_rows.append({"side": side, "bsp_family": family, "layer": layer, "rank": rank, "feature": feature})
        side_bundle = SideBspSplitBetaEdgeModelBundle(side=side, family_bundles=family_bundles, route=SPLIT_ROUTE_NAME)
        side_bundle.save(out_dir / f"{side}_model.pkl")
        enabled_thresholds = [
            float(bundle.threshold)
            for bundle in family_bundles.values()
            if float(bundle.threshold) < float(side_bundle.disabled_threshold)
        ]
        side_thresholds[side] = float(min(enabled_thresholds)) if enabled_thresholds else float(side_bundle.disabled_threshold)

    pd.DataFrame(manifest_rows).to_csv(out_dir / "selected_feature_manifest.csv", index=False)
    policy = {
        "route": SPLIT_ROUTE_NAME,
        "default_thresholds": {"buy": float(side_thresholds.get("buy", 0.99)), "sell": float(side_thresholds.get("sell", 0.99))},
        "symbol_thresholds": {
            "BTCUSDT": {"buy": float(side_thresholds.get("buy", 0.99)), "sell": float(side_thresholds.get("sell", 0.99))}
        },
        "quality_filter": {"enabled": False},
        "gate_filter": {"enabled": False},
        "position_sizing": {"enabled": False},
        "validation": {
            "train_start": str(train_start),
            "valid_start": str(valid_start),
            "confirm_start": str(confirm_start),
            "test_start": str(test_start),
            "model_kind": str(model_kind),
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

    parser = argparse.ArgumentParser(description="Train BTC v3 beta edge models split by BSP family and side")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--model-dir", default=str(DEFAULT_SPLIT_MODEL_DIR))
    parser.add_argument("--train-start", default="2021-01-01")
    parser.add_argument("--valid-start", default="2024-01-01")
    parser.add_argument("--confirm-start", default="2025-01-01")
    parser.add_argument("--test-start", default="2026-01-01")
    parser.add_argument("--test-end", default="")
    parser.add_argument("--model-kind", default="stable_rank")
    parser.add_argument("--selection-mode", default="temporal_stability")
    parser.add_argument("--top-k-features", type=int, default=20)
    parser.add_argument("--min-feature-release-auc", type=float, default=0.505)
    parser.add_argument("--min-final-auc", type=float, default=0.52)
    parser.add_argument("--min-precision", type=float, default=0.52)
    parser.add_argument("--min-profit-factor", type=float, default=1.01)
    parser.add_argument("--min-trades", type=int, default=12)
    parser.add_argument("--min-daily-trades", type=float, default=0.01)
    parser.add_argument("--min-family-train-rows", type=int, default=1500)
    parser.add_argument("--min-family-valid-rows", type=int, default=500)
    parser.add_argument("--min-family-confirm-rows", type=int, default=500)
    args = parser.parse_args()
    result = train_bsp_split_models(
        dataset_path=args.dataset,
        model_dir=args.model_dir,
        train_start=args.train_start,
        valid_start=args.valid_start,
        confirm_start=args.confirm_start,
        test_start=args.test_start,
        test_end=args.test_end or None,
        model_kind=args.model_kind,
        selection_mode=args.selection_mode,
        top_k_features=args.top_k_features,
        min_feature_release_auc=args.min_feature_release_auc,
        min_final_auc=args.min_final_auc,
        min_precision=args.min_precision,
        min_profit_factor=args.min_profit_factor,
        min_trades=args.min_trades,
        min_daily_trades=args.min_daily_trades,
        min_family_train_rows=args.min_family_train_rows,
        min_family_valid_rows=args.min_family_valid_rows,
        min_family_confirm_rows=args.min_family_confirm_rows,
    )
    print(json.dumps(json_safe(result["models"]), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
