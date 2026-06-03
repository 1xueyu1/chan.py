from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from ML.features import build_event_features
from ML.model import ModelBundle
from ML.registry import validate_model_manifest
from ML.shared.structure_position_sizing import bsp2_structure_quality_and_sizing
from ML.shared.threshold_policy import CapabilityV2ThresholdPolicy, add_policy_columns
from .time_utils import event_signal_time

from .types import RawBSPEvent, ScoredSignalEvent


@lru_cache(maxsize=8)
def _load_bundle(path: str) -> ModelBundle:
    bundle = ModelBundle.load(path)
    validate_model_manifest(path, list(bundle.feature_columns))
    return bundle


def _score_side(bundle: ModelBundle, features: pd.DataFrame, threshold: float, signal_value: int) -> tuple[np.ndarray, np.ndarray]:
    if features.empty:
        return np.empty(0, dtype="float64"), np.empty(0, dtype=bool)
    probability = bundle.predict_proba(features)
    if hasattr(bundle, "predict_qualified"):
        qualified = bundle.predict_qualified(features, default_threshold=float(threshold))
    else:
        qualified = probability >= float(threshold)
    return probability, np.asarray(qualified, dtype=bool)


def _finite_or_nan(value) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _prepare_bundle_features(features: pd.DataFrame, bundles: list[ModelBundle]) -> pd.DataFrame:
    out = features
    for bundle in bundles:
        prepare = getattr(bundle, "prepare_features", None)
        if prepare is None:
            continue
        prepared = prepare(out)
        for col in prepared.columns:
            if col not in out.columns or str(col).startswith("v3_"):
                out[col] = prepared[col]
    return out


def _coverage_time(event: RawBSPEvent) -> pd.Timestamp:
    actual = getattr(event, "actual_exec_time", None)
    if actual is not None and not pd.isna(actual):
        return pd.to_datetime(actual, utc=True)
    return pd.to_datetime(event_signal_time(event), utc=True)


def _apply_daily_coverage_policy(
    policy: CapabilityV2ThresholdPolicy,
    raw_events: List[RawBSPEvent],
    probabilities: np.ndarray,
    feature_rows: list[pd.Series],
    decisions: list,
) -> list:
    if not getattr(policy, "daily_coverage_enabled", False):
        return decisions

    start_hour = float(getattr(policy, "daily_coverage_start_hour_utc", 12.0))
    max_per_day = max(1, int(getattr(policy, "daily_coverage_max_signals_per_day", 1)))
    indexed_times = [(idx, _coverage_time(event)) for idx, event in enumerate(raw_events)]
    qualified_days = {
        (str(raw_events[idx].symbol), ts.date())
        for idx, ts in indexed_times
        if bool(decisions[idx].qualified)
    }
    fallback_counts: dict[tuple[str, object], int] = {}

    for idx, ts in sorted(indexed_times, key=lambda item: item[1]):
        event = raw_events[idx]
        key = (str(event.symbol), ts.date())
        if key in qualified_days:
            continue
        if fallback_counts.get(key, 0) >= max_per_day:
            continue
        hour_float = float(ts.hour) + float(ts.minute) / 60.0 + float(ts.second) / 3600.0
        if hour_float < start_hour:
            continue
        side = "buy" if bool(event.is_buy) else "sell"
        fallback_decision = policy.daily_coverage_decide(
            event.symbol,
            side,
            float(probabilities[idx]),
            feature_rows[idx],
            decisions[idx],
        )
        if bool(fallback_decision.qualified):
            decisions[idx] = fallback_decision
            fallback_counts[key] = fallback_counts.get(key, 0) + 1
            if fallback_counts[key] >= max_per_day:
                qualified_days.add(key)
    return decisions


def score_raw_events_with_ml(
    bars: pd.DataFrame,
    raw_events: List[RawBSPEvent],
    buy_model_path: Optional[str],
    sell_model_path: Optional[str],
    both_model_path: Optional[str],
    buy_threshold: float,
    sell_threshold: float,
    rust_config_path: Optional[str] = None,
    threshold_policy_path: Optional[str] = None,
) -> List[ScoredSignalEvent]:
    if not raw_events:
        return []

    required_columns = set()
    loaded_bundles: list[ModelBundle] = []
    for model_path in (buy_model_path, sell_model_path, both_model_path):
        if model_path:
            bundle = _load_bundle(str(model_path))
            loaded_bundles.append(bundle)
            required_columns.update(bundle.feature_columns)

    features = build_event_features(
        bars,
        raw_events,
        required_columns=required_columns or None,
        rust_config_path=rust_config_path,
    )
    if features.empty:
        return []
    if loaded_bundles:
        features = _prepare_bundle_features(features, loaded_bundles)

    probabilities = np.zeros(len(raw_events), dtype="float64")
    gate_probabilities = np.full(len(raw_events), np.nan, dtype="float64")
    thresholds = np.zeros(len(raw_events), dtype="float64")

    buy_mask = features["is_buy"].astype(bool).to_numpy()
    sell_mask = ~buy_mask

    bundle_has_gate = False
    bundle_quality_threshold = 0.0
    gate_thresholds = {"buy": np.nan, "sell": np.nan}

    if both_model_path:
        bundle = _load_bundle(str(both_model_path))
        probabilities[:] = bundle.predict_proba(features)
        thresholds[:] = np.where(buy_mask, float(buy_threshold), float(sell_threshold))
        if hasattr(bundle, "predict_gate_proba"):
            gate_probabilities[:] = np.asarray(bundle.predict_gate_proba(features), dtype="float64")
            bundle_has_gate = True
        gate_threshold = _finite_or_nan(getattr(bundle, "gate_threshold", np.nan))
        if np.isfinite(gate_threshold):
            gate_thresholds["buy"] = gate_thresholds["sell"] = max(
                float(gate_thresholds["buy"]) if np.isfinite(gate_thresholds["buy"]) else float("-inf"),
                float(gate_threshold),
            )
        q_threshold = _finite_or_nan(getattr(bundle, "quality_threshold", np.nan))
        if np.isfinite(q_threshold):
            bundle_quality_threshold = max(bundle_quality_threshold, q_threshold)

    if buy_model_path and buy_mask.any():
        bundle = _load_bundle(str(buy_model_path))
        buy_probability, _ = _score_side(bundle, features.loc[buy_mask], float(buy_threshold), 1)
        probabilities[buy_mask] = buy_probability
        if hasattr(bundle, "threshold_for_rows"):
            thresholds[buy_mask] = np.asarray(bundle.threshold_for_rows(features.loc[buy_mask]), dtype="float64")
        else:
            thresholds[buy_mask] = float(buy_threshold)
        if hasattr(bundle, "predict_gate_proba"):
            gate_probabilities[buy_mask] = np.asarray(bundle.predict_gate_proba(features.loc[buy_mask]), dtype="float64")
            bundle_has_gate = True
        gate_threshold = _finite_or_nan(getattr(bundle, "gate_threshold", np.nan))
        if np.isfinite(gate_threshold):
            gate_thresholds["buy"] = max(
                float(gate_thresholds["buy"]) if np.isfinite(gate_thresholds["buy"]) else float("-inf"),
                float(gate_threshold),
            )
        q_threshold = _finite_or_nan(getattr(bundle, "quality_threshold", np.nan))
        if np.isfinite(q_threshold):
            bundle_quality_threshold = max(bundle_quality_threshold, q_threshold)

    if sell_model_path and sell_mask.any():
        bundle = _load_bundle(str(sell_model_path))
        sell_probability, _ = _score_side(bundle, features.loc[sell_mask], float(sell_threshold), -1)
        probabilities[sell_mask] = sell_probability
        if hasattr(bundle, "threshold_for_rows"):
            thresholds[sell_mask] = np.asarray(bundle.threshold_for_rows(features.loc[sell_mask]), dtype="float64")
        else:
            thresholds[sell_mask] = float(sell_threshold)
        if hasattr(bundle, "predict_gate_proba"):
            gate_probabilities[sell_mask] = np.asarray(bundle.predict_gate_proba(features.loc[sell_mask]), dtype="float64")
            bundle_has_gate = True
        gate_threshold = _finite_or_nan(getattr(bundle, "gate_threshold", np.nan))
        if np.isfinite(gate_threshold):
            gate_thresholds["sell"] = max(
                float(gate_thresholds["sell"]) if np.isfinite(gate_thresholds["sell"]) else float("-inf"),
                float(gate_threshold),
            )
        q_threshold = _finite_or_nan(getattr(bundle, "quality_threshold", np.nan))
        if np.isfinite(q_threshold):
            bundle_quality_threshold = max(bundle_quality_threshold, q_threshold)

    features["primary_probability"] = probabilities
    features["ml_primary_probability"] = probabilities
    features["primary_margin"] = probabilities - thresholds
    features["ml_primary_margin"] = probabilities - thresholds
    if bundle_has_gate:
        features["ml_gate_probability"] = gate_probabilities
    if threshold_policy_path or bundle_has_gate or bundle_quality_threshold > 0.0:
        features = add_policy_columns(features)

    if threshold_policy_path and Path(threshold_policy_path).exists():
        policy_payload = json.loads(Path(threshold_policy_path).read_text(encoding="utf-8"))
    else:
        policy_payload = {
            "default_thresholds": {"buy": float(buy_threshold), "sell": float(sell_threshold)},
            "symbol_thresholds": {},
            "market_state_thresholds": {},
        }

    if bundle_has_gate:
        gate_filter = dict(policy_payload.get("gate_filter") or {})
        gate_filter["enabled"] = True
        gate_model_threshold_mode = str(gate_filter.get("model_threshold_mode", "max")).lower()
        for side_name in ("buy", "sell"):
            gate_side = dict(gate_filter.get(side_name) or {})
            gate_threshold = _finite_or_nan(gate_side.get("min_score", gate_side.get("threshold", np.nan)))
            model_gate_threshold = _finite_or_nan(gate_thresholds.get(side_name, np.nan))
            if gate_model_threshold_mode in {"policy", "override", "ignore"}:
                pass
            elif np.isfinite(gate_threshold) and np.isfinite(model_gate_threshold):
                gate_threshold = max(float(gate_threshold), float(model_gate_threshold))
            elif not np.isfinite(gate_threshold):
                gate_threshold = model_gate_threshold
            if np.isfinite(gate_threshold):
                gate_side["min_score"] = float(gate_threshold)
                gate_filter[side_name] = gate_side
        policy_payload["gate_filter"] = gate_filter

    if bundle_quality_threshold > 0.0:
        quality_filter = dict(policy_payload.get("quality_filter") or {})
        quality_filter["enabled"] = True
        quality_model_threshold_mode = str(quality_filter.get("model_threshold_mode", "max")).lower()
        if quality_model_threshold_mode not in {"policy", "override", "ignore"}:
            quality_filter["min_score"] = max(
                float(quality_filter.get("min_score", 0.0) or 0.0),
                float(bundle_quality_threshold),
            )
        elif "min_score" not in quality_filter:
            quality_filter["min_score"] = float(bundle_quality_threshold)
        policy_payload["quality_filter"] = quality_filter

    policy = CapabilityV2ThresholdPolicy(policy_payload)
    policy_decisions = []
    feature_rows = [row for _, row in features.iterrows()]
    for event, probability, row in zip(raw_events, probabilities, feature_rows):
        side = "buy" if bool(event.is_buy) else "sell"
        policy_decisions.append(policy.decide(event.symbol, side, float(probability), row))
    policy_decisions = _apply_daily_coverage_policy(policy, raw_events, probabilities, feature_rows, policy_decisions)
    qualified = np.asarray([decision.qualified for decision in policy_decisions], dtype=bool)

    scored: List[ScoredSignalEvent] = []
    for idx, (event, probability, is_qualified, decision) in enumerate(zip(raw_events, probabilities, qualified, policy_decisions)):
        raw_signal = 1 if bool(event.is_buy) else -1
        sizing = bsp2_structure_quality_and_sizing(
            feature_rows[idx],
            enabled=True,
            target_risk_pct=0.01,
            min_stake_multiplier=0.15,
            tier_a_max_stake=1.0,
            tier_b_max_stake=0.55,
            tier_c_max_stake=0.25,
            tier_a_risk_boost=3.0,
            tier_b_risk_boost=1.0,
            tier_c_risk_boost=0.6,
            cost_rate=0.001,
        )
        scored.append(
            ScoredSignalEvent(
                symbol=event.symbol,
                exec_time=event_signal_time(event),
                bsp_time=event.bsp_time,
                is_buy=event.is_buy,
                bsp_type=event.bsp_type,
                bsp_types_str=event.bsp_types_str,
                trade_price=event.trade_price,
                klu_idx=event.klu_idx,
                probability=float(probability),
                qualified=bool(is_qualified),
                signal=raw_signal if bool(is_qualified) else 0,
                actual_exec_time=getattr(event, "actual_exec_time", None),
                threshold=float(decision.threshold),
                market_state=str(decision.market_state),
                quality_score=float(decision.quality_score),
                gate_probability=float(gate_probabilities[idx]) if np.isfinite(gate_probabilities[idx]) else float("nan"),
                threshold_reason=str(decision.reason),
                structure_confidence_tier=str(sizing["confidence_tier"]),
                structure_quality_score=float(sizing["structure_quality_score"]),
                structure_risk_pct=float(sizing["structure_risk_pct"]),
                risk_size_multiplier=float(sizing["risk_size_multiplier"]),
                quality_size_multiplier=float(sizing["quality_size_multiplier"]),
                stake_multiplier=float(sizing["stake_multiplier"]),
            )
        )
    return scored
