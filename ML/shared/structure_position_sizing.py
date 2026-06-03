from __future__ import annotations

from typing import Any, Mapping

import numpy as np


def _finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


def bsp2_structure_quality_and_sizing(
    row: Mapping[str, Any],
    *,
    enabled: bool,
    target_risk_pct: float = 0.01,
    min_stake_multiplier: float = 0.08,
    tier_a_max_stake: float = 1.0,
    tier_b_max_stake: float = 0.55,
    tier_c_max_stake: float = 0.25,
    tier_a_risk_boost: float = 3.0,
    tier_b_risk_boost: float = 1.0,
    tier_c_risk_boost: float = 0.6,
    cost_rate: float = 0.001,
) -> dict[str, Any]:
    if not enabled:
        return {
            "confidence_tier": "fixed",
            "confidence_score": 1.0,
            "structure_quality_score": float("nan"),
            "structure_risk_pct": float("nan"),
            "risk_size_multiplier": 1.0,
            "quality_size_multiplier": 1.0,
            "stake_multiplier": 1.0,
            "leverage": 1.0,
        }

    is_buy = bool(row.get("is_buy"))
    entry_price = _finite_float(row.get("entry_price", row.get("trade_price", row.get("close", np.nan))))
    invalid_price = _finite_float(row.get("label_bsp2_invalid_price", row.get("chan_bsp2_invalid_price", np.nan)))
    risk_pct = float("nan")
    if np.isfinite(entry_price) and entry_price > 0 and np.isfinite(invalid_price) and invalid_price > 0:
        raw_risk = (entry_price - invalid_price) / entry_price if is_buy else (invalid_price - entry_price) / entry_price
        risk_pct = max(0.0, float(raw_risk + float(cost_rate)))

    break_strength = _finite_float(row.get("chan_bsp_2_break_strength", np.nan))
    retrace_rate = _finite_float(row.get("chan_bsp2_retrace_rate", np.nan))
    support_dist = _finite_float(row.get("chan_fx_signal_support_dist_atr", np.nan))
    close_location = _finite_float(row.get("v3_signal_close_location_score", np.nan))
    ma_conflict = _finite_float(row.get("tech_mtf_ma_conflict_count", np.nan))
    family = str(row.get("bsp2_candidate_family", row.get("bsp2_family", "")) or "").lower()
    mtf_conflict = _finite_float(row.get("cap_chan_mtf_direction_conflict_score", np.nan))
    boundary_alignment = _finite_float(row.get("cap_chan_structure_boundary_alignment", np.nan))
    structure_purity = _finite_float(row.get("cap_chan_structure_purity_score", np.nan))
    identity_prob = _finite_float(row.get("bsp2_identity_prob", np.nan))
    invalid_prob = _finite_float(row.get("bsp2_pre_invalid_prob", np.nan))
    path_weak_prob = _finite_float(row.get("bsp2_path_prob_weak_no_confirm", np.nan))
    path_invalid_prob = _finite_float(row.get("bsp2_path_prob_invalid_before_confirm", np.nan))
    path_pred = str(row.get("bsp2_path_pred", "") or "").lower()

    score = 0
    if np.isfinite(break_strength) and break_strength >= 2.59:
        score += 1
    if np.isfinite(retrace_rate) and retrace_rate <= 0.386:
        score += 1
    if np.isfinite(support_dist) and support_dist >= 1.47:
        score += 1
    if np.isfinite(close_location) and close_location >= 0.846:
        score += 1
    if np.isfinite(ma_conflict) and ma_conflict <= 2:
        score += 1
    if family in {"bsp2_near_origin_zs_boundary", "bsp2_t3_overlap"}:
        score += 1
    if np.isfinite(boundary_alignment) and boundary_alignment >= 0.55:
        score += 1
    if np.isfinite(structure_purity) and structure_purity >= 0.55:
        score += 1
    if np.isfinite(identity_prob) and identity_prob >= 0.70:
        score += 1

    penalty = 0
    if family == "standard_bsp2":
        penalty += 1
    if np.isfinite(retrace_rate) and retrace_rate >= 0.535:
        penalty += 1
    if np.isfinite(close_location) and close_location <= 0.556:
        penalty += 1
    if np.isfinite(ma_conflict) and ma_conflict >= 3:
        penalty += 1
    if np.isfinite(support_dist) and support_dist < 0.887:
        penalty += 1
    if np.isfinite(mtf_conflict) and mtf_conflict >= 0.65:
        penalty += 2
    elif np.isfinite(mtf_conflict) and mtf_conflict >= 0.45:
        penalty += 1
    if np.isfinite(risk_pct) and risk_pct >= 0.025:
        penalty += 1
    if np.isfinite(risk_pct) and risk_pct >= 0.045:
        penalty += 1
    if np.isfinite(identity_prob) and identity_prob < 0.25:
        penalty += 4
    elif np.isfinite(identity_prob) and identity_prob < 0.35:
        penalty += 2
    elif np.isfinite(identity_prob) and identity_prob < 0.50:
        penalty += 1
    if np.isfinite(invalid_prob) and invalid_prob >= 0.80:
        penalty += 4
    elif np.isfinite(invalid_prob) and invalid_prob >= 0.65:
        penalty += 3
    elif np.isfinite(invalid_prob) and invalid_prob >= 0.55:
        penalty += 2
    if np.isfinite(path_invalid_prob) and path_invalid_prob >= 0.70:
        penalty += 3
    elif np.isfinite(path_invalid_prob) and path_invalid_prob >= 0.55:
        penalty += 1
    if np.isfinite(path_weak_prob) and path_weak_prob >= 0.70:
        penalty += 2
    elif np.isfinite(path_weak_prob) and path_weak_prob >= 0.55:
        penalty += 1
    if path_pred == "invalid_before_confirm":
        penalty += 2
    elif path_pred == "weak_no_confirm":
        penalty += 1

    adjusted_score = int(score - penalty)
    if np.isfinite(risk_pct) and risk_pct >= 0.03:
        adjusted_score = min(adjusted_score, 0)
    elif np.isfinite(risk_pct) and risk_pct >= 0.02:
        adjusted_score = min(adjusted_score, 2)
    if np.isfinite(invalid_prob) and invalid_prob >= 0.65:
        adjusted_score = min(adjusted_score, -1)
    if np.isfinite(identity_prob) and identity_prob < 0.25:
        adjusted_score = min(adjusted_score, -2)
    if np.isfinite(path_invalid_prob) and path_invalid_prob >= 0.70:
        adjusted_score = min(adjusted_score, -2)
    tier_a_eligible = True
    if np.isfinite(invalid_prob) and invalid_prob >= 0.10:
        tier_a_eligible = False
    if np.isfinite(identity_prob) and identity_prob < 0.78:
        tier_a_eligible = False
    if np.isfinite(path_invalid_prob) and path_invalid_prob >= 0.10:
        tier_a_eligible = False
    if np.isfinite(mtf_conflict) and mtf_conflict >= 0.55:
        tier_a_eligible = False
    if np.isfinite(risk_pct) and risk_pct >= 0.015:
        tier_a_eligible = False
    if np.isfinite(retrace_rate) and retrace_rate >= 0.55:
        tier_a_eligible = False
    if path_pred == "bsp2_t3_overlap_trend":
        tier_a_eligible = False

    if adjusted_score >= 3 and tier_a_eligible:
        tier = "A"
        quality_multiplier = float(tier_a_max_stake)
        risk_budget = float(target_risk_pct) * float(tier_a_risk_boost)
    elif adjusted_score >= 1:
        tier = "B"
        quality_multiplier = float(tier_b_max_stake)
        risk_budget = float(target_risk_pct) * float(tier_b_risk_boost)
    else:
        tier = "C"
        quality_multiplier = float(tier_c_max_stake)
        risk_budget = float(target_risk_pct) * float(tier_c_risk_boost)

    risk_multiplier = 1.0
    if np.isfinite(risk_pct) and risk_pct > 0 and risk_budget > 0:
        risk_multiplier = min(1.0, float(risk_budget) / float(risk_pct))
    stake_multiplier = max(float(min_stake_multiplier), min(1.0, quality_multiplier * risk_multiplier))

    decision_cap = 1.0
    if np.isfinite(invalid_prob):
        if invalid_prob >= 0.85:
            decision_cap = min(decision_cap, 0.03)
        elif invalid_prob >= 0.80:
            decision_cap = min(decision_cap, 0.05)
        elif invalid_prob >= 0.65:
            decision_cap = min(decision_cap, 0.08)
        elif invalid_prob >= 0.55:
            decision_cap = min(decision_cap, 0.15)
    if np.isfinite(identity_prob):
        if identity_prob <= 0.20:
            decision_cap = min(decision_cap, 0.04)
        elif identity_prob <= 0.25:
            decision_cap = min(decision_cap, 0.06)
        elif identity_prob <= 0.35:
            decision_cap = min(decision_cap, 0.12)
    if np.isfinite(path_invalid_prob):
        if path_invalid_prob >= 0.80:
            decision_cap = min(decision_cap, 0.04)
        elif path_invalid_prob >= 0.70:
            decision_cap = min(decision_cap, 0.07)
        elif path_invalid_prob >= 0.55:
            decision_cap = min(decision_cap, 0.12)
    if np.isfinite(path_weak_prob) and path_weak_prob >= 0.70:
        decision_cap = min(decision_cap, 0.12)
    if path_pred == "invalid_before_confirm":
        decision_cap = min(decision_cap, 0.08)
    elif path_pred == "weak_no_confirm":
        decision_cap = min(decision_cap, 0.15)
    stake_multiplier = max(0.0, min(stake_multiplier, float(decision_cap)))

    return {
        "confidence_tier": tier,
        "confidence_score": float(max(0, min(5, adjusted_score)) / 5.0),
        "structure_quality_score": float(adjusted_score),
        "structure_risk_pct": float(risk_pct) if np.isfinite(risk_pct) else float("nan"),
        "risk_size_multiplier": float(risk_multiplier),
        "quality_size_multiplier": float(quality_multiplier),
        "decision_size_cap": float(decision_cap),
        "stake_multiplier": float(stake_multiplier),
        "leverage": 1.0,
    }
