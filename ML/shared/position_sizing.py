from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PositionSizingConfig:
    enabled: bool = True
    low_score: float = 0.0
    medium_score: float = 0.34
    high_score: float = 0.62
    low_stake_multiplier: float = 0.50
    medium_stake_multiplier: float = 1.00
    high_stake_multiplier: float = 1.50
    low_leverage: float = 1.00
    medium_leverage: float = 2.00
    high_leverage: float = 3.00
    max_leverage: float = 3.00
    quality_floor: float = 0.75
    daily_coverage_score_multiplier: float = 0.70
    high_vol_score_multiplier: float = 0.90
    daily_coverage_max_tier: str = "medium"


@dataclass(frozen=True)
class PositionSizingDecision:
    confidence_score: float
    probability_margin: float
    tier: str
    stake_multiplier: float
    leverage: float
    risk_reason: str


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


def _clip(value: float, low: float, high: float) -> float:
    return float(min(max(float(value), float(low)), float(high)))


def _tier_rank(tier: str) -> int:
    return {"low": 0, "medium": 1, "high": 2}.get(str(tier), 0)


def _cap_tier(tier: str, max_tier: str) -> str:
    if _tier_rank(tier) <= _tier_rank(max_tier):
        return tier
    return str(max_tier) if str(max_tier) in {"low", "medium", "high"} else "medium"


def _reason_float(reason_text: str, name: str, default: float) -> float:
    match = re.search(rf"(?:^|\+){re.escape(name)}=([0-9]+(?:\.[0-9]+)?)", reason_text)
    if not match:
        return float(default)
    return _finite_float(match.group(1), default)


def _reason_tier(reason_text: str, name: str, default: str) -> str:
    match = re.search(rf"(?:^|\+){re.escape(name)}=(low|medium|high)", reason_text)
    if not match:
        return str(default)
    return str(match.group(1))


def decide_position_sizing(
    probability: Any,
    threshold: Any,
    quality_score: Any,
    threshold_reason: str = "",
    market_state: str = "",
    config: PositionSizingConfig | None = None,
) -> PositionSizingDecision:
    cfg = config or PositionSizingConfig()
    if not cfg.enabled:
        return PositionSizingDecision(1.0, 0.0, "medium", 1.0, 1.0, "disabled")

    prob = _finite_float(probability, 0.0)
    th = _clip(_finite_float(threshold, 1.0), 0.0, 1.0)
    quality = _clip(_finite_float(quality_score, cfg.quality_floor), 0.0, 1.0)
    reason_text = str(threshold_reason or "").lower()
    state_text = str(market_state or "").lower()

    margin = max(0.0, prob - th)
    span = max(0.05, 1.0 - th)
    probability_strength = _clip(margin / span, 0.0, 1.0)
    quality_strength = _clip((quality - float(cfg.quality_floor)) / max(1e-9, 1.0 - float(cfg.quality_floor)), 0.0, 1.0)
    score = 0.72 * probability_strength + 0.28 * quality_strength

    risk_reasons: list[str] = []
    if "daily_coverage" in reason_text:
        score *= float(cfg.daily_coverage_score_multiplier)
        risk_reasons.append("daily_coverage")
    if "validation_block" in reason_text or "group_disabled" in reason_text:
        score *= 0.25
        risk_reasons.append("blocked_reason")
    if "high_vol" in state_text:
        score *= float(cfg.high_vol_score_multiplier)
        risk_reasons.append("high_vol")
    if "range_high_vol" in state_text or "trend_mixed_high_vol" in state_text:
        score *= 0.85
        risk_reasons.append("unstable_state")
    elif "range_" in state_text:
        score *= 0.95
        risk_reasons.append("range_state")

    risk_score_multiplier = _reason_float(reason_text, "risk_score_multiplier", 1.0)
    if risk_score_multiplier != 1.0:
        score *= max(0.0, float(risk_score_multiplier))
        risk_reasons.append(f"risk_score_multiplier={risk_score_multiplier:.3f}")

    score = _clip(score, 0.0, 1.0)
    if score >= float(cfg.high_score):
        tier = "high"
    elif score >= float(cfg.medium_score):
        tier = "medium"
    else:
        tier = "low"

    if "daily_coverage" in reason_text:
        tier = _cap_tier(tier, cfg.daily_coverage_max_tier)
    tier = _cap_tier(tier, _reason_tier(reason_text, "risk_max_tier", "high"))

    if tier == "high":
        stake_multiplier = float(cfg.high_stake_multiplier)
        leverage = float(cfg.high_leverage)
    elif tier == "medium":
        stake_multiplier = float(cfg.medium_stake_multiplier)
        leverage = float(cfg.medium_leverage)
    else:
        stake_multiplier = float(cfg.low_stake_multiplier)
        leverage = float(cfg.low_leverage)

    max_leverage = max(1.0, float(cfg.max_leverage))
    risk_leverage_cap = _reason_float(reason_text, "risk_leverage_cap", max_leverage)
    max_leverage = min(max_leverage, max(1.0, float(risk_leverage_cap)))
    leverage = _clip(leverage, 1.0, max_leverage)
    stake_scale = _reason_float(reason_text, "risk_stake_scale", 1.0)
    if stake_scale != 1.0:
        risk_reasons.append(f"risk_stake_scale={stake_scale:.3f}")
    stake_multiplier = max(0.0, float(stake_multiplier) * max(0.0, float(stake_scale)))
    risk_reason = "+".join(risk_reasons) if risk_reasons else "normal"
    return PositionSizingDecision(
        confidence_score=float(score),
        probability_margin=float(margin),
        tier=tier,
        stake_multiplier=float(stake_multiplier),
        leverage=float(leverage),
        risk_reason=risk_reason,
    )


def config_from_mapping(values: dict[str, Any] | None) -> PositionSizingConfig:
    if not values:
        return PositionSizingConfig()
    defaults = PositionSizingConfig()
    return PositionSizingConfig(
        enabled=bool(values.get("enabled", defaults.enabled)),
        low_score=_finite_float(values.get("low_score", defaults.low_score), defaults.low_score),
        medium_score=_finite_float(values.get("medium_score", defaults.medium_score), defaults.medium_score),
        high_score=_finite_float(values.get("high_score", defaults.high_score), defaults.high_score),
        low_stake_multiplier=_finite_float(
            values.get("low_stake_multiplier", defaults.low_stake_multiplier),
            defaults.low_stake_multiplier,
        ),
        medium_stake_multiplier=_finite_float(
            values.get("medium_stake_multiplier", defaults.medium_stake_multiplier),
            defaults.medium_stake_multiplier,
        ),
        high_stake_multiplier=_finite_float(
            values.get("high_stake_multiplier", defaults.high_stake_multiplier),
            defaults.high_stake_multiplier,
        ),
        low_leverage=_finite_float(values.get("low_leverage", defaults.low_leverage), defaults.low_leverage),
        medium_leverage=_finite_float(
            values.get("medium_leverage", defaults.medium_leverage),
            defaults.medium_leverage,
        ),
        high_leverage=_finite_float(values.get("high_leverage", defaults.high_leverage), defaults.high_leverage),
        max_leverage=_finite_float(values.get("max_leverage", defaults.max_leverage), defaults.max_leverage),
        quality_floor=_finite_float(values.get("quality_floor", defaults.quality_floor), defaults.quality_floor),
        daily_coverage_score_multiplier=_finite_float(
            values.get("daily_coverage_score_multiplier", defaults.daily_coverage_score_multiplier),
            defaults.daily_coverage_score_multiplier,
        ),
        high_vol_score_multiplier=_finite_float(
            values.get("high_vol_score_multiplier", defaults.high_vol_score_multiplier),
            defaults.high_vol_score_multiplier,
        ),
        daily_coverage_max_tier=str(values.get("daily_coverage_max_tier", defaults.daily_coverage_max_tier)),
    )
