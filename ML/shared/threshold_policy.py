from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd


DEFAULT_BUY_THRESHOLD = 0.74
DEFAULT_SELL_THRESHOLD = 0.74


def normalize_symbol(symbol: str) -> str:
    text = str(symbol or "").upper().replace("/", "").replace(":", "")
    if text.endswith("USDTUSDT"):
        text = text[: -len("USDT")]
    if not text.endswith("USDT"):
        text = f"{text}USDT"
    return text


def _finite_float(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


def _row_value(row: Mapping[str, Any], name: str, default: float = np.nan) -> float:
    if hasattr(row, "get"):
        return _finite_float(row.get(name, default), default)
    return float(default)


def _mean_present(values: list[float], default: float = np.nan) -> float:
    arr = np.asarray([x for x in values if np.isfinite(x)], dtype="float64")
    if arr.size == 0:
        return float(default)
    return float(arr.mean())


def _max_present(values: list[float], default: float = np.nan) -> float:
    arr = np.asarray([x for x in values if np.isfinite(x)], dtype="float64")
    if arr.size == 0:
        return float(default)
    return float(arr.max())


def market_state_from_row(row: Mapping[str, Any]) -> str:
    trend_eff = _max_present(
        [
            _row_value(row, "bar_trend_efficiency_96"),
            _row_value(row, "bar_trend_efficiency_32"),
        ],
        default=0.0,
    )
    atr_z = _row_value(row, "bar_atr_z_64", 0.0)
    rv_ratio = _row_value(row, "bar_rv_ratio_16_64", 1.0)
    ma55 = _row_value(row, "bar_ma_dist_55", 0.0)
    don55 = _row_value(row, "bar_donchian_pos_55", 0.5)

    vol = "high_vol" if atr_z >= 1.0 or rv_ratio >= 1.45 else ("low_vol" if atr_z <= -0.5 and rv_ratio <= 0.85 else "normal_vol")
    if trend_eff >= 0.35:
        if ma55 >= 0.0 and don55 >= 0.55:
            return f"trend_up_{vol}"
        if ma55 <= 0.0 and don55 <= 0.45:
            return f"trend_down_{vol}"
        return f"trend_mixed_{vol}"
    return f"range_{vol}"


def chan_entry_quality_score(row: Mapping[str, Any]) -> float:
    consistency = _max_present(
        [
            _row_value(row, "cap_chan_structure_consistency_15m_1h_4h_1d"),
            _row_value(row, "cap_chan_mtf_direction_consensus_score"),
            _row_value(row, "cap_chan_mtf_consensus_strength"),
        ],
        default=0.5,
    )
    conflict = _row_value(row, "cap_chan_mtf_direction_conflict_score", 0.0)
    boundary = _max_present(
        [
            _row_value(row, "cap_chan_1h_zs_signal_boundary_score"),
            _row_value(row, "cap_chan_4h_zs_signal_boundary_score"),
            _row_value(row, "cap_chan_1d_zs_signal_boundary_score"),
        ],
        default=0.5,
    )
    divergence_boundary = _max_present(
        [
            _row_value(row, "cap_chan_1h_divergence_boundary_score"),
            _row_value(row, "cap_chan_4h_divergence_boundary_score"),
            _row_value(row, "cap_chan_1d_divergence_boundary_score"),
        ],
        default=0.0,
    )
    maturity = _mean_present(
        [
            _row_value(row, "cap_chan_15m_structure_maturity_score"),
            _row_value(row, "cap_chan_1h_structure_maturity_score"),
            _row_value(row, "cap_chan_4h_structure_maturity_score"),
            _row_value(row, "cap_chan_1d_structure_maturity_score"),
            _row_value(row, "cap_chan_structure_maturity_mean"),
        ],
        default=0.5,
    )
    structure_state = _max_present(
        [
            _row_value(row, "cap_chan_1h_trend_continuation_score"),
            _row_value(row, "cap_chan_4h_trend_continuation_score"),
            _row_value(row, "cap_chan_1d_trend_continuation_score"),
            _row_value(row, "cap_chan_1h_range_rebound_score"),
            _row_value(row, "cap_chan_4h_range_rebound_score"),
            _row_value(row, "cap_chan_1d_range_rebound_score"),
            _row_value(row, "cap_chan_1h_zs_departure_score"),
            _row_value(row, "cap_chan_4h_zs_departure_score"),
            _row_value(row, "cap_chan_1d_zs_departure_score"),
            _row_value(row, "cap_chan_entry_style_strength"),
        ],
        default=0.5,
    )
    purity = _row_value(row, "cap_chan_structure_purity_score", 0.5)
    nested_divergence = _row_value(row, "cap_chan_nested_divergence_score", 0.5)

    score = (
        0.22 * consistency
        + 0.16 * boundary
        + 0.12 * divergence_boundary
        + 0.12 * maturity
        + 0.12 * structure_state
        + 0.10 * purity
        + 0.08 * nested_divergence
        - 0.20 * max(0.0, conflict)
    )
    return float(np.clip(score, 0.0, 1.0))


@dataclass(frozen=True)
class ThresholdDecision:
    threshold: float
    qualified: bool
    market_state: str
    quality_score: float
    reason: str


class CapabilityV2ThresholdPolicy:
    def __init__(self, payload: Mapping[str, Any] | None = None):
        payload = dict(payload or {})
        defaults = payload.get("default_thresholds") or {}
        self.buy_threshold = float(defaults.get("buy", payload.get("buy_threshold", DEFAULT_BUY_THRESHOLD)))
        self.sell_threshold = float(defaults.get("sell", payload.get("sell_threshold", DEFAULT_SELL_THRESHOLD)))
        self.symbol_thresholds = payload.get("symbol_thresholds") or {}
        self.market_state_thresholds = payload.get("market_state_thresholds") or {}
        quality = payload.get("quality_filter") or {}
        self.quality_enabled = bool(quality.get("enabled", False))
        self.min_quality = float(quality.get("min_score", 0.0))
        gate = payload.get("gate_filter") or {}
        self.gate_enabled = bool(gate.get("enabled", False))
        self.gate_filter = gate
        self.min_primary_margin = float(gate.get("min_primary_margin", 0.0))
        frequency = payload.get("frequency_filter") or {}
        self.frequency_enabled = bool(frequency.get("enabled", False))
        self.frequency_filter = frequency
        self.probability_margin = float(frequency.get("probability_margin", 0.0))
        group_frequency = payload.get("group_frequency_filter") or {}
        self.group_frequency_enabled = bool(group_frequency.get("enabled", False))
        self.group_frequency_filter = group_frequency
        self.group_frequency_global = group_frequency.get("global") or {}
        self.group_frequency_groups = group_frequency.get("groups") or {}
        self.group_frequency_fallback_order = list(
            group_frequency.get("fallback_order")
            or ["symbol_side_state", "symbol_side", "state_side", "side", "global"]
        )
        structure_pool = payload.get("structure_pool_filter") or {}
        self.structure_pool_enabled = bool(structure_pool.get("enabled", False))
        self.structure_pool_filter = structure_pool
        self.structure_pool_require_configured = bool(structure_pool.get("require_configured_pool", False))
        self.structure_pool_threshold_mode = str(structure_pool.get("threshold_mode", "override")).lower()
        self.structure_pool_groups = structure_pool.get("groups") or {}
        self.structure_pool_global = structure_pool.get("global") or {}
        self.structure_pool_fallback_order = list(
            structure_pool.get("fallback_order")
            or ["pool_side_state", "pool_side", "side_pool", "pool", "side", "global"]
        )
        validation_filter = payload.get("validation_filter") or {}
        self.validation_filter_enabled = bool(validation_filter.get("enabled", False))
        self.validation_filter = validation_filter
        daily_coverage = payload.get("daily_coverage_filter") or {}
        self.daily_coverage_enabled = bool(daily_coverage.get("enabled", False))
        self.daily_coverage_filter = daily_coverage
        self.daily_coverage_start_hour_utc = float(daily_coverage.get("start_hour_utc", 12.0))
        self.daily_coverage_probability_margin = float(daily_coverage.get("probability_margin", -0.10))
        self.daily_coverage_gate_min = float(daily_coverage.get("gate_min", 0.0))
        self.daily_coverage_quality_min = float(daily_coverage.get("quality_min", 0.0))
        self.daily_coverage_max_signals_per_day = int(daily_coverage.get("max_signals_per_day", 1))
        self.daily_coverage_keep_validation_filter = bool(daily_coverage.get("keep_validation_filter", True))
        self.daily_coverage_allow_group_disabled = bool(daily_coverage.get("allow_group_disabled", False))
        risk_profile = payload.get("risk_profile_filter") or {}
        self.risk_profile_enabled = bool(risk_profile.get("enabled", False))
        self.risk_profile_filter = risk_profile
        self.risk_profile_global = risk_profile.get("global") or {}
        self.risk_profile_groups = risk_profile.get("groups") or {}
        self.risk_profile_fallback_order = list(
            risk_profile.get("fallback_order")
            or ["symbol_side_state", "symbol_side", "state_side", "side", "global"]
        )

    @classmethod
    def from_file(cls, path: str | Path | None) -> "CapabilityV2ThresholdPolicy":
        if not path:
            return cls()
        p = Path(path)
        if not p.exists():
            return cls()
        return cls(json.loads(p.read_text(encoding="utf-8")))

    def threshold_for(self, symbol: str, side: str, row: Mapping[str, Any] | None = None) -> tuple[float, str]:
        side = "buy" if str(side).lower() == "buy" else "sell"
        threshold = self.buy_threshold if side == "buy" else self.sell_threshold
        reason = "default"

        symbol_key = normalize_symbol(symbol)
        sym_cfg = self.symbol_thresholds.get(symbol_key) or self.symbol_thresholds.get(str(symbol))
        if isinstance(sym_cfg, Mapping) and side in sym_cfg:
            threshold = float(sym_cfg[side])
            reason = "symbol"

        if row is not None:
            state = market_state_from_row(row)
            state_cfg = self.market_state_thresholds.get(state)
            if isinstance(state_cfg, Mapping) and side in state_cfg:
                threshold = max(float(threshold), float(state_cfg[side]))
                reason = f"{reason}+market_state"

        return float(threshold), reason

    def _group_key(self, symbol: str, side: str, state: str, level: str) -> str:
        symbol_key = normalize_symbol(symbol)
        if level == "symbol_side_state":
            return f"{symbol_key}|{side}|{state}"
        if level == "symbol_side":
            return f"{symbol_key}|{side}"
        if level == "state_side":
            return f"{state}|{side}"
        if level == "side":
            return side
        return "global"

    def _group_cfg_for(self, symbol: str, side: str, state: str) -> tuple[Mapping[str, Any] | None, str]:
        if not self.group_frequency_enabled:
            return None, ""
        groups = self.group_frequency_groups if isinstance(self.group_frequency_groups, Mapping) else {}
        for level in self.group_frequency_fallback_order:
            if level == "global":
                cfg = self.group_frequency_global
                if isinstance(cfg, Mapping):
                    return cfg, "global"
                continue
            key = self._group_key(symbol, side, state, level)
            level_groups = groups.get(level) if isinstance(groups, Mapping) else None
            cfg = None
            if isinstance(level_groups, Mapping):
                cfg = level_groups.get(key)
            if cfg is None and isinstance(groups, Mapping):
                cfg = groups.get(key)
            if isinstance(cfg, Mapping):
                return cfg, level
        if isinstance(self.group_frequency_global, Mapping):
            return self.group_frequency_global, "global"
        return None, ""

    def _structure_pool_key(self, side: str, state: str, pool: str, level: str) -> str:
        if level == "pool_side_state":
            return f"{pool}|{side}|{state}"
        if level == "pool_side":
            return f"{pool}|{side}"
        if level == "side_pool":
            return f"{side}|{pool}"
        if level == "pool":
            return pool
        if level == "side":
            return side
        return "global"

    def _structure_pool_cfg_for(self, side: str, state: str, pool: str) -> tuple[Mapping[str, Any] | None, str]:
        if not self.structure_pool_enabled:
            return None, ""
        groups = self.structure_pool_groups if isinstance(self.structure_pool_groups, Mapping) else {}
        for level in self.structure_pool_fallback_order:
            if level == "global":
                cfg = self.structure_pool_global
                if isinstance(cfg, Mapping):
                    return cfg, "global"
                continue
            key = self._structure_pool_key(side, state, pool, level)
            level_groups = groups.get(level) if isinstance(groups, Mapping) else None
            cfg = None
            if isinstance(level_groups, Mapping):
                cfg = level_groups.get(key)
            if cfg is None and isinstance(groups, Mapping):
                cfg = groups.get(key)
            if isinstance(cfg, Mapping):
                return cfg, level
        if isinstance(self.structure_pool_global, Mapping):
            return self.structure_pool_global, "global"
        return None, ""

    def _risk_profile_cfg_for(self, symbol: str, side: str, state: str) -> tuple[Mapping[str, Any] | None, str]:
        if not self.risk_profile_enabled:
            return None, ""
        groups = self.risk_profile_groups if isinstance(self.risk_profile_groups, Mapping) else {}
        for level in self.risk_profile_fallback_order:
            if level == "global":
                cfg = self.risk_profile_global
                if isinstance(cfg, Mapping):
                    return cfg, "global"
                continue
            key = self._group_key(symbol, side, state, level)
            level_groups = groups.get(level) if isinstance(groups, Mapping) else None
            cfg = None
            if isinstance(level_groups, Mapping):
                cfg = level_groups.get(key)
            if cfg is None and isinstance(groups, Mapping):
                cfg = groups.get(key)
            if isinstance(cfg, Mapping):
                return cfg, level
        if isinstance(self.risk_profile_global, Mapping):
            return self.risk_profile_global, "global"
        return None, ""

    def _append_risk_profile_reason(
        self,
        reason: str,
        symbol: str,
        side: str,
        state: str,
    ) -> tuple[str, bool]:
        cfg, level = self._risk_profile_cfg_for(symbol, side, state)
        if cfg is None:
            return reason, True

        enabled = bool(cfg.get("enabled", True))
        pieces = [reason, f"risk_profile_{level}"]
        if not enabled:
            pieces.append("risk_disabled")
        max_tier = str(cfg.get("max_tier", "") or "").lower()
        if max_tier in {"low", "medium", "high"}:
            pieces.append(f"risk_max_tier={max_tier}")
        for key in ("score_multiplier", "stake_scale", "leverage_cap"):
            if key not in cfg:
                continue
            value = _finite_float(cfg.get(key), np.nan)
            if np.isfinite(value):
                reason_key = {
                    "score_multiplier": "risk_score_multiplier",
                    "stake_scale": "risk_stake_scale",
                    "leverage_cap": "risk_leverage_cap",
                }[key]
                pieces.append(f"{reason_key}={float(value):.6g}")
        return "+".join(piece for piece in pieces if piece), enabled

    def decide(self, symbol: str, side: str, probability: float, row: Mapping[str, Any] | None = None) -> ThresholdDecision:
        row_map: Mapping[str, Any] = row if row is not None else {}
        state = market_state_from_row(row_map)
        quality = chan_entry_quality_score(row_map)
        threshold, reason = self.threshold_for(symbol=symbol, side=side, row=row_map)
        group_cfg, group_level = self._group_cfg_for(symbol, side, state)
        group_enabled = True
        if group_cfg is not None:
            group_enabled = bool(group_cfg.get("enabled", True))
            margin = _finite_float(group_cfg.get("probability_margin", self.probability_margin), self.probability_margin)
            if np.isfinite(float(margin)) and float(margin) != 0.0:
                threshold = float(np.clip(float(threshold) + float(margin), 0.0, 1.0))
                reason = f"{reason}+group_frequency_{group_level}"
        elif self.frequency_enabled:
            side_margins = self.frequency_filter.get("side_probability_margin") or {}
            margin = self.probability_margin
            if isinstance(side_margins, Mapping) and side in side_margins:
                margin = _finite_float(side_margins.get(side), margin)
            if np.isfinite(float(margin)) and float(margin) != 0.0:
                threshold = float(np.clip(float(threshold) + float(margin), 0.0, 1.0))
                reason = f"{reason}+frequency_margin"
        active_filter_cfg: Mapping[str, Any] | None = group_cfg
        if self.structure_pool_enabled:
            pool = str(row_map.get("v3_structure_pool", "") or "")
            pool_cfg, pool_level = self._structure_pool_cfg_for(side, state, pool)
            if pool_cfg is None:
                if self.structure_pool_require_configured:
                    group_enabled = False
                    reason = f"{reason}+structure_pool_missing"
            else:
                pool_enabled = bool(pool_cfg.get("enabled", True))
                if not pool_enabled:
                    group_enabled = False
                    reason = f"{reason}+structure_pool_disabled_{pool_level}"
                pool_threshold = _finite_float(pool_cfg.get("threshold", np.nan), np.nan)
                if np.isfinite(pool_threshold):
                    pool_mode = str(pool_cfg.get("threshold_mode", self.structure_pool_threshold_mode)).lower()
                    if pool_mode in {"override", "replace", "pool"}:
                        threshold = float(pool_threshold)
                    else:
                        threshold = max(float(threshold), float(pool_threshold))
                    reason = f"{reason}+structure_pool_{pool_level}"
                pool_margin = _finite_float(pool_cfg.get("probability_margin", np.nan), np.nan)
                if np.isfinite(pool_margin) and float(pool_margin) != 0.0:
                    threshold = float(np.clip(float(threshold) + float(pool_margin), 0.0, 1.0))
                    reason = f"{reason}+structure_pool_margin"
                merged_cfg = dict(active_filter_cfg or {})
                merged_cfg.update(dict(pool_cfg))
                active_filter_cfg = merged_cfg
        qualified = float(probability) >= threshold
        if not group_enabled:
            qualified = False
            reason = f"{reason}+group_disabled"
        if self.gate_enabled:
            gate_cfg = self.gate_filter.get(side)
            gate_threshold = np.nan
            if isinstance(gate_cfg, Mapping):
                gate_threshold = _finite_float(gate_cfg.get("min_score", gate_cfg.get("threshold", np.nan)))
                if active_filter_cfg is not None and "gate_min" in active_filter_cfg:
                    gate_threshold = _finite_float(active_filter_cfg.get("gate_min"), gate_threshold)
                if np.isfinite(gate_threshold):
                    gate_probability = _row_value(row_map, "ml_gate_probability", np.nan)
                    if not np.isfinite(gate_probability):
                        qualified = False
                        reason = f"{reason}+gate_missing"
                    elif gate_probability < gate_threshold:
                        qualified = False
                        reason = f"{reason}+gate_block"
                    elif self.min_primary_margin > 0.0:
                        primary_margin = _row_value(row_map, "ml_primary_margin", np.nan)
                        if not np.isfinite(primary_margin) or primary_margin < float(self.min_primary_margin):
                            qualified = False
                            reason = f"{reason}+margin_block"
        min_quality = self.min_quality
        if active_filter_cfg is not None and "quality_min" in active_filter_cfg:
            min_quality = _finite_float(active_filter_cfg.get("quality_min"), min_quality)
        if self.quality_enabled and quality < min_quality:
            qualified = False
            reason = f"{reason}+quality_block"
        if self.validation_filter_enabled:
            blocked_states = self.validation_filter.get("blocked_market_states") or {}
            blocked_symbols = self.validation_filter.get("blocked_symbols") or {}
            allowed_states = self.validation_filter.get("allowed_market_states") or {}
            allowed_symbols = self.validation_filter.get("allowed_symbols") or {}
            symbol_key = normalize_symbol(symbol)
            if isinstance(blocked_states, Mapping) and state in set(blocked_states.get(side) or []):
                qualified = False
                reason = f"{reason}+state_validation_block"
            if isinstance(blocked_symbols, Mapping) and symbol_key in set(blocked_symbols.get(side) or []):
                qualified = False
                reason = f"{reason}+symbol_validation_block"
            if isinstance(allowed_states, Mapping) and side in allowed_states and state not in set(allowed_states.get(side) or []):
                qualified = False
                reason = f"{reason}+state_not_allowed"
            if isinstance(allowed_symbols, Mapping) and side in allowed_symbols and symbol_key not in set(allowed_symbols.get(side) or []):
                qualified = False
                reason = f"{reason}+symbol_not_allowed"
        reason, risk_enabled = self._append_risk_profile_reason(reason, symbol, side, state)
        if not risk_enabled:
            qualified = False
        return ThresholdDecision(
            threshold=float(threshold),
            qualified=bool(qualified),
            market_state=state,
            quality_score=float(quality),
            reason=reason,
        )

    def daily_coverage_decide(
        self,
        symbol: str,
        side: str,
        probability: float,
        row: Mapping[str, Any] | None,
        base_decision: ThresholdDecision,
    ) -> ThresholdDecision:
        if not self.daily_coverage_enabled:
            return base_decision
        if "risk_disabled" in str(base_decision.reason):
            return base_decision
        if (not self.daily_coverage_allow_group_disabled) and "group_disabled" in str(base_decision.reason):
            return base_decision

        row_map: Mapping[str, Any] = row if row is not None else {}
        threshold = float(np.clip(float(base_decision.threshold) + float(self.daily_coverage_probability_margin), 0.0, 1.0))
        quality = float(base_decision.quality_score)
        reason = f"{base_decision.reason}+daily_coverage"
        qualified = float(probability) >= threshold

        gate_min = _finite_float(self.daily_coverage_gate_min, np.nan)
        if np.isfinite(gate_min) and gate_min > 0.0:
            gate_probability = _row_value(row_map, "ml_gate_probability", np.nan)
            if not np.isfinite(gate_probability):
                qualified = False
                reason = f"{reason}+gate_missing"
            elif gate_probability < float(gate_min):
                qualified = False
                reason = f"{reason}+gate_block"

        quality_min = _finite_float(self.daily_coverage_quality_min, 0.0)
        if quality < float(quality_min):
            qualified = False
            reason = f"{reason}+quality_block"

        if self.validation_filter_enabled and self.daily_coverage_keep_validation_filter:
            blocked_states = self.validation_filter.get("blocked_market_states") or {}
            blocked_symbols = self.validation_filter.get("blocked_symbols") or {}
            allowed_states = self.validation_filter.get("allowed_market_states") or {}
            allowed_symbols = self.validation_filter.get("allowed_symbols") or {}
            symbol_key = normalize_symbol(symbol)
            state = str(base_decision.market_state)
            if isinstance(blocked_states, Mapping) and state in set(blocked_states.get(side) or []):
                qualified = False
                reason = f"{reason}+state_validation_block"
            if isinstance(blocked_symbols, Mapping) and symbol_key in set(blocked_symbols.get(side) or []):
                qualified = False
                reason = f"{reason}+symbol_validation_block"
            if isinstance(allowed_states, Mapping) and side in allowed_states and state not in set(allowed_states.get(side) or []):
                qualified = False
                reason = f"{reason}+state_not_allowed"
            if isinstance(allowed_symbols, Mapping) and side in allowed_symbols and symbol_key not in set(allowed_symbols.get(side) or []):
                qualified = False
                reason = f"{reason}+symbol_not_allowed"

        return ThresholdDecision(
            threshold=float(threshold),
            qualified=bool(qualified),
            market_state=str(base_decision.market_state),
            quality_score=float(quality),
            reason=reason,
        )


def add_policy_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if out.empty:
        out["cap_v2_market_state"] = pd.Series(dtype="object")
        out["cap_v2_quality_score"] = pd.Series(dtype="float64")
        return out
    out["cap_v2_market_state"] = [market_state_from_row(row) for row in out.to_dict("records")]
    out["cap_v2_quality_score"] = [chan_entry_quality_score(row) for row in out.to_dict("records")]
    return out
