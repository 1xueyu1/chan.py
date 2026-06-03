from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


BSP2_FAMILY_ORDER = (
    "standard_bsp2",
    "subclass_bsp2s",
    "bsp2_t3_overlap",
    "bsp2_after_bsp1",
    "bsp2_near_origin_zs_boundary",
)


def _finite(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


def _types(row: pd.Series | dict[str, Any]) -> set[str]:
    raw = str(row.get("bsp_types_str", row.get("bsp_type", "")) or "").lower()
    parts = raw.replace("|", ",").replace(";", ",").replace("+", ",").split(",")
    return {p.strip().replace("'", "p") for p in parts if p.strip()}


def _has_type(row: pd.Series | dict[str, Any], prefix: str) -> bool:
    return any(t.startswith(prefix) for t in _types(row))


def _origin_zs_available(row: pd.Series | dict[str, Any]) -> bool:
    return np.isfinite(_finite(row.get("chan_bsp2_origin_zs_low", np.nan))) and np.isfinite(
        _finite(row.get("chan_bsp2_origin_zs_high", np.nan))
    )


def _near_origin_zs_boundary(row: pd.Series | dict[str, Any]) -> bool:
    price = _finite(row.get("trade_price", row.get("entry_price", np.nan)))
    low = _finite(row.get("chan_bsp2_origin_zs_low", np.nan))
    high = _finite(row.get("chan_bsp2_origin_zs_high", np.nan))
    atr_pct = _finite(row.get("bar_atr_pct_14", row.get("tech_15m_atr_pct", np.nan)), 0.0)
    if not (np.isfinite(price) and np.isfinite(low) and np.isfinite(high) and price > 0):
        return False
    boundary_dist = min(abs(price - low), abs(price - high)) / price
    buffer = max(0.003, min(0.02, atr_pct * 1.5 if np.isfinite(atr_pct) and atr_pct > 0 else 0.006))
    return bool(boundary_dist <= buffer)


def bsp2_candidate_family(row: pd.Series | dict[str, Any]) -> str:
    """Classify a raw BSP event into the expanded second-class family.

    The categories are intentionally conservative: every non-standard family
    still has to be structurally adjacent to a second-class BSP concept.
    """
    types = _types(row)
    has_2 = any(t == "2" or t.startswith("2,") for t in types)
    has_2s = any(t.startswith("2s") for t in types)
    has_3 = any(t.startswith("3") for t in types)
    has_1 = any(t.startswith("1") for t in types)

    if has_2 and has_3:
        return "bsp2_t3_overlap"
    if has_2s:
        return "subclass_bsp2s"
    if has_2:
        if _origin_zs_available(row) and _near_origin_zs_boundary(row):
            return "bsp2_near_origin_zs_boundary"
        return "standard_bsp2"
    if has_1 and _origin_zs_available(row) and _near_origin_zs_boundary(row):
        return "bsp2_after_bsp1"
    if has_3 and _origin_zs_available(row) and _near_origin_zs_boundary(row):
        return "bsp2_t3_overlap"
    return "other"


def attach_bsp2_candidate_family(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["bsp2_candidate_family"] = [bsp2_candidate_family(row) for _, row in out.iterrows()]
    out["bsp2_family"] = out["bsp2_candidate_family"]
    out["is_bsp2_family_candidate"] = out["bsp2_candidate_family"].isin(BSP2_FAMILY_ORDER).astype(float)
    for family in BSP2_FAMILY_ORDER:
        out[f"bsp2_family_is_{family}"] = (out["bsp2_candidate_family"] == family).astype(float)
    return out
