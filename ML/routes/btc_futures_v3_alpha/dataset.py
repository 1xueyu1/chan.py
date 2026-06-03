from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ML.routes.btc_futures_v2_stable.dataset import (
    DEFAULT_DATASET_PATH as V2_DATASET_PATH,
    relabel_dataset_with_1m_path,
)
from ML.shared.threshold_policy import market_state_from_row


ROUTE_NAME = "btc_futures_v3_alpha"
DEFAULT_BASE_DATASET = V2_DATASET_PATH
DEFAULT_DATASET_PATH = Path("data/btc_futures_v3_alpha/btc_futures_v3_alpha_dataset.parquet")
DEFAULT_MODEL_DIR = Path("result/ml/btc_futures_v3_alpha")
DEFAULT_BACKTEST_DIR = Path("result/btc_futures_v3_alpha")

STRUCTURE_POOLS = (
    "nested_boundary",
    "rebound_boundary",
    "rebound_mid",
    "zs_departure_resonant",
    "zs_departure_weak",
    "trend_continuation_resonant",
    "trend_continuation_mixed",
    "boundary_probe",
    "direction_conflict",
    "mixed_low_edge",
)

STYLE_NAMES = ("continuation", "rebound", "departure", "nested")
SESSION_NAMES = ("asia", "europe", "us", "off")

V3_FEATURE_PREFIX = "v3_"


def _num(frame: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[col], errors="coerce").astype("float64")


def _max_cols(frame: pd.DataFrame, cols: Iterable[str], default: float = np.nan) -> pd.Series:
    present = [_num(frame, col) for col in cols if col in frame.columns]
    if not present:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.concat(present, axis=1).max(axis=1)


def _mean_cols(frame: pd.DataFrame, cols: Iterable[str], default: float = np.nan) -> pd.Series:
    present = [_num(frame, col) for col in cols if col in frame.columns]
    if not present:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.concat(present, axis=1).mean(axis=1)


def _sum_cols(frame: pd.DataFrame, cols: Iterable[str]) -> pd.Series:
    present = [_num(frame, col) for col in cols if col in frame.columns]
    if not present:
        return pd.Series(0.0, index=frame.index, dtype="float64")
    return pd.concat(present, axis=1).fillna(0.0).sum(axis=1).astype("float64")


def _safe_clip(series: pd.Series, low: float = 0.0, high: float = 1.0) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").clip(lower=low, upper=high).astype("float64")


def _style_name(continuation: float, rebound: float, departure: float, nested: float) -> str:
    values = {
        "continuation": continuation,
        "rebound": rebound,
        "departure": departure,
        "nested": nested,
    }
    clean = {name: (float(value) if np.isfinite(value) else float("-inf")) for name, value in values.items()}
    return max(clean, key=clean.get)


def _pool_name(row: pd.Series) -> str:
    consensus = float(row.get("v3_htf_consensus_score", np.nan))
    conflict = float(row.get("v3_htf_conflict_score", np.nan))
    boundary = float(row.get("v3_boundary_score", np.nan))
    maturity = float(row.get("v3_maturity_score", np.nan))
    continuation = float(row.get("v3_style_continuation_score", np.nan))
    rebound = float(row.get("v3_style_rebound_score", np.nan))
    departure = float(row.get("v3_style_departure_score", np.nan))
    nested = float(row.get("v3_style_nested_score", np.nan))

    consensus = 0.0 if not np.isfinite(consensus) else consensus
    conflict = 0.0 if not np.isfinite(conflict) else conflict
    boundary = 0.0 if not np.isfinite(boundary) else boundary
    maturity = 0.0 if not np.isfinite(maturity) else maturity
    continuation = 0.0 if not np.isfinite(continuation) else continuation
    rebound = 0.0 if not np.isfinite(rebound) else rebound
    departure = 0.0 if not np.isfinite(departure) else departure
    nested = 0.0 if not np.isfinite(nested) else nested

    if conflict >= 0.67 and consensus <= 0.45:
        return "direction_conflict"
    if nested >= 0.45 and boundary >= 0.62:
        return "nested_boundary"
    if rebound >= max(continuation, departure) and rebound >= 0.35:
        return "rebound_boundary" if boundary >= 0.62 else "rebound_mid"
    if departure >= max(continuation, rebound) and departure >= 0.35:
        return "zs_departure_resonant" if consensus >= 0.55 and maturity >= 0.35 else "zs_departure_weak"
    if continuation >= 0.35:
        return "trend_continuation_resonant" if consensus >= 0.60 else "trend_continuation_mixed"
    if boundary >= 0.75:
        return "boundary_probe"
    return "mixed_low_edge"


def _session_name(hour: int) -> str:
    if 0 <= hour < 8:
        return "asia"
    if 8 <= hour < 14:
        return "europe"
    if 14 <= hour < 22:
        return "us"
    return "off"


def required_base_columns_for_v3() -> list[str]:
    cols = {
        "exec_time",
        "is_buy",
        "bsp_types_str",
        "bar_atr_z_64",
        "bar_atr_pct_14",
        "bar_rv_ratio_16_64",
        "bar_trend_efficiency_32",
        "bar_trend_efficiency_96",
        "bar_ma_dist_55",
        "bar_donchian_pos_55",
        "bar_volume_ratio_20",
        "bar_ret_3",
        "bar_ret_8",
        "bar_close_pos",
        "bar_lower_shadow_pct",
        "bar_upper_shadow_pct",
        "cap_micro_confirm_score",
        "cap_chan_15m_structure_maturity_score",
        "cap_chan_structure_consistency_15m_1h_4h_1d",
        "cap_chan_mtf_consensus_strength",
        "cap_chan_mtf_direction_consensus_score",
        "cap_chan_mtf_direction_conflict_score",
        "cap_chan_structure_maturity_mean",
        "cap_chan_structure_boundary_alignment",
        "cap_chan_nested_divergence_score",
        "cap_chan_regime_continuation_strength",
        "cap_chan_regime_rebound_strength",
        "cap_chan_regime_departure_strength",
        "cap_chan_entry_style_strength",
        "cap_chan_structure_purity_score",
        "cap_chan_consensus_x_maturity",
        "cap_chan_boundary_x_maturity",
        "cap_chan_divergence_boundary_x_maturity",
        "cap_chan_rebound_vs_departure",
        "cap_chan_mtf_bi_align_count",
        "cap_chan_mtf_seg_align_count",
        "cap_chan_mtf_zs_mid_align_count",
        "cap_chan_mtf_bi_conflict_count",
        "cap_chan_mtf_seg_conflict_count",
        "cap_chan_mtf_zs_mid_conflict_count",
        "chan_ctx_bsp_inside_last_zs",
        "chan_ctx_bsp_after_last_zs_bi_cnt",
        "chan_ctx_bsp_pos_in_parent_seg",
        "chan_ctx_bsp_bi_idx_pct",
        "chan_ctx_bsp_bi_klu_cnt",
        "chan_ctx_bsp_bi_amp_pct",
        "chan_ctx_bsp_pre_bi_amp_ratio",
        "chan_ctx_bsp_pre2_bi_amp_ratio",
        "chan_ctx_close_vs_last_zs_mid_pct",
        "chan_ctx_close_vs_last_zs_high_pct",
        "chan_ctx_close_vs_last_zs_low_pct",
        "chan_ctx_last_zs_width_pct",
        "chan_ctx_last_zs_bi_cnt",
        "chan_ctx_last_seg_is_up",
        "chan_ctx_last_seg_is_sure",
        "chan_ctx_last_seg_bi_cnt",
        "chan_ctx_last_seg_zs_cnt",
        "chan_ctx_last_seg_multi_zs_cnt",
        "chan_ctx_parent_seg_is_up",
        "chan_ctx_parent_seg_is_sure",
        "chan_ctx_parent_seg_bi_cnt",
    }
    for level in ("1h", "4h", "1d"):
        cols.update(
            {
                f"mtf_{level}_signal_align_ret_3",
                f"mtf_{level}_signal_align_ma_21",
                f"mtf_{level}_event_pos_20",
                f"mtf_{level}_event_pos_55",
                f"mtf_{level}_close_pos_55",
                f"chan_mtf_{level}_event_inside_last_zs",
                f"chan_mtf_{level}_event_pos_last_zs",
                f"chan_mtf_{level}_last_zs_exists",
                f"chan_mtf_{level}_last_zs_is_sure",
                f"chan_mtf_{level}_last_zs_width_pct",
                f"chan_mtf_{level}_last_zs_bi_cnt",
                f"chan_mtf_{level}_signal_align_last_bi",
                f"chan_mtf_{level}_signal_align_last_seg",
                f"chan_mtf_{level}_signal_align_last_zs_mid",
                f"cap_chan_{level}_zs_signal_boundary_score",
                f"cap_chan_{level}_near_signal_boundary",
                f"cap_chan_{level}_near_opposite_boundary",
                f"cap_chan_{level}_inside_sure_zs",
                f"cap_chan_{level}_zs_leave_distance",
                f"cap_chan_{level}_structure_maturity_score",
                f"cap_chan_{level}_divergence_near_signal_boundary",
                f"cap_chan_{level}_divergence_inside_zs",
                f"cap_chan_{level}_divergence_boundary_score",
                f"cap_chan_{level}_trend_continuation_score",
                f"cap_chan_{level}_range_rebound_score",
                f"cap_chan_{level}_zs_departure_score",
            }
        )
    return sorted(cols)


def enhance_btc_futures_v3_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add deterministic, no-lookahead v3 features.

    The function only uses event-time bar/Chan/context columns. Columns derived
    from labels or future trade outcomes are intentionally not read.
    """
    out = frame.copy()
    if out.empty:
        out["v3_structure_pool"] = pd.Series(dtype="object")
        return out

    is_buy = out["is_buy"].astype(bool) if "is_buy" in out.columns else pd.Series(False, index=out.index)
    signal_dir = pd.Series(np.where(is_buy, 1.0, -1.0), index=out.index, dtype="float64")
    out["v3_signal_dir"] = signal_dir

    consensus = _max_cols(
        out,
        [
            "cap_chan_structure_consistency_15m_1h_4h_1d",
            "cap_chan_mtf_consensus_strength",
            "cap_chan_mtf_direction_consensus_score",
        ],
        default=0.5,
    )
    conflict = _max_cols(
        out,
        [
            "cap_chan_mtf_direction_conflict_score",
            "cap_chan_1h_direction_conflict",
            "cap_chan_4h_direction_conflict",
            "cap_chan_1d_direction_conflict",
        ],
        default=0.5,
    )
    maturity = _mean_cols(
        out,
        [
            "cap_chan_15m_structure_maturity_score",
            "cap_chan_1h_structure_maturity_score",
            "cap_chan_4h_structure_maturity_score",
            "cap_chan_1d_structure_maturity_score",
            "cap_chan_structure_maturity_mean",
        ],
        default=0.5,
    )
    boundary = _max_cols(
        out,
        [
            "cap_chan_1h_zs_signal_boundary_score",
            "cap_chan_4h_zs_signal_boundary_score",
            "cap_chan_1d_zs_signal_boundary_score",
            "cap_chan_structure_boundary_alignment",
        ],
        default=0.5,
    )
    inside_levels = _sum_cols(
        out,
        [
            "cap_chan_1h_inside_sure_zs",
            "cap_chan_4h_inside_sure_zs",
            "cap_chan_1d_inside_sure_zs",
        ],
    )
    near_signal_boundary_levels = _sum_cols(
        out,
        [
            "cap_chan_1h_near_signal_boundary",
            "cap_chan_4h_near_signal_boundary",
            "cap_chan_1d_near_signal_boundary",
        ],
    )
    near_opposite_boundary_levels = _sum_cols(
        out,
        [
            "cap_chan_1h_near_opposite_boundary",
            "cap_chan_4h_near_opposite_boundary",
            "cap_chan_1d_near_opposite_boundary",
        ],
    )

    continuation = _max_cols(
        out,
        [
            "cap_chan_1h_trend_continuation_score",
            "cap_chan_4h_trend_continuation_score",
            "cap_chan_1d_trend_continuation_score",
            "cap_chan_regime_continuation_strength",
        ],
        default=0.0,
    )
    rebound = _max_cols(
        out,
        [
            "cap_chan_1h_range_rebound_score",
            "cap_chan_4h_range_rebound_score",
            "cap_chan_1d_range_rebound_score",
            "cap_chan_regime_rebound_strength",
        ],
        default=0.0,
    )
    departure = _max_cols(
        out,
        [
            "cap_chan_1h_zs_departure_score",
            "cap_chan_4h_zs_departure_score",
            "cap_chan_1d_zs_departure_score",
            "cap_chan_regime_departure_strength",
        ],
        default=0.0,
    )
    nested = _max_cols(
        out,
        [
            "cap_chan_1h_divergence_boundary_score",
            "cap_chan_4h_divergence_boundary_score",
            "cap_chan_1d_divergence_boundary_score",
            "cap_chan_nested_divergence_score",
            "cap_chan_divergence_boundary_x_maturity",
        ],
        default=0.0,
    )

    out["v3_htf_consensus_score"] = _safe_clip(consensus)
    out["v3_htf_conflict_score"] = _safe_clip(conflict)
    out["v3_maturity_score"] = _safe_clip(maturity)
    out["v3_boundary_score"] = _safe_clip(boundary)
    out["v3_inside_sure_zs_levels"] = inside_levels
    out["v3_near_signal_boundary_levels"] = near_signal_boundary_levels
    out["v3_near_opposite_boundary_levels"] = near_opposite_boundary_levels
    out["v3_boundary_maturity_score"] = _safe_clip(boundary * maturity)
    out["v3_consensus_maturity_score"] = _safe_clip(consensus * maturity)
    out["v3_conflict_penalty_score"] = _safe_clip(conflict * (1.0 - consensus))

    out["v3_style_continuation_score"] = _safe_clip(continuation)
    out["v3_style_rebound_score"] = _safe_clip(rebound)
    out["v3_style_departure_score"] = _safe_clip(departure)
    out["v3_style_nested_score"] = _safe_clip(nested)
    style_frame = out[
        [
            "v3_style_continuation_score",
            "v3_style_rebound_score",
            "v3_style_departure_score",
            "v3_style_nested_score",
        ]
    ].copy()
    out["v3_style_best_score"] = style_frame.max(axis=1)
    out["v3_style_gap_score"] = style_frame.max(axis=1) - style_frame.apply(
        lambda row: row.nlargest(2).iloc[-1] if row.notna().sum() >= 2 else 0.0,
        axis=1,
    )
    out["v3_structure_purity_score"] = _safe_clip(
        0.25 * out["v3_htf_consensus_score"]
        + 0.20 * out["v3_maturity_score"]
        + 0.20 * out["v3_boundary_score"]
        + 0.15 * out["v3_style_best_score"]
        + 0.10 * _num(out, "cap_chan_structure_purity_score", 0.5)
        + 0.10 * _num(out, "cap_micro_confirm_score", 0.5)
        - 0.18 * out["v3_htf_conflict_score"]
    )

    align_count = _sum_cols(
        out,
        [
            "cap_chan_mtf_bi_align_count",
            "cap_chan_mtf_seg_align_count",
            "cap_chan_mtf_zs_mid_align_count",
        ],
    )
    conflict_count = _sum_cols(
        out,
        [
            "cap_chan_mtf_bi_conflict_count",
            "cap_chan_mtf_seg_conflict_count",
            "cap_chan_mtf_zs_mid_conflict_count",
        ],
    )
    out["v3_mtf_align_count"] = align_count
    out["v3_mtf_conflict_count"] = conflict_count
    out["v3_mtf_align_ratio"] = align_count / (align_count + conflict_count + 1e-12)

    ret_align = _mean_cols(
        out,
        [
            "mtf_1h_signal_align_ret_3",
            "mtf_4h_signal_align_ret_3",
            "mtf_1d_signal_align_ret_3",
            "mtf_1h_signal_align_ma_21",
            "mtf_4h_signal_align_ma_21",
            "mtf_1d_signal_align_ma_21",
        ],
        default=0.5,
    )
    out["v3_price_mtf_align_score"] = _safe_clip(ret_align)

    ctx_mid = _num(out, "chan_ctx_close_vs_last_zs_mid_pct", 0.0)
    ctx_high = _num(out, "chan_ctx_close_vs_last_zs_high_pct", np.nan)
    ctx_low = _num(out, "chan_ctx_close_vs_last_zs_low_pct", np.nan)
    out["v3_15m_zs_mid_signal_side"] = signal_dir * ctx_mid
    out["v3_15m_zs_boundary_distance"] = pd.concat([ctx_high.abs(), ctx_low.abs()], axis=1).min(axis=1)
    out["v3_bsp_inside_last_zs"] = _num(out, "chan_ctx_bsp_inside_last_zs", 0.0)
    out["v3_bsp_after_zs_bi_cnt_norm"] = np.tanh(_num(out, "chan_ctx_bsp_after_last_zs_bi_cnt", 0.0) / 5.0)
    out["v3_bsp_parent_pos_signal"] = np.where(
        is_buy,
        1.0 - _num(out, "chan_ctx_bsp_pos_in_parent_seg", 0.5),
        _num(out, "chan_ctx_bsp_pos_in_parent_seg", 0.5),
    )
    out["v3_bsp_bi_age_score"] = np.tanh(_num(out, "chan_ctx_bsp_bi_klu_cnt", 0.0) / 32.0)
    out["v3_bsp_amp_relative_score"] = _num(out, "chan_ctx_bsp_bi_amp_pct", 0.0) / (
        _num(out, "bar_atr_pct_14", 0.01).abs() + 1e-12
    )
    out["v3_bsp_pre_amp_balance"] = _num(out, "chan_ctx_bsp_pre_bi_amp_ratio", 1.0) - _num(
        out, "chan_ctx_bsp_pre2_bi_amp_ratio", 1.0
    )

    atr_z = _num(out, "bar_atr_z_64", 0.0)
    rv_ratio = _num(out, "bar_rv_ratio_16_64", 1.0)
    trend_eff = _max_cols(out, ["bar_trend_efficiency_32", "bar_trend_efficiency_96"], default=0.0)
    don55 = _num(out, "bar_donchian_pos_55", 0.5)
    ma55 = _num(out, "bar_ma_dist_55", 0.0)
    out["v3_volatility_expansion_score"] = _safe_clip((atr_z + 1.0) / 3.0) * 0.5 + _safe_clip((rv_ratio - 0.6) / 1.4) * 0.5
    out["v3_trend_strength_score"] = _safe_clip(trend_eff)
    out["v3_signal_trend_align_score"] = np.where(
        is_buy,
        ((ma55 >= 0.0) & (don55 >= 0.5)).astype("float64"),
        ((ma55 <= 0.0) & (don55 <= 0.5)).astype("float64"),
    )
    out["v3_signal_close_location_score"] = np.where(
        is_buy,
        _num(out, "bar_close_pos", 0.5),
        1.0 - _num(out, "bar_close_pos", 0.5),
    )
    out["v3_signal_shadow_support_score"] = np.where(
        is_buy,
        _num(out, "bar_lower_shadow_pct", 0.0),
        _num(out, "bar_upper_shadow_pct", 0.0),
    )

    out["v3_alpha_structure_score"] = _safe_clip(
        0.18 * out["v3_structure_purity_score"]
        + 0.16 * out["v3_consensus_maturity_score"]
        + 0.14 * out["v3_boundary_maturity_score"]
        + 0.12 * out["v3_style_best_score"]
        + 0.10 * out["v3_price_mtf_align_score"]
        + 0.10 * out["v3_signal_trend_align_score"]
        + 0.08 * _num(out, "cap_micro_confirm_score", 0.5)
        + 0.07 * out["v3_bsp_parent_pos_signal"]
        + 0.05 * out["v3_bsp_bi_age_score"]
        - 0.16 * out["v3_conflict_penalty_score"]
    )

    styles = [
        _style_name(c, r, d, n)
        for c, r, d, n in zip(
            out["v3_style_continuation_score"],
            out["v3_style_rebound_score"],
            out["v3_style_departure_score"],
            out["v3_style_nested_score"],
        )
    ]
    out["v3_style_name"] = styles
    for style in STYLE_NAMES:
        out[f"v3_style_is_{style}"] = (out["v3_style_name"] == style).astype("float64")

    out["v3_structure_pool"] = out.apply(_pool_name, axis=1)
    for pool in STRUCTURE_POOLS:
        out[f"v3_pool_is_{pool}"] = (out["v3_structure_pool"] == pool).astype("float64")
    out["v3_structure_pool_id"] = out["v3_structure_pool"].map({name: idx for idx, name in enumerate(STRUCTURE_POOLS)}).fillna(-1).astype("float64")

    states = [market_state_from_row(row) for row in out.to_dict("records")]
    out["v3_market_state"] = states
    out["v3_market_is_trend_up"] = pd.Series(states, index=out.index).str.startswith("trend_up").astype("float64")
    out["v3_market_is_trend_down"] = pd.Series(states, index=out.index).str.startswith("trend_down").astype("float64")
    out["v3_market_is_trend_mixed"] = pd.Series(states, index=out.index).str.startswith("trend_mixed").astype("float64")
    out["v3_market_is_range"] = pd.Series(states, index=out.index).str.startswith("range").astype("float64")
    out["v3_market_is_high_vol"] = pd.Series(states, index=out.index).str.contains("high_vol", regex=False).astype("float64")
    out["v3_market_is_low_vol"] = pd.Series(states, index=out.index).str.contains("low_vol", regex=False).astype("float64")

    times = pd.to_datetime(out["exec_time"], utc=True, errors="coerce") if "exec_time" in out.columns else pd.Series(pd.NaT, index=out.index)
    hour = times.dt.hour.fillna(0).astype(int)
    minute = times.dt.minute.fillna(0).astype(int)
    hour_float = hour.astype("float64") + minute.astype("float64") / 60.0
    out["v3_hour_sin"] = np.sin(2.0 * np.pi * hour_float / 24.0)
    out["v3_hour_cos"] = np.cos(2.0 * np.pi * hour_float / 24.0)
    out["v3_weekday"] = times.dt.weekday.fillna(0).astype("float64")
    sessions = [_session_name(int(h)) for h in hour]
    out["v3_session"] = sessions
    for session in SESSION_NAMES:
        out[f"v3_session_is_{session}"] = (out["v3_session"] == session).astype("float64")

    bsp_text = out["bsp_types_str"].astype(str) if "bsp_types_str" in out.columns else pd.Series("", index=out.index)
    out["v3_bsp_has_1"] = bsp_text.str.contains("1", regex=False).astype("float64")
    out["v3_bsp_has_2"] = bsp_text.str.contains("2", regex=False).astype("float64")
    out["v3_bsp_has_3"] = bsp_text.str.contains("3", regex=False).astype("float64")
    out["v3_bsp_has_prime"] = bsp_text.str.contains("p", regex=False).astype("float64")
    out["v3_bsp_has_second"] = bsp_text.str.contains("s", regex=False).astype("float64")

    numeric_v3_cols = [col for col in out.columns if col.startswith(V3_FEATURE_PREFIX) and pd.api.types.is_numeric_dtype(out[col])]
    out[numeric_v3_cols] = out[numeric_v3_cols].replace([np.inf, -np.inf], np.nan)
    return out


def build_btc_futures_v3_alpha_dataset(
    base_dataset: str | Path = DEFAULT_BASE_DATASET,
    output_path: str | Path = DEFAULT_DATASET_PATH,
    force: bool = False,
    ensure_base: bool = True,
) -> pd.DataFrame:
    output = Path(output_path)
    if output.exists() and not force:
        return pd.read_parquet(output)

    base = Path(base_dataset)
    if not base.exists():
        if not ensure_base:
            raise FileNotFoundError(f"base dataset not found: {base}")
        relabel_dataset_with_1m_path(output_path=base, force=False)

    dataset = pd.read_parquet(base)
    for col in ("exec_time", "entry_time", "exit_time", "signal_available_time"):
        if col in dataset.columns:
            dataset[col] = pd.to_datetime(dataset[col], utc=True, errors="coerce")
    dataset = dataset.dropna(subset=["label", "exec_time", "entry_time", "exit_time"]).sort_values("exec_time").reset_index(drop=True)
    out = enhance_btc_futures_v3_features(dataset)
    out["route"] = ROUTE_NAME
    if not bool((out["entry_time"] >= out["exec_time"] + pd.Timedelta(minutes=15)).all()):
        raise RuntimeError("lookahead audit failed: entry_time is earlier than closed signal bar")

    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output, index=False)
    meta = {
        "route": ROUTE_NAME,
        "base_dataset": str(base),
        "output_path": str(output),
        "rows": int(len(out)),
        "columns": int(len(out.columns)),
        "v3_numeric_features": int(
            len([col for col in out.columns if col.startswith(V3_FEATURE_PREFIX) and pd.api.types.is_numeric_dtype(out[col])])
        ),
        "structure_pools": {str(k): int(v) for k, v in out["v3_structure_pool"].value_counts().sort_index().items()},
        "label_rate": float(pd.to_numeric(out["label"], errors="coerce").mean()),
        "avg_net_return": float(pd.to_numeric(out["net_return"], errors="coerce").mean()),
        "lookahead_check": "entry_time >= exec_time + 15min",
    }
    output.with_name("btc_futures_v3_alpha_dataset_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build BTC futures v3 alpha enhanced dataset")
    parser.add_argument("--base-dataset", default=str(DEFAULT_BASE_DATASET))
    parser.add_argument("--output", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-ensure-base", action="store_true")
    args = parser.parse_args()
    frame = build_btc_futures_v3_alpha_dataset(
        base_dataset=args.base_dataset,
        output_path=args.output,
        force=bool(args.force),
        ensure_base=not bool(args.no_ensure_base),
    )
    print(f"built {ROUTE_NAME} dataset rows={len(frame)} output={args.output}")


if __name__ == "__main__":
    main()
