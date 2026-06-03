from __future__ import annotations

import json
import os
import warnings
from dataclasses import asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd
import numpy as np
from pandas.errors import PerformanceWarning

from Backtest.chan_signal_extractor import extract_raw_bsp_events_from_bars
from Backtest.config import BacktestConfig
from ChanConfig import CChanConfig
from Common.CEnum import KL_TYPE
from ML.features import (
    build_capability_chan_feature_frame,
    build_capability_context_feature_frame,
    build_event_features,
    normalize_bars,
)
from RustCore import rust_config_path_from_chan_config

from .data import (
    DEFAULT_DATASET_PATH,
    DEFAULT_FUTURES_DATA_DIR,
    DEFAULT_SOURCE_PATH,
    PEER_MARKET_SYMBOLS,
    SYMBOL,
    TRAIN_SYMBOLS,
    futures_1m_path,
    load_1m_futures_bars,
    normalize_symbol,
    parse_utc,
    resample_ohlcv_open_labeled,
    symbol_asset,
)
from .labels import Bsp2StructureLabelConfig, label_bsp2_structure_with_1m_path


DEFAULT_CACHE_DIR = Path("data/btc_futures_v2_bsp2_family/cache")
LABEL_PREFIXES = ("label_",)
OUTCOME_COLUMNS = {
    "signal_available_time",
    "entry_time",
    "entry_price",
    "exit_time",
    "exit_price",
    "exit_reason",
    "gross_return",
    "net_return",
    "mfe",
    "mae",
    "entry_bar_idx",
    "route",
    "source_path",
    "label_max_holding_minutes",
}


def make_chan_config(symbol: str = SYMBOL) -> BacktestConfig:
    cfg = BacktestConfig(
        symbols=[normalize_symbol(symbol)],
        begin_time="1970-01-01",
        end_time="2099-12-31",
        kl_type=KL_TYPE.K_15M,
        allow_short=True,
    )
    cfg.chan_config.update(
        {
            "use_rust_core": True,
            "mtf_chan_features": True,
            "trigger_step": True,
            "skip_step": 0,
            "bi_strict": True,
            "bs_type": "1,2,3a,1p,2s,3b",
            "min_zs_cnt": 1,
            "bsp1_only_multibi_zs": True,
            "bsp2_follow_1": True,
            "bsp3_follow_1": True,
            "bs1_peak": True,
        }
    )
    return cfg


def _num(frame: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(default, index=frame.index, dtype="float64")
    return pd.to_numeric(frame[col], errors="coerce").astype("float64")


def _safe_clip(series: pd.Series, low: float = 0.0, high: float = 1.0) -> pd.Series:
    values = pd.Series(series) if not isinstance(series, pd.Series) else series
    return pd.to_numeric(values, errors="coerce").clip(lower=low, upper=high).astype("float64")


def enhance_chan_bi_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add Chan BI-focused features for v1.

    These features only combine event-time Chan context columns already
    generated before labeling. They do not read future path or outcome columns.
    """
    out = frame.copy()
    if out.empty:
        return out

    is_buy = out["is_buy"].astype(bool) if "is_buy" in out.columns else pd.Series(False, index=out.index)
    signal_dir = pd.Series(np.where(is_buy, 1.0, -1.0), index=out.index, dtype="float64")

    bsp_bi_up = _num(out, "chan_ctx_bsp_bi_is_up")
    last_bi_up = _num(out, "chan_ctx_last_bi_is_up")
    bsp_bi_sure = _num(out, "chan_ctx_bsp_bi_is_sure", 0.0)
    last_bi_sure = _num(out, "chan_ctx_last_bi_is_sure", 0.0)
    bsp_bi_klu = _num(out, "chan_ctx_bsp_bi_klu_cnt", 0.0)
    bsp_bi_klc = _num(out, "chan_ctx_bsp_bi_klc_cnt", 0.0)
    last_bi_klu = _num(out, "chan_ctx_last_bi_klu_cnt", 0.0)
    last_bi_klc = _num(out, "chan_ctx_last_bi_klc_cnt", 0.0)
    bsp_amp = _num(out, "chan_ctx_bsp_bi_amp_pct", 0.0).abs()
    last_amp = _num(out, "chan_ctx_last_bi_amp_pct", 0.0).abs()
    recent_amp_mean = _num(out, "chan_ctx_recent_bi_amp_mean", np.nan).abs()
    recent_amp_std = _num(out, "chan_ctx_recent_bi_amp_std", np.nan).abs()
    recent_klu_mean = _num(out, "chan_ctx_recent_bi_klu_mean", np.nan)
    recent_up_cnt = _num(out, "chan_ctx_recent_bi_up_cnt", np.nan)
    recent_down_cnt = _num(out, "chan_ctx_recent_bi_down_cnt", np.nan)
    recent_cnt = recent_up_cnt + recent_down_cnt
    atr = _num(out, "bar_atr_pct_14", 0.01).abs()

    out["chan_bi_15m_bsp_direction_align"] = np.where(
        np.isfinite(bsp_bi_up),
        np.where(is_buy, bsp_bi_up <= 0.5, bsp_bi_up > 0.5).astype("float64"),
        np.nan,
    )
    out["chan_bi_15m_last_direction_align"] = np.where(
        np.isfinite(last_bi_up),
        np.where(is_buy, last_bi_up <= 0.5, last_bi_up > 0.5).astype("float64"),
        np.nan,
    )
    out["chan_bi_15m_bsp_last_same_direction"] = np.where(
        np.isfinite(bsp_bi_up + last_bi_up),
        (bsp_bi_up.round() == last_bi_up.round()).astype("float64"),
        np.nan,
    )
    out["chan_bi_15m_bsp_maturity"] = _safe_clip(np.tanh(bsp_bi_klu / 32.0) * bsp_bi_sure)
    out["chan_bi_15m_last_maturity"] = _safe_clip(np.tanh(last_bi_klu / 32.0) * last_bi_sure)
    out["chan_bi_15m_bsp_klu_norm"] = np.tanh(bsp_bi_klu / 48.0)
    out["chan_bi_15m_bsp_klc_norm"] = np.tanh(bsp_bi_klc / 16.0)
    out["chan_bi_15m_last_klu_norm"] = np.tanh(last_bi_klu / 48.0)
    out["chan_bi_15m_bsp_amp_atr_ratio"] = bsp_amp / (atr + 1e-12)
    out["chan_bi_15m_bsp_amp_recent_ratio"] = bsp_amp / (recent_amp_mean + 1e-12)
    out["chan_bi_15m_bsp_amp_zscore"] = (bsp_amp - recent_amp_mean) / (recent_amp_std + 1e-12)
    out["chan_bi_15m_bsp_klu_recent_ratio"] = bsp_bi_klu / (recent_klu_mean + 1e-12)
    out["chan_bi_15m_bsp_speed"] = bsp_amp / (bsp_bi_klu + 1.0)
    out["chan_bi_15m_last_speed"] = last_amp / (last_bi_klu + 1.0)
    out["chan_bi_15m_bsp_speed_vs_last"] = out["chan_bi_15m_bsp_speed"] / (out["chan_bi_15m_last_speed"].abs() + 1e-12)
    out["chan_bi_15m_bsp_pre_amp_ratio"] = _num(out, "chan_ctx_bsp_pre_bi_amp_ratio", np.nan)
    out["chan_bi_15m_bsp_pre2_amp_ratio"] = _num(out, "chan_ctx_bsp_pre2_bi_amp_ratio", np.nan)
    out["chan_bi_15m_bsp_amp_accel"] = out["chan_bi_15m_bsp_pre_amp_ratio"] - out["chan_bi_15m_bsp_pre2_amp_ratio"]
    out["chan_bi_15m_bsp_pre_klu_ratio"] = _num(out, "chan_ctx_bsp_pre_bi_klu_ratio", np.nan)
    out["chan_bi_15m_recent_up_ratio"] = recent_up_cnt / (recent_cnt + 1e-12)
    out["chan_bi_15m_recent_signal_bias"] = np.where(
        is_buy,
        out["chan_bi_15m_recent_up_ratio"],
        1.0 - out["chan_bi_15m_recent_up_ratio"],
    )
    out["chan_bi_15m_recent_direction_imbalance"] = (recent_up_cnt - recent_down_cnt).abs() / (recent_cnt + 1e-12)

    idx_pct = _num(out, "chan_ctx_bsp_bi_idx_pct", np.nan)
    parent_pos = _num(out, "chan_ctx_bsp_pos_in_parent_seg", np.nan)
    out["chan_bi_15m_bsp_idx_pct"] = idx_pct
    out["chan_bi_15m_bsp_idx_edge_score"] = (idx_pct - 0.5).abs() * 2.0
    out["chan_bi_15m_bsp_parent_pos_signal"] = np.where(is_buy, 1.0 - parent_pos, parent_pos)
    out["chan_bi_15m_bsp_is_last_bi"] = _num(out, "chan_ctx_bsp_is_last_bi", np.nan)
    out["chan_bi_15m_bsp_after_zs_norm"] = np.tanh(_num(out, "chan_ctx_bsp_after_last_zs_bi_cnt", 0.0) / 5.0)
    out["chan_bi_15m_bsp_inside_zs"] = _num(out, "chan_ctx_bsp_inside_last_zs", 0.0)
    out["chan_bi_15m_bsp_leave_zs_score"] = out["chan_bi_15m_bsp_after_zs_norm"] * (1.0 - out["chan_bi_15m_bsp_inside_zs"])
    out["chan_bi_15m_bsp_quality_score"] = _safe_clip(
        0.20 * out["chan_bi_15m_bsp_direction_align"]
        + 0.18 * out["chan_bi_15m_bsp_maturity"]
        + 0.15 * _safe_clip(out["chan_bi_15m_bsp_amp_atr_ratio"] / 3.0)
        + 0.14 * _safe_clip(out["chan_bi_15m_bsp_amp_recent_ratio"] / 2.0)
        + 0.12 * out["chan_bi_15m_recent_signal_bias"]
        + 0.11 * _safe_clip(out["chan_bi_15m_bsp_parent_pos_signal"])
        + 0.10 * _safe_clip(out["chan_bi_15m_bsp_leave_zs_score"])
    )

    mtf_align_cols: list[pd.Series] = []
    mtf_maturity_cols: list[pd.Series] = []
    mtf_pos_cols: list[pd.Series] = []
    mtf_amp_cols: list[pd.Series] = []
    for level in ("1h", "4h", "1d"):
        prefix = f"chan_mtf_{level}"
        align = _num(out, f"{prefix}_signal_align_last_bi", np.nan)
        sure = _num(out, f"{prefix}_last_bi_is_sure", np.nan)
        klu = _num(out, f"{prefix}_last_bi_klu_cnt", np.nan)
        amp = _num(out, f"{prefix}_last_bi_amp_pct", np.nan).abs()
        pos = _num(out, f"{prefix}_event_pos_last_bi", np.nan)
        end_dist = _num(out, f"{prefix}_event_vs_last_bi_end_pct", np.nan)
        out[f"chan_bi_{level}_align"] = align
        out[f"chan_bi_{level}_maturity"] = _safe_clip(np.tanh(klu / 32.0) * sure)
        out[f"chan_bi_{level}_amp_atr_ratio"] = amp / (atr + 1e-12)
        out[f"chan_bi_{level}_event_pos_signal"] = np.where(is_buy, pos, 1.0 - pos)
        out[f"chan_bi_{level}_event_near_bi_end"] = np.exp(-end_dist.abs() * 50.0)
        mtf_align_cols.append(out[f"chan_bi_{level}_align"])
        mtf_maturity_cols.append(out[f"chan_bi_{level}_maturity"])
        mtf_pos_cols.append(out[f"chan_bi_{level}_event_pos_signal"])
        mtf_amp_cols.append(out[f"chan_bi_{level}_amp_atr_ratio"])

    align_frame = pd.concat(mtf_align_cols, axis=1)
    maturity_frame = pd.concat(mtf_maturity_cols, axis=1)
    pos_frame = pd.concat(mtf_pos_cols, axis=1)
    amp_frame = pd.concat(mtf_amp_cols, axis=1)
    out["chan_bi_mtf_align_ratio"] = align_frame.mean(axis=1)
    out["chan_bi_mtf_align_count"] = align_frame.sum(axis=1)
    out["chan_bi_mtf_conflict_count"] = align_frame.notna().sum(axis=1) - align_frame.sum(axis=1)
    out["chan_bi_mtf_maturity_mean"] = maturity_frame.mean(axis=1)
    out["chan_bi_mtf_event_pos_signal_mean"] = pos_frame.mean(axis=1)
    out["chan_bi_mtf_amp_atr_ratio_mean"] = amp_frame.mean(axis=1)
    out["chan_bi_mtf_resonance_score"] = _safe_clip(
        0.35 * out["chan_bi_mtf_align_ratio"]
        + 0.25 * out["chan_bi_mtf_maturity_mean"]
        + 0.20 * _safe_clip(out["chan_bi_mtf_event_pos_signal_mean"])
        + 0.20 * _safe_clip(out["chan_bi_mtf_amp_atr_ratio_mean"] / 3.0)
    )
    out["chan_bi_total_edge_score"] = _safe_clip(
        0.55 * out["chan_bi_15m_bsp_quality_score"]
        + 0.35 * out["chan_bi_mtf_resonance_score"]
        + 0.10 * _num(out, "v3_alpha_structure_score", 0.5)
    )

    bi_cols = [col for col in out.columns if col.startswith("chan_bi_") and pd.api.types.is_numeric_dtype(out[col])]
    out[bi_cols] = out[bi_cols].replace([np.inf, -np.inf], np.nan)
    return out


def enhance_chan_zs_bsp_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add richer Zhongshu and BSP-derived features for the v1 route."""
    out = frame.copy()
    if out.empty:
        return out

    is_buy = out["is_buy"].astype(bool) if "is_buy" in out.columns else pd.Series(False, index=out.index)
    atr = _num(out, "bar_atr_pct_14", 0.01).abs()
    zs_width = _num(out, "chan_ctx_last_zs_width_pct", np.nan).abs()
    zs_peak_width = _num(out, "chan_ctx_last_zs_peak_width_pct", np.nan).abs()
    zs_bi_cnt = _num(out, "chan_ctx_last_zs_bi_cnt", np.nan)
    zs_cnt = _num(out, "chan_ctx_zs_cnt", 0.0)
    close_mid = _num(out, "chan_ctx_close_vs_last_zs_mid_pct", np.nan)
    close_high = _num(out, "chan_ctx_close_vs_last_zs_high_pct", np.nan)
    close_low = _num(out, "chan_ctx_close_vs_last_zs_low_pct", np.nan)
    inside = _num(out, "chan_ctx_bsp_inside_last_zs", 0.0)
    after_zs = _num(out, "chan_ctx_bsp_after_last_zs_bi_cnt", 0.0)

    out["chan_zs_15m_width_atr_ratio"] = zs_width / (atr + 1e-12)
    out["chan_zs_15m_peak_width_ratio"] = zs_peak_width / (zs_width + 1e-12)
    out["chan_zs_15m_maturity"] = _safe_clip(np.tanh(zs_bi_cnt / 5.0))
    out["chan_zs_15m_density_in_context"] = zs_cnt / (_num(out, "chan_ctx_bi_cnt", 0.0) + 1e-12)
    out["chan_zs_15m_inside"] = inside
    out["chan_zs_15m_after_bi_norm"] = np.tanh(after_zs / 5.0)
    out["chan_zs_15m_mid_signal_side"] = np.where(is_buy, -close_mid, close_mid)
    out["chan_zs_15m_signal_boundary_dist"] = np.where(is_buy, close_low.abs(), close_high.abs())
    out["chan_zs_15m_opposite_boundary_dist"] = np.where(is_buy, close_high.abs(), close_low.abs())
    out["chan_zs_15m_near_signal_boundary_score"] = np.exp(-out["chan_zs_15m_signal_boundary_dist"].abs() / (zs_width.abs() + 1e-4))
    out["chan_zs_15m_near_opposite_boundary_score"] = np.exp(-out["chan_zs_15m_opposite_boundary_dist"].abs() / (zs_width.abs() + 1e-4))
    out["chan_zs_15m_leave_score"] = out["chan_zs_15m_after_bi_norm"] * (1.0 - inside)
    out["chan_zs_15m_rebound_score"] = _safe_clip(
        inside * out["chan_zs_15m_near_signal_boundary_score"] * out["chan_zs_15m_maturity"]
    )
    out["chan_zs_15m_departure_score"] = _safe_clip(
        (1.0 - inside) * out["chan_zs_15m_after_bi_norm"] * out["chan_zs_15m_maturity"]
    )
    out["chan_zs_15m_mid_reclaim_score"] = _safe_clip(
        np.where(is_buy, (close_mid >= 0.0).astype("float64"), (close_mid <= 0.0).astype("float64"))
        * out["chan_zs_15m_maturity"]
    )

    mtf_inside: list[pd.Series] = []
    mtf_maturity: list[pd.Series] = []
    mtf_boundary: list[pd.Series] = []
    mtf_departure: list[pd.Series] = []
    for level in ("1h", "4h", "1d"):
        prefix = f"chan_mtf_{level}"
        pos = _num(out, f"{prefix}_event_pos_last_zs", np.nan)
        exists = _num(out, f"{prefix}_last_zs_exists", np.nan)
        sure = _num(out, f"{prefix}_last_zs_is_sure", np.nan)
        width = _num(out, f"{prefix}_last_zs_width_pct", np.nan).abs()
        bi_cnt = _num(out, f"{prefix}_last_zs_bi_cnt", np.nan)
        level_inside = _num(out, f"{prefix}_event_inside_last_zs", np.nan)
        mid_align = _num(out, f"{prefix}_signal_align_last_zs_mid", np.nan)
        signal_boundary = np.where(is_buy, 1.0 - pos, pos)
        maturity = _safe_clip(np.tanh(bi_cnt / 5.0) * sure)
        departure = _safe_clip((1.0 - level_inside.fillna(0.0)) * mid_align.fillna(0.0) * maturity)
        out[f"chan_zs_{level}_inside"] = level_inside
        out[f"chan_zs_{level}_maturity"] = maturity
        out[f"chan_zs_{level}_width_atr_ratio"] = width / (atr + 1e-12)
        out[f"chan_zs_{level}_signal_boundary_score"] = _safe_clip(signal_boundary)
        out[f"chan_zs_{level}_mid_align"] = mid_align
        out[f"chan_zs_{level}_departure_score"] = departure.where(exists > 0.0)
        mtf_inside.append(level_inside)
        mtf_maturity.append(maturity)
        mtf_boundary.append(out[f"chan_zs_{level}_signal_boundary_score"])
        mtf_departure.append(out[f"chan_zs_{level}_departure_score"])

    out["chan_zs_mtf_inside_count"] = pd.concat(mtf_inside, axis=1).sum(axis=1)
    out["chan_zs_mtf_maturity_mean"] = pd.concat(mtf_maturity, axis=1).mean(axis=1)
    out["chan_zs_mtf_signal_boundary_mean"] = pd.concat(mtf_boundary, axis=1).mean(axis=1)
    out["chan_zs_mtf_departure_mean"] = pd.concat(mtf_departure, axis=1).mean(axis=1)
    out["chan_zs_total_rebound_score"] = _safe_clip(
        0.45 * out["chan_zs_15m_rebound_score"]
        + 0.25 * out["chan_zs_mtf_signal_boundary_mean"]
        + 0.20 * out["chan_zs_mtf_maturity_mean"]
        + 0.10 * out["chan_zs_mtf_inside_count"] / 3.0
    )
    out["chan_zs_total_departure_score"] = _safe_clip(
        0.45 * out["chan_zs_15m_departure_score"]
        + 0.35 * out["chan_zs_mtf_departure_mean"]
        + 0.20 * out["chan_zs_mtf_maturity_mean"]
    )

    bsp_text = out["bsp_types_str"].astype(str).str.lower() if "bsp_types_str" in out.columns else pd.Series("", index=out.index)
    out["chan_bsp_family_1"] = bsp_text.str.contains("1", regex=False).astype("float64")
    out["chan_bsp_family_2"] = bsp_text.str.contains("2", regex=False).astype("float64")
    out["chan_bsp_family_3"] = bsp_text.str.contains("3", regex=False).astype("float64")
    out["chan_bsp_variant_prime"] = bsp_text.str.contains("p", regex=False).astype("float64")
    out["chan_bsp_variant_second"] = bsp_text.str.contains("s", regex=False).astype("float64")
    out["chan_bsp_variant_3a"] = bsp_text.str.contains("3a", regex=False).astype("float64")
    out["chan_bsp_variant_3b"] = bsp_text.str.contains("3b", regex=False).astype("float64")
    bsp_amp = _num(out, "chan_bsp_bi_amp", np.nan).abs()
    out["chan_bsp_amp_atr_ratio"] = bsp_amp / (atr + 1e-12)
    out["chan_bsp_type_complexity"] = (
        out["chan_bsp_family_1"] + out["chan_bsp_family_2"] + out["chan_bsp_family_3"]
        + out["chan_bsp_variant_prime"] + out["chan_bsp_variant_second"]
    )
    out["chan_bsp_2_retrace_control"] = 1.0 - _num(out, "chan_bsp2_retrace_rate", np.nan).abs()
    out["chan_bsp_2_break_strength"] = _num(out, "chan_bsp2_break_bi_amp", np.nan).abs() / (bsp_amp + 1e-12)
    out["chan_bsp_2s_retrace_control"] = 1.0 - _num(out, "chan_bsp2s_retrace_rate", np.nan).abs()
    out["chan_bsp_2s_break_strength"] = _num(out, "chan_bsp2s_break_bi_amp", np.nan).abs() / (bsp_amp + 1e-12)
    out["chan_bsp_3_zs_height_atr_ratio"] = _num(out, "chan_bsp3_zs_height", np.nan).abs() / (atr + 1e-12)
    out["chan_bsp_parent_seg_end_score"] = _num(out, "chan_ctx_parent_seg_end_is_bsp", 0.0)
    out["chan_bsp_bars_since_signal_norm"] = np.tanh(_num(out, "chan_ctx_bars_since_bsp_klu", 0.0) / 8.0)
    out["chan_bsp_context_score"] = _safe_clip(
        0.20 * _num(out, "chan_bi_total_edge_score", 0.5)
        + 0.20 * out["chan_zs_total_rebound_score"].fillna(0.0)
        + 0.20 * out["chan_zs_total_departure_score"].fillna(0.0)
        + 0.15 * _safe_clip(out["chan_bsp_amp_atr_ratio"] / 3.0)
        + 0.15 * out["chan_bsp_parent_seg_end_score"]
        + 0.10 * (1.0 - out["chan_bsp_bars_since_signal_norm"])
    )

    new_cols = [
        col
        for col in out.columns
        if (col.startswith("chan_zs_") or col.startswith("chan_bsp_"))
        and pd.api.types.is_numeric_dtype(out[col])
    ]
    out[new_cols] = out[new_cols].replace([np.inf, -np.inf], np.nan)
    return out


def _market_state_from_15m(bars_15m: pd.DataFrame, prefix: str) -> pd.DataFrame:
    close = pd.to_numeric(bars_15m["close"], errors="coerce")
    high = pd.to_numeric(bars_15m["high"], errors="coerce")
    low = pd.to_numeric(bars_15m["low"], errors="coerce")
    volume = pd.to_numeric(bars_15m["volume"], errors="coerce").fillna(0.0)
    out = pd.DataFrame(index=bars_15m.index)
    ret1 = close.pct_change()
    out[f"{prefix}_ret_1"] = ret1
    for window in (4, 16, 96):
        out[f"{prefix}_ret_{window}"] = close.pct_change(window)
    for window in (16, 96):
        rv = ret1.rolling(window, min_periods=max(4, window // 4)).std()
        out[f"{prefix}_rv_{window}"] = rv
    out[f"{prefix}_rv_ratio_16_96"] = out[f"{prefix}_rv_16"] / (out[f"{prefix}_rv_96"] + 1e-12)
    for window in (21, 96):
        ma = close.rolling(window, min_periods=max(5, window // 4)).mean()
        out[f"{prefix}_ma_dist_{window}"] = (close - ma) / (ma + 1e-12)
    roll_high = high.rolling(96, min_periods=24).max()
    roll_low = low.rolling(96, min_periods=24).min()
    out[f"{prefix}_range_pos_96"] = (close - roll_low) / (roll_high - roll_low + 1e-12)
    vol20 = volume.rolling(20, min_periods=5).mean()
    vol96 = volume.rolling(96, min_periods=24).mean()
    out[f"{prefix}_volume_ratio_20_96"] = vol20 / (vol96 + 1e-12)
    return out.replace([np.inf, -np.inf], np.nan)


def build_cross_market_feature_frame(
    events: pd.DataFrame,
    btc_bars_15m: pd.DataFrame,
    *,
    data_dir: str | Path = DEFAULT_FUTURES_DATA_DIR,
    peer_symbols: tuple[str, ...] = PEER_MARKET_SYMBOLS,
    begin_time: Any | None = None,
    end_time: Any | None = None,
) -> pd.DataFrame:
    """Build no-lookahead cross-market features sampled at signal close.

    Peer 1m data is resampled to open-labeled 15m bars. Event rows sample the
    latest 15m bar whose open time is <= exec_time, which matches the existing
    v1 signal availability rule: the bar is tradable only after exec_time + 15m.
    """
    if events.empty:
        return pd.DataFrame(index=events.index)

    btc_state = _market_state_from_15m(btc_bars_15m, "btc_ref")
    btc_ret = pd.to_numeric(btc_state["btc_ref_ret_1"], errors="coerce")
    state_parts: list[pd.DataFrame] = []
    peer_ret_cols: list[str] = []
    peer_ret16_cols: list[str] = []
    peer_rv_cols: list[str] = []
    peer_volume_cols: list[str] = []

    load_begin = parse_utc(begin_time)
    if load_begin is not None:
        load_begin = load_begin - pd.Timedelta(days=5)
    load_end = parse_utc(end_time, end_of_day=True)
    for symbol in peer_symbols:
        path = futures_1m_path(symbol, data_dir)
        if not path.exists():
            continue
        try:
            peer_1m = load_1m_futures_bars(path, begin_time=load_begin, end_time=load_end)
        except Exception:
            continue
        if peer_1m.empty:
            continue
        peer_15m = resample_ohlcv_open_labeled(peer_1m, "15min")
        prefix = f"peer_{symbol.lower()}"
        peer_state = _market_state_from_15m(peer_15m, prefix).reindex(btc_bars_15m.index)
        peer_ret = pd.to_numeric(peer_state[f"{prefix}_ret_1"], errors="coerce")
        for window in (1, 4, 16, 96):
            peer_state[f"{prefix}_rel_ret_{window}"] = peer_state[f"{prefix}_ret_{window}"] - btc_state[f"btc_ref_ret_{window}"]
        cov = btc_ret.rolling(96, min_periods=24).cov(peer_ret)
        var = btc_ret.rolling(96, min_periods=24).var()
        peer_state[f"{prefix}_btc_beta_96"] = cov / (var + 1e-12)
        peer_state[f"{prefix}_btc_corr_96"] = btc_ret.rolling(96, min_periods=24).corr(peer_ret)
        peer_state[f"{prefix}_risk_on_score"] = (
            peer_state[f"{prefix}_rel_ret_16"].fillna(0.0) * 100.0
            + np.log(peer_state[f"{prefix}_volume_ratio_20_96"].clip(lower=1e-6)).fillna(0.0)
            - peer_state[f"{prefix}_rv_ratio_16_96"].fillna(1.0)
        )
        keep = [
            f"{prefix}_ret_1",
            f"{prefix}_ret_4",
            f"{prefix}_ret_16",
            f"{prefix}_ret_96",
            f"{prefix}_rel_ret_1",
            f"{prefix}_rel_ret_4",
            f"{prefix}_rel_ret_16",
            f"{prefix}_rel_ret_96",
            f"{prefix}_rv_ratio_16_96",
            f"{prefix}_ma_dist_21",
            f"{prefix}_ma_dist_96",
            f"{prefix}_range_pos_96",
            f"{prefix}_volume_ratio_20_96",
            f"{prefix}_btc_beta_96",
            f"{prefix}_btc_corr_96",
            f"{prefix}_risk_on_score",
        ]
        state_parts.append(peer_state[keep])
        peer_ret_cols.append(f"{prefix}_ret_1")
        peer_ret16_cols.append(f"{prefix}_ret_16")
        peer_rv_cols.append(f"{prefix}_rv_ratio_16_96")
        peer_volume_cols.append(f"{prefix}_volume_ratio_20_96")

    if not state_parts:
        return pd.DataFrame(index=events.index)

    state = pd.concat(state_parts, axis=1)
    state["peer_mkt_ret_1_mean"] = state[peer_ret_cols].mean(axis=1)
    state["peer_mkt_ret_1_up_ratio"] = (state[peer_ret_cols] > 0.0).mean(axis=1)
    state["peer_mkt_ret_1_dispersion"] = state[peer_ret_cols].std(axis=1)
    state["peer_mkt_ret_16_mean"] = state[peer_ret16_cols].mean(axis=1)
    state["peer_mkt_ret_16_up_ratio"] = (state[peer_ret16_cols] > 0.0).mean(axis=1)
    state["peer_mkt_rv_ratio_mean"] = state[peer_rv_cols].mean(axis=1)
    state["peer_mkt_volume_ratio_mean"] = state[peer_volume_cols].mean(axis=1)
    state["peer_mkt_btc_rel_strength_16"] = btc_state["btc_ref_ret_16"] - state["peer_mkt_ret_16_mean"]
    state["peer_mkt_risk_on_score"] = (
        state["peer_mkt_ret_16_up_ratio"]
        + np.log(state["peer_mkt_volume_ratio_mean"].clip(lower=1e-6)).fillna(0.0)
        - state["peer_mkt_rv_ratio_mean"].fillna(1.0)
    )

    event_time = pd.to_datetime(events["exec_time"], utc=True, errors="coerce")
    event_ns = pd.DatetimeIndex(event_time).astype("datetime64[ns, UTC]").view("int64")
    state_index_ns = state.index.view("int64")
    positions = np.searchsorted(state_index_ns, event_ns, side="right") - 1
    sampled = pd.DataFrame(index=events.index, columns=state.columns, dtype="float64")
    valid = positions >= 0
    if bool(valid.any()):
        sampled.loc[valid, :] = state.iloc[positions[valid]].to_numpy(dtype="float64")

    is_buy = events["is_buy"].astype(bool) if "is_buy" in events.columns else pd.Series(False, index=events.index)
    sampled["peer_mkt_signal_align_1"] = np.where(
        is_buy,
        (sampled["peer_mkt_ret_1_mean"] > 0.0).astype("float64"),
        (sampled["peer_mkt_ret_1_mean"] < 0.0).astype("float64"),
    )
    sampled["peer_mkt_signal_align_16"] = np.where(
        is_buy,
        (sampled["peer_mkt_ret_16_mean"] > 0.0).astype("float64"),
        (sampled["peer_mkt_ret_16_mean"] < 0.0).astype("float64"),
    )
    sampled["peer_mkt_signal_conflict_16"] = 1.0 - sampled["peer_mkt_signal_align_16"]
    return sampled.replace([np.inf, -np.inf], np.nan)


def _technical_state_from_bars(bars: pd.DataFrame, prefix: str) -> pd.DataFrame:
    close = pd.to_numeric(bars["close"], errors="coerce")
    open_ = pd.to_numeric(bars["open"], errors="coerce")
    high = pd.to_numeric(bars["high"], errors="coerce")
    low = pd.to_numeric(bars["low"], errors="coerce")
    volume = pd.to_numeric(bars["volume"], errors="coerce").fillna(0.0)
    out = pd.DataFrame(index=bars.index)
    ret1 = close.pct_change()
    out[f"{prefix}_ret_1"] = ret1
    for window in (3, 8, 16, 32):
        out[f"{prefix}_ret_{window}"] = close.pct_change(window)
    for window in (5, 13, 21, 34, 55, 89):
        ma = close.rolling(window, min_periods=max(3, window // 4)).mean()
        out[f"{prefix}_ma{window}_dist"] = (close - ma) / (ma + 1e-12)
    ema12 = close.ewm(span=12, adjust=False, min_periods=6).mean()
    ema26 = close.ewm(span=26, adjust=False, min_periods=13).mean()
    dif = ema12 - ema26
    dea = dif.ewm(span=9, adjust=False, min_periods=5).mean()
    hist = dif - dea
    out[f"{prefix}_macd_dif_pct"] = dif / (close + 1e-12)
    out[f"{prefix}_macd_dea_pct"] = dea / (close + 1e-12)
    out[f"{prefix}_macd_hist_pct"] = hist / (close + 1e-12)
    out[f"{prefix}_macd_hist_slope"] = hist.diff() / (close + 1e-12)
    out[f"{prefix}_macd_area_8"] = hist.rolling(8, min_periods=3).sum() / (close + 1e-12)
    out[f"{prefix}_macd_area_21"] = hist.rolling(21, min_periods=6).sum() / (close + 1e-12)
    out[f"{prefix}_macd_hist_max_21"] = hist.rolling(21, min_periods=6).max() / (close + 1e-12)
    out[f"{prefix}_macd_hist_min_21"] = hist.rolling(21, min_periods=6).min() / (close + 1e-12)
    boll_mid = close.rolling(20, min_periods=8).mean()
    boll_std = close.rolling(20, min_periods=8).std()
    boll_up = boll_mid + 2.0 * boll_std
    boll_low = boll_mid - 2.0 * boll_std
    out[f"{prefix}_boll_width"] = (boll_up - boll_low) / (boll_mid + 1e-12)
    out[f"{prefix}_boll_pos"] = (close - boll_low) / (boll_up - boll_low + 1e-12)
    out[f"{prefix}_boll_mid_dist"] = (close - boll_mid) / (boll_mid + 1e-12)
    out[f"{prefix}_boll_squeeze"] = out[f"{prefix}_boll_width"] / (
        out[f"{prefix}_boll_width"].rolling(96, min_periods=24).mean() + 1e-12
    )
    prev_close = close.shift(1)
    tr = pd.concat([(high - low), (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=5).mean()
    out[f"{prefix}_atr_pct"] = atr / (close + 1e-12)
    roll_high = high.rolling(55, min_periods=14).max()
    roll_low = low.rolling(55, min_periods=14).min()
    out[f"{prefix}_range_pos_55"] = (close - roll_low) / (roll_high - roll_low + 1e-12)
    out[f"{prefix}_range_width_55"] = (roll_high - roll_low) / (close + 1e-12)
    price_high = close.rolling(21, min_periods=8).max()
    price_low = close.rolling(21, min_periods=8).min()
    macd_high = hist.rolling(21, min_periods=8).max()
    macd_low = hist.rolling(21, min_periods=8).min()
    out[f"{prefix}_bear_divergence_proxy"] = ((close >= price_high.shift(1)) & (hist < macd_high.shift(1))).astype("float64")
    out[f"{prefix}_bull_divergence_proxy"] = ((close <= price_low.shift(1)) & (hist > macd_low.shift(1))).astype("float64")
    out[f"{prefix}_ma_bull_stack"] = ((ema12 > ema26) & (close > ema12)).astype("float64")
    out[f"{prefix}_ma_bear_stack"] = ((ema12 < ema26) & (close < ema12)).astype("float64")
    out[f"{prefix}_volume_ratio_20"] = volume / (volume.rolling(20, min_periods=5).mean() + 1e-12)
    return out.replace([np.inf, -np.inf], np.nan)


def _sample_state_at_event_close(events: pd.DataFrame, state: pd.DataFrame, duration_minutes: int) -> pd.DataFrame:
    if events.empty or state.empty:
        return pd.DataFrame(index=events.index)
    exec_time = pd.to_datetime(events["exec_time"], utc=True, errors="coerce")
    sample_time = exec_time + pd.Timedelta(minutes=15 - int(duration_minutes))
    state_index_ns = state.index.view("int64")
    event_ns = pd.DatetimeIndex(sample_time).astype("datetime64[ns, UTC]").view("int64")
    positions = np.searchsorted(state_index_ns, event_ns, side="right") - 1
    sampled = pd.DataFrame(index=events.index, columns=state.columns, dtype="float64")
    valid = positions >= 0
    if bool(valid.any()):
        sampled.loc[valid, :] = state.iloc[positions[valid]].to_numpy(dtype="float64")
    return sampled


def build_technical_mtf_feature_frame(events: pd.DataFrame, bars_1m: pd.DataFrame) -> pd.DataFrame:
    """Build 5m/15m/1h/4h/1d leakage-free technical and nesting features."""
    if events.empty:
        return pd.DataFrame(index=events.index)
    rules = {
        "5m": ("5min", 5),
        "15m": ("15min", 15),
        "1h": ("1h", 60),
        "4h": ("4h", 240),
        "1d": ("1D", 1440),
    }
    sampled_parts: list[pd.DataFrame] = []
    for label, (rule, minutes) in rules.items():
        bars_tf = resample_ohlcv_open_labeled(bars_1m, rule)
        state = _technical_state_from_bars(bars_tf, f"tech_{label}")
        sampled_parts.append(_sample_state_at_event_close(events, state, minutes))
    out = pd.concat(sampled_parts, axis=1) if sampled_parts else pd.DataFrame(index=events.index)
    if out.empty:
        return out

    is_buy = events["is_buy"].astype(bool) if "is_buy" in events.columns else pd.Series(False, index=events.index)
    align_cols = []
    conflict_cols = []
    divergence_cols = []
    for label in rules:
        bull = pd.to_numeric(out.get(f"tech_{label}_ma_bull_stack"), errors="coerce")
        bear = pd.to_numeric(out.get(f"tech_{label}_ma_bear_stack"), errors="coerce")
        align = np.where(is_buy, bull, bear)
        conflict = np.where(is_buy, bear, bull)
        out[f"tech_{label}_signal_ma_align"] = align
        out[f"tech_{label}_signal_ma_conflict"] = conflict
        align_cols.append(out[f"tech_{label}_signal_ma_align"])
        conflict_cols.append(out[f"tech_{label}_signal_ma_conflict"])
        bull_div = pd.to_numeric(out.get(f"tech_{label}_bull_divergence_proxy"), errors="coerce")
        bear_div = pd.to_numeric(out.get(f"tech_{label}_bear_divergence_proxy"), errors="coerce")
        out[f"tech_{label}_signal_divergence_proxy"] = np.where(is_buy, bull_div, bear_div)
        divergence_cols.append(out[f"tech_{label}_signal_divergence_proxy"])

    out["tech_mtf_ma_align_count"] = pd.concat(align_cols, axis=1).sum(axis=1)
    out["tech_mtf_ma_conflict_count"] = pd.concat(conflict_cols, axis=1).sum(axis=1)
    out["tech_mtf_divergence_count"] = pd.concat(divergence_cols, axis=1).sum(axis=1)
    for lo, hi in (("5m", "15m"), ("15m", "1h"), ("1h", "4h"), ("4h", "1d")):
        out[f"tech_nested_{lo}_{hi}_ma_align"] = (
            pd.to_numeric(out.get(f"tech_{lo}_signal_ma_align"), errors="coerce")
            * pd.to_numeric(out.get(f"tech_{hi}_signal_ma_align"), errors="coerce")
        )
        out[f"tech_nested_{lo}_{hi}_boll_pos_diff"] = (
            pd.to_numeric(out.get(f"tech_{lo}_boll_pos"), errors="coerce")
            - pd.to_numeric(out.get(f"tech_{hi}_boll_pos"), errors="coerce")
        )
        out[f"tech_nested_{lo}_{hi}_macd_hist_ratio"] = (
            pd.to_numeric(out.get(f"tech_{lo}_macd_hist_pct"), errors="coerce")
            / (pd.to_numeric(out.get(f"tech_{hi}_macd_hist_pct"), errors="coerce").abs() + 1e-12)
        )
    out["tech_nested_5m_15m_1h_resonance"] = (
        pd.to_numeric(out.get("tech_nested_5m_15m_ma_align"), errors="coerce")
        * pd.to_numeric(out.get("tech_nested_15m_1h_ma_align"), errors="coerce")
    )
    out["tech_nested_15m_1h_4h_resonance"] = (
        pd.to_numeric(out.get("tech_nested_15m_1h_ma_align"), errors="coerce")
        * pd.to_numeric(out.get("tech_nested_1h_4h_ma_align"), errors="coerce")
    )
    return out.replace([np.inf, -np.inf], np.nan)


def _feature_cache_path(cache_dir: str | Path, symbol: str) -> Path:
    return Path(cache_dir) / f"btc_futures_v2_bsp2_family_features_{normalize_symbol(symbol).lower()}.parquet"


def _drop_label_and_outcome_columns(frame: pd.DataFrame) -> pd.DataFrame:
    drop_cols = [
        col
        for col in frame.columns
        if str(col).startswith(LABEL_PREFIXES) or col in OUTCOME_COLUMNS
    ]
    return frame.drop(columns=drop_cols, errors="ignore")


def _finalize_labeled_dataset(dataset: pd.DataFrame, symbol: str, begin: pd.Timestamp | None, end: pd.Timestamp | None) -> pd.DataFrame:
    for col in ("exec_time", "entry_time", "exit_time", "signal_available_time"):
        if col in dataset.columns:
            dataset[col] = pd.to_datetime(dataset[col], utc=True, errors="coerce")

    if begin is not None:
        dataset = dataset.loc[dataset["exec_time"] >= begin]
    if end is not None:
        dataset = dataset.loc[dataset["exec_time"] <= end]
    primary_label = "label_bsp2_chan_entry_quality" if "label_bsp2_chan_entry_quality" in dataset.columns else "label_bsp2_entry_quality"
    dataset = dataset.dropna(subset=[primary_label, "entry_time", "exit_time"]).sort_values("exec_time").reset_index(drop=True)

    min_entry_time = dataset["exec_time"]
    if not bool((dataset["entry_time"] >= min_entry_time).all()):
        raise RuntimeError(f"lookahead audit failed for {symbol}: entry_time is earlier than signal event time")
    return dataset


def label_symbol_feature_dataset(
    feature_dataset: pd.DataFrame,
    source_path: str | Path,
    *,
    symbol: str,
    begin_time: Any | None = "2021-01-01",
    end_time: Any | None = None,
    max_holding_minutes: int = 1440,
) -> pd.DataFrame:
    symbol = normalize_symbol(symbol)
    begin = parse_utc(begin_time)
    end = parse_utc(end_time, end_of_day=True)
    features_only = _drop_label_and_outcome_columns(feature_dataset).copy()
    if features_only.empty:
        return features_only

    exec_times = pd.to_datetime(features_only["exec_time"], utc=True, errors="coerce")
    load_begin = exec_times.min() - pd.Timedelta(days=2)
    load_end = exec_times.max() + pd.Timedelta(minutes=int(max_holding_minutes) + 24 * 60)
    if begin is not None:
        load_begin = min(load_begin, begin)
    if end is not None:
        load_end = max(load_end, end)
    bars_1m = load_1m_futures_bars(source_path, begin_time=load_begin, end_time=load_end)
    label_cfg = Bsp2StructureLabelConfig(
        max_observe_minutes=int(max_holding_minutes),
        signal_timeframe_minutes=0,
    )
    labels = label_bsp2_structure_with_1m_path(features_only, bars_1m, label_cfg)
    dataset = pd.concat([features_only.reset_index(drop=True), labels.reset_index(drop=True)], axis=1)
    dataset["route"] = "btc_futures_v2_bsp2_family"
    dataset["source_path"] = str(Path(source_path))
    dataset["label_max_holding_minutes"] = int(max_holding_minutes)
    return _finalize_labeled_dataset(dataset, symbol, begin, end)


def build_symbol_feature_dataset(
    symbol: str,
    source_path: str | Path,
    *,
    peer_data_dir: str | Path = DEFAULT_FUTURES_DATA_DIR,
    include_cross_market: bool = False,
    begin_time: Any | None = "2021-01-01",
    end_time: Any | None = None,
    warmup_days: int = 120,
) -> pd.DataFrame:
    symbol = normalize_symbol(symbol)
    asset = symbol_asset(symbol)
    begin = parse_utc(begin_time)
    load_begin = begin - pd.Timedelta(days=int(warmup_days)) if begin is not None else None
    end = parse_utc(end_time, end_of_day=True)

    bars_1m = load_1m_futures_bars(source_path, begin_time=load_begin, end_time=end)
    bars_15m = resample_ohlcv_open_labeled(bars_1m, "15min")
    cfg = make_chan_config(symbol)
    raw_events = extract_raw_bsp_events_from_bars(cfg, symbol, bars_15m)
    if not raw_events:
        raise RuntimeError(f"no Chan BSP events extracted for {symbol}")

    normalized_15m = normalize_bars(bars_15m)
    rust_config_path = rust_config_path_from_chan_config(CChanConfig(dict(cfg.chan_config)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PerformanceWarning)
        features = build_event_features(
            normalized_15m,
            raw_events,
            required_columns=None,
            rust_config_path=rust_config_path,
            normalized_bars=normalized_15m,
        )
        context = build_capability_context_feature_frame(raw_events, required_columns=None)
        chan_capability = build_capability_chan_feature_frame(features, required_columns=None)
        if include_cross_market:
            cross_market = build_cross_market_feature_frame(
                features,
                normalized_15m,
                data_dir=peer_data_dir,
                begin_time=load_begin,
                end_time=end,
            )
        else:
            cross_market = pd.DataFrame(index=features.index)
        technical_mtf = build_technical_mtf_feature_frame(features, bars_1m)
    dataset = pd.concat(
        [
            features.reset_index(drop=True),
            context.reset_index(drop=True),
            chan_capability.reset_index(drop=True),
            cross_market.reset_index(drop=True),
            technical_mtf.reset_index(drop=True),
        ],
        axis=1,
    )
    dataset = dataset.loc[:, ~dataset.columns.duplicated(keep="first")]
    bsp_text = dataset.get("bsp_types_str", dataset.get("bsp_type", pd.Series("", index=dataset.index))).astype(str).str.lower()
    dataset = dataset.loc[bsp_text.str.startswith("2")].reset_index(drop=True)
    dataset["symbol"] = symbol
    dataset["asset"] = asset
    for train_symbol in TRAIN_SYMBOLS:
        train_asset = symbol_asset(train_symbol).lower()
        dataset[f"asset_is_{train_asset}"] = 1.0 if normalize_symbol(train_symbol) == symbol else 0.0

    try:
        from ML.routes.btc_futures_v3_alpha.dataset import enhance_btc_futures_v3_features

        dataset = enhance_btc_futures_v3_features(dataset)
    except Exception as exc:
        raise RuntimeError(f"failed to apply v3 structure feature expansion to {symbol} dataset: {exc}") from exc
    dataset = enhance_chan_bi_features(dataset)
    dataset = enhance_chan_zs_bsp_features(dataset)
    return _drop_label_and_outcome_columns(dataset)


def build_symbol_futures_dataset(
    symbol: str,
    source_path: str | Path,
    *,
    peer_data_dir: str | Path = DEFAULT_FUTURES_DATA_DIR,
    include_cross_market: bool = False,
    begin_time: Any | None = "2021-01-01",
    end_time: Any | None = None,
    warmup_days: int = 120,
    take_profit_pct: float = 0.01,
    stop_loss_pct: float = 0.01,
    max_holding_minutes: int = 1440,
) -> pd.DataFrame:
    symbol = normalize_symbol(symbol)
    features_only = build_symbol_feature_dataset(
        symbol=symbol,
        source_path=source_path,
        peer_data_dir=peer_data_dir,
        include_cross_market=include_cross_market,
        begin_time=begin_time,
        end_time=end_time,
        warmup_days=warmup_days,
    )
    return label_symbol_feature_dataset(
        features_only,
        source_path=source_path,
        symbol=symbol,
        begin_time=begin_time,
        end_time=end_time,
        max_holding_minutes=max_holding_minutes,
    )


def _build_cached_symbol_dataset(args: dict[str, Any]) -> tuple[str, str, int, str]:
    symbol = normalize_symbol(args["symbol"])
    source_path = Path(args["source_path"])
    cache_path = Path(args["cache_path"])
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists() and not bool(args["force_features"]):
        features_only = pd.read_parquet(cache_path)
        source = "feature_cache"
    else:
        features_only = build_symbol_feature_dataset(
            symbol=symbol,
            source_path=source_path,
            peer_data_dir=args["peer_data_dir"],
            include_cross_market=bool(args["include_cross_market"]),
            begin_time=args["begin_time"],
            end_time=args["end_time"],
            warmup_days=int(args["warmup_days"]),
        )
        features_only.to_parquet(cache_path, index=False)
        source = "rebuilt_features"

    if bool(args["features_only"]):
        return symbol, str(cache_path), int(len(features_only)), source

    dataset = label_symbol_feature_dataset(
        features_only,
        source_path=source_path,
        symbol=symbol,
        begin_time=args["begin_time"],
        end_time=args["end_time"],
        max_holding_minutes=int(args["max_holding_minutes"]),
    )
    part_path = Path(args["part_path"])
    part_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(part_path, index=False)
    return symbol, str(part_path), int(len(dataset)), source


def build_btc_futures_dataset(
    source_path: str | Path = DEFAULT_SOURCE_PATH,
    output_path: str | Path = DEFAULT_DATASET_PATH,
    peer_data_dir: str | Path = DEFAULT_FUTURES_DATA_DIR,
    symbols: tuple[str, ...] = TRAIN_SYMBOLS,
    include_cross_market: bool = False,
    begin_time: Any | None = "2021-01-01",
    end_time: Any | None = None,
    warmup_days: int = 120,
    take_profit_pct: float = 0.01,
    stop_loss_pct: float = 0.01,
    max_holding_minutes: int = 1440,
    force: bool = False,
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    force_features: bool = False,
    refresh_labels_only: bool = False,
    workers: int = 1,
    features_only: bool = False,
) -> pd.DataFrame:
    output = Path(output_path)
    if output.exists() and not force:
        return pd.read_parquet(output)

    begin = parse_utc(begin_time)
    end = parse_utc(end_time, end_of_day=True)
    requested_symbols = tuple(normalize_symbol(s) for s in symbols)
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    part_root = cache_root / "labeled_parts"
    frames: list[pd.DataFrame] = []
    skipped: dict[str, str] = {}
    tasks: list[dict[str, Any]] = []
    seed_dataset = None
    if refresh_labels_only and output.exists():
        try:
            seed_dataset = pd.read_parquet(output)
            print(f"[DATASET][CACHE] seed feature caches from existing dataset: {output}")
        except Exception as exc:
            print(f"[DATASET][CACHE] cannot seed from existing dataset {output}: {exc}")
    for symbol in requested_symbols:
        path = Path(source_path) if symbol == normalize_symbol(SYMBOL) and source_path else futures_1m_path(symbol, peer_data_dir)
        if not path.exists():
            skipped[symbol] = f"missing source file: {path}"
            continue
        cache_path = _feature_cache_path(cache_root, symbol)
        if refresh_labels_only and not cache_path.exists():
            if seed_dataset is not None and "symbol" in seed_dataset.columns:
                seed_group = seed_dataset.loc[seed_dataset["symbol"].astype(str).eq(symbol)]
                if not seed_group.empty:
                    _drop_label_and_outcome_columns(seed_group).to_parquet(cache_path, index=False)
                    print(f"[DATASET][CACHE] seeded feature cache from existing dataset: {symbol}")
            legacy_path = Path(f"data/btc_futures_v2_bsp2_family/btc_futures_v2_bsp2_family_dataset_{symbol_asset(symbol).lower()}_only.parquet")
            if not cache_path.exists() and legacy_path.exists():
                legacy = pd.read_parquet(legacy_path)
                _drop_label_and_outcome_columns(legacy).to_parquet(cache_path, index=False)
                print(f"[DATASET][CACHE] seeded feature cache from {legacy_path}: {symbol}")
            v1_cache_path = Path(f"data/btc_futures_v1/cache/btc_futures_v1_features_{normalize_symbol(symbol).lower()}.parquet")
            if not cache_path.exists() and v1_cache_path.exists():
                v1_cache = pd.read_parquet(v1_cache_path)
                _drop_label_and_outcome_columns(v1_cache).to_parquet(cache_path, index=False)
                print(f"[DATASET][CACHE] seeded feature cache from {v1_cache_path}: {symbol}")
        tasks.append(
            {
                "symbol": symbol,
                "source_path": str(path),
                "cache_path": str(cache_path),
                "part_path": str(part_root / f"btc_futures_v2_bsp2_family_labeled_{symbol.lower()}.parquet"),
                "peer_data_dir": str(peer_data_dir),
                "include_cross_market": bool(include_cross_market),
                "begin_time": begin_time,
                "end_time": end_time,
                "warmup_days": int(warmup_days),
                "max_holding_minutes": int(max_holding_minutes),
                "force_features": bool(force_features and not refresh_labels_only),
                "features_only": bool(features_only),
            }
        )

    if tasks:
        worker_count = max(1, int(workers))
        print(
            f"[DATASET] symbols={len(tasks)} workers={worker_count} "
            f"force_features={bool(force_features and not refresh_labels_only)} refresh_labels_only={bool(refresh_labels_only)}"
        )
        results: list[tuple[str, str, int, str]] = []
        if worker_count == 1:
            for task in tasks:
                symbol = task["symbol"]
                print(f"[DATASET][START] {symbol}")
                try:
                    result = _build_cached_symbol_dataset(task)
                    print(f"[DATASET][DONE] {result[0]} rows={result[2]} source={result[3]}")
                    results.append(result)
                except Exception as exc:
                    skipped[symbol] = str(exc)
                    print(f"[DATASET][SKIP] {symbol}: {exc}")
        else:
            with ProcessPoolExecutor(max_workers=worker_count) as executor:
                future_map = {executor.submit(_build_cached_symbol_dataset, task): task["symbol"] for task in tasks}
                for future in as_completed(future_map):
                    symbol = future_map[future]
                    try:
                        result = future.result()
                        print(f"[DATASET][DONE] {result[0]} rows={result[2]} source={result[3]}")
                        results.append(result)
                    except Exception as exc:
                        skipped[symbol] = str(exc)
                        print(f"[DATASET][SKIP] {symbol}: {exc}")

        if features_only:
            if results:
                print(f"[DATASET] feature caches ready: {len(results)}")
            return pd.DataFrame(
                [{"symbol": symbol, "path": path, "rows": rows, "source": source} for symbol, path, rows, source in results]
            )

        for symbol, part_path, _, _ in sorted(results):
            frames.append(pd.read_parquet(part_path))

    if not frames:
        raise RuntimeError(f"no symbol datasets built; skipped={skipped}")

    dataset = pd.concat(frames, axis=0, ignore_index=True, sort=False)
    dataset = dataset.sort_values(["exec_time", "symbol"]).reset_index(drop=True)
    label_cfg = Bsp2StructureLabelConfig(
        max_observe_minutes=int(max_holding_minutes),
        signal_timeframe_minutes=0,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output, index=False)
    by_symbol = {
        str(symbol): {
            "rows": int(len(group)),
            "buy_rows": int(group["is_buy"].astype(bool).sum()),
            "sell_rows": int((~group["is_buy"].astype(bool)).sum()),
            "exec_time_min": str(pd.to_datetime(group["exec_time"], utc=True).min()),
            "exec_time_max": str(pd.to_datetime(group["exec_time"], utc=True).max()),
        }
        for symbol, group in dataset.groupby("symbol", dropna=False)
    }
    metadata = {
        "route": "btc_futures_v2_bsp2_family",
        "dataset_scope": "multi-symbol futures BSP samples",
        "source_path": str(Path(source_path)),
        "peer_data_dir": str(Path(peer_data_dir)),
        "symbols": list(requested_symbols),
        "built_symbols": sorted(by_symbol),
        "skipped_symbols": skipped,
        "include_cross_market": bool(include_cross_market),
        "peer_symbols": list(PEER_MARKET_SYMBOLS) if include_cross_market else [],
        "output_path": str(output),
        "begin_time": str(begin) if begin is not None else None,
        "end_time": str(end) if end is not None else None,
        "warmup_days": int(warmup_days),
        "cache_dir": str(cache_root),
        "force_features": bool(force_features and not refresh_labels_only),
        "refresh_labels_only": bool(refresh_labels_only),
        "workers": int(workers),
        "rows": int(len(dataset)),
        "by_symbol": by_symbol,
        "buy_rows": int(dataset["is_buy"].astype(bool).sum()),
        "sell_rows": int((~dataset["is_buy"].astype(bool)).sum()),
        "label_config": asdict(label_cfg),
        "primary_label": "label_bsp2_family_valid",
        "label_design": (
            "Expanded Chan BSP2 family label: classify standard BSP2, subclass BSP2s, 2/3 overlap, "
            "BSP2 after BSP1, and origin-ZS-boundary confirmation candidates. A valid sample must "
            "avoid structural invalidation and weak-no-confirm paths; follow classes drive exits."
        ),
        "structure_invalid_price": "chan_bsp2_invalid_price / chan_bsp2s_invalid_price from Chan BSP elements; fallback is audited by label_bsp2_invalid_source",
        "feature_expansion": (
            "v1 base features + cap_chan features + v3 alpha structure features + "
            "chan_bi features + chan_fx features + chan_zs features + chan_bsp features + "
            "asset identity features + 5m/15m/1h/4h/1d technical nesting features"
        )
        + (" + optional cross-market peer futures features" if include_cross_market else ""),
        "lookahead_check": "exec_time is signal availability time; entry_time >= exec_time",
    }
    output.with_name("btc_futures_v2_bsp2_family_dataset_meta.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return dataset


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build futures v2 BSP2-family ML dataset")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE_PATH))
    parser.add_argument("--output", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--peer-data-dir", default=str(DEFAULT_FUTURES_DATA_DIR))
    parser.add_argument("--symbols", default=",".join(TRAIN_SYMBOLS))
    parser.add_argument("--include-cross-market", action="store_true")
    parser.add_argument("--begin-time", default="2021-01-01")
    parser.add_argument("--end-time", default="")
    parser.add_argument("--warmup-days", type=int, default=120)
    parser.add_argument("--take-profit-pct", type=float, default=0.01)
    parser.add_argument("--stop-loss-pct", type=float, default=0.01)
    parser.add_argument("--max-holding-minutes", type=int, default=1440)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    parser.add_argument("--force-features", action="store_true", help="rebuild per-symbol feature cache before labeling")
    parser.add_argument("--refresh-labels-only", action="store_true", help="reuse cached features and only rebuild labels/outcomes")
    parser.add_argument("--workers", type=int, default=1, help="parallel symbol workers")
    parser.add_argument("--features-only", action="store_true", help="only build/update feature caches")
    args = parser.parse_args()

    dataset = build_btc_futures_dataset(
        source_path=args.source,
        output_path=args.output,
        peer_data_dir=args.peer_data_dir,
        symbols=tuple(s.strip() for s in args.symbols.split(",") if s.strip()),
        include_cross_market=bool(args.include_cross_market),
        begin_time=args.begin_time,
        end_time=args.end_time or None,
        warmup_days=args.warmup_days,
        take_profit_pct=args.take_profit_pct,
        stop_loss_pct=args.stop_loss_pct,
        max_holding_minutes=args.max_holding_minutes,
        force=args.force,
        cache_dir=args.cache_dir,
        force_features=args.force_features,
        refresh_labels_only=args.refresh_labels_only,
        workers=args.workers,
        features_only=args.features_only,
    )
    print(f"built futures dataset: rows={len(dataset)} output={args.output}")


if __name__ == "__main__":
    main()

