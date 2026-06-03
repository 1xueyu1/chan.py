from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ML.routes.btc_futures_v1.data import DEFAULT_SOURCE_PATH, load_1m_futures_bars
from ML.routes.btc_futures_v3_alpha.dataset import (
    DEFAULT_DATASET_PATH as ALPHA_DATASET_PATH,
    build_btc_futures_v3_alpha_dataset,
    enhance_btc_futures_v3_features,
    required_base_columns_for_v3,
)
from ML.shared.json_utils import json_safe

from .entry_timing import EntryTimingConfig, build_entry_signal_samples


ROUTE_NAME = "btc_futures_v3_beta_edge"
DEFAULT_BASE_DATASET = ALPHA_DATASET_PATH
DEFAULT_DATASET_PATH = Path("data/btc_futures_v3_beta_edge/btc_futures_v3_beta_edge_dataset.parquet")
DEFAULT_MODEL_DIR = Path("result/ml/btc_futures_v3_beta_edge")
DEFAULT_BACKTEST_DIR = Path("result/btc_futures_v3_beta_edge")
DEFAULT_DELAYS = (0,)


def required_base_columns_for_beta() -> list[str]:
    cols = set(required_base_columns_for_v3())
    cols.update(
        {
            "target_pct",
            "beta_entry_delay_minutes",
            "beta_pre_entry_minutes",
            "beta_pre_entry_bar_count",
            "beta_pre_entry_signal_return",
            "beta_pre_entry_entry_move",
            "beta_pre_entry_runup",
            "beta_pre_entry_drawdown",
            "beta_pre_entry_range_pct",
            "beta_pre_entry_close_location",
            "beta_pre_entry_volatility",
            "beta_pre_entry_volume_ratio_60",
            "beta_confirm_momentum",
            "beta_confirm_breakout",
            "beta_confirm_pullback_control",
            "beta_confirm_volume",
            "beta_micro_signed_ret_5",
            "beta_micro_signed_ret_15",
            "beta_micro_signed_ret_30",
            "beta_micro_signed_ret_60",
            "beta_micro_signed_ret_120",
            "beta_micro_momentum_accel_5_30",
            "beta_micro_momentum_decay_30_60",
            "beta_micro_vol_expansion_5_30",
            "beta_micro_compression_5_30",
            "beta_micro_compression_30_120",
            "beta_micro_volume_impulse_5_60",
            "beta_micro_price_volume_confirm",
            "beta_micro_breakout_dist_30",
            "beta_micro_breakout_dist_120",
            "beta_micro_pullback_dist_30",
            "beta_micro_pullback_dist_120",
            "beta_micro_shadow_balance",
            "beta_micro_trend_efficiency_30",
            "beta_micro_trend_efficiency_120",
            "beta_micro_return_skew_30",
            "beta_micro_return_skew_120",
            "beta_second_signal_ret_1",
            "beta_second_signal_ret_3",
            "beta_second_signal_ret_5",
            "beta_second_signal_ret_10",
            "beta_second_signal_ret_all",
            "beta_second_signal_accel",
            "beta_second_directional_close_ratio",
            "beta_second_directional_body_ratio",
            "beta_second_directional_body_strength",
            "beta_second_favorable_excursion",
            "beta_second_adverse_excursion",
            "beta_second_adverse_control",
            "beta_second_favorable_efficiency",
            "beta_second_breakout_hold",
            "beta_second_failed_breakout",
            "beta_second_reclaim_after_pullback",
            "beta_second_early_follow_through",
            "beta_second_late_follow_through",
            "beta_second_confirmation_persistence",
            "beta_second_volume_confirmation",
            "beta_second_counter_impulse",
            "beta_second_clean_confirm_score",
            "beta_second_failure_risk_score",
        }
    )
    return sorted(cols)


def enhance_btc_futures_v3_beta_features(frame: pd.DataFrame) -> pd.DataFrame:
    out = enhance_btc_futures_v3_features(frame)
    if out.empty:
        return out

    def col(name: str, default: float) -> pd.Series:
        if name in out.columns:
            return pd.to_numeric(out[name], errors="coerce").astype("float64")
        return pd.Series(default, index=out.index, dtype="float64")

    delay = col("beta_entry_delay_minutes", 0.0).fillna(0.0)
    target = col("target_pct", 0.01).replace(0, np.nan).fillna(0.01)
    pre_return = col("beta_pre_entry_signal_return", np.nan)
    pre_runup = col("beta_pre_entry_runup", np.nan)
    pre_drawdown = col("beta_pre_entry_drawdown", np.nan)
    pre_range = col("beta_pre_entry_range_pct", np.nan)
    pre_vol = col("beta_pre_entry_volatility", np.nan)
    volume_ratio = col("beta_pre_entry_volume_ratio_60", np.nan)
    confirmations = [
        col("beta_confirm_momentum", np.nan),
        col("beta_confirm_breakout", np.nan),
        col("beta_confirm_pullback_control", np.nan),
        col("beta_confirm_volume", np.nan),
    ]
    confirm_matrix = pd.concat(confirmations, axis=1)
    out["beta_delay_norm"] = np.tanh(delay / 60.0)
    out["beta_pre_return_vs_target"] = pre_return / (target + 1e-12)
    out["beta_pre_runup_vs_target"] = pre_runup / (target + 1e-12)
    out["beta_pre_drawdown_vs_target"] = pre_drawdown.abs() / (target + 1e-12)
    out["beta_pre_range_vs_target"] = pre_range / (target + 1e-12)
    out["beta_pre_noise_score"] = pre_vol / (target + 1e-12)
    out["beta_confirmation_count"] = confirm_matrix.fillna(0.0).sum(axis=1)
    out["beta_confirmation_ratio"] = out["beta_confirmation_count"] / confirm_matrix.notna().sum(axis=1).replace(0, np.nan)
    out["beta_confirmation_strength"] = (
        0.30 * out["beta_confirmation_ratio"].fillna(0.0)
        + 0.20 * col("beta_confirm_momentum", 0.0).fillna(0.0)
        + 0.20 * col("beta_confirm_pullback_control", 0.0).fillna(0.0)
        + 0.15 * np.clip(volume_ratio.fillna(1.0) / 2.0, 0.0, 1.0)
        + 0.15 * col("v3_alpha_structure_score", 0.0).fillna(0.0)
    )
    out["beta_structure_x_confirmation"] = (
        col("v3_alpha_structure_score", 0.0).fillna(0.0)
        * out["beta_confirmation_strength"].fillna(0.0)
    )
    out["beta_conflict_adjusted_confirmation"] = out["beta_confirmation_strength"].fillna(0.0) - col(
        "v3_htf_conflict_score", 0.0
    ).fillna(0.0) * 0.25
    out["beta_micro_ret5_vs_target"] = col("beta_micro_signed_ret_5", 0.0) / (target + 1e-12)
    out["beta_micro_ret30_vs_target"] = col("beta_micro_signed_ret_30", 0.0) / (target + 1e-12)
    out["beta_micro_accel_vs_target"] = col("beta_micro_momentum_accel_5_30", 0.0) / (target + 1e-12)
    out["beta_micro_breakout30_vs_target"] = col("beta_micro_breakout_dist_30", 0.0) / (target + 1e-12)
    out["beta_micro_breakout120_vs_target"] = col("beta_micro_breakout_dist_120", 0.0) / (target + 1e-12)
    out["beta_micro_pullback30_vs_target"] = col("beta_micro_pullback_dist_30", 0.0) / (target + 1e-12)
    out["beta_micro_pressure_adjusted_score"] = (
        col("beta_micro_body_strength", 0.0).fillna(0.0)
        + col("beta_micro_shadow_balance", 0.0).fillna(0.0)
        + col("beta_micro_price_volume_confirm", 0.0).fillna(0.0)
        - col("beta_micro_vol_expansion_5_30", 1.0).fillna(1.0).clip(lower=0.0, upper=4.0) * 0.10
    )
    out["beta_micro_clean_breakout_score"] = (
        (col("beta_micro_breakout_dist_30", 0.0).fillna(0.0) > 0.0).astype("float64")
        + (col("beta_micro_pullback_dist_30", 0.0).fillna(0.0).abs() <= target * 1.25).astype("float64")
        + (col("beta_micro_volume_impulse_5_60", 1.0).fillna(1.0) >= 1.05).astype("float64")
        + (col("beta_micro_momentum_accel_5_30", 0.0).fillna(0.0) >= 0.0).astype("float64")
    ) / 4.0
    out["beta_micro_reversal_risk_score"] = (
        (col("beta_micro_momentum_accel_5_30", 0.0).fillna(0.0) < 0.0).astype("float64")
        + (col("beta_micro_shadow_balance", 0.0).fillna(0.0) < -0.15).astype("float64")
        + (col("beta_micro_return_skew_30", 0.0).fillna(0.0) < -0.25).astype("float64")
        + (col("beta_micro_vol_expansion_5_30", 1.0).fillna(1.0) > 1.8).astype("float64")
    ) / 4.0
    out["beta_micro_edge_score"] = (
        0.28 * out["beta_micro_clean_breakout_score"].fillna(0.0)
        + 0.22 * out["beta_micro_pressure_adjusted_score"].fillna(0.0)
        + 0.20 * col("beta_micro_price_volume_confirm", 0.0).fillna(0.0)
        + 0.16 * col("beta_micro_trend_efficiency_30", 0.0).fillna(0.0)
        + 0.14 * out["beta_confirmation_strength"].fillna(0.0)
        - 0.25 * out["beta_micro_reversal_risk_score"].fillna(0.0)
    )
    out["beta_micro_structure_edge_score"] = (
        out["beta_micro_edge_score"].fillna(0.0)
        * col("v3_alpha_structure_score", 0.5).fillna(0.5)
        * (1.0 - col("v3_htf_conflict_score", 0.0).fillna(0.0).clip(lower=0.0, upper=1.0) * 0.5)
    )
    second_ret3_vs_target = col("beta_second_signal_ret_3", 0.0) / (target + 1e-12)
    second_ret10_vs_target = col("beta_second_signal_ret_10", 0.0) / (target + 1e-12)
    second_favorable_vs_target = col("beta_second_favorable_excursion", 0.0) / (target + 1e-12)
    second_adverse_vs_target = col("beta_second_adverse_excursion", 0.0) / (target + 1e-12)
    second_confirm_quality_score = (
        0.22 * col("beta_second_clean_confirm_score", 0.0).fillna(0.0)
        + 0.18 * col("beta_second_confirmation_persistence", 0.0).fillna(0.0)
        + 0.14 * col("beta_second_breakout_hold", 0.0).fillna(0.0)
        + 0.12 * col("beta_second_reclaim_after_pullback", 0.0).fillna(0.0)
        + 0.12 * col("beta_second_directional_body_ratio", 0.0).fillna(0.0)
        + 0.10 * np.clip(col("beta_second_volume_confirmation", 1.0).fillna(1.0) / 2.0, 0.0, 1.0)
        + 0.12 * np.clip(col("beta_second_adverse_control", 0.0).fillna(0.0) / 2.0, 0.0, 1.0)
    )
    second_confirm_risk_score = (
        0.28 * col("beta_second_failure_risk_score", 0.0).fillna(0.0)
        + 0.22 * col("beta_second_failed_breakout", 0.0).fillna(0.0)
        + 0.18 * col("beta_second_counter_impulse", 0.0).fillna(0.0)
        + 0.16 * (col("beta_second_signal_accel", 0.0).fillna(0.0) < -target * 0.15).astype("float64")
        + 0.16 * (col("beta_second_adverse_excursion", 0.0).fillna(0.0) > target * 0.45).astype("float64")
    )
    second_structure_confirm_score = (
        second_confirm_quality_score.fillna(0.0)
        * col("v3_alpha_structure_score", 0.5).fillna(0.5)
        * (1.0 - col("v3_htf_conflict_score", 0.0).fillna(0.0).clip(lower=0.0, upper=1.0) * 0.35)
        - second_confirm_risk_score.fillna(0.0) * 0.30
    )
    second_micro_alignment_score = (
        0.35 * second_structure_confirm_score.fillna(0.0)
        + 0.30 * out["beta_micro_structure_edge_score"].fillna(0.0)
        + 0.20 * out["beta_conflict_adjusted_confirmation"].fillna(0.0)
        + 0.15 * col("beta_second_late_follow_through", 0.0).fillna(0.0)
    )
    second_features = pd.DataFrame(
        {
            "beta_second_ret3_vs_target": second_ret3_vs_target,
            "beta_second_ret10_vs_target": second_ret10_vs_target,
            "beta_second_favorable_vs_target": second_favorable_vs_target,
            "beta_second_adverse_vs_target": second_adverse_vs_target,
            "beta_second_confirm_quality_score": second_confirm_quality_score,
            "beta_second_confirm_risk_score": second_confirm_risk_score,
            "beta_second_structure_confirm_score": second_structure_confirm_score,
            "beta_second_micro_alignment_score": second_micro_alignment_score,
        },
        index=out.index,
    )
    out = out.drop(columns=[name for name in second_features.columns if name in out.columns], errors="ignore")
    out = pd.concat([out, second_features], axis=1).copy()
    numeric_cols = [col for col in out.columns if col.startswith("beta_") and pd.api.types.is_numeric_dtype(out[col])]
    out[numeric_cols] = out[numeric_cols].replace([np.inf, -np.inf], np.nan)
    return out


def build_btc_futures_v3_beta_edge_dataset(
    base_dataset: str | Path = DEFAULT_BASE_DATASET,
    source_path: str | Path = DEFAULT_SOURCE_PATH,
    output_path: str | Path = DEFAULT_DATASET_PATH,
    delays_minutes: tuple[int, ...] = DEFAULT_DELAYS,
    max_holding_minutes: int = 1440,
    force: bool = False,
) -> pd.DataFrame:
    output = Path(output_path)
    if output.exists() and not force:
        return pd.read_parquet(output)

    base_path = Path(base_dataset)
    if not base_path.exists():
        build_btc_futures_v3_alpha_dataset(output_path=base_path)
    base = pd.read_parquet(base_path)
    for col in ("exec_time", "entry_time", "exit_time", "signal_available_time"):
        if col in base.columns:
            base[col] = pd.to_datetime(base[col], utc=True, errors="coerce")
    base = base.dropna(subset=["label", "exec_time"]).sort_values("exec_time").reset_index(drop=True)
    begin = base["exec_time"].min() - pd.Timedelta(days=2)
    end = base["exec_time"].max() + pd.Timedelta(days=2)
    bars_1m = load_1m_futures_bars(source_path, begin_time=begin, end_time=end)
    cfg = EntryTimingConfig(delays_minutes=(0,), max_holding_minutes=int(max_holding_minutes))
    samples = build_entry_signal_samples(base, bars_1m, config=cfg)
    out = enhance_btc_futures_v3_beta_features(samples)
    out["route"] = ROUTE_NAME
    out["label_take_profit_pct"] = pd.to_numeric(out["target_pct"], errors="coerce")
    out["label_stop_loss_pct"] = pd.to_numeric(out["target_pct"], errors="coerce")
    out["label_max_holding_minutes"] = int(max_holding_minutes)
    out = out.dropna(subset=["label_tp_first", "entry_time", "exit_time"]).sort_values(["exec_time", "beta_event_id"]).reset_index(drop=True)
    if not bool((out["entry_time"] >= out["exec_time"] + pd.Timedelta(minutes=15)).all()):
        raise RuntimeError("lookahead audit failed: beta entry_time is earlier than closed signal bar")
    output.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output, index=False)
    meta = {
        "route": ROUTE_NAME,
        "base_dataset": str(base_path),
        "source_path": str(source_path),
        "output_path": str(output),
        "rows": int(len(out)),
        "events": int(out["beta_event_id"].nunique()),
        "sample_expansion_enabled": False,
        "delays_minutes": [0],
        "columns": int(len(out.columns)),
        "label_tp_first_rate": float(pd.to_numeric(out["label_tp_first"], errors="coerce").mean()),
        "avg_net_return": float(pd.to_numeric(out["net_return"], errors="coerce").mean()),
        "lookahead_check": "one sample per BSP event; entry_time >= exec_time + 15min; beta features use pre-entry 1m bars only",
    }
    output.with_name("btc_futures_v3_beta_edge_dataset_meta.json").write_text(
        json.dumps(json_safe(meta), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return out


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Build BTC futures v3 beta edge one-sample-per-signal dataset")
    parser.add_argument("--base-dataset", default=str(DEFAULT_BASE_DATASET))
    parser.add_argument("--source", default=str(DEFAULT_SOURCE_PATH))
    parser.add_argument("--output", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--delays", default="0", help="kept for compatibility; sample expansion is disabled and only 0 is used")
    parser.add_argument("--max-holding-minutes", type=int, default=1440)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    delays = tuple(int(x.strip()) for x in str(args.delays).split(",") if x.strip())
    frame = build_btc_futures_v3_beta_edge_dataset(
        base_dataset=args.base_dataset,
        source_path=args.source,
        output_path=args.output,
        delays_minutes=delays,
        max_holding_minutes=args.max_holding_minutes,
        force=bool(args.force),
    )
    print(f"built {ROUTE_NAME} dataset rows={len(frame)} output={args.output}")


if __name__ == "__main__":
    main()
