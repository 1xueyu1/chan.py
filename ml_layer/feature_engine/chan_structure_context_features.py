from __future__ import annotations

from typing import Dict

import numpy as np


def compute_chan_structure_context_features(sample: Dict) -> Dict[str, float]:
    chan_struct = sample.get("chan_struct", {})
    bsp_type = str(sample.get("bsp_main_type", ""))
    entry = float(sample.get("trade_price", np.nan))
    zs_low = float(sample.get("zs_low", np.nan))
    zs_high = float(sample.get("zs_high", np.nan))
    bi_start = float(sample.get("bi_start_price", np.nan))

    zs_center = (zs_low + zs_high) / 2.0 if np.isfinite(zs_low) and np.isfinite(zs_high) else entry
    zs_half = abs(zs_high - zs_low) / 2.0 if np.isfinite(zs_low) and np.isfinite(zs_high) else np.nan

    prev_bi = float(chan_struct.get("bi_length", np.nan))
    bi_ret_ratio = float(chan_struct.get("bi_return_ratio", np.nan))

    out = {
        "bsp_type_1": 1.0 if bsp_type == "1" else 0.0,
        "bsp_type_2": 1.0 if bsp_type == "2" else 0.0,
        "bsp_type_3": 1.0 if bsp_type == "3" else 0.0,
        "is_buy_signal": 1.0 if bool(sample.get("is_buy")) else 0.0,
        "bsp_type_1b": 1.0 if bsp_type == "1" and bool(sample.get("is_buy")) else 0.0,
        "bsp_type_2b": 1.0 if bsp_type == "2" and bool(sample.get("is_buy")) else 0.0,
        "bsp_type_3b": 1.0 if bsp_type == "3" and bool(sample.get("is_buy")) else 0.0,
        "bsp_type_1s": 1.0 if bsp_type == "1" and not bool(sample.get("is_buy")) else 0.0,
        "bsp_type_2s": 1.0 if bsp_type == "2" and not bool(sample.get("is_buy")) else 0.0,
        "bsp_type_3s": 1.0 if bsp_type == "3" and not bool(sample.get("is_buy")) else 0.0,
        "beichi_strength": float(chan_struct.get("beichi_strength", 0.0)),
        "zs_width_ratio": float((zs_high - zs_low) / (abs(zs_low) + 1e-9)) if np.isfinite(zs_low) and np.isfinite(zs_high) else 0.0,
        "bi_count_in_zs": float(chan_struct.get("zs_bi_count", 0.0)),
        "zs_expand_count": float(chan_struct.get("zs_expand_count", 0.0)),
        "bi_length_ratio": float(chan_struct.get("bi_length_ratio", bi_ret_ratio if np.isfinite(bi_ret_ratio) else 0.0)),
        "dist_to_zs_center": float((entry - zs_center) / (zs_half + 1e-9)) if np.isfinite(zs_half) and zs_half > 0 else 0.0,
        "bi_start_gap": float((entry - bi_start) / (abs(entry) + 1e-9)) if np.isfinite(bi_start) else 0.0,
        "bi_length": float(prev_bi if np.isfinite(prev_bi) else 0.0),
    }
    return out
