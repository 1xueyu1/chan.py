from __future__ import annotations

from typing import Dict

import numpy as np


def compute_multi_timeframe_resonance_features(
    sample: Dict,
    same_symbol_recent: Dict[str, float],
) -> Dict[str, float]:
    chan_struct = sample.get("chan_struct", {})
    seg_dir = float(chan_struct.get("seg_direction", 0.0))
    ma_align = float(chan_struct.get("ma_trend_aligned", 0.0))
    event_dir = 1.0 if bool(sample.get("is_buy")) else -1.0

    htf_trend_dir = 1.0 if seg_dir > 0 else (-1.0 if seg_dir < 0 else 0.0)
    ltf_confirm = 1.0 if event_dir * ma_align > 0 else 0.0
    resonance = float((htf_trend_dir == event_dir) + (ma_align == event_dir) + (seg_dir == event_dir))

    out = {
        "htf_trend_dir": htf_trend_dir,
        "htf_zs_count_aligned": float(max(0.0, chan_struct.get("seg_bi_count", 0.0))),
        "ltf_bi_confirmed": ltf_confirm,
        "level_resonance_score": resonance,
        "htf_dist_to_nearest_zs": float(chan_struct.get("distance_to_zhongshu_center", 0.0)),
        "same_dir_bsp_count_20": float(same_symbol_recent.get("same_dir_bsp_count_20", 0.0)),
        "last_same_bsp_ret": float(same_symbol_recent.get("last_same_bsp_ret", 0.0)),
        "htf_bsp_alignment": 1.0 if resonance >= 2.0 else 0.0,
    }
    return out
