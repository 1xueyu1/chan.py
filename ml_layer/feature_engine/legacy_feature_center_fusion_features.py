from __future__ import annotations

from typing import Dict


# 旧 ChanModel/feature_center.py 中对外输出的主特征清单。
LEGACY_FEATURE_CENTER_KEYS = [
    "momentum_5",
    "price_acceleration",
    "price_pos_20",
    "price_ma20_dist",
    "price_ma60_dist",
    "ma20_ma60_cross",
    "ma_trend_aligned",
    "vol_ratio_20",
    "vol_change",
    "vol_amount_std_20",
    "upper_shadow_ratio",
    "lower_shadow_ratio",
    "bar_range",
    "bar_body_position",
    "candle_strength",
    "rsi_14",
    "macd_hist",
    "kdj_k",
    "boll_bandwidth",
    "atr_expanding",
    "cci_value",
    "bi_length",
    "bi_return_ratio",
    "bi_length_ratio",
    "bi_macd_area",
    "distance_to_zhongshu_center",
    "zs_bi_count",
    "zs_peak_range",
    "seg_direction",
    "seg_bi_count",
]


def compute_legacy_feature_center_fusion_features(
    sample: Dict,
    event_bar_row: Dict,
    structural_features: Dict,
    resonance_features: Dict,
    microstructure_features: Dict,
) -> Dict[str, float]:
    """
    将旧 feature_center 特征逐项融合到新特征层。

    说明:
    - 显式列出并映射每个旧特征，避免黑盒透传。
    - 保留原特征键名，确保与现有回测事件特征兼容。
    """
    chan_struct = sample.get("chan_struct", {})
    fused: Dict[str, float] = {
        # 动量 / 趋势 / 均线
        "momentum_5": float(event_bar_row.get("momentum_5", 0.0)),
        "price_acceleration": float(event_bar_row.get("price_acceleration", 0.0)),
        "price_pos_20": float(event_bar_row.get("price_pos_20", event_bar_row.get("price_position_20", 0.5))),
        "price_ma20_dist": float(event_bar_row.get("price_ma20_dist", 0.0)),
        "price_ma60_dist": float(event_bar_row.get("price_ma60_dist", 0.0)),
        "ma20_ma60_cross": float(event_bar_row.get("ma20_ma60_cross", 0.0)),
        "ma_trend_aligned": float(event_bar_row.get("ma_trend_aligned", 0.0)),
        # 成交量 / K线
        "vol_ratio_20": float(event_bar_row.get("vol_ratio_20", 1.0)),
        "vol_change": float(event_bar_row.get("vol_change", 0.0)),
        "vol_amount_std_20": float(event_bar_row.get("vol_amount_std_20", 0.0)),
        "upper_shadow_ratio": float(event_bar_row.get("upper_shadow_ratio", 0.0)),
        "lower_shadow_ratio": float(event_bar_row.get("lower_shadow_ratio", 0.0)),
        "bar_range": float(event_bar_row.get("bar_range", 0.0)),
        "bar_body_position": float(event_bar_row.get("bar_body_position", 0.5)),
        "candle_strength": float(event_bar_row.get("candle_strength", 0.0)),
        # 技术指标
        "rsi_14": float(event_bar_row.get("rsi_14", 50.0)),
        "macd_hist": float(event_bar_row.get("macd_hist", 0.0)),
        "kdj_k": float(event_bar_row.get("kdj_k", 50.0)),
        "boll_bandwidth": float(event_bar_row.get("boll_bandwidth", 0.0)),
        "atr_expanding": float(event_bar_row.get("atr_expanding", 0.0)),
        "cci_value": float(event_bar_row.get("cci_value", 0.0)),
        # 缠论结构
        "bi_length": float(chan_struct.get("bi_length", structural_features.get("bi_length", 0.0))),
        "bi_return_ratio": float(chan_struct.get("bi_return_ratio", 0.0)),
        "bi_length_ratio": float(chan_struct.get("bi_length_ratio", structural_features.get("bi_length_ratio", 0.0))),
        "bi_macd_area": float(chan_struct.get("bi_macd_area", 0.0)),
        "distance_to_zhongshu_center": float(chan_struct.get("distance_to_zhongshu_center", resonance_features.get("htf_dist_to_nearest_zs", 0.0))),
        "zs_bi_count": float(chan_struct.get("zs_bi_count", structural_features.get("bi_count_in_zs", 0.0))),
        "zs_peak_range": float(chan_struct.get("zs_peak_range", 0.0)),
        "seg_direction": float(chan_struct.get("seg_direction", resonance_features.get("htf_trend_dir", 0.0))),
        "seg_bi_count": float(chan_struct.get("seg_bi_count", resonance_features.get("htf_zs_count_aligned", 0.0))),
    }
    # 保底补齐，确保键完整。
    for key in LEGACY_FEATURE_CENTER_KEYS:
        if key not in fused:
            fused[key] = 0.0
    return fused
