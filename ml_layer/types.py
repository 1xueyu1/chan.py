from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class EventSample:
    symbol: str
    open_time: str
    t0_ts: float
    t0_pos: int
    is_buy: bool
    bsp_main_type: str
    trade_price: float
    bi_start_price: float
    zs_low: float
    zs_high: float
    feature_map: Dict[str, float]


@dataclass
class LabeledSample:
    symbol: str
    open_time: str
    t0_ts: float
    t1_ts: float
    t0_pos: int
    t1_pos: int
    is_buy: bool
    bsp_main_type: str
    trade_price: float
    label_raw: int
    label: int
    label_text: str
    hit_event: str
    entry_price: float
    sl_price: float
    pt_price: float
    holding_bars: int
    timeout_used: int
    realized_return: float
    sample_weight: float
    overlap_count: float
    holding_time_normalized: float
    feature_map: Dict[str, float]
