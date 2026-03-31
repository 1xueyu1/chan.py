from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from ml_layer.config import FeatureConfig
from ml_layer.feature_engine import FeatureEngine

from .types import RawBSPEvent


def _event_pairs_sorted(
    events: List[RawBSPEvent],
) -> List[Tuple[int, RawBSPEvent]]:
    pairs = list(enumerate(events))
    pairs.sort(key=lambda x: (x[1].exec_time, x[1].klu_idx, x[0]))
    return pairs


def _lookup_t0_pos(index: pd.DatetimeIndex, ts: pd.Timestamp) -> int:
    # Use floor alignment to avoid any chance of referencing future bars.
    loc = int(index.searchsorted(ts, side="right")) - 1
    if loc < 0:
        return 0
    if loc >= len(index):
        return len(index) - 1
    return loc


def _build_chan_struct(feature_map: Dict[str, float]) -> Dict[str, float]:
    def _val(name: str) -> float:
        v = feature_map.get(name, 0.0)
        try:
            fv = float(v)
            return fv if np.isfinite(fv) else 0.0
        except Exception:
            return 0.0

    return {
        "beichi_strength": _val("beichi_strength"),
        "zs_bi_count": _val("zs_bi_count"),
        "zs_expand_count": _val("zs_expand_count"),
        "bi_length": _val("bi_length"),
        "bi_return_ratio": _val("bi_return_ratio"),
        "bi_length_ratio": _val("bi_length_ratio"),
        "bi_macd_area": _val("bi_macd_area"),
        "distance_to_zhongshu_center": _val("distance_to_zhongshu_center"),
        "zs_peak_range": _val("zs_peak_range"),
        "seg_direction": _val("seg_direction"),
        "seg_bi_count": _val("seg_bi_count"),
    }


def enrich_raw_events_with_feature_engine(
    symbol: str,
    bars: pd.DataFrame,
    raw_events: List[RawBSPEvent],
) -> List[RawBSPEvent]:
    if not raw_events or bars.empty:
        return raw_events

    fe = FeatureEngine(FeatureConfig(symbol_workers=1))
    pairs = _event_pairs_sorted(raw_events)
    bars_index = bars.index
    bars_time_text = bars_index.strftime("%Y-%m-%d %H:%M:%S")
    bars_open_ts = bars_index.asi8.astype(np.float64) / 1_000_000_000.0

    samples = []
    for _, ev in pairs:
        t0_pos = _lookup_t0_pos(bars_index, ev.exec_time)
        samples.append(
            {
                "symbol": symbol,
                "open_time": bars_time_text[t0_pos],
                "t0_ts": float(ev.exec_time.timestamp()),
                "t0_pos": int(t0_pos),
                "is_buy": bool(ev.is_buy),
                "bsp_main_type": str(ev.bsp_type),
                "realized_return": 0.0,
                "klu_idx": int(t0_pos),
                "chan_struct": _build_chan_struct(ev.feature_map),
            }
        )

    opens = bars["open"].to_numpy(dtype=np.float64, copy=False)
    highs = bars["high"].to_numpy(dtype=np.float64, copy=False)
    lows = bars["low"].to_numpy(dtype=np.float64, copy=False)
    closes = bars["close"].to_numpy(dtype=np.float64, copy=False)
    volumes = bars["volume"].to_numpy(dtype=np.float64, copy=False)
    bars_records = [
        {
            "klu_idx": int(pos),
            "open_ts": float(bars_open_ts[pos]),
            "open": float(opens[pos]),
            "high": float(highs[pos]),
            "low": float(lows[pos]),
            "close": float(closes[pos]),
            "volume": float(volumes[pos]),
            "time": bars_time_text[pos],
        }
        for pos in range(len(bars_index))
    ]

    sample_df = fe._build_sample_df(samples).reset_index(drop=True)
    sample_df = sample_df.sort_values("t0_ts")
    symbol_bars = fe._build_symbol_bar_frame(
        {symbol: bars_records}
    ).get(symbol)

    _, feature_rows = fe._transform_symbol_samples(
        symbol,
        sample_df,
        symbol_bars,
    )
    feat_df = pd.DataFrame(feature_rows)
    feat_df = feat_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    if len(feat_df) > 0:
        feat_df = fe.normalizer.fit_transform(
            feat_df,
            ordered_index=pd.RangeIndex(len(feat_df)),
        )

    feat_cols = list(feat_df.columns)
    feat_values = feat_df.to_numpy(dtype=np.float32, copy=False) if len(feat_df) > 0 else np.empty((0, 0), dtype=np.float32)

    for i, (orig_idx, _) in enumerate(pairs):
        if i >= feat_values.shape[0]:
            break
        merged = dict(raw_events[orig_idx].feature_map)
        row_values = feat_values[i]
        for col_idx, k in enumerate(feat_cols):
            v = row_values[col_idx]
            try:
                fv = float(v)
            except Exception:
                fv = 0.0
            merged[k] = fv if np.isfinite(fv) else 0.0
        raw_events[orig_idx].feature_map = merged

    return raw_events
