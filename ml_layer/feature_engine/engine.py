from __future__ import annotations

# flake8: noqa: E501

from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict, deque
from typing import Deque, Dict, List, Tuple

import numpy as np
import pandas as pd

from ..config import FeatureConfig
from .chan_structure_context_features import compute_chan_structure_context_features
from .expanding_window_normalizer import ExpandingWindowNormalizer
from .legacy_feature_center_fusion_features import (
    compute_legacy_feature_center_fusion_features,
)
from .market_regime_state_features import compute_market_regime_state_features
from .multi_timeframe_resonance_features import (
    compute_multi_timeframe_resonance_features,
)
from .price_volume_microstructure_features import (
    build_price_volume_microstructure_indicators,
    compute_price_volume_microstructure_features,
)


class FeatureEngine:
    def __init__(self, config: FeatureConfig):
        self.config = config
        self.normalizer = ExpandingWindowNormalizer(
            categorical_features={
                "bsp_type_1",
                "bsp_type_2",
                "bsp_type_3",
                "is_buy_signal",
                "bsp_type_1b",
                "bsp_type_2b",
                "bsp_type_3b",
                "bsp_type_1s",
                "bsp_type_2s",
                "bsp_type_3s",
                "vol_regime",
                "market_regime",
            }
        )

    @staticmethod
    def _build_sample_df(samples: List[Dict]) -> pd.DataFrame:
        rows = [
            {
                "symbol": item["symbol"],
                "open_time": item["open_time"],
                "t0_ts": float(item["t0_ts"]),
                "t0_pos": int(item["t0_pos"]),
                "is_buy": bool(item["is_buy"]),
                "bsp_main_type": str(item["bsp_main_type"]),
                "realized_return": float(item.get("realized_return", 0.0)),
                "raw": item,
            }
            for item in samples
        ]
        df = pd.DataFrame(rows).sort_values(["t0_ts", "symbol", "t0_pos"]).reset_index(drop=True)
        df.index = pd.to_datetime(df["t0_ts"], unit="s", utc=True)
        return df

    @staticmethod
    def _build_symbol_bar_frame(bars_by_symbol: Dict[str, List[Dict]]) -> Dict[str, pd.DataFrame]:
        output: Dict[str, pd.DataFrame] = {}
        for symbol, bars in bars_by_symbol.items():
            if not bars:
                continue
            bdf = pd.DataFrame(bars).sort_values("klu_idx").reset_index(drop=True)
            ind = build_price_volume_microstructure_indicators(bdf)
            ind["symbol"] = symbol
            output[symbol] = ind
        return output

    @staticmethod
    def _build_symbol_event_bar_lookup(symbol_bars: pd.DataFrame) -> Tuple[Dict[int, Dict], Dict[int, int]]:
        if symbol_bars is None or len(symbol_bars) == 0:
            return {}, {}
        event_bar_lookup: Dict[int, Dict] = {}
        pos_lookup: Dict[int, int] = {}
        for pos, row in enumerate(symbol_bars.to_dict(orient="records")):
            key = int(row.get("klu_idx", -1))
            event_bar_lookup[key] = row
            pos_lookup[key] = pos
        return event_bar_lookup, pos_lookup

    @staticmethod
    def _precompute_bar_regime(symbol_bars: pd.DataFrame | None) -> Tuple[np.ndarray, np.ndarray]:
        if symbol_bars is None or len(symbol_bars) == 0:
            return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)

        close = pd.to_numeric(symbol_bars["close"], errors="coerce")
        returns = close.pct_change()
        annualizer = float(np.sqrt(252.0))

        rv5 = (
            returns.rolling(window=5, min_periods=5).std(ddof=0) * annualizer
        ).fillna(0.0).to_numpy(dtype=np.float32, copy=False)

        full_vol = returns.rolling(window=252, min_periods=20).std(ddof=0) * annualizer
        q1 = full_vol.rolling(window=252, min_periods=20).quantile(0.33)
        q2 = full_vol.rolling(window=252, min_periods=20).quantile(0.66)

        full_arr = full_vol.to_numpy(dtype=np.float64, copy=False)
        q1_arr = q1.to_numpy(dtype=np.float64, copy=False)
        q2_arr = q2.to_numpy(dtype=np.float64, copy=False)

        vol_regime = np.ones(len(symbol_bars), dtype=np.float32)
        finite_low = np.isfinite(full_arr) & np.isfinite(q1_arr)
        finite_high = np.isfinite(full_arr) & np.isfinite(q2_arr)
        vol_regime[finite_low & (full_arr < q1_arr)] = 0.0
        vol_regime[finite_high & (full_arr >= q2_arr)] = 2.0

        return rv5, vol_regime

    @staticmethod
    def _transform_symbol_samples(
        symbol: str,
        symbol_sample_df: pd.DataFrame,
        symbol_bars: pd.DataFrame | None,
    ) -> Tuple[List[pd.Timestamp], List[Dict[str, float]]]:
        same_dir_count_20: Dict[int, Deque[int]] = defaultdict(deque)
        same_type_last_ret: Dict[Tuple[str, int], float] = {}
        same_dir_win_hist: Dict[int, Deque[int]] = defaultdict(deque)
        last_signal_ts: float | None = None

        feature_rows: List[Dict[str, float]] = []
        feature_index: List[pd.Timestamp] = []

        event_bar_lookup, pos_lookup = FeatureEngine._build_symbol_event_bar_lookup(symbol_bars)
        rv5_arr, vol_regime_arr = FeatureEngine._precompute_bar_regime(symbol_bars)

        for row in symbol_sample_df.itertuples(index=True):
            ts = row.Index
            raw = row.raw
            event_dir = 1 if bool(row.is_buy) else -1
            bsp_type = str(row.bsp_main_type)

            structural_features = compute_chan_structure_context_features(raw)

            dir_hist = same_dir_count_20[event_dir]
            dir_hist.append(int(raw["t0_pos"]))
            while dir_hist and (int(raw["t0_pos"]) - dir_hist[0] > 20):
                dir_hist.popleft()
            b_ctx = {
                "same_dir_bsp_count_20": float(len(dir_hist)),
                "last_same_bsp_ret": float(same_type_last_ret.get((bsp_type, event_dir), 0.0)),
            }
            resonance_features = compute_multi_timeframe_resonance_features(raw, b_ctx)

            event_klu_idx = int(raw["klu_idx"])
            event_bar = event_bar_lookup.get(event_klu_idx, {})
            microstructure_features = compute_price_volume_microstructure_features(event_bar)

            ret_hist = same_dir_win_hist[event_dir]
            ret_hist.append(1 if float(row.realized_return) > 0 else 0)
            while len(ret_hist) > 20:
                ret_hist.popleft()

            days_since = 0.0
            cur_ts = float(row.t0_ts)
            if last_signal_ts is not None:
                days_since = max(0.0, (cur_ts - last_signal_ts) / 86400.0)
            last_signal_ts = cur_ts

            rv5 = 0.0
            vol_regime = 1.0
            trend_proxy = abs(float(raw.get("chan_struct", {}).get("seg_direction", 0.0))) * 25.0
            if symbol_bars is not None and event_bar:
                cur_pos = pos_lookup.get(event_klu_idx, -1)
                if 0 <= cur_pos < len(rv5_arr):
                    rv5 = float(rv5_arr[cur_pos])
                if 0 <= cur_pos < len(vol_regime_arr):
                    vol_regime = float(vol_regime_arr[cur_pos])

            d_ctx = {
                "realized_vol_5": rv5,
                "vol_regime": vol_regime,
                "trend_strength_adx": trend_proxy,
                "rolling_win_rate_20": float(np.mean(ret_hist)) if ret_hist else 0.5,
                "days_since_last_signal": days_since,
            }
            regime_features = compute_market_regime_state_features(d_ctx)
            legacy_fused_features = compute_legacy_feature_center_fusion_features(
                sample=raw,
                event_bar_row=event_bar,
                structural_features=structural_features,
                resonance_features=resonance_features,
                microstructure_features=microstructure_features,
            )

            merged = {}
            merged.update(structural_features)
            merged.update(resonance_features)
            merged.update(microstructure_features)
            merged.update(regime_features)
            merged.update(legacy_fused_features)

            same_type_last_ret[(bsp_type, event_dir)] = float(row.realized_return)

            feature_rows.append(merged)
            feature_index.append(ts)

        return feature_index, feature_rows

    def transform(
        self,
        samples: List[Dict],
        bars_by_symbol: Dict[str, List[Dict]],
        normalize: bool = True,
    ) -> pd.DataFrame:
        sample_df = self._build_sample_df(samples)
        sample_df = sample_df.reset_index(drop=True)  # Remove 't0_ts' index to avoid ambiguity in sort_values
        bar_df_map = self._build_symbol_bar_frame(bars_by_symbol)

        feature_rows: List[Dict[str, float]] = []
        feature_index: List[pd.Timestamp] = []

        symbol_frames = {
            symbol: sdf.sort_values("t0_ts")
            for symbol, sdf in sample_df.groupby("symbol", sort=False)
        }

        workers = max(1, int(getattr(self.config, "symbol_workers", 1)))
        workers = min(workers, len(symbol_frames)) if symbol_frames else 1

        if workers <= 1:
            for symbol, sdf in symbol_frames.items():
                idx, rows = self._transform_symbol_samples(symbol, sdf, bar_df_map.get(symbol))
                feature_index.extend(idx)
                feature_rows.extend(rows)
        else:
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = [
                    ex.submit(
                        self._transform_symbol_samples,
                        symbol,
                        sdf,
                        bar_df_map.get(symbol),
                    )
                    for symbol, sdf in symbol_frames.items()
                ]
                for fut in futures:
                    idx, rows = fut.result()
                    feature_index.extend(idx)
                    feature_rows.extend(rows)

        feature_df = pd.DataFrame(feature_rows, index=feature_index).sort_index()
        feature_df = feature_df.replace([np.inf, -np.inf], np.nan)
        feature_df = feature_df.fillna(0.0)

        if normalize and len(feature_df) > 0:
            feature_df = self.normalizer.fit_transform(feature_df, ordered_index=feature_df.index)

        return feature_df

    def transform_realtime(self, feature_row: pd.Series) -> pd.Series:
        return self.normalizer.transform_realtime(feature_row)
