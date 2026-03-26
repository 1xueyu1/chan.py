from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import xgboost as xgb

from .config import BacktestConfig
from .types import RawBSPEvent, ScoredSignalEvent


class DualModelGate:
    _EMPTY_SIGNAL_TARGET_RATE = 0.05

    def __init__(self, config: BacktestConfig):
        self.config = config

        self.model_buy = xgb.Booster()
        self.model_buy.load_model(config.model_buy_path)

        self.model_sell = xgb.Booster()
        self.model_sell.load_model(config.model_sell_path)

        self.meta_model = None
        meta_model_path = str(
            getattr(config, "meta_model_path", "") or ""
        ).strip()
        if not meta_model_path:
            # Default to train artifact directory convention.
            guessed = (
                Path(config.model_buy_path).resolve().parent
                / "meta_model.pkl"
            )
            if guessed.exists():
                meta_model_path = str(guessed)
        if meta_model_path:
            p = Path(meta_model_path)
            if p.exists():
                with open(p, "rb") as f:
                    self.meta_model = pickle.load(f)

        with open(config.meta_buy_path, "r", encoding="utf-8") as f:
            self.meta_buy: Dict[str, int] = json.load(f)
        with open(config.meta_sell_path, "r", encoding="utf-8") as f:
            self.meta_sell: Dict[str, int] = json.load(f)

        self.fnames_buy = [""] * len(self.meta_buy)
        for name, idx in self.meta_buy.items():
            self.fnames_buy[idx] = name

        self.fnames_sell = [""] * len(self.meta_sell)
        for name, idx in self.meta_sell.items():
            self.fnames_sell[idx] = name

    def _vectorize(
        self,
        event: RawBSPEvent,
        is_buy: bool,
    ) -> tuple[np.ndarray, Dict[str, int], List[str]]:
        meta = self.meta_buy if is_buy else self.meta_sell
        fnames = self.fnames_buy if is_buy else self.fnames_sell

        arr = np.zeros(len(meta), dtype=np.float32)

        if "bsp_type_1" in meta:
            arr[meta["bsp_type_1"]] = 1.0 if event.bsp_type == "1" else 0.0
        if "bsp_type_2" in meta:
            arr[meta["bsp_type_2"]] = 1.0 if event.bsp_type == "2" else 0.0
        if "bsp_type_3" in meta:
            arr[meta["bsp_type_3"]] = 1.0 if event.bsp_type == "3" else 0.0
        if "is_buy_signal" in meta:
            arr[meta["is_buy_signal"]] = 1.0 if bool(event.is_buy) else 0.0
        if "bsp_type_1b" in meta:
            arr[meta["bsp_type_1b"]] = (
                1.0
                if bool(event.is_buy) and event.bsp_type == "1"
                else 0.0
            )
        if "bsp_type_2b" in meta:
            arr[meta["bsp_type_2b"]] = (
                1.0
                if bool(event.is_buy) and event.bsp_type == "2"
                else 0.0
            )
        if "bsp_type_3b" in meta:
            arr[meta["bsp_type_3b"]] = (
                1.0
                if bool(event.is_buy) and event.bsp_type == "3"
                else 0.0
            )
        if "bsp_type_1s" in meta:
            arr[meta["bsp_type_1s"]] = (
                1.0
                if (not bool(event.is_buy)) and event.bsp_type == "1"
                else 0.0
            )
        if "bsp_type_2s" in meta:
            arr[meta["bsp_type_2s"]] = (
                1.0
                if (not bool(event.is_buy)) and event.bsp_type == "2"
                else 0.0
            )
        if "bsp_type_3s" in meta:
            arr[meta["bsp_type_3s"]] = (
                1.0
                if (not bool(event.is_buy)) and event.bsp_type == "3"
                else 0.0
            )

        for feat_name, feat_value in event.feature_map.items():
            if feat_name in meta:
                arr[meta[feat_name]] = (
                    0.0 if not np.isfinite(feat_value) else float(feat_value)
                )

        # Compatible aliases across older/newer feature naming.
        if (
            "dist_to_zs_center" in meta
            and "distance_to_zhongshu_center" in event.feature_map
        ):
            val = float(
                event.feature_map.get("distance_to_zhongshu_center", 0.0)
            )
            arr[meta["dist_to_zs_center"]] = (
                0.0 if not np.isfinite(val) else val
            )
        if "bi_count_in_zs" in meta and "zs_bi_count" in event.feature_map:
            val = float(event.feature_map.get("zs_bi_count", 0.0))
            arr[meta["bi_count_in_zs"]] = 0.0 if not np.isfinite(val) else val

        return arr, meta, fnames

    def _vectorize_batch(
        self,
        events: Sequence[RawBSPEvent],
        is_buy: bool,
    ) -> tuple[np.ndarray, List[str]]:
        meta = self.meta_buy if is_buy else self.meta_sell
        fnames = self.fnames_buy if is_buy else self.fnames_sell

        if not events:
            return np.empty((0, len(meta)), dtype=np.float32), fnames

        matrix = np.zeros((len(events), len(meta)), dtype=np.float32)

        if "bsp_type_1" in meta:
            idx = meta["bsp_type_1"]
            for row_idx, ev in enumerate(events):
                matrix[row_idx, idx] = 1.0 if ev.bsp_type == "1" else 0.0
        if "bsp_type_2" in meta:
            idx = meta["bsp_type_2"]
            for row_idx, ev in enumerate(events):
                matrix[row_idx, idx] = 1.0 if ev.bsp_type == "2" else 0.0
        if "bsp_type_3" in meta:
            idx = meta["bsp_type_3"]
            for row_idx, ev in enumerate(events):
                matrix[row_idx, idx] = 1.0 if ev.bsp_type == "3" else 0.0
        if "is_buy_signal" in meta:
            idx = meta["is_buy_signal"]
            for row_idx, ev in enumerate(events):
                matrix[row_idx, idx] = 1.0 if bool(ev.is_buy) else 0.0
        if "bsp_type_1b" in meta:
            idx = meta["bsp_type_1b"]
            for row_idx, ev in enumerate(events):
                matrix[row_idx, idx] = (
                    1.0
                    if bool(ev.is_buy) and ev.bsp_type == "1"
                    else 0.0
                )
        if "bsp_type_2b" in meta:
            idx = meta["bsp_type_2b"]
            for row_idx, ev in enumerate(events):
                matrix[row_idx, idx] = (
                    1.0
                    if bool(ev.is_buy) and ev.bsp_type == "2"
                    else 0.0
                )
        if "bsp_type_3b" in meta:
            idx = meta["bsp_type_3b"]
            for row_idx, ev in enumerate(events):
                matrix[row_idx, idx] = (
                    1.0
                    if bool(ev.is_buy) and ev.bsp_type == "3"
                    else 0.0
                )
        if "bsp_type_1s" in meta:
            idx = meta["bsp_type_1s"]
            for row_idx, ev in enumerate(events):
                matrix[row_idx, idx] = (
                    1.0
                    if (not bool(ev.is_buy)) and ev.bsp_type == "1"
                    else 0.0
                )
        if "bsp_type_2s" in meta:
            idx = meta["bsp_type_2s"]
            for row_idx, ev in enumerate(events):
                matrix[row_idx, idx] = (
                    1.0
                    if (not bool(ev.is_buy)) and ev.bsp_type == "2"
                    else 0.0
                )
        if "bsp_type_3s" in meta:
            idx = meta["bsp_type_3s"]
            for row_idx, ev in enumerate(events):
                matrix[row_idx, idx] = (
                    1.0
                    if (not bool(ev.is_buy)) and ev.bsp_type == "3"
                    else 0.0
                )

        for row_idx, ev in enumerate(events):
            for feat_name, feat_value in ev.feature_map.items():
                col_idx = meta.get(feat_name)
                if col_idx is not None:
                    matrix[row_idx, col_idx] = (
                        0.0
                        if not np.isfinite(feat_value)
                        else float(feat_value)
                    )

            if (
                "dist_to_zs_center" in meta
                and "distance_to_zhongshu_center" in ev.feature_map
            ):
                val = float(
                    ev.feature_map.get("distance_to_zhongshu_center", 0.0)
                )
                matrix[row_idx, meta["dist_to_zs_center"]] = (
                    0.0 if not np.isfinite(val) else val
                )
            if "bi_count_in_zs" in meta and "zs_bi_count" in ev.feature_map:
                val = float(ev.feature_map.get("zs_bi_count", 0.0))
                matrix[row_idx, meta["bi_count_in_zs"]] = (
                    0.0 if not np.isfinite(val) else val
                )

        return matrix, fnames

    def _predict_batch(
        self,
        events: Sequence[RawBSPEvent],
        is_buy: bool,
    ) -> np.ndarray:
        if not events:
            return np.empty((0,), dtype=np.float32)

        matrix, fnames = self._vectorize_batch(events, is_buy)
        dmat = xgb.DMatrix(
            matrix,
            feature_names=fnames,
            missing=np.nan,
        )
        model = self.model_buy if is_buy else self.model_sell
        probs = model.predict(dmat)
        arr = np.asarray(probs)
        if arr.ndim == 1:
            # Binary fallback: convert to 3-class compatible matrix.
            return np.stack([1.0 - arr, np.zeros_like(arr), arr], axis=1)
        if arr.ndim == 2 and arr.shape[1] == 2:
            # Binary model outputs [p_sl, p_pt] -> align to [p_sl, p_timeout, p_pt].
            return np.stack([arr[:, 0], np.zeros(arr.shape[0], dtype=arr.dtype), arr[:, 1]], axis=1)
        return arr

    @staticmethod
    def _direction_from_primary(primary_probs: np.ndarray) -> np.ndarray:
        classes = np.argmax(primary_probs, axis=1).astype(np.int32)
        # class map: 0=SL(-1), 1=TIMEOUT(0), 2=PT(+1)
        direction = np.zeros_like(classes, dtype=np.int32)
        direction[classes == 0] = -1
        direction[classes == 2] = 1
        return direction

    def _meta_prob_batch(
        self,
        X: np.ndarray,
        p_primary: np.ndarray,
        direction: np.ndarray,
    ) -> np.ndarray:
        out = np.zeros(X.shape[0], dtype=np.float32)
        active = direction != 0
        if not np.any(active):
            return out
        if self.meta_model is None:
            # Legacy fallback: max non-timeout class probability.
            out[active] = np.max(
                p_primary[active][:, [0, 2]],
                axis=1,
            ).astype(np.float32)
            return out

        p_block = p_primary[active]
        base_meta = getattr(self.meta_model, "model", self.meta_model)
        expected_total = getattr(base_meta, "n_features_in_", None)
        if isinstance(expected_total, (int, np.integer)):
            expect_primary_cols = int(expected_total) - int(X.shape[1])
            if expect_primary_cols == 2 and p_block.shape[1] >= 3:
                p_block = p_block[:, [0, 2]]
            elif expect_primary_cols == 1:
                p_block = np.max(
                    p_block[:, [0, 2]] if p_block.shape[1] >= 3 else p_block,
                    axis=1,
                    keepdims=True,
                )

        meta_input = np.hstack(
            [np.nan_to_num(X[active], nan=0.0), p_block]
        )
        out[active] = np.asarray(
            self.meta_model.predict_proba(meta_input),
            dtype=np.float32,
        )
        return out

    def score_event(self, event: RawBSPEvent) -> ScoredSignalEvent:
        scored = self.score_events([event])
        return scored[0]

    def score_events(
        self,
        events: List[RawBSPEvent],
    ) -> List[ScoredSignalEvent]:
        if not events:
            return []

        buy_idx: List[int] = []
        sell_idx: List[int] = []
        buy_events: List[RawBSPEvent] = []
        sell_events: List[RawBSPEvent] = []

        for idx, ev in enumerate(events):
            if bool(ev.is_buy):
                buy_idx.append(idx)
                buy_events.append(ev)
            else:
                sell_idx.append(idx)
                sell_events.append(ev)

        buy_primary = self._predict_batch(buy_events, True)
        sell_primary = self._predict_batch(sell_events, False)

        primary_probs = np.zeros((len(events), 3), dtype=np.float32)
        for i, idx in enumerate(buy_idx):
            primary_probs[idx] = buy_primary[i]
        for i, idx in enumerate(sell_idx):
            primary_probs[idx] = sell_primary[i]

        # Rebuild raw feature matrix in original event order for meta gating.
        X = np.full(
            (len(events), len(self.meta_buy)),
            np.nan,
            dtype=np.float32,
        )
        for idx, ev in enumerate(events):
            row, _, _ = self._vectorize(ev, bool(ev.is_buy))
            X[idx] = row

        direction = self._direction_from_primary(primary_probs)
        probs = self._meta_prob_batch(X, primary_probs, direction)

        scored: List[ScoredSignalEvent] = []
        threshold = self.config.signal_threshold + self.config.signal_margin
        primary_conf = np.max(primary_probs[:, [0, 2]], axis=1)

        qualified_arr = np.zeros(len(events), dtype=bool)
        for idx in range(len(events)):
            qualified_arr[idx] = bool(
                (float(probs[idx]) >= threshold) and (direction[idx] != 0)
            )

        # Robust fallback for fully empty backtests.
        if not np.any(qualified_arr) and len(events) > 0:
            q = max(0.0, min(1.0, 1.0 - self._EMPTY_SIGNAL_TARGET_RATE))
            adaptive_thr = float(np.quantile(primary_conf, q))
            qualified_arr = primary_conf >= adaptive_thr
            probs = primary_conf.astype(np.float32)
            direction = np.array(
                [1 if bool(ev.is_buy) else -1 for ev in events],
                dtype=np.int32,
            )

        for idx, ev in enumerate(events):
            prob = float(probs[idx])
            qualified = bool(qualified_arr[idx])
            if not qualified or direction[idx] == 0:
                signal = 0
            else:
                signal = int(direction[idx])

            scored.append(
                ScoredSignalEvent(
                    symbol=ev.symbol,
                    exec_time=ev.exec_time,
                    bsp_time=ev.bsp_time,
                    is_buy=ev.is_buy,
                    bsp_type=ev.bsp_type,
                    bsp_types_str=ev.bsp_types_str,
                    trade_price=ev.trade_price,
                    klu_idx=ev.klu_idx,
                    probability=prob,
                    qualified=qualified,
                    signal=signal,
                )
            )

        return scored
