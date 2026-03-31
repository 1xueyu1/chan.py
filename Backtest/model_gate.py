from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Dict, List, Sequence

import math
import numpy as np
import xgboost as xgb

from .config import BacktestConfig
from .types import RawBSPEvent, ScoredSignalEvent


_GATE_CACHE: Dict[tuple, "DualModelGate"] = {}


def _gate_cache_key(config: BacktestConfig) -> tuple:
    return (
        str(config.model_buy_path),
        str(config.model_sell_path),
        str(config.meta_buy_path),
        str(config.meta_sell_path),
        str(getattr(config, "meta_model_path", "") or ""),
        float(config.signal_threshold),
        float(config.signal_margin),
        tuple(sorted((getattr(config, "meta_threshold_by_direction", {}) or {}).items())),
        tuple(sorted((getattr(config, "meta_threshold_by_bsp", {}) or {}).items())),
    )


def get_dual_model_gate(config: BacktestConfig) -> "DualModelGate":
    key = _gate_cache_key(config)
    gate = _GATE_CACHE.get(key)
    if gate is None:
        gate = DualModelGate(config)
        _GATE_CACHE[key] = gate
    return gate


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
        self.has_meta_model = self.meta_model is not None

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

        self._base_threshold = float(self.config.signal_threshold + self.config.signal_margin)
        self._threshold_by_direction: Dict[str, float] = {}
        self._threshold_by_bsp: Dict[str, float] = {}
        self._threshold_by_combo: Dict[str, float] = {}

        for key, value in (getattr(self.config, "meta_threshold_by_direction", {}) or {}).items():
            k = str(key).strip().lower()
            if k in {"buy", "sell"}:
                self._threshold_by_direction[k] = float(value)

        for key, value in (getattr(self.config, "meta_threshold_by_bsp", {}) or {}).items():
            k = str(key).strip().lower()
            if not k:
                continue
            if "_" in k:
                self._threshold_by_combo[k] = float(value)
            else:
                self._threshold_by_bsp[k] = float(value)

    @staticmethod
    def _clip_threshold(value: float) -> float:
        return float(max(0.0, min(1.0, value)))

    def _event_threshold(
        self,
        event: RawBSPEvent,
        direction: int,
    ) -> float:
        threshold = self._base_threshold
        if direction == 0:
            return self._clip_threshold(threshold)

        dir_key = "buy" if int(direction) > 0 else "sell"

        if dir_key in self._threshold_by_direction:
            threshold = self._threshold_by_direction[dir_key]

        bsp_key = str(getattr(event, "bsp_type", "")).lower()
        combo_key = f"{dir_key}_{bsp_key}"
        if bsp_key in self._threshold_by_bsp:
            threshold = self._threshold_by_bsp[bsp_key]
        if combo_key in self._threshold_by_combo:
            threshold = self._threshold_by_combo[combo_key]

        return self._clip_threshold(threshold)

    def _vectorize(
        self,
        event: RawBSPEvent,
        is_buy: bool,
    ) -> tuple[np.ndarray, Dict[str, int], List[str]]:
        meta = self.meta_buy if is_buy else self.meta_sell
        matrix, fnames = self._vectorize_batch([event], is_buy)
        if matrix.shape[0] == 0:
            return np.zeros(len(meta), dtype=np.float32), meta, fnames
        return matrix[0], meta, fnames

    def _vectorize_batch(
        self,
        events: Sequence[RawBSPEvent],
        is_buy: bool,
    ) -> tuple[np.ndarray, List[str]]:
        meta = self.meta_buy if is_buy else self.meta_sell
        fnames = self.fnames_buy if is_buy else self.fnames_sell

        if not events:
            return np.empty((0, len(meta)), dtype=np.float32), fnames

        n_rows = len(events)
        matrix = np.zeros((n_rows, len(meta)), dtype=np.float32)

        bsp_types = [str(ev.bsp_type) for ev in events]
        is_buy_flags = np.fromiter((1.0 if bool(ev.is_buy) else 0.0 for ev in events), dtype=np.float32, count=n_rows)
        is_sell_flags = 1.0 - is_buy_flags

        idx = meta.get("bsp_type_1")
        if idx is not None:
            matrix[:, idx] = np.fromiter((1.0 if t == "1" else 0.0 for t in bsp_types), dtype=np.float32, count=n_rows)
        idx = meta.get("bsp_type_2")
        if idx is not None:
            matrix[:, idx] = np.fromiter((1.0 if t == "2" else 0.0 for t in bsp_types), dtype=np.float32, count=n_rows)
        idx = meta.get("bsp_type_3")
        if idx is not None:
            matrix[:, idx] = np.fromiter((1.0 if t == "3" else 0.0 for t in bsp_types), dtype=np.float32, count=n_rows)

        idx = meta.get("is_buy_signal")
        if idx is not None:
            matrix[:, idx] = is_buy_flags

        idx_1 = np.fromiter((1.0 if t == "1" else 0.0 for t in bsp_types), dtype=np.float32, count=n_rows)
        idx_2 = np.fromiter((1.0 if t == "2" else 0.0 for t in bsp_types), dtype=np.float32, count=n_rows)
        idx_3 = np.fromiter((1.0 if t == "3" else 0.0 for t in bsp_types), dtype=np.float32, count=n_rows)

        idx = meta.get("bsp_type_1b")
        if idx is not None:
            matrix[:, idx] = is_buy_flags * idx_1
        idx = meta.get("bsp_type_2b")
        if idx is not None:
            matrix[:, idx] = is_buy_flags * idx_2
        idx = meta.get("bsp_type_3b")
        if idx is not None:
            matrix[:, idx] = is_buy_flags * idx_3

        idx = meta.get("bsp_type_1s")
        if idx is not None:
            matrix[:, idx] = is_sell_flags * idx_1
        idx = meta.get("bsp_type_2s")
        if idx is not None:
            matrix[:, idx] = is_sell_flags * idx_2
        idx = meta.get("bsp_type_3s")
        if idx is not None:
            matrix[:, idx] = is_sell_flags * idx_3

        meta_get = meta.get
        alias_dist_idx = meta_get("dist_to_zs_center")
        alias_bi_idx = meta_get("bi_count_in_zs")

        for row_idx, ev in enumerate(events):
            feature_map = ev.feature_map
            for feat_name, feat_value in feature_map.items():
                col_idx = meta_get(feat_name)
                if col_idx is None:
                    continue
                fv = float(feat_value)
                matrix[row_idx, col_idx] = fv if np.isfinite(fv) else 0.0

            if alias_dist_idx is not None and "distance_to_zhongshu_center" in feature_map:
                val = float(feature_map.get("distance_to_zhongshu_center", 0.0))
                matrix[row_idx, alias_dist_idx] = val if np.isfinite(val) else 0.0
            if alias_bi_idx is not None and "zs_bi_count" in feature_map:
                val = float(feature_map.get("zs_bi_count", 0.0))
                matrix[row_idx, alias_bi_idx] = val if np.isfinite(val) else 0.0

        return matrix, fnames

    def _predict_from_matrix(
        self,
        matrix: np.ndarray,
        fnames: List[str],
        is_buy: bool,
    ) -> np.ndarray:
        if matrix.shape[0] == 0:
            return np.empty((0, 3), dtype=np.float32)

        model = self.model_buy if is_buy else self.model_sell
        try:
            # Avoid DMatrix construction overhead on hot path.
            probs = model.inplace_predict(matrix)
        except Exception:
            dmat = xgb.DMatrix(
                matrix,
                feature_names=fnames,
                missing=np.nan,
            )
            probs = model.predict(dmat)
        arr = np.asarray(probs)
        if arr.ndim == 1:
            # Binary fallback: convert to 3-class compatible matrix.
            return np.stack([1.0 - arr, np.zeros_like(arr), arr], axis=1)
        if arr.ndim == 2 and arr.shape[1] == 2:
            # Binary model outputs [p_sl, p_pt] -> align to [p_sl, p_timeout, p_pt].
            return np.stack([arr[:, 0], np.zeros(arr.shape[0], dtype=arr.dtype), arr[:, 1]], axis=1)
        return arr

    def _predict_batch(
        self,
        events: Sequence[RawBSPEvent],
        is_buy: bool,
    ) -> np.ndarray:
        if not events:
            return np.empty((0, 3), dtype=np.float32)

        matrix, fnames = self._vectorize_batch(events, is_buy)
        return self._predict_from_matrix(matrix, fnames, is_buy)

    @staticmethod
    def _direction_from_primary(primary_probs: np.ndarray) -> np.ndarray:
        classes = np.argmax(primary_probs, axis=1).astype(np.int32)
        # class map: 0=SL(-1), 1=TIMEOUT(0), 2=PT(+1)
        direction = np.zeros_like(classes, dtype=np.int32)
        direction[classes == 0] = -1
        direction[classes == 2] = 1
        return direction

    @staticmethod
    def _direction_from_event_side(events: Sequence[RawBSPEvent]) -> np.ndarray:
        return np.array(
            [1 if bool(ev.is_buy) else -1 for ev in events],
            dtype=np.int32,
        )

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
            # Legacy fallback: use PT probability as execution confidence.
            if p_primary.shape[1] >= 3:
                out[active] = p_primary[active, 2].astype(np.float32)
            else:
                out[active] = p_primary[active, -1].astype(np.float32)
            return out

        p_block = p_primary[active]
        base_meta = getattr(self.meta_model, "model", self.meta_model)
        expected_total = getattr(base_meta, "n_features_in_", None)
        if isinstance(expected_total, (int, np.integer)):
            expect_primary_cols = int(expected_total) - int(X.shape[1])
            if expect_primary_cols == 2 and p_block.shape[1] >= 3:
                p_block = p_block[:, [0, 2]]
            elif expect_primary_cols == 1:
                if p_block.shape[1] >= 3:
                    p_block = p_block[:, [2]]
                else:
                    p_block = p_block[:, -1].reshape(-1, 1)

        meta_input = np.hstack(
            [np.nan_to_num(X[active], nan=0.0), p_block]
        )
        out[active] = np.asarray(
            self.meta_model.predict_proba(meta_input),
            dtype=np.float32,
        )
        return out

    @staticmethod
    def _primary_pt_selected(primary_probs: np.ndarray) -> np.ndarray:
        """Return whether primary model selects PT class for each event.

        Supports both legacy 3-class and binary models.
        """
        if primary_probs.ndim != 2 or primary_probs.shape[0] == 0:
            return np.zeros(primary_probs.shape[0], dtype=bool)

        if primary_probs.shape[1] >= 3:
            classes = np.argmax(primary_probs, axis=1).astype(np.int32)
            return classes == 2

        # Binary fallback: treat p(class=1) >= 0.5 as PT-selected.
        if primary_probs.shape[1] == 2:
            return primary_probs[:, 1] >= 0.5

        # Degenerate single-column fallback.
        return primary_probs[:, 0] >= 0.5

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

        n_events = len(events)
        probs = np.zeros(n_events, dtype=np.float32)
        direction = np.zeros(n_events, dtype=np.int32)
        primary_pt_selected = np.zeros(n_events, dtype=bool)
        primary_pt_conf = np.zeros(n_events, dtype=np.float32)

        if buy_events:
            buy_X, buy_fnames = self._vectorize_batch(buy_events, True)
            buy_primary = self._predict_from_matrix(buy_X, buy_fnames, True)
            buy_direction = np.ones(len(buy_events), dtype=np.int32)
            buy_probs = self._meta_prob_batch(buy_X, buy_primary, buy_direction)
            buy_pt_selected = self._primary_pt_selected(buy_primary)
            buy_pt_conf = (
                buy_primary[:, 2].astype(np.float32)
                if buy_primary.shape[1] >= 3
                else buy_primary[:, -1].astype(np.float32)
            )
            for i, event_idx in enumerate(buy_idx):
                probs[event_idx] = buy_probs[i]
                direction[event_idx] = 1
                primary_pt_selected[event_idx] = bool(buy_pt_selected[i])
                primary_pt_conf[event_idx] = buy_pt_conf[i]

        if sell_events:
            sell_X, sell_fnames = self._vectorize_batch(sell_events, False)
            sell_primary = self._predict_from_matrix(sell_X, sell_fnames, False)
            sell_direction = -np.ones(len(sell_events), dtype=np.int32)
            sell_probs = self._meta_prob_batch(sell_X, sell_primary, sell_direction)
            sell_pt_selected = self._primary_pt_selected(sell_primary)
            sell_pt_conf = (
                sell_primary[:, 2].astype(np.float32)
                if sell_primary.shape[1] >= 3
                else sell_primary[:, -1].astype(np.float32)
            )
            for i, event_idx in enumerate(sell_idx):
                probs[event_idx] = sell_probs[i]
                direction[event_idx] = -1
                primary_pt_selected[event_idx] = bool(sell_pt_selected[i])
                primary_pt_conf[event_idx] = sell_pt_conf[i]

        scored: List[ScoredSignalEvent] = []

        thresholds = np.fromiter(
            (self._event_threshold(events[idx], int(direction[idx])) for idx in range(n_events)),
            dtype=np.float32,
            count=n_events,
        )

        use_primary_gate = not self.has_meta_model
        if use_primary_gate:
            qualified_arr = np.logical_and(primary_pt_selected, probs >= thresholds)
        else:
            qualified_arr = probs >= thresholds

        # Optional fallback for fully empty symbols.
        use_fallback = bool(getattr(self.config, "empty_signal_fallback", False))
        target_rate = float(
            getattr(self.config, "empty_signal_target_rate", self._EMPTY_SIGNAL_TARGET_RATE)
        )
        target_rate = max(0.0, min(1.0, target_rate))
        if use_fallback and target_rate > 0.0 and (not np.any(qualified_arr)) and n_events > 0:
            q = max(0.0, min(1.0, 1.0 - target_rate))
            fallback_conf = probs if self.has_meta_model else primary_pt_conf
            adaptive_thr = float(np.nanquantile(fallback_conf, q))
            if not math.isfinite(adaptive_thr):
                adaptive_thr = float(np.nan_to_num(adaptive_thr, nan=1.0, posinf=1.0, neginf=0.0))
            normalized_conf = np.nan_to_num(fallback_conf, nan=0.0, posinf=1.0, neginf=0.0)
            qualified_arr = normalized_conf >= adaptive_thr
            probs = fallback_conf.astype(np.float32)
            direction = self._direction_from_event_side(events)

        for idx, ev in enumerate(events):
            prob = float(probs[idx])
            qualified = bool(qualified_arr[idx])
            if not qualified:
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
