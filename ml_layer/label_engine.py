from __future__ import annotations

# flake8: noqa: E501

from typing import Dict, List

import numpy as np

from .config import LabelConfig


class LabelEngine:
    def __init__(self, config: LabelConfig):
        self.config = config

    @staticmethod
    def _safe_float(v) -> float:
        try:
            return float(v)
        except Exception:
            return float("nan")

    @staticmethod
    def _label_to_class(label_raw: int) -> int:
        return int(label_raw + 1)

    def _compute_structural_sl(self, event: Dict) -> float:
        entry = float(event["trade_price"])
        is_buy = bool(event["is_buy"])
        bi_start = self._safe_float(event.get("bi_start_price", float("nan")))
        zs_low = self._safe_float(event.get("zs_low", float("nan")))
        zs_high = self._safe_float(event.get("zs_high", float("nan")))
        candidates = []

        if is_buy:
            if np.isfinite(bi_start) and bi_start < entry:
                candidates.append(bi_start)
            if np.isfinite(zs_low) and zs_low < entry:
                candidates.append(zs_low)
            return max(candidates) if candidates else entry * 0.99

        if np.isfinite(bi_start) and bi_start > entry:
            candidates.append(bi_start)
        if np.isfinite(zs_high) and zs_high > entry:
            candidates.append(zs_high)
        return min(candidates) if candidates else entry * 1.01

    def _compute_tw_ibs_weights(self, samples: List[Dict]) -> None:
        if not samples:
            return
        max_pos = max(int(item["t1_pos"]) for item in samples)
        diff = np.zeros(max_pos + 3, dtype=np.float64)
        for item in samples:
            s = int(item["t0_pos"])
            e = int(item["t1_pos"])
            diff[s] += 1.0
            diff[e + 1] -= 1.0

        active = np.cumsum(diff)
        prefix = np.cumsum(active)

        ws = []
        for item in samples:
            s = int(item["t0_pos"])
            e = int(item["t1_pos"])
            overlap_avg = float((prefix[e] - (prefix[s - 1] if s > 0 else 0.0)) / max(1, e - s + 1))
            overlap_avg = max(1.0, overlap_avg)
            timeout_used = max(1, int(item.get("timeout_used", 1)))
            holding_norm = max(0.05, min(1.0, float(item["holding_bars"]) / float(timeout_used)))
            raw_w = holding_norm / overlap_avg
            item["overlap_count"] = overlap_avg
            item["holding_time_normalized"] = holding_norm
            item["sample_weight"] = raw_w
            ws.append(raw_w)

        mean_w = float(np.mean(ws)) if ws else 1.0
        if mean_w > 0:
            for item in samples:
                item["sample_weight"] = float(item["sample_weight"] / mean_w)

    def transform(self, events: List[Dict], bars: List[Dict]) -> List[Dict]:
        if not events or not bars:
            return []

        weak_types = {x.strip() for x in str(self.config.weak_bsp_types).split(",") if x.strip()}
        idx2pos = {int(b["klu_idx"]): i for i, b in enumerate(bars)}
        highs = np.asarray([float(b["high"]) for b in bars], dtype=np.float64)
        lows = np.asarray([float(b["low"]) for b in bars], dtype=np.float64)
        closes = np.asarray([float(b["close"]) for b in bars], dtype=np.float64)
        open_ts = np.asarray([float(b["open_ts"]) for b in bars], dtype=np.float64)

        labeled: List[Dict] = []
        for event in events:
            entry = float(event["trade_price"])
            if entry <= 0:
                continue

            t0_pos = idx2pos.get(int(event["klu_idx"]))
            if t0_pos is None:
                continue

            timeout_used = self.config.weak_timeout_bars if event["bsp_main_type"] in weak_types else self.config.timeout_bars
            timeout_used = max(1, int(timeout_used))

            sl = self._compute_structural_sl(event)
            risk = abs(entry - sl)
            if risk <= 1e-8:
                risk = entry * 0.005
                sl = entry - risk if bool(event["is_buy"]) else entry + risk

            pt = entry + risk * self.config.pt_multiplier if bool(event["is_buy"]) else entry - risk * self.config.pt_multiplier
            end_pos = min(len(bars) - 1, t0_pos + timeout_used)

            label_raw = 0
            hit_event = "timeout"
            t1_pos = end_pos

            for pos in range(t0_pos + 1, end_pos + 1):
                high = float(highs[pos])
                low = float(lows[pos])
                if bool(event["is_buy"]):
                    hit_pt = high >= pt
                    hit_sl = low <= sl
                else:
                    hit_pt = low <= pt
                    hit_sl = high >= sl

                if hit_pt and hit_sl:
                    label_raw = -1
                    hit_event = "sl_tie"
                    t1_pos = pos
                    break
                if hit_pt:
                    label_raw = 1
                    hit_event = "pt"
                    t1_pos = pos
                    break
                if hit_sl:
                    label_raw = -1
                    hit_event = "sl"
                    t1_pos = pos
                    break

            exit_price = float(closes[t1_pos])
            if bool(event["is_buy"]):
                realized_ret = (exit_price - entry) / (entry + 1e-9)
            else:
                realized_ret = (entry - exit_price) / (entry + 1e-9)

            if abs(realized_ret) < float(self.config.min_ret_threshold) and label_raw != 0:
                label_raw = 0
                hit_event = "ret_floor_timeout"

            labeled.append(
                {
                    **event,
                    "label_raw": int(label_raw),
                    "label": self._label_to_class(int(label_raw)),
                    "label_text": "PT" if label_raw == 1 else ("SL" if label_raw == -1 else "TIMEOUT"),
                    "entry_price": float(entry),
                    "sl_price": float(sl),
                    "pt_price": float(pt),
                    "hit_event": hit_event,
                    "timeout_used": int(timeout_used),
                    "t0_pos": int(t0_pos),
                    "t1_pos": int(t1_pos),
                    "t0_ts": float(event["open_ts"]),
                    "t1_ts": float(open_ts[t1_pos]),
                    "holding_bars": int(t1_pos - t0_pos),
                    "realized_return": float(realized_ret),
                }
            )

        self._compute_tw_ibs_weights(labeled)
        return labeled
