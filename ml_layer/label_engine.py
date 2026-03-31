from __future__ import annotations

# flake8: noqa: E501

from typing import Dict, List

import numpy as np

from .config import LabelConfig


class LabelEngine:
    def __init__(self, config: LabelConfig):
        self.config = config

    @staticmethod
    def _rolling_volatility(closes: np.ndarray, window: int) -> np.ndarray:
        n = int(closes.size)
        out = np.zeros(n, dtype=np.float64)
        if n <= 2:
            return out
        safe_close = np.maximum(closes.astype(np.float64), 1e-9)
        log_ret = np.diff(np.log(safe_close))
        w = max(8, int(window))
        for i in range(1, n):
            s = max(0, i - w)
            seg = log_ret[s:i]
            out[i] = float(np.std(seg)) if seg.size > 1 else 0.0
        return out

    def _resolve_dynamic_pt_multiplier(
        self,
        vol_value: float,
        vol_q_low: float,
        vol_q_high: float,
    ) -> tuple[float, str]:
        if not bool(self.config.dynamic_pt_enabled):
            return float(self.config.pt_multiplier), "static"

        if not np.isfinite(vol_value) or vol_q_high <= vol_q_low:
            return float(self.config.pt_mid_vol_multiplier), "mid"

        if vol_value <= vol_q_low:
            return float(self.config.pt_low_vol_multiplier), "low"
        if vol_value >= vol_q_high:
            return float(self.config.pt_high_vol_multiplier), "high"
        return float(self.config.pt_mid_vol_multiplier), "mid"

    @staticmethod
    def _safe_float(v) -> float:
        try:
            return float(v)
        except Exception:
            return float("nan")

    @staticmethod
    def _label_to_class(label_raw: int) -> int:
        # Binary class map: SL(-1)->0, PT(+1)->1
        return 1 if int(label_raw) > 0 else 0

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
            holding_norm = 1.0
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

        idx2pos = {int(b["klu_idx"]): i for i, b in enumerate(bars)}
        highs = np.asarray([float(b["high"]) for b in bars], dtype=np.float64)
        lows = np.asarray([float(b["low"]) for b in bars], dtype=np.float64)
        closes = np.asarray([float(b["close"]) for b in bars], dtype=np.float64)
        open_ts = np.asarray([float(b["open_ts"]) for b in bars], dtype=np.float64)

        rolling_vol = None
        vol_q_low = 0.0
        vol_q_high = 0.0
        if bool(self.config.dynamic_pt_enabled):
            rolling_vol = self._rolling_volatility(
                closes=closes,
                window=max(8, int(self.config.vol_window)),
            )
            finite_vol = rolling_vol[np.isfinite(rolling_vol)]
            if finite_vol.size >= 20:
                q_low = float(np.clip(float(self.config.vol_quantile_low), 0.01, 0.98))
                q_high = float(np.clip(float(self.config.vol_quantile_high), q_low + 0.01, 0.99))
                vol_q_low = float(np.quantile(finite_vol, q_low))
                vol_q_high = float(np.quantile(finite_vol, q_high))
            else:
                rolling_vol = None

        labeled: List[Dict] = []
        for event in events:
            entry = float(event["trade_price"])
            if entry <= 0:
                continue

            t0_pos = idx2pos.get(int(event["klu_idx"]))
            if t0_pos is None:
                continue

            sl = self._compute_structural_sl(event)
            risk = abs(entry - sl)
            if risk <= 1e-8:
                risk = entry * 0.005
                sl = entry - risk if bool(event["is_buy"]) else entry + risk

            vol_value = float("nan")
            if rolling_vol is not None and 0 <= int(t0_pos) < int(rolling_vol.size):
                vol_value = float(rolling_vol[int(t0_pos)])

            pt_multiplier_used, vol_regime = self._resolve_dynamic_pt_multiplier(
                vol_value=vol_value,
                vol_q_low=vol_q_low,
                vol_q_high=vol_q_high,
            )

            pt = (
                entry + risk * pt_multiplier_used
                if bool(event["is_buy"])
                else entry - risk * pt_multiplier_used
            )
            end_pos = len(bars) - 1

            label_raw = -1
            hit_event = "end_of_data"
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

            # No timeout class: if PT/SL not triggered until data end, use realized sign.
            if hit_event == "end_of_data":
                if abs(realized_ret) < float(self.config.min_ret_threshold):
                    label_raw = -1
                    hit_event = "ret_floor_sl"
                else:
                    label_raw = 1 if realized_ret > 0 else -1
                    hit_event = "ret_sign_pt" if label_raw > 0 else "ret_sign_sl"

            labeled.append(
                {
                    **event,
                    "label_raw": int(label_raw),
                    "label": self._label_to_class(int(label_raw)),
                    "label_text": "PT" if label_raw == 1 else "SL",
                    "entry_price": float(entry),
                    "sl_price": float(sl),
                    "pt_price": float(pt),
                    "pt_multiplier_used": float(pt_multiplier_used),
                    "volatility_value": float(vol_value) if np.isfinite(vol_value) else None,
                    "volatility_regime": str(vol_regime),
                    "hit_event": hit_event,
                    "timeout_used": 0,
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
