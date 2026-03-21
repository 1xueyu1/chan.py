from __future__ import annotations

import json
from typing import Dict, List

import numpy as np
import xgboost as xgb

from .config import BacktestConfig
from .types import RawBSPEvent, ScoredSignalEvent


class DualModelGate:
    def __init__(self, config: BacktestConfig):
        self.config = config

        self.model_buy = xgb.Booster()
        self.model_buy.load_model(config.model_buy_path)

        self.model_sell = xgb.Booster()
        self.model_sell.load_model(config.model_sell_path)

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

        arr = np.full(len(meta), np.nan, dtype=np.float32)

        if "bsp_type_1" in meta:
            arr[meta["bsp_type_1"]] = 1.0 if event.bsp_type == "1" else 0.0
        if "bsp_type_2" in meta:
            arr[meta["bsp_type_2"]] = 1.0 if event.bsp_type == "2" else 0.0
        if "bsp_type_3" in meta:
            arr[meta["bsp_type_3"]] = 1.0 if event.bsp_type == "3" else 0.0

        for feat_name, feat_value in event.feature_map.items():
            if feat_name in meta:
                arr[meta[feat_name]] = feat_value

        return arr, meta, fnames

    def score_event(self, event: RawBSPEvent) -> ScoredSignalEvent:
        is_buy = bool(event.is_buy)
        arr, _meta, fnames = self._vectorize(event, is_buy)

        dmat = xgb.DMatrix(
            arr.reshape(1, -1),
            feature_names=fnames,
            missing=np.nan,
        )
        model = self.model_buy if is_buy else self.model_sell
        prob = float(model.predict(dmat)[0])

        qualified = prob >= self.config.signal_threshold
        if not qualified:
            signal = 0
        else:
            signal = 1 if is_buy else -1

        return ScoredSignalEvent(
            symbol=event.symbol,
            exec_time=event.exec_time,
            bsp_time=event.bsp_time,
            is_buy=event.is_buy,
            bsp_type=event.bsp_type,
            bsp_types_str=event.bsp_types_str,
            trade_price=event.trade_price,
            klu_idx=event.klu_idx,
            probability=prob,
            qualified=qualified,
            signal=signal,
        )

    def score_events(
        self,
        events: List[RawBSPEvent],
    ) -> List[ScoredSignalEvent]:
        return [self.score_event(ev) for ev in events]
