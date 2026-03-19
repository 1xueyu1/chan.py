from __future__ import annotations

from typing import Dict, List

import pandas as pd

from Chan import CChan
from ChanConfig import CChanConfig
from ChanModel.feature_center import build_features
from Common.CEnum import AUTYPE

from .config import BacktestConfig
from .types import RawBSPEvent


def _safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _make_exec_time(snapshot) -> pd.Timestamp:
    last_klu = snapshot[0][-1][-1]
    return pd.to_datetime(float(last_klu.time.ts), unit="s", utc=True)


def extract_raw_bsp_events(config: BacktestConfig, symbol: str) -> List[RawBSPEvent]:
    chan_cfg = dict(config.chan_config)
    chan_cfg["trigger_step"] = True
    chan_cfg.setdefault("skip_step", 0)

    chan = CChan(
        code=symbol,
        begin_time=config.begin_time,
        end_time=config.end_time,
        data_src=config.data_src,
        lv_list=[config.kl_type],
        config=CChanConfig(chan_cfg),
        autype=AUTYPE.NONE,
    )

    events: List[RawBSPEvent] = []
    seen_klu_idx = set()

    for snapshot in chan.step_load():
        if len(snapshot[0]) == 0 or len(snapshot[0][-1]) == 0:
            continue

        bsp_list = snapshot.get_latest_bsp()
        if not bsp_list:
            continue

        last_bsp = bsp_list[0]
        cur_lv_chan = snapshot[0]
        if last_bsp.klu.idx in seen_klu_idx:
            continue
        if len(cur_lv_chan) < 2 or cur_lv_chan[-2].idx != last_bsp.klu.klc.idx:
            continue

        seen_klu_idx.add(last_bsp.klu.idx)

        last_klu = cur_lv_chan[-1][-1]
        extra_feat = build_features(klu=last_klu, history=cur_lv_chan.lst, chan=cur_lv_chan)
        last_bsp.features.add_feat(extra_feat)

        feature_map = {
            feat_name: _safe_float(feat_value)
            for feat_name, feat_value in last_bsp.features.items()
        }

        event = RawBSPEvent(
            symbol=symbol,
            exec_time=_make_exec_time(snapshot),
            bsp_time=str(last_bsp.klu.time),
            is_buy=bool(last_bsp.is_buy),
            bsp_type=last_bsp.type[0].value[0],
            bsp_types_str=last_bsp.type2str(),
            trade_price=float(last_klu.close),
            klu_idx=int(last_bsp.klu.idx),
            feature_map=feature_map,
        )
        events.append(event)

    events.sort(key=lambda x: (x.exec_time, x.klu_idx))
    return events


def extract_multi_symbol_raw_events(config: BacktestConfig) -> Dict[str, List[RawBSPEvent]]:
    result = {}
    for symbol in config.normalized_symbols():
        result[symbol] = extract_raw_bsp_events(config, symbol)
    return result
