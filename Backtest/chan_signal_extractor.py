from __future__ import annotations

from typing import Dict, List

import pandas as pd

from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import AUTYPE, KL_TYPE
from Common.ChanException import CChanException, ErrCode
from RustCore import RustChanEngine, RustStepSnapshot, freq_from_kl_type, rust_config_path_from_chan_config

from .config import BacktestConfig
from .types import RawBSPEvent


def _safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _make_exec_time(snapshot) -> pd.Timestamp:
    if isinstance(snapshot, RustStepSnapshot):
        return _make_chan_ctime_exec_time(snapshot.latest_base_klu.time)
    last_klu = snapshot[0][-1][-1]
    return _make_chan_ctime_exec_time(last_klu.time)


def _make_chan_ctime_exec_time(value) -> pd.Timestamp:
    return pd.Timestamp(
        year=int(value.year),
        month=int(value.month),
        day=int(value.day),
        hour=int(value.hour),
        minute=int(value.minute),
        second=int(getattr(value, "second", 0)),
        tz="UTC",
    )


def _make_legacy_ctime_exec_time(ts) -> pd.Timestamp:
    dt = pd.to_datetime(ts, utc=True, errors="coerce")
    if pd.isna(dt):
        raise ValueError(f"invalid event timestamp: {ts!r}")
    return pd.Timestamp(dt).floor("s")


def _extract_feature_map(last_bsp) -> Dict[str, float]:
    features = getattr(last_bsp, "features", None)
    if features is None:
        return {}
    out: Dict[str, float] = {}
    try:
        for feat_name, feat_value in features.items():
            out[str(feat_name)] = _safe_float(feat_value)
    except Exception:
        return {}
    return out


def _normalize_ohlcv_bars(bars: pd.DataFrame) -> pd.DataFrame:
    if bars is None or len(bars) == 0:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"], index=pd.DatetimeIndex([], name="time"))

    frame = bars.copy()
    time_source = None
    for col in ("date", "open_time", "time", "datetime"):
        if col in frame.columns:
            time_source = frame[col]
            break
    if time_source is None:
        time_source = frame.index

    time_index = pd.to_datetime(time_source, utc=True, errors="coerce")
    required = ["open", "high", "low", "close"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"bars missing OHLC columns: {missing}")

    if "volume" not in frame.columns:
        frame["volume"] = 0.0

    frame = frame.assign(_time=time_index)
    for col in ["open", "high", "low", "close", "volume"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame = (
        frame.dropna(subset=["_time", "open", "high", "low", "close"])
        .drop_duplicates(subset=["_time"], keep="last")
        .sort_values("_time")
        .set_index("_time")
    )
    frame.index.name = "time"
    return frame[["open", "high", "low", "close", "volume"]].copy()


def _bool_float(value) -> float:
    return 1.0 if bool(value) else 0.0


def _ratio(num: float, den: float) -> float:
    return float(num) / (float(den) + 1e-12)


_MTF_CHAN_LEVEL_LABELS = {
    KL_TYPE.K_60M: "1h",
    KL_TYPE.K_DAY: "1d",
}


def _direction_is_up(value) -> bool:
    return str(value).lower().endswith("up")


def _direction_is_down(value) -> bool:
    return str(value).lower().endswith("down")


def _range_pos(value: float, low: float, high: float) -> float:
    return _ratio(float(value) - float(low), float(high) - float(low))


def _chan_mtf_lv_list(config: BacktestConfig) -> List[KL_TYPE]:
    if not bool(config.chan_config.get("use_rust_core", False)):
        return [config.kl_type]
    if not bool(config.chan_config.get("mtf_chan_features", False)):
        return [config.kl_type]

    candidates = [KL_TYPE.K_DAY, KL_TYPE.K_60M]
    levels = [level for level in candidates if level.value > config.kl_type.value]
    levels.append(config.kl_type)
    return levels


def _extract_mtf_chan_context_features(snapshot: RustStepSnapshot, base_idx: int, last_bsp, trade_price: float) -> Dict[str, float]:
    out: Dict[str, float] = {}
    signal_dir = 1.0 if bool(last_bsp.is_buy) else -1.0

    for lv_idx, lv in enumerate(snapshot.lv_list):
        if lv_idx == base_idx:
            continue
        label = _MTF_CHAN_LEVEL_LABELS.get(lv)
        if not label:
            continue
        prefix = f"mtf_{label}"
        try:
            state = snapshot.get_rust_snapshot(lv_idx)
        except Exception:
            continue

        counts = state.get("counts") or {}
        out[f"{prefix}_klc_cnt"] = float(counts.get("klcs", 0) or 0)
        out[f"{prefix}_bi_cnt"] = float(counts.get("bis", 0) or 0)
        out[f"{prefix}_seg_cnt"] = float(counts.get("segs", 0) or 0)
        out[f"{prefix}_segseg_cnt"] = float(counts.get("segs_seg", 0) or 0)
        out[f"{prefix}_zs_cnt"] = float(counts.get("bzs", 0) or 0)
        out[f"{prefix}_seg_zs_cnt"] = float(counts.get("szs", 0) or 0)
        out[f"{prefix}_bi_bsp_cnt"] = float(counts.get("bi_bsp", 0) or 0)
        out[f"{prefix}_seg_bsp_cnt"] = float(counts.get("seg_bsp", 0) or 0)

        last_klc = state.get("last_klc") or {}
        klc_high = _safe_float(last_klc.get("high"))
        klc_low = _safe_float(last_klc.get("low"))
        if pd.notna(klc_high) and pd.notna(klc_low):
            out[f"{prefix}_last_klc_unit_cnt"] = float(last_klc.get("unit_count", 0) or 0)
            out[f"{prefix}_last_klc_range_pct"] = _ratio(klc_high - klc_low, trade_price)
            out[f"{prefix}_event_pos_last_klc"] = _range_pos(trade_price, klc_low, klc_high)

        bis = state.get("bis") or []
        if bis:
            last_bi = bis[-1]
            begin_val = _safe_float(last_bi.get("begin_val"))
            end_val = _safe_float(last_bi.get("end_val"))
            bi_high = max(begin_val, end_val)
            bi_low = min(begin_val, end_val)
            is_up = _direction_is_up(last_bi.get("direction"))
            is_down = _direction_is_down(last_bi.get("direction"))
            out[f"{prefix}_last_bi_exists"] = 1.0
            out[f"{prefix}_last_bi_is_sure"] = _bool_float(last_bi.get("is_sure"))
            out[f"{prefix}_last_bi_is_up"] = _bool_float(is_up)
            out[f"{prefix}_last_bi_amp_pct"] = _ratio(last_bi.get("amp", bi_high - bi_low), begin_val)
            out[f"{prefix}_last_bi_klu_cnt"] = float(last_bi.get("klu_cnt", 0) or 0)
            out[f"{prefix}_last_bi_klc_cnt"] = float(last_bi.get("klc_cnt", 0) or 0)
            out[f"{prefix}_event_pos_last_bi"] = _range_pos(trade_price, bi_low, bi_high)
            out[f"{prefix}_event_vs_last_bi_end_pct"] = _ratio(trade_price - end_val, end_val)
            out[f"{prefix}_signal_align_last_bi"] = _bool_float((signal_dir > 0 and is_up) or (signal_dir < 0 and is_down))
        else:
            out[f"{prefix}_last_bi_exists"] = 0.0

        segs = state.get("segs") or []
        if segs:
            last_seg = segs[-1]
            begin_val = _safe_float(last_seg.get("begin_val"))
            end_val = _safe_float(last_seg.get("end_val"))
            seg_high = max(begin_val, end_val)
            seg_low = min(begin_val, end_val)
            is_up = _direction_is_up(last_seg.get("direction"))
            is_down = _direction_is_down(last_seg.get("direction"))
            out[f"{prefix}_last_seg_exists"] = 1.0
            out[f"{prefix}_last_seg_is_sure"] = _bool_float(last_seg.get("is_sure"))
            out[f"{prefix}_last_seg_is_up"] = _bool_float(is_up)
            out[f"{prefix}_last_seg_amp_pct"] = _ratio(last_seg.get("amp", seg_high - seg_low), begin_val)
            out[f"{prefix}_last_seg_klu_cnt"] = float(last_seg.get("klu_cnt", 0) or 0)
            out[f"{prefix}_last_seg_bi_cnt"] = float(last_seg.get("element_cnt", 0) or 0)
            out[f"{prefix}_last_seg_zs_cnt"] = float(len(last_seg.get("zs_lst") or []))
            out[f"{prefix}_last_seg_multi_zs_cnt"] = float(last_seg.get("multi_bi_zs_cnt", 0) or 0)
            out[f"{prefix}_event_pos_last_seg"] = _range_pos(trade_price, seg_low, seg_high)
            out[f"{prefix}_event_vs_last_seg_end_pct"] = _ratio(trade_price - end_val, end_val)
            out[f"{prefix}_signal_align_last_seg"] = _bool_float((signal_dir > 0 and is_up) or (signal_dir < 0 and is_down))
        else:
            out[f"{prefix}_last_seg_exists"] = 0.0

        zss = state.get("bzs") or []
        if zss:
            last_zs = zss[-1]
            zs_low = _safe_float(last_zs.get("low"))
            zs_high = _safe_float(last_zs.get("high"))
            zs_mid = _safe_float(last_zs.get("mid"))
            peak_high = _safe_float(last_zs.get("peak_high", zs_high))
            peak_low = _safe_float(last_zs.get("peak_low", zs_low))
            element_list = last_zs.get("element_list") or []
            out[f"{prefix}_last_zs_exists"] = 1.0
            out[f"{prefix}_last_zs_is_sure"] = _bool_float(last_zs.get("is_sure"))
            out[f"{prefix}_last_zs_width_pct"] = _ratio(zs_high - zs_low, zs_mid)
            out[f"{prefix}_last_zs_peak_width_pct"] = _ratio(peak_high - peak_low, zs_mid)
            out[f"{prefix}_last_zs_bi_cnt"] = float(len(element_list))
            out[f"{prefix}_event_inside_last_zs"] = _bool_float(zs_low <= trade_price <= zs_high)
            out[f"{prefix}_event_pos_last_zs"] = _range_pos(trade_price, zs_low, zs_high)
            out[f"{prefix}_event_vs_last_zs_mid_pct"] = _ratio(trade_price - zs_mid, zs_mid)
            out[f"{prefix}_event_vs_last_zs_high_pct"] = _ratio(trade_price - zs_high, zs_high)
            out[f"{prefix}_event_vs_last_zs_low_pct"] = _ratio(trade_price - zs_low, zs_low)
            out[f"{prefix}_signal_align_last_zs_mid"] = _bool_float(
                (signal_dir > 0 and trade_price >= zs_mid) or (signal_dir < 0 and trade_price <= zs_mid)
            )
        else:
            out[f"{prefix}_last_zs_exists"] = 0.0

    return out


def _extract_chan_context_features(cur_lv_chan, last_bsp, last_klu) -> Dict[str, float]:
    out: Dict[str, float] = {}

    bi_list = cur_lv_chan.bi_list
    seg_list = cur_lv_chan.seg_list
    zs_list = cur_lv_chan.zs_list
    bsp_bi = last_bsp.bi

    out["ctx_klc_cnt"] = float(len(cur_lv_chan))
    out["ctx_bi_cnt"] = float(len(bi_list))
    out["ctx_seg_cnt"] = float(len(seg_list))
    out["ctx_zs_cnt"] = float(len(zs_list))
    out["ctx_bsp_bi_idx_pct"] = _ratio(bsp_bi.idx + 1, max(len(bi_list), 1))
    out["ctx_bars_since_bsp_klu"] = float(last_klu.idx - last_bsp.klu.idx)

    if len(cur_lv_chan) > 0:
        last_klc = cur_lv_chan[-1]
        out["ctx_last_klc_unit_cnt"] = float(len(last_klc))
        out["ctx_last_klc_range_pct"] = _ratio(last_klc.high - last_klc.low, last_klu.close)
        out["ctx_last_klc_close_pos"] = _ratio(last_klu.close - last_klc.low, last_klc.high - last_klc.low)

    if len(bi_list) > 0:
        last_bi = bi_list[-1]
        out["ctx_last_bi_is_sure"] = _bool_float(last_bi.is_sure)
        out["ctx_last_bi_is_up"] = _bool_float(last_bi.is_up())
        out["ctx_last_bi_amp_pct"] = _ratio(last_bi.amp(), last_bi.get_begin_val())
        out["ctx_last_bi_klu_cnt"] = float(last_bi.get_klu_cnt())
        out["ctx_last_bi_klc_cnt"] = float(last_bi.get_klc_cnt())
        out["ctx_bsp_is_last_bi"] = _bool_float(last_bi.idx == bsp_bi.idx)

    if bsp_bi is not None:
        out["ctx_bsp_bi_is_sure"] = _bool_float(bsp_bi.is_sure)
        out["ctx_bsp_bi_is_up"] = _bool_float(bsp_bi.is_up())
        out["ctx_bsp_bi_amp_pct"] = _ratio(bsp_bi.amp(), bsp_bi.get_begin_val())
        out["ctx_bsp_bi_klu_cnt"] = float(bsp_bi.get_klu_cnt())
        out["ctx_bsp_bi_klc_cnt"] = float(bsp_bi.get_klc_cnt())
        if bsp_bi.pre is not None:
            out["ctx_bsp_pre_bi_amp_ratio"] = _ratio(bsp_bi.amp(), bsp_bi.pre.amp())
            out["ctx_bsp_pre_bi_klu_ratio"] = _ratio(bsp_bi.get_klu_cnt(), bsp_bi.pre.get_klu_cnt())
            out["ctx_bsp_end_vs_pre_end_pct"] = _ratio(bsp_bi.get_end_val() - bsp_bi.pre.get_end_val(), bsp_bi.pre.get_end_val())
        if bsp_bi.pre is not None and bsp_bi.pre.pre is not None:
            out["ctx_bsp_pre2_bi_amp_ratio"] = _ratio(bsp_bi.amp(), bsp_bi.pre.pre.amp())

    if len(bi_list) >= 3:
        recent = list(bi_list[-5:])
        amps = [bi.amp() for bi in recent]
        klu_cnts = [bi.get_klu_cnt() for bi in recent]
        out["ctx_recent_bi_amp_mean"] = float(sum(amps) / len(amps))
        out["ctx_recent_bi_amp_std"] = float(pd.Series(amps).std(ddof=0))
        out["ctx_recent_bi_klu_mean"] = float(sum(klu_cnts) / len(klu_cnts))
        out["ctx_recent_bi_up_cnt"] = float(sum(1 for bi in recent if bi.is_up()))
        out["ctx_recent_bi_down_cnt"] = float(sum(1 for bi in recent if bi.is_down()))

    parent_seg = getattr(bsp_bi, "parent_seg", None)
    if parent_seg is not None:
        out["ctx_parent_seg_is_sure"] = _bool_float(parent_seg.is_sure)
        out["ctx_parent_seg_is_up"] = _bool_float(parent_seg.is_up())
        out["ctx_parent_seg_bi_cnt"] = float(parent_seg.cal_bi_cnt())
        out["ctx_parent_seg_klu_cnt"] = float(parent_seg.get_klu_cnt())
        out["ctx_parent_seg_amp_pct"] = _ratio(parent_seg.amp(), parent_seg.get_begin_val())
        out["ctx_bsp_pos_in_parent_seg"] = _ratio(bsp_bi.idx - parent_seg.start_bi.idx, max(parent_seg.end_bi.idx - parent_seg.start_bi.idx, 1))
        out["ctx_parent_seg_zs_cnt"] = float(len(parent_seg.zs_lst))
        out["ctx_parent_seg_multi_zs_cnt"] = float(parent_seg.get_multi_bi_zs_cnt())
        out["ctx_parent_seg_end_is_bsp"] = _bool_float(parent_seg.end_bi.idx == bsp_bi.idx)

    if len(seg_list) > 0:
        last_seg = seg_list[-1]
        out["ctx_last_seg_is_sure"] = _bool_float(last_seg.is_sure)
        out["ctx_last_seg_is_up"] = _bool_float(last_seg.is_up())
        out["ctx_last_seg_bi_cnt"] = float(last_seg.cal_bi_cnt())
        out["ctx_last_seg_klu_cnt"] = float(last_seg.get_klu_cnt())
        out["ctx_last_seg_amp_pct"] = _ratio(last_seg.amp(), last_seg.get_begin_val())
        out["ctx_last_seg_zs_cnt"] = float(len(last_seg.zs_lst))
        out["ctx_last_seg_multi_zs_cnt"] = float(last_seg.get_multi_bi_zs_cnt())

    if len(zs_list) > 0:
        last_zs = zs_list[-1]
        out["ctx_last_zs_width_pct"] = _ratio(last_zs.high - last_zs.low, last_zs.mid)
        out["ctx_last_zs_peak_width_pct"] = _ratio(last_zs.peak_high - last_zs.peak_low, last_zs.mid)
        out["ctx_last_zs_bi_cnt"] = float(last_zs.end_bi.idx - last_zs.begin_bi.idx + 1)
        out["ctx_bsp_inside_last_zs"] = _bool_float(last_zs.begin_bi.idx <= bsp_bi.idx <= last_zs.end_bi.idx)
        out["ctx_bsp_after_last_zs_bi_cnt"] = float(bsp_bi.idx - last_zs.end_bi.idx)
        out["ctx_close_vs_last_zs_mid_pct"] = _ratio(last_klu.close - last_zs.mid, last_zs.mid)
        out["ctx_close_vs_last_zs_high_pct"] = _ratio(last_klu.close - last_zs.high, last_zs.high)
        out["ctx_close_vs_last_zs_low_pct"] = _ratio(last_klu.close - last_zs.low, last_zs.low)

    return out


def extract_raw_bsp_events(config: BacktestConfig, symbol: str) -> List[RawBSPEvent]:
    chan_cfg = dict(config.chan_config)
    chan_cfg["trigger_step"] = True
    chan_cfg.setdefault("skip_step", 0)
    lv_list = _chan_mtf_lv_list(config)
    base_idx = lv_list.index(config.kl_type)

    chan = CChan(
        code=symbol,
        begin_time=config.begin_time,
        end_time=config.end_time,
        data_src=config.data_src,
        lv_list=lv_list,
        config=CChanConfig(chan_cfg),
        autype=AUTYPE.NONE,
    )

    events: List[RawBSPEvent] = []
    seen_klu_idx = set()

    try:
        for snapshot in chan.step_load():
            event_exec_time = None
            event_trade_price = None
            feature_map = None
            if isinstance(snapshot, RustStepSnapshot):
                query_idx = base_idx if len(lv_list) > 1 else None
                fast_event = snapshot.get_latest_bsp_event_fast(query_idx)
                if fast_event is not None:
                    last_bsp = fast_event["bsp"]
                    if last_bsp.klu.idx in seen_klu_idx:
                        continue
                    last_klc_idx = snapshot.get_last_klc_idx_fast(query_idx)
                    bsp_klc_idx = getattr(getattr(last_bsp.klu, "klc", None), "idx", None)
                    if last_klc_idx is None or bsp_klc_idx is None or int(last_klc_idx) - 1 != int(bsp_klc_idx):
                        continue
                    feature_map = _extract_feature_map(last_bsp)
                    for feat_name, feat_value in fast_event.get("context_features", []):
                        feat_name = str(feat_name)
                        value = _safe_float(feat_value)
                        if feat_name.startswith("ctx_"):
                            feature_map[feat_name] = value
                        else:
                            feature_map.setdefault(feat_name, value)
                    first_type = str(last_bsp.type[0].value) if getattr(last_bsp, "type", None) else ""
                    if first_type.startswith("2"):
                        last_bsp = snapshot.enrich_bsp(last_bsp, query_idx)
                        feature_map.update(_extract_feature_map(last_bsp))
                    event_exec_time = _make_chan_ctime_exec_time(snapshot.latest_base_klu.time)
                    event_trade_price = float(snapshot.latest_base_klu.close)
                else:
                    bsp_list = snapshot.get_latest_bsp_fast(query_idx)
                    if not bsp_list:
                        continue
                    last_bsp = bsp_list[0]
                    if last_bsp.klu.idx in seen_klu_idx:
                        continue
                    last_klc_idx = snapshot.get_last_klc_idx_fast(query_idx)
                    if last_klc_idx is None or int(last_klc_idx) - 1 != last_bsp.klu.klc.idx:
                        continue
                    cur_lv_chan = snapshot[base_idx]
                    last_bsp = snapshot.enrich_bsp(last_bsp, query_idx)
                    last_klu = cur_lv_chan[-1][-1]
            else:
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
                last_klu = cur_lv_chan[-1][-1]

            seen_klu_idx.add(last_bsp.klu.idx)

            if feature_map is None:
                feature_map = _extract_feature_map(last_bsp)
                try:
                    feature_map.update(_extract_chan_context_features(cur_lv_chan, last_bsp, last_klu))
                except Exception:
                    pass
            if event_exec_time is None:
                event_exec_time = _make_exec_time(snapshot)
            if event_trade_price is None:
                event_trade_price = float(last_klu.close)
            if isinstance(snapshot, RustStepSnapshot) and len(lv_list) > 1:
                feature_map.update(_extract_mtf_chan_context_features(snapshot, base_idx, last_bsp, event_trade_price))
            event = RawBSPEvent(
                symbol=symbol,
                exec_time=event_exec_time,
                bsp_time=str(last_bsp.klu.time),
                is_buy=bool(last_bsp.is_buy),
                bsp_type=last_bsp.type[0].value[0],
                bsp_types_str=last_bsp.type2str(),
                trade_price=event_trade_price,
                klu_idx=int(last_bsp.klu.idx),
                feature_map=feature_map,
            )
            events.append(event)
    except CChanException as ex:
        if ex.errcode == ErrCode.NO_DATA:
            print(f"[BACKTEST][WARN] {symbol} no data in range, skip signal extraction.")
            return []
        raise

    events.sort(key=lambda x: (x.exec_time, x.klu_idx))
    return events


def extract_raw_bsp_events_from_bars(config: BacktestConfig, symbol: str, bars: pd.DataFrame) -> List[RawBSPEvent]:
    """Extract BSP events from an in-memory OHLCV frame.

    This is the live/dry-run path used by freqtrade.  It avoids the historical
    parquet data source and only consumes the candles supplied by the caller.
    """
    frame = _normalize_ohlcv_bars(bars)
    if frame.empty:
        return []

    chan_cfg = dict(config.chan_config)
    chan_cfg["trigger_step"] = True
    chan_cfg.setdefault("skip_step", 0)
    rust_config_path = rust_config_path_from_chan_config(CChanConfig(chan_cfg))
    base_freq = freq_from_kl_type(config.kl_type)
    engine = RustChanEngine(freq=base_freq, config_path=rust_config_path)

    events: List[RawBSPEvent] = []
    seen_klu_idx = set()

    for ts, row in frame.iterrows():
        engine.push_bar(ts, row.open, row.high, row.low, row.close, row.volume)
        engine.cal_seg_and_zs()

        fast_event = engine.latest_bi_bsp_event()
        if fast_event is None:
            continue

        last_bsp = fast_event["bsp"]
        if last_bsp.klu.idx in seen_klu_idx:
            continue

        last_klc_idx = engine.last_klc_idx()
        bsp_klc_idx = getattr(getattr(last_bsp.klu, "klc", None), "idx", None)
        if last_klc_idx is None or bsp_klc_idx is None or int(last_klc_idx) - 1 != int(bsp_klc_idx):
            continue

        seen_klu_idx.add(last_bsp.klu.idx)
        feature_map = _extract_feature_map(last_bsp)
        for feat_name, feat_value in fast_event.get("context_features", []):
            feat_name = str(feat_name)
            value = _safe_float(feat_value)
            if feat_name.startswith("ctx_"):
                feature_map[feat_name] = value
            else:
                feature_map.setdefault(feat_name, value)
        first_type = str(last_bsp.type[0].value) if getattr(last_bsp, "type", None) else ""
        if first_type.startswith("2"):
            last_bsp = engine.enrich_bsp(last_bsp)
            feature_map.update(_extract_feature_map(last_bsp))

        actual_exec_time = pd.to_datetime(ts, utc=True)
        events.append(
            RawBSPEvent(
                symbol=symbol,
                exec_time=_make_legacy_ctime_exec_time(actual_exec_time),
                bsp_time=str(last_bsp.klu.time),
                is_buy=bool(last_bsp.is_buy),
                bsp_type=first_type[0] if first_type else "",
                bsp_types_str=last_bsp.type2str(),
                trade_price=float(row.close),
                klu_idx=int(last_bsp.klu.idx),
                feature_map=feature_map,
                actual_exec_time=actual_exec_time,
            )
        )

    events.sort(key=lambda x: (x.exec_time, x.klu_idx))
    return events


def extract_multi_symbol_raw_events(config: BacktestConfig) -> Dict[str, List[RawBSPEvent]]:
    result = {}
    for symbol in config.normalized_symbols():
        result[symbol] = extract_raw_bsp_events(config, symbol)
    return result
