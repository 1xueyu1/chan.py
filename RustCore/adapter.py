"""Thin Python adapter for the Rust chan-core extension.

The adapter keeps Rust integration optional: existing Python code can import this
module safely, then enable the Rust path only when ``chan_core_py`` is installed.
"""

from __future__ import annotations

import json
import hashlib
import math
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union

from ChanModel.Features import CFeatures
from Common.CEnum import BSP_TYPE
from Common.CTime import CTime

try:
    import chan_core_py
except ImportError:  # pragma: no cover - depends on local Rust build
    chan_core_py = None


TimestampLike = Union[int, float, str, datetime]
BarTuple = Tuple[TimestampLike, float, float, float, float, float]


_KL_TYPE_TO_FREQ = {
    "K_1S": "1s",
    "K_3S": "3s",
    "K_5S": "5s",
    "K_10S": "10s",
    "K_15S": "15s",
    "K_20S": "20s",
    "K_30S": "30s",
    "K_1M": "1m",
    "K_3M": "3m",
    "K_5M": "5m",
    "K_10M": "10m",
    "K_15M": "15m",
    "K_30M": "30m",
    "K_60M": "1h",
    "K_DAY": "1d",
    "K_WEEK": "1w",
}

_FREQ_TO_SECONDS = {
    "1s": 1,
    "3s": 3,
    "5s": 5,
    "10s": 10,
    "15s": 15,
    "20s": 20,
    "30s": 30,
    "1m": 60,
    "3m": 180,
    "5m": 300,
    "10m": 600,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "1d": 86400,
    "1w": 604800,
}


class RustCoreUnavailable(RuntimeError):
    """Raised when the Rust extension has not been built or installed."""


def is_rust_core_available() -> bool:
    return chan_core_py is not None


def _require_rust_core():
    if chan_core_py is None:
        raise RustCoreUnavailable(
            "chan_core_py is not installed. Install/build the optional "
            "Rust extension, for example from the chan-core-py crate with "
            "`maturin develop --release`."
        )
    return chan_core_py


_BSP_TYPE_TO_RUST = {
    "1": "T1",
    "1p": "T1P",
    "2": "T2",
    "2s": "T2S",
    "3a": "T3A",
    "3b": "T3B",
}

_MACD_ALGO_TO_RUST = {
    "AREA": "area",
    "PEAK": "peak",
    "FULL_AREA": "fullarea",
    "DIFF": "diff",
    "SLOPE": "slope",
    "AMP": "amp",
    "AMOUNT": "amount",
    "VOLUMN": "volume",
    "VOLUME": "volume",
    "VOLUMN_AVG": "volumeavg",
    "VOLUME_AVG": "volumeavg",
    "AMOUNT_AVG": "amountavg",
    "TURNRATE_AVG": "turnrateavg",
    "RSI": "rsi",
}

_FX_CHECK_TO_RUST = {
    "STRICT": "strict",
    "LOSS": "loose",
    "LOOSE": "loose",
    "HALF": "half",
    "TOTALLY": "totally",
}


def _enum_name(value) -> str:
    name = getattr(value, "name", None)
    if name is not None:
        return str(name)
    text = str(value)
    return text.rsplit(".", 1)[-1]


def _enum_text(value) -> str:
    raw = getattr(value, "value", value)
    if isinstance(raw, str):
        return raw
    return _enum_name(value).lower()


def _finite_or_inf(value):
    number = float(value)
    if math.isinf(number):
        return "inf" if number > 0 else "-inf"
    return number


def _macd_algo_to_rust(value) -> str:
    name = _enum_name(value).upper()
    if name in _MACD_ALGO_TO_RUST:
        return _MACD_ALGO_TO_RUST[name]
    text = str(value).lower().replace("_", "")
    return text


def _fx_check_to_rust(value) -> str:
    name = _enum_name(value).upper()
    return _FX_CHECK_TO_RUST.get(name, name.lower())


def _bsp_target_types_to_rust(target_types) -> list[str]:
    result = []
    for item in target_types:
        value = _enum_text(item)
        result.append(_BSP_TYPE_TO_RUST.get(value, value))
    return result


def _point_config_to_rust(conf) -> dict:
    return {
        "divergence_rate": _finite_or_inf(conf.divergence_rate),
        "min_zs_cnt": int(conf.min_zs_cnt),
        "bsp1_only_multibi_zs": bool(conf.bsp1_only_multibi_zs),
        "max_bs2_rate": float(conf.max_bs2_rate),
        "macd_algo": _macd_algo_to_rust(conf.macd_algo),
        "bs1_peak": bool(conf.bs1_peak),
        "target_types": _bsp_target_types_to_rust(conf.target_types),
        "bsp2_follow_1": bool(conf.bsp2_follow_1),
        "bsp3_follow_1": bool(conf.bsp3_follow_1),
        "bsp3_peak": bool(conf.bsp3_peak),
        "bsp2s_follow_2": bool(conf.bsp2s_follow_2),
        "max_bsp2s_lv": None if conf.max_bsp2s_lv is None else int(conf.max_bsp2s_lv),
        "strict_bsp3": bool(conf.strict_bsp3),
        "bsp3a_max_zs_cnt": int(conf.bsp3a_max_zs_cnt),
    }


def _bs_point_config_to_rust(conf) -> dict:
    return {
        "b_conf": _point_config_to_rust(conf.b_conf),
        "s_conf": _point_config_to_rust(conf.s_conf),
    }


def rust_config_payload_from_chan_config(chan_config) -> dict:
    """Build a Rust ChanConfig payload from the active Python CChanConfig."""
    macd_config = getattr(chan_config, "macd_config", {}) or {}
    return {
        "bi_config": {
            "bi_algo": getattr(chan_config.bi_conf, "bi_algo", "normal"),
            "is_strict": bool(getattr(chan_config.bi_conf, "is_strict", True)),
            "bi_fx_check": _fx_check_to_rust(getattr(chan_config.bi_conf, "bi_fx_check", "strict")),
            "gap_as_kl": bool(getattr(chan_config.bi_conf, "gap_as_kl", False)),
            "bi_end_is_peak": bool(getattr(chan_config.bi_conf, "bi_end_is_peak", True)),
            "bi_allow_sub_peak": bool(getattr(chan_config.bi_conf, "bi_allow_sub_peak", True)),
        },
        "seg_config": {
            "seg_algo": getattr(chan_config.seg_conf, "seg_algo", "chan"),
            "left_seg_method": _enum_text(getattr(chan_config.seg_conf, "left_method", "peak")),
        },
        "zs_config": {
            "need_combine": bool(getattr(chan_config.zs_conf, "need_combine", True)),
            "zs_combine_mode": getattr(chan_config.zs_conf, "zs_combine_mode", "zs"),
            "one_bi_zs": bool(getattr(chan_config.zs_conf, "one_bi_zs", False)),
            "zs_algo": getattr(chan_config.zs_conf, "zs_algo", "normal"),
        },
        "bs_point_config": _bs_point_config_to_rust(chan_config.bs_point_conf),
        "seg_bs_point_config": _bs_point_config_to_rust(chan_config.seg_bs_point_conf),
        "global_config": {
            "trigger_step": bool(getattr(chan_config, "trigger_step", True)),
            "skip_step": int(getattr(chan_config, "skip_step", 0)),
            "kl_data_check": bool(getattr(chan_config, "kl_data_check", True)),
            "max_kl_misalign_cnt": int(getattr(chan_config, "max_kl_misalgin_cnt", 0)),
            "max_kl_inconsistent_cnt": int(getattr(chan_config, "max_kl_inconsistent_cnt", 0)),
            "auto_skip_illegal_sub_lv": bool(getattr(chan_config, "auto_skip_illegal_sub_lv", False)),
            "print_warning": bool(getattr(chan_config, "print_warning", True)),
            "print_err_time": bool(getattr(chan_config, "print_err_time", False)),
            "mean_metrics": [int(item) for item in getattr(chan_config, "mean_metrics", [])],
            "trend_metrics": [int(item) for item in getattr(chan_config, "trend_metrics", [])],
        },
        "indicators": {
            "macd": {
                "fast": int(macd_config.get("fast", 12)),
                "slow": int(macd_config.get("slow", 26)),
                "signal": int(macd_config.get("signal", 9)),
            },
            "boll": {
                "period": int(getattr(chan_config, "boll_n", 20)),
                "std_dev": 2.0,
            },
            "rsi": {
                "period": int(getattr(chan_config, "rsi_cycle", 14)),
            },
            "kdj": {
                "n": int(getattr(chan_config, "kdj_cycle", 9)),
            },
        },
    }


def rust_config_path_from_chan_config(chan_config) -> Optional[str]:
    """Return an explicit or generated Rust config path for a Python config."""
    if chan_config is None:
        return None
    explicit_path = getattr(chan_config, "rust_core_config_path", None)
    if explicit_path:
        return str(explicit_path)

    payload = rust_config_payload_from_chan_config(chan_config)
    text = json.dumps(payload, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
    digest = hashlib.blake2b(text.encode("utf-8"), digest_size=12).hexdigest()
    cache_dir = Path(tempfile.gettempdir()) / "chan_py_rust_core_configs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / f"{digest}.json"
    if not path.exists():
        tmp_path = cache_dir / f"{digest}.{os.getpid()}.tmp"
        tmp_path.write_text(text, encoding="utf-8")
        tmp_path.replace(path)
    return str(path)


def _to_timestamp_seconds(value: TimestampLike) -> int:
    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())

    if isinstance(value, str):
        text = value.strip()
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            dt = datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return int(dt.timestamp())

    ts = int(value)
    # Accept common millisecond timestamps while keeping second timestamps cheap.
    if abs(ts) > 10_000_000_000:
        return ts // 1000
    return ts


def freq_from_kl_type(kl_type, default: str = "1m") -> str:
    name = getattr(kl_type, "name", str(kl_type))
    return _KL_TYPE_TO_FREQ.get(name, default)


def freq_seconds(freq: str) -> int:
    return _FREQ_TO_SECONDS.get(freq, 60)


def freqs_from_lv_list(lv_list) -> list[str]:
    return [freq_from_kl_type(lv) for lv in lv_list]


def _timestamp_from_ctime(value) -> int:
    dt = datetime(
        int(value.year),
        int(value.month),
        int(value.day),
        int(value.hour),
        int(value.minute),
        int(getattr(value, "second", 0)),
        tzinfo=timezone.utc,
    )
    return int(dt.timestamp())


def _bar_from_klu(klu) -> BarTuple:
    return (
        _timestamp_from_ctime(klu.time),
        float(klu.open),
        float(klu.high),
        float(klu.low),
        float(klu.close),
        float(getattr(klu, "vol", 0.0) or 0.0),
    )


class RustBSPType:
    def __init__(self, value: str):
        self.value = value

    def __repr__(self):
        return f"RustBSPType({self.value!r})"


def _bsp_type(value: str):
    try:
        return BSP_TYPE(value)
    except Exception:
        return RustBSPType(value)


class RustKLineRef:
    def __init__(self, raw: dict):
        self.idx = int(raw.get("klu_idx", -1))
        ts_ms = raw.get("klu_time")
        self.ts = (float(ts_ms) / 1000.0) if ts_ms is not None else 0.0
        dt = datetime.fromtimestamp(self.ts, tz=timezone.utc)
        self.time = CTime(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second, auto=False)
        self.open = raw.get("open")
        self.high = raw.get("high")
        self.low = raw.get("low")
        self.close = raw.get("close")
        self.klc = type("RustKlcRef", (), {"idx": raw.get("klc_idx")})()


class RustElementRef:
    def __init__(self, raw: dict):
        self.idx = int(raw.get("element", -1))
        self.bsp = None
        self._klu = RustKLineRef(raw)
        self.is_sure = True

    def get_end_klu(self):
        return self._klu


class RustKLineIndexRef:
    def __init__(self, idx: Optional[int], value: Optional[float] = None):
        self.idx = int(idx) if idx is not None else -1
        self.open = value
        self.high = value
        self.low = value
        self.close = value
        self.klc = type("RustKlcRef", (), {"idx": None})()


def _is_up_direction(value) -> bool:
    return str(value).lower().endswith("up")


def _is_down_direction(value) -> bool:
    return str(value).lower().endswith("down")


class RustBiView:
    def __init__(self, raw: dict):
        self.raw = raw
        self.idx = int(raw.get("idx", -1))
        self.begin_klc_idx = int(raw.get("begin_klc_idx", -1))
        self.end_klc_idx = int(raw.get("end_klc_idx", -1))
        self.begin_klc = type("RustKlcRef", (), {"idx": self.begin_klc_idx})()
        self.end_klc = type("RustKlcRef", (), {"idx": self.end_klc_idx})()
        self.dir = raw.get("direction")
        self.type = raw.get("typ")
        self.is_sure = bool(raw.get("is_sure"))
        self.seg_idx = raw.get("seg_idx")
        self.parent_seg = None
        self.bsp = None
        self.pre = None
        self.next = None

    def get_begin_val(self):
        return float(self.raw.get("begin_val", 0.0))

    def get_end_val(self):
        return float(self.raw.get("end_val", 0.0))

    def get_begin_klu(self):
        return RustKLineIndexRef(self.raw.get("begin_klu_idx"), self.get_begin_val())

    def get_end_klu(self):
        return RustKLineIndexRef(self.raw.get("end_klu_idx"), self.get_end_val())

    def amp(self):
        return float(self.raw.get("amp", abs(self.get_end_val() - self.get_begin_val())))

    def get_klu_cnt(self):
        return int(self.raw.get("klu_cnt", 0))

    def get_klc_cnt(self):
        return int(self.raw.get("klc_cnt", self.end_klc_idx - self.begin_klc_idx + 1))

    def _high(self):
        return max(self.get_begin_val(), self.get_end_val())

    def _low(self):
        return min(self.get_begin_val(), self.get_end_val())

    def is_down(self):
        return _is_down_direction(self.dir)

    def is_up(self):
        return _is_up_direction(self.dir)


class RustSegView:
    def __init__(self, raw: dict, bi_list: list[RustBiView], zs_list: list["RustZsView"]):
        self.raw = raw
        self.idx = int(raw.get("idx", -1))
        self.start_element_id = int(raw.get("start_element_id", -1))
        self.end_element_id = int(raw.get("end_element_id", -1))
        self.start_bi = bi_list[self.start_element_id] if 0 <= self.start_element_id < len(bi_list) else None
        self.end_bi = bi_list[self.end_element_id] if 0 <= self.end_element_id < len(bi_list) else None
        self.is_sure = bool(raw.get("is_sure"))
        self.dir = raw.get("direction")
        self.zs_lst = [zs_list[idx] for idx in raw.get("zs_lst", []) if 0 <= idx < len(zs_list)]
        self.seg_idx = raw.get("seg_idx")
        self.parent_seg = None
        self.pre = None
        self.next = None
        self.reason = raw.get("reason")
        self.ele_inside_is_sure = bool(raw.get("ele_inside_is_sure"))

    def cal_bi_cnt(self):
        return int(self.raw.get("element_cnt", self.end_element_id - self.start_element_id + 1))

    def get_begin_val(self):
        return float(self.raw.get("begin_val", 0.0))

    def get_end_val(self):
        return float(self.raw.get("end_val", 0.0))

    def get_begin_klu(self):
        return RustKLineIndexRef(self.raw.get("begin_klu_idx"), self.get_begin_val())

    def get_end_klu(self):
        return RustKLineIndexRef(self.raw.get("end_klu_idx"), self.get_end_val())

    def amp(self):
        return float(self.raw.get("amp", abs(self.get_end_val() - self.get_begin_val())))

    def get_klu_cnt(self):
        return int(self.raw.get("klu_cnt", 0))

    def get_multi_bi_zs_cnt(self):
        return int(self.raw.get("multi_bi_zs_cnt", 0))

    def is_down(self):
        return _is_down_direction(self.dir)

    def is_up(self):
        return _is_up_direction(self.dir)


class RustZsView:
    def __init__(self, raw: dict, bi_list: list[RustBiView]):
        self.raw = raw
        self.is_sure = bool(raw.get("is_sure"))
        self.begin_element_id = int(raw.get("begin_element_id", -1))
        self.end_element_id = int(raw.get("end_element_id", -1))
        self.begin_bi = bi_list[self.begin_element_id] if 0 <= self.begin_element_id < len(bi_list) else None
        self.end_bi = bi_list[self.end_element_id] if 0 <= self.end_element_id < len(bi_list) else None
        self.low = float(raw.get("low", 0.0))
        self.high = float(raw.get("high", 0.0))
        self.mid = float(raw.get("mid", 0.0))
        self.peak_high = float(raw.get("peak_high", self.high))
        self.peak_low = float(raw.get("peak_low", self.low))
        self.bi_in = raw.get("bi_in_id")
        self.bi_out = raw.get("bi_out_id")

    def is_one_bi_zs(self):
        return self.begin_bi is not None and self.end_bi is not None and self.begin_bi.idx == self.end_bi.idx


class RustBSP:
    """Small attribute-compatible view for Rust BSP dictionaries."""

    def __init__(self, raw: dict, element: Optional[RustBiView] = None):
        self.raw = raw
        self.bi = element if element is not None else RustElementRef(raw)
        self.klu = RustKLineRef(raw)
        self.is_buy = bool(raw.get("is_buy"))
        self.type = [_bsp_type(value) for value in raw.get("types", [])]
        self.relate_bsp1 = raw.get("relate_bsp1_element")
        self.features = CFeatures(dict(raw.get("features", [])))
        self.is_segbsp = bool(raw.get("is_segbsp"))
        self.is_target = bool(raw.get("is_target"))
        if hasattr(self.bi, "bsp"):
            self.bi.bsp = self
        self._add_missing_compat_features()

    def _add_missing_compat_features(self):
        if not hasattr(self.bi, "amp"):
            return
        existing = {name for name, _ in self.features.items()}
        types = {item.value for item in self.type}
        additions = {}
        bi_amp = float(self.bi.amp())
        if "bsp_bi_amp" not in existing:
            additions["bsp_bi_amp"] = bi_amp
        parent_seg = getattr(self.bi, "parent_seg", None)
        if parent_seg is not None and "zs_cnt" not in existing and "1" in types:
            additions["zs_cnt"] = float(len(getattr(parent_seg, "zs_lst", [])))
        if "1p" in types and "bsp1_bi_amp" not in existing:
            additions["bsp1_bi_amp"] = bi_amp
        if "2" in types:
            pre_bi = getattr(self.bi, "pre", None)
            if pre_bi is not None and hasattr(pre_bi, "amp"):
                break_amp = float(pre_bi.amp())
                invalid_bi = getattr(pre_bi, "pre", None) or pre_bi
                if "bsp2_bi_amp" not in existing:
                    additions["bsp2_bi_amp"] = bi_amp
                if "bsp2_break_bi_amp" not in existing:
                    additions["bsp2_break_bi_amp"] = break_amp
                if "bsp2_break_bi_low" not in existing:
                    additions["bsp2_break_bi_low"] = float(pre_bi._low())
                if "bsp2_break_bi_high" not in existing:
                    additions["bsp2_break_bi_high"] = float(pre_bi._high())
                if "bsp2_break_bi_end_price" not in existing:
                    additions["bsp2_break_bi_end_price"] = float(pre_bi.get_end_val())
                if "bsp2_retrace_rate" not in existing:
                    additions["bsp2_retrace_rate"] = bi_amp / (break_amp + 1e-12)
                if "bsp2_invalid_price" not in existing:
                    additions["bsp2_invalid_price"] = float(invalid_bi._low() if self.bi.is_down() else invalid_bi._high())
                if "bsp2_invalid_bi_idx" not in existing:
                    additions["bsp2_invalid_bi_idx"] = float(getattr(invalid_bi, "idx", -1))
        if "2s" in types:
            bsp2_bi = None
            cursor = getattr(self.bi, "pre", None)
            while cursor is not None:
                cursor_bsp = getattr(cursor, "bsp", None)
                cursor_types = {item.value for item in getattr(cursor_bsp, "type", [])} if cursor_bsp else set()
                if "2" in cursor_types:
                    bsp2_bi = cursor
                    break
                cursor = getattr(cursor, "pre", None)
            break_bi = None
            relate_bsp1_idx = self.raw.get("relate_bsp1_element")
            level = getattr(self.bi, "level", None)
            bi_lookup = getattr(level, "_bi_lookup", {}) if level is not None else {}
            if relate_bsp1_idx is not None:
                break_bi = bi_lookup.get(int(relate_bsp1_idx) + 1)
            if break_bi is None:
                break_bi = getattr(bsp2_bi, "pre", None) if bsp2_bi is not None else getattr(self.bi, "pre", None)
            if break_bi is not None and hasattr(break_bi, "amp"):
                break_amp = float(break_bi.amp())
                if "bsp2s_bi_amp" not in existing:
                    additions["bsp2s_bi_amp"] = bi_amp
                if "bsp2s_break_bi_amp" not in existing:
                    additions["bsp2s_break_bi_amp"] = break_amp
                if "bsp2s_retrace_rate" not in existing:
                    additions["bsp2s_retrace_rate"] = abs(self.bi.get_end_val() - break_bi.get_end_val()) / (break_amp + 1e-12)
                if "bsp2s_lv" not in existing:
                    level = (self.bi.idx - bsp2_bi.idx) / 2.0 if bsp2_bi is not None else 1.0
                    additions["bsp2s_lv"] = level
                if "bsp2s_invalid_price" not in existing:
                    additions["bsp2s_invalid_price"] = float(break_bi._low() if self.bi.is_down() else break_bi._high())
                if "bsp2s_invalid_bi_idx" not in existing:
                    additions["bsp2s_invalid_bi_idx"] = float(getattr(break_bi, "idx", -1))
        if "3a" in types or "3b" in types:
            if "bsp3_bi_amp" not in existing:
                additions["bsp3_bi_amp"] = bi_amp
            if "bsp3_zs_height" not in existing:
                zs_lst = getattr(parent_seg, "zs_lst", []) if parent_seg is not None else []
                if not zs_lst:
                    level = getattr(self.bi, "level", None)
                    zs_lst = getattr(level, "zs_list", []) if level is not None else []
                if zs_lst:
                    zs = zs_lst[-1]
                    additions["bsp3_zs_height"] = (float(zs.high) - float(zs.low)) / (float(zs.low) + 1e-12)
        if additions:
            self.features.add_feat(additions)

    def type2str(self):
        return ",".join(item.value for item in self.type)

    def add_feat(self, inp1, inp2=None):
        self.features.add_feat(inp1, inp2)

    def to_dict(self):
        return dict(self.raw)

    def __getitem__(self, key):
        return self.raw[key]

    def __repr__(self):
        side = "buy" if self.is_buy else "sell"
        return f"RustBSP({side}, types={self.type2str()}, klu_idx={self.klu.idx})"


def _wrap_bsp(raw, as_dict: bool = False, element_lookup: Optional[dict[int, RustBiView]] = None):
    if raw is None or as_dict:
        return raw
    element = element_lookup.get(int(raw.get("element", -1))) if element_lookup else None
    return RustBSP(raw, element=element)


def _wrap_bsp_list(items, as_dict: bool = False, element_lookup: Optional[dict[int, RustBiView]] = None):
    if as_dict:
        return items
    wrapped = [_wrap_bsp(item, element_lookup=element_lookup) for item in items]
    for item in wrapped:
        if isinstance(item, RustBSP):
            item._add_missing_compat_features()
    return wrapped


def _wrap_bsp_event(raw, as_dict: bool = False, element_lookup: Optional[dict[int, RustBiView]] = None):
    if raw is None or as_dict:
        return raw
    item = dict(raw)
    item["bsp"] = _wrap_bsp(item.get("bsp"), element_lookup=element_lookup)
    return item


def _raw_bi_high(raw_bi: dict) -> float:
    return max(float(raw_bi.get("begin_val", 0.0)), float(raw_bi.get("end_val", 0.0)))


def _raw_bi_low(raw_bi: dict) -> float:
    return min(float(raw_bi.get("begin_val", 0.0)), float(raw_bi.get("end_val", 0.0)))


def _raw_bi_is_down(raw_bi: dict) -> bool:
    return _is_down_direction(raw_bi.get("direction"))


def _raw_feature_names(raw_bsp: dict) -> set[str]:
    return {str(name) for name, _ in raw_bsp.get("features", [])}


def _add_raw_features(raw_bsp: dict, additions: dict[str, float]) -> None:
    if not additions:
        return
    existing = _raw_feature_names(raw_bsp)
    features = list(raw_bsp.get("features", []))
    for name, value in additions.items():
        if name not in existing:
            features.append((name, float(value)))
    raw_bsp["features"] = features


def _add_bsp2_invalid_features_from_snapshot(raw_bsp: dict, snapshot: dict) -> None:
    """Add actual BSP2 structural invalidation fields without building RustLevelView."""
    if not isinstance(raw_bsp, dict):
        return
    types = {str(value) for value in raw_bsp.get("types", [])}
    if not ({"2", "2s"} & types):
        return
    bis = snapshot.get("bis", []) if isinstance(snapshot, dict) else []
    if not bis:
        return

    bi_by_idx = {}
    for raw_bi in bis:
        try:
            bi_by_idx[int(raw_bi.get("idx", -1))] = raw_bi
        except Exception:
            continue

    try:
        element_idx = int(raw_bsp.get("element", -1))
    except Exception:
        element_idx = -1
    current_bi = bi_by_idx.get(element_idx)
    if current_bi is None:
        return

    additions: dict[str, float] = {}
    if "2" in types:
        break_bi = bi_by_idx.get(element_idx - 1)
        invalid_bi = bi_by_idx.get(element_idx - 2) or break_bi
        if break_bi is not None:
            additions["bsp2_break_bi_low"] = _raw_bi_low(break_bi)
            additions["bsp2_break_bi_high"] = _raw_bi_high(break_bi)
            additions["bsp2_break_bi_end_price"] = float(break_bi.get("end_val", float("nan")))
        if invalid_bi is not None:
            additions["bsp2_invalid_price"] = _raw_bi_low(invalid_bi) if _raw_bi_is_down(current_bi) else _raw_bi_high(invalid_bi)
            additions["bsp2_invalid_bi_idx"] = float(invalid_bi.get("idx", -1))
        relate_bsp1_idx = raw_bsp.get("relate_bsp1_element")
        if relate_bsp1_idx is not None:
            try:
                origin_idx = int(relate_bsp1_idx)
            except Exception:
                origin_idx = -1
            origin_seg = None
            for raw_seg in snapshot.get("segs", []) if isinstance(snapshot, dict) else []:
                try:
                    if int(raw_seg.get("end_element_id", -1)) == origin_idx:
                        origin_seg = raw_seg
                        break
                except Exception:
                    continue
            if origin_seg is not None:
                zs_indices = list(origin_seg.get("zs_lst") or [])
                zss = snapshot.get("bzs", []) if isinstance(snapshot, dict) else []
                origin_zs = None
                for zs_idx in reversed(zs_indices):
                    try:
                        candidate = zss[int(zs_idx)]
                    except Exception:
                        continue
                    element_list = candidate.get("element_list") or []
                    if len(element_list) >= 3:
                        origin_zs = candidate
                        break
                    if origin_zs is None:
                        origin_zs = candidate
                if origin_zs is not None:
                    additions["bsp2_origin_zs_low"] = float(origin_zs.get("low", float("nan")))
                    additions["bsp2_origin_zs_mid"] = float(origin_zs.get("mid", float("nan")))
                    additions["bsp2_origin_zs_high"] = float(origin_zs.get("high", float("nan")))
                    additions["bsp2_origin_zs_peak_low"] = float(origin_zs.get("peak_low", origin_zs.get("low", float("nan"))))
                    additions["bsp2_origin_zs_peak_high"] = float(origin_zs.get("peak_high", origin_zs.get("high", float("nan"))))
                    additions["bsp2_origin_zs_begin_bi_idx"] = float(origin_zs.get("begin_element_id", float("nan")))
                    additions["bsp2_origin_zs_end_bi_idx"] = float(origin_zs.get("end_element_id", float("nan")))

    if "2s" in types:
        break_bi = None
        relate_bsp1_idx = raw_bsp.get("relate_bsp1_element")
        if relate_bsp1_idx is not None:
            try:
                break_bi = bi_by_idx.get(int(relate_bsp1_idx) + 1)
            except Exception:
                break_bi = None
        if break_bi is None:
            break_bi = bi_by_idx.get(element_idx - 1)
        if break_bi is not None:
            additions["bsp2s_invalid_price"] = _raw_bi_low(break_bi) if _raw_bi_is_down(current_bi) else _raw_bi_high(break_bi)
            additions["bsp2s_invalid_bi_idx"] = float(break_bi.get("idx", -1))

    _add_raw_features(raw_bsp, additions)


class RustKLineUnitView:
    def __init__(self, klu, idx: Optional[int] = None):
        self.idx = int(idx if idx is not None else getattr(klu, "idx", -1))
        self.time = klu.time
        self.open = float(klu.open)
        self.high = float(klu.high)
        self.low = float(klu.low)
        self.close = float(klu.close)
        self.vol = float(getattr(klu, "vol", 0.0) or 0.0)
        self.klc = None


class RustKLineCombinedView:
    def __init__(self, idx: int, klu: Optional[RustKLineUnitView], raw: Optional[dict] = None):
        self.idx = int(idx)
        self.lst = [klu] if klu is not None else []
        self.high = float(raw.get("high")) if raw and raw.get("high") is not None else (klu.high if klu else 0.0)
        self.low = float(raw.get("low")) if raw and raw.get("low") is not None else (klu.low if klu else 0.0)
        self._unit_count = int(raw.get("unit_count", len(self.lst))) if raw else len(self.lst)
        self.kl_type = None
        if klu is not None:
            try:
                klu.klc = self
            except Exception:
                pass

    def __len__(self):
        return self._unit_count

    def __getitem__(self, index):
        return self.lst[index]


class RustLevelView:
    def __init__(self, engine, freq: str, latest_klu: Optional[RustKLineUnitView]):
        self.engine = engine
        self.freq = freq
        self.latest_klu = latest_klu
        self._snapshot = engine.snapshot(freq) if isinstance(engine, RustMultiChanEngine) else engine.snapshot()
        self._counts = self._snapshot["counts"]
        self._last_klc = self._snapshot.get("last_klc")
        self.bi_list = [RustBiView(raw) for raw in self._snapshot.get("bis", [])]
        self.zs_list = [RustZsView(raw, self.bi_list) for raw in self._snapshot.get("bzs", [])]
        self.seg_list = [RustSegView(raw, self.bi_list, self.zs_list) for raw in self._snapshot.get("segs", [])]
        self._bi_lookup = {bi.idx: bi for bi in self.bi_list}
        for pos, bi in enumerate(self.bi_list):
            bi.level = self
            bi.pre = self.bi_list[pos - 1] if pos > 0 else None
            next_idx = bi.raw.get("next_idx")
            if next_idx is not None and 0 <= int(next_idx) < len(self.bi_list):
                bi.next = self.bi_list[int(next_idx)]
            elif pos + 1 < len(self.bi_list):
                bi.next = self.bi_list[pos + 1]
            parent_idx = bi.raw.get("parent_seg_idx")
            if parent_idx is None:
                parent_idx = bi.raw.get("seg_idx")
            if parent_idx is not None and 0 <= int(parent_idx) < len(self.seg_list):
                bi.parent_seg = self.seg_list[int(parent_idx)]
        for pos, seg in enumerate(self.seg_list):
            seg.pre = self.seg_list[pos - 1] if pos > 0 else None
            next_idx = seg.raw.get("next_idx")
            if next_idx is not None and 0 <= int(next_idx) < len(self.seg_list):
                seg.next = self.seg_list[int(next_idx)]
            elif pos + 1 < len(self.seg_list):
                seg.next = self.seg_list[pos + 1]

    def __len__(self):
        return int(self._counts.get("klcs", 0))

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(*index.indices(len(self)))]
        if index < 0:
            idx = len(self) + index
        else:
            idx = index
        if idx < 0 or idx >= len(self):
            raise IndexError(index)
        if self.latest_klu is None:
            raise IndexError(index)
        raw = self._last_klc if self._last_klc and int(self._last_klc.get("idx", -1)) == idx else None
        return RustKLineCombinedView(idx, self.latest_klu, raw)

    def bi_bsp(self, latest_first: bool = False, as_dict: bool = False) -> list:
        items = list(self._snapshot.get("bi_bsp", []))
        if latest_first:
            items.reverse()
        return _wrap_bsp_list(items, as_dict=as_dict, element_lookup=self._bi_lookup)

    def seg_bsp(self, latest_first: bool = False, as_dict: bool = False) -> list:
        items = list(self._snapshot.get("seg_bsp", []))
        if latest_first:
            items.reverse()
        return _wrap_bsp_list(items, as_dict=as_dict)


class RustStepSnapshot:
    def __init__(self, engine, lv_list, latest_base_klu, base_lv):
        self.rust_engine = engine
        self.lv_list = lv_list
        self.base_lv = base_lv
        self.latest_base_klu = RustKLineUnitView(latest_base_klu, getattr(latest_base_klu, "idx", -1))
        self._levels: dict[str, RustLevelView] = {}

    def _freq_for_query(self, idx=None) -> str:
        if idx is None:
            if len(self.lv_list) != 1:
                raise IndexError("multi-level Rust snapshot requires an index or KL_TYPE")
            lv = self.lv_list[0]
        elif isinstance(idx, int):
            lv = self.lv_list[idx]
        else:
            lv = idx
        return freq_from_kl_type(lv)

    def __getitem__(self, index):
        freq = self._freq_for_query(index)
        return self._level_for_freq(freq)

    def _level_for_freq(self, freq: str) -> RustLevelView:
        if freq not in self._levels:
            latest = self.latest_base_klu if freq == freq_from_kl_type(self.base_lv) else None
            self._levels[freq] = RustLevelView(self.rust_engine, freq, latest)
        return self._levels[freq]

    def _level_for_query(self, idx=None) -> RustLevelView:
        freq = self._freq_for_query(idx)
        return self._level_for_freq(freq)

    def __len__(self):
        return len(self.lv_list)

    def get_rust_snapshot(self, idx=None):
        return self._level_for_query(idx)._snapshot

    def get_rust_seg_bsp(self, idx=None, latest_first=True):
        return self._level_for_query(idx).seg_bsp(latest_first=latest_first)

    def get_bsp(self, idx=None):
        return self._level_for_query(idx).bi_bsp(latest_first=False)

    def get_latest_bsp(self, idx=None, number=1):
        items = self._level_for_query(idx).bi_bsp(latest_first=True)
        return items if number == 0 else items[:number]

    def get_latest_bsp_fast(self, idx=None, number=1, as_dict: bool = False):
        if not isinstance(self.rust_engine, RustMultiChanEngine) and number == 1:
            item = self.rust_engine.latest_bi_bsp(as_dict=as_dict)
            return [] if item is None else [item]
        if isinstance(self.rust_engine, RustMultiChanEngine):
            items = self.rust_engine.bi_bsp(self._freq_for_query(idx), latest_first=True, as_dict=as_dict)
        else:
            items = self.rust_engine.bi_bsp(latest_first=True, as_dict=as_dict)
        return items if number == 0 else items[:number]

    def get_latest_bsp_event_fast(self, idx=None, as_dict: bool = False):
        if isinstance(self.rust_engine, RustMultiChanEngine):
            return self.rust_engine.latest_bi_bsp_event(self._freq_for_query(idx), as_dict=as_dict)
        return self.rust_engine.latest_bi_bsp_event(as_dict=as_dict)

    def enrich_bsp(self, bsp, idx=None):
        raw = bsp if isinstance(bsp, dict) else getattr(bsp, "raw", bsp)
        if not isinstance(raw, dict):
            return bsp
        freq = self._freq_for_query(idx)
        snapshot = self.rust_engine.snapshot(freq) if isinstance(self.rust_engine, RustMultiChanEngine) else self.rust_engine.snapshot()
        _add_bsp2_invalid_features_from_snapshot(raw, snapshot)
        return _wrap_bsp(raw)

    def get_rust_counts(self, idx=None):
        if isinstance(self.rust_engine, RustMultiChanEngine):
            if idx is None and len(self.lv_list) != 1:
                return self.rust_engine.counts()
            return self.rust_engine.counts(self._freq_for_query(idx))
        return self.rust_engine.counts()

    def get_last_klc_idx_fast(self, idx=None):
        if isinstance(self.rust_engine, RustMultiChanEngine):
            return self.rust_engine.last_klc_idx(self._freq_for_query(idx))
        return self.rust_engine.last_klc_idx()


class RustChanEngine:
    """Small compatibility wrapper around ``chan_core_py.ChanEngine``."""

    def __init__(self, freq: str = "1m", config_path: Optional[Union[str, Path]] = None):
        ext = _require_rust_core()
        self.freq = freq
        self.config_path = str(config_path) if config_path is not None else None
        self._engine = ext.ChanEngine(freq, self.config_path)

    def push_bar(
        self,
        dt: TimestampLike,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: float = 0.0,
    ) -> None:
        self._engine.push_bar_ts(
            _to_timestamp_seconds(dt),
            float(open),
            float(high),
            float(low),
            float(close),
            float(volume),
        )

    def push_bars(self, bars: Iterable[Sequence[Union[TimestampLike, float]]]) -> int:
        rust_bars = []
        for row in bars:
            if len(row) == 5:
                dt, open_, high, low, close = row
                volume = 0.0
            elif len(row) == 6:
                dt, open_, high, low, close, volume = row
            else:
                raise ValueError("each bar must be (dt, open, high, low, close[, volume])")
            rust_bars.append(
                (
                    _to_timestamp_seconds(dt),
                    float(open_),
                    float(high),
                    float(low),
                    float(close),
                    float(volume),
                )
            )
        return self._engine.push_bars_ts(rust_bars)

    def push_klu(self, klu) -> None:
        self.push_bar(*_bar_from_klu(klu))

    def push_klu_step(self, klu) -> None:
        self.push_klu(klu)
        self.cal_seg_and_zs()

    def push_klus(self, klus: Iterable) -> int:
        return self.push_bars(_bar_from_klu(klu) for klu in klus)

    def push_klus_chunked(self, klus: Iterable, chunk_size: int = 10000) -> int:
        total = 0
        chunk = []
        for klu in klus:
            chunk.append(_bar_from_klu(klu))
            if len(chunk) >= chunk_size:
                total += self.push_bars(chunk)
                chunk.clear()
        if chunk:
            total += self.push_bars(chunk)
        return total

    def cal_seg_and_zs(self) -> None:
        self._engine.cal_seg_and_zs()

    def counts(self) -> dict:
        return json.loads(self._engine.counts_json())

    def last_klc_idx(self):
        return self._engine.last_klc_idx()

    def bi_bsp(self, latest_first: bool = False, as_dict: bool = False) -> list:
        return _wrap_bsp_list(json.loads(self._engine.bi_bsp_json(latest_first)), as_dict=as_dict)

    def seg_bsp(self, latest_first: bool = False, as_dict: bool = False) -> list:
        return _wrap_bsp_list(json.loads(self._engine.seg_bsp_json(latest_first)), as_dict=as_dict)

    def latest_bi_bsp(self, as_dict: bool = False):
        return _wrap_bsp(json.loads(self._engine.latest_bi_bsp_json()), as_dict=as_dict)

    def latest_bi_bsp_event(self, as_dict: bool = False):
        return _wrap_bsp_event(json.loads(self._engine.latest_bi_bsp_event_json()), as_dict=as_dict)

    def enrich_bsp(self, bsp):
        raw = bsp if isinstance(bsp, dict) else getattr(bsp, "raw", bsp)
        if not isinstance(raw, dict):
            return bsp
        _add_bsp2_invalid_features_from_snapshot(raw, self.snapshot())
        return _wrap_bsp(raw)

    def latest_seg_bsp(self, as_dict: bool = False):
        return _wrap_bsp(json.loads(self._engine.latest_seg_bsp_json()), as_dict=as_dict)

    def snapshot(self) -> dict:
        return json.loads(self._engine.snapshot_json())

    def tail_snapshot(self) -> dict:
        return json.loads(self._engine.tail_snapshot_json())

    @classmethod
    def from_klus(
        cls,
        klus: Iterable,
        freq: Optional[str] = None,
        config_path: Optional[Union[str, Path]] = None,
    ) -> "RustChanEngine":
        iterator = iter(klus)
        buffered = []
        try:
            first = next(iterator)
        except StopIteration:
            engine = cls(freq or "1m", config_path=config_path)
            return engine

        buffered.append(first)
        inferred_freq = freq or freq_from_kl_type(getattr(first, "kl_type", None))
        engine = cls(inferred_freq, config_path=config_path)
        engine.push_klus(buffered)
        engine.push_klus(iterator)
        return engine

    @property
    def raw(self):
        return self._engine


class RustMultiChanEngine:
    """Python adapter around Rust ``CMultiChan``."""

    def __init__(
        self,
        freqs: Union[str, Iterable[str]],
        base_freq: Optional[str] = None,
        config_path: Optional[Union[str, Path]] = None,
        auto_snapshot: bool = True,
        timeline_stride: Optional[int] = None,
    ):
        ext = _require_rust_core()
        if isinstance(freqs, str):
            freq_text = freqs
        else:
            freq_text = ",".join(freqs)
        self.freqs = [item.strip() for item in freq_text.split(",") if item.strip()]
        self.base_freq = base_freq or min(self.freqs, key=freq_seconds)
        self.config_path = str(config_path) if config_path is not None else None
        self._engine = ext.MultiChanEngine(
            freq_text,
            self.base_freq,
            self.config_path,
            bool(auto_snapshot),
            timeline_stride,
        )
        self._dirty = False

    def push_bar(
        self,
        dt: TimestampLike,
        open: float,
        high: float,
        low: float,
        close: float,
        volume: float = 0.0,
    ) -> None:
        self._engine.push_bar_ts(
            _to_timestamp_seconds(dt),
            float(open),
            float(high),
            float(low),
            float(close),
            float(volume),
        )
        self._dirty = True

    def push_bars(self, bars: Iterable[Sequence[Union[TimestampLike, float]]]) -> int:
        rust_bars = []
        for row in bars:
            if len(row) == 5:
                dt, open_, high, low, close = row
                volume = 0.0
            elif len(row) == 6:
                dt, open_, high, low, close, volume = row
            else:
                raise ValueError("each bar must be (dt, open, high, low, close[, volume])")
            rust_bars.append(
                (
                    _to_timestamp_seconds(dt),
                    float(open_),
                    float(high),
                    float(low),
                    float(close),
                    float(volume),
                )
            )
        pushed = self._engine.push_bars_ts(rust_bars)
        self._dirty = self._dirty or pushed > 0
        return pushed

    def push_klus_chunked(self, klus: Iterable, chunk_size: int = 10000) -> int:
        total = 0
        chunk = []
        for klu in klus:
            chunk.append(_bar_from_klu(klu))
            if len(chunk) >= chunk_size:
                total += self.push_bars(chunk)
                chunk.clear()
        if chunk:
            total += self.push_bars(chunk)
        return total

    def push_klu_step(self, klu) -> None:
        self.push_bar(*_bar_from_klu(klu))
        self.finalize()

    def finalize(self) -> None:
        if self._dirty:
            self._engine.finalize()
            self._dirty = False

    def _ensure_finalized(self) -> None:
        self.finalize()

    def available_freqs(self) -> list[str]:
        return json.loads(self._engine.freqs_json())

    def counts(self, freq: Optional[str] = None):
        self._ensure_finalized()
        return json.loads(self._engine.counts_json(freq))

    def last_klc_idx(self, freq: str):
        self._ensure_finalized()
        return self._engine.last_klc_idx(freq)

    def bi_bsp(self, freq: str, latest_first: bool = False, as_dict: bool = False) -> list:
        self._ensure_finalized()
        return _wrap_bsp_list(
            json.loads(self._engine.bi_bsp_json(freq, latest_first)),
            as_dict=as_dict,
        )

    def seg_bsp(self, freq: str, latest_first: bool = False, as_dict: bool = False) -> list:
        self._ensure_finalized()
        return _wrap_bsp_list(
            json.loads(self._engine.seg_bsp_json(freq, latest_first)),
            as_dict=as_dict,
        )

    def latest_bi_bsp_event(self, freq: str, as_dict: bool = False):
        self._ensure_finalized()
        return _wrap_bsp_event(json.loads(self._engine.latest_bi_bsp_event_json(freq)), as_dict=as_dict)

    def snapshot(self, freq: str) -> dict:
        self._ensure_finalized()
        return json.loads(self._engine.snapshot_json(freq))

    def snapshots(self) -> list[dict]:
        self._ensure_finalized()
        return json.loads(self._engine.snapshots_json())

    @property
    def raw(self):
        return self._engine
