from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import xgboost as xgb

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE


CHAN_CONFIG = {
    "trigger_step": True,
    "bi_strict": True,
    "skip_step": 0,
    "divergence_rate": float("inf"),
    "bsp2_follow_1": False,
    "bsp3_follow_1": False,
    "min_zs_cnt": 0,
    "bs1_peak": False,
    "macd_algo": "peak",
    "bs_type": "1,2,3a,1p,2s,3b",
    "print_warning": False,
    "zs_algo": "normal",
}


@dataclass
class Sample:
    symbol: str
    bsp_time: str
    trade_price: float
    is_buy: bool
    bsp_type: str
    bsp_types_str: str
    feature_map: Dict[str, float]


def _kl_type_from_text(text: str) -> KL_TYPE:
    mapping = {
        "1m": KL_TYPE.K_1M,
        "3m": KL_TYPE.K_3M,
        "5m": KL_TYPE.K_5M,
        "10m": KL_TYPE.K_10M,
        "15m": KL_TYPE.K_15M,
        "30m": KL_TYPE.K_30M,
        "1h": KL_TYPE.K_60M,
        "1d": KL_TYPE.K_DAY,
    }
    if text not in mapping:
        raise ValueError(f"Unsupported kl-type: {text}")
    return mapping[text]


def _safe_float(v: object) -> float:
    try:
        fv = float(v)
        if np.isfinite(fv):
            return fv
    except Exception:
        pass
    return 0.0


def discover_latest_train_dir(project_root: Path) -> Path:
    runs_dir = project_root / "Debug" / "runs"
    cands = list(runs_dir.glob("*/train/model_buy.json"))
    if not cands:
        raise FileNotFoundError("No train artifact found under Debug/runs/*/train/model_buy.json")
    cands.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return cands[0].parent


def load_artifacts(
    train_dir: Path,
    side: str,
) -> Tuple[xgb.Booster, Dict[str, int], List[str], object]:
    if side not in {"buy", "sell"}:
        raise ValueError("side must be buy or sell")

    model_path = train_dir / f"model_{side}.json"
    meta_path = train_dir / f"meta_{side}.json"
    meta_model_path = train_dir / "meta_model.pkl"

    if not model_path.exists():
        raise FileNotFoundError(str(model_path))
    if not meta_path.exists():
        raise FileNotFoundError(str(meta_path))

    booster = xgb.Booster()
    booster.load_model(str(model_path))

    with open(meta_path, "r", encoding="utf-8") as f:
        meta: Dict[str, int] = json.load(f)

    fnames = [""] * len(meta)
    for name, idx in meta.items():
        fnames[idx] = name

    meta_model = None
    if meta_model_path.exists():
        with open(meta_model_path, "rb") as f:
            meta_model = pickle.load(f)

    return booster, meta, fnames, meta_model


def sample_from_json(path: Path, expected_is_buy: bool) -> Sample:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    feature_map = {
        str(k): _safe_float(v)
        for k, v in (raw.get("feature_map") or {}).items()
    }

    return Sample(
        symbol=str(raw.get("symbol") or "UNKNOWN"),
        bsp_time=str(raw.get("bsp_time") or "UNKNOWN"),
        trade_price=_safe_float(raw.get("trade_price", 0.0)),
        is_buy=bool(raw.get("is_buy", expected_is_buy)),
        bsp_type=str(raw.get("bsp_type") or "1"),
        bsp_types_str=str(raw.get("bsp_types_str") or ""),
        feature_map=feature_map,
    )


def sample_from_latest_bsp(
    code: str,
    begin_time: str,
    end_time: str,
    kl_type_text: str,
    expected_is_buy: bool,
) -> Sample:
    chan = CChan(
        code=code,
        begin_time=begin_time,
        end_time=end_time,
        data_src=DATA_SRC.PARQUET,
        lv_list=[_kl_type_from_text(kl_type_text)],
        config=CChanConfig(CHAN_CONFIG),
        autype=AUTYPE.NONE,
    )

    last = None
    seen = set()
    for snapshot in chan.step_load():
        bsp_list = snapshot.get_latest_bsp()
        if not bsp_list:
            continue

        bsp = bsp_list[0]
        cur_lv = snapshot[0]
        if len(cur_lv) < 2 or cur_lv[-2].idx != bsp.klu.klc.idx:
            continue
        if bool(bsp.is_buy) != bool(expected_is_buy):
            continue
        if bsp.klu.idx in seen:
            continue
        seen.add(bsp.klu.idx)

        fmap = {str(k): _safe_float(v) for k, v in bsp.features.items()}
        last = Sample(
            symbol=code,
            bsp_time=bsp.klu.time.to_str(),
            trade_price=_safe_float(bsp.klu.close),
            is_buy=bool(bsp.is_buy),
            bsp_type=str(bsp.type[0].value[0]),
            bsp_types_str=str(bsp.type2str()),
            feature_map=fmap,
        )

    if last is None:
        side_text = "buy" if expected_is_buy else "sell"
        raise RuntimeError(
            f"No {side_text} BSP found for {code} in [{begin_time}, {end_time}]"
        )
    return last


def build_feature_vector(
    sample: Sample,
    meta: Dict[str, int],
) -> np.ndarray:
    arr = np.zeros(len(meta), dtype=np.float32)

    if "bsp_type_1" in meta:
        arr[meta["bsp_type_1"]] = 1.0 if sample.bsp_type == "1" else 0.0
    if "bsp_type_2" in meta:
        arr[meta["bsp_type_2"]] = 1.0 if sample.bsp_type == "2" else 0.0
    if "bsp_type_3" in meta:
        arr[meta["bsp_type_3"]] = 1.0 if sample.bsp_type == "3" else 0.0
    if "is_buy_signal" in meta:
        arr[meta["is_buy_signal"]] = 1.0 if bool(sample.is_buy) else 0.0

    if "bsp_type_1b" in meta:
        arr[meta["bsp_type_1b"]] = 1.0 if sample.is_buy and sample.bsp_type == "1" else 0.0
    if "bsp_type_2b" in meta:
        arr[meta["bsp_type_2b"]] = 1.0 if sample.is_buy and sample.bsp_type == "2" else 0.0
    if "bsp_type_3b" in meta:
        arr[meta["bsp_type_3b"]] = 1.0 if sample.is_buy and sample.bsp_type == "3" else 0.0
    if "bsp_type_1s" in meta:
        arr[meta["bsp_type_1s"]] = 1.0 if (not sample.is_buy) and sample.bsp_type == "1" else 0.0
    if "bsp_type_2s" in meta:
        arr[meta["bsp_type_2s"]] = 1.0 if (not sample.is_buy) and sample.bsp_type == "2" else 0.0
    if "bsp_type_3s" in meta:
        arr[meta["bsp_type_3s"]] = 1.0 if (not sample.is_buy) and sample.bsp_type == "3" else 0.0

    for name, val in sample.feature_map.items():
        idx = meta.get(name)
        if idx is not None:
            arr[idx] = _safe_float(val)

    if "dist_to_zs_center" in meta and "distance_to_zhongshu_center" in sample.feature_map:
        arr[meta["dist_to_zs_center"]] = _safe_float(sample.feature_map["distance_to_zhongshu_center"])
    if "bi_count_in_zs" in meta and "zs_bi_count" in sample.feature_map:
        arr[meta["bi_count_in_zs"]] = _safe_float(sample.feature_map["zs_bi_count"])

    return arr


def _normalize_primary_probs(pred: np.ndarray) -> np.ndarray:
    arr = np.asarray(pred)
    if arr.ndim == 1:
        arr = np.stack([1.0 - arr, np.zeros_like(arr), arr], axis=1)
    elif arr.ndim == 2 and arr.shape[1] == 2:
        arr = np.stack(
            [arr[:, 0], np.zeros(arr.shape[0], dtype=arr.dtype), arr[:, 1]],
            axis=1,
        )
    return arr


def analyze_one_sample(
    booster: xgb.Booster,
    meta_model: object,
    fnames: List[str],
    X: np.ndarray,
    threshold: float,
) -> Dict[str, object]:
    dtest = xgb.DMatrix(X.reshape(1, -1), feature_names=fnames, missing=np.nan)
    primary_probs = _normalize_primary_probs(booster.predict(dtest))[0]

    cls = int(np.argmax(primary_probs))
    direction = 0
    if cls == 0:
        direction = -1
    elif cls == 2:
        direction = 1

    primary_conf = float(np.max(primary_probs[[0, 2]]))

    if direction == 0:
        meta_prob = 0.0
    elif meta_model is None:
        meta_prob = primary_conf
    else:
        p_block = primary_probs.reshape(1, -1)
        base_meta = getattr(meta_model, "model", meta_model)
        expected_total = getattr(base_meta, "n_features_in_", None)
        if isinstance(expected_total, (int, np.integer)):
            expect_primary_cols = int(expected_total) - int(X.shape[0])
            if expect_primary_cols == 2 and p_block.shape[1] >= 3:
                p_block = p_block[:, [0, 2]]
            elif expect_primary_cols == 1:
                p_block = np.max(
                    p_block[:, [0, 2]] if p_block.shape[1] >= 3 else p_block,
                    axis=1,
                    keepdims=True,
                )

        meta_input = np.hstack([np.nan_to_num(X, nan=0.0).reshape(1, -1), p_block])
        p_meta = np.asarray(meta_model.predict_proba(meta_input), dtype=np.float32)
        meta_prob = float(p_meta[0, -1])

    qualified = bool(meta_prob >= float(threshold) and direction != 0)

    contrib_raw = np.asarray(booster.predict(dtest, pred_contribs=True))
    if contrib_raw.ndim == 3:
        # Multiclass: [n_samples, n_classes, n_features+1], pick predicted class.
        contrib = contrib_raw[0, cls, :]
    else:
        contrib = contrib_raw[0]
    bias = float(contrib[-1])
    feats = []
    for i, name in enumerate(fnames):
        feats.append(
            {
                "name": name,
                "value": float(X[i]),
                "contrib": float(contrib[i]),
                "abs_contrib": float(abs(contrib[i])),
            }
        )
    feats.sort(key=lambda item: item["abs_contrib"], reverse=True)

    return {
        "primary_probs": {
            "class_0_sl": float(primary_probs[0]),
            "class_1_timeout": float(primary_probs[1]),
            "class_2_pt": float(primary_probs[2]),
        },
        "primary_class": cls,
        "direction": direction,
        "primary_conf": primary_conf,
        "meta_prob": meta_prob,
        "threshold": float(threshold),
        "qualified": qualified,
        "signal": int(direction if qualified else 0),
        "bias": bias,
        "top_features": feats[:20],
    }


def _print_result(sample: Sample, side: str, result: Dict[str, object], missing_names: List[str]):
    probs = result["primary_probs"]
    print("=" * 72)
    print(f"MODEL SIDE: {side}")
    print(f"SYMBOL    : {sample.symbol}")
    print(f"BSP TIME  : {sample.bsp_time}")
    print(f"BSP TYPE  : {sample.bsp_type} ({sample.bsp_types_str})")
    print(f"PRICE     : {sample.trade_price:.6f}")
    print("-" * 72)
    print(
        "PRIMARY   : "
        f"SL={probs['class_0_sl']:.4f}, "
        f"TIMEOUT={probs['class_1_timeout']:.4f}, "
        f"PT={probs['class_2_pt']:.4f}"
    )
    print(f"CLASS     : {result['primary_class']} (direction={result['direction']})")
    print(f"META PROB : {result['meta_prob']:.4f} (threshold={result['threshold']:.4f})")
    print(f"QUALIFIED : {result['qualified']}")
    print(f"SIGNAL    : {result['signal']}")
    print("-" * 72)
    print(f"MISSING FEATURE COUNT: {len(missing_names)}")
    if missing_names:
        preview = ", ".join(missing_names[:15])
        print(f"MISSING TOP 15: {preview}")
    print("-" * 72)
    print("TOP FEATURE CONTRIBUTIONS")
    for i, item in enumerate(result["top_features"], start=1):
        print(
            f"{i:>2}. {item['name']:<36} "
            f"value={item['value']:>10.5f} contrib={item['contrib']:>+10.5f}"
        )
    print("=" * 72)


def run_side_cli(side: str):
    parser = argparse.ArgumentParser(
        description=f"Single-sample decision analyzer for {side} model"
    )
    parser.add_argument("--train-dir", default="", help="Path like Debug/runs/<run_id>/train")
    parser.add_argument("--feature-json", default="", help="JSON input with feature_map")
    parser.add_argument("--code", default="BTCUSDT")
    parser.add_argument("--begin-time", default="2025-01-01")
    parser.add_argument("--end-time", default="2026-02-28")
    parser.add_argument("--kl-type", default="15m")
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument("--output-json", default="")
    ns = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    train_dir = Path(ns.train_dir) if ns.train_dir else discover_latest_train_dir(root)
    train_dir = train_dir.resolve()

    booster, meta, fnames, meta_model = load_artifacts(train_dir, side)

    expect_buy = side == "buy"
    if ns.feature_json:
        sample = sample_from_json(Path(ns.feature_json), expected_is_buy=expect_buy)
    else:
        sample = sample_from_latest_bsp(
            code=ns.code,
            begin_time=ns.begin_time,
            end_time=ns.end_time,
            kl_type_text=ns.kl_type,
            expected_is_buy=expect_buy,
        )

    X = build_feature_vector(sample, meta)
    missing = [name for name in fnames if name and (name not in sample.feature_map)]
    result = analyze_one_sample(
        booster=booster,
        meta_model=meta_model,
        fnames=fnames,
        X=X,
        threshold=float(ns.threshold),
    )

    _print_result(sample, side, result, missing)

    payload = {
        "side": side,
        "train_dir": str(train_dir),
        "sample": {
            "symbol": sample.symbol,
            "bsp_time": sample.bsp_time,
            "trade_price": sample.trade_price,
            "is_buy": sample.is_buy,
            "bsp_type": sample.bsp_type,
            "bsp_types_str": sample.bsp_types_str,
        },
        "result": result,
        "missing_features": missing,
    }

    if ns.output_json:
        out = Path(ns.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"JSON report written: {out}")
