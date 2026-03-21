"""
XGBoost + SHAP 多币种训练流程 — 双模型 (买点/卖点质量评估)
============================================================
核心变化:
  1. 数据源切换为 DataAPI/parquetAPI.py，直接读取 data/*.parquet
  2. 多币种训练，按币种维度 4:1 划分训练币种/保留币种
  3. 每个币种内部按时间 4:1 划分，训练集只使用训练币种前 80% 时间
  4. 测试集包含:
       - 训练币种后 20% 时间 (时间外推)
       - 保留币种全部时间窗口 (跨币种泛化)
  5. 多进程并行收集每个币种的 BSP 特征
  6. 训练模式可选 CPU / GPU / AUTO

输出 (每个方向各一套):
  - Debug/model_buy.json / model_sell.json
  - Debug/meta_buy.json / meta_sell.json
  - Debug/shap_report_buy.html / shap_report_sell.html
  - Debug/metrics_buy.json / metrics_sell.json
  - Debug/filter_log_buy.json / filter_log_sell.json
============================================================
"""

# flake8: noqa: E501, E402

import argparse
import json
import multiprocessing as mp
import os
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from Chan import CChan
from ChanConfig import CChanConfig
from ChanModel.feature_center import build_features
from ChanModel.shap_analyzer import SHAPAnalyzer
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE


BEGIN_TIME = "2020-01-01"
END_TIME = "2026-03-16"
DATA_SRC_TYPE = DATA_SRC.PARQUET
TRAIN_KL_TYPE_TEXT = "15m"
TRAIN_RATIO = 0.8
SYMBOL_TRAIN_RATIO = 0.8
QUALIFIED_THRESHOLD = 0.02
TRAIN_MODE = "auto"
NUM_WORKERS = max(1, (os.cpu_count() or 2) - 1)
SHAP_SAMPLE_LIMIT = 5000

PARQUET_REQUIRED_COLUMNS = ["open_time", "open", "high", "low", "close", "volume"]

XGB_BASE_PARAMS = {
    "max_depth": 3,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.7,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "min_child_weight": 10,
    "gamma": 0.3,
    "reg_alpha": 0.5,
    "reg_lambda": 2.0,
    "seed": 42,
}
NUM_BOOST_ROUND = 300
EARLY_STOPPING_ROUNDS = 30

CORR_THRESHOLD = 0.85
HIGH_NAN_THRESHOLD = 0.90
LOW_INFO_UNIQUE_THRESHOLD = 1
LOW_INFO_DOMINANT_RATIO = 0.995

FEATURE_PRUNE_MODE = "aggressive"
MODEL_PRUNE_TOPK = 40
MODEL_PRUNE_MIN_GAIN = 0.0
MODEL_PRUNE_ROUNDS = 120

CORE_FEATURE_ALLOWLIST = {
    "momentum_5",
    "price_acceleration",
    "price_pos_20",
    "price_ma20_dist",
    "price_ma60_dist",
    "ma20_ma60_cross",
    "ma_trend_aligned",
    "vol_ratio_20",
    "vol_change",
    "vol_amount_std_20",
    "upper_shadow_ratio",
    "lower_shadow_ratio",
    "bar_range",
    "bar_body_position",
    "candle_strength",
    "rsi_14",
    "macd_hist",
    "kdj_k",
    "boll_bandwidth",
    "atr_expanding",
    "cci_value",
    "bi_length",
    "bi_return_ratio",
    "bi_length_ratio",
    "bi_macd_area",
    "distance_to_zhongshu_center",
    "zs_bi_count",
    "zs_peak_range",
    "seg_direction",
    "seg_bi_count",
}

EXTRA_ALWAYS_KEEP_PREFIXES = (
    "bsp",
    "zs_",
    "seg_",
    "bi_",
    "divergence_",
)

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

OUTPUT_DIR = "Debug"


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()

    def isatty(self):
        return any(
            getattr(stream, "isatty", lambda: False)()
            for stream in self.streams
        )


def setup_train_log():
    project_root = Path(__file__).resolve().parents[1]
    result_dir = project_root / "result"
    result_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = result_dir / f"xgb_train_{timestamp}.log"
    log_fp = open(log_path, "w", encoding="utf-8")

    sys.stdout = TeeStream(sys.__stdout__, log_fp)
    sys.stderr = TeeStream(sys.__stderr__, log_fp)
    print(f"[LOG] 训练输出将保存到: {log_path}")
    return log_fp, str(log_path)


def close_train_log(log_fp, log_path: str):
    if log_fp is None:
        return

    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        log_fp.close()

    if log_path:
        print(f"[LOG] 训练日志已保存: {log_path}")


@dataclass
class TrainArgs:
    symbols: List[str]
    begin_time: str
    end_time: str
    kl_type: KL_TYPE
    train_ratio: float
    symbol_train_ratio: float
    qualified_threshold: float
    train_mode: str
    num_workers: int
    shap_sample_limit: int
    output_dir: str
    feature_prune_mode: str
    model_prune_topk: int


def parse_args() -> TrainArgs:
    parser = argparse.ArgumentParser(description="多币种 XGBoost + SHAP 训练")
    parser.add_argument("--symbols", nargs="*", help="币种列表，如 BTC ETH BTCUSDT")
    parser.add_argument("--begin-time", default=BEGIN_TIME)
    parser.add_argument("--end-time", default=END_TIME)
    parser.add_argument(
        "--kl-type",
        default=TRAIN_KL_TYPE_TEXT,
        choices=[TRAIN_KL_TYPE_TEXT],
        help="训练周期固定为15m（由5m K线自动合并）",
    )
    parser.add_argument("--train-ratio", type=float, default=TRAIN_RATIO)
    parser.add_argument(
        "--symbol-train-ratio",
        type=float,
        default=SYMBOL_TRAIN_RATIO,
    )
    parser.add_argument(
        "--qualified-threshold",
        type=float,
        default=QUALIFIED_THRESHOLD,
    )
    parser.add_argument(
        "--train-mode",
        choices=["cpu", "gpu", "auto"],
        default=TRAIN_MODE,
    )
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument(
        "--shap-sample-limit",
        type=int,
        default=SHAP_SAMPLE_LIMIT,
    )
    parser.add_argument(
        "--feature-prune-mode",
        choices=["off", "basic", "aggressive"],
        default=FEATURE_PRUNE_MODE,
        help="off=仅基础校验, basic=缺失+低信息+相关性, aggressive=额外按模型gain裁剪",
    )
    parser.add_argument(
        "--model-prune-topk",
        type=int,
        default=MODEL_PRUNE_TOPK,
        help="aggressive模式下按gain保留的最大特征数",
    )
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    ns = parser.parse_args()

    symbols = normalize_symbols(ns.symbols) if ns.symbols else discover_symbols("5m")
    ensure_15m_parquet_from_5m(symbols)
    if len(symbols) < 2:
        raise ValueError("至少需要 2 个币种，才能执行 4:1 币种切分")

    return TrainArgs(
        symbols=symbols,
        begin_time=ns.begin_time,
        end_time=ns.end_time,
        kl_type=kl_type_from_text(ns.kl_type),
        train_ratio=ns.train_ratio,
        symbol_train_ratio=ns.symbol_train_ratio,
        qualified_threshold=ns.qualified_threshold,
        train_mode=ns.train_mode,
        num_workers=max(1, ns.num_workers),
        shap_sample_limit=max(100, ns.shap_sample_limit),
        output_dir=ns.output_dir,
        feature_prune_mode=ns.feature_prune_mode,
        model_prune_topk=max(10, ns.model_prune_topk),
    )


def kl_type_from_text(text: str) -> KL_TYPE:
    mapping = {
        "1m": KL_TYPE.K_1M,
        "3m": KL_TYPE.K_3M,
        "5m": KL_TYPE.K_5M,
        "10m": KL_TYPE.K_10M,
        "15m": KL_TYPE.K_15M,
        "30m": KL_TYPE.K_30M,
        "1h": KL_TYPE.K_60M,
        "1d": KL_TYPE.K_DAY,
        "1w": KL_TYPE.K_WEEK,
        "1mo": KL_TYPE.K_MON,
    }
    return mapping[text]


def discover_symbols(interval_text: str) -> List[str]:
    data_dir = Path(__file__).resolve().parents[1] / "data"
    symbols = []
    for path in sorted(data_dir.glob(f"*_{interval_text}.parquet")):
        base = path.stem.rsplit("_", 1)[0]
        symbols.append(f"{base.upper()}USDT")
    if not symbols and interval_text != "5m":
        for path in sorted(data_dir.glob("*_5m.parquet")):
            base = path.stem.rsplit("_", 1)[0]
            symbols.append(f"{base.upper()}USDT")
    if not symbols:
        raise FileNotFoundError(
            f"未在 {data_dir} 找到 *_{interval_text}.parquet 文件"
        )
    return symbols


def ensure_15m_parquet_from_5m(symbols: Sequence[str]):
    """
    将 data/*_5m.parquet 合并为 data/*_15m.parquet。

    合并规则采用业界常用 OHLCV 聚合：
      - open: first
      - high: max
      - low : min
      - close: last
      - volume: sum
    时间桶按 UTC 15 分钟左闭左标（00/15/30/45）对齐。
    """
    data_dir = Path(__file__).resolve().parents[1] / "data"
    generated = 0
    reused = 0

    for symbol in symbols:
        base = symbol.upper().replace("USDT", "")
        source_5m = data_dir / f"{base}_5m.parquet"
        target_15m = data_dir / f"{base}_{TRAIN_KL_TYPE_TEXT}.parquet"

        if target_15m.exists():
            reused += 1
            continue
        if not source_5m.exists():
            raise FileNotFoundError(
                f"未找到5m源文件，无法合并15m: {source_5m}"
            )

        df = pd.read_parquet(source_5m, columns=PARQUET_REQUIRED_COLUMNS)
        missing = [
            col for col in PARQUET_REQUIRED_COLUMNS
            if col not in df.columns
        ]
        if missing:
            raise ValueError(f"{source_5m} 缺少字段: {missing}")

        df = df[PARQUET_REQUIRED_COLUMNS].copy()
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        for col in PARQUET_REQUIRED_COLUMNS[1:]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = (
            df.dropna(subset=["open_time", "open", "high", "low", "close"])
            .drop_duplicates(subset=["open_time"])
            .sort_values("open_time")
            .reset_index(drop=True)
        )

        agg = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
        df_15m = (
            df.set_index("open_time")
            .resample("15min", label="left", closed="left", origin="epoch")
            .agg(agg)
            .dropna(subset=["open", "high", "low", "close"])
            .reset_index()
        )
        df_15m["open_time"] = (
            df_15m["open_time"].astype("int64") // 1_000_000
        ).astype("int64")
        df_15m.to_parquet(target_15m, index=False)
        generated += 1
        print(f"  [K线合并] 5m -> 15m: {source_5m.name} -> {target_15m.name}")

    print(
        f"[K线合并] 15m文件准备完成: 新生成={generated}, 已存在复用={reused}"
    )


def normalize_symbols(symbols: Sequence[str]) -> List[str]:
    result = []
    for symbol in symbols:
        item = symbol.upper()
        if not item.endswith("USDT"):
            item = f"{item}USDT"
        result.append(item)
    return sorted(dict.fromkeys(result))


def parse_user_time(text: str, is_end: bool = False) -> datetime:
    normalized = text.replace("/", "-").strip()
    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(normalized, fmt)
            if fmt == "%Y-%m-%d" and is_end:
                return dt.replace(hour=23, minute=59, second=59)
            return dt
        except ValueError:
            continue
    raise ValueError(f"无法解析时间: {text}")


def compute_split_ts(begin_time: str, end_time: str, ratio: float) -> float:
    begin_dt = parse_user_time(begin_time)
    end_dt = parse_user_time(end_time, is_end=True)
    span = end_dt - begin_dt
    split_dt = begin_dt + timedelta(seconds=span.total_seconds() * ratio)
    return split_dt.timestamp()


def split_symbols(
    symbols: Sequence[str],
    ratio: float,
) -> Tuple[List[str], List[str]]:
    train_count = int(len(symbols) * ratio)
    train_count = max(1, min(len(symbols) - 1, train_count))
    return list(symbols[:train_count]), list(symbols[train_count:])


def build_output_paths(output_dir: str):
    return {
        "model_buy": os.path.join(output_dir, "model_buy.json"),
        "meta_buy": os.path.join(output_dir, "meta_buy.json"),
        "report_buy": os.path.join(output_dir, "shap_report_buy.html"),
        "metrics_buy": os.path.join(output_dir, "metrics_buy.json"),
        "libsvm_buy": os.path.join(output_dir, "feature_buy.libsvm"),
        "filter_buy": os.path.join(output_dir, "filter_log_buy.json"),
        "model_sell": os.path.join(output_dir, "model_sell.json"),
        "meta_sell": os.path.join(output_dir, "meta_sell.json"),
        "report_sell": os.path.join(output_dir, "shap_report_sell.html"),
        "metrics_sell": os.path.join(output_dir, "metrics_sell.json"),
        "libsvm_sell": os.path.join(output_dir, "feature_sell.libsvm"),
        "filter_sell": os.path.join(output_dir, "filter_log_sell.json"),
    }


def safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def _should_keep_feature_at_source(feat_name: str) -> bool:
    if feat_name in CORE_FEATURE_ALLOWLIST:
        return True
    return feat_name.startswith(EXTRA_ALWAYS_KEEP_PREFIXES)


def _prefilter_feature_map(feature_map: Dict[str, float]) -> Dict[str, float]:
    return {
        feat_name: feat_value
        for feat_name, feat_value in feature_map.items()
        if _should_keep_feature_at_source(feat_name)
    }


def collect_symbol_samples_worker(
    symbol: str,
    begin_time: str,
    end_time: str,
    lv_list: List[KL_TYPE],
    chan_config: Dict,
    qualified_threshold: float,
    split_ts: float,
    is_train_symbol: bool,
):
    config = CChanConfig(chan_config)
    chan = CChan(
        code=symbol,
        begin_time=begin_time,
        end_time=end_time,
        data_src=DATA_SRC_TYPE,
        lv_list=lv_list,
        config=config,
        autype=AUTYPE.NONE,
    )

    raw_samples = []
    seen_idx = set()
    step_count = 0

    for chan_snapshot in chan.step_load():
        step_count += 1
        last_klu = chan_snapshot[0][-1][-1]
        bsp_list = chan_snapshot.get_latest_bsp()
        if not bsp_list:
            continue

        last_bsp = bsp_list[0]
        cur_lv_chan = chan_snapshot[0]
        if last_bsp.klu.idx in seen_idx:
            continue
        if len(cur_lv_chan) < 2 or cur_lv_chan[-2].idx != last_bsp.klu.klc.idx:
            continue

        seen_idx.add(last_bsp.klu.idx)

        extra_feat = build_features(
            klu=last_klu,
            history=cur_lv_chan.lst,
            chan=cur_lv_chan,
        )
        last_bsp.features.add_feat(extra_feat)

        feature_map = {
            feat_name: safe_float(feat_value)
            for feat_name, feat_value in last_bsp.features.items()
        }
        feature_map = _prefilter_feature_map(feature_map)

        raw_samples.append({
            "symbol": symbol,
            "klu_idx": last_bsp.klu.idx,
            "feature": feature_map,
            "is_buy": bool(last_bsp.is_buy),
            "open_time": last_klu.time.to_str(),
            "open_ts": float(last_klu.time.ts),
            "trade_price": float(last_klu.close),
            "bsp_main_type": last_bsp.type[0].value[0],
            "bsp_types_str": last_bsp.type2str(),
        })

    raw_samples.sort(key=lambda item: item["open_ts"])
    early_segment = [
        sample for sample in raw_samples if sample["open_ts"] < split_ts
    ]
    late_segment = [
        sample for sample in raw_samples if sample["open_ts"] >= split_ts
    ]

    early_buy, early_sell, early_stats = label_bsp_quality(
        early_segment,
        qualified_threshold,
        partition="early_window",
        symbol_group="train_symbol" if is_train_symbol else "holdout_symbol",
    )
    late_buy, late_sell, late_stats = label_bsp_quality(
        late_segment,
        qualified_threshold,
        partition="late_window",
        symbol_group="train_symbol" if is_train_symbol else "holdout_symbol",
    )

    train_buy = early_buy if is_train_symbol else []
    train_sell = early_sell if is_train_symbol else []
    test_buy = late_buy if is_train_symbol else early_buy + late_buy
    test_sell = late_sell if is_train_symbol else early_sell + late_sell

    for sample in train_buy + train_sell:
        sample["partition"] = "train_time"
    for sample in late_buy + late_sell:
        if is_train_symbol:
            sample["partition"] = "test_time"
        else:
            sample["partition"] = "test_symbol_late"
    if not is_train_symbol:
        for sample in early_buy + early_sell:
            sample["partition"] = "test_symbol_early"

    return {
        "symbol": symbol,
        "is_train_symbol": is_train_symbol,
        "step_count": step_count,
        "raw_count": len(raw_samples),
        "train_buy": train_buy,
        "train_sell": train_sell,
        "test_buy": test_buy,
        "test_sell": test_sell,
        "early_stats": early_stats,
        "late_stats": late_stats,
    }


def label_bsp_quality(
    samples,
    threshold: float,
    partition: str,
    symbol_group: str,
):
    buy_samples = []
    sell_samples = []
    unpaired = 0

    next_opposite_idx = [-1] * len(samples)
    next_buy_idx = -1
    next_sell_idx = -1

    for i in range(len(samples) - 1, -1, -1):
        if samples[i]["is_buy"]:
            next_opposite_idx[i] = next_sell_idx
            next_buy_idx = i
        else:
            next_opposite_idx[i] = next_buy_idx
            next_sell_idx = i

    for i, bsp in enumerate(samples):
        partner_idx = next_opposite_idx[i]
        if partner_idx < 0:
            unpaired += 1
            continue
        partner = samples[partner_idx]

        if bsp["is_buy"]:
            change_pct = (
                (partner["trade_price"] - bsp["trade_price"])
                / (bsp["trade_price"] + 1e-9)
            )
        else:
            change_pct = (
                (bsp["trade_price"] - partner["trade_price"])
                / (bsp["trade_price"] + 1e-9)
            )

        labeled = {
            "symbol": bsp["symbol"],
            "symbol_group": symbol_group,
            "partition": partition,
            "klu_idx": bsp["klu_idx"],
            "feature": bsp["feature"],
            "is_buy": bsp["is_buy"],
            "open_time": bsp["open_time"],
            "open_ts": bsp["open_ts"],
            "trade_price": bsp["trade_price"],
            "bsp_main_type": bsp["bsp_main_type"],
            "bsp_types_str": bsp["bsp_types_str"],
            "label": 1 if change_pct >= threshold else 0,
            "change_pct": change_pct,
            "partner_time": partner["open_time"],
            "holding_bars": partner["klu_idx"] - bsp["klu_idx"],
        }
        if bsp["is_buy"]:
            buy_samples.append(labeled)
        else:
            sell_samples.append(labeled)

    stats = {
        "partition": partition,
        "symbol_group": symbol_group,
        "raw_samples": len(samples),
        "labeled_buy": len(buy_samples),
        "labeled_sell": len(sell_samples),
        "unpaired": unpaired,
    }
    return buy_samples, sell_samples, stats


def build_feature_meta(
    train_samples: List[Dict],
) -> Tuple[List[str], Dict[str, int]]:
    feature_meta = {
        "bsp_type_1": 0,
        "bsp_type_2": 1,
        "bsp_type_3": 2,
    }
    cursor = len(feature_meta)
    for sample in train_samples:
        for feat_name in sample["feature"]:
            if feat_name not in feature_meta:
                feature_meta[feat_name] = cursor
                cursor += 1
    feature_names = [""] * len(feature_meta)
    for name, idx in feature_meta.items():
        feature_names[idx] = name
    return feature_names, feature_meta


def vectorize_samples(samples: List[Dict], feature_meta: Dict[str, int]):
    ordered = sorted(
        samples,
        key=lambda item: (item["open_ts"], item["symbol"], item["klu_idx"]),
    )
    X = np.full((len(ordered), len(feature_meta)), np.nan, dtype=np.float32)
    y = np.zeros(len(ordered), dtype=np.int32)
    sample_info = []

    for row_idx, sample in enumerate(ordered):
        y[row_idx] = sample["label"]
        X[row_idx, feature_meta["bsp_type_1"]] = (
            1.0 if sample["bsp_main_type"] == "1" else 0.0
        )
        X[row_idx, feature_meta["bsp_type_2"]] = (
            1.0 if sample["bsp_main_type"] == "2" else 0.0
        )
        X[row_idx, feature_meta["bsp_type_3"]] = (
            1.0 if sample["bsp_main_type"] == "3" else 0.0
        )

        for feat_name, feat_value in sample["feature"].items():
            if feat_name in feature_meta:
                X[row_idx, feature_meta[feat_name]] = feat_value

        sample_info.append({
            "symbol": sample["symbol"],
            "partition": sample["partition"],
            "symbol_group": sample["symbol_group"],
            "open_time": sample["open_time"],
            "label": sample["label"],
            "change_pct": sample["change_pct"],
            "bsp_main_type": sample["bsp_main_type"],
        })
    return X, y, sample_info


def _low_info_filter(X_train, feature_names):
    keep_mask = np.ones(len(feature_names), dtype=bool)
    removed = []

    for idx, name in enumerate(feature_names):
        col = X_train[:, idx]
        valid = col[~np.isnan(col)]
        nan_ratio = float(np.isnan(col).mean())

        if nan_ratio >= HIGH_NAN_THRESHOLD:
            keep_mask[idx] = False
            removed.append(
                {
                    "name": name,
                    "reason": "high_nan",
                    "nan_ratio": round(nan_ratio, 4),
                }
            )
            continue

        if len(valid) == 0:
            keep_mask[idx] = False
            removed.append(
                {
                    "name": name,
                    "reason": "all_nan",
                    "nan_ratio": round(nan_ratio, 4),
                }
            )
            continue

        uniq, counts = np.unique(valid, return_counts=True)
        if len(uniq) <= LOW_INFO_UNIQUE_THRESHOLD:
            keep_mask[idx] = False
            removed.append(
                {
                    "name": name,
                    "reason": "constant",
                    "unique": int(len(uniq)),
                    "nan_ratio": round(nan_ratio, 4),
                }
            )
            continue

        dominant_ratio = float(counts.max() / len(valid))
        if len(uniq) <= 8 and dominant_ratio >= LOW_INFO_DOMINANT_RATIO:
            keep_mask[idx] = False
            removed.append(
                {
                    "name": name,
                    "reason": "near_constant",
                    "dominant_ratio": round(dominant_ratio, 4),
                    "unique": int(len(uniq)),
                    "nan_ratio": round(nan_ratio, 4),
                }
            )

    return keep_mask, removed


def _correlation_filter(X_train, feature_names):
    n_features = len(feature_names)
    if n_features <= 1:
        return np.ones(n_features, dtype=bool), []

    df = pd.DataFrame(X_train, columns=feature_names)
    corr_df = df.corr(method="spearman", min_periods=30).abs()

    nan_ratios = np.isnan(X_train).mean(axis=0)
    variances = np.nanvar(X_train, axis=0)

    to_remove = set()
    removed_pairs = []

    for i in range(n_features):
        if i in to_remove:
            continue
        for j in range(i + 1, n_features):
            if j in to_remove:
                continue

            corr_val = corr_df.iat[i, j]
            if np.isnan(corr_val) or corr_val <= CORR_THRESHOLD:
                continue

            # 优先保留缺失更少的特征；若相同则保留方差更大的特征。
            score_i = (nan_ratios[i], -variances[i])
            score_j = (nan_ratios[j], -variances[j])

            if score_i <= score_j:
                keep_idx, remove_idx = i, j
            else:
                keep_idx, remove_idx = j, i

            to_remove.add(remove_idx)
            removed_pairs.append(
                {
                    "removed": feature_names[remove_idx],
                    "kept": feature_names[keep_idx],
                    "corr": round(float(corr_val), 4),
                }
            )

    keep_mask = np.array([i not in to_remove for i in range(n_features)], dtype=bool)
    return keep_mask, removed_pairs


def _model_gain_prune(
    X_train,
    y_train,
    X_test,
    y_test,
    feature_names,
    train_mode,
    model_prune_topk,
):
    n_pos = int(y_train.sum())
    n_neg = len(y_train) - n_pos
    scale_pw = n_neg / max(n_pos, 1)

    params = build_xgb_params(train_mode, scale_pw)
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names, missing=np.nan)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names, missing=np.nan)

    booster = xgb.train(
        params,
        dtrain=dtrain,
        num_boost_round=MODEL_PRUNE_ROUNDS,
        evals=[(dtest, "eval")],
        early_stopping_rounds=20,
        verbose_eval=False,
    )

    gain = booster.get_score(importance_type="gain")
    gain_items = sorted(
        [(name, float(gain.get(name, 0.0))) for name in feature_names],
        key=lambda item: item[1],
        reverse=True,
    )

    mandatory = {
        name
        for name in feature_names
        if (name in CORE_FEATURE_ALLOWLIST or name.startswith(EXTRA_ALWAYS_KEEP_PREFIXES))
    }

    topk = min(max(10, model_prune_topk), len(feature_names))
    min_keep = min(len(feature_names), max(20, topk // 2))

    selected = []
    selected_set = set()

    for name in sorted(mandatory):
        selected.append(name)
        selected_set.add(name)

    for name, g in gain_items:
        if name in selected_set:
            continue
        if len(selected) >= topk and g <= MODEL_PRUNE_MIN_GAIN:
            continue
        selected.append(name)
        selected_set.add(name)
        if len(selected) >= topk:
            break

    if len(selected) < min_keep:
        for name, _ in gain_items:
            if name in selected_set:
                continue
            selected.append(name)
            selected_set.add(name)
            if len(selected) >= min_keep:
                break

    keep_mask = np.array([name in selected_set for name in feature_names], dtype=bool)
    removed = [
        {
            "name": name,
            "gain": round(float(g), 6),
        }
        for name, g in gain_items
        if name not in selected_set
    ]
    return keep_mask, removed


def filter_features(
    X_train,
    y_train,
    X_test,
    y_test,
    feature_names,
    direction_name="",
    train_mode="auto",
    feature_prune_mode="basic",
    model_prune_topk=40,
):
    print("\n" + "=" * 60)
    print(f"[阶段4] 特征过滤 [{direction_name}] 模式={feature_prune_mode}")
    print("=" * 60)

    filter_log = {
        "original_count": len(feature_names),
        "feature_prune_mode": feature_prune_mode,
        "low_info_removed": [],
        "correlated_removed": [],
        "model_gain_removed": [],
    }

    if feature_prune_mode != "off":
        keep_mask, removed = _low_info_filter(X_train, feature_names)
        filter_log["low_info_removed"] = removed
        X_train = X_train[:, keep_mask]
        X_test = X_test[:, keep_mask]
        feature_names = [name for name, keep in zip(feature_names, keep_mask) if keep]

    if feature_prune_mode in {"basic", "aggressive"} and len(feature_names) > 1:
        keep_mask, removed = _correlation_filter(X_train, feature_names)
        filter_log["correlated_removed"] = removed
        X_train = X_train[:, keep_mask]
        X_test = X_test[:, keep_mask]
        feature_names = [name for name, keep in zip(feature_names, keep_mask) if keep]

    if feature_prune_mode == "aggressive" and len(feature_names) > 10:
        keep_mask, removed = _model_gain_prune(
            X_train,
            y_train,
            X_test,
            y_test,
            feature_names,
            train_mode,
            model_prune_topk,
        )
        filter_log["model_gain_removed"] = removed
        X_train = X_train[:, keep_mask]
        X_test = X_test[:, keep_mask]
        feature_names = [name for name, keep in zip(feature_names, keep_mask) if keep]

    filter_log["final_count"] = len(feature_names)
    filter_log["final_features"] = feature_names
    print(f"  保留特征数: {len(feature_names)}")
    return X_train, X_test, feature_names, filter_log


def build_xgb_params(train_mode: str, scale_pos_weight: float):
    params = {
        **XGB_BASE_PARAMS,
        "scale_pos_weight": scale_pos_weight,
        "tree_method": "hist",
    }
    if train_mode == "gpu":
        params["device"] = "cuda"
    elif train_mode == "cpu":
        params["device"] = "cpu"
    else:
        params["device"] = "cuda"
    return params


def infer_booster_device(booster) -> str:
    try:
        config = json.loads(booster.save_config())
        return (
            config
            .get("learner", {})
            .get("generic_param", {})
            .get("device", "")
        )
    except Exception:
        return ""


def device_label(device_text: str) -> str:
    text = (device_text or "").lower()
    if "cuda" in text or "gpu" in text:
        return "GPU"
    if "cpu" in text:
        return "CPU"
    return "UNKNOWN"


def compute_binary_metrics(y_true, y_pred, y_prob):
    auc = (
        float(roc_auc_score(y_true, y_prob))
        if len(np.unique(y_true)) > 1
        else 0.0
    )
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "AUC": auc,
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "Positive_Rate": float(np.mean(y_true)) if len(y_true) else 0.0,
    }


def build_partition_metrics(sample_info, y_true, y_pred, y_prob):
    grouped = defaultdict(list)
    for idx, info in enumerate(sample_info):
        grouped[info["partition"]].append(idx)

    result = {}
    for partition, idxs in grouped.items():
        part_true = y_true[idxs]
        part_pred = y_pred[idxs]
        part_prob = y_prob[idxs]
        metrics = compute_binary_metrics(part_true, part_pred, part_prob)
        metrics["samples"] = len(idxs)
        result[partition] = metrics
    return result


def train_and_evaluate(
    X_train,
    y_train,
    X_test,
    y_test,
    feature_names,
    direction_name,
    train_mode,
    sample_info,
):
    print("\n" + "=" * 60)
    print(f"[阶段5] 模型训练与评估 [{direction_name}]")
    print("=" * 60)

    n_pos_train = int(y_train.sum())
    n_neg_train = len(y_train) - n_pos_train
    scale_pw = n_neg_train / max(n_pos_train, 1)
    params = build_xgb_params(train_mode, scale_pw)
    requested_device = params.get("device", "cpu")

    dtrain = xgb.DMatrix(
        X_train,
        label=y_train,
        feature_names=feature_names,
        missing=np.nan,
    )
    dtest = xgb.DMatrix(
        X_test,
        label=y_test,
        feature_names=feature_names,
        missing=np.nan,
    )
    evals_result = {}

    print(f"  训练集: {len(y_train)}  测试集: {len(y_test)}")
    print(f"  训练正例率: {y_train.mean():.1%}  测试正例率: {y_test.mean():.1%}")
    print(f"  训练模式: {train_mode}  scale_pos_weight: {scale_pw:.2f}")
    print(
        f"  [设备] 请求训练设备: {device_label(requested_device)} "
        f"({requested_device})"
    )

    try:
        booster = xgb.train(
            params,
            dtrain=dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            evals=[(dtrain, "train"), (dtest, "test")],
            evals_result=evals_result,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=20,
        )
        effective_mode = "gpu" if device_label(requested_device) == "GPU" else "cpu"
    except xgb.core.XGBoostError:
        if train_mode != "auto":
            raise
        print("  [Info] GPU 不可用，自动回退到 CPU")
        params = build_xgb_params("cpu", scale_pw)
        booster = xgb.train(
            params,
            dtrain=dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            evals=[(dtrain, "train"), (dtest, "test")],
            evals_result=evals_result,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=20,
        )
        effective_mode = "cpu"

    actual_device = infer_booster_device(booster) or params.get("device", "unknown")
    actual_device_kind = device_label(actual_device)
    print(f"  [设备] 实际训练设备: {actual_device_kind} ({actual_device})")

    y_pred_prob = booster.predict(dtest)
    y_pred = (y_pred_prob >= 0.5).astype(int)
    metrics = compute_binary_metrics(y_test, y_pred, y_pred_prob)
    metrics.update({
        "direction": direction_name,
        "Train_Samples": int(len(y_train)),
        "Test_Samples": int(len(y_test)),
        "Best_Iteration": (
            int(booster.best_iteration)
            if hasattr(booster, "best_iteration")
            else NUM_BOOST_ROUND
        ),
        "Best_AUC_Train": float(max(evals_result["train"]["auc"])),
        "Best_AUC_Test": float(max(evals_result["test"]["auc"])),
        "scale_pos_weight": round(scale_pw, 4),
        "n_features": len(feature_names),
        "train_mode": effective_mode,
        "train_mode_request": train_mode,
        "train_device_request": requested_device,
        "train_device": actual_device,
        "train_device_kind": actual_device_kind,
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=["不合格", "合格"],
            zero_division=0,
            output_dict=True,
        ),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "test_partition_metrics": build_partition_metrics(
            sample_info,
            y_test,
            y_pred,
            y_pred_prob,
        ),
    })

    print(f"\n  ═══════ [{direction_name}] 测试集评估 ═══════")
    print(f"  Accuracy  : {metrics['Accuracy']:.4f}")
    print(f"  AUC       : {metrics['AUC']:.4f}")
    print(f"  Precision : {metrics['Precision']:.4f}")
    print(f"  Recall    : {metrics['Recall']:.4f}")
    print(f"  F1 Score  : {metrics['F1']:.4f}")
    print(f"  Best Iter : {metrics['Best_Iteration']}")

    return booster, metrics


def sample_for_shap(X, y, sample_limit: int):
    if len(y) <= sample_limit:
        return X, y
    idx = np.linspace(0, len(y) - 1, num=sample_limit, dtype=int)
    return X[idx], y[idx]


def shap_analysis(
    bst,
    X,
    y,
    feature_names,
    metrics,
    report_path,
    direction_name="",
):
    print("\n" + "=" * 60)
    print(f"[阶段6] SHAP 可解释性分析 [{direction_name}]")
    print("=" * 60)

    analyzer = SHAPAnalyzer(model=bst, feature_names=feature_names)
    t0 = time.time()
    result = analyzer.analyze(X, y)
    elapsed = time.time() - t0
    print(f"  ✓ SHAP 完成 ({elapsed:.1f}s, 样本={len(y)})")

    analyzer.generate_report(
        result,
        output_path=report_path,
        model_metrics=metrics,
        top_dependence=6,
    )
    return result


def time_series_cv(X, y, feature_names, direction_name, train_mode):
    print("\n" + "=" * 60)
    print(f"[阶段7] TimeSeriesSplit 交叉验证 [{direction_name}]")
    print("=" * 60)
    if len(y) < 200:
        print("  样本过少，跳过 TimeSeriesSplit")
        return []

    n_splits = min(5, max(2, len(y) // 500))
    if n_splits < 2:
        return []

    fold_aucs = []
    tscv = TimeSeriesSplit(n_splits=n_splits)
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        scale_pw = (len(y_tr) - int(y_tr.sum())) / max(int(y_tr.sum()), 1)
        params = build_xgb_params(
            train_mode if train_mode != "auto" else "cpu",
            scale_pw,
        )
        dtrain = xgb.DMatrix(
            X_tr,
            label=y_tr,
            feature_names=feature_names,
            missing=np.nan,
        )
        dtest = xgb.DMatrix(
            X_te,
            label=y_te,
            feature_names=feature_names,
            missing=np.nan,
        )
        bst = xgb.train(
            params,
            dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            evals=[(dtest, "eval")],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=False,
        )
        pred_prob = bst.predict(dtest)
        if len(np.unique(y_te)) < 2:
            print(f"  Fold {fold}: 跳过 (测试集只有单类)")
            continue
        auc = roc_auc_score(y_te, pred_prob)
        fold_aucs.append(float(auc))
        print(f"  Fold {fold}: AUC={auc:.4f} (训练={len(y_tr)}, 测试={len(y_te)})")
    return fold_aucs


def save_json(path, payload):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def save_libsvm(path, X, y):
    with open(path, "w", encoding="utf-8") as file:
        for row_idx in range(len(y)):
            pairs = [
                f"{col_idx}:{float(X[row_idx, col_idx])}"
                for col_idx in range(X.shape[1])
                if not np.isnan(X[row_idx, col_idx])
            ]
            file.write(f"{int(y[row_idx])} {' '.join(pairs)}\n")


def summarize_samples(name, samples: List[Dict]):
    if not samples:
        print(f"  {name}: 0 样本")
        return
    pos = sum(sample["label"] for sample in samples)
    type_counts = Counter(sample["bsp_main_type"] for sample in samples)
    part_counts = Counter(sample["partition"] for sample in samples)
    print(f"  {name}: {len(samples)} 样本, 正例={pos}({pos / len(samples):.1%})")
    print(f"    类型分布: {dict(type_counts)}")
    print(f"    分区分布: {dict(part_counts)}")


def train_direction_pipeline(
    train_samples,
    test_samples,
    direction_name,
    model_path,
    meta_path,
    report_path,
    metrics_path,
    libsvm_path,
    filter_log_path,
    args: TrainArgs,
    split_summary: Dict,
):
    print("\n" + "▓" * 60)
    print(f"  >>> {direction_name}模型训练流水线 <<<")
    print("▓" * 60)

    if len(train_samples) < 20 or len(test_samples) < 20:
        print(
            f"  [跳过] {direction_name} 样本不足: "
            f"train={len(train_samples)} test={len(test_samples)}"
        )
        return None

    summarize_samples("训练集", train_samples)
    summarize_samples("测试集", test_samples)

    feature_names, feature_meta = build_feature_meta(train_samples)
    X_train, y_train, _ = vectorize_samples(train_samples, feature_meta)
    X_test, y_test, test_info = vectorize_samples(test_samples, feature_meta)

    X_train, X_test, feature_names, filter_log = filter_features(
        X_train,
        y_train,
        X_test,
        y_test,
        feature_names,
        direction_name=direction_name,
        train_mode=args.train_mode,
        feature_prune_mode=args.feature_prune_mode,
        model_prune_topk=args.model_prune_topk,
    )

    filtered_meta = {name: idx for idx, name in enumerate(feature_names)}
    save_json(meta_path, filtered_meta)
    save_json(filter_log_path, filter_log)
    save_libsvm(libsvm_path, X_train, y_train)
    print(f"  ✓ Meta: {meta_path}")
    print(f"  ✓ 过滤日志: {filter_log_path}")
    print(f"  ✓ LibSVM: {libsvm_path}")

    booster, metrics = train_and_evaluate(
        X_train,
        y_train,
        X_test,
        y_test,
        feature_names,
        direction_name,
        args.train_mode,
        test_info,
    )
    booster.save_model(model_path)
    print(f"  ✓ 模型已保存: {model_path}")

    X_shap, y_shap = sample_for_shap(X_train, y_train, args.shap_sample_limit)
    shap_result = shap_analysis(
        booster,
        X_shap,
        y_shap,
        feature_names,
        metrics,
        report_path,
        direction_name=direction_name,
    )

    cv_aucs = time_series_cv(
        X_train,
        y_train,
        feature_names,
        direction_name,
        metrics["train_mode"],
    )
    metrics["cv_auc_scores"] = cv_aucs
    metrics["cv_auc_mean"] = float(np.mean(cv_aucs)) if cv_aucs else None
    metrics["cv_auc_std"] = float(np.std(cv_aucs)) if cv_aucs else None
    metrics["split_summary"] = split_summary
    save_json(metrics_path, metrics)
    print(f"  ✓ 指标: {metrics_path}")

    return {
        "model": booster,
        "metrics": metrics,
        "feature_names": feature_names,
        "shap_result": shap_result,
    }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    output_paths = build_output_paths(args.output_dir)

    split_ts = compute_split_ts(
        args.begin_time,
        args.end_time,
        args.train_ratio,
    )
    split_dt = datetime.fromtimestamp(split_ts)
    train_symbols, holdout_symbols = split_symbols(
        args.symbols,
        args.symbol_train_ratio,
    )

    split_summary = {
        "begin_time": args.begin_time,
        "end_time": args.end_time,
        "time_train_ratio": args.train_ratio,
        "symbol_train_ratio": args.symbol_train_ratio,
        "time_split": split_dt.strftime("%Y-%m-%d %H:%M:%S"),
        "train_symbols": train_symbols,
        "holdout_symbols": holdout_symbols,
        "kl_type": args.kl_type.name,
    }

    print("\n" + "★" * 60)
    print("   XGBoost + SHAP 多币种训练流程")
    print(f"   数据源: {DATA_SRC_TYPE.name}  周期: {args.kl_type.name}")
    print(f"   区间: {args.begin_time} ~ {args.end_time}")
    print(f"   时间切分: 4:1  →  {split_summary['time_split']}")
    print(f"   训练币种({len(train_symbols)}): {', '.join(train_symbols)}")
    print(f"   保留币种({len(holdout_symbols)}): {', '.join(holdout_symbols)}")
    print(f"   训练模式: {args.train_mode}  并行进程: {args.num_workers}")
    print(f"   特征过滤模式: {args.feature_prune_mode}  topk: {args.model_prune_topk}")
    print("★" * 60)

    start_time = time.time()
    jobs = []
    buy_train_samples = []
    sell_train_samples = []
    buy_test_samples = []
    sell_test_samples = []
    collect_stats = []

    with ProcessPoolExecutor(
        max_workers=min(args.num_workers, len(args.symbols)),
    ) as executor:
        for symbol in args.symbols:
            jobs.append(
                executor.submit(
                    collect_symbol_samples_worker,
                    symbol,
                    args.begin_time,
                    args.end_time,
                    [args.kl_type],
                    CHAN_CONFIG,
                    args.qualified_threshold,
                    split_ts,
                    symbol in train_symbols,
                )
            )

        for future in as_completed(jobs):
            result = future.result()
            collect_stats.append(result)
            buy_train_samples.extend(result["train_buy"])
            sell_train_samples.extend(result["train_sell"])
            buy_test_samples.extend(result["test_buy"])
            sell_test_samples.extend(result["test_sell"])
            print(
                f"  [完成] {result['symbol']}: raw={result['raw_count']} "
                f"train_buy={len(result['train_buy'])} "
                f"train_sell={len(result['train_sell'])} "
                f"test_buy={len(result['test_buy'])} "
                f"test_sell={len(result['test_sell'])}"
            )

    split_summary["symbol_stats"] = {
        item["symbol"]: {
            "raw_count": item["raw_count"],
            "step_count": item["step_count"],
            "train_symbol": item["is_train_symbol"],
            "early_stats": item["early_stats"],
            "late_stats": item["late_stats"],
        }
        for item in sorted(collect_stats, key=lambda row: row["symbol"])
    }

    buy_result = train_direction_pipeline(
        buy_train_samples,
        buy_test_samples,
        "买点",
        output_paths["model_buy"],
        output_paths["meta_buy"],
        output_paths["report_buy"],
        output_paths["metrics_buy"],
        output_paths["libsvm_buy"],
        output_paths["filter_buy"],
        args,
        split_summary,
    )
    sell_result = train_direction_pipeline(
        sell_train_samples,
        sell_test_samples,
        "卖点",
        output_paths["model_sell"],
        output_paths["meta_sell"],
        output_paths["report_sell"],
        output_paths["metrics_sell"],
        output_paths["libsvm_sell"],
        output_paths["filter_sell"],
        args,
        split_summary,
    )

    total_time = time.time() - start_time
    print(f"\n{'★' * 60}")
    print(f"   全部完成! 耗时: {total_time:.1f}s")
    if buy_result:
        print(f"   买点模型: {output_paths['model_buy']}")
        print(f"   买点指标: {output_paths['metrics_buy']}")
    if sell_result:
        print(f"   卖点模型: {output_paths['model_sell']}")
        print(f"   卖点指标: {output_paths['metrics_sell']}")
    print(f"{'★' * 60}")


if __name__ == "__main__":
    mp.freeze_support()
    _log_fp = None
    _log_path = ""
    try:
        _log_fp, _log_path = setup_train_log()
        main()
    finally:
        close_train_log(_log_fp, _log_path)
