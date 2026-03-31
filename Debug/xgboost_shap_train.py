"""
L3 + L4 重构训练入口
- L3: ml_layer.feature_engine.FeatureEngine
- L4: ml_layer.label_engine.LabelEngine + ml_layer.train_validator.TrainValidator

兼容性:
- 保留 run_pipeline.py 现有参数
- 保留 model_buy/model_sell/meta_buy/meta_sell 产物命名
"""

# flake8: noqa: E501, E402

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from Chan import CChan
from ChanConfig import CChanConfig
from ChanModel.shap_analyzer import SHAPAnalyzer
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE
from ml_layer.cache_layer import (
    _config_hash,
    cache_stats,
    clear_all_caches,
    clear_cache_namespace,
    get_feature_cache,
    get_label_cache,
)
from ml_layer.config import FeatureConfig, LabelConfig, ModelConfig, TrainValidatorConfig
from ml_layer.feature_engine import FeatureEngine, CachedFeatureEngine
from ml_layer.label_engine import LabelEngine
from ml_layer.train_validator import TrainValidator
from ml_layer.utils.dual_model_diagnostics import build_dual_model_diagnostics
from ml_layer.utils.model_visual_reports import generate_meta_model_visual_report
from ml_layer.utils.optimization_objective import BacktestObjectiveConfig


BEGIN_TIME = "2020-01-01"
END_TIME = "2026-03-16"
TRAIN_KL_TYPE_TEXT = "15m"
DATA_SRC_TYPE = DATA_SRC.PARQUET
NUM_WORKERS = max(1, (os.cpu_count() or 2) - 1)
OUTPUT_DIR = "Debug"

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


class TeeStream:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for stream in self.streams:
            stream.write(data)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def _is_true_env(name: str) -> bool:
    value = str(os.environ.get(name, "")).strip().lower()
    return value in {"1", "true", "yes", "on"}


def setup_train_log():
    if _is_true_env("XGB_DISABLE_INNER_FILE_LOG"):
        run_id = str(os.environ.get("XGB_PIPELINE_RUN_ID", "")).strip()
        stage = str(os.environ.get("XGB_PIPELINE_STAGE", "train")).strip() or "train"
        mode = str(os.environ.get("XGB_PIPELINE_MODE", "")).strip()
        header = f"run_id={run_id}, stage={stage}"
        if mode:
            header = f"{header}, mode={mode}"
        print(f"[LOG] 内层文件日志已禁用，统一使用外层 pipeline 日志 ({header})")
        return None, ""

    project_root = Path(__file__).resolve().parents[1]
    log_dir_env = str(os.environ.get("XGB_TRAIN_LOG_DIR", "")).strip()
    if log_dir_env:
        log_dir = Path(log_dir_env)
    else:
        log_dir = project_root / "result"
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = str(os.environ.get("XGB_PIPELINE_RUN_ID", "")).strip()
    if run_id:
        stage = str(os.environ.get("XGB_PIPELINE_STAGE", "train")).strip() or "train"
        log_name = f"{run_id}_{stage}_{timestamp}.log"
    else:
        log_name = f"train_validator_{timestamp}.log"
    log_path = log_dir / log_name

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


def snapshot_architecture_docs(output_dir: str) -> Dict[str, str]:
    root = Path(__file__).resolve().parents[1]
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    snapshots: Dict[str, str] = {}
    for name in ["MODEL_TRAINING_ARCHITECTURE.md", "chanlun_L3_L4_architecture.md"]:
        src = root / "Debug" / name
        if not src.exists():
            continue
        dst = out / f"{src.stem}_{ts}.md"
        dst.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        snapshots[name] = str(dst)
        print(f"[Snapshot] {name} -> {dst}")
    return snapshots


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
    if text not in mapping:
        raise ValueError(f"不支持的周期: {text}")
    return mapping[text]


def normalize_symbols(symbols: Sequence[str]) -> List[str]:
    out = []
    for symbol in symbols:
        item = symbol.upper()
        if not item.endswith("USDT"):
            item = f"{item}USDT"
        out.append(item)
    return sorted(dict.fromkeys(out))


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
        raise FileNotFoundError(f"未在 {data_dir} 找到 *_{interval_text}.parquet 文件")
    return symbols


def ensure_15m_parquet_from_5m(symbols: Sequence[str]):
    data_dir = Path(__file__).resolve().parents[1] / "data"
    required_cols = ["open_time", "open", "high", "low", "close", "volume"]
    generated = 0
    reused = 0
    for symbol in symbols:
        base = symbol.replace("USDT", "")
        src = data_dir / f"{base}_5m.parquet"
        dst = data_dir / f"{base}_15m.parquet"
        if dst.exists():
            reused += 1
            continue
        if not src.exists():
            raise FileNotFoundError(f"缺少5m数据: {src}")
        df = pd.read_parquet(src, columns=required_cols)
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        for col in required_cols[1:]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["open_time", "open", "high", "low", "close"]).sort_values("open_time")
        agg = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
        df15 = (
            df.set_index("open_time")
            .resample("15min", label="left", closed="left", origin="epoch")
            .agg(agg)
            .dropna(subset=["open", "high", "low", "close"])
            .reset_index()
        )
        df15["open_time"] = (df15["open_time"].astype("int64") // 1_000_000).astype("int64")
        df15.to_parquet(dst, index=False)
        generated += 1
    print(f"[K线合并] 15m文件准备完成: 新生成={generated}, 已存在复用={reused}")


def safe_float(value) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def collect_symbol_events_worker(
    symbol: str,
    begin_time: str,
    end_time: str,
    lv_list: List[KL_TYPE],
    chan_config: Dict,
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

    events = []
    bars = []
    seen_event_idx = set()
    seen_bar_idx = set()

    for snapshot in chan.step_load():
        last_klu = snapshot[0][-1][-1]
        if last_klu.idx not in seen_bar_idx:
            seen_bar_idx.add(last_klu.idx)
            bars.append(
                {
                    "klu_idx": int(last_klu.idx),
                    "open_ts": float(last_klu.time.ts),
                    "open": float(getattr(last_klu, "open", last_klu.close)),
                    "high": float(getattr(last_klu, "high", last_klu.close)),
                    "low": float(getattr(last_klu, "low", last_klu.close)),
                    "close": float(last_klu.close),
                    "volume": float(getattr(last_klu, "vol", 0.0)),
                    "time": last_klu.time.to_str(),
                }
            )

        bsp_list = snapshot.get_latest_bsp()
        if not bsp_list:
            continue

        last_bsp = bsp_list[0]
        cur_lv = snapshot[0]
        if last_bsp.klu.idx in seen_event_idx:
            continue
        if len(cur_lv) < 2 or cur_lv[-2].idx != last_bsp.klu.klc.idx:
            continue
        seen_event_idx.add(last_bsp.klu.idx)

        f_map = {k: safe_float(v) for k, v in last_bsp.features.items()}

        chan_struct = {
            "beichi_strength": safe_float(f_map.get("beichi_strength", 0.0)),
            "zs_bi_count": safe_float(f_map.get("zs_bi_count", 0.0)),
            "zs_expand_count": safe_float(f_map.get("zs_expand_count", 0.0)),
            "bi_length": safe_float(f_map.get("bi_length", 0.0)),
            "bi_return_ratio": safe_float(f_map.get("bi_return_ratio", 0.0)),
            "bi_length_ratio": safe_float(f_map.get("bi_length_ratio", 0.0)),
            "bi_macd_area": safe_float(f_map.get("bi_macd_area", 0.0)),
            "distance_to_zhongshu_center": safe_float(f_map.get("distance_to_zhongshu_center", 0.0)),
            "zs_peak_range": safe_float(f_map.get("zs_peak_range", 0.0)),
            "seg_direction": safe_float(f_map.get("seg_direction", 0.0)),
            "seg_bi_count": safe_float(f_map.get("seg_bi_count", 0.0)),
        }

        zs_low = float("nan")
        zs_high = float("nan")
        zs_list = getattr(cur_lv, "zs_list", None)
        if zs_list is not None and len(zs_list) > 0:
            try:
                last_zs = zs_list[-1]
                zs_low = safe_float(getattr(last_zs, "low", float("nan")))
                zs_high = safe_float(getattr(last_zs, "high", float("nan")))
            except Exception:
                pass

        events.append(
            {
                "symbol": symbol,
                "klu_idx": int(last_bsp.klu.idx),
                "open_ts": float(last_klu.time.ts),
                "open_time": last_klu.time.to_str(),
                "trade_price": float(last_klu.close),
                "is_buy": bool(last_bsp.is_buy),
                "bsp_main_type": last_bsp.type[0].value[0],
                "bsp_types_str": last_bsp.type2str(),
                "bi_start_price": safe_float(last_bsp.bi.get_begin_val()),
                "zs_low": zs_low,
                "zs_high": zs_high,
                "chan_struct": chan_struct,
            }
        )

    events.sort(key=lambda x: x["open_ts"])
    bars.sort(key=lambda x: x["open_ts"])
    return {"symbol": symbol, "events": events, "bars": bars}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="L3+L4 分层训练引擎")
    parser.add_argument("--symbols", nargs="*", help="币种列表")
    parser.add_argument("--begin-time", default=BEGIN_TIME)
    parser.add_argument("--end-time", default=END_TIME)
    parser.add_argument("--kl-type", default=TRAIN_KL_TYPE_TEXT, choices=[TRAIN_KL_TYPE_TEXT])
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--feature-symbol-workers", type=int, default=max(1, min(8, NUM_WORKERS // 2)))
    parser.add_argument("--train-mode", choices=["cpu", "gpu", "auto"], default="auto")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)

    parser.add_argument("--labeling-strategy", default="trainvalidator_hierarchical")
    parser.add_argument("--labeling-name", default="TrainValidator_v2")
    parser.add_argument("--pt-multiplier", type=float, default=2.0)
    parser.add_argument("--dynamic-pt-enabled", action="store_true", help="启用按波动状态动态PT倍数")
    parser.add_argument("--pt-low-vol-multiplier", type=float, default=1.8)
    parser.add_argument("--pt-mid-vol-multiplier", type=float, default=2.0)
    parser.add_argument("--pt-high-vol-multiplier", type=float, default=2.2)
    parser.add_argument("--pt-vol-window", type=int, default=96)
    parser.add_argument("--pt-vol-quantile-low", type=float, default=0.33)
    parser.add_argument("--pt-vol-quantile-high", type=float, default=0.67)
    parser.add_argument("--timeout-bars", type=int, default=20, help="deprecated: ignored in binary PT/SL labeling")
    parser.add_argument("--weak-timeout-bars", type=int, default=10, help="deprecated: ignored in binary PT/SL labeling")
    parser.add_argument("--weak-bsp-types", default="3")

    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument("--embargo-bars", type=int, default=10)
    parser.add_argument("--meta-threshold", type=float, default=0.55)
    parser.add_argument("--primary-model", choices=["xgboost"], default="xgboost")
    parser.add_argument("--meta-model", choices=["logistic"], default="logistic")
    parser.add_argument("--meta-calibration", choices=["none", "platt", "isotonic"], default="none")
    parser.add_argument("--meta-calibration-ratio", type=float, default=0.2)
    parser.add_argument("--optuna-trials", type=int, default=0)
    parser.add_argument("--pass-positive-month-ratio", type=float, default=0.70)
    parser.add_argument("--mda-max-samples", type=int, default=6000)
    parser.add_argument("--mda-n-jobs", type=int, default=max(1, min(8, NUM_WORKERS // 2)))
    parser.add_argument("--shap-sample-limit", type=int, default=5000)
    parser.add_argument("--disable-label-cache", action="store_true", help="禁用标签缓存")
    parser.add_argument("--disable-feature-cache", action="store_true", help="禁用特征缓存")
    parser.add_argument("--clear-cache", action="store_true", help="清空所有缓存并退出")
    parser.add_argument(
        "--cache-mode",
        choices=["resume", "fresh"],
        default="resume",
        help=(
            "缓存模式: resume=续跑模式，优先复用历史缓存；"
            "fresh=全新模式，先清空命名空间并禁用缓存。"
        ),
    )
    parser.add_argument(
        "--cache-namespace",
        default="default",
        help="缓存命名空间（同一实验使用同名可续跑，默认default）",
    )
    return parser


def sort_samples(samples: List[Dict]) -> List[Dict]:
    return sorted(samples, key=lambda x: (float(x["t0_ts"]), x["symbol"], int(x["klu_idx"])))


def build_global_pos(samples: List[Dict]) -> tuple[np.ndarray, np.ndarray]:
    ts_all = sorted({float(s["t0_ts"]) for s in samples} | {float(s["t1_ts"]) for s in samples})
    ts_pos = {ts: i for i, ts in enumerate(ts_all)}
    t0 = np.array([int(ts_pos[float(s["t0_ts"])]) for s in samples], dtype=np.int32)
    t1 = np.array([int(ts_pos[float(s["t1_ts"])]) for s in samples], dtype=np.int32)
    return t0, t1


def sample_for_shap(X: np.ndarray, y: np.ndarray, sample_limit: int) -> tuple[np.ndarray, np.ndarray]:
    n = int(X.shape[0])
    if n <= sample_limit:
        return X, y
    idx = np.linspace(0, n - 1, num=int(sample_limit), dtype=int)
    return X[idx], y[idx]


def save_outputs(
    output_dir: str,
    feature_meta: Dict[str, int],
    report: Dict,
    mdi_df: pd.DataFrame,
    mda_df: pd.DataFrame,
    selected_features: List[str],
    primary_model,
    meta_model,
    samples: List[Dict],
    architecture_snapshots: Dict[str, str],
    visual_reports: Dict[str, str],
):
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    primary_path = out / "primary_model.json"
    primary_model.save(str(primary_path))

    with open(out / "primary_feature_meta.json", "w", encoding="utf-8") as f:
        json.dump(feature_meta, f, indent=2, ensure_ascii=False)

    with open(out / "meta_model.pkl", "wb") as f:
        pickle.dump(meta_model, f)

    report["architecture_doc_snapshots"] = architecture_snapshots
    report["visual_reports"] = visual_reports
    with open(out / "train_validator_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    mdi_df.to_csv(out / "feature_importance_mdi.csv", index=False, encoding="utf-8")
    mda_df.to_csv(out / "feature_importance_mda.csv", index=False, encoding="utf-8")
    with open(out / "selected_features.json", "w", encoding="utf-8") as f:
        json.dump({"selected": selected_features}, f, indent=2, ensure_ascii=False)

    # 兼容旧回测入口
    primary_model.save(str(out / "model_buy.json"))
    primary_model.save(str(out / "model_sell.json"))
    with open(out / "meta_buy.json", "w", encoding="utf-8") as f:
        json.dump(feature_meta, f, indent=2, ensure_ascii=False)
    with open(out / "meta_sell.json", "w", encoding="utf-8") as f:
        json.dump(feature_meta, f, indent=2, ensure_ascii=False)

    rec = pd.DataFrame(
        [
            {
                "symbol": s["symbol"],
                "open_time": s["open_time"],
                "is_buy": int(bool(s["is_buy"])),
                "bsp_main_type": s["bsp_main_type"],
                "label": int(s["label"]),
                "label_raw": int(s["label_raw"]),
                "label_text": s["label_text"],
                "entry_price": float(s["entry_price"]),
                "sl_price": float(s["sl_price"]),
                "pt_price": float(s["pt_price"]),
                "pt_multiplier_used": float(s.get("pt_multiplier_used", 0.0)),
                "volatility_value": s.get("volatility_value", None),
                "volatility_regime": str(s.get("volatility_regime", "")),
                "t0_pos": int(s["t0_pos"]),
                "t1_pos": int(s["t1_pos"]),
                "holding_bars": int(s["holding_bars"]),
                "sample_weight": float(s.get("sample_weight", 1.0)),
                "realized_return": float(s.get("realized_return", 0.0)),
            }
            for s in samples
        ]
    )
    rec.to_csv(out / "label_records_all.csv", index=False, encoding="utf-8")


def main():
    parser = build_parser()
    ns = parser.parse_args()

    cache_namespace = str(ns.cache_namespace or "").strip() or "default"
    cache_mode = str(ns.cache_mode or "resume").strip().lower()

    # 处理清除缓存指令
    if ns.clear_cache:
        if cache_namespace == "default":
            clear_all_caches()
            print("✓ 已清空所有缓存（全部命名空间）")
        else:
            clear_cache_namespace(cache_namespace)
            print(f"✓ 已清空缓存命名空间: {cache_namespace}")
        return

    if cache_mode == "fresh":
        clear_cache_namespace(cache_namespace)
        disable_label_cache = True
        disable_feature_cache = True
        print(
            f"[Cache] mode=fresh, namespace={cache_namespace} -> 已清空命名空间并禁用缓存"
        )
    else:
        disable_label_cache = bool(ns.disable_label_cache)
        disable_feature_cache = bool(ns.disable_feature_cache)
        print(
            "[Cache] mode=resume, "
            f"namespace={cache_namespace}, "
            f"label_cache={'off' if disable_label_cache else 'on'}, "
            f"feature_cache={'off' if disable_feature_cache else 'on'}"
        )

    symbols = normalize_symbols(ns.symbols) if ns.symbols else discover_symbols("5m")
    ensure_15m_parquet_from_5m(symbols)

    label_config = LabelConfig(
        pt_multiplier=max(0.5, float(ns.pt_multiplier)),
        dynamic_pt_enabled=bool(ns.dynamic_pt_enabled),
        pt_low_vol_multiplier=max(0.5, float(ns.pt_low_vol_multiplier)),
        pt_mid_vol_multiplier=max(0.5, float(ns.pt_mid_vol_multiplier)),
        pt_high_vol_multiplier=max(0.5, float(ns.pt_high_vol_multiplier)),
        vol_window=max(8, int(ns.pt_vol_window)),
        vol_quantile_low=min(0.98, max(0.01, float(ns.pt_vol_quantile_low))),
        vol_quantile_high=min(0.99, max(0.02, float(ns.pt_vol_quantile_high))),
        timeout_bars=max(2, int(ns.timeout_bars)),
        weak_timeout_bars=max(1, int(ns.weak_timeout_bars)),
        weak_bsp_types=ns.weak_bsp_types,
    )
    if label_config.vol_quantile_high <= label_config.vol_quantile_low:
        label_config.vol_quantile_high = min(0.99, label_config.vol_quantile_low + 0.05)

    feature_config = FeatureConfig(
        symbol_workers=max(1, int(ns.feature_symbol_workers)),
    )
    model_config = ModelConfig(
        primary_model_type=ns.primary_model,
        meta_model_type=ns.meta_model,
        meta_threshold=min(0.95, max(0.05, float(ns.meta_threshold))),
        meta_calibration=str(ns.meta_calibration),
        meta_calibration_ratio=min(0.5, max(0.05, float(ns.meta_calibration_ratio))),
    )
    validator_config = TrainValidatorConfig(
        n_splits=max(3, int(ns.cv_splits)),
        embargo_bars=max(0, int(ns.embargo_bars)),
        optuna_trials=max(0, int(ns.optuna_trials)),
        pass_positive_month_ratio=min(1.0, max(0.0, float(ns.pass_positive_month_ratio))),
        mda_max_samples=max(1000, int(ns.mda_max_samples)),
        mda_n_jobs=max(1, int(ns.mda_n_jobs)),
    )

    output_dir = ns.output_dir
    os.makedirs(output_dir, exist_ok=True)
    snapshots = snapshot_architecture_docs(output_dir)

    print("\n" + "★" * 60)
    print(" L3+L4 重构训练: LabelEngine + FeatureEngine + TrainValidator ")
    print(f" 数据源={DATA_SRC_TYPE.name}, 周期={ns.kl_type}")
    print(f" 区间={ns.begin_time} ~ {ns.end_time}")
    print(f" 标注策略={ns.labeling_strategy}, 名称={ns.labeling_name}")
    print(f" 二分类标注: pt={label_config.pt_multiplier}, labels={{SL(-1), PT(+1)}}")
    print(
        f" 动态PT: enabled={label_config.dynamic_pt_enabled}, "
        f"low/mid/high={label_config.pt_low_vol_multiplier:.2f}/"
        f"{label_config.pt_mid_vol_multiplier:.2f}/"
        f"{label_config.pt_high_vol_multiplier:.2f}"
    )
    print(f" 验证: folds={validator_config.n_splits}, embargo={validator_config.embargo_bars}")
    print(
        f" Meta校准: method={model_config.meta_calibration}, "
        f"ratio={model_config.meta_calibration_ratio:.2f}, "
        f"pass_positive_month_ratio={validator_config.pass_positive_month_ratio:.2f}"
    )
    print("★" * 60)

    t_start = time.time()
    args_lv = [kl_type_from_text(ns.kl_type)]

    # 初始化缓存
    label_cache = get_label_cache(
        enable=not disable_label_cache,
        namespace=cache_namespace,
    )
    feature_cache = get_feature_cache(
        enable=not disable_feature_cache,
        namespace=cache_namespace,
    )
    label_config_hash = _config_hash(label_config)
    feature_config_hash = _config_hash(feature_config)
    
    bars_by_symbol: Dict[str, List[Dict]] = {}
    labeled_all: List[Dict] = []
    label_engine = LabelEngine(label_config)

    # 续跑快路径：标签和特征缓存都命中的symbol，直接复用，跳过事件采集。
    fast_cached_symbols: List[str] = []
    fast_cached_set = set()
    fast_cached_labels_by_symbol: Dict[str, List[Dict]] = {}
    fast_cached_features_by_symbol: Dict[str, pd.DataFrame] = {}
    if label_cache.enable and feature_cache.enable:
        for sym in symbols:
            label_path = label_cache.get_cache_path(sym, label_config_hash)
            feature_path = feature_cache.get_cache_path(sym, feature_config_hash)
            if not (label_path.exists() and feature_path.exists()):
                continue
            cached_labels = label_cache.load(sym, label_config_hash)
            cached_features = feature_cache.load(sym, feature_config_hash)
            if cached_labels is None or cached_features is None:
                continue
            if len(cached_features) != len(cached_labels):
                print(
                    f"[CacheFast][WARN] {sym} 标签与特征缓存行数不一致 "
                    f"(labels={len(cached_labels)}, features={len(cached_features)}), "
                    "回退到常规路径"
                )
                continue
            labeled_all.extend(cached_labels)
            fast_cached_symbols.append(sym)
            fast_cached_set.add(sym)
            fast_cached_labels_by_symbol[sym] = cached_labels
            fast_cached_features_by_symbol[sym] = cached_features.reset_index(drop=True)
            print(f"[完成] {sym}: events=0, labeled={len(cached_labels)} (cache-fast)")

        if fast_cached_symbols:
            print(
                f"[CacheFast] 已跳过事件采集并复用标签+特征缓存: "
                f"{len(fast_cached_symbols)}/{len(symbols)} symbols"
            )

    symbols_to_collect = [sym for sym in symbols if sym not in fast_cached_set]

    if symbols_to_collect:
        with ProcessPoolExecutor(
            max_workers=min(max(1, int(ns.num_workers)), len(symbols_to_collect))
        ) as ex:
            jobs = [
                ex.submit(
                    collect_symbol_events_worker,
                    sym,
                    ns.begin_time,
                    ns.end_time,
                    args_lv,
                    CHAN_CONFIG,
                )
                for sym in symbols_to_collect
            ]
            for fut in as_completed(jobs):
                res = fut.result()
                symbol = res["symbol"]
                bars = res["bars"]
                bars_by_symbol[symbol] = bars

                # 尝试从缓存加载标签
                cached_labels = label_cache.load(symbol, label_config_hash)
                if cached_labels is not None:
                    labeled = cached_labels
                    label_src = "cache"
                else:
                    labeled = label_engine.transform(res["events"], bars)
                    label_cache.save(symbol, label_config_hash, labeled)
                    label_src = "compute"

                labeled_all.extend(labeled)
                print(
                    f"[完成] {symbol}: events={len(res['events'])}, "
                    f"labeled={len(labeled)} ({label_src})"
                )
    else:
        print("[CacheFast] 所有symbol均命中标签+特征缓存，跳过事件采集阶段")

    if len(labeled_all) < 300:
        raise RuntimeError("样本不足，建议扩大时间范围或币种数量")

    labeled_all = sort_samples(labeled_all)

    # 使用支持缓存的 FeatureEngine。
    # 当全部symbol命中cache-fast时，直接拼接特征缓存并做安全标准化，
    # 避免重复索引在normalizer里发生标签扩容导致内存爆炸。
    if (
        not symbols_to_collect
        and len(fast_cached_set) == len(symbols)
        and fast_cached_features_by_symbol
    ):
        merged_feature_frames: List[pd.DataFrame] = []
        for sym in symbols:
            labels_sym = fast_cached_labels_by_symbol.get(sym)
            features_sym = fast_cached_features_by_symbol.get(sym)
            if labels_sym is None or features_sym is None:
                raise RuntimeError(f"cache-fast缺少symbol缓存内容: {sym}")
            if len(features_sym) != len(labels_sym):
                raise RuntimeError(
                    f"cache-fast行数不一致: {sym}, "
                    f"labels={len(labels_sym)}, features={len(features_sym)}"
                )

            frame = features_sym.copy()
            frame["__t0_ts"] = [float(item["t0_ts"]) for item in labels_sym]
            frame["__symbol"] = [str(item["symbol"]) for item in labels_sym]
            frame["__klu_idx"] = [int(item["klu_idx"]) for item in labels_sym]
            merged_feature_frames.append(frame)

        merged_feature_df = pd.concat(merged_feature_frames, ignore_index=True)
        merged_feature_df = merged_feature_df.sort_values(
            ["__t0_ts", "__symbol", "__klu_idx"],
            kind="mergesort",
        ).reset_index(drop=True)

        feature_df = merged_feature_df.drop(columns=["__t0_ts", "__symbol", "__klu_idx"])
        if len(feature_df) != len(labeled_all):
            raise RuntimeError(
                "cache-fast样本与特征数量不一致: "
                f"samples={len(labeled_all)}, features={len(feature_df)}"
            )

        feature_engine = CachedFeatureEngine(
            feature_config,
            enable_cache=not disable_feature_cache,
            cache_namespace=cache_namespace,
        )
        feature_df = feature_engine.normalizer.fit_transform(
            feature_df.reset_index(drop=True),
            ordered_index=None,
        )
        print("[CacheFast] 全量命中时已直接拼接特征缓存并完成标准化")
    else:
        feature_engine = CachedFeatureEngine(
            feature_config,
            enable_cache=not disable_feature_cache,
            cache_namespace=cache_namespace,
        )
        feature_df = feature_engine.transform(labeled_all, bars_by_symbol, normalize=True)

    y = np.array([int(s["label"]) for s in labeled_all], dtype=np.int32)
    w = np.array([float(s.get("sample_weight", 1.0)) for s in labeled_all], dtype=np.float32)
    t0_pos, t1_pos = build_global_pos(labeled_all)
    rets = np.array([float(s.get("realized_return", 0.0)) for s in labeled_all], dtype=np.float32)
    X = feature_df.to_numpy(dtype=np.float32)

    feature_names = list(feature_df.columns)
    feature_meta = {name: idx for idx, name in enumerate(feature_names)}

    trainer = TrainValidator(model_config=model_config, validator_config=validator_config)
    report, mdi_df, mda_df, selected, oos_cls, oos_exec = trainer.fit(
        X=X,
        y=y,
        sample_weight=w,
        t0_pos=t0_pos,
        t1_pos=t1_pos,
        returns=rets,
        feature_names=feature_names,
        train_mode=ns.train_mode,
    )
    report["optimization_objective"] = BacktestObjectiveConfig().__dict__
    report["cache"] = {
        "mode": cache_mode,
        "namespace": cache_namespace,
        "label_cache_enabled": bool(not disable_label_cache),
        "feature_cache_enabled": bool(not disable_feature_cache),
    }

    visual_reports: Dict[str, str] = {}

    p_all = trainer.primary_model.predict_proba(X)
    c_all = np.argmax(p_all, axis=1)
    sig_all = np.ones(len(c_all), dtype=bool)

    # 组织可视化指标：补充 AUC/LogLoss/覆盖率等。
    label_total = max(1, int(len(y)))
    label_pt_ratio = float(np.sum(y == 1) / label_total)
    label_sl_ratio = float(np.sum(y == 0) / label_total)

    primary_auc_ovr_macro = 0.0
    primary_logloss = 0.0
    try:
        if len(np.unique(y)) >= 2:
            p_pos = p_all[:, -1] if p_all.ndim == 2 else p_all
            primary_auc_ovr_macro = float(roc_auc_score(y, p_pos))
            primary_logloss = float(log_loss(y, np.clip(p_pos, 1e-8, 1.0), labels=[0, 1]))
    except Exception:
        pass

    primary_signal_precision = 0.0
    if np.any(sig_all):
        primary_signal_precision = float(np.mean((c_all[sig_all] == y[sig_all]).astype(float)))

    primary_visual_metrics = {
        "oos_sharpe": float(report["oos_metrics"].get("sharpe", 0.0)),
        "oos_precision": float(report["oos_metrics"].get("precision", 0.0)),
        "oos_macro_f1": float(report["oos_metrics"].get("macro_f1", 0.0)),
        "primary_auc_ovr_macro": float(primary_auc_ovr_macro),
        "primary_logloss": float(primary_logloss),
        "primary_signal_coverage": float(np.mean(sig_all.astype(float))),
        "primary_signal_precision": float(primary_signal_precision),
        "label_pt_ratio": label_pt_ratio,
        "label_sl_ratio": label_sl_ratio,
    }

    # Primary模型（XGBoost）可视化。
    try:
        shap_analyzer = SHAPAnalyzer(trainer.primary_model.booster, feature_names)
        shap_X, shap_y = sample_for_shap(X, (y == 1).astype(int), max(500, int(ns.shap_sample_limit)))
        shap_result = shap_analyzer.analyze(shap_X, y=shap_y)
        shap_report_path = Path(output_dir) / "primary_shap_report.html"
        shap_analyzer.generate_report(
            shap_result,
            output_path=str(shap_report_path),
            model_metrics=primary_visual_metrics,
        )
        visual_reports["primary_shap_report"] = str(shap_report_path)
    except Exception as exc:
        print(f"[WARN] Primary SHAP报告生成失败: {exc}")

    # Meta模型可视化：扩展指标 + 阈值敏感性 + 结果解读。
    try:
        if np.any(sig_all):
            meta_X = np.hstack([np.nan_to_num(X[sig_all], nan=0.0), p_all[sig_all]])
            meta_y = (c_all[sig_all] == y[sig_all]).astype(int)
            meta_prob_cols = [f"p_cls_{i}" for i in range(p_all.shape[1])] if p_all.ndim == 2 else ["p_cls_1"]
            meta_feature_names = feature_names + meta_prob_cols
            meta_report_path = Path(output_dir) / "meta_visual_report.html"
            generate_meta_model_visual_report(
                output_path=str(meta_report_path),
                meta_model=trainer.meta_model,
                meta_X=meta_X,
                meta_y=meta_y,
                feature_names=meta_feature_names,
                threshold=model_config.meta_threshold,
                context_metrics={
                    "oos_metrics": report.get("oos_metrics", {}),
                    "fold_summary": report.get("fold_summary", {}),
                    "primary_metrics": primary_visual_metrics,
                    "meta_threshold": float(model_config.meta_threshold),
                },
            )
            visual_reports["meta_visual_report"] = str(meta_report_path)
    except Exception as exc:
        print(f"[WARN] Meta可视化报告生成失败: {exc}")

    # 双模型诊断：Primary/Meta偏置、分桶收益、阈值敏感性、校准、误杀漏放、依赖集中度。
    try:
        model_exec_prob = np.zeros(len(y), dtype=np.float32)
        if trainer.meta_model is not None:
            model_exec_prob = trainer.meta_model.predict_proba(
                np.hstack([np.nan_to_num(X, nan=0.0), p_all])
            )

        samples_df = pd.DataFrame(
            {
                "symbol": [str(s["symbol"]) for s in labeled_all],
                "open_time": [str(s["open_time"]) for s in labeled_all],
                "bsp_main_type": [str(s.get("bsp_main_type", "")) for s in labeled_all],
                "realized_return": [float(s.get("realized_return", 0.0)) for s in labeled_all],
            }
        )
        diagnostics = build_dual_model_diagnostics(
            samples_df=samples_df,
            y=y,
            primary_probs=p_all,
            primary_pred=c_all,
            meta_exec_prob=model_exec_prob,
            meta_threshold=float(model_config.meta_threshold),
            meta_model=trainer.meta_model,
            base_feature_names=feature_names,
        )

        positive_month_ratio = float(
            diagnostics.get("primary", {})
            .get("time_stability", {})
            .get("positive_month_ratio", 0.0)
            or 0.0
        )
        stability_passed = bool(
            positive_month_ratio >= float(validator_config.pass_positive_month_ratio)
        )
        pass_block = report.get("pass_criteria", {})
        base_passed = bool(pass_block.get("passed", False))
        pass_block["stability_positive_month_ratio"] = positive_month_ratio
        pass_block["stability_threshold"] = float(validator_config.pass_positive_month_ratio)
        pass_block["stability_passed"] = stability_passed
        pass_block["base_passed"] = base_passed
        pass_block["passed"] = bool(base_passed and stability_passed)
        pass_block["rule"] = (
            f"{pass_block.get('rule', '')} and "
            f"positive_month_ratio>={validator_config.pass_positive_month_ratio}"
        )
        report["pass_criteria"] = pass_block

        diag_path = Path(output_dir) / "dual_model_diagnostics.json"
        diag_path.write_text(json.dumps(diagnostics, indent=2, ensure_ascii=False), encoding="utf-8")
        visual_reports["dual_model_diagnostics"] = str(diag_path)

        primary_buckets = diagnostics.get("primary", {}).get("confidence_bucket_returns", [])
        if isinstance(primary_buckets, list) and primary_buckets:
            pd.DataFrame(primary_buckets).to_csv(
                Path(output_dir) / "primary_confidence_bucket_returns.csv",
                index=False,
                encoding="utf-8",
            )

        meta_scan = diagnostics.get("meta", {}).get("threshold_scan", [])
        if isinstance(meta_scan, list) and meta_scan:
            pd.DataFrame(meta_scan).to_csv(
                Path(output_dir) / "meta_threshold_sensitivity.csv",
                index=False,
                encoding="utf-8",
            )
    except Exception as exc:
        print(f"[WARN] 双模型诊断报告生成失败: {exc}")

    save_outputs(
        output_dir=output_dir,
        feature_meta=feature_meta,
        report=report,
        mdi_df=mdi_df,
        mda_df=mda_df,
        selected_features=selected,
        primary_model=trainer.primary_model,
        meta_model=trainer.meta_model,
        samples=labeled_all,
        architecture_snapshots=snapshots,
        visual_reports=visual_reports,
    )

    print("\n" + "=" * 60)
    print("[✓ 完成] L3+L4 重构训练完成")
    print(f"[样本] total={len(labeled_all)}")
    print(
        f"[OOS] sharpe={report['oos_metrics']['sharpe']:.4f}, "
        f"precision={report['oos_metrics']['precision']:.4f}, "
        f"macro_f1={report['oos_metrics']['macro_f1']:.4f}"
    )
    
    # 打印缓存统计
    try:
        stats = cache_stats(namespace=cache_namespace)
        print(f"[缓存] 标签: hits={stats['labels']['hits']}, misses={stats['labels']['misses']}, "
              f"hit_rate={stats['labels']['hit_rate']}, size={stats['labels']['total_size_mb']}MB")
        print(f"       特征: hits={stats['features']['hits']}, misses={stats['features']['misses']}, "
              f"hit_rate={stats['features']['hit_rate']}, size={stats['features']['total_size_mb']}MB")
    except Exception as e:
        print(f"[缓存] 统计获取失败: {e}")
    
    print(f"[产物] {output_dir}")
    print(f"[耗时] {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    mp.freeze_support()
    _log_fp = None
    _log_path = ""
    try:
        _log_fp, _log_path = setup_train_log()
        main()
    finally:
        close_train_log(_log_fp, _log_path)
