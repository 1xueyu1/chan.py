"""
统一的全流程管理脚本，支持两种模式：
1. 全流程模式（train → predict → backtest）
2. 预测+回测模式（快速迭代，复用已有训练模型）
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List


_CPU_COUNT = os.cpu_count() or 4
_DEFAULT_NUM_WORKERS = max(1, _CPU_COUNT - 1)
_DEFAULT_SYMBOL_WORKERS = max(1, min(8, _CPU_COUNT // 2))


def _default_run_id() -> str:
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "统一的全流程管理脚本：支持完整训练→预测→回测，"
            "或者快速预测+回测模式（复用已有训练模型）"
        ),
    )

    # 基础参数
    parser.add_argument(
        "--mode",
        choices=["full", "predict-backtest"],
        default="full",
        help="运行模式：full=完整流程(训练→预测→回测)，predict-backtest=仅预测+回测",
    )
    parser.add_argument(
        "--stage",
        choices=["train", "predict", "backtest"],
        default=None,
        help="（可选）仅运行指定阶段。不指定时按 --mode 运行完整流程。",
    )
    parser.add_argument(
        "--run-id",
        default="",
        help="run 文件夹名；为空时自动生成时间戳",
    )
    parser.add_argument(
        "--debug-root",
        default="Debug",
        help="debug 根目录",
    )

    # 训练参数（仅在 mode=full 时使用）
    parser.add_argument(
        "--train-symbols",
        nargs="+",
        default=["ADAUSDT", "AVAXUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT", "DOTUSDT", "ETHUSDT", "LTCUSDT"],
        help="训练币种（默认8个币种进行训练）",
    )
    parser.add_argument(
        "--test-symbols",
        nargs="+",
        default=["SOLUSDT", "XRPUSDT"],
        help="测试币种（默认2个币种用于验证）",
    )

    # 时间参数
    parser.add_argument(
        "--begin-time",
        default="2024-01-01",
        help="训练/预测/回测的开始时间",
    )
    parser.add_argument(
        "--end-time",
        default="2026-01-01",
        help="训练/预测/回测的结束时间",
    )

    # 训练模式参数
    parser.add_argument(
        "--train-mode",
        choices=["cpu", "gpu", "auto"],
        default="auto",
        help="训练设备模式",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=_DEFAULT_NUM_WORKERS,
        help="训练阶段并行采样进程数（默认按CPU自动设置）",
    )
    parser.add_argument(
        "--labeling-strategy",
        choices=[
            "original_return",
            "chan_structure_invalidation",
            "multicycle_resonance",
            "divergence_focus",
            "ensemble4_consensus",
        ],
        default="original_return",
        help="训练标注策略",
    )
    parser.add_argument(
        "--labeling-name",
        default="ChanLabelingSuite_v2",
        help="标注方案名称（写入训练指标）",
    )
    parser.add_argument(
        "--ensemble-min-votes",
        type=int,
        default=3,
        help="ensemble4_consensus策略下的最小共识票数",
    )
    parser.add_argument(
        "--label-max-holding-bars",
        type=int,
        default=96,
        help="结构标注中允许的最大持有bar数",
    )

    # 预测参数
    parser.add_argument(
        "--predict-symbol",
        default="BTCUSDT",
        help="生成 SHAP 解释报告的币种",
    )

    # 回测参数
    parser.add_argument(
        "--backtest-symbols",
        nargs="+",
        default=["ADAUSDT", "AVAXUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT", "DOTUSDT", "ETHUSDT", "LTCUSDT", "SOLUSDT", "XRPUSDT"],
        help="回测使用的全量币种（默认10个币种）",
    )
    parser.add_argument(
        "--symbol-workers",
        type=int,
        default=_DEFAULT_SYMBOL_WORKERS,
        help="回测时按symbol并行worker数（默认按CPU自动设置）",
    )
    parser.add_argument(
        "--backtest-parallel-mode",
        choices=["process", "thread"],
        default="process",
        help="回测symbol并行模式（默认process）",
    )
    parser.add_argument(
        "--backtest-fast-mode",
        action="store_true",
        default=True,
        help="回测快速模式：禁用非必要输出与HTML，提升速度（默认开启）",
    )
    parser.add_argument(
        "--backtest-full-artifacts",
        action="store_true",
        help="关闭fast-mode，保留events/bars/html等完整产物",
    )

    # 信号阈值
    parser.add_argument(
        "--signal-threshold",
        type=float,
        default=0.55,
        help="买卖点判定阈值",
    )

    # 报告参数
    parser.add_argument(
        "--save-html-detail-report",
        action="store_true",
        help="回测时额外生成详细的 HTML 报告",
    )
    parser.add_argument(
        "--enable-predict",
        action="store_true",
        help="在full模式下启用predict阶段（默认full只做训练+回测）",
    )

    # 复用已有训练（predict-backtest 模式专用）
    parser.add_argument(
        "--train-dir",
        default="",
        help="（predict-backtest 模式）已有的训练产物目录，包含 model_*.json / meta_*.json",
    )
    parser.add_argument(
        "--source-run-id",
        default="",
        help="（predict-backtest 模式）源 run_id，自动指向 Debug/runs/<source-run-id>/train",
    )

    # 跳过参数
    parser.add_argument("--skip-predict", action="store_true", help="跳过预测阶段")
    parser.add_argument("--skip-backtest", action="store_true", help="跳过回测阶段")

    return parser


def _run_cmd(
    cmd: List[str],
    log_path: Path,
    extra_env: Dict[str, str] | None = None,
) -> float:
    """执行命令并返回耗时（秒）"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    start_time = time.time()
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write("$ " + " ".join(cmd) + "\n\n")
        base_env = dict(os.environ)
        # Ensure child process stdout uses UTF-8 on Windows to avoid GBK encoding errors.
        base_env["PYTHONIOENCODING"] = "utf-8"
        base_env["PYTHONUTF8"] = "1"
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            env={**base_env, **extra_env} if extra_env else base_env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        rc = process.wait()
        if rc != 0:
            raise RuntimeError(f"命令失败 (exit={rc}): {' '.join(cmd)}")
    elapsed = time.time() - start_time
    return elapsed


def _write_manifest(path: Path, payload: Dict[str, object]) -> None:
    """写入 manifest 文件"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _run_full_pipeline(args: argparse.Namespace) -> None:
    """完整流程：训练 → 预测 → 回测"""
    run_id = args.run_id or _default_run_id()

    root = Path(args.debug_root) / "runs" / run_id
    train_dir = root / "train"
    predict_dir = root / "predict"
    backtest_dir = root / "backtest"
    logs_dir = root / "logs"

    for d in [train_dir, predict_dir, backtest_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    python_exec = sys.executable
    project_root = Path(__file__).resolve().parents[1]

    model_buy = train_dir / "model_buy.json"
    model_sell = train_dir / "model_sell.json"
    meta_buy = train_dir / "meta_buy.json"
    meta_sell = train_dir / "meta_sell.json"

    # 构建训练命令
    train_cmd = [
        python_exec,
        str(project_root / "Debug" / "xgboost_shap_train.py"),
        "--symbols",
        *args.train_symbols,
        "--begin-time",
        args.begin_time,
        "--end-time",
        args.end_time,
        "--train-mode",
        args.train_mode,
        "--num-workers",
        str(args.num_workers),
        "--labeling-strategy",
        args.labeling_strategy,
        "--labeling-name",
        args.labeling_name,
        "--ensemble-min-votes",
        str(args.ensemble_min_votes),
        "--label-max-holding-bars",
        str(args.label_max_holding_bars),
        "--output-dir",
        str(train_dir),
    ]

    timing_stats: Dict[str, float] = {}
    pipeline_start = time.time()

    print(f"[Pipeline] run_id={run_id}")
    print("[Stage] train（训练模型）")
    train_env = {
        "XGB_TRAIN_LOG_DIR": str(logs_dir),
    }
    merged_train_env = dict(os.environ)
    merged_train_env.update(train_env)
    train_elapsed = _run_cmd(train_cmd, logs_dir / "train.log", merged_train_env)
    timing_stats["train"] = train_elapsed
    print(f"[Timing] train: {train_elapsed:.2f}s (即 {train_elapsed/60:.2f} 分钟)\n")

    # 预测阶段
    predict_cmd: List[str] = []
    predict_elapsed = 0.0
    if not args.skip_predict:
        predict_cmd = [
            python_exec,
            str(project_root / "Debug" / "xgboost_shap_predict.py"),
            "--code",
            args.predict_symbol,
            "--begin-time",
            args.begin_time,
            "--end-time",
            args.end_time,
            "--model-buy-path",
            str(model_buy),
            "--model-sell-path",
            str(model_sell),
            "--meta-buy-path",
            str(meta_buy),
            "--meta-sell-path",
            str(meta_sell),
            "--signal-threshold",
            str(args.signal_threshold),
            "--output-dir",
            str(predict_dir),
            "--report-path",
            str(predict_dir / "shap_predict_report.html"),
        ]
        print("[Stage] predict（预测和 SHAP 解释）")
        predict_elapsed = _run_cmd(predict_cmd, logs_dir / "predict.log")
        timing_stats["predict"] = predict_elapsed
        print(f"[Timing] predict: {predict_elapsed:.2f}s (即 {predict_elapsed/60:.2f} 分钟)\n")

    # 回测阶段
    backtest_cmd: List[str] = []
    backtest_elapsed = 0.0
    if not args.skip_backtest:
        backtest_cmd = [
            python_exec,
            str(
                project_root
                / "Backtest"
                / "examples"
                / "run_vectorbt_backtest.py"
            ),
            "--symbols",
            *args.backtest_symbols,
            "--begin-time",
            args.begin_time,
            "--end-time",
            args.end_time,
            "--model-buy-path",
            str(model_buy),
            "--model-sell-path",
            str(model_sell),
            "--meta-buy-path",
            str(meta_buy),
            "--meta-sell-path",
            str(meta_sell),
            "--signal-threshold",
            str(args.signal_threshold),
            "--output-dir",
            str(backtest_dir),
            "--symbol-workers",
            str(args.symbol_workers),
            "--parallel-mode",
            args.backtest_parallel_mode,
        ]
        if args.backtest_fast_mode and not args.backtest_full_artifacts:
            backtest_cmd.append("--fast-mode")
        if args.save_html_detail_report:
            backtest_cmd.append("--save-html-detail-report")

        print("[Stage] backtest（全量回测，包括10个币种）")
        backtest_elapsed = _run_cmd(backtest_cmd, logs_dir / "backtest.log")
        timing_stats["backtest"] = backtest_elapsed
        print(f"[Timing] backtest: {backtest_elapsed:.2f}s (即 {backtest_elapsed/60:.2f} 分钟)\n")

    pipeline_total = time.time() - pipeline_start

    manifest = {
        "run_id": run_id,
        "mode": "full",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "train_config": {
            "train_symbols": args.train_symbols,
            "test_symbols": args.test_symbols,
            "time_range": {
                "begin": args.begin_time,
                "end": args.end_time,
            },
            "train_mode": args.train_mode,
            "num_workers": args.num_workers,
            "labeling_strategy": args.labeling_strategy,
            "labeling_name": args.labeling_name,
            "ensemble_min_votes": args.ensemble_min_votes,
            "label_max_holding_bars": args.label_max_holding_bars,
        },
        "backtest_config": {
            "symbols": args.backtest_symbols,
            "symbol_workers": args.symbol_workers,
            "parallel_mode": args.backtest_parallel_mode,
            "fast_mode": bool(args.backtest_fast_mode and not args.backtest_full_artifacts),
            "signal_threshold": args.signal_threshold,
        },
        "paths": {
            "root": str(root),
            "train": str(train_dir),
            "predict": str(predict_dir),
            "backtest": str(backtest_dir),
            "logs": str(logs_dir),
        },
        "commands": {
            "train": train_cmd,
            "predict": predict_cmd,
            "backtest": backtest_cmd,
        },
        "artifacts": {
            "model_buy": str(model_buy),
            "model_sell": str(model_sell),
            "meta_buy": str(meta_buy),
            "meta_sell": str(meta_sell),
            "predict_report": str(predict_dir / "shap_predict_report.html"),
            "backtest_metrics": str(backtest_dir / "backtest_metrics.json"),
            "backtest_report": str(backtest_dir / "xgb_backtest_report.html"),
            "backtest_report_detail": str(
                backtest_dir / "xgb_backtest_report_detail.html"
            ),
        },
        "timing_seconds": {
            "train": round(timing_stats.get("train", 0), 2),
            "predict": round(timing_stats.get("predict", 0), 2),
            "backtest": round(timing_stats.get("backtest", 0), 2),
            "total": round(pipeline_total, 2),
        },
    }
    _write_manifest(root / "run_manifest.json", manifest)

    print("\n" + "=" * 60)
    print("[✓ 完成] 全流程执行完毕")
    print("=" * 60)
    print(f"[Artifacts Root] {root}")
    print(f"[Manifest] {root / 'run_manifest.json'}")
    print("\n=== 耗时统计 ===")
    for stage, elapsed in timing_stats.items():
        print(f"  {stage}: {elapsed:.2f}s ({elapsed/60:.2f}min)")
    print(f"  total: {pipeline_total:.2f}s ({pipeline_total/3600:.2f}h)")


def _run_predict_backtest_only(args: argparse.Namespace) -> None:
    """快速模式：仅预测 + 回测（复用已有训练模型）"""
    run_id = args.run_id or _default_run_id()

    # 解析训练目录
    if args.train_dir:
        train_dir = Path(args.train_dir)
    elif args.source_run_id:
        train_dir = Path(args.debug_root) / "runs" / args.source_run_id / "train"
    else:
        raise ValueError("predict-backtest 模式需要 --train-dir 或 --source-run-id 之一")

    model_buy = train_dir / "model_buy.json"
    model_sell = train_dir / "model_sell.json"
    meta_buy = train_dir / "meta_buy.json"
    meta_sell = train_dir / "meta_sell.json"
    required_files = [model_buy, model_sell, meta_buy, meta_sell]
    missing = [str(p) for p in required_files if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "缺失必需的训练产物: " + "; ".join(missing)
        )

    root = Path(args.debug_root) / "runs" / run_id
    predict_dir = root / "predict"
    backtest_dir = root / "backtest"
    logs_dir = root / "logs"
    for d in [predict_dir, backtest_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).resolve().parents[1]
    python_exec = sys.executable

    timing_stats: Dict[str, float] = {}
    pipeline_start = time.time()

    print(f"[Pipeline] run_id={run_id} (predict-backtest 模式)")
    print(f"[复用训练模型] {train_dir}")

    # 预测阶段
    predict_cmd: List[str] = []
    predict_elapsed = 0.0
    if not args.skip_predict:
        predict_cmd = [
            python_exec,
            str(project_root / "Debug" / "xgboost_shap_predict.py"),
            "--code",
            args.predict_symbol,
            "--begin-time",
            args.begin_time,
            "--end-time",
            args.end_time,
            "--model-buy-path",
            str(model_buy),
            "--model-sell-path",
            str(model_sell),
            "--meta-buy-path",
            str(meta_buy),
            "--meta-sell-path",
            str(meta_sell),
            "--signal-threshold",
            str(args.signal_threshold),
            "--output-dir",
            str(predict_dir),
            "--report-path",
            str(predict_dir / "shap_predict_report.html"),
        ]
        print("[Stage] predict（预测和 SHAP 解释）")
        predict_elapsed = _run_cmd(predict_cmd, logs_dir / "predict.log")
        timing_stats["predict"] = predict_elapsed
        print(f"[Timing] predict: {predict_elapsed:.2f}s (即 {predict_elapsed/60:.2f} 分钟)\n")

    # 回测阶段
    backtest_cmd: List[str] = []
    backtest_elapsed = 0.0
    if not args.skip_backtest:
        backtest_cmd = [
            python_exec,
            str(
                project_root
                / "Backtest"
                / "examples"
                / "run_vectorbt_backtest.py"
            ),
            "--symbols",
            *args.backtest_symbols,
            "--begin-time",
            args.begin_time,
            "--end-time",
            args.end_time,
            "--model-buy-path",
            str(model_buy),
            "--model-sell-path",
            str(model_sell),
            "--meta-buy-path",
            str(meta_buy),
            "--meta-sell-path",
            str(meta_sell),
            "--signal-threshold",
            str(args.signal_threshold),
            "--output-dir",
            str(backtest_dir),
            "--symbol-workers",
            str(args.symbol_workers),
            "--parallel-mode",
            args.backtest_parallel_mode,
        ]
        if args.backtest_fast_mode and not args.backtest_full_artifacts:
            backtest_cmd.append("--fast-mode")
        if args.save_html_detail_report:
            backtest_cmd.append("--save-html-detail-report")

        print("[Stage] backtest（全量回测，包括10个币种）")
        backtest_elapsed = _run_cmd(backtest_cmd, logs_dir / "backtest.log")
        timing_stats["backtest"] = backtest_elapsed
        print(f"[Timing] backtest: {backtest_elapsed:.2f}s (即 {backtest_elapsed/60:.2f} 分钟)\n")

    pipeline_total = time.time() - pipeline_start

    manifest = {
        "run_id": run_id,
        "mode": "predict-backtest",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "train_dir": str(train_dir),
        "backtest_config": {
            "symbols": args.backtest_symbols,
            "symbol_workers": args.symbol_workers,
            "signal_threshold": args.signal_threshold,
        },
        "paths": {
            "root": str(root),
            "predict": str(predict_dir),
            "backtest": str(backtest_dir),
            "logs": str(logs_dir),
        },
        "commands": {
            "predict": predict_cmd,
            "backtest": backtest_cmd,
        },
        "timing_seconds": {
            "predict": round(timing_stats.get("predict", 0), 2),
            "backtest": round(timing_stats.get("backtest", 0), 2),
            "total": round(pipeline_total, 2),
        },
    }
    _write_manifest(root / "run_manifest_predict_backtest.json", manifest)

    print("\n" + "=" * 60)
    print("[✓ 完成] 预测+回测执行完毕")
    print("=" * 60)
    print(f"[Artifacts Root] {root}")
    print(f"[Manifest] {root / 'run_manifest_predict_backtest.json'}")
    print("\n=== 耗时统计 ===")
    for stage, elapsed in timing_stats.items():
        print(f"  {stage}: {elapsed:.2f}s ({elapsed/60:.2f}min)")
    print(f"  total: {pipeline_total:.2f}s ({pipeline_total/3600:.2f}h)")


def main() -> None:
    args = _build_parser().parse_args()

    # 默认full流程为训练+回测，predict按需显式开启。
    if args.mode == "full" and args.stage is None and not args.enable_predict:
        args.skip_predict = True

    # Stage 优先于 mode：允许通过统一脚本独立执行任一阶段
    if args.stage is not None:
        if args.stage == "train":
            args.mode = "full"
            args.skip_predict = True
            args.skip_backtest = True
            _run_full_pipeline(args)
            return

        if args.stage == "predict":
            args.mode = "predict-backtest"
            args.skip_backtest = True
            _run_predict_backtest_only(args)
            return

        if args.stage == "backtest":
            args.mode = "predict-backtest"
            args.skip_predict = True
            _run_predict_backtest_only(args)
            return

    if args.mode == "full":
        _run_full_pipeline(args)
    elif args.mode == "predict-backtest":
        _run_predict_backtest_only(args)
    else:
        raise ValueError(f"未知模式: {args.mode}")


if __name__ == "__main__":
    main()
