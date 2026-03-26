"""
统一的全流程管理脚本，支持两种模式：
1. 全流程模式（train → predict → backtest）
2. 预测+回测模式（快速迭代，复用已有训练模型）

架构维护约定：
    - 修改本脚本中与训练/标签/回测参数语义相关逻辑时，
        需同步更新 Debug/MODEL_TRAINING_ARCHITECTURE.md。
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

try:
    import psutil
except ImportError:
    psutil = None  # Graceful fallback if psutil not installed


_CPU_COUNT = os.cpu_count() or 4
_DEFAULT_NUM_WORKERS = max(1, _CPU_COUNT - 1)
_DEFAULT_SYMBOL_WORKERS = max(1, min(8, _CPU_COUNT // 2))
_DEFAULT_FEATURE_SYMBOL_WORKERS = max(1, min(8, _DEFAULT_NUM_WORKERS // 2))
_MAX_TRAIN_TIME_SEC = 4 * 3600  # 4小时超时
_MAX_BACKTEST_TIME_SEC = 3 * 3600  # 3小时超时
_MIN_AVAILABLE_MEMORY_GB = 2.0  # 最少保留2GB内存  
_BATCH_STRATEGIES = [
    "trainvalidator_hierarchical",
]


def _check_system_resources(stage: str) -> None:
    """检查系统资源，不足时告警"""
    if not psutil:
        return
    mem = psutil.virtual_memory()
    available_gb = mem.available / (1024 ** 3)
    used_pct = mem.percent
    
    if available_gb < _MIN_AVAILABLE_MEMORY_GB:
        print(
            f"[WARN] {stage} 可用内存仅 {available_gb:.2f}GB"
            f"（低于建议 {_MIN_AVAILABLE_MEMORY_GB}GB），"
            f"建议关闭其他程序"
        )
    if used_pct > 90:
        print(f"[WARN] {stage} 内存使用率 {used_pct:.1f}%（>90%），可能影响性能")


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
        "--feature-symbol-workers",
        type=int,
        default=_DEFAULT_FEATURE_SYMBOL_WORKERS,
        help="训练阶段特征构建并行symbol数（默认按CPU自动设置）",
    )
    parser.add_argument(
        "--labeling-strategy",
        choices=["trainvalidator_hierarchical"],
        default="trainvalidator_hierarchical",
        help="训练策略（TrainValidator 分层建模）",
    )
    parser.add_argument(
        "--batch-labeling-strategies",
        nargs="+",
        choices=_BATCH_STRATEGIES,
        default=None,
        help="批量运行多个标注策略（仅full模式生效）",
    )
    parser.add_argument(
        "--batch-start-from-strategy",
        choices=_BATCH_STRATEGIES,
        default=None,
        help="批量模式下，从指定策略开始执行（含该策略）",
    )
    parser.add_argument(
        "--batch-run-prefix",
        default="labeling_batch",
        help="批量模式下run-id前缀",
    )
    parser.add_argument(
        "--labeling-name",
        default="TrainValidator_v1",
        help="标注方案名称（写入训练指标）",
    )
    parser.add_argument(
        "--pt-multiplier",
        type=float,
        default=2.0,
        help="三重障碍止盈倍数（相对结构止损风险R）",
    )
    parser.add_argument(
        "--timeout-bars",
        type=int,
        default=20,
        help="已弃用（为兼容保留）：二分类标签模式下不参与标注",
    )
    parser.add_argument(
        "--weak-timeout-bars",
        type=int,
        default=10,
        help="已弃用（为兼容保留）：二分类标签模式下不参与标注",
    )
    parser.add_argument(
        "--weak-bsp-types",
        default="3",
        help="弱信号主类型，逗号分隔（默认: 3）",
    )
    parser.add_argument("--cv-splits", type=int, default=5, help="PurgedKFold 折数")
    parser.add_argument("--embargo-bars", type=int, default=10, help="PurgedKFold 隔离带bar数")
    parser.add_argument("--meta-threshold", type=float, default=0.55, help="Meta执行阈值")
    parser.add_argument("--primary-model", choices=["xgboost"], default="xgboost")
    parser.add_argument("--meta-model", choices=["logistic"], default="logistic")
    parser.add_argument("--optuna-trials", type=int, default=0, help="超参搜索次数，0为关闭")

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
        default=False,
        help="回测快速模式：禁用非必要输出与HTML，提升速度（默认关闭）",
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
    parser.add_argument(
        "--signal-margin",
        type=float,
        default=0.0,
        help="信号边际阈值：实际触发阈值=signal-threshold+signal-margin",
    )
    parser.add_argument(
        "--cooldown-bars",
        type=int,
        default=0,
        help="交易冷却bar数；>0时，开/平仓后在冷却窗口内忽略新信号",
    )

    # 报告参数
    parser.add_argument(
        "--save-html-detail-report",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="回测时是否生成详细 HTML 报告（默认开启，可用 --no-save-html-detail-report 关闭）",
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
    max_time_sec: float | None = None,
    stage_name: str = "",
) -> float:
    """执行命令，返回耗时（秒），支持超时保护"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 检查资源
    if stage_name:
        _check_system_resources(stage_name)
    
    start_time = time.time()
    if max_time_sec:
        print(f"[Watchdog] {stage_name} max_time={max_time_sec:.0f}s")
    
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write("$ " + " ".join(cmd) + "\n\n")
        base_env = dict(os.environ)
        base_env["PYTHONIOENCODING"] = "utf-8"
        base_env["PYTHONUTF8"] = "1"
        
        last_activity = time.time()
        stall_timeout_sec = 900  # 15分钟无日志输出 -> 告警
        
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
        
        while True:
            try:
                line = process.stdout.readline()
                if not line:
                    break
                
                print(line, end="")
                log_file.write(line)
                last_activity = time.time()
                
                # 检查超时
                elapsed = time.time() - start_time
                if max_time_sec and elapsed > max_time_sec:
                    print(f"[ERROR] 超时 ({elapsed:.0f}s > {max_time_sec:.0f}s)，终止进程...")
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    raise RuntimeError(
                        f"命令超时 ({elapsed:.0f}s): {' '.join(cmd)}"
                    )
                
                # 检查卡住
                idle_time = time.time() - last_activity
                if idle_time > stall_timeout_sec:
                    print(
                        f"[WARN] 进程卡住 (无输出 {idle_time:.0f}s)，"
                        f"尝试 graceful shutdown..."
                    )
                    process.terminate()
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        process.kill()
                    raise RuntimeError(
                        f"进程卡住 (idle {idle_time:.0f}s > {stall_timeout_sec}s)"
                    )
                    
            except KeyboardInterrupt:
                print("\n[INTERRUPTED] 用户中断，正在清理...")
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                raise
        
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


def _append_timing_history(
    history_csv: Path,
    run_id: str,
    mode: str,
    created_at: str,
    stage_timeline: List[Dict[str, object]],
) -> None:
    """将阶段耗时追加到全局历史CSV，便于长期性能对比。"""
    history_csv.parent.mkdir(parents=True, exist_ok=True)
    exists = history_csv.exists()
    with open(history_csv, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(
                [
                    "created_at",
                    "run_id",
                    "mode",
                    "stage",
                    "status",
                    "elapsed_seconds",
                    "started_at",
                    "ended_at",
                ]
            )
        for row in stage_timeline:
            writer.writerow(
                [
                    created_at,
                    run_id,
                    mode,
                    str(row.get("stage", "")),
                    str(row.get("status", "")),
                    float(row.get("elapsed_seconds", 0.0) or 0.0),
                    str(row.get("started_at", "")),
                    str(row.get("ended_at", "")),
                ]
            )


def _snapshot_architecture_doc(
    project_root: Path,
    destination_dir: Path,
) -> str:
    """将当前架构说明文档快照到本次运行目录，便于后续归档追溯。"""
    source = project_root / "Debug" / "MODEL_TRAINING_ARCHITECTURE.md"
    if not source.exists():
        print(f"[Warn] 架构文档不存在，跳过快照: {source}")
        return ""

    destination_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = destination_dir / f"MODEL_TRAINING_ARCHITECTURE_{ts}.md"
    target.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"[Snapshot] 架构文档快照: {target}")
    return str(target)


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
    architecture_snapshot_path = _snapshot_architecture_doc(project_root, train_dir)

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
        "--feature-symbol-workers",
        str(args.feature_symbol_workers),
        "--labeling-strategy",
        args.labeling_strategy,
        "--labeling-name",
        args.labeling_name,
        "--pt-multiplier",
        str(args.pt_multiplier),
        "--timeout-bars",
        str(args.timeout_bars),
        "--weak-timeout-bars",
        str(args.weak_timeout_bars),
        "--weak-bsp-types",
        args.weak_bsp_types,
        "--cv-splits",
        str(args.cv_splits),
        "--embargo-bars",
        str(args.embargo_bars),
        "--meta-threshold",
        str(args.meta_threshold),
        "--primary-model",
        args.primary_model,
        "--meta-model",
        args.meta_model,
        "--optuna-trials",
        str(args.optuna_trials),
        "--output-dir",
        str(train_dir),
    ]

    timing_stats: Dict[str, float] = {}
    stage_timeline: List[Dict[str, object]] = []
    pipeline_start = time.time()

    print(f"[Pipeline] run_id={run_id}")
    print("[Stage] train（训练模型）")
    train_env = {
        "XGB_TRAIN_LOG_DIR": str(logs_dir),
    }
    merged_train_env = dict(os.environ)
    merged_train_env.update(train_env)
    train_started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    train_elapsed = _run_cmd(
        train_cmd,
        logs_dir / "train.log",
        merged_train_env,
        max_time_sec=_MAX_TRAIN_TIME_SEC,
        stage_name="[train]",
    )
    train_ended_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    timing_stats["train"] = train_elapsed
    stage_timeline.append(
        {
            "stage": "train",
            "status": "done",
            "elapsed_seconds": round(train_elapsed, 2),
            "started_at": train_started_at,
            "ended_at": train_ended_at,
            "log": str(logs_dir / "train.log"),
        }
    )
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
        predict_started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        predict_elapsed = _run_cmd(predict_cmd, logs_dir / "predict.log")
        predict_ended_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        timing_stats["predict"] = predict_elapsed
        stage_timeline.append(
            {
                "stage": "predict",
                "status": "done",
                "elapsed_seconds": round(predict_elapsed, 2),
                "started_at": predict_started_at,
                "ended_at": predict_ended_at,
                "log": str(logs_dir / "predict.log"),
            }
        )
        print(f"[Timing] predict: {predict_elapsed:.2f}s (即 {predict_elapsed/60:.2f} 分钟)\n")
    else:
        stage_timeline.append(
            {
                "stage": "predict",
                "status": "skipped",
                "elapsed_seconds": 0.0,
                "started_at": "",
                "ended_at": "",
                "log": str(logs_dir / "predict.log"),
            }
        )

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
            "--signal-margin",
            str(args.signal_margin),
            "--cooldown-bars",
            str(args.cooldown_bars),
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
        backtest_started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        backtest_elapsed = _run_cmd(backtest_cmd, logs_dir / "backtest.log")
        backtest_ended_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        timing_stats["backtest"] = backtest_elapsed
        stage_timeline.append(
            {
                "stage": "backtest",
                "status": "done",
                "elapsed_seconds": round(backtest_elapsed, 2),
                "started_at": backtest_started_at,
                "ended_at": backtest_ended_at,
                "log": str(logs_dir / "backtest.log"),
            }
        )
        print(f"[Timing] backtest: {backtest_elapsed:.2f}s (即 {backtest_elapsed/60:.2f} 分钟)\n")
    else:
        stage_timeline.append(
            {
                "stage": "backtest",
                "status": "skipped",
                "elapsed_seconds": 0.0,
                "started_at": "",
                "ended_at": "",
                "log": str(logs_dir / "backtest.log"),
            }
        )

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
            "feature_symbol_workers": args.feature_symbol_workers,
            "labeling_strategy": args.labeling_strategy,
            "labeling_name": args.labeling_name,
            "pt_multiplier": args.pt_multiplier,
            "timeout_bars": args.timeout_bars,
            "weak_timeout_bars": args.weak_timeout_bars,
            "weak_bsp_types": args.weak_bsp_types,
            "cv_splits": args.cv_splits,
            "embargo_bars": args.embargo_bars,
            "meta_threshold": args.meta_threshold,
            "primary_model": args.primary_model,
            "meta_model": args.meta_model,
            "optuna_trials": args.optuna_trials,
        },
        "backtest_config": {
            "symbols": args.backtest_symbols,
            "symbol_workers": args.symbol_workers,
            "parallel_mode": args.backtest_parallel_mode,
            "fast_mode": bool(args.backtest_fast_mode and not args.backtest_full_artifacts),
            "signal_threshold": args.signal_threshold,
            "signal_margin": args.signal_margin,
            "cooldown_bars": args.cooldown_bars,
            "save_html_detail_report": bool(args.save_html_detail_report),
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
            "architecture_doc_snapshot": architecture_snapshot_path,
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
        "timing_timeline": stage_timeline,
    }
    _write_manifest(root / "run_manifest.json", manifest)
    timing_summary = {
        "run_id": run_id,
        "mode": "full",
        "created_at": manifest["created_at"],
        "timing_seconds": manifest["timing_seconds"],
        "stages": stage_timeline,
    }
    _write_manifest(root / "timing_summary.json", timing_summary)
    _append_timing_history(
        Path(args.debug_root) / "runs" / "timing_history.csv",
        run_id,
        "full",
        str(manifest["created_at"]),
        stage_timeline,
    )

    print("\n" + "=" * 60)
    print("[✓ 完成] 全流程执行完毕")
    print("=" * 60)
    print(f"[Artifacts Root] {root}")
    print(f"[Manifest] {root / 'run_manifest.json'}")
    print(f"[Timing Summary] {root / 'timing_summary.json'}")
    print(f"[Timing History] {Path(args.debug_root) / 'runs' / 'timing_history.csv'}")
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
    architecture_snapshot_path = _snapshot_architecture_doc(
        project_root,
        root / "docs_snapshot",
    )

    timing_stats: Dict[str, float] = {}
    stage_timeline: List[Dict[str, object]] = []
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
        predict_started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        predict_elapsed = _run_cmd(predict_cmd, logs_dir / "predict.log")
        predict_ended_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        timing_stats["predict"] = predict_elapsed
        stage_timeline.append(
            {
                "stage": "predict",
                "status": "done",
                "elapsed_seconds": round(predict_elapsed, 2),
                "started_at": predict_started_at,
                "ended_at": predict_ended_at,
                "log": str(logs_dir / "predict.log"),
            }
        )
        print(f"[Timing] predict: {predict_elapsed:.2f}s (即 {predict_elapsed/60:.2f} 分钟)\n")
    else:
        stage_timeline.append(
            {
                "stage": "predict",
                "status": "skipped",
                "elapsed_seconds": 0.0,
                "started_at": "",
                "ended_at": "",
                "log": str(logs_dir / "predict.log"),
            }
        )

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
            "--signal-margin",
            str(args.signal_margin),
            "--cooldown-bars",
            str(args.cooldown_bars),
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
        backtest_started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        backtest_elapsed = _run_cmd(backtest_cmd, logs_dir / "backtest.log")
        backtest_ended_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        timing_stats["backtest"] = backtest_elapsed
        stage_timeline.append(
            {
                "stage": "backtest",
                "status": "done",
                "elapsed_seconds": round(backtest_elapsed, 2),
                "started_at": backtest_started_at,
                "ended_at": backtest_ended_at,
                "log": str(logs_dir / "backtest.log"),
            }
        )
        print(f"[Timing] backtest: {backtest_elapsed:.2f}s (即 {backtest_elapsed/60:.2f} 分钟)\n")
    else:
        stage_timeline.append(
            {
                "stage": "backtest",
                "status": "skipped",
                "elapsed_seconds": 0.0,
                "started_at": "",
                "ended_at": "",
                "log": str(logs_dir / "backtest.log"),
            }
        )

    pipeline_total = time.time() - pipeline_start

    manifest = {
        "run_id": run_id,
        "mode": "predict-backtest",
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "train_dir": str(train_dir),
        "architecture_doc_snapshot": architecture_snapshot_path,
        "backtest_config": {
            "symbols": args.backtest_symbols,
            "symbol_workers": args.symbol_workers,
            "parallel_mode": args.backtest_parallel_mode,
            "fast_mode": bool(args.backtest_fast_mode and not args.backtest_full_artifacts),
            "signal_threshold": args.signal_threshold,
            "signal_margin": args.signal_margin,
            "cooldown_bars": args.cooldown_bars,
            "save_html_detail_report": bool(args.save_html_detail_report),
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
        "timing_timeline": stage_timeline,
    }
    _write_manifest(root / "run_manifest_predict_backtest.json", manifest)
    timing_summary = {
        "run_id": run_id,
        "mode": "predict-backtest",
        "created_at": manifest["created_at"],
        "timing_seconds": manifest["timing_seconds"],
        "stages": stage_timeline,
    }
    _write_manifest(root / "timing_summary.json", timing_summary)
    _append_timing_history(
        Path(args.debug_root) / "runs" / "timing_history.csv",
        run_id,
        "predict-backtest",
        str(manifest["created_at"]),
        stage_timeline,
    )

    print("\n" + "=" * 60)
    print("[✓ 完成] 预测+回测执行完毕")
    print("=" * 60)
    print(f"[Artifacts Root] {root}")
    print(f"[Manifest] {root / 'run_manifest_predict_backtest.json'}")
    print(f"[Timing Summary] {root / 'timing_summary.json'}")
    print(f"[Timing History] {Path(args.debug_root) / 'runs' / 'timing_history.csv'}")
    print("\n=== 耗时统计 ===")
    for stage, elapsed in timing_stats.items():
        print(f"  {stage}: {elapsed:.2f}s ({elapsed/60:.2f}min)")
    print(f"  total: {pipeline_total:.2f}s ({pipeline_total/3600:.2f}h)")


def main() -> None:
    args = _build_parser().parse_args()

    # 批量策略运行：将原 run_labeling_strategy_full_batch.py 能力并入统一入口。
    if args.batch_labeling_strategies:
        if args.mode != "full":
            raise ValueError("--batch-labeling-strategies 仅支持 --mode full")
        if args.stage is not None:
            raise ValueError("批量策略模式不支持 --stage")

        strategies = list(args.batch_labeling_strategies)
        if args.batch_start_from_strategy:
            if args.batch_start_from_strategy not in strategies:
                raise ValueError("--batch-start-from-strategy 必须在 --batch-labeling-strategies 内")
            start_idx = strategies.index(args.batch_start_from_strategy)
            strategies = strategies[start_idx:]

        batch_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        summary: List[Dict[str, object]] = []
        for strategy in strategies:
            run_args = argparse.Namespace(**vars(args))
            run_args.labeling_strategy = strategy
            run_args.run_id = f"{args.batch_run_prefix}_{strategy}_{batch_ts}"
            run_args.skip_predict = True if not args.enable_predict else args.skip_predict
            print("\n" + "=" * 72)
            print(f"[Batch] strategy={strategy} run_id={run_args.run_id}")
            print("=" * 72)
            try:
                _run_full_pipeline(run_args)
                summary.append({
                    "strategy": strategy,
                    "run_id": run_args.run_id,
                    "status": "success",
                })
            except Exception as exc:
                summary.append({
                    "strategy": strategy,
                    "run_id": run_args.run_id,
                    "status": "failed",
                    "error": str(exc),
                })
                break

        summary_path = (
            Path(args.debug_root)
            / "runs"
            / f"{args.batch_run_prefix}_summary_{batch_ts}.json"
        )
        _write_manifest(
            summary_path,
            {
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "config": {
                    "begin_time": args.begin_time,
                    "end_time": args.end_time,
                    "train_mode": args.train_mode,
                    "num_workers": args.num_workers,
                    "symbol_workers": args.symbol_workers,
                    "pt_multiplier": args.pt_multiplier,
                    "timeout_bars": args.timeout_bars,
                    "weak_timeout_bars": args.weak_timeout_bars,
                    "weak_bsp_types": args.weak_bsp_types,
                    "cv_splits": args.cv_splits,
                    "embargo_bars": args.embargo_bars,
                    "meta_threshold": args.meta_threshold,
                    "primary_model": args.primary_model,
                    "meta_model": args.meta_model,
                    "optuna_trials": args.optuna_trials,
                },
                "batch": summary,
            },
        )
        print(f"\n[Batch] summary: {summary_path}")
        return

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
