"""
统一的全流程管理脚本，支持两种模式：
1. 全流程模式（train → predict → backtest）
2. 预测+回测模式（快速迭代，复用已有训练模型）

运行环境约定：
    - 统一使用已存在的 conda 环境：chan。
    - 执行前默认已完成：conda activate chan。
    - 本脚本不再重复做环境提醒，直接按当前解释器执行。

架构维护约定：
    - 修改本脚本中与训练/标签/回测参数语义相关逻辑时，
        需同步更新 Debug/MODEL_TRAINING_ARCHITECTURE.md。
"""
from __future__ import annotations

# flake8: noqa: E501, W291, W293

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import psutil  # type: ignore[reportMissingImports]
except ImportError:
    psutil = None  # Graceful fallback if psutil not installed


_CPU_COUNT = os.cpu_count() or 4
_DEFAULT_NUM_WORKERS = max(1, _CPU_COUNT - 1)
_DEFAULT_SYMBOL_WORKERS = max(1, min(8, _CPU_COUNT // 2))
_DEFAULT_FEATURE_SYMBOL_WORKERS = max(1, min(8, _DEFAULT_NUM_WORKERS // 2))
_MAX_TRAIN_TIME_SEC = 4 * 3600  # 4小时超时
_MAX_BACKTEST_TIME_SEC = 3 * 3600  # 3小时超时
_MIN_AVAILABLE_MEMORY_GB = 2.0  # 最少保留2GB内存  
_DEFAULT_CONDA_ENV = "chan"
_TARGET_MEMORY_RESERVE_RATIO = 0.30  # 默认预留30%内存给其他任务
_BATCH_STRATEGIES = [
    "trainvalidator_hierarchical",
]

_SENTINEL_DYNAMIC_SYMBOLS = ["__AUTO_FROM_DATA__"]


def _discover_data_symbols(project_root: Path) -> List[str]:
    """从 data 目录动态发现全部币种（优先15m，缺失时回退5m）。"""
    data_dir = project_root / "data"
    if not data_dir.exists():
        raise FileNotFoundError(f"未找到数据目录: {data_dir}")

    def _collect(interval: str) -> List[str]:
        symbols: List[str] = []
        for path in sorted(data_dir.glob(f"*_{interval}.parquet")):
            base = path.stem.rsplit("_", 1)[0].upper()
            if not base:
                continue
            symbols.append(f"{base}USDT")
        return symbols

    discovered = _collect("15m")
    if not discovered:
        discovered = _collect("5m")
    if not discovered:
        raise FileNotFoundError(
            f"未在 {data_dir} 发现 *_15m.parquet 或 *_5m.parquet 文件"
        )
    return sorted(dict.fromkeys(discovered))


def _split_symbols_4_to_1(symbols: List[str]) -> Tuple[List[str], List[str]]:
    """按4:1拆分训练/测试币种；至少保证测试集1个。"""
    if len(symbols) < 2:
        raise ValueError("币种数量不足，至少需要2个币种才能做4:1划分")
    split_idx = int(len(symbols) * 0.8)
    split_idx = max(1, min(len(symbols) - 1, split_idx))
    return symbols[:split_idx], symbols[split_idx:]


def _resolve_worker_count(
    requested: int,
    cpu_default: int,
    available_mem_gb: float,
    reserve_ratio: float,
    memory_per_worker_gb: float,
    min_workers: int = 1,
) -> int:
    """在用户给定值与内存预留策略之间折中，避免吃满机器内存。"""
    requested_workers = max(min_workers, int(requested or cpu_default))
    if available_mem_gb <= 0 or memory_per_worker_gb <= 0:
        return requested_workers
    usable = max(1.0, available_mem_gb * (1.0 - reserve_ratio))
    by_memory = max(min_workers, int(math.floor(usable / memory_per_worker_gb)))
    return max(min_workers, min(requested_workers, by_memory))


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


def _safe_cmdline(proc) -> str:
    try:
        return " ".join(proc.cmdline())
    except Exception:
        return ""


def _terminate_process_tree(proc, timeout_sec: float = 8.0) -> bool:
    """尽量优雅终止进程树，必要时强杀。"""
    try:
        children = proc.children(recursive=True)
    except Exception:
        children = []

    all_targets = children + [proc]
    for p in all_targets:
        try:
            p.terminate()
        except Exception:
            pass

    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        alive = []
        for p in all_targets:
            try:
                if p.is_running() and p.status() != "zombie":
                    alive.append(p)
            except Exception:
                continue
        if not alive:
            return True
        time.sleep(0.2)

    for p in all_targets:
        try:
            p.kill()
        except Exception:
            pass
    return False


def _kill_existing_experiment_processes(
    project_root: Path,
    current_pid: int,
) -> List[Dict[str, object]]:
    """
    在启动新的 full run 前，清理同项目下遗留的 run 相关进程。
    """
    if not psutil:
        print("[RunCleanup] psutil 不可用，跳过旧进程清理")
        return []

    try:
        current_proc = psutil.Process(current_pid)
        protected_pids = {current_pid}
        for parent in current_proc.parents():
            protected_pids.add(parent.pid)
    except Exception:
        protected_pids = {current_pid}

    script_markers = [
        "Debug/run_pipeline.py",
        "Debug/xgboost_shap_train.py",
        "Debug/xgboost_shap_predict.py",
        "Backtest/examples/run_vectorbt_backtest.py",
    ]
    project_text = str(project_root)
    killed: List[Dict[str, object]] = []

    for proc in psutil.process_iter(attrs=["pid", "name"]):
        pid = int(proc.info.get("pid", 0) or 0)
        if pid <= 0 or pid in protected_pids:
            continue
        cmdline = _safe_cmdline(proc)
        if not cmdline:
            continue
        if project_text not in cmdline and not any(marker in cmdline for marker in script_markers):
            continue
        if not any(marker in cmdline for marker in script_markers):
            continue

        terminated = _terminate_process_tree(proc)
        killed.append(
            {
                "pid": pid,
                "name": str(proc.info.get("name") or ""),
                "terminated": bool(terminated),
                "cmdline": cmdline,
            }
        )

    if killed:
        print(f"[RunCleanup] 已处理旧实验进程 {len(killed)} 个")
        for item in killed:
            status = "ok" if item["terminated"] else "force-kill"
            print(f"  - pid={item['pid']} status={status}")
    else:
        print("[RunCleanup] 未发现需要清理的旧实验进程")

    return killed


def _default_run_id() -> str:
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


def _artifact_root(debug_root: Path, artifact_kind: str) -> Path:
    kind = str(artifact_kind or "run").strip().lower()
    if kind == "test":
        return debug_root / "tests"
    return debug_root / "runs"


def _sanitize_token(text: str, default: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_]+", "_", str(text or "").strip()).strip("_")
    return token or default


def _strip_tail_timestamp(token: str) -> str:
    return re.sub(r"(?:_\d{8}(?:_\d{6})?)$", "", token)


def _make_timestamped_name(prefix: str, suffix: str, timestamp: str | None = None) -> str:
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    p = _strip_tail_timestamp(_sanitize_token(prefix, "pipeline"))
    s = _sanitize_token(suffix, "run")
    return f"{p}_{ts}_{s}"


def _resolve_run_id(
    provided_run_id: str,
    run_prefix: str,
    run_suffix: str,
    mode_suffix: str,
) -> str:
    provided = str(provided_run_id or "").strip()
    if provided:
        return provided
    suffix = str(run_suffix or "").strip() or mode_suffix
    return _make_timestamped_name(run_prefix, suffix)


def _resolve_train_cache_namespace(
    cache_namespace: str,
    cache_mode: str,
    resume_from_run_id: str,
    run_id: str,
) -> str:
    explicit = str(cache_namespace or "").strip()
    if explicit:
        return explicit

    mode = str(cache_mode or "resume").strip().lower()
    if mode == "fresh":
        return f"fresh_{_sanitize_token(run_id, 'run')}"

    resume_id = str(resume_from_run_id or "").strip()
    if resume_id:
        return resume_id

    return "default"


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
        "--run-prefix",
        default="pipeline",
        help="自动命名时的前缀（run-id为空时生效）",
    )
    parser.add_argument(
        "--run-suffix",
        default="",
        help="自动命名时的后缀（run-id为空时生效）",
    )
    parser.add_argument(
        "--artifact-kind",
        choices=["run", "test"],
        default="run",
        help="产物类别：run=正式产物，test=测试产物",
    )
    parser.add_argument(
        "--source-artifact-kind",
        choices=["run", "test"],
        default="run",
        help="source-run-id 所在产物类别（predict-backtest模式）",
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
        default=list(_SENTINEL_DYNAMIC_SYMBOLS),
        help="训练币种（默认从data目录动态发现全量币种后按4:1自动切分）",
    )
    parser.add_argument(
        "--test-symbols",
        nargs="+",
        default=list(_SENTINEL_DYNAMIC_SYMBOLS),
        help="测试币种（默认由train-symbols自动4:1切分得到）",
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
        "--dynamic-pt-enabled",
        action="store_true",
        help="启用按波动状态动态调整pt_multiplier",
    )
    parser.add_argument("--pt-low-vol-multiplier", type=float, default=1.8)
    parser.add_argument("--pt-mid-vol-multiplier", type=float, default=2.0)
    parser.add_argument("--pt-high-vol-multiplier", type=float, default=2.2)
    parser.add_argument("--pt-vol-window", type=int, default=96)
    parser.add_argument("--pt-vol-quantile-low", type=float, default=0.33)
    parser.add_argument("--pt-vol-quantile-high", type=float, default=0.67)
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
    parser.add_argument(
        "--meta-calibration",
        choices=["none", "platt", "isotonic"],
        default="none",
        help="Meta概率校准方法",
    )
    parser.add_argument("--meta-calibration-ratio", type=float, default=0.2, help="Meta校准集占比")
    parser.add_argument(
        "--pass-positive-month-ratio",
        type=float,
        default=0.70,
        help="训练准入约束：月度正收益占比下限",
    )
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
        default=list(_SENTINEL_DYNAMIC_SYMBOLS),
        help="回测币种（默认从data目录动态发现全量币种）",
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
        "--backtest-data-cache-size",
        type=int,
        default=8,
        help="回测数据缓存条目数（每进程，0表示关闭）",
    )
    parser.add_argument(
        "--backtest-preload-bars",
        action="store_true",
        default=False,
        help="回测前预加载K线（thread并行模式下更有效）",
    )
    parser.add_argument(
        "--event-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="回测事件缓存开关（默认开启，可用 --no-event-cache 关闭）",
    )
    parser.add_argument(
        "--event-cache-dir",
        default="data/cache/backtest_events",
        help="回测事件缓存目录",
    )
    parser.add_argument(
        "--event-cache-namespace",
        default="default",
        help="回测事件缓存命名空间（用于隔离实验）",
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
        "--meta-threshold-by-bsp",
        default="",
        help="回测分组阈值(JSON)：如 {'1':0.53,'3':0.62,'buy_1':0.51,'sell_3':0.65}",
    )
    parser.add_argument(
        "--meta-threshold-by-direction",
        default="",
        help="回测方向阈值(JSON)：如 {'buy':0.55,'sell':0.60}",
    )
    parser.add_argument(
        "--cooldown-bars",
        type=int,
        default=0,
        help="交易冷却bar数；>0时，开/平仓后在冷却窗口内忽略新信号",
    )
    parser.add_argument(
        "--empty-signal-fallback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="空信号补单开关（默认关闭，可用 --empty-signal-fallback 开启）",
    )
    parser.add_argument(
        "--empty-signal-target-rate",
        type=float,
        default=0.05,
        help="空信号补单目标比例（仅在 --empty-signal-fallback 开启时生效）",
    )
    parser.add_argument("--backtest-fee", type=float, default=0.0004, help="回测手续费")
    parser.add_argument("--backtest-slippage", type=float, default=0.0001, help="回测滑点")

    # 报告参数
    parser.add_argument(
        "--save-html-detail-report",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="回测时是否生成详细 HTML 报告（默认开启，可用 --no-save-html-detail-report 关闭）",
    )
    parser.add_argument(
        "--save-trades-csv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="回测时是否导出 executed_trades.csv（默认开启）",
    )
    parser.add_argument(
        "--save-equity-csv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="回测时是否导出 portfolio_equity_curve.csv（默认开启）",
    )
    parser.add_argument(
        "--enable-predict",
        action="store_true",
        help="在full模式下启用predict阶段（默认full只做训练+回测）",
    )

    # 缓存参数
    parser.add_argument(
        "--disable-label-cache",
        action="store_true",
        help="禁用训练时的标签缓存",
    )
    parser.add_argument(
        "--disable-feature-cache",
        action="store_true",
        help="禁用训练时的特征缓存",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        help="清空所有缓存并退出",
    )
    parser.add_argument(
        "--cache-mode",
        choices=["resume", "fresh"],
        default="resume",
        help=(
            "训练缓存模式：resume=续跑复用缓存；"
            "fresh=全新重跑，不使用缓存。"
        ),
    )
    parser.add_argument(
        "--cache-namespace",
        default="",
        help="训练缓存命名空间；为空时自动解析（resume_from_run_id/fresh_run_id/default）",
    )
    parser.add_argument(
        "--resume-from-run-id",
        default="",
        help="续跑来源 run_id（仅 cache-mode=resume 且未显式 cache-namespace 时生效）",
    )
    parser.add_argument(
        "--kill-existing-runs-before-start",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="full模式启动前，先终止同项目旧 run 相关进程（默认开启）",
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
    parser.add_argument(
        "--backup-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否在执行前将当前代码快照统一备份到backup目录（默认开启）",
    )
    parser.add_argument(
        "--backup-tag",
        default="pipeline_backup",
        help="backup目录快照前缀",
    )

    return parser


def _run_cmd(
    cmd: List[str],
    log_path: Path,
    extra_env: Dict[str, str] | None = None,
    max_time_sec: float | None = None,
    stage_name: str = "",
    run_id: str = "",
    mode: str = "",
    stage_alias: str = "",
) -> float:
    """执行命令并返回耗时（秒）。"""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 检查资源
    if stage_name:
        _check_system_resources(stage_name)
    
    start_time = time.time()
    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    with open(log_path, "w", encoding="utf-8", buffering=1) as log_file:
        log_file.write("=" * 88 + "\n")
        log_file.write(f"RUN_ID: {run_id or '--'}\n")
        log_file.write(f"MODE: {mode or '--'}\n")
        log_file.write(f"STAGE: {stage_alias or stage_name or '--'}\n")
        log_file.write(f"STARTED_AT: {started_at}\n")
        log_file.write(f"LOG_PATH: {log_path}\n")
        log_file.write(f"WORKDIR: {Path.cwd()}\n")
        log_file.write("=" * 88 + "\n")
        log_file.write("$ " + " ".join(cmd) + "\n\n")
        # Make stage header visible immediately even when child process has no early stdout.
        log_file.flush()
        base_env = dict(os.environ)
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
        
        while True:
            try:
                line = process.stdout.readline()
                if not line:
                    break
                
                print(line, end="")
                line_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                log_file.write(f"[{line_ts}] {line}")
                log_file.flush()
                    
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
            finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            elapsed = time.time() - start_time
            log_file.write("\n" + "=" * 88 + "\n")
            log_file.write(f"FINISHED_AT: {finished_at}\n")
            log_file.write(f"ELAPSED_SECONDS: {elapsed:.2f}\n")
            log_file.write(f"EXIT_CODE: {rc}\n")
            log_file.write("=" * 88 + "\n")
            log_file.flush()
            raise RuntimeError(f"命令失败 (exit={rc}): {' '.join(cmd)}")
        finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elapsed = time.time() - start_time
        log_file.write("\n" + "=" * 88 + "\n")
        log_file.write(f"FINISHED_AT: {finished_at}\n")
        log_file.write(f"ELAPSED_SECONDS: {elapsed:.2f}\n")
        log_file.write("EXIT_CODE: 0\n")
        log_file.write("=" * 88 + "\n")
        log_file.flush()
    
    elapsed = time.time() - start_time
    return elapsed


def _write_manifest(path: Path, payload: Dict[str, object]) -> None:
    """写入 manifest 文件"""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _append_pipeline_log(logs_dir: Path, message: str) -> None:
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(logs_dir / "pipeline.log", "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {message}\n")


def _update_run_index(
    artifact_root: Path,
    run_id: str,
    mode: str,
    status: str,
    current_stage: str,
    root: Path,
    logs_dir: Path,
    started_at: str,
    ended_at: str = "",
) -> None:
    logs = {
        "pipeline": str(logs_dir / "pipeline.log"),
        "train": str(logs_dir / "train.log"),
        "predict": str(logs_dir / "predict.log"),
        "backtest": str(logs_dir / "backtest.log"),
    }
    active_log = logs.get(current_stage, logs["pipeline"])
    payload: Dict[str, object] = {
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_id": run_id,
        "mode": mode,
        "status": status,
        "current_stage": current_stage,
        "started_at": started_at,
        "ended_at": ended_at,
        "root": str(root),
        "logs": logs,
        "active_log": str(active_log),
    }

    latest_path = artifact_root / "latest_run.json"
    current_path = artifact_root / "current_run.json"
    _write_manifest(latest_path, payload)
    if status == "running":
        _write_manifest(current_path, payload)
    else:
        if current_path.exists():
            try:
                current_path.unlink()
            except OSError:
                pass


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
            elapsed_raw = row.get("elapsed_seconds", 0.0)
            elapsed_seconds = 0.0
            if isinstance(elapsed_raw, (int, float)):
                elapsed_seconds = float(elapsed_raw)
            elif isinstance(elapsed_raw, str):
                try:
                    elapsed_seconds = float(elapsed_raw)
                except ValueError:
                    elapsed_seconds = 0.0
            writer.writerow(
                [
                    created_at,
                    run_id,
                    mode,
                    str(row.get("stage", "")),
                    str(row.get("status", "")),
                    elapsed_seconds,
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


def _safe_copy_file(src: Path, dst: Path) -> bool:
    if not src.exists() or not src.is_file():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def _safe_copy_tree(src: Path, dst: Path) -> bool:
    if not src.exists() or not src.is_dir():
        return False
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(
        src,
        dst,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", ".pytest_cache", ".mypy_cache"),
    )
    return True


def _create_unified_code_backup(
    project_root: Path,
    run_id: str,
    mode: str,
    backup_tag: str,
) -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = project_root / "backup" / f"{backup_tag}_{run_id}_{ts}"
    code_dir = backup_dir / "code"
    docs_dir = backup_dir / "docs"
    for d in [backup_dir, code_dir, docs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    copied_files: List[str] = []
    copied_dirs: List[str] = []
    code_files = [
        project_root / "Debug" / "run_pipeline.py",
        project_root / "Debug" / "xgboost_shap_train.py",
        project_root / "Debug" / "xgboost_shap_predict.py",
        project_root / "Debug" / "tune_backtest_threshold.py",
        project_root / "Debug" / "MODEL_TRAINING_ARCHITECTURE.md",
        project_root / "Debug" / "PIPELINE_USAGE.md",
        project_root / "backup" / "README.md",
    ]
    for src in code_files:
        rel = src.relative_to(project_root)
        if _safe_copy_file(src, code_dir / rel):
            copied_files.append(str(rel))

    for rel_dir in ["ml_layer", "Backtest"]:
        src_dir = project_root / rel_dir
        if _safe_copy_tree(src_dir, code_dir / rel_dir):
            copied_dirs.append(rel_dir)

    notes_path = docs_dir / "backup_notes.md"
    notes_path.write_text(
        "\n".join(
            [
                "# 统一备份快照",
                "",
                f"- created_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                f"- run_id: {run_id}",
                f"- mode: {mode}",
                "- scope: 当前代码与架构文档快照",
            ]
        ),
        encoding="utf-8",
    )

    manifest = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_id": run_id,
        "mode": mode,
        "copied_files": copied_files,
        "copied_dirs": copied_dirs,
        "backup_notes": str(notes_path),
    }
    _write_manifest(backup_dir / "manifest.json", manifest)
    print(f"[Backup] 代码快照已创建: {backup_dir}")
    return backup_dir


def _append_run_artifacts_to_backup(
    backup_dir: Path,
    run_root: Path,
    source_train_dir: Path | None = None,
) -> Dict[str, str]:
    artifacts_dir = backup_dir / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    copied: Dict[str, str] = {}

    if _safe_copy_tree(run_root, artifacts_dir / "current_run"):
        copied["current_run"] = str(artifacts_dir / "current_run")

    if source_train_dir is not None:
        if _safe_copy_tree(source_train_dir, artifacts_dir / "source_train"):
            copied["source_train"] = str(artifacts_dir / "source_train")

    if copied:
        print(f"[Backup] 产物归档完成: {artifacts_dir}")
    return copied


def _write_iteration_dashboard(
    root: Path,
    run_id: str,
    mode: str,
    signal_threshold: float,
    signal_margin: float,
    cooldown_bars: int,
    stage_timeline: List[Dict[str, object]],
    timing_seconds: Dict[str, float],
) -> str:
    backtest_metrics_path = root / "backtest" / "backtest_metrics.json"
    aggregate = {}
    if backtest_metrics_path.exists():
        try:
            payload = json.loads(backtest_metrics_path.read_text(encoding="utf-8"))
            aggregate = payload.get("aggregate", {}) or {}
        except Exception:
            aggregate = {}

    dashboard = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_id": run_id,
        "mode": mode,
        "signal_threshold": float(signal_threshold),
        "signal_margin": float(signal_margin),
        "cooldown_bars": int(cooldown_bars),
        "metrics": {
            "annualized_return_pct": float(aggregate.get("annualized_return_pct", 0.0) or 0.0),
            "max_drawdown_pct": float(aggregate.get("max_drawdown_pct", 0.0) or 0.0),
            "sharpe": float(aggregate.get("sharpe", 0.0) or 0.0),
            "win_rate_pct": float(aggregate.get("win_rate_pct", 0.0) or 0.0),
            "total_trades": float(aggregate.get("total_trades", 0.0) or 0.0),
            "total_return_pct": float(aggregate.get("total_return_pct", 0.0) or 0.0),
        },
        "timing_seconds": timing_seconds,
        "stage_timeline": stage_timeline,
    }

    out = root / "iteration_dashboard.json"
    _write_manifest(out, dashboard)
    return str(out)


def _build_backtest_cmd(
    python_exec: str,
    project_root: Path,
    model_buy: Path,
    model_sell: Path,
    meta_buy: Path,
    meta_sell: Path,
    backtest_dir: Path,
    args: argparse.Namespace,
) -> List[str]:
    cmd = [
        python_exec,
        str(project_root / "Backtest" / "examples" / "run_vectorbt_backtest.py"),
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
        "--fee",
        str(args.backtest_fee),
        "--slippage",
        str(args.backtest_slippage),
        "--cooldown-bars",
        str(args.cooldown_bars),
        "--output-dir",
        str(backtest_dir),
        "--symbol-workers",
        str(args.symbol_workers),
        "--parallel-mode",
        args.backtest_parallel_mode,
        "--data-cache-size",
        str(int(args.backtest_data_cache_size)),
    ]
    if str(args.meta_threshold_by_bsp).strip():
        cmd.extend(["--meta-threshold-by-bsp", str(args.meta_threshold_by_bsp)])
    if str(args.meta_threshold_by_direction).strip():
        cmd.extend([
            "--meta-threshold-by-direction",
            str(args.meta_threshold_by_direction),
        ])
    if not bool(args.event_cache):
        cmd.append("--disable-event-cache")
    if str(args.event_cache_dir).strip():
        cmd.extend(["--event-cache-dir", str(args.event_cache_dir)])
    if str(args.event_cache_namespace).strip():
        cmd.extend(["--event-cache-namespace", str(args.event_cache_namespace)])
    if bool(args.empty_signal_fallback):
        cmd.append("--enable-empty-signal-fallback")
    cmd.extend(["--empty-signal-target-rate", str(args.empty_signal_target_rate)])
    if not bool(args.save_trades_csv):
        cmd.append("--no-trades-csv")
    if not bool(args.save_equity_csv):
        cmd.append("--no-equity-csv")
    if args.backtest_fast_mode and not args.backtest_full_artifacts:
        cmd.append("--fast-mode")
    if bool(args.backtest_preload_bars):
        cmd.append("--preload-bars")
    if args.save_html_detail_report:
        cmd.append("--save-html-detail-report")
    return cmd


def _run_full_pipeline(args: argparse.Namespace) -> None:
    """完整流程：训练 → 预测 → 回测"""
    run_id = _resolve_run_id(
        provided_run_id=args.run_id,
        run_prefix=args.run_prefix,
        run_suffix=args.run_suffix,
        mode_suffix="full",
    )

    debug_root = Path(args.debug_root)
    artifact_root = _artifact_root(debug_root, args.artifact_kind)
    root = artifact_root / run_id
    train_dir = root / "train"
    predict_dir = root / "predict"
    backtest_dir = root / "backtest"
    logs_dir = root / "logs"

    for d in [train_dir, predict_dir, backtest_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    python_exec = sys.executable
    project_root = Path(__file__).resolve().parents[1]
    train_cache_namespace = _resolve_train_cache_namespace(
        cache_namespace=str(args.cache_namespace),
        cache_mode=str(args.cache_mode),
        resume_from_run_id=str(args.resume_from_run_id),
        run_id=run_id,
    )

    cleanup_result: List[Dict[str, object]] = []
    if bool(args.kill_existing_runs_before_start):
        cleanup_result = _kill_existing_experiment_processes(
            project_root=project_root,
            current_pid=os.getpid(),
        )

    backup_dir: Path | None = None
    if bool(args.backup_run):
        backup_dir = _create_unified_code_backup(
            project_root=project_root,
            run_id=run_id,
            mode="full",
            backup_tag=str(args.backup_tag),
        )
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
        "--pt-low-vol-multiplier",
        str(args.pt_low_vol_multiplier),
        "--pt-mid-vol-multiplier",
        str(args.pt_mid_vol_multiplier),
        "--pt-high-vol-multiplier",
        str(args.pt_high_vol_multiplier),
        "--pt-vol-window",
        str(args.pt_vol_window),
        "--pt-vol-quantile-low",
        str(args.pt_vol_quantile_low),
        "--pt-vol-quantile-high",
        str(args.pt_vol_quantile_high),
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
        "--meta-calibration",
        str(args.meta_calibration),
        "--meta-calibration-ratio",
        str(args.meta_calibration_ratio),
        "--pass-positive-month-ratio",
        str(args.pass_positive_month_ratio),
        "--primary-model",
        args.primary_model,
        "--meta-model",
        args.meta_model,
        "--optuna-trials",
        str(args.optuna_trials),
        "--output-dir",
        str(train_dir),
        "--cache-mode",
        str(args.cache_mode),
        "--cache-namespace",
        train_cache_namespace,
    ]
    if bool(args.dynamic_pt_enabled):
        train_cmd.append("--dynamic-pt-enabled")
    if bool(args.disable_label_cache):
        train_cmd.append("--disable-label-cache")
    if bool(args.disable_feature_cache):
        train_cmd.append("--disable-feature-cache")
    if bool(args.clear_cache):
        train_cmd.append("--clear-cache")

    timing_stats: Dict[str, float] = {}
    stage_timeline: List[Dict[str, object]] = []
    pipeline_start = time.time()
    pipeline_started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    _update_run_index(
        artifact_root=artifact_root,
        run_id=run_id,
        mode="full",
        status="running",
        current_stage="init",
        root=root,
        logs_dir=logs_dir,
        started_at=pipeline_started_at,
    )
    _append_pipeline_log(
        logs_dir,
        f"[RUN_START] run_id={run_id} mode=full root={root}",
    )

    print(f"[Pipeline] run_id={run_id}, artifact_kind={args.artifact_kind}")
    print("[Stage] train（训练模型）")
    _update_run_index(
        artifact_root=artifact_root,
        run_id=run_id,
        mode="full",
        status="running",
        current_stage="train",
        root=root,
        logs_dir=logs_dir,
        started_at=pipeline_started_at,
    )
    _append_pipeline_log(logs_dir, "[STAGE_START] train")
    train_env = {
        "XGB_TRAIN_LOG_DIR": str(logs_dir),
        "XGB_DISABLE_INNER_FILE_LOG": "1",
        "XGB_PIPELINE_RUN_ID": run_id,
        "XGB_PIPELINE_MODE": "full",
        "XGB_PIPELINE_STAGE": "train",
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
        run_id=run_id,
        mode="full",
        stage_alias="train",
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
    _append_pipeline_log(logs_dir, f"[STAGE_DONE] train elapsed={train_elapsed:.2f}s")

    # 预测阶段
    predict_cmd: List[str] = []
    predict_elapsed = 0.0
    if not args.skip_predict:
        _update_run_index(
            artifact_root=artifact_root,
            run_id=run_id,
            mode="full",
            status="running",
            current_stage="predict",
            root=root,
            logs_dir=logs_dir,
            started_at=pipeline_started_at,
        )
        _append_pipeline_log(logs_dir, "[STAGE_START] predict")
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
        predict_elapsed = _run_cmd(
            predict_cmd,
            logs_dir / "predict.log",
            stage_name="[predict]",
            run_id=run_id,
            mode="full",
            stage_alias="predict",
        )
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
        _append_pipeline_log(logs_dir, f"[STAGE_DONE] predict elapsed={predict_elapsed:.2f}s")
    else:
        _append_pipeline_log(logs_dir, "[STAGE_SKIPPED] predict")
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
        _update_run_index(
            artifact_root=artifact_root,
            run_id=run_id,
            mode="full",
            status="running",
            current_stage="backtest",
            root=root,
            logs_dir=logs_dir,
            started_at=pipeline_started_at,
        )
        _append_pipeline_log(logs_dir, "[STAGE_START] backtest")
        backtest_cmd = _build_backtest_cmd(
            python_exec=python_exec,
            project_root=project_root,
            model_buy=model_buy,
            model_sell=model_sell,
            meta_buy=meta_buy,
            meta_sell=meta_sell,
            backtest_dir=backtest_dir,
            args=args,
        )

        print(f"[Stage] backtest（全量回测，共{len(args.backtest_symbols)}个币种）")
        backtest_started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        backtest_elapsed = _run_cmd(
            backtest_cmd,
            logs_dir / "backtest.log",
            max_time_sec=_MAX_BACKTEST_TIME_SEC,
            stage_name="[backtest]",
            run_id=run_id,
            mode="full",
            stage_alias="backtest",
        )
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
        _append_pipeline_log(logs_dir, f"[STAGE_DONE] backtest elapsed={backtest_elapsed:.2f}s")
    else:
        _append_pipeline_log(logs_dir, "[STAGE_SKIPPED] backtest")
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
            "dynamic_pt_enabled": bool(args.dynamic_pt_enabled),
            "pt_low_vol_multiplier": args.pt_low_vol_multiplier,
            "pt_mid_vol_multiplier": args.pt_mid_vol_multiplier,
            "pt_high_vol_multiplier": args.pt_high_vol_multiplier,
            "pt_vol_window": args.pt_vol_window,
            "pt_vol_quantile_low": args.pt_vol_quantile_low,
            "pt_vol_quantile_high": args.pt_vol_quantile_high,
            "timeout_bars": args.timeout_bars,
            "weak_timeout_bars": args.weak_timeout_bars,
            "weak_bsp_types": args.weak_bsp_types,
            "cv_splits": args.cv_splits,
            "embargo_bars": args.embargo_bars,
            "meta_threshold": args.meta_threshold,
            "meta_calibration": args.meta_calibration,
            "meta_calibration_ratio": args.meta_calibration_ratio,
            "pass_positive_month_ratio": args.pass_positive_month_ratio,
            "primary_model": args.primary_model,
            "meta_model": args.meta_model,
            "optuna_trials": args.optuna_trials,
            "cache_mode": str(args.cache_mode),
            "cache_namespace": train_cache_namespace,
            "resume_from_run_id": str(args.resume_from_run_id),
        },
        "run_start_policy": {
            "kill_existing_runs_before_start": bool(
                args.kill_existing_runs_before_start
            ),
            "killed_processes": cleanup_result,
        },
        "backtest_config": {
            "symbols": args.backtest_symbols,
            "symbol_workers": args.symbol_workers,
            "parallel_mode": args.backtest_parallel_mode,
            "fast_mode": bool(args.backtest_fast_mode and not args.backtest_full_artifacts),
            "event_cache": bool(args.event_cache),
            "event_cache_dir": str(args.event_cache_dir),
            "event_cache_namespace": str(args.event_cache_namespace),
            "signal_threshold": args.signal_threshold,
            "signal_margin": args.signal_margin,
            "meta_threshold_by_bsp": str(args.meta_threshold_by_bsp),
            "meta_threshold_by_direction": str(args.meta_threshold_by_direction),
            "cooldown_bars": args.cooldown_bars,
            "empty_signal_fallback": bool(args.empty_signal_fallback),
            "empty_signal_target_rate": args.empty_signal_target_rate,
            "fee": args.backtest_fee,
            "slippage": args.backtest_slippage,
            "save_html_detail_report": bool(args.save_html_detail_report),
            "save_trades_csv": bool(args.save_trades_csv),
            "save_equity_csv": bool(args.save_equity_csv),
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

    dashboard_path = _write_iteration_dashboard(
        root=root,
        run_id=run_id,
        mode="full",
        signal_threshold=float(args.signal_threshold),
        signal_margin=float(args.signal_margin),
        cooldown_bars=int(args.cooldown_bars),
        stage_timeline=stage_timeline,
        timing_seconds=manifest["timing_seconds"],
    )
    manifest["artifacts"]["iteration_dashboard"] = dashboard_path

    if backup_dir is not None:
        copied = _append_run_artifacts_to_backup(backup_dir=backup_dir, run_root=root)
        manifest["backup"] = {
            "enabled": True,
            "backup_dir": str(backup_dir),
            "copied_artifacts": copied,
        }
    else:
        manifest["backup"] = {
            "enabled": False,
            "backup_dir": "",
            "copied_artifacts": {},
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
        artifact_root / "timing_history.csv",
        run_id,
        "full",
        str(manifest["created_at"]),
        stage_timeline,
    )
    pipeline_ended_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _update_run_index(
        artifact_root=artifact_root,
        run_id=run_id,
        mode="full",
        status="done",
        current_stage="done",
        root=root,
        logs_dir=logs_dir,
        started_at=pipeline_started_at,
        ended_at=pipeline_ended_at,
    )
    _append_pipeline_log(
        logs_dir,
        (
            "[RUN_DONE] "
            f"run_id={run_id} total={pipeline_total:.2f}s "
            f"manifest={root / 'run_manifest.json'}"
        ),
    )

    print("\n" + "=" * 60)
    print("[✓ 完成] 全流程执行完毕")
    print("=" * 60)
    print(f"[Artifacts Root] {root}")
    print(f"[Manifest] {root / 'run_manifest.json'}")
    print(f"[Timing Summary] {root / 'timing_summary.json'}")
    print(f"[Timing History] {artifact_root / 'timing_history.csv'}")
    print("\n=== 耗时统计 ===")
    for stage, elapsed in timing_stats.items():
        print(f"  {stage}: {elapsed:.2f}s ({elapsed/60:.2f}min)")
    print(f"  total: {pipeline_total:.2f}s ({pipeline_total/3600:.2f}h)")


def _run_predict_backtest_only(args: argparse.Namespace) -> None:
    """快速模式：仅预测 + 回测（复用已有训练模型）"""
    run_id = _resolve_run_id(
        provided_run_id=args.run_id,
        run_prefix=args.run_prefix,
        run_suffix=args.run_suffix,
        mode_suffix="predict_backtest",
    )

    # 解析训练目录
    if args.train_dir:
        train_dir = Path(args.train_dir)
    elif args.source_run_id:
        source_root = _artifact_root(Path(args.debug_root), args.source_artifact_kind)
        train_dir = source_root / args.source_run_id / "train"
        if not train_dir.exists():
            fallback_kind = "test" if str(args.source_artifact_kind) == "run" else "run"
            fallback_root = _artifact_root(Path(args.debug_root), fallback_kind)
            fallback_train_dir = fallback_root / args.source_run_id / "train"
            if fallback_train_dir.exists():
                print(
                    "[Pipeline][WARN] source_artifact_kind 与实际目录不一致，"
                    f"自动回退到 {fallback_kind}: {fallback_train_dir}"
                )
                train_dir = fallback_train_dir
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

    debug_root = Path(args.debug_root)
    artifact_root = _artifact_root(debug_root, args.artifact_kind)
    root = artifact_root / run_id
    predict_dir = root / "predict"
    backtest_dir = root / "backtest"
    logs_dir = root / "logs"
    for d in [predict_dir, backtest_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    project_root = Path(__file__).resolve().parents[1]
    python_exec = sys.executable
    backup_dir: Path | None = None
    if bool(args.backup_run):
        backup_dir = _create_unified_code_backup(
            project_root=project_root,
            run_id=run_id,
            mode="predict-backtest",
            backup_tag=str(args.backup_tag),
        )
    architecture_snapshot_path = _snapshot_architecture_doc(
        project_root,
        root / "docs_snapshot",
    )

    timing_stats: Dict[str, float] = {}
    stage_timeline: List[Dict[str, object]] = []
    pipeline_start = time.time()
    pipeline_started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    _update_run_index(
        artifact_root=artifact_root,
        run_id=run_id,
        mode="predict-backtest",
        status="running",
        current_stage="init",
        root=root,
        logs_dir=logs_dir,
        started_at=pipeline_started_at,
    )
    _append_pipeline_log(
        logs_dir,
        f"[RUN_START] run_id={run_id} mode=predict-backtest root={root}",
    )

    print(
        f"[Pipeline] run_id={run_id} (predict-backtest 模式), "
        f"artifact_kind={args.artifact_kind}"
    )
    print(f"[复用训练模型] {train_dir}")

    # 预测阶段
    predict_cmd: List[str] = []
    predict_elapsed = 0.0
    if not args.skip_predict:
        _update_run_index(
            artifact_root=artifact_root,
            run_id=run_id,
            mode="predict-backtest",
            status="running",
            current_stage="predict",
            root=root,
            logs_dir=logs_dir,
            started_at=pipeline_started_at,
        )
        _append_pipeline_log(logs_dir, "[STAGE_START] predict")
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
        predict_elapsed = _run_cmd(
            predict_cmd,
            logs_dir / "predict.log",
            stage_name="[predict]",
            run_id=run_id,
            mode="predict-backtest",
            stage_alias="predict",
        )
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
        _append_pipeline_log(logs_dir, f"[STAGE_DONE] predict elapsed={predict_elapsed:.2f}s")
    else:
        _append_pipeline_log(logs_dir, "[STAGE_SKIPPED] predict")
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
        _update_run_index(
            artifact_root=artifact_root,
            run_id=run_id,
            mode="predict-backtest",
            status="running",
            current_stage="backtest",
            root=root,
            logs_dir=logs_dir,
            started_at=pipeline_started_at,
        )
        _append_pipeline_log(logs_dir, "[STAGE_START] backtest")
        backtest_cmd = _build_backtest_cmd(
            python_exec=python_exec,
            project_root=project_root,
            model_buy=model_buy,
            model_sell=model_sell,
            meta_buy=meta_buy,
            meta_sell=meta_sell,
            backtest_dir=backtest_dir,
            args=args,
        )

        print(f"[Stage] backtest（全量回测，共{len(args.backtest_symbols)}个币种）")
        backtest_started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        backtest_elapsed = _run_cmd(
            backtest_cmd,
            logs_dir / "backtest.log",
            max_time_sec=_MAX_BACKTEST_TIME_SEC,
            stage_name="[backtest]",
            run_id=run_id,
            mode="predict-backtest",
            stage_alias="backtest",
        )
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
        _append_pipeline_log(logs_dir, f"[STAGE_DONE] backtest elapsed={backtest_elapsed:.2f}s")
    else:
        _append_pipeline_log(logs_dir, "[STAGE_SKIPPED] backtest")
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
            "event_cache": bool(args.event_cache),
            "event_cache_dir": str(args.event_cache_dir),
            "event_cache_namespace": str(args.event_cache_namespace),
            "signal_threshold": args.signal_threshold,
            "signal_margin": args.signal_margin,
            "meta_threshold_by_bsp": str(args.meta_threshold_by_bsp),
            "meta_threshold_by_direction": str(args.meta_threshold_by_direction),
            "cooldown_bars": args.cooldown_bars,
            "empty_signal_fallback": bool(args.empty_signal_fallback),
            "empty_signal_target_rate": args.empty_signal_target_rate,
            "fee": args.backtest_fee,
            "slippage": args.backtest_slippage,
            "save_html_detail_report": bool(args.save_html_detail_report),
            "save_trades_csv": bool(args.save_trades_csv),
            "save_equity_csv": bool(args.save_equity_csv),
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

    dashboard_path = _write_iteration_dashboard(
        root=root,
        run_id=run_id,
        mode="predict-backtest",
        signal_threshold=float(args.signal_threshold),
        signal_margin=float(args.signal_margin),
        cooldown_bars=int(args.cooldown_bars),
        stage_timeline=stage_timeline,
        timing_seconds=manifest["timing_seconds"],
    )
    manifest["iteration_dashboard"] = dashboard_path

    if backup_dir is not None:
        copied = _append_run_artifacts_to_backup(
            backup_dir=backup_dir,
            run_root=root,
            source_train_dir=train_dir,
        )
        manifest["backup"] = {
            "enabled": True,
            "backup_dir": str(backup_dir),
            "copied_artifacts": copied,
        }
    else:
        manifest["backup"] = {
            "enabled": False,
            "backup_dir": "",
            "copied_artifacts": {},
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
        artifact_root / "timing_history.csv",
        run_id,
        "predict-backtest",
        str(manifest["created_at"]),
        stage_timeline,
    )
    pipeline_ended_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    _update_run_index(
        artifact_root=artifact_root,
        run_id=run_id,
        mode="predict-backtest",
        status="done",
        current_stage="done",
        root=root,
        logs_dir=logs_dir,
        started_at=pipeline_started_at,
        ended_at=pipeline_ended_at,
    )
    _append_pipeline_log(
        logs_dir,
        (
            "[RUN_DONE] "
            f"run_id={run_id} total={pipeline_total:.2f}s "
            f"manifest={root / 'run_manifest_predict_backtest.json'}"
        ),
    )

    print("\n" + "=" * 60)
    print("[✓ 完成] 预测+回测执行完毕")
    print("=" * 60)
    print(f"[Artifacts Root] {root}")
    print(f"[Manifest] {root / 'run_manifest_predict_backtest.json'}")
    print(f"[Timing Summary] {root / 'timing_summary.json'}")
    print(f"[Timing History] {artifact_root / 'timing_history.csv'}")
    print("\n=== 耗时统计 ===")
    for stage, elapsed in timing_stats.items():
        print(f"  {stage}: {elapsed:.2f}s ({elapsed/60:.2f}min)")
    print(f"  total: {pipeline_total:.2f}s ({pipeline_total/3600:.2f}h)")


def main() -> None:
    args = _build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]

    # 动态币种解析：默认每次从data目录发现全量币种，不再依赖固定10币种。
    using_dynamic_train = args.train_symbols == _SENTINEL_DYNAMIC_SYMBOLS
    using_dynamic_test = args.test_symbols == _SENTINEL_DYNAMIC_SYMBOLS
    using_dynamic_backtest = args.backtest_symbols == _SENTINEL_DYNAMIC_SYMBOLS

    if using_dynamic_train or using_dynamic_test or using_dynamic_backtest:
        all_symbols = _discover_data_symbols(project_root)
        train_symbols, test_symbols = _split_symbols_4_to_1(all_symbols)

        if using_dynamic_train:
            args.train_symbols = train_symbols
        if using_dynamic_test:
            args.test_symbols = test_symbols
        if using_dynamic_backtest:
            args.backtest_symbols = all_symbols

        print(
            "[Symbols] 已启用data目录动态全量币种: "
            f"total={len(all_symbols)}, train={len(args.train_symbols)}, "
            f"test={len(args.test_symbols)}, backtest={len(args.backtest_symbols)}"
        )

    # 默认并行度做内存预留：保留约30%可用内存，避免占满机器影响其他任务。
    if psutil:
        available_gb = max(1.0, psutil.virtual_memory().available / (1024 ** 3))
        default_num = _resolve_worker_count(
            requested=args.num_workers,
            cpu_default=_DEFAULT_NUM_WORKERS,
            available_mem_gb=available_gb,
            reserve_ratio=_TARGET_MEMORY_RESERVE_RATIO,
            memory_per_worker_gb=1.6,
            min_workers=1,
        )
        default_feature = _resolve_worker_count(
            requested=args.feature_symbol_workers,
            cpu_default=_DEFAULT_FEATURE_SYMBOL_WORKERS,
            available_mem_gb=available_gb,
            reserve_ratio=_TARGET_MEMORY_RESERVE_RATIO,
            memory_per_worker_gb=1.2,
            min_workers=1,
        )
        default_backtest = _resolve_worker_count(
            requested=args.symbol_workers,
            cpu_default=_DEFAULT_SYMBOL_WORKERS,
            available_mem_gb=available_gb,
            reserve_ratio=_TARGET_MEMORY_RESERVE_RATIO,
            memory_per_worker_gb=0.8,
            min_workers=1,
        )

        args.num_workers = min(default_num, max(1, len(args.train_symbols)))
        args.feature_symbol_workers = min(default_feature, max(1, len(args.train_symbols)))
        args.symbol_workers = min(default_backtest, max(1, len(args.backtest_symbols)))
        print(
            "[Workers] memory-safe并发已应用: "
            f"num_workers={args.num_workers}, "
            f"feature_symbol_workers={args.feature_symbol_workers}, "
            f"symbol_workers={args.symbol_workers}"
        )

    if bool(args.clear_cache):
        clear_ns = str(args.cache_namespace or "").strip() or "default"
        clear_cmd = [
            sys.executable,
            str(project_root / "Debug" / "xgboost_shap_train.py"),
            "--clear-cache",
            "--cache-mode",
            str(args.cache_mode),
            "--cache-namespace",
            clear_ns,
        ]
        print("[Pipeline] clear-cache: 清空训练缓存并退出")
        subprocess.run(clear_cmd, check=True)
        return

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

        batch_root = _artifact_root(Path(args.debug_root), args.artifact_kind)
        summary_path = batch_root / f"{args.batch_run_prefix}_summary_{batch_ts}.json"
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
