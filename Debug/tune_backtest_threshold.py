from __future__ import annotations

# flake8: noqa: E402, E501

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from ml_layer.utils.optimization_objective import BacktestObjectiveConfig, score_backtest_metrics


@dataclass
class TuneResult:
    threshold: float
    total_return_pct: float
    annualized_return_pct: float
    max_drawdown_pct: float
    sharpe: float
    profit_factor: float
    total_trades: float
    score: float
    passed_constraints: bool
    objective: dict
    run_id: str
    metrics_path: str


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Grid-search signal threshold by backtest metrics",
    )
    parser.add_argument("--source-run-id", required=True)
    parser.add_argument("--begin-time", default="2020-01-01")
    parser.add_argument("--end-time", default="2026-02-28")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
    )
    parser.add_argument("--threshold-start", type=float, default=0.45)
    parser.add_argument("--threshold-stop", type=float, default=0.75)
    parser.add_argument("--threshold-step", type=float, default=0.02)
    parser.add_argument("--symbol-workers", type=int, default=4)
    parser.add_argument("--parallel-mode", choices=["process", "thread"], default="process")
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
        default="",
        help="回测事件缓存命名空间（为空则复用 source-run-id）",
    )
    parser.add_argument("--signal-margin", type=float, default=0.0)
    parser.add_argument("--cooldown-bars", type=int, default=0)
    parser.add_argument(
        "--empty-signal-fallback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="空信号补单开关（默认关闭）",
    )
    parser.add_argument(
        "--empty-signal-target-rate",
        type=float,
        default=0.05,
        help="空信号补单目标比例（仅 fallback 开启时生效）",
    )
    parser.add_argument("--backtest-fee", type=float, default=0.0004)
    parser.add_argument("--backtest-slippage", type=float, default=0.0001)
    parser.add_argument(
        "--save-trades-csv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否导出 executed_trades.csv（默认开启）",
    )
    parser.add_argument(
        "--save-equity-csv",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="是否导出 portfolio_equity_curve.csv（默认开启）",
    )
    parser.add_argument("--meta-threshold-by-bsp", default="")
    parser.add_argument("--meta-threshold-by-direction", default="")
    parser.add_argument(
        "--max-concurrent-runs",
        type=int,
        default=1,
        help="并发执行的阈值子任务数量（每个子任务内部仍可按symbol并行）",
    )
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="若子run的 backtest_metrics.json 已存在，则跳过重跑并直接复用",
    )
    parser.add_argument(
        "--artifact-kind",
        choices=["run", "test"],
        default="test",
        help="本次阈值迭代产物类别；默认归档到 Debug/tests",
    )
    parser.add_argument(
        "--source-artifact-kind",
        choices=["run", "test"],
        default="run",
        help="source-run-id 所在产物类别",
    )
    parser.add_argument(
        "--enable-backup-run",
        action="store_true",
        help="默认关闭每轮阈值子run备份；显式开启时会保留每轮备份快照",
    )
    parser.add_argument(
        "--backtest-fast-mode",
        action="store_true",
        default=False,
        help="将子run回测切换为fast-mode",
    )
    parser.add_argument("--debug-root", default="Debug")
    parser.add_argument("--run-prefix", default="threshold_tune")
    parser.add_argument("--objective-ret-weight", type=float, default=0.45)
    parser.add_argument("--objective-mdd-weight", type=float, default=0.25)
    parser.add_argument("--objective-sharpe-weight", type=float, default=0.20)
    parser.add_argument("--objective-trade-weight", type=float, default=0.10)
    parser.add_argument("--objective-mdd-tolerance", type=float, default=1.0)
    parser.add_argument("--objective-sharpe-tolerance", type=float, default=0.05)
    parser.add_argument("--objective-trade-min-ratio", type=float, default=0.60)
    parser.add_argument("--objective-trade-max-ratio", type=float, default=1.40)
    parser.add_argument("--objective-trade-abs-min", type=float, default=30.0)
    parser.add_argument("--objective-trade-abs-max", type=float, default=5000.0)
    return parser


def _frange(start: float, stop: float, step: float) -> List[float]:
    values: List[float] = []
    x = start
    while x <= stop + 1e-12:
        values.append(round(x, 4))
        x += step
    return values


def _run_cmd(cmd: List[str]) -> None:
    proc = subprocess.Popen(cmd)
    rc = proc.wait()
    if rc != 0:
        raise RuntimeError(f"Command failed with exit={rc}: {' '.join(cmd)}")


def _safe_float(value, default=0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _artifact_root(project_root: Path, debug_root: str, kind: str) -> Path:
    sub = "tests" if str(kind).strip().lower() == "test" else "runs"
    return project_root / debug_root / sub


def _strip_tail_timestamp(token: str) -> str:
    return re.sub(r"(?:_\d{8}(?:_\d{6})?)$", "", str(token or "").strip())


def _find_prefix_matched_baseline(
    root_dir: Path,
    source_run_id: str,
) -> Path | None:
    pattern = f"{source_run_id}*"
    cands = sorted(
        root_dir.glob(pattern),
        key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
        reverse=True,
    )
    for folder in cands:
        path = folder / "backtest" / "backtest_metrics.json"
        if path.exists():
            return path
    return None


def _resolve_baseline_metrics_path(
    project_root: Path,
    debug_root: str,
    source_run_id: str,
    source_artifact_kind: str,
) -> Path:
    preferred_root = _artifact_root(project_root, debug_root, source_artifact_kind)
    preferred = preferred_root / source_run_id / "backtest" / "backtest_metrics.json"
    if preferred.exists():
        return preferred

    fallback_kind = "test" if str(source_artifact_kind) == "run" else "run"
    fallback_root = _artifact_root(project_root, debug_root, fallback_kind)
    fallback = fallback_root / source_run_id / "backtest" / "backtest_metrics.json"
    if fallback.exists():
        print(
            "[Tune][WARN] source_artifact_kind 与实际目录不一致，"
            f"自动回退到 {fallback_kind}: {fallback}"
        )
        return fallback

    matched = _find_prefix_matched_baseline(preferred_root, source_run_id)
    if matched is not None:
        print(
            "[Tune][WARN] 未找到精确source-run-id，"
            f"自动匹配到最近前缀目录: {matched}"
        )
        return matched

    matched_fb = _find_prefix_matched_baseline(fallback_root, source_run_id)
    if matched_fb is not None:
        print(
            "[Tune][WARN] 未找到精确source-run-id，"
            f"在回退目录自动匹配到最近前缀目录: {matched_fb}"
        )
        return matched_fb

    raise FileNotFoundError(
        "Baseline metrics not found in both locations: "
        f"{preferred} ; {fallback}"
    )


def _build_subrun_cmd(
    args: argparse.Namespace,
    project_root: Path,
    python_exec: str,
    run_id: str,
    threshold: float,
) -> List[str]:
    cmd = [
        python_exec,
        str(project_root / "Debug" / "run_pipeline.py"),
        "--mode",
        "predict-backtest",
        "--artifact-kind",
        str(args.artifact_kind),
        "--source-artifact-kind",
        str(args.source_artifact_kind),
        "--source-run-id",
        args.source_run_id,
        "--run-id",
        run_id,
        "--begin-time",
        args.begin_time,
        "--end-time",
        args.end_time,
        "--backtest-symbols",
        *args.symbols,
        "--signal-threshold",
        str(threshold),
        "--symbol-workers",
        str(args.symbol_workers),
        "--backtest-parallel-mode",
        args.parallel_mode,
        "--event-cache-dir",
        str(args.event_cache_dir),
        "--event-cache-namespace",
        str(args.event_cache_namespace or args.source_run_id),
        "--signal-margin",
        str(args.signal_margin),
        "--cooldown-bars",
        str(args.cooldown_bars),
        "--empty-signal-target-rate",
        str(args.empty_signal_target_rate),
        "--backtest-fee",
        str(args.backtest_fee),
        "--backtest-slippage",
        str(args.backtest_slippage),
        "--skip-predict",
    ]
    if not bool(args.event_cache):
        cmd.append("--no-event-cache")
    if bool(args.empty_signal_fallback):
        cmd.append("--empty-signal-fallback")
    if not bool(args.save_trades_csv):
        cmd.append("--no-save-trades-csv")
    if not bool(args.save_equity_csv):
        cmd.append("--no-save-equity-csv")
    if not bool(args.enable_backup_run):
        cmd.append("--no-backup-run")
    if bool(args.backtest_fast_mode):
        cmd.append("--backtest-fast-mode")
        cmd.append("--no-save-html-detail-report")
    if str(args.meta_threshold_by_bsp).strip():
        cmd.extend(["--meta-threshold-by-bsp", str(args.meta_threshold_by_bsp)])
    if str(args.meta_threshold_by_direction).strip():
        cmd.extend(["--meta-threshold-by-direction", str(args.meta_threshold_by_direction)])
    return cmd


def _run_one_threshold(
    args: argparse.Namespace,
    project_root: Path,
    python_exec: str,
    run_id: str,
    threshold: float,
    baseline_aggregate: dict,
    objective_config: BacktestObjectiveConfig,
) -> TuneResult:
    metrics_path = (
        _artifact_root(project_root, args.debug_root, args.artifact_kind)
        / run_id
        / "backtest"
        / "backtest_metrics.json"
    )

    if not (bool(args.resume_existing) and metrics_path.exists()):
        cmd = _build_subrun_cmd(
            args=args,
            project_root=project_root,
            python_exec=python_exec,
            run_id=run_id,
            threshold=threshold,
        )
        _run_cmd(cmd)

    payload = json.loads(metrics_path.read_text(encoding="utf-8"))
    agg = payload.get("aggregate", {})
    objective = score_backtest_metrics(
        baseline_metrics=baseline_aggregate,
        candidate_metrics=agg,
        config=objective_config,
    )

    return TuneResult(
        threshold=threshold,
        total_return_pct=_safe_float(agg.get("total_return_pct")),
        annualized_return_pct=_safe_float(agg.get("annualized_return_pct")),
        max_drawdown_pct=_safe_float(agg.get("max_drawdown_pct")),
        sharpe=_safe_float(agg.get("sharpe")),
        profit_factor=_safe_float(agg.get("profit_factor")),
        total_trades=_safe_float(agg.get("total_trades")),
        score=float(objective.score),
        passed_constraints=bool(objective.passed_constraints),
        objective=objective.to_dict(),
        run_id=run_id,
        metrics_path=str(metrics_path),
    )


def main() -> None:
    args = _build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    python_exec = sys.executable

    objective_config = BacktestObjectiveConfig(
        ret_weight=float(args.objective_ret_weight),
        mdd_weight=float(args.objective_mdd_weight),
        sharpe_weight=float(args.objective_sharpe_weight),
        trade_weight=float(args.objective_trade_weight),
        mdd_worsen_tolerance_pct=float(args.objective_mdd_tolerance),
        sharpe_drop_tolerance=float(args.objective_sharpe_tolerance),
        trade_min_ratio=float(args.objective_trade_min_ratio),
        trade_max_ratio=float(args.objective_trade_max_ratio),
        trade_abs_min=float(args.objective_trade_abs_min),
        trade_abs_max=float(args.objective_trade_abs_max),
    )

    baseline_metrics_path = _resolve_baseline_metrics_path(
        project_root=project_root,
        debug_root=args.debug_root,
        source_run_id=args.source_run_id,
        source_artifact_kind=args.source_artifact_kind,
    )
    baseline_payload = json.loads(baseline_metrics_path.read_text(encoding="utf-8"))
    baseline_aggregate = baseline_payload.get("aggregate", {})

    thresholds = _frange(args.threshold_start, args.threshold_stop, args.threshold_step)
    if not thresholds:
        raise ValueError("No threshold generated, please check start/stop/step")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = _strip_tail_timestamp(args.run_prefix)
    if not prefix:
        prefix = "threshold_tune"

    if len(thresholds) > 1:
        print(
            f"[Tune] threshold grid count={len(thresholds)}, "
            f"range=[{thresholds[0]:.4f}, {thresholds[-1]:.4f}]"
        )

    run_records: List[TuneResult] = []

    run_plan = []
    for i, threshold in enumerate(thresholds, start=1):
        threshold_tag = str(threshold).replace(".", "p")
        run_id = f"{prefix}_{timestamp}_th{threshold_tag}"
        run_plan.append((i, threshold, run_id))

    max_workers = max(1, int(args.max_concurrent_runs))
    cpu_cnt = os.cpu_count() or 4
    est_worker_load = max_workers * max(1, int(args.symbol_workers))
    if est_worker_load > cpu_cnt * 2:
        print(
            "[Tune][WARN] 预计并发负载较高: "
            f"max_concurrent_runs({max_workers}) * symbol_workers({args.symbol_workers}) "
            f"= {est_worker_load}, CPU={cpu_cnt}. "
            "建议适当降低并发参数，避免资源争用。"
        )

    if max_workers <= 1:
        for i, threshold, run_id in run_plan:
            print(
                f"[Tune] ({i}/{len(thresholds)}) running "
                f"threshold={threshold:.4f} run_id={run_id}"
            )
            run_records.append(
                _run_one_threshold(
                    args=args,
                    project_root=project_root,
                    python_exec=python_exec,
                    run_id=run_id,
                    threshold=threshold,
                    baseline_aggregate=baseline_aggregate,
                    objective_config=objective_config,
                )
            )
    else:
        print(f"[Tune] concurrent mode enabled: max_concurrent_runs={max_workers}")
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            fut_to_meta = {}
            for i, threshold, run_id in run_plan:
                print(
                    f"[Tune] ({i}/{len(thresholds)}) queued "
                    f"threshold={threshold:.4f} run_id={run_id}"
                )
                fut = ex.submit(
                    _run_one_threshold,
                    args,
                    project_root,
                    python_exec,
                    run_id,
                    threshold,
                    baseline_aggregate,
                    objective_config,
                )
                fut_to_meta[fut] = (i, threshold, run_id)

            for fut in as_completed(fut_to_meta):
                i, threshold, run_id = fut_to_meta[fut]
                result = fut.result()
                run_records.append(result)
                print(
                    f"[Tune] ({i}/{len(thresholds)}) done "
                    f"threshold={threshold:.4f} run_id={run_id}"
                )

    passed = [r for r in run_records if r.passed_constraints]
    best_pool = passed if passed else run_records
    best = max(best_pool, key=lambda r: r.score)

    summary = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_run_id": args.source_run_id,
        "objective_version": objective_config.version,
        "objective_config": objective_config.__dict__,
        "baseline_metrics_path": str(baseline_metrics_path),
        "baseline_aggregate": baseline_aggregate,
        "symbols": args.symbols,
        "begin_time": args.begin_time,
        "end_time": args.end_time,
        "signal_margin": args.signal_margin,
        "cooldown_bars": args.cooldown_bars,
        "selection_rule": "best score among passed constraints"
        if passed
        else "fallback best score (no run passed hard constraints)",
        "passed_count": int(len(passed)),
        "total_count": int(len(run_records)),
        "best": best.__dict__,
        "all": [item.__dict__ for item in sorted(run_records, key=lambda r: r.threshold)],
    }

    out_path = (
        _artifact_root(project_root, args.debug_root, args.artifact_kind)
        / f"{prefix}_{timestamp}_summary.json"
    )
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n[Tune] completed")
    print(f"[Tune] best threshold={best.threshold:.4f}, score={best.score:.4f}")
    print(f"[Tune] constraints_passed={best.passed_constraints}")
    print(f"[Tune] best run_id={best.run_id}")
    print(f"[Tune] summary={out_path}")


if __name__ == "__main__":
    main()
