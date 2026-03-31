from __future__ import annotations

# flake8: noqa: E501

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class TuneCandidateResult:
    calibration: str
    train_run_id: str
    bsp_threshold_map: str
    direction_threshold_map: str
    tune_summary_path: str
    best_subrun_id: str
    best_threshold: float
    objective_score: float
    passed_constraints: bool
    metrics_path: str


def _safe_float(v: object, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _artifact_root(project_root: Path, debug_root: str, kind: str) -> Path:
    sub = "tests" if str(kind).strip().lower() == "test" else "runs"
    return project_root / debug_root / sub


def _latest_summary_by_prefix(root_dir: Path, prefix: str) -> Path:
    cands = sorted(
        root_dir.glob(f"{prefix}_*_summary.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not cands:
        raise FileNotFoundError(f"No summary found for prefix={prefix} under {root_dir}")
    return cands[0]


def _resolve_metrics_path(
    project_root: Path,
    debug_root: str,
    run_id: str,
    artifact_kind: str,
) -> Path:
    preferred_root = _artifact_root(project_root, debug_root, artifact_kind)
    preferred = preferred_root / run_id / "backtest" / "backtest_metrics.json"
    if preferred.exists():
        return preferred

    fallback_kind = "test" if str(artifact_kind).strip().lower() == "run" else "run"
    fallback_root = _artifact_root(project_root, debug_root, fallback_kind)
    fallback = fallback_root / run_id / "backtest" / "backtest_metrics.json"
    if fallback.exists():
        print(
            "[Suite][WARN] metrics path artifact kind mismatch, "
            f"fallback to {fallback_kind}: {fallback}"
        )
        return fallback

    raise FileNotFoundError(
        "Backtest metrics not found in both locations: "
        f"{preferred}; {fallback}"
    )


def _resolve_train_dir(
    project_root: Path,
    debug_root: str,
    run_id: str,
    artifact_kind: str,
) -> Path:
    preferred_root = _artifact_root(project_root, debug_root, artifact_kind)
    preferred = preferred_root / run_id / "train"
    if preferred.exists():
        return preferred

    fallback_kind = "test" if str(artifact_kind).strip().lower() == "run" else "run"
    fallback_root = _artifact_root(project_root, debug_root, fallback_kind)
    fallback = fallback_root / run_id / "train"
    if fallback.exists():
        print(
            "[Suite][WARN] train dir artifact kind mismatch, "
            f"fallback to {fallback_kind}: {fallback}"
        )
        return fallback

    raise FileNotFoundError(
        "Train dir not found in both locations: "
        f"{preferred}; {fallback}"
    )


def _run_cmd(cmd: List[str]) -> None:
    rc = subprocess.call(cmd)
    if rc != 0:
        raise RuntimeError(f"Command failed rc={rc}: {' '.join(cmd)}")


def _parse_reuse_map(raw: str) -> Dict[str, str]:
    text = str(raw or "").strip()
    if not text:
        return {}
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("--reuse-calibration-run-map must be a JSON object")
    out: Dict[str, str] = {}
    for k, v in payload.items():
        key = str(k).strip().lower()
        val = str(v).strip()
        if key and val:
            out[key] = val
    return out


def _build_train_cmd(
    python_exec: str,
    project_root: Path,
    args: argparse.Namespace,
    run_id: str,
    calibration: str,
) -> List[str]:
    cmd = [
        python_exec,
        str(project_root / "Debug" / "run_pipeline.py"),
        "--mode",
        "full",
        "--stage",
        "train",
        "--artifact-kind",
        str(args.train_artifact_kind),
        "--run-id",
        run_id,
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
        "--meta-calibration",
        calibration,
        "--meta-calibration-ratio",
        str(args.meta_calibration_ratio),
        "--pass-positive-month-ratio",
        str(args.pass_positive_month_ratio),
        "--cache-mode",
        str(args.train_cache_mode),
    ]

    if str(args.train_cache_namespace).strip():
        cmd.extend(["--cache-namespace", str(args.train_cache_namespace)])

    if args.train_symbols:
        cmd.extend(["--train-symbols", *args.train_symbols])
    if args.test_symbols:
        cmd.extend(["--test-symbols", *args.test_symbols])

    if bool(args.dynamic_pt_enabled):
        cmd.append("--dynamic-pt-enabled")
    if not bool(args.enable_backup_run):
        cmd.append("--no-backup-run")

    return cmd


def _build_tune_cmd(
    python_exec: str,
    project_root: Path,
    args: argparse.Namespace,
    train_run_id: str,
    bsp_threshold_map: str,
    direction_threshold_map: str,
    run_prefix: str,
) -> List[str]:
    cmd = [
        python_exec,
        str(project_root / "Debug" / "tune_backtest_threshold.py"),
        "--source-run-id",
        train_run_id,
        "--source-artifact-kind",
        str(args.train_artifact_kind),
        "--artifact-kind",
        str(args.tune_artifact_kind),
        "--debug-root",
        str(args.debug_root),
        "--begin-time",
        args.begin_time,
        "--end-time",
        args.end_time,
        "--threshold-start",
        str(args.threshold_start),
        "--threshold-stop",
        str(args.threshold_stop),
        "--threshold-step",
        str(args.threshold_step),
        "--symbol-workers",
        str(args.symbol_workers),
        "--parallel-mode",
        str(args.backtest_parallel_mode),
        "--event-cache-dir",
        str(args.event_cache_dir),
        "--event-cache-namespace",
        str(args.event_cache_namespace or train_run_id),
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
        "--run-prefix",
        run_prefix,
        "--max-concurrent-runs",
        str(args.max_concurrent_runs),
    ]

    if args.backtest_symbols:
        cmd.extend(["--symbols", *args.backtest_symbols])

    if not bool(args.event_cache):
        cmd.append("--no-event-cache")
    if bool(args.empty_signal_fallback):
        cmd.append("--empty-signal-fallback")
    if not bool(args.save_trades_csv):
        cmd.append("--no-save-trades-csv")
    if not bool(args.save_equity_csv):
        cmd.append("--no-save-equity-csv")
    if bool(args.backtest_fast_mode):
        cmd.append("--backtest-fast-mode")
    if bool(args.resume_existing):
        cmd.append("--resume-existing")
    if bool(args.enable_backup_run):
        cmd.append("--enable-backup-run")

    if str(bsp_threshold_map).strip():
        cmd.extend(["--meta-threshold-by-bsp", str(bsp_threshold_map)])
    if str(direction_threshold_map).strip():
        cmd.extend(["--meta-threshold-by-direction", str(direction_threshold_map)])

    return cmd


def _select_best(candidates: List[TuneCandidateResult]) -> TuneCandidateResult:
    passed = [c for c in candidates if c.passed_constraints]
    pool = passed if passed else candidates
    return max(pool, key=lambda c: c.objective_score)


def _to_markdown(
    suite_id: str,
    baseline_run_id: str,
    baseline_metrics_path: str,
    rows: List[Dict[str, object]],
    out_json: Path,
) -> str:
    lines: List[str] = []
    lines.append("# Meta Calibration + Threshold 搜索联动对比")
    lines.append("")
    lines.append(f"- suite_id: {suite_id}")
    lines.append(f"- created_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- baseline_run_id: {baseline_run_id or '-'}")
    lines.append(f"- baseline_metrics_path: {baseline_metrics_path or '-'}")
    lines.append(f"- result_json: {out_json}")
    lines.append("")
    lines.append("| calibration | train_run_id | best_threshold | bsp_map | direction_map | score | passed | ann_ret% | mdd% | sharpe | total_ret% | trades | d_ret% | d_mdd% | d_sharpe |")
    lines.append("|---|---|---:|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    for row in rows:
        lines.append(
            "| {calibration} | {train_run_id} | {best_threshold:.4f} | {bsp_threshold_map} | {direction_threshold_map} | "
            "{objective_score:.4f} | {passed_constraints} | {annualized_return_pct:.4f} | {max_drawdown_pct:.4f} | "
            "{sharpe:.4f} | {total_return_pct:.4f} | {total_trades:.0f} | {delta_total_return_pct:.4f} | "
            "{delta_max_drawdown_pct:.4f} | {delta_sharpe:.4f} |".format(**row)
        )

    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run meta-calibration suite (none/platt/isotonic), then perform BSP/方向阈值搜索, "
            "and auto-generate comparison report"
        )
    )

    parser.add_argument("--debug-root", default="Debug")
    parser.add_argument("--begin-time", default="2020-01-01")
    parser.add_argument("--end-time", default="2026-02-28")

    parser.add_argument("--run-prefix", default="meta_calibration_suite")
    parser.add_argument("--baseline-run-id", default="")
    parser.add_argument(
        "--reuse-calibration-run-map",
        default="",
        help='JSON object, e.g. {"none":"run_a","platt":"run_b","isotonic":"run_c"}',
    )

    parser.add_argument(
        "--calibrations",
        nargs="+",
        default=["platt", "isotonic"],
        help="calibration list to execute",
    )
    parser.add_argument(
        "--include-none-calibration",
        action="store_true",
        help="include calibration=none in this suite",
    )
    parser.add_argument("--meta-calibration-ratio", type=float, default=0.2)
    parser.add_argument("--pass-positive-month-ratio", type=float, default=0.70)

    parser.add_argument("--train-artifact-kind", choices=["run", "test"], default="run")
    parser.add_argument("--tune-artifact-kind", choices=["run", "test"], default="test")
    parser.add_argument("--report-artifact-kind", choices=["run", "test"], default="run")

    parser.add_argument("--train-mode", choices=["cpu", "gpu", "auto"], default="auto")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--feature-symbol-workers", type=int, default=4)
    parser.add_argument("--symbol-workers", type=int, default=8)
    parser.add_argument("--backtest-parallel-mode", choices=["process", "thread"], default="process")

    parser.add_argument("--train-symbols", nargs="+", default=[])
    parser.add_argument("--test-symbols", nargs="+", default=[])
    parser.add_argument("--backtest-symbols", nargs="+", default=[])

    parser.add_argument("--threshold-start", type=float, default=0.45)
    parser.add_argument("--threshold-stop", type=float, default=0.75)
    parser.add_argument("--threshold-step", type=float, default=0.02)
    parser.add_argument("--signal-margin", type=float, default=0.0)
    parser.add_argument("--cooldown-bars", type=int, default=0)
    parser.add_argument("--backtest-fee", type=float, default=0.0004)
    parser.add_argument("--backtest-slippage", type=float, default=0.0001)

    parser.add_argument(
        "--bsp-threshold-candidate",
        action="append",
        default=[],
        help="single BSP map candidate, e.g. 3=0.62,sell_3=0.65 ; can repeat",
    )
    parser.add_argument(
        "--direction-threshold-candidate",
        action="append",
        default=[],
        help="single direction map candidate, e.g. buy=0.55,sell=0.60 ; can repeat",
    )

    parser.add_argument("--max-concurrent-runs", type=int, default=1)
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--backtest-fast-mode", action="store_true")

    parser.add_argument(
        "--event-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="event cache switch for tuned backtests",
    )
    parser.add_argument("--event-cache-dir", default="data/cache/backtest_events")
    parser.add_argument("--event-cache-namespace", default="")
    parser.add_argument(
        "--empty-signal-fallback",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--empty-signal-target-rate", type=float, default=0.05)
    parser.add_argument(
        "--save-trades-csv",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--save-equity-csv",
        action=argparse.BooleanOptionalAction,
        default=True,
    )

    parser.add_argument("--train-cache-mode", choices=["resume", "fresh"], default="resume")
    parser.add_argument("--train-cache-namespace", default="")

    parser.add_argument(
        "--enable-backup-run",
        action="store_true",
        help="enable run backup for train/tune subprocesses",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    project_root = Path(__file__).resolve().parents[1]
    python_exec = sys.executable
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suite_id = f"{args.run_prefix}_{timestamp}"

    report_root = _artifact_root(project_root, str(args.debug_root), str(args.report_artifact_kind)) / suite_id
    report_root.mkdir(parents=True, exist_ok=True)

    reuse_map = _parse_reuse_map(args.reuse_calibration_run_map)

    calibrations = [str(c).strip().lower() for c in args.calibrations if str(c).strip()]
    if bool(args.include_none_calibration) and "none" not in calibrations:
        calibrations = ["none", *calibrations]
    calibrations = list(dict.fromkeys(calibrations))

    if not calibrations:
        raise ValueError("No calibration specified")

    bsp_candidates = [str(v).strip() for v in args.bsp_threshold_candidate if str(v).strip()]
    direction_candidates = [str(v).strip() for v in args.direction_threshold_candidate if str(v).strip()]
    if not bsp_candidates:
        bsp_candidates = [""]
    if not direction_candidates:
        direction_candidates = [""]

    all_candidates: List[TuneCandidateResult] = []
    best_rows: List[Dict[str, object]] = []

    for calibration in calibrations:
        if calibration in reuse_map:
            train_run_id = reuse_map[calibration]
            _resolve_train_dir(
                project_root=project_root,
                debug_root=str(args.debug_root),
                run_id=train_run_id,
                artifact_kind=str(args.train_artifact_kind),
            )
            print(f"[Suite] calibration={calibration} reuse train_run_id={train_run_id}")
        else:
            train_run_id = f"{suite_id}_{calibration}_train"
            train_cmd = _build_train_cmd(
                python_exec=python_exec,
                project_root=project_root,
                args=args,
                run_id=train_run_id,
                calibration=calibration,
            )
            print(f"[Suite] calibration={calibration} train run_id={train_run_id}")
            _run_cmd(train_cmd)

        per_cal_candidates: List[TuneCandidateResult] = []
        pair_idx = 0
        for bsp_map in bsp_candidates:
            for direction_map in direction_candidates:
                pair_idx += 1
                tune_prefix = f"{suite_id}_{calibration}_pair{pair_idx:02d}"
                tune_cmd = _build_tune_cmd(
                    python_exec=python_exec,
                    project_root=project_root,
                    args=args,
                    train_run_id=train_run_id,
                    bsp_threshold_map=bsp_map,
                    direction_threshold_map=direction_map,
                    run_prefix=tune_prefix,
                )
                print(
                    "[Suite] tune "
                    f"calibration={calibration} pair={pair_idx} bsp='{bsp_map or '-'}' direction='{direction_map or '-'}'"
                )
                _run_cmd(tune_cmd)

                summary_path = _latest_summary_by_prefix(
                    _artifact_root(project_root, str(args.debug_root), str(args.tune_artifact_kind)),
                    tune_prefix,
                )
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                best = summary.get("best", {}) or {}

                candidate = TuneCandidateResult(
                    calibration=calibration,
                    train_run_id=train_run_id,
                    bsp_threshold_map=bsp_map,
                    direction_threshold_map=direction_map,
                    tune_summary_path=str(summary_path),
                    best_subrun_id=str(best.get("run_id", "")),
                    best_threshold=_safe_float(best.get("threshold")),
                    objective_score=_safe_float(best.get("score")),
                    passed_constraints=bool(best.get("passed_constraints", False)),
                    metrics_path=str(best.get("metrics_path", "")),
                )
                per_cal_candidates.append(candidate)
                all_candidates.append(candidate)

        if not per_cal_candidates:
            raise RuntimeError(f"No tune candidate generated for calibration={calibration}")

        best_candidate = _select_best(per_cal_candidates)

        metrics_payload = json.loads(Path(best_candidate.metrics_path).read_text(encoding="utf-8"))
        agg = metrics_payload.get("aggregate", {}) or {}

        best_rows.append(
            {
                "calibration": calibration,
                "train_run_id": best_candidate.train_run_id,
                "bsp_threshold_map": best_candidate.bsp_threshold_map or "-",
                "direction_threshold_map": best_candidate.direction_threshold_map or "-",
                "tune_summary_path": best_candidate.tune_summary_path,
                "best_subrun_id": best_candidate.best_subrun_id,
                "best_threshold": best_candidate.best_threshold,
                "objective_score": best_candidate.objective_score,
                "passed_constraints": best_candidate.passed_constraints,
                "metrics_path": best_candidate.metrics_path,
                "annualized_return_pct": _safe_float(agg.get("annualized_return_pct")),
                "max_drawdown_pct": _safe_float(agg.get("max_drawdown_pct")),
                "sharpe": _safe_float(agg.get("sharpe")),
                "total_return_pct": _safe_float(agg.get("total_return_pct")),
                "total_trades": _safe_float(agg.get("total_trades")),
                "win_rate_pct": _safe_float(agg.get("win_rate_pct")),
            }
        )

    baseline_metrics_path = ""
    baseline_agg: Dict[str, float] = {}
    baseline_run_id = str(args.baseline_run_id or "").strip()
    if baseline_run_id:
        baseline_metrics = _resolve_metrics_path(
            project_root=project_root,
            debug_root=str(args.debug_root),
            run_id=baseline_run_id,
            artifact_kind=str(args.train_artifact_kind),
        )
        baseline_metrics_path = str(baseline_metrics)
        payload = json.loads(baseline_metrics.read_text(encoding="utf-8"))
        baseline_agg = payload.get("aggregate", {}) or {}

    baseline_ret = _safe_float(baseline_agg.get("total_return_pct"))
    baseline_mdd = _safe_float(baseline_agg.get("max_drawdown_pct"))
    baseline_sharpe = _safe_float(baseline_agg.get("sharpe"))

    for row in best_rows:
        row["delta_total_return_pct"] = _safe_float(row["total_return_pct"]) - baseline_ret
        row["delta_max_drawdown_pct"] = _safe_float(row["max_drawdown_pct"]) - baseline_mdd
        row["delta_sharpe"] = _safe_float(row["sharpe"]) - baseline_sharpe

    payload = {
        "suite_id": suite_id,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "debug_root": args.debug_root,
            "begin_time": args.begin_time,
            "end_time": args.end_time,
            "calibrations": calibrations,
            "bsp_candidates": bsp_candidates,
            "direction_candidates": direction_candidates,
            "threshold_grid": {
                "start": args.threshold_start,
                "stop": args.threshold_stop,
                "step": args.threshold_step,
            },
            "signal_margin": args.signal_margin,
            "cooldown_bars": args.cooldown_bars,
            "event_cache": bool(args.event_cache),
            "event_cache_dir": str(args.event_cache_dir),
            "event_cache_namespace": str(args.event_cache_namespace),
            "empty_signal_fallback": bool(args.empty_signal_fallback),
            "empty_signal_target_rate": args.empty_signal_target_rate,
            "save_trades_csv": bool(args.save_trades_csv),
            "save_equity_csv": bool(args.save_equity_csv),
            "backtest_fast_mode": bool(args.backtest_fast_mode),
            "max_concurrent_runs": int(args.max_concurrent_runs),
        },
        "baseline": {
            "run_id": baseline_run_id,
            "metrics_path": baseline_metrics_path,
            "aggregate": baseline_agg,
        },
        "best_by_calibration": best_rows,
        "all_candidates": [item.__dict__ for item in all_candidates],
    }

    out_json = report_root / "suite_comparison.json"
    out_md = report_root / "suite_comparison.md"
    out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    out_md.write_text(
        _to_markdown(
            suite_id=suite_id,
            baseline_run_id=baseline_run_id,
            baseline_metrics_path=baseline_metrics_path,
            rows=best_rows,
            out_json=out_json,
        ),
        encoding="utf-8",
    )

    print("[Suite] completed")
    print(f"[Suite] report_json={out_json}")
    print(f"[Suite] report_md={out_md}")


if __name__ == "__main__":
    main()
