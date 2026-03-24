from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List


@dataclass
class TuneResult:
    threshold: float
    total_return_pct: float
    max_drawdown_pct: float
    profit_factor: float
    total_trades: float
    score: float
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
    parser.add_argument("--signal-margin", type=float, default=0.0)
    parser.add_argument("--cooldown-bars", type=int, default=0)
    parser.add_argument("--debug-root", default="Debug")
    parser.add_argument("--run-prefix", default="threshold_tune")
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


def _score(metrics: dict) -> float:
    total_return = _safe_float(metrics.get("total_return_pct"), -1e9)
    drawdown = abs(_safe_float(metrics.get("max_drawdown_pct"), 100.0))
    profit_factor = _safe_float(metrics.get("profit_factor"), 0.0)
    # Higher is better: reward return/profit-factor, penalize drawdown.
    return total_return + (profit_factor - 1.0) * 25.0 - drawdown * 0.35


def main() -> None:
    args = _build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    python_exec = sys.executable

    thresholds = _frange(args.threshold_start, args.threshold_stop, args.threshold_step)
    if not thresholds:
        raise ValueError("No threshold generated, please check start/stop/step")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_records: List[TuneResult] = []

    for threshold in thresholds:
        threshold_tag = str(threshold).replace(".", "p")
        run_id = f"{args.run_prefix}_{timestamp}_th{threshold_tag}"
        cmd = [
            python_exec,
            str(project_root / "Debug" / "run_pipeline.py"),
            "--mode",
            "predict-backtest",
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
            "--signal-margin",
            str(args.signal_margin),
            "--cooldown-bars",
            str(args.cooldown_bars),
            "--skip-predict",
        ]
        print(f"[Tune] running threshold={threshold:.4f} run_id={run_id}")
        _run_cmd(cmd)

        metrics_path = (
            project_root
            / args.debug_root
            / "runs"
            / run_id
            / "backtest"
            / "backtest_metrics.json"
        )
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        agg = payload.get("aggregate", {})
        record = TuneResult(
            threshold=threshold,
            total_return_pct=_safe_float(agg.get("total_return_pct")),
            max_drawdown_pct=_safe_float(agg.get("max_drawdown_pct")),
            profit_factor=_safe_float(agg.get("profit_factor")),
            total_trades=_safe_float(agg.get("total_trades")),
            score=_score(agg),
            run_id=run_id,
            metrics_path=str(metrics_path),
        )
        run_records.append(record)

    best = max(run_records, key=lambda r: r.score)

    summary = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_run_id": args.source_run_id,
        "symbols": args.symbols,
        "begin_time": args.begin_time,
        "end_time": args.end_time,
        "signal_margin": args.signal_margin,
        "cooldown_bars": args.cooldown_bars,
        "best": best.__dict__,
        "all": [item.__dict__ for item in sorted(run_records, key=lambda r: r.threshold)],
    }

    out_path = (
        project_root
        / args.debug_root
        / "runs"
        / f"{args.run_prefix}_{timestamp}_summary.json"
    )
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n[Tune] completed")
    print(f"[Tune] best threshold={best.threshold:.4f}, score={best.score:.4f}")
    print(f"[Tune] best run_id={best.run_id}")
    print(f"[Tune] summary={out_path}")


if __name__ == "__main__":
    main()
