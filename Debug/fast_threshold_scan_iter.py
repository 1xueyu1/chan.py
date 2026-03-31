from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ml_layer.utils.optimization_objective import BacktestObjectiveConfig, score_backtest_metrics


def _safe_float(v: object, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    python_exec = sys.executable

    source_run_id = "full_allparts_20260327_1"
    symbols = [
        "ADAUSDT",
        "AVAXUSDT",
        "BNBUSDT",
        "BTCUSDT",
        "DOGEUSDT",
        "DOTUSDT",
        "ETHUSDT",
        "LTCUSDT",
        "SOLUSDT",
        "XRPUSDT",
    ]
    thresholds = [0.45, 0.50, 0.55, 0.60, 0.65, 0.70]

    baseline_metrics_path = (
        project_root
        / "Debug"
        / "runs"
        / source_run_id
        / "backtest"
        / "backtest_metrics.json"
    )
    baseline_payload = json.loads(baseline_metrics_path.read_text(encoding="utf-8"))
    baseline_aggregate = baseline_payload.get("aggregate", {})

    objective_cfg = BacktestObjectiveConfig()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    records = []

    for threshold in thresholds:
        run_id = f"fast_tune_iter1_{ts}_th{str(threshold).replace('.', 'p')}"
        cmd = [
            python_exec,
            str(project_root / "Debug" / "run_pipeline.py"),
            "--mode",
            "predict-backtest",
            "--source-run-id",
            source_run_id,
            "--run-id",
            run_id,
            "--begin-time",
            "2020-01-01",
            "--end-time",
            "2026-02-28",
            "--backtest-symbols",
            *symbols,
            "--signal-threshold",
            str(threshold),
            "--symbol-workers",
            "4",
            "--backtest-parallel-mode",
            "process",
            "--signal-margin",
            "0.0",
            "--cooldown-bars",
            "0",
            "--skip-predict",
            "--backtest-fast-mode",
            "--no-save-html-detail-report",
        ]

        print(f"[FAST-TUNE] threshold={threshold:.2f} run_id={run_id}")
        rc = subprocess.call(cmd)
        if rc != 0:
            raise RuntimeError(f"Backtest run failed: threshold={threshold:.2f}, rc={rc}")

        metrics_path = (
            project_root
            / "Debug"
            / "runs"
            / run_id
            / "backtest"
            / "backtest_metrics.json"
        )
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        agg = payload.get("aggregate", {})

        objective = score_backtest_metrics(
            baseline_metrics=baseline_aggregate,
            candidate_metrics=agg,
            config=objective_cfg,
        )

        records.append(
            {
                "threshold": threshold,
                "run_id": run_id,
                "metrics_path": str(metrics_path),
                "annualized_return_pct": _safe_float(agg.get("annualized_return_pct")),
                "max_drawdown_pct": _safe_float(agg.get("max_drawdown_pct")),
                "sharpe": _safe_float(agg.get("sharpe")),
                "total_trades": _safe_float(agg.get("total_trades")),
                "total_return_pct": _safe_float(agg.get("total_return_pct")),
                "objective": objective.to_dict(),
                "score": float(objective.score),
                "passed_constraints": bool(objective.passed_constraints),
            }
        )

    passed = [r for r in records if r["passed_constraints"]]
    best_pool = passed if passed else records
    best = max(best_pool, key=lambda x: x["score"])

    summary = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source_run_id": source_run_id,
        "objective_config": asdict(objective_cfg),
        "baseline_metrics_path": str(baseline_metrics_path),
        "baseline_aggregate": baseline_aggregate,
        "selection_rule": "best score among passed constraints" if passed else "fallback best score",
        "passed_count": len(passed),
        "total_count": len(records),
        "best": best,
        "all": sorted(records, key=lambda x: x["threshold"]),
    }

    out = project_root / "Debug" / "runs" / f"fast_threshold_tune_iter1_{ts}_summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[FAST-TUNE] completed")
    print(f"[FAST-TUNE] best_threshold={best['threshold']:.2f}")
    print(f"[FAST-TUNE] best_run_id={best['run_id']}")
    print(f"[FAST-TUNE] summary={out}")


if __name__ == "__main__":
    main()
