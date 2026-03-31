from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def _safe_float(v: object, default: float = 0.0) -> float:
    try:
        x = float(v)
        if np.isfinite(x):
            return x
    except Exception:
        pass
    return float(default)


def _load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _nearest_threshold_row(rows: List[Dict[str, object]], target: float) -> Dict[str, object]:
    if not rows:
        return {}
    return min(rows, key=lambda x: abs(_safe_float(x.get("threshold"), 0.0) - target))


def _best_threshold_row(rows: List[Dict[str, object]], base_trades: float) -> Dict[str, object]:
    if not rows:
        return {}

    trade_min = max(30.0, base_trades * 0.60)
    trade_max = max(trade_min, base_trades * 1.40)
    constrained = [
        r
        for r in rows
        if trade_min <= _safe_float(r.get("trade_count"), 0.0) <= trade_max
    ]
    pool = constrained if constrained else rows

    def _rank_key(row: Dict[str, object]) -> tuple[float, float, float]:
        return (
            _safe_float(row.get("sharpe"), 0.0),
            _safe_float(row.get("total_return_pct"), 0.0),
            -abs(_safe_float(row.get("max_drawdown_pct"), 0.0)),
        )

    return max(pool, key=_rank_key)


def _confidence_drift(primary: Dict[str, object]) -> Dict[str, object]:
    buckets = list(primary.get("confidence_bucket_returns", []) or [])
    if len(buckets) < 3:
        return {
            "detected": False,
            "message": "insufficient buckets",
        }

    low_bucket = buckets[0]
    high_bucket = buckets[-1]
    low_ret = _safe_float(low_bucket.get("mean_return_pct"), 0.0)
    high_ret = _safe_float(high_bucket.get("mean_return_pct"), 0.0)

    detected = bool(high_ret < low_ret)
    return {
        "detected": detected,
        "low_bucket": low_bucket,
        "high_bucket": high_bucket,
        "delta_mean_return_pct": float(high_ret - low_ret),
        "message": "high confidence has lower return than low confidence" if detected else "bucket return ordering acceptable",
    }


def _build_findings(
    train_report: Dict[str, object],
    diagnostics: Dict[str, object],
) -> List[Dict[str, object]]:
    findings: List[Dict[str, object]] = []

    primary = diagnostics.get("primary", {})
    meta = diagnostics.get("meta", {})
    summary = diagnostics.get("summary", {})

    class_dist = primary.get("class_distribution", {})
    direction_bias = _safe_float(class_dist.get("direction_bias"), 0.0)
    if abs(direction_bias) >= 0.08:
        findings.append(
            {
                "severity": "high",
                "area": "primary",
                "title": "Direction bias is too strong",
                "detail": {
                    "direction_bias": direction_bias,
                    "pred_pt_ratio": _safe_float(class_dist.get("pred_pt_ratio"), 0.0),
                    "label_pt_ratio": _safe_float(class_dist.get("label_pt_ratio"), 0.0),
                },
            }
        )

    bsp_rows = list(primary.get("bsp_type_accuracy", []) or [])
    if bsp_rows:
        worst_by_return = min(bsp_rows, key=lambda x: _safe_float(x.get("mean_return_pct"), 0.0))
        if _safe_float(worst_by_return.get("mean_return_pct"), 0.0) < -0.02:
            findings.append(
                {
                    "severity": "medium",
                    "area": "primary",
                    "title": "One BSP type drags return",
                    "detail": worst_by_return,
                }
            )

    conf_drift = _confidence_drift(primary)
    if bool(conf_drift.get("detected", False)):
        findings.append(
            {
                "severity": "high",
                "area": "primary",
                "title": "Confidence-return relation is inverted",
                "detail": conf_drift,
            }
        )

    stability = primary.get("time_stability", {})
    positive_month_ratio = _safe_float(stability.get("positive_month_ratio"), 0.0)
    min_ratio = _safe_float(
        train_report.get("pass_criteria", {}).get("positive_month_ratio_min", 0.70),
        0.70,
    )
    if positive_month_ratio < min_ratio:
        findings.append(
            {
                "severity": "high",
                "area": "framework",
                "title": "Time-segment stability below constraint",
                "detail": {
                    "positive_month_ratio": positive_month_ratio,
                    "required_min": min_ratio,
                },
            }
        )

    threshold_rows = list(meta.get("threshold_scan", []) or [])
    if threshold_rows:
        current_threshold = _safe_float(train_report.get("architecture", {}).get("meta_threshold"), 0.55)
        current_row = _nearest_threshold_row(threshold_rows, current_threshold)
        current_trades = _safe_float(current_row.get("trade_count"), _safe_float(summary.get("sample_count"), 0.0) * _safe_float(summary.get("meta_keep_rate"), 0.0))
        best_row = _best_threshold_row(threshold_rows, max(30.0, current_trades))
        if best_row and current_row:
            cur_sharpe = _safe_float(current_row.get("sharpe"), 0.0)
            best_sharpe = _safe_float(best_row.get("sharpe"), 0.0)
            if best_sharpe > cur_sharpe + 0.15:
                findings.append(
                    {
                        "severity": "medium",
                        "area": "meta",
                        "title": "Meta threshold has room for improvement",
                        "detail": {
                            "current": current_row,
                            "candidate": best_row,
                        },
                    }
                )

    calibration = meta.get("calibration", {})
    brier = _safe_float(calibration.get("brier"), 0.0)
    logloss = _safe_float(calibration.get("logloss"), 0.0)
    if brier > 0.20 or logloss > 0.60:
        findings.append(
            {
                "severity": "medium",
                "area": "meta",
                "title": "Calibration quality is weak",
                "detail": {
                    "brier": brier,
                    "logloss": logloss,
                },
            }
        )

    fr_fa = meta.get("false_reject_false_accept", {})
    fr = _safe_float(fr_fa.get("false_reject_ratio"), 0.0)
    fa = _safe_float(fr_fa.get("false_accept_ratio"), 0.0)
    if fr > 0.30 or fa > 0.30:
        findings.append(
            {
                "severity": "medium",
                "area": "meta",
                "title": "False-reject or false-accept ratio too high",
                "detail": {
                    "false_reject_ratio": fr,
                    "false_accept_ratio": fa,
                },
            }
        )

    dep = meta.get("feature_dependency", {})
    top1_share = _safe_float(dep.get("top_feature_abs_share"), 0.0)
    top5_share = _safe_float(dep.get("top5_abs_share"), 0.0)
    if top1_share > 0.35 or top5_share > 0.80:
        findings.append(
            {
                "severity": "medium",
                "area": "meta",
                "title": "Meta feature dependency is too concentrated",
                "detail": {
                    "top_feature_abs_share": top1_share,
                    "top5_abs_share": top5_share,
                    "top_features": dep.get("top_features", [])[:5],
                },
            }
        )

    return findings


def _default_actions() -> List[str]:
    return [
        "Primary: enable dynamic pt multiplier by volatility regime and keep PT/SL binary labels.",
        "Primary: continue feature pruning for stable non-positive MDA features; add cross-timeframe consistency features.",
        "Primary: run small-step hyperparameter search (20-40 trials) and rank by backtest objective, not CV score only.",
        "Meta: enable probability calibration (platt or isotonic) before threshold search.",
        "Meta: use grouped thresholds by bsp_type and direction instead of single threshold.",
        "Framework: keep fee/slippage explicit and require positive period ratio in pass criteria.",
    ]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze primary/meta model issues from dual diagnostics")
    parser.add_argument("--run-id", required=True, help="run id under Debug/runs")
    parser.add_argument("--debug-root", default="Debug")
    parser.add_argument("--output-json", default="", help="default: Debug/runs/<run-id>/train/dual_model_issue_focus.json")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    run_root = project_root / args.debug_root / "runs" / args.run_id
    train_dir = run_root / "train"

    diagnostics_path = train_dir / "dual_model_diagnostics.json"
    train_report_path = train_dir / "train_validator_report.json"

    if not diagnostics_path.exists():
        raise FileNotFoundError(f"Missing diagnostics: {diagnostics_path}")
    if not train_report_path.exists():
        raise FileNotFoundError(f"Missing train report: {train_report_path}")

    diagnostics = _load_json(diagnostics_path)
    train_report = _load_json(train_report_path)

    findings = _build_findings(train_report=train_report, diagnostics=diagnostics)
    findings_sorted = sorted(
        findings,
        key=lambda x: {"high": 0, "medium": 1, "low": 2}.get(str(x.get("severity", "low")), 3),
    )

    output = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "run_id": args.run_id,
        "train_dir": str(train_dir),
        "inputs": {
            "train_report": str(train_report_path),
            "dual_model_diagnostics": str(diagnostics_path),
        },
        "summary": {
            "finding_count": int(len(findings_sorted)),
            "high_count": int(sum(1 for x in findings_sorted if str(x.get("severity")) == "high")),
            "medium_count": int(sum(1 for x in findings_sorted if str(x.get("severity")) == "medium")),
        },
        "findings": findings_sorted,
        "recommended_actions": _default_actions(),
    }

    out_path = Path(args.output_json) if str(args.output_json).strip() else (train_dir / "dual_model_issue_focus.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")

    print("[IssueAnalysis] completed")
    print(f"[IssueAnalysis] run_id={args.run_id}")
    print(f"[IssueAnalysis] finding_count={len(findings_sorted)}")
    print(f"[IssueAnalysis] output={out_path}")


if __name__ == "__main__":
    main()
