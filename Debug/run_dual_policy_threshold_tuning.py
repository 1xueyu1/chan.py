from __future__ import annotations

# flake8: noqa: E501

import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List


@dataclass
class PolicyConfig:
    name: str
    run_prefix: str
    objective_trade_max_ratio: float


def _safe_float(v: object, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _latest_summary_by_prefix(tests_dir: Path, prefix: str) -> Path:
    cands = sorted(
        tests_dir.glob(f"{prefix}_*_summary.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not cands:
        raise FileNotFoundError(f"No summary found for prefix={prefix}")
    return cands[0]


def _run_one_policy(project_root: Path, policy: PolicyConfig) -> Dict[str, object]:
    py = sys.executable
    debug_script = project_root / "Debug" / "tune_backtest_threshold.py"

    cmd = [
        py,
        str(debug_script),
        "--source-run-id",
        "current_model_full_20260326",
        "--source-artifact-kind",
        "run",
        "--artifact-kind",
        "test",
        "--begin-time",
        "2020-01-01",
        "--end-time",
        "2026-02-28",
        "--symbols",
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
        "--threshold-start",
        "0.50",
        "--threshold-stop",
        "0.55",
        "--threshold-step",
        "0.025",
        "--symbol-workers",
        "3",
        "--parallel-mode",
        "process",
        "--signal-margin",
        "0.0",
        "--cooldown-bars",
        "0",
        "--backtest-fee",
        "0.0004",
        "--backtest-slippage",
        "0.0001",
        "--meta-threshold-by-bsp",
        "3=0.62,sell_3=0.65",
        "--backtest-fast-mode",
        "--run-prefix",
        policy.run_prefix,
        "--max-concurrent-runs",
        "3",
        "--resume-existing",
        "--objective-trade-max-ratio",
        str(policy.objective_trade_max_ratio),
    ]

    print(f"\n[DualPolicy] Running policy={policy.name}")
    print(f"[DualPolicy] trade_max_ratio={policy.objective_trade_max_ratio}")
    rc = subprocess.call(cmd)
    if rc != 0:
        raise RuntimeError(f"policy={policy.name} failed with exit={rc}")

    tests_dir = project_root / "Debug" / "tests"
    summary_path = _latest_summary_by_prefix(tests_dir, policy.run_prefix)
    payload = json.loads(summary_path.read_text(encoding="utf-8"))

    best = payload.get("best", {})
    return {
        "policy": policy.name,
        "trade_max_ratio": policy.objective_trade_max_ratio,
        "summary_path": str(summary_path),
        "passed_count": int(payload.get("passed_count", 0) or 0),
        "total_count": int(payload.get("total_count", 0) or 0),
        "selection_rule": str(payload.get("selection_rule", "")),
        "best_threshold": _safe_float(best.get("threshold")),
        "best_score": _safe_float(best.get("score")),
        "best_passed_constraints": bool(best.get("passed_constraints", False)),
        "best_annualized_return_pct": _safe_float(best.get("annualized_return_pct")),
        "best_max_drawdown_pct": _safe_float(best.get("max_drawdown_pct")),
        "best_sharpe": _safe_float(best.get("sharpe")),
        "best_total_trades": _safe_float(best.get("total_trades")),
        "best_run_id": str(best.get("run_id", "")),
    }


def _to_markdown(rows: List[Dict[str, object]], out_json: Path) -> str:
    lines: List[str] = []
    lines.append("# 双策略阈值调优对比（自动执行）")
    lines.append("")
    lines.append(f"- created_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("- source: current_model_full_20260326")
    lines.append(f"- result_json: {out_json}")
    lines.append("")
    lines.append("| policy | trade_max_ratio | passed_count/total | selection_rule | best_threshold | best_score | ann_ret% | mdd% | sharpe | trades | passed_constraints |")
    lines.append("|---|---:|---|---|---:|---:|---:|---:|---:|---:|---|")
    for r in rows:
        lines.append(
            "| {policy} | {trade_max_ratio:.2f} | {passed_count}/{total_count} | {selection_rule} | "
            "{best_threshold:.3f} | {best_score:.4f} | {best_annualized_return_pct:.4f} | "
            "{best_max_drawdown_pct:.4f} | {best_sharpe:.4f} | {best_total_trades:.0f} | {best_passed_constraints} |".format(**r)
        )

    lines.append("")
    lines.append("## 结论")
    if len(rows) >= 2:
        strict = rows[0]
        balanced = rows[1]
        lines.append(
            "- 约束优先与收益优先在本轮的最佳解已自动对比，可直接按 `passed_count` 与 `best_passed_constraints` 判断是否满足硬约束。"
        )
        lines.append(
            "- 若两者都未满足硬约束，优先查看 `best_total_trades` 与 `best_threshold`，再决定是放宽交易数上限还是继续细化阈值网格。"
        )
        lines.append(
            f"- strict best: threshold={strict['best_threshold']:.3f}, score={strict['best_score']:.4f}, trades={strict['best_total_trades']:.0f}"
        )
        lines.append(
            f"- balanced best: threshold={balanced['best_threshold']:.3f}, score={balanced['best_score']:.4f}, trades={balanced['best_total_trades']:.0f}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    policies = [
        PolicyConfig(
            name="strict_constraints_first",
            run_prefix=f"dual_policy_strict_{stamp}",
            objective_trade_max_ratio=1.40,
        ),
        PolicyConfig(
            name="balanced_return_with_control",
            run_prefix=f"dual_policy_balanced_{stamp}",
            objective_trade_max_ratio=2.20,
        ),
    ]

    rows: List[Dict[str, object]] = []
    for policy in policies:
        rows.append(_run_one_policy(project_root, policy))

    out_json = project_root / "Debug" / "tests" / f"dual_policy_compare_{stamp}.json"
    out_md = project_root / "Debug" / "tests" / f"dual_policy_compare_{stamp}.md"

    out_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    out_md.write_text(_to_markdown(rows, out_json), encoding="utf-8")

    print("\n[DualPolicy] done")
    print(f"[DualPolicy] json={out_json}")
    print(f"[DualPolicy] md={out_md}")


if __name__ == "__main__":
    main()
