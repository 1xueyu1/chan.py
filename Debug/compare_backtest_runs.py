from __future__ import annotations

# flake8: noqa: E501

import argparse
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def _safe_float(v: object, default: float = 0.0) -> float:
    try:
        out = float(v)
        if math.isnan(out) or math.isinf(out):
            return float(default)
        return out
    except Exception:
        return float(default)


def _per_symbol_rows(metrics: Dict[str, object]) -> List[Dict[str, object]]:
    rows = metrics.get("per_symbol", []) or []
    if not rows:
        # Backward compatibility for older report schema.
        rows = metrics.get("symbols", []) or []
    return [row for row in rows if isinstance(row, dict)]


def _artifact_root(project_root: Path, debug_root: str, kind: str) -> Path:
    sub = "tests" if str(kind).strip().lower() == "test" else "runs"
    return project_root / debug_root / sub


def _load_metrics(
    project_root: Path,
    debug_root: str,
    run_id: str,
    artifact_kind: str,
) -> Dict[str, object]:
    preferred_root = _artifact_root(project_root, debug_root, artifact_kind)
    preferred = preferred_root / run_id / "backtest" / "backtest_metrics.json"
    if preferred.exists():
        return json.loads(preferred.read_text(encoding="utf-8"))

    fallback_kind = "test" if str(artifact_kind).strip().lower() == "run" else "run"
    fallback_root = _artifact_root(project_root, debug_root, fallback_kind)
    fallback = fallback_root / run_id / "backtest" / "backtest_metrics.json"
    if fallback.exists():
        print(
            "[Compare][WARN] artifact kind mismatch, "
            f"fallback to {fallback_kind}: {fallback}"
        )
        return json.loads(fallback.read_text(encoding="utf-8"))

    raise FileNotFoundError(f"metrics file not found: {preferred} ; {fallback}")


def _summarize_symbols(metrics: Dict[str, object]) -> Dict[str, float]:
    rows = _per_symbol_rows(metrics)
    values = [_safe_float(r.get("total_return_pct")) for r in rows]
    if not values:
        return {
            "symbol_count": 0.0,
            "positive_count": 0.0,
            "negative_count": 0.0,
            "positive_ratio": 0.0,
            "median_total_return_pct": 0.0,
            "worst_total_return_pct": 0.0,
            "best_total_return_pct": 0.0,
            "deep_drawdown_60_count": 0.0,
        }

    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mid = n // 2
    median = sorted_vals[mid] if n % 2 == 1 else (sorted_vals[mid - 1] + sorted_vals[mid]) / 2.0

    positive = sum(1 for v in sorted_vals if v > 0)
    negative = sum(1 for v in sorted_vals if v < 0)
    deep_dd = 0
    for row in rows:
        mdd = _safe_float(row.get("max_drawdown_pct"))
        if mdd <= -60.0:
            deep_dd += 1

    return {
        "symbol_count": float(n),
        "positive_count": float(positive),
        "negative_count": float(negative),
        "positive_ratio": float(positive) / float(n),
        "median_total_return_pct": float(median),
        "worst_total_return_pct": float(sorted_vals[0]),
        "best_total_return_pct": float(sorted_vals[-1]),
        "deep_drawdown_60_count": float(deep_dd),
    }


def _symbol_return_map(metrics: Dict[str, object]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for row in _per_symbol_rows(metrics):
        symbol = str(row.get("symbol") or "").strip().upper()
        if not symbol:
            continue
        out[symbol] = _safe_float(row.get("total_return_pct"))
    return out


def _top_symbol_deltas(
    old_map: Dict[str, float],
    new_map: Dict[str, float],
    top_n: int,
) -> Dict[str, List[Dict[str, float | str]]]:
    rows: List[Dict[str, float | str]] = []
    for symbol, old_ret in old_map.items():
        if symbol not in new_map:
            continue
        new_ret = new_map[symbol]
        delta = new_ret - old_ret
        rows.append(
            {
                "symbol": symbol,
                "old_total_return_pct": old_ret,
                "new_total_return_pct": new_ret,
                "delta_total_return_pct": delta,
            }
        )

    improved = sorted(rows, key=lambda x: float(x["delta_total_return_pct"]), reverse=True)[:top_n]
    worsened = sorted(rows, key=lambda x: float(x["delta_total_return_pct"]))[:top_n]
    return {
        "top_improved": improved,
        "top_worsened": worsened,
    }


def _to_markdown(compare: Dict[str, object]) -> str:
    old_run_id = str(compare["old_run_id"])
    new_run_id = str(compare["new_run_id"])
    agg_rows = compare["aggregate_rows"]
    symbol_rows = compare["symbol_distribution_rows"]

    lines: List[str] = []
    lines.append("# 回测对比报告")
    lines.append("")
    lines.append(f"- created_at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"- old_run_id: {old_run_id}")
    lines.append(f"- new_run_id: {new_run_id}")
    lines.append("")

    lines.append("## 聚合指标")
    lines.append("")
    lines.append("| metric | old | new | delta(new-old) |")
    lines.append("|---|---:|---:|---:|")
    for row in agg_rows:
        lines.append(
            "| {metric} | {old:.6f} | {new:.6f} | {delta:.6f} |".format(**row)
        )

    lines.append("")
    lines.append("## 分标的分布")
    lines.append("")
    lines.append("| metric | old | new | delta(new-old) |")
    lines.append("|---|---:|---:|---:|")
    for row in symbol_rows:
        lines.append(
            "| {metric} | {old:.6f} | {new:.6f} | {delta:.6f} |".format(**row)
        )

    lines.append("")
    lines.append("## Top 改善标的")
    lines.append("")
    lines.append("| symbol | old_total_ret% | new_total_ret% | delta |")
    lines.append("|---|---:|---:|---:|")
    for row in compare["top_symbol_deltas"]["top_improved"]:
        lines.append(
            "| {symbol} | {old_total_return_pct:.6f} | {new_total_return_pct:.6f} | {delta_total_return_pct:.6f} |".format(**row)
        )

    lines.append("")
    lines.append("## Top 恶化标的")
    lines.append("")
    lines.append("| symbol | old_total_ret% | new_total_ret% | delta |")
    lines.append("|---|---:|---:|---:|")
    for row in compare["top_symbol_deltas"]["top_worsened"]:
        lines.append(
            "| {symbol} | {old_total_return_pct:.6f} | {new_total_return_pct:.6f} | {delta_total_return_pct:.6f} |".format(**row)
        )

    return "\n".join(lines) + "\n"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare two backtest runs and generate markdown/json report")
    parser.add_argument("--old-run-id", required=True)
    parser.add_argument("--new-run-id", required=True)
    parser.add_argument("--debug-root", default="Debug")
    parser.add_argument("--artifact-kind", choices=["run", "test"], default="run")
    parser.add_argument("--top-n", type=int, default=20)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]

    old_metrics = _load_metrics(
        project_root=project_root,
        debug_root=str(args.debug_root),
        run_id=str(args.old_run_id),
        artifact_kind=str(args.artifact_kind),
    )
    new_metrics = _load_metrics(
        project_root=project_root,
        debug_root=str(args.debug_root),
        run_id=str(args.new_run_id),
        artifact_kind=str(args.artifact_kind),
    )

    old_agg = old_metrics.get("aggregate", {}) or {}
    new_agg = new_metrics.get("aggregate", {}) or {}

    agg_metrics = [
        "total_return_pct",
        "annualized_return_pct",
        "max_drawdown_pct",
        "sharpe",
        "profit_factor",
        "win_rate_pct",
        "total_trades",
    ]

    aggregate_rows = []
    for metric in agg_metrics:
        old_val = _safe_float(old_agg.get(metric))
        new_val = _safe_float(new_agg.get(metric))
        aggregate_rows.append(
            {
                "metric": metric,
                "old": old_val,
                "new": new_val,
                "delta": new_val - old_val,
            }
        )

    old_symbol_summary = _summarize_symbols(old_metrics)
    new_symbol_summary = _summarize_symbols(new_metrics)
    symbol_metrics = [
        "symbol_count",
        "positive_count",
        "negative_count",
        "positive_ratio",
        "median_total_return_pct",
        "worst_total_return_pct",
        "best_total_return_pct",
        "deep_drawdown_60_count",
    ]

    symbol_distribution_rows = []
    for metric in symbol_metrics:
        old_val = _safe_float(old_symbol_summary.get(metric))
        new_val = _safe_float(new_symbol_summary.get(metric))
        symbol_distribution_rows.append(
            {
                "metric": metric,
                "old": old_val,
                "new": new_val,
                "delta": new_val - old_val,
            }
        )

    symbol_deltas = _top_symbol_deltas(
        old_map=_symbol_return_map(old_metrics),
        new_map=_symbol_return_map(new_metrics),
        top_n=max(1, int(args.top_n)),
    )

    compare = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "old_run_id": str(args.old_run_id),
        "new_run_id": str(args.new_run_id),
        "aggregate_rows": aggregate_rows,
        "symbol_distribution_rows": symbol_distribution_rows,
        "top_symbol_deltas": symbol_deltas,
    }

    out_dir = _artifact_root(project_root, str(args.debug_root), str(args.artifact_kind)) / str(args.new_run_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_json = out_dir / f"backtest_compare_vs_{args.old_run_id}.json"
    out_md = out_dir / f"backtest_compare_vs_{args.old_run_id}.md"
    out_json.write_text(json.dumps(compare, indent=2, ensure_ascii=False), encoding="utf-8")
    out_md.write_text(_to_markdown(compare), encoding="utf-8")

    print("[Compare] completed")
    print(f"[Compare] json={out_json}")
    print(f"[Compare] md={out_md}")


if __name__ == "__main__":
    main()
