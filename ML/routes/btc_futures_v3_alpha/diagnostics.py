from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ML.shared.json_utils import json_safe

from .dataset import DEFAULT_DATASET_PATH, DEFAULT_MODEL_DIR, ROUTE_NAME, build_btc_futures_v3_alpha_dataset


DEFAULT_DIAGNOSTICS_DIR = DEFAULT_MODEL_DIR / "diagnostics"


def _profit_factor(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    wins = float(values.loc[values > 0].sum())
    losses = float(-values.loc[values < 0].sum())
    if losses <= 0:
        return float("inf") if wins > 0 else float("nan")
    return float(wins / losses)


def _mean_col(frame: pd.DataFrame, col: str) -> float:
    if col not in frame.columns:
        return float("nan")
    return float(pd.to_numeric(frame[col], errors="coerce").mean())


def _agg(group: pd.DataFrame) -> pd.Series:
    returns = pd.to_numeric(group["net_return"], errors="coerce")
    labels = pd.to_numeric(group["label"], errors="coerce")
    return pd.Series(
        {
            "rows": int(len(group)),
            "label_rate": float(labels.mean()),
            "avg_net_return": float(returns.mean()),
            "profit_factor": _profit_factor(returns),
            "tp_count": int((group["exit_reason"].astype(str) == "take_profit").sum()) if "exit_reason" in group else 0,
            "sl_count": int((group["exit_reason"].astype(str) == "stop_loss").sum()) if "exit_reason" in group else 0,
            "timeout_count": int((group["exit_reason"].astype(str) == "timeout").sum()) if "exit_reason" in group else 0,
            "avg_target_pct": _mean_col(group, "target_pct"),
            "avg_alpha_structure_score": _mean_col(group, "v3_alpha_structure_score"),
            "avg_consensus": _mean_col(group, "v3_htf_consensus_score"),
            "avg_conflict": _mean_col(group, "v3_htf_conflict_score"),
            "avg_boundary": _mean_col(group, "v3_boundary_score"),
            "avg_maturity": _mean_col(group, "v3_maturity_score"),
        }
    )


def _fmt_pct(value: float) -> str:
    return "N/A" if not np.isfinite(value) else f"{value:.2%}"


def _fmt_float(value: float) -> str:
    if np.isinf(value):
        return "inf"
    return "N/A" if not np.isfinite(value) else f"{value:.4f}"


def _write_markdown(summary: dict[str, Any], pool_summary: pd.DataFrame, yearly: pd.DataFrame, path: Path) -> None:
    lines = [
        "# BTC 合约 v3 Alpha 结构池诊断",
        "",
        "## 目的",
        "",
        "这份报告只做信号池体检，不做调参。它用已经修复过未来函数口径的数据，观察不同缠论结构池在历史年份中的胜率、平均净收益和稳定性。",
        "",
        "## 总览",
        "",
        f"- 路线：`{summary['route']}`",
        f"- 数据集：`{summary['dataset_path']}`",
        f"- 样本数：{summary['rows']}",
        f"- 整体标签胜率：{_fmt_pct(summary['label_rate'])}",
        f"- 整体平均净收益：{_fmt_pct(summary['avg_net_return'])}",
        "",
        "## 结构池汇总",
        "",
        "| 结构池 | 方向 | 样本数 | 胜率 | 平均净收益 | PF | Alpha分 | 共振 | 冲突 | 边界 | 成熟度 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    show = pool_summary.sort_values(["avg_net_return", "label_rate", "rows"], ascending=[False, False, False]).head(40)
    for _, row in show.iterrows():
        side = "多" if bool(row["is_buy"]) else "空"
        lines.append(
            f"| {row['v3_structure_pool']} | {side} | {int(row['rows'])} | "
            f"{_fmt_pct(row['label_rate'])} | {_fmt_pct(row['avg_net_return'])} | {_fmt_float(row['profit_factor'])} | "
            f"{_fmt_float(row['avg_alpha_structure_score'])} | {_fmt_float(row['avg_consensus'])} | "
            f"{_fmt_float(row['avg_conflict'])} | {_fmt_float(row['avg_boundary'])} | {_fmt_float(row['avg_maturity'])} |"
        )
    lines.extend(
        [
            "",
            "## 年度稳定性",
            "",
            "| 年份 | 结构池 | 方向 | 样本数 | 胜率 | 平均净收益 | PF |",
            "|---:|---|---|---:|---:|---:|---:|",
        ]
    )
    yearly_show = yearly.sort_values(["year", "avg_net_return", "rows"], ascending=[True, False, False])
    for _, row in yearly_show.head(120).iterrows():
        side = "多" if bool(row["is_buy"]) else "空"
        lines.append(
            f"| {int(row['year'])} | {row['v3_structure_pool']} | {side} | {int(row['rows'])} | "
            f"{_fmt_pct(row['label_rate'])} | {_fmt_pct(row['avg_net_return'])} | {_fmt_float(row['profit_factor'])} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def _group_apply(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    grouped = frame.groupby(keys, dropna=False)
    try:
        return grouped.apply(_agg, include_groups=False).reset_index()
    except TypeError:
        return grouped.apply(_agg).reset_index()


def run_structure_pool_diagnostics(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    output_dir: str | Path = DEFAULT_DIAGNOSTICS_DIR,
    build_if_missing: bool = True,
) -> dict[str, Any]:
    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        if not build_if_missing:
            raise FileNotFoundError(f"dataset not found: {dataset_file}")
        build_btc_futures_v3_alpha_dataset(output_path=dataset_file)

    frame = pd.read_parquet(dataset_file)
    for col in ("exec_time", "entry_time", "exit_time"):
        if col in frame.columns:
            frame[col] = pd.to_datetime(frame[col], utc=True, errors="coerce")
    frame = frame.dropna(subset=["label", "exec_time"]).copy()
    frame["year"] = frame["exec_time"].dt.year.astype(int)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    pool_summary = _group_apply(frame, ["v3_structure_pool", "is_buy"])
    yearly = _group_apply(frame, ["year", "v3_structure_pool", "is_buy"])
    state_summary = _group_apply(frame, ["v3_market_state", "v3_structure_pool", "is_buy"])

    pool_summary.to_csv(out / "pool_summary.csv", index=False)
    yearly.to_csv(out / "pool_yearly_summary.csv", index=False)
    state_summary.to_csv(out / "pool_market_state_summary.csv", index=False)

    summary = {
        "route": ROUTE_NAME,
        "dataset_path": str(dataset_file),
        "output_dir": str(out),
        "rows": int(len(frame)),
        "label_rate": float(pd.to_numeric(frame["label"], errors="coerce").mean()),
        "avg_net_return": float(pd.to_numeric(frame["net_return"], errors="coerce").mean()),
        "pool_count": int(frame["v3_structure_pool"].nunique(dropna=True)),
        "year_range": [int(frame["year"].min()), int(frame["year"].max())],
    }
    (out / "diagnostics_summary.json").write_text(json.dumps(json_safe(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(summary, pool_summary, yearly, out / "POOL_DIAGNOSTICS.md")
    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run v3 structure-pool diagnostics")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_DIAGNOSTICS_DIR))
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()
    summary = run_structure_pool_diagnostics(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        build_if_missing=not bool(args.no_build),
    )
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
