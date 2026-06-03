from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ML.shared.json_utils import json_safe

from .dataset import DEFAULT_DATASET_PATH, DEFAULT_MODEL_DIR, ROUTE_NAME, build_btc_futures_v3_beta_edge_dataset


DEFAULT_DIAGNOSTICS_DIR = DEFAULT_MODEL_DIR / "diagnostics"


def _profit_factor(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    wins = float(values.loc[values > 0].sum())
    losses = float(-values.loc[values < 0].sum())
    if losses <= 0:
        return float("inf") if wins > 0 else float("nan")
    return float(wins / losses)


def _agg(group: pd.DataFrame) -> pd.Series:
    def mean_col(name: str) -> float:
        if name not in group.columns:
            return float("nan")
        return float(pd.to_numeric(group[name], errors="coerce").mean())

    return pd.Series(
        {
            "rows": int(len(group)),
            "events": int(group["beta_event_id"].nunique()),
            "tp_first_rate": float(pd.to_numeric(group["label_tp_first"], errors="coerce").mean()),
            "entry_timing_rate": float("nan"),
            "tradeable_rate": float("nan"),
            "avg_net_return": float(pd.to_numeric(group["net_return"], errors="coerce").mean()),
            "profit_factor": _profit_factor(group["net_return"]),
            "avg_delay": mean_col("beta_entry_delay_minutes"),
            "avg_confirmation": mean_col("beta_confirmation_strength"),
        }
    )


def _group_apply(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    grouped = frame.groupby(keys, dropna=False)
    try:
        return grouped.apply(_agg, include_groups=False).reset_index()
    except TypeError:
        return grouped.apply(_agg).reset_index()


def _fmt_pct(value: float) -> str:
    return "N/A" if not np.isfinite(value) else f"{value:.2%}"


def _fmt_float(value: float) -> str:
    if np.isinf(value):
        return "inf"
    return "N/A" if not np.isfinite(value) else f"{value:.4f}"


def _write_markdown(summary: dict[str, Any], pool: pd.DataFrame, delay: pd.DataFrame, path: Path) -> None:
    lines = [
        "# BTC 合约 v3 Beta Edge 诊断",
        "",
        "## 总览",
        "",
        f"- 样本行数：{summary['rows']}",
        f"- 原始事件数：{summary['events']}",
        f"- TP first 标签率：{_fmt_pct(summary['tp_first_rate'])}",
        f"- 入场时机标签率：{_fmt_pct(summary['entry_timing_rate'])}",
        f"- 可交易标签率：{_fmt_pct(summary['tradeable_rate'])}",
        "",
        "## 结构池",
        "",
        "| 结构池 | 方向 | 行数 | 事件数 | 可交易率 | 平均净收益 | PF |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in pool.sort_values(["tradeable_rate", "avg_net_return"], ascending=[False, False]).head(40).iterrows():
        side = "多" if bool(row["is_buy"]) else "空"
        lines.append(
            f"| {row['v3_structure_pool']} | {side} | {int(row['rows'])} | {int(row['events'])} | "
            f"{_fmt_pct(row['tradeable_rate'])} | {_fmt_pct(row['avg_net_return'])} | {_fmt_float(row['profit_factor'])} |"
        )
    lines.extend(["", "## 延迟入场", "", "| 延迟分钟 | 方向 | 行数 | 可交易率 | 平均净收益 | PF |", "|---:|---|---:|---:|---:|---:|"])
    for _, row in delay.sort_values(["beta_entry_delay_minutes", "is_buy"]).iterrows():
        side = "多" if bool(row["is_buy"]) else "空"
        lines.append(
            f"| {int(row['beta_entry_delay_minutes'])} | {side} | {int(row['rows'])} | "
            f"{_fmt_pct(row['tradeable_rate'])} | {_fmt_pct(row['avg_net_return'])} | {_fmt_float(row['profit_factor'])} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_beta_edge_diagnostics(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    output_dir: str | Path = DEFAULT_DIAGNOSTICS_DIR,
    build_if_missing: bool = True,
) -> dict[str, Any]:
    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        if not build_if_missing:
            raise FileNotFoundError(f"dataset not found: {dataset_file}")
        build_btc_futures_v3_beta_edge_dataset(output_path=dataset_file)
    frame = pd.read_parquet(dataset_file)
    frame["exec_time"] = pd.to_datetime(frame["exec_time"], utc=True, errors="coerce")
    frame["year"] = frame["exec_time"].dt.year.astype(int)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pool = _group_apply(frame, ["v3_structure_pool", "is_buy"])
    delay = _group_apply(frame, ["beta_entry_delay_minutes", "is_buy"])
    yearly = _group_apply(frame, ["year", "v3_structure_pool", "is_buy"])
    pool.to_csv(out / "pool_summary.csv", index=False)
    delay.to_csv(out / "delay_summary.csv", index=False)
    yearly.to_csv(out / "year_pool_summary.csv", index=False)
    summary = {
        "route": ROUTE_NAME,
        "dataset_path": str(dataset_file),
        "output_dir": str(out),
        "rows": int(len(frame)),
        "events": int(frame["beta_event_id"].nunique()),
        "tp_first_rate": float(pd.to_numeric(frame["label_tp_first"], errors="coerce").mean()),
        "entry_timing_rate": float("nan"),
        "tradeable_rate": float("nan"),
        "avg_net_return": float(pd.to_numeric(frame["net_return"], errors="coerce").mean()),
    }
    (out / "diagnostics_summary.json").write_text(json.dumps(json_safe(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(summary, pool, delay, out / "BETA_EDGE_DIAGNOSTICS.md")
    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run v3 beta edge diagnostics")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_DIAGNOSTICS_DIR))
    parser.add_argument("--no-build", action="store_true")
    args = parser.parse_args()
    summary = run_beta_edge_diagnostics(args.dataset, args.output_dir, build_if_missing=not bool(args.no_build))
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
