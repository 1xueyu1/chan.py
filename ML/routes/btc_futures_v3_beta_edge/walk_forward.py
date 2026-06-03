from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ML.routes.btc_futures_v1.data import parse_utc
from ML.shared.json_utils import json_safe

from .backtest import run_btc_futures_v3_beta_edge_backtest
from .dataset import DEFAULT_DATASET_PATH, ROUTE_NAME, build_btc_futures_v3_beta_edge_dataset
from .train import train_btc_futures_v3_beta_edge_models


DEFAULT_MODEL_ROOT = Path("result/ml/btc_futures_v3_beta_edge_walk_forward")
DEFAULT_OUTPUT_ROOT = Path("result/btc_futures_v3_beta_edge_walk_forward")


def _date(year: int) -> str:
    return f"{year:04d}-01-01"


def _year_end(year: int) -> str:
    return f"{year:04d}-12-31"


def _finite(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _fold_row(year: int, train_metrics: dict[str, Any], bt: dict[str, Any]) -> dict[str, Any]:
    models = train_metrics.get("models") or {}
    buy = models.get("buy") or {}
    sell = models.get("sell") or {}
    splits = train_metrics.get("splits") or {}
    return {
        "test_year": int(year),
        "train_rows": int((splits.get("train") or {}).get("rows", 0)),
        "valid_rows": int((splits.get("valid") or {}).get("rows", 0)),
        "confirm_rows": int((splits.get("confirm") or {}).get("rows", 0)),
        "test_rows": int((splits.get("test") or {}).get("rows", 0)),
        "buy_enabled": bool(buy.get("enabled", False)),
        "sell_enabled": bool(sell.get("enabled", False)),
        "buy_threshold": _finite(buy.get("threshold")),
        "sell_threshold": _finite(sell.get("threshold")),
        "buy_valid_auc": _finite(((buy.get("final") or {}).get("valid") or {}).get("auc")),
        "buy_confirm_auc": _finite(((buy.get("final") or {}).get("confirm") or {}).get("auc")),
        "sell_valid_auc": _finite(((sell.get("final") or {}).get("valid") or {}).get("auc")),
        "sell_confirm_auc": _finite(((sell.get("final") or {}).get("confirm") or {}).get("auc")),
        "trades": int(bt.get("trades", 0)),
        "qualified_candidates": int(bt.get("qualified_candidates", 0)),
        "win_rate": _finite(bt.get("win_rate")),
        "total_return": _finite(bt.get("total_return")),
        "profit_factor": _finite(bt.get("profit_factor")),
        "max_drawdown": _finite(bt.get("max_drawdown")),
    }


def _fmt_pct(value: float) -> str:
    return "N/A" if not np.isfinite(value) else f"{value:.2%}"


def _fmt_float(value: float) -> str:
    if np.isinf(value):
        return "inf"
    return "N/A" if not np.isfinite(value) else f"{value:.3f}"


def _write_markdown(summary: dict[str, Any], table: pd.DataFrame, path: Path) -> None:
    lines = [
        "# BTC 合约 v3 Beta Edge 年度 Walk-Forward",
        "",
        "## 口径",
        "",
        "- 每个测试年只使用过去数据训练。",
        "- 测试年前两年分别作为验证年和确认年。",
        "- 阈值必须在验证年和确认年同时过成本线。",
        "- 候选入场行只使用候选入场前的 1m 数据做确认特征。",
        "",
        "## 汇总",
        "",
        f"- Fold 数：{summary['folds']}",
        f"- 总交易数：{summary['total_trades']}",
        f"- 加权胜率：{_fmt_pct(summary['weighted_win_rate'])}",
        f"- 年度复合收益：{_fmt_pct(summary['compounded_return'])}",
        "",
        "| 年份 | 交易数 | 胜率 | 收益 | PF | 买确认AUC | 卖确认AUC |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"| {int(row['test_year'])} | {int(row['trades'])} | {_fmt_pct(row['win_rate'])} | "
            f"{_fmt_pct(row['total_return'])} | {_fmt_float(row['profit_factor'])} | "
            f"{_fmt_float(row['buy_confirm_auc'])} | {_fmt_float(row['sell_confirm_auc'])} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_yearly_walk_forward(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    model_root: str | Path = DEFAULT_MODEL_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    train_start: Any | None = "2021-01-01",
    first_test_year: int = 2024,
    last_test_year: int | None = None,
    model_kind: str = "stable_rank",
    min_final_auc: float = 0.55,
    min_precision: float = 0.55,
    min_profit_factor: float = 1.05,
    initial_cash: float = 100000.0,
) -> dict[str, Any]:
    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        build_btc_futures_v3_beta_edge_dataset(output_path=dataset_file)
    meta = pd.read_parquet(dataset_file, columns=["exec_time"])
    meta["exec_time"] = pd.to_datetime(meta["exec_time"], utc=True, errors="coerce")
    data_end = meta["exec_time"].max()
    final_year = int(last_test_year or pd.Timestamp(data_end).year)
    model_root = Path(model_root)
    output_root = Path(output_root)
    model_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    train_start_ts = parse_utc(train_start)
    for year in range(int(first_test_year), final_year + 1):
        valid_start = _date(year - 2)
        confirm_start = _date(year - 1)
        test_start = _date(year)
        test_end = _date(year + 1)
        if train_start_ts is not None and parse_utc(valid_start) <= train_start_ts:
            continue
        if parse_utc(test_start) > data_end:
            continue
        fold_model_dir = model_root / f"test_{year}"
        fold_output_dir = output_root / f"test_{year}"
        print(
            f"[WF-BETA] year={year} train=[{train_start},{valid_start}) "
            f"valid=[{valid_start},{confirm_start}) confirm=[{confirm_start},{test_start}) test=[{test_start},{test_end})",
            flush=True,
        )
        train_metrics = train_btc_futures_v3_beta_edge_models(
            dataset_path=dataset_file,
            model_dir=fold_model_dir,
            train_start=train_start,
            valid_start=valid_start,
            confirm_start=confirm_start,
            test_start=test_start,
            test_end=test_end,
            model_kind=model_kind,
            min_final_auc=min_final_auc,
            min_precision=min_precision,
            min_profit_factor=min_profit_factor,
        )
        bt_end = _year_end(year)
        if year == int(pd.Timestamp(data_end).year):
            bt_end = data_end.strftime("%Y-%m-%d")
        bt = run_btc_futures_v3_beta_edge_backtest(
            dataset_path=dataset_file,
            model_dir=fold_model_dir,
            output_dir=fold_output_dir,
            begin_time=test_start,
            end_time=bt_end,
            initial_cash=initial_cash,
        )
        row = _fold_row(year, train_metrics, bt)
        rows.append(row)
        print(f"[WF-BETA] done year={year} trades={row['trades']} return={_fmt_pct(row['total_return'])}", flush=True)
    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("no beta edge walk-forward folds were produced")
    table.to_csv(output_root / "fold_summary.csv", index=False)
    total_trades = int(table["trades"].sum())
    win_table = table.loc[(table["trades"] > 0) & table["win_rate"].notna()]
    weighted_win_rate = (
        float(np.average(win_table["win_rate"], weights=win_table["trades"]))
        if int(win_table["trades"].sum()) > 0
        else float("nan")
    )
    summary = {
        "route": ROUTE_NAME,
        "dataset_path": str(dataset_file),
        "model_root": str(model_root),
        "output_root": str(output_root),
        "folds": int(len(table)),
        "total_trades": total_trades,
        "weighted_win_rate": weighted_win_rate,
        "compounded_return": float(np.prod(1.0 + table["total_return"].astype(float)) - 1.0),
        "mean_year_return": float(table["total_return"].mean()),
        "worst_year_return": float(table["total_return"].min()),
        "worst_year_drawdown": float(table["max_drawdown"].min()),
        "lookahead_policy": "train < valid < confirm < test; candidate features use pre-entry 1m only",
    }
    (output_root / "walk_forward_summary.json").write_text(json.dumps(json_safe(summary), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_markdown(summary, table, output_root / "WALK_FORWARD_SUMMARY.md")
    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run yearly walk-forward for BTC futures v3 beta edge")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--model-root", default=str(DEFAULT_MODEL_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--train-start", default="2021-01-01")
    parser.add_argument("--first-test-year", type=int, default=2024)
    parser.add_argument("--last-test-year", type=int, default=0)
    parser.add_argument("--model-kind", default="stable_rank")
    parser.add_argument("--min-final-auc", type=float, default=0.55)
    parser.add_argument("--min-precision", type=float, default=0.55)
    parser.add_argument("--min-profit-factor", type=float, default=1.05)
    args = parser.parse_args()
    summary = run_yearly_walk_forward(
        dataset_path=args.dataset,
        model_root=args.model_root,
        output_root=args.output_root,
        train_start=args.train_start,
        first_test_year=args.first_test_year,
        last_test_year=args.last_test_year or None,
        model_kind=args.model_kind,
        min_final_auc=args.min_final_auc,
        min_precision=args.min_precision,
        min_profit_factor=args.min_profit_factor,
    )
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
