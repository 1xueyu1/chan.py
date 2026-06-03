from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ML.routes.btc_futures_v1.data import parse_utc
from ML.shared.json_utils import json_safe

from .backtest import run_btc_futures_v3_alpha_backtest
from .dataset import DEFAULT_DATASET_PATH, ROUTE_NAME, build_btc_futures_v3_alpha_dataset
from .train import train_btc_futures_v3_alpha_models


DEFAULT_MODEL_ROOT = Path("result/ml/btc_futures_v3_alpha_walk_forward")
DEFAULT_OUTPUT_ROOT = Path("result/btc_futures_v3_alpha_walk_forward")


def _date_text(year: int) -> str:
    return f"{year:04d}-01-01"


def _year_end_text(year: int) -> str:
    return f"{year:04d}-12-31"


def _finite_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _fold_row(year: int, train_metrics: dict[str, Any], backtest_metrics: dict[str, Any]) -> dict[str, Any]:
    models = train_metrics.get("models") or {}
    buy = models.get("buy") or {}
    sell = models.get("sell") or {}
    splits = train_metrics.get("splits") or {}
    return {
        "test_year": int(year),
        "train_rows": int((splits.get("train") or {}).get("rows", 0)),
        "valid_rows": int((splits.get("valid") or {}).get("rows", 0)),
        "test_rows": int((splits.get("test") or {}).get("rows", 0)),
        "buy_enabled": bool(buy.get("enabled", False)),
        "sell_enabled": bool(sell.get("enabled", False)),
        "buy_global_threshold": _finite_float(buy.get("global_threshold")),
        "sell_global_threshold": _finite_float(sell.get("global_threshold")),
        "buy_valid_auc": _finite_float((buy.get("valid") or {}).get("auc")),
        "sell_valid_auc": _finite_float((sell.get("valid") or {}).get("auc")),
        "buy_pool_model_count": int(buy.get("pool_model_count", 0)),
        "sell_pool_model_count": int(sell.get("pool_model_count", 0)),
        "trades": int(backtest_metrics.get("trades", 0)),
        "qualified_signals": int(backtest_metrics.get("qualified_signals", 0)),
        "win_rate": _finite_float(backtest_metrics.get("win_rate")),
        "total_return": _finite_float(backtest_metrics.get("total_return")),
        "profit_factor": _finite_float(backtest_metrics.get("profit_factor")),
        "max_drawdown": _finite_float(backtest_metrics.get("max_drawdown")),
        "final_equity": _finite_float(backtest_metrics.get("final_equity")),
    }


def _fmt_pct(value: float) -> str:
    return "N/A" if not np.isfinite(value) else f"{value:.2%}"


def _fmt_float(value: float) -> str:
    if np.isinf(value):
        return "inf"
    return "N/A" if not np.isfinite(value) else f"{value:.3f}"


def _write_markdown(summary: dict[str, Any], table: pd.DataFrame, path: Path) -> None:
    lines = [
        "# BTC 合约 v3 Alpha 年度 Walk-Forward",
        "",
        "## 验证口径",
        "",
        "- 每个测试年只使用该年之前的数据。",
        "- 测试年前 `validation_years` 年作为验证集，用于选特征、选结构池阈值和决定哪些池允许交易。",
        "- 测试年不参与训练、特征筛选、阈值选择。",
        "- 入场时间必须满足 `entry_time >= exec_time + 15min`。",
        "- 回测仍使用修复后的同口径事件回放，不使用未来函数。",
        "",
        "## 汇总",
        "",
        f"- Fold 数：{summary['folds']}",
        f"- 总交易数：{summary['total_trades']}",
        f"- 加权胜率：{_fmt_pct(summary['weighted_win_rate'])}",
        f"- 年度复合收益：{_fmt_pct(summary['compounded_return'])}",
        f"- 平均年度收益：{_fmt_pct(summary['mean_year_return'])}",
        f"- 最差年度收益：{_fmt_pct(summary['worst_year_return'])}",
        f"- 最差年度回撤：{_fmt_pct(summary['worst_year_drawdown'])}",
        "",
        "## 分年结果",
        "",
        "| 年份 | 交易数 | 胜率 | 收益 | 最大回撤 | PF | 买池模型 | 卖池模型 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"| {int(row['test_year'])} | {int(row['trades'])} | {_fmt_pct(row['win_rate'])} | "
            f"{_fmt_pct(row['total_return'])} | {_fmt_pct(row['max_drawdown'])} | {_fmt_float(row['profit_factor'])} | "
            f"{int(row['buy_pool_model_count'])} | {int(row['sell_pool_model_count'])} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def run_yearly_walk_forward(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    model_root: str | Path = DEFAULT_MODEL_ROOT,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    train_start: Any | None = "2021-01-01",
    first_test_year: int = 2023,
    last_test_year: int | None = None,
    validation_years: int = 2,
    model_kind: str = "auto",
    min_precision: float = 0.60,
    min_profit_factor: float = 1.20,
    min_side_auc: float = 0.54,
    min_pool_auc: float = 0.54,
    top_k_features: int = 180,
    top_k_pool_features: int = 90,
    initial_cash: float = 100000.0,
) -> dict[str, Any]:
    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        build_btc_futures_v3_alpha_dataset(output_path=dataset_file)
    dataset = pd.read_parquet(dataset_file, columns=["exec_time"])
    dataset["exec_time"] = pd.to_datetime(dataset["exec_time"], utc=True, errors="coerce")
    data_end = dataset["exec_time"].max()
    if pd.isna(data_end):
        raise ValueError("dataset has no valid exec_time")
    final_year = int(last_test_year or pd.Timestamp(data_end).year)
    train_start_ts = parse_utc(train_start)
    model_root_path = Path(model_root)
    output_root_path = Path(output_root)
    model_root_path.mkdir(parents=True, exist_ok=True)
    output_root_path.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for year in range(int(first_test_year), final_year + 1):
        valid_start = _date_text(year - int(validation_years))
        test_start = _date_text(year)
        test_end_exclusive = _date_text(year + 1)
        if train_start_ts is not None and parse_utc(valid_start) <= train_start_ts:
            continue
        if parse_utc(test_start) > data_end:
            continue
        fold_model_dir = model_root_path / f"test_{year}"
        fold_output_dir = output_root_path / f"test_{year}"
        print(
            f"[WF3] year={year} train=[{train_start},{valid_start}) "
            f"valid=[{valid_start},{test_start}) test=[{test_start},{test_end_exclusive})",
            flush=True,
        )
        train_metrics = train_btc_futures_v3_alpha_models(
            dataset_path=dataset_file,
            model_dir=fold_model_dir,
            train_start=train_start,
            valid_start=valid_start,
            test_start=test_start,
            test_end=test_end_exclusive,
            model_kind=model_kind,
            min_precision=min_precision,
            min_profit_factor=min_profit_factor,
            min_side_auc=min_side_auc,
            min_pool_auc=min_pool_auc,
            top_k_features=top_k_features,
            top_k_pool_features=top_k_pool_features,
        )
        backtest_end = _year_end_text(year)
        if year == int(pd.Timestamp(data_end).year):
            backtest_end = data_end.strftime("%Y-%m-%d")
        backtest_metrics = run_btc_futures_v3_alpha_backtest(
            dataset_path=dataset_file,
            model_dir=fold_model_dir,
            output_dir=fold_output_dir,
            begin_time=test_start,
            end_time=backtest_end,
            initial_cash=initial_cash,
        )
        row = _fold_row(year, train_metrics, backtest_metrics)
        rows.append(row)
        print(
            f"[WF3] done year={year} trades={row['trades']} "
            f"win_rate={_fmt_pct(row['win_rate'])} return={_fmt_pct(row['total_return'])}",
            flush=True,
        )

    table = pd.DataFrame(rows)
    if table.empty:
        raise ValueError("no walk-forward folds were produced")
    table.to_csv(output_root_path / "fold_summary.csv", index=False)
    total_trades = int(table["trades"].sum())
    win_table = table.loc[(table["trades"] > 0) & table["win_rate"].notna()].copy()
    weighted_win_rate = (
        float(np.average(win_table["win_rate"], weights=win_table["trades"]))
        if int(win_table["trades"].sum()) > 0
        else float("nan")
    )
    summary = {
        "route": ROUTE_NAME,
        "dataset_path": str(dataset_file),
        "model_root": str(model_root_path),
        "output_root": str(output_root_path),
        "train_start": str(train_start),
        "first_test_year": int(first_test_year),
        "last_test_year": int(final_year),
        "validation_years": int(validation_years),
        "folds": int(len(table)),
        "total_trades": total_trades,
        "weighted_win_rate": weighted_win_rate,
        "compounded_return": float(np.prod(1.0 + table["total_return"].astype(float)) - 1.0),
        "mean_year_return": float(table["total_return"].mean()),
        "worst_year_return": float(table["total_return"].min()),
        "best_year_return": float(table["total_return"].max()),
        "worst_year_drawdown": float(table["max_drawdown"].min()),
        "lookahead_policy": "train < valid < test; entry_time >= exec_time + 15min",
    }
    (output_root_path / "walk_forward_summary.json").write_text(
        json.dumps(json_safe(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_markdown(summary, table, output_root_path / "WALK_FORWARD_SUMMARY.md")
    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run yearly walk-forward for BTC futures v3 alpha")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--model-root", default=str(DEFAULT_MODEL_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--train-start", default="2021-01-01")
    parser.add_argument("--first-test-year", type=int, default=2023)
    parser.add_argument("--last-test-year", type=int, default=0)
    parser.add_argument("--validation-years", type=int, default=2)
    parser.add_argument("--model-kind", default="auto")
    parser.add_argument("--min-precision", type=float, default=0.60)
    parser.add_argument("--min-profit-factor", type=float, default=1.20)
    parser.add_argument("--min-side-auc", type=float, default=0.54)
    parser.add_argument("--min-pool-auc", type=float, default=0.54)
    parser.add_argument("--top-k-features", type=int, default=180)
    parser.add_argument("--top-k-pool-features", type=int, default=90)
    parser.add_argument("--initial-cash", type=float, default=100000.0)
    args = parser.parse_args()
    summary = run_yearly_walk_forward(
        dataset_path=args.dataset,
        model_root=args.model_root,
        output_root=args.output_root,
        train_start=args.train_start,
        first_test_year=args.first_test_year,
        last_test_year=args.last_test_year or None,
        validation_years=args.validation_years,
        model_kind=args.model_kind,
        min_precision=args.min_precision,
        min_profit_factor=args.min_profit_factor,
        min_side_auc=args.min_side_auc,
        min_pool_auc=args.min_pool_auc,
        top_k_features=args.top_k_features,
        top_k_pool_features=args.top_k_pool_features,
        initial_cash=args.initial_cash,
    )
    print(json.dumps(json_safe(summary), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
