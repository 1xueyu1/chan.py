from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .backtest import run_btc_futures_backtest
from .data import DEFAULT_DATASET_PATH, parse_utc
from .decision_models import train_bsp2_decision_models
from .train import train_btc_futures_models


DEFAULT_WALK_FORWARD_MODEL_ROOT = Path("result/ml/btc_futures_v2_bsp2_family_wf")
DEFAULT_WALK_FORWARD_OUTPUT_ROOT = Path("result/btc_futures_v2_bsp2_family_wf")


def _date_text(year: int, month: int = 1, day: int = 1) -> str:
    return f"{year:04d}-{month:02d}-{day:02d}"


def _year_end_text(year: int) -> str:
    return f"{year:04d}-12-31"


def _finite_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if np.isfinite(out) else float("nan")


def _side_model_metric(models: dict[str, Any], side: str, metric: str) -> float:
    side_model = models.get(side) or {}
    if metric in side_model:
        return _finite_float(side_model.get(metric))
    values: list[float] = []
    if isinstance(side_model, dict):
        for child in side_model.values():
            if isinstance(child, dict) and metric in child:
                value = _finite_float(child.get(metric))
                if np.isfinite(value):
                    values.append(value)
    return float(np.mean(values)) if values else float("nan")


def _side_model_auc(models: dict[str, Any], side: str, split: str) -> float:
    side_model = models.get(side) or {}
    if isinstance(side_model.get(split), dict):
        return _finite_float(side_model.get(split, {}).get("auc"))
    values: list[float] = []
    if isinstance(side_model, dict):
        for child in side_model.values():
            if isinstance(child, dict):
                value = _finite_float((child.get(split) or {}).get("auc"))
                if np.isfinite(value):
                    values.append(value)
    return float(np.mean(values)) if values else float("nan")


def _fold_row(year: int, train_metrics: dict[str, Any], backtest_metrics: dict[str, Any]) -> dict[str, Any]:
    models = train_metrics.get("models") or {}
    splits = train_metrics.get("splits") or {}
    return {
        "test_year": int(year),
        "train_start": (splits.get("train") or {}).get("start"),
        "train_end": (splits.get("train") or {}).get("end"),
        "train_rows": int((splits.get("train") or {}).get("rows", 0)),
        "valid_start": (splits.get("valid") or {}).get("start"),
        "valid_end": (splits.get("valid") or {}).get("end"),
        "valid_rows": int((splits.get("valid") or {}).get("rows", 0)),
        "test_start": (splits.get("test") or {}).get("start"),
        "test_end": (splits.get("test") or {}).get("end"),
        "test_rows": int((splits.get("test") or {}).get("rows", 0)),
        "feature_count": int(train_metrics.get("feature_count", 0)),
        "buy_threshold": _side_model_metric(models, "buy", "threshold"),
        "sell_threshold": _side_model_metric(models, "sell", "threshold"),
        "buy_valid_auc": _side_model_auc(models, "buy", "valid"),
        "sell_valid_auc": _side_model_auc(models, "sell", "valid"),
        "buy_test_auc": _side_model_auc(models, "buy", "test"),
        "sell_test_auc": _side_model_auc(models, "sell", "test"),
        "trades": int(backtest_metrics.get("trades", 0)),
        "qualified_signals": int(backtest_metrics.get("qualified_signals", 0)),
        "days_per_trade": _finite_float(backtest_metrics.get("days_per_trade")),
        "trades_per_day": _finite_float(backtest_metrics.get("trades_per_day")),
        "win_rate": _finite_float(backtest_metrics.get("win_rate")),
        "total_return": _finite_float(backtest_metrics.get("total_return")),
        "profit_factor": _finite_float(backtest_metrics.get("profit_factor")),
        "max_drawdown": _finite_float(backtest_metrics.get("max_drawdown")),
        "final_equity": _finite_float(backtest_metrics.get("final_equity")),
    }


def _split_metrics_from_dataset(
    dataset: pd.DataFrame,
    *,
    train_start: Any | None,
    valid_start: str,
    test_start: str,
    test_end: str,
) -> dict[str, Any]:
    frame = dataset.copy()
    frame["exec_time"] = pd.to_datetime(frame["exec_time"], utc=True, errors="coerce")
    train_start_ts = parse_utc(train_start)
    valid_start_ts = parse_utc(valid_start)
    test_start_ts = parse_utc(test_start)
    test_end_ts = parse_utc(test_end)
    train_mask = frame["exec_time"] < valid_start_ts
    if train_start_ts is not None:
        train_mask &= frame["exec_time"] >= train_start_ts
    valid_mask = (frame["exec_time"] >= valid_start_ts) & (frame["exec_time"] < test_start_ts)
    test_mask = frame["exec_time"] >= test_start_ts
    if test_end_ts is not None:
        test_mask &= frame["exec_time"] < test_end_ts

    def part(mask: pd.Series) -> dict[str, Any]:
        chunk = frame.loc[mask]
        if chunk.empty:
            return {"start": None, "end": None, "rows": 0}
        return {
            "start": str(chunk["exec_time"].min()),
            "end": str(chunk["exec_time"].max()),
            "rows": int(len(chunk)),
        }

    return {
        "route": "btc_futures_v2_bsp2_family",
        "models": {},
        "feature_count": 0,
        "splits": {
            "train": part(train_mask),
            "valid": part(valid_mask),
            "test": part(test_mask),
        },
        "training_skipped": True,
        "skip_reason": "use_ml_filter=false",
    }


def _write_markdown(summary: dict[str, Any], fold_table: pd.DataFrame, path: Path) -> None:
    lines = [
        "# BTC 合约年度 Walk-Forward 验证",
        "",
        "## 口径",
        "",
        "- 每个测试年只使用该年份之前的数据训练。",
        "- 测试年前一年作为验证集，用于选择买入/卖出概率阈值。",
        "- 测试年只做离线回测，不参与模型训练和阈值选择。",
        "- 信号来自 15m 缠论二类买卖点，入场从信号可用时间开始，不额外推迟 15m。",
        f"- 出场模式：`{summary.get('exit_mode')}`；结构目标层级：`{summary.get('structure_target_level')}`。",
        "- 当前组合：二买/二卖模型入场，第三段兑现后移动止损，之后等待同级别任意反向买卖点出场。",
        "",
        "## 汇总",
        "",
        f"- Fold 数：{summary['folds']}",
        f"- 总交易数：{summary['total_trades']}",
        f"- 加权胜率：{summary['weighted_win_rate']:.2%}" if np.isfinite(summary["weighted_win_rate"]) else "- 加权胜率：N/A",
        f"- 年度复合收益：{summary['compounded_return']:.2%}",
        f"- 平均年度收益：{summary['mean_year_return']:.2%}",
        f"- 最差年度收益：{summary['worst_year_return']:.2%}",
        f"- 最差年度回撤：{summary['worst_year_drawdown']:.2%}",
        "",
        "## 分年结果",
        "",
        "| 年份 | 交易数 | 天/笔 | 胜率 | 收益 | 最大回撤 | PF | 买阈值 | 卖阈值 |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in fold_table.iterrows():
        pf = row["profit_factor"]
        pf_text = "inf" if np.isinf(pf) else f"{pf:.3f}" if np.isfinite(pf) else "N/A"
        days_per_trade = row["days_per_trade"]
        days_text = f"{days_per_trade:.2f}" if np.isfinite(days_per_trade) else "N/A"
        lines.append(
            "| "
            f"{int(row['test_year'])} | "
            f"{int(row['trades'])} | "
            f"{days_text} | "
            f"{row['win_rate']:.2%} | "
            f"{row['total_return']:.2%} | "
            f"{row['max_drawdown']:.2%} | "
            f"{pf_text} | "
            f"{row['buy_threshold']:.2f} | "
            f"{row['sell_threshold']:.2f} |"
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def run_yearly_walk_forward(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    model_root: str | Path = DEFAULT_WALK_FORWARD_MODEL_ROOT,
    output_root: str | Path = DEFAULT_WALK_FORWARD_OUTPUT_ROOT,
    train_start: Any | None = "2021-01-01",
    first_test_year: int = 2023,
    last_test_year: int | None = None,
    model_kind: str = "auto",
    min_feature_coverage: float = 0.05,
    min_threshold_trades: int = 40,
    min_daily_trades: float = 0.35,
    min_precision: float = 0.52,
    min_trade_win_rate: float = 0.50,
    min_avg_net_return: float = 0.0,
    hard_negative_weight: float = 3.0,
    severe_negative_weight: float = 5.0,
    efficient_positive_weight: float = 1.2,
    primary_label_column: str = "label_bsp2_family_valid",
    initial_cash: float = 100000.0,
    stake_fraction: float = 1.0,
    exit_mode: str = "family_realtime",
    structure_target_level: str = "third",
    pure_bsp2_only: bool = False,
    enable_structure_risk_sizing: bool = True,
    target_trade_risk_pct: float = 0.01,
    min_stake_multiplier: float = 0.08,
    tier_a_max_stake: float = 1.0,
    tier_b_max_stake: float = 0.55,
    tier_c_max_stake: float = 0.25,
    tier_a_risk_boost: float = 3.0,
    tier_b_risk_boost: float = 1.0,
    tier_c_risk_boost: float = 0.6,
    no_timeout_max_holding_minutes: int = 43200,
    disable_before_third_timeout: bool = False,
    position_scope: str = "symbol",
    use_ml_filter: bool = False,
    train_models: bool | None = None,
    train_decision_models: bool = True,
) -> dict[str, Any]:
    dataset = pd.read_parquet(dataset_path)
    dataset["exec_time"] = pd.to_datetime(dataset["exec_time"], utc=True, errors="coerce")
    data_end = dataset["exec_time"].max()
    if pd.isna(data_end):
        raise ValueError("dataset has no valid exec_time")

    inferred_last_year = int(pd.Timestamp(data_end).year)
    final_year = int(last_test_year or inferred_last_year)
    if final_year < int(first_test_year):
        raise ValueError("last_test_year must be >= first_test_year")

    train_start_ts = parse_utc(train_start)
    model_root_path = Path(model_root)
    output_root_path = Path(output_root)
    model_root_path.mkdir(parents=True, exist_ok=True)
    output_root_path.mkdir(parents=True, exist_ok=True)

    fold_rows: list[dict[str, Any]] = []
    for year in range(int(first_test_year), final_year + 1):
        valid_start = _date_text(year - 1)
        test_start = _date_text(year)
        test_end_exclusive = _date_text(year + 1)
        if train_start_ts is not None and parse_utc(valid_start) <= train_start_ts:
            continue
        if parse_utc(test_start) > data_end:
            continue

        fold_model_dir = model_root_path / f"test_{year}"
        fold_output_dir = output_root_path / f"test_{year}"
        print(
            f"[WF] year={year} train=[{train_start},{valid_start}) "
            f"valid=[{valid_start},{test_start}) test=[{test_start},{test_end_exclusive})",
            flush=True,
        )
        should_train = bool(use_ml_filter if train_models is None else train_models)
        if should_train:
            train_metrics = train_btc_futures_models(
                dataset_path=dataset_path,
                model_dir=fold_model_dir,
                train_start=train_start,
                valid_start=valid_start,
                test_start=test_start,
                test_end=test_end_exclusive,
                model_kind=model_kind,
                min_feature_coverage=min_feature_coverage,
                min_threshold_trades=min_threshold_trades,
                min_daily_trades=min_daily_trades,
                min_precision=min_precision,
                min_trade_win_rate=min_trade_win_rate,
                min_avg_net_return=min_avg_net_return,
                hard_negative_weight=hard_negative_weight,
                severe_negative_weight=severe_negative_weight,
                efficient_positive_weight=efficient_positive_weight,
                primary_label_column=primary_label_column,
            )
        else:
            train_metrics = _split_metrics_from_dataset(
                dataset,
                train_start=train_start,
                valid_start=valid_start,
                test_start=test_start,
                test_end=test_end_exclusive,
            )
            if train_decision_models:
                fold_model_dir.mkdir(parents=True, exist_ok=True)
                train_metrics["decision_models"] = train_bsp2_decision_models(
                    dataset,
                    model_dir=fold_model_dir,
                    train_start=train_start,
                    valid_start=valid_start,
                    model_kind=model_kind,
                    min_feature_coverage=min_feature_coverage,
                )
        backtest_end = _year_end_text(year)
        if year == inferred_last_year:
            backtest_end = data_end.strftime("%Y-%m-%d")
        backtest_metrics = run_btc_futures_backtest(
            dataset_path=dataset_path,
            model_dir=fold_model_dir,
            output_dir=fold_output_dir,
            begin_time=test_start,
            end_time=backtest_end,
            initial_cash=initial_cash,
            stake_fraction=stake_fraction,
            exit_mode=exit_mode,
            structure_target_level=structure_target_level,
            pure_bsp2_only=pure_bsp2_only,
            enable_structure_risk_sizing=enable_structure_risk_sizing,
            target_trade_risk_pct=target_trade_risk_pct,
            min_stake_multiplier=min_stake_multiplier,
            tier_a_max_stake=tier_a_max_stake,
            tier_b_max_stake=tier_b_max_stake,
            tier_c_max_stake=tier_c_max_stake,
            tier_a_risk_boost=tier_a_risk_boost,
            tier_b_risk_boost=tier_b_risk_boost,
            tier_c_risk_boost=tier_c_risk_boost,
            no_timeout_max_holding_minutes=no_timeout_max_holding_minutes,
            disable_before_third_timeout=disable_before_third_timeout,
            position_scope=position_scope,
            use_ml_filter=bool(use_ml_filter),
            use_decision_models=bool(train_decision_models),
        )
        fold_row = _fold_row(year, train_metrics, backtest_metrics)
        fold_rows.append(fold_row)
        print(
            f"[WF] done year={year} trades={fold_row['trades']} "
            f"win_rate={fold_row['win_rate']:.2%} return={fold_row['total_return']:.2%}",
            flush=True,
        )

    fold_table = pd.DataFrame(fold_rows)
    if fold_table.empty:
        raise ValueError("no walk-forward folds were produced")

    fold_table.to_csv(output_root_path / "fold_summary.csv", index=False)
    total_trades = int(fold_table["trades"].sum())
    weighted_win_rate = float(np.average(fold_table["win_rate"], weights=fold_table["trades"])) if total_trades > 0 else float("nan")
    compounded_return = float(np.prod(1.0 + fold_table["total_return"].astype(float)) - 1.0)
    summary = {
        "route": "btc_futures_v2_bsp2_family",
        "dataset_path": str(dataset_path),
        "model_root": str(model_root_path),
        "output_root": str(output_root_path),
        "train_start": str(train_start),
        "first_test_year": int(first_test_year),
        "last_test_year": int(final_year),
        "folds": int(len(fold_table)),
        "total_trades": total_trades,
        "weighted_win_rate": weighted_win_rate,
        "compounded_return": compounded_return,
        "mean_year_return": float(fold_table["total_return"].mean()),
        "worst_year_return": float(fold_table["total_return"].min()),
        "best_year_return": float(fold_table["total_return"].max()),
        "worst_year_drawdown": float(fold_table["max_drawdown"].min()),
        "exit_mode": str(exit_mode),
        "structure_target_level": str(structure_target_level),
        "pure_bsp2_only": bool(pure_bsp2_only),
        "position_scope": str(position_scope),
        "use_ml_filter": bool(use_ml_filter),
        "train_models": bool(use_ml_filter if train_models is None else train_models),
        "train_decision_models": bool(train_decision_models),
        "threshold_constraints": {
            "min_threshold_trades": int(min_threshold_trades),
            "min_daily_trades": float(min_daily_trades),
            "min_precision": float(min_precision),
            "min_trade_win_rate": float(min_trade_win_rate),
            "min_avg_net_return": float(min_avg_net_return),
        },
        "sample_weighting": {
            "hard_negative_weight": float(hard_negative_weight),
            "severe_negative_weight": float(severe_negative_weight),
            "efficient_positive_weight": float(efficient_positive_weight),
        },
        "primary_label": str(primary_label_column),
        "structure_risk_sizing": {
            "enabled": bool(enable_structure_risk_sizing),
            "target_trade_risk_pct": float(target_trade_risk_pct),
            "min_stake_multiplier": float(min_stake_multiplier),
            "tier_a_max_stake": float(tier_a_max_stake),
            "tier_b_max_stake": float(tier_b_max_stake),
            "tier_c_max_stake": float(tier_c_max_stake),
            "tier_a_risk_boost": float(tier_a_risk_boost),
            "tier_b_risk_boost": float(tier_b_risk_boost),
            "tier_c_risk_boost": float(tier_c_risk_boost),
        },
        "timeout_policy": {
            "exit_mode": str(exit_mode),
            "position_scope": str(position_scope),
            "no_timeout_max_holding_minutes": int(no_timeout_max_holding_minutes),
            "disable_before_third_timeout": bool(disable_before_third_timeout),
        },
        "lookahead_policy": "train < valid < test; exec_time is signal availability time; entry_time >= exec_time",
    }
    (output_root_path / "walk_forward_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_markdown(summary, fold_table, output_root_path / "WALK_FORWARD_SUMMARY.md")
    return summary


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run yearly walk-forward validation for BTC futures v2 BSP2 family")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--model-root", default=str(DEFAULT_WALK_FORWARD_MODEL_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_WALK_FORWARD_OUTPUT_ROOT))
    parser.add_argument("--train-start", default="2021-01-01")
    parser.add_argument("--first-test-year", type=int, default=2023)
    parser.add_argument("--last-test-year", type=int, default=0)
    parser.add_argument("--model-kind", default="auto")
    parser.add_argument("--min-feature-coverage", type=float, default=0.05)
    parser.add_argument("--min-threshold-trades", type=int, default=40)
    parser.add_argument("--min-daily-trades", type=float, default=0.35)
    parser.add_argument("--min-precision", type=float, default=0.52)
    parser.add_argument("--min-trade-win-rate", type=float, default=0.50)
    parser.add_argument("--min-avg-net-return", type=float, default=0.0)
    parser.add_argument("--hard-negative-weight", type=float, default=3.0)
    parser.add_argument("--severe-negative-weight", type=float, default=5.0)
    parser.add_argument("--efficient-positive-weight", type=float, default=1.2)
    parser.add_argument("--primary-label-column", default="label_bsp2_family_valid")
    parser.add_argument("--initial-cash", type=float, default=100000.0)
    parser.add_argument("--stake-fraction", type=float, default=1.0)
    parser.add_argument(
        "--exit-mode",
        default="family_realtime",
        choices=["post_management", "fixed_label", "structure_target", "opposite_bsp", "third_then_opposite_bsp", "third_then_opposite_no_after_timeout", "family_adaptive", "family_realtime"],
    )
    parser.add_argument("--structure-target-level", default="third", choices=["third", "t1", "t2", "t3"])
    parser.add_argument("--include-bsp2s-family", action="store_true")
    parser.add_argument("--enable-structure-risk-sizing", action="store_true")
    parser.add_argument("--disable-structure-risk-sizing", action="store_true")
    parser.add_argument("--target-trade-risk-pct", type=float, default=0.01)
    parser.add_argument("--min-stake-multiplier", type=float, default=0.08)
    parser.add_argument("--tier-a-max-stake", type=float, default=1.0)
    parser.add_argument("--tier-b-max-stake", type=float, default=0.55)
    parser.add_argument("--tier-c-max-stake", type=float, default=0.25)
    parser.add_argument("--tier-a-risk-boost", type=float, default=3.0)
    parser.add_argument("--tier-b-risk-boost", type=float, default=1.0)
    parser.add_argument("--tier-c-risk-boost", type=float, default=0.6)
    parser.add_argument("--no-timeout-max-holding-minutes", type=int, default=43200)
    parser.add_argument("--disable-before-third-timeout", action="store_true")
    parser.add_argument("--position-scope", default="symbol", choices=["symbol", "global"])
    parser.add_argument("--enable-ml-filter", action="store_true")
    parser.add_argument("--train-models", action="store_true")
    parser.add_argument("--disable-decision-models", action="store_true")
    args = parser.parse_args()

    summary = run_yearly_walk_forward(
        dataset_path=args.dataset,
        model_root=args.model_root,
        output_root=args.output_root,
        train_start=args.train_start,
        first_test_year=args.first_test_year,
        last_test_year=args.last_test_year or None,
        model_kind=args.model_kind,
        min_feature_coverage=args.min_feature_coverage,
        min_threshold_trades=args.min_threshold_trades,
        min_daily_trades=args.min_daily_trades,
        min_precision=args.min_precision,
        min_trade_win_rate=args.min_trade_win_rate,
        min_avg_net_return=args.min_avg_net_return,
        hard_negative_weight=args.hard_negative_weight,
        severe_negative_weight=args.severe_negative_weight,
        efficient_positive_weight=args.efficient_positive_weight,
        primary_label_column=args.primary_label_column,
        initial_cash=args.initial_cash,
        stake_fraction=args.stake_fraction,
        exit_mode=args.exit_mode,
        structure_target_level=args.structure_target_level,
        pure_bsp2_only=False,
        enable_structure_risk_sizing=bool((args.enable_structure_risk_sizing or not args.disable_structure_risk_sizing) and not args.disable_structure_risk_sizing),
        target_trade_risk_pct=args.target_trade_risk_pct,
        min_stake_multiplier=args.min_stake_multiplier,
        tier_a_max_stake=args.tier_a_max_stake,
        tier_b_max_stake=args.tier_b_max_stake,
        tier_c_max_stake=args.tier_c_max_stake,
        tier_a_risk_boost=args.tier_a_risk_boost,
        tier_b_risk_boost=args.tier_b_risk_boost,
        tier_c_risk_boost=args.tier_c_risk_boost,
        no_timeout_max_holding_minutes=args.no_timeout_max_holding_minutes,
        disable_before_third_timeout=bool(args.disable_before_third_timeout),
        position_scope=args.position_scope,
        use_ml_filter=bool(args.enable_ml_filter),
        train_models=bool(args.train_models) if args.train_models else None,
        train_decision_models=not args.disable_decision_models,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

