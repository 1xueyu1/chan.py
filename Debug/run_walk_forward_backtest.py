from __future__ import annotations

# flake8: noqa: E501

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import pandas as pd


@dataclass
class WindowResult:
    index: int
    train_begin: str
    train_end: str
    test_begin: str
    test_end: str
    train_run_id: str
    eval_run_id: str
    annualized_return_pct: float
    max_drawdown_pct: float
    sharpe: float
    total_trades: float
    total_return_pct: float
    win_rate_pct: float


def _safe_float(v: object, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return float(default)


def _run_cmd(cmd: List[str]) -> None:
    rc = subprocess.call(cmd)
    if rc != 0:
        raise RuntimeError(f"Command failed rc={rc}: {' '.join(cmd)}")


def _load_agg_metrics(project_root: Path, debug_root: str, run_id: str) -> Dict[str, float]:
    path = project_root / debug_root / "runs" / run_id / "backtest" / "backtest_metrics.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    agg = payload.get("aggregate", {}) or {}
    return {
        "annualized_return_pct": _safe_float(agg.get("annualized_return_pct")),
        "max_drawdown_pct": _safe_float(agg.get("max_drawdown_pct")),
        "sharpe": _safe_float(agg.get("sharpe")),
        "total_trades": _safe_float(agg.get("total_trades")),
        "total_return_pct": _safe_float(agg.get("total_return_pct")),
        "win_rate_pct": _safe_float(agg.get("win_rate_pct")),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Walk-forward rolling train and out-of-sample backtest")
    parser.add_argument("--debug-root", default="Debug")
    parser.add_argument("--begin-time", default="2020-01-01")
    parser.add_argument("--end-time", default="2026-02-28")
    parser.add_argument("--train-days", type=int, default=720)
    parser.add_argument("--test-days", type=int, default=120)
    parser.add_argument("--step-days", type=int, default=120)
    parser.add_argument("--max-windows", type=int, default=0, help="0 means all windows")

    parser.add_argument(
        "--train-symbols",
        nargs="+",
        default=["ADAUSDT", "AVAXUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT", "DOTUSDT", "ETHUSDT", "LTCUSDT"],
    )
    parser.add_argument(
        "--backtest-symbols",
        nargs="+",
        default=["ADAUSDT", "AVAXUSDT", "BNBUSDT", "BTCUSDT", "DOGEUSDT", "DOTUSDT", "ETHUSDT", "LTCUSDT", "SOLUSDT", "XRPUSDT"],
    )

    parser.add_argument("--dynamic-pt-enabled", action="store_true")
    parser.add_argument("--pt-low-vol-multiplier", type=float, default=1.8)
    parser.add_argument("--pt-mid-vol-multiplier", type=float, default=2.0)
    parser.add_argument("--pt-high-vol-multiplier", type=float, default=2.2)
    parser.add_argument("--meta-calibration", choices=["none", "platt", "isotonic"], default="none")
    parser.add_argument("--meta-calibration-ratio", type=float, default=0.2)
    parser.add_argument("--pass-positive-month-ratio", type=float, default=0.70)

    parser.add_argument("--signal-threshold", type=float, default=0.55)
    parser.add_argument("--signal-margin", type=float, default=0.0)
    parser.add_argument("--meta-threshold-by-bsp", default="")
    parser.add_argument("--meta-threshold-by-direction", default="")
    parser.add_argument("--cooldown-bars", type=int, default=0)
    parser.add_argument("--backtest-fee", type=float, default=0.0004)
    parser.add_argument("--backtest-slippage", type=float, default=0.0001)

    parser.add_argument("--symbol-workers", type=int, default=4)
    parser.add_argument("--backtest-parallel-mode", choices=["process", "thread"], default="process")
    parser.add_argument("--backtest-fast-mode", action="store_true")
    parser.add_argument("--save-html-detail-report", action="store_true", default=False)
    parser.add_argument("--enable-backup-run", action="store_true")
    parser.add_argument("--run-prefix", default="walk_forward")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[1]
    python_exec = sys.executable

    begin = pd.Timestamp(args.begin_time)
    end = pd.Timestamp(args.end_time)

    if args.train_days <= 0 or args.test_days <= 0 or args.step_days <= 0:
        raise ValueError("train-days/test-days/step-days must be positive")

    cursor = begin
    idx = 0
    rows: List[WindowResult] = []
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    while True:
        train_begin = cursor
        train_end = train_begin + pd.Timedelta(days=int(args.train_days))
        test_begin = train_end
        test_end = test_begin + pd.Timedelta(days=int(args.test_days))

        if test_end > end:
            break

        idx += 1
        if int(args.max_windows) > 0 and idx > int(args.max_windows):
            break

        train_run_id = f"{args.run_prefix}_train_{ts}_w{idx:02d}"
        eval_run_id = f"{args.run_prefix}_eval_{ts}_w{idx:02d}"

        train_cmd = [
            python_exec,
            str(project_root / "Debug" / "run_pipeline.py"),
            "--mode",
            "full",
            "--run-id",
            train_run_id,
            "--begin-time",
            train_begin.strftime("%Y-%m-%d"),
            "--end-time",
            train_end.strftime("%Y-%m-%d"),
            "--train-symbols",
            *args.train_symbols,
            "--backtest-symbols",
            *args.backtest_symbols,
            "--skip-predict",
            "--skip-backtest",
            "--meta-calibration",
            str(args.meta_calibration),
            "--meta-calibration-ratio",
            str(args.meta_calibration_ratio),
            "--pass-positive-month-ratio",
            str(args.pass_positive_month_ratio),
            "--pt-low-vol-multiplier",
            str(args.pt_low_vol_multiplier),
            "--pt-mid-vol-multiplier",
            str(args.pt_mid_vol_multiplier),
            "--pt-high-vol-multiplier",
            str(args.pt_high_vol_multiplier),
        ]
        if bool(args.dynamic_pt_enabled):
            train_cmd.append("--dynamic-pt-enabled")
        if not bool(args.enable_backup_run):
            train_cmd.append("--no-backup-run")

        print(f"[WF] window={idx} train {train_begin.date()} -> {train_end.date()} run={train_run_id}")
        _run_cmd(train_cmd)

        eval_cmd = [
            python_exec,
            str(project_root / "Debug" / "run_pipeline.py"),
            "--mode",
            "predict-backtest",
            "--source-run-id",
            train_run_id,
            "--run-id",
            eval_run_id,
            "--begin-time",
            test_begin.strftime("%Y-%m-%d"),
            "--end-time",
            test_end.strftime("%Y-%m-%d"),
            "--backtest-symbols",
            *args.backtest_symbols,
            "--skip-predict",
            "--signal-threshold",
            str(args.signal_threshold),
            "--signal-margin",
            str(args.signal_margin),
            "--cooldown-bars",
            str(args.cooldown_bars),
            "--backtest-fee",
            str(args.backtest_fee),
            "--backtest-slippage",
            str(args.backtest_slippage),
            "--symbol-workers",
            str(args.symbol_workers),
            "--backtest-parallel-mode",
            str(args.backtest_parallel_mode),
        ]
        if str(args.meta_threshold_by_bsp).strip():
            eval_cmd.extend(["--meta-threshold-by-bsp", str(args.meta_threshold_by_bsp)])
        if str(args.meta_threshold_by_direction).strip():
            eval_cmd.extend(["--meta-threshold-by-direction", str(args.meta_threshold_by_direction)])
        if bool(args.backtest_fast_mode):
            eval_cmd.append("--backtest-fast-mode")
            eval_cmd.append("--no-save-html-detail-report")
        elif bool(args.save_html_detail_report):
            eval_cmd.append("--save-html-detail-report")

        if not bool(args.enable_backup_run):
            eval_cmd.append("--no-backup-run")

        print(f"[WF] window={idx} test  {test_begin.date()} -> {test_end.date()} run={eval_run_id}")
        _run_cmd(eval_cmd)

        agg = _load_agg_metrics(project_root, str(args.debug_root), eval_run_id)
        rows.append(
            WindowResult(
                index=idx,
                train_begin=train_begin.strftime("%Y-%m-%d"),
                train_end=train_end.strftime("%Y-%m-%d"),
                test_begin=test_begin.strftime("%Y-%m-%d"),
                test_end=test_end.strftime("%Y-%m-%d"),
                train_run_id=train_run_id,
                eval_run_id=eval_run_id,
                annualized_return_pct=agg["annualized_return_pct"],
                max_drawdown_pct=agg["max_drawdown_pct"],
                sharpe=agg["sharpe"],
                total_trades=agg["total_trades"],
                total_return_pct=agg["total_return_pct"],
                win_rate_pct=agg["win_rate_pct"],
            )
        )

        cursor = cursor + pd.Timedelta(days=int(args.step_days))

    if not rows:
        raise RuntimeError("No walk-forward window generated. Please adjust begin/end/train-days/test-days.")

    df = pd.DataFrame([r.__dict__ for r in rows])
    positive_ratio = float((df["total_return_pct"] > 0).mean())

    summary = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "config": {
            "begin_time": args.begin_time,
            "end_time": args.end_time,
            "train_days": int(args.train_days),
            "test_days": int(args.test_days),
            "step_days": int(args.step_days),
            "max_windows": int(args.max_windows),
            "signal_threshold": float(args.signal_threshold),
            "signal_margin": float(args.signal_margin),
            "cooldown_bars": int(args.cooldown_bars),
            "meta_calibration": str(args.meta_calibration),
            "dynamic_pt_enabled": bool(args.dynamic_pt_enabled),
        },
        "window_count": int(len(df)),
        "aggregate": {
            "annualized_return_pct_mean": _safe_float(df["annualized_return_pct"].mean()),
            "max_drawdown_pct_mean": _safe_float(df["max_drawdown_pct"].mean()),
            "sharpe_mean": _safe_float(df["sharpe"].mean()),
            "total_return_pct_mean": _safe_float(df["total_return_pct"].mean()),
            "total_trades_mean": _safe_float(df["total_trades"].mean()),
            "positive_window_ratio": positive_ratio,
        },
        "windows": [r.__dict__ for r in rows],
    }

    out_dir = project_root / str(args.debug_root) / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"{args.run_prefix}_{ts}_summary.json"
    out_csv = out_dir / f"{args.run_prefix}_{ts}_summary.csv"

    out_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    df.to_csv(out_csv, index=False, encoding="utf-8")

    print("[WF] completed")
    print(f"[WF] windows={len(df)}")
    print(f"[WF] positive_window_ratio={positive_ratio:.4f}")
    print(f"[WF] summary_json={out_json}")
    print(f"[WF] summary_csv={out_csv}")


if __name__ == "__main__":
    main()
