from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List


def _default_run_id() -> str:
    return datetime.now().strftime("run_%Y%m%d_%H%M%S")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run full lifecycle pipeline "
            "(train -> predict -> backtest) with organized artifacts."
        ),
    )

    parser.add_argument(
        "--run-id",
        default="",
        help="run folder name; auto-generated when empty",
    )
    parser.add_argument(
        "--debug-root",
        default="Debug",
        help="debug root directory",
    )

    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTCUSDT", "ETHUSDT"],
        help="symbols for training",
    )
    parser.add_argument(
        "--begin-time",
        default="2024-01-01",
        help="train/predict/backtest begin time",
    )
    parser.add_argument(
        "--end-time",
        default="2026-01-01",
        help="train/predict/backtest end time",
    )
    parser.add_argument(
        "--train-mode",
        choices=["cpu", "gpu", "auto"],
        default="auto",
    )
    parser.add_argument("--num-workers", type=int, default=4)

    parser.add_argument(
        "--predict-symbol",
        default="BTCUSDT",
        help="symbol used by predict stage",
    )
    parser.add_argument("--signal-threshold", type=float, default=0.55)

    parser.add_argument(
        "--backtest-symbols",
        nargs="+",
        default=["BTCUSDT", "ETHUSDT"],
        help="symbols for backtest",
    )
    parser.add_argument(
        "--symbol-workers",
        type=int,
        default=1,
        help="parallel workers for symbol backtest",
    )
    parser.add_argument(
        "--save-html-detail-report",
        action="store_true",
        help="save detail html report for backtest",
    )

    parser.add_argument("--skip-predict", action="store_true")
    parser.add_argument("--skip-backtest", action="store_true")

    return parser


def _run_cmd(
    cmd: List[str],
    log_path: Path,
    extra_env: Dict[str, str] | None = None,
) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write("$ " + " ".join(cmd) + "\n\n")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**dict(), **extra_env} if extra_env else None,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_file.write(line)
        rc = process.wait()
        if rc != 0:
            raise RuntimeError(f"Command failed (exit={rc}): {' '.join(cmd)}")


def _write_manifest(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def main() -> None:
    args = _build_parser().parse_args()
    run_id = args.run_id or _default_run_id()

    root = Path(args.debug_root) / "runs" / run_id
    train_dir = root / "train"
    predict_dir = root / "predict"
    backtest_dir = root / "backtest"
    logs_dir = root / "logs"

    for d in [train_dir, predict_dir, backtest_dir, logs_dir]:
        d.mkdir(parents=True, exist_ok=True)

    python_exec = sys.executable
    project_root = Path(__file__).resolve().parents[1]

    model_buy = train_dir / "model_buy.json"
    model_sell = train_dir / "model_sell.json"
    meta_buy = train_dir / "meta_buy.json"
    meta_sell = train_dir / "meta_sell.json"

    train_cmd = [
        python_exec,
        str(project_root / "Debug" / "xgboost_shap_train.py"),
        "--symbols",
        *args.symbols,
        "--begin-time",
        args.begin_time,
        "--end-time",
        args.end_time,
        "--train-mode",
        args.train_mode,
        "--num-workers",
        str(args.num_workers),
        "--signal-threshold",
        str(args.signal_threshold),
        "--output-dir",
        str(train_dir),
    ]

    print(f"[Pipeline] run_id={run_id}")
    print("[Stage] train")
    train_env = {
        "XGB_TRAIN_LOG_DIR": str(logs_dir),
    }
    merged_train_env = dict(os.environ)
    merged_train_env.update(train_env)
    _run_cmd(train_cmd, logs_dir / "train.log", merged_train_env)

    predict_cmd: List[str] = []
    if not args.skip_predict:
        predict_cmd = [
            python_exec,
            str(project_root / "Debug" / "xgboost_shap_predict.py"),
            "--code",
            args.predict_symbol,
            "--begin-time",
            args.begin_time,
            "--end-time",
            args.end_time,
            "--model-buy-path",
            str(model_buy),
            "--model-sell-path",
            str(model_sell),
            "--meta-buy-path",
            str(meta_buy),
            "--meta-sell-path",
            str(meta_sell),
            "--signal-threshold",
            str(args.signal_threshold),
            "--output-dir",
            str(predict_dir),
            "--report-path",
            str(predict_dir / "shap_predict_report.html"),
        ]
        print("[Stage] predict")
        _run_cmd(predict_cmd, logs_dir / "predict.log")

    backtest_cmd: List[str] = []
    if not args.skip_backtest:
        backtest_cmd = [
            python_exec,
            str(
                project_root
                / "Backtest"
                / "examples"
                / "run_vectorbt_backtest.py"
            ),
            "--symbols",
            *args.backtest_symbols,
            "--begin-time",
            args.begin_time,
            "--end-time",
            args.end_time,
            "--model-buy-path",
            str(model_buy),
            "--model-sell-path",
            str(model_sell),
            "--meta-buy-path",
            str(meta_buy),
            "--meta-sell-path",
            str(meta_sell),
            "--signal-threshold",
            str(args.signal_threshold),
            "--output-dir",
            str(backtest_dir),
            "--symbol-workers",
            str(args.symbol_workers),
        ]
        if args.save_html_detail_report:
            backtest_cmd.append("--save-html-detail-report")

        print("[Stage] backtest")
        _run_cmd(backtest_cmd, logs_dir / "backtest.log")

    manifest = {
        "run_id": run_id,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "paths": {
            "root": str(root),
            "train": str(train_dir),
            "predict": str(predict_dir),
            "backtest": str(backtest_dir),
            "logs": str(logs_dir),
        },
        "commands": {
            "train": train_cmd,
            "predict": predict_cmd,
            "backtest": backtest_cmd,
        },
        "artifacts": {
            "model_buy": str(model_buy),
            "model_sell": str(model_sell),
            "meta_buy": str(meta_buy),
            "meta_sell": str(meta_sell),
            "predict_report": str(predict_dir / "shap_predict_report.html"),
            "backtest_metrics": str(backtest_dir / "backtest_metrics.json"),
            "backtest_report": str(backtest_dir / "xgb_backtest_report.html"),
            "backtest_report_detail": str(
                backtest_dir / "xgb_backtest_report_detail.html"
            ),
        },
    }
    _write_manifest(root / "run_manifest.json", manifest)

    print("\n[Done] Full pipeline finished.")
    print(f"[Artifacts Root] {root}")
    print(f"[Manifest] {root / 'run_manifest.json'}")


if __name__ == "__main__":
    main()
