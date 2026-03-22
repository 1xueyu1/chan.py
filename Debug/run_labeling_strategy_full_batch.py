from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


_CPU_COUNT = os.cpu_count() or 4
_DEFAULT_NUM_WORKERS = max(1, _CPU_COUNT - 1)
_DEFAULT_SYMBOL_WORKERS = max(1, min(8, _CPU_COUNT // 2))


STRATEGY_RUNS = [
    ("original_return", "baseline_original"),
    ("chan_structure_invalidation", "chan_structure"),
    ("multicycle_resonance", "multicycle_resonance"),
    ("divergence_focus", "divergence_focus"),
    ("ensemble4_consensus", "ensemble_consensus"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按4种标注策略 + 集成策略，批量执行 full 流程并归档结果"
    )
    parser.add_argument("--begin-time", default="2020-01-01")
    parser.add_argument("--end-time", default="2026-02-28")
    parser.add_argument(
        "--train-mode",
        choices=["cpu", "gpu", "auto"],
        default="auto",
    )
    parser.add_argument("--num-workers", type=int, default=_DEFAULT_NUM_WORKERS)
    parser.add_argument("--symbol-workers", type=int, default=_DEFAULT_SYMBOL_WORKERS)
    parser.add_argument("--signal-threshold", type=float, default=0.55)
    parser.add_argument("--labeling-name", default="ChanLabelingSuite_v2")
    parser.add_argument("--ensemble-min-votes", type=int, default=3)
    parser.add_argument("--label-max-holding-bars", type=int, default=96)
    parser.add_argument("--save-html-detail-report", action="store_true")
    parser.add_argument(
        "--archive",
        action="store_true",
        help="每次运行后自动归档到backup",
    )
    parser.add_argument("--run-prefix", default="labeling_batch")
    parser.add_argument(
        "--start-from-strategy",
        choices=[item[0] for item in STRATEGY_RUNS],
        default="original_return",
        help="从指定策略开始执行（含该策略）",
    )
    return parser.parse_args()


def run_cmd(cmd: list[str]) -> int:
    print("$ " + " ".join(cmd))
    proc = subprocess.Popen(cmd)
    return proc.wait()


def main() -> None:
    args = parse_args()
    project_root = Path(__file__).resolve().parents[1]
    python_exec = sys.executable
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    batch_records = []

    strategy_names = [item[0] for item in STRATEGY_RUNS]
    start_idx = strategy_names.index(args.start_from_strategy)
    runs = STRATEGY_RUNS[start_idx:]

    for strategy, tag in runs:
        run_id = f"{args.run_prefix}_{tag}_{timestamp}"
        print("\n" + "=" * 72)
        print(f"[Batch] Start strategy={strategy} run_id={run_id}")
        print("=" * 72)

        cmd = [
            python_exec,
            str(project_root / "Debug" / "run_pipeline.py"),
            "--mode",
            "full",
            "--run-id",
            run_id,
            "--begin-time",
            args.begin_time,
            "--end-time",
            args.end_time,
            "--train-mode",
            args.train_mode,
            "--num-workers",
            str(args.num_workers),
            "--symbol-workers",
            str(args.symbol_workers),
            "--signal-threshold",
            str(args.signal_threshold),
            "--labeling-strategy",
            strategy,
            "--labeling-name",
            args.labeling_name,
            "--ensemble-min-votes",
            str(args.ensemble_min_votes),
            "--label-max-holding-bars",
            str(args.label_max_holding_bars),
        ]
        if args.save_html_detail_report:
            cmd.append("--save-html-detail-report")

        t0 = time.time()
        rc = run_cmd(cmd)
        elapsed = round(time.time() - t0, 2)

        run_root = project_root / "Debug" / "runs" / run_id
        record = {
            "strategy": strategy,
            "tag": tag,
            "run_id": run_id,
            "exit_code": rc,
            "elapsed_seconds": elapsed,
            "run_root": str(run_root),
        }

        if args.archive and rc == 0:
            archive_name = f"{run_id}_archive"
            archive_cmd = [
                python_exec,
                str(project_root / "backup" / "archive_version.py"),
                "--version-name",
                archive_name,
                "--train-dir",
                str(run_root / "train"),
                "--predict-dir",
                str(run_root / "predict"),
                "--backtest-dir",
                str(run_root / "backtest"),
                "--notes",
                f"strategy={strategy}, labeling_name={args.labeling_name}",
            ]
            arc_rc = run_cmd(archive_cmd)
            record["archive_exit_code"] = arc_rc
            record["archive_version"] = archive_name

        batch_records.append(record)

        if rc != 0:
            print(f"[Batch] Stop on failure: strategy={strategy}, exit={rc}")
            break

    summary_path = (
        project_root
        / "Debug"
        / "runs"
        / f"{args.run_prefix}_summary_{timestamp}.json"
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "batch": batch_records,
        "config": {
            "begin_time": args.begin_time,
            "end_time": args.end_time,
            "train_mode": args.train_mode,
            "num_workers": args.num_workers,
            "symbol_workers": args.symbol_workers,
            "signal_threshold": args.signal_threshold,
            "labeling_name": args.labeling_name,
            "ensemble_min_votes": args.ensemble_min_votes,
            "label_max_holding_bars": args.label_max_holding_bars,
            "save_html_detail_report": args.save_html_detail_report,
            "archive": args.archive,
        },
    }
    summary_path.write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("[Batch] Completed")
    print(f"[Batch] Summary: {summary_path}")
    for rec in batch_records:
        print(
            f"  - {rec['strategy']}: exit={rec['exit_code']} "
            f"elapsed={rec['elapsed_seconds']}s run_id={rec['run_id']}"
        )


if __name__ == "__main__":
    main()
