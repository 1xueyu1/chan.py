from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .model import ModelBundle
from .validation import time_split


def threshold_report(frame: pd.DataFrame, probability: np.ndarray, thresholds: np.ndarray) -> pd.DataFrame:
    rows = []
    labels = frame["label"].astype(int).to_numpy()
    returns = frame["net_return"].astype(float).to_numpy()
    for threshold in thresholds:
        mask = probability >= float(threshold)
        selected = int(mask.sum())
        if selected == 0:
            rows.append(
                {
                    "threshold": float(threshold),
                    "selected": 0,
                    "coverage": 0.0,
                    "precision": np.nan,
                    "win_rate": np.nan,
                    "avg_net_return": np.nan,
                    "median_net_return": np.nan,
                    "sum_net_return": 0.0,
                }
            )
            continue
        selected_labels = labels[mask]
        selected_returns = returns[mask]
        rows.append(
            {
                "threshold": float(threshold),
                "selected": selected,
                "coverage": selected / len(frame),
                "precision": selected_labels.mean(),
                "win_rate": (selected_returns > 0).mean(),
                "avg_net_return": selected_returns.mean(),
                "median_net_return": np.median(selected_returns),
                "sum_net_return": selected_returns.sum(),
            }
        )
    return pd.DataFrame(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained Chan ML model by probability threshold")
    parser.add_argument("--dataset", default="data/btc_futures_v1/btc_futures_v1_dataset.parquet")
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", default="result/ml/btc_futures_v1/threshold_report.csv")
    parser.add_argument("--valid-start", default="2025-01-01")
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument("--side", default="auto", choices=["auto", "both", "buy", "sell"])
    parser.add_argument("--threshold-min", type=float, default=0.3)
    parser.add_argument("--threshold-max", type=float, default=0.8)
    parser.add_argument("--threshold-step", type=float, default=0.02)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    dataset = pd.read_parquet(args.dataset)
    _, valid = time_split(dataset, valid_start=args.valid_start, valid_ratio=args.valid_ratio)
    bundle = ModelBundle.load(args.model)

    side = bundle.side if args.side == "auto" else args.side
    if side == "buy":
        valid = valid[valid["is_buy"].astype(bool)].copy()
    elif side == "sell":
        valid = valid[~valid["is_buy"].astype(bool)].copy()

    probability = bundle.predict_proba(valid)
    thresholds = np.arange(args.threshold_min, args.threshold_max + 1e-12, args.threshold_step)
    report = threshold_report(valid, probability, thresholds)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(path, index=False)
    print(report.sort_values(["avg_net_return", "selected"], ascending=[False, False]).head(10).to_string(index=False))
    print(f"[ML] saved threshold report: {path}")


if __name__ == "__main__":
    main()
