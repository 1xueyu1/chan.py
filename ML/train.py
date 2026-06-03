from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .model import train_classifier
from .validation import classification_report_dict, time_split


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a fast Chan ML classifier")
    parser.add_argument("--dataset", default="data/btc_futures_v1/btc_futures_v1_dataset.parquet")
    parser.add_argument("--model-output", default="result/ml/btc_futures_v1/model.pkl")
    parser.add_argument("--metrics-output", default="result/ml/btc_futures_v1/generic_metrics.json")
    parser.add_argument("--model-kind", default="auto", choices=["auto", "lightgbm", "xgboost", "sklearn"])
    parser.add_argument("--side", default="both", choices=["both", "buy", "sell"])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--valid-start", default=None)
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    dataset = pd.read_parquet(args.dataset)
    train_frame, valid_frame = time_split(dataset, valid_start=args.valid_start, valid_ratio=args.valid_ratio)

    bundle = train_classifier(
        train_frame=train_frame,
        model_kind=args.model_kind,
        threshold=args.threshold,
        side=args.side,
    )
    bundle.save(args.model_output)

    metrics = {"train_rows": len(train_frame), "valid_rows": len(valid_frame), "feature_count": len(bundle.feature_columns)}
    if not valid_frame.empty:
        valid_use = valid_frame
        if args.side == "buy":
            valid_use = valid_use[valid_use["is_buy"].astype(bool)]
        elif args.side == "sell":
            valid_use = valid_use[~valid_use["is_buy"].astype(bool)]
        if not valid_use.empty:
            score = bundle.predict_proba(valid_use)
            metrics.update({f"valid_{k}": v for k, v in classification_report_dict(valid_use["label"].astype(int), score, args.threshold).items()})

    metrics_path = Path(args.metrics_output)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
