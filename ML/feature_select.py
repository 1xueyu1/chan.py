from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from .features import infer_feature_columns
from .model import train_classifier
from .validation import classification_report_dict, time_split


def _side_filter(frame: pd.DataFrame, side: str) -> pd.DataFrame:
    if side == "buy":
        return frame[frame["is_buy"].astype(bool)].copy()
    if side == "sell":
        return frame[~frame["is_buy"].astype(bool)].copy()
    return frame.copy()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Select lightweight Chan ML features")
    parser.add_argument("--dataset", default="data/btc_futures_v1/btc_futures_v1_dataset.parquet")
    parser.add_argument("--side", choices=["buy", "sell", "both"], required=True)
    parser.add_argument("--valid-start", default="2025-01-01")
    parser.add_argument("--model-kind", default="sklearn", choices=["auto", "lightgbm", "xgboost", "sklearn"])
    parser.add_argument("--sample-size", type=int, default=20000)
    parser.add_argument("--top-k", nargs="+", type=int, default=[20, 30, 40, 60])
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--output-dir", default="result/ml/btc_futures_v1/feature_select")
    parser.add_argument("--random-state", type=int, default=42)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dataset = pd.read_parquet(args.dataset)
    train_frame, valid_frame = time_split(dataset, valid_start=args.valid_start)
    train_frame = _side_filter(train_frame, args.side)
    valid_frame = _side_filter(valid_frame, args.side)
    feature_columns = infer_feature_columns(train_frame)

    full_bundle = train_classifier(
        train_frame=train_frame,
        feature_columns=feature_columns,
        model_kind=args.model_kind,
        threshold=args.threshold,
        side="both",
        random_state=args.random_state,
    )

    valid_sample = valid_frame
    if len(valid_sample) > args.sample_size:
        valid_sample = valid_sample.sample(args.sample_size, random_state=args.random_state)
    x_valid = full_bundle.transform(valid_sample)
    y_valid = valid_sample["label"].astype(int)

    importance = permutation_importance(
        full_bundle.estimator,
        x_valid,
        y_valid,
        scoring="average_precision",
        n_repeats=2,
        random_state=args.random_state,
        n_jobs=-1,
    )
    ranking = (
        pd.DataFrame(
            {
                "feature": full_bundle.feature_columns,
                "importance_mean": importance.importances_mean,
                "importance_std": importance.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    ranking_path = out_dir / f"feature_ranking_{args.side}.csv"
    ranking.to_csv(ranking_path, index=False)

    rows = []
    for top_k in args.top_k:
        selected = ranking.head(int(top_k))["feature"].tolist()
        bundle = train_classifier(
            train_frame=train_frame,
            feature_columns=selected,
            model_kind=args.model_kind,
            threshold=args.threshold,
            side="both",
            random_state=args.random_state,
        )
        score = bundle.predict_proba(valid_frame)
        metrics = classification_report_dict(valid_frame["label"].astype(int), score, args.threshold)
        model_path = out_dir / f"model_{args.side}_top{top_k}.pkl"
        bundle.save(model_path)
        row = {
            "side": args.side,
            "top_k": int(top_k),
            "model_path": str(model_path),
            "feature_count": len(bundle.feature_columns),
            **metrics,
        }
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values("average_precision", ascending=False)
    summary_path = out_dir / f"summary_{args.side}.csv"
    summary.to_csv(summary_path, index=False)

    payload = {
        "ranking": str(ranking_path),
        "summary": str(summary_path),
        "best": summary.iloc[0].to_dict() if not summary.empty else {},
    }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
