from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from Backtest.config import build_arg_parser, config_from_args
from Backtest.engine import run_vectorbt_backtest


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    cfg = config_from_args(args)

    result = run_vectorbt_backtest(cfg)

    print("Backtest completed.")
    print("Aggregate metrics:")
    for k, v in result.aggregate_metrics.items():
        print(f"  {k}: {v}")

    if result.artifacts:
        print("Artifacts:")
        if result.artifacts.events_csv is not None:
            print(f"  events:  {result.artifacts.events_csv}")
        if result.artifacts.bars_csv is not None:
            print(f"  bars:    {result.artifacts.bars_csv}")
        if result.artifacts.metrics_json is not None:
            print(f"  metrics: {result.artifacts.metrics_json}")
        if result.artifacts.report_html is not None:
            print(f"  report:  {result.artifacts.report_html}")
        if result.artifacts.trades_csv is not None:
            print(f"  trades:  {result.artifacts.trades_csv}")
        if result.artifacts.equity_csv is not None:
            print(f"  equity:  {result.artifacts.equity_csv}")


if __name__ == "__main__":
    main()
