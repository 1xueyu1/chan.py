# Backtest Framework (vectorbt)

This folder contains a vectorbt-based backtesting framework for the chan project.

## Key Modules
- config.py: runtime config and CLI parser
- data_loader.py: load parquet bars and resample from 5m fallback
- chan_signal_extractor.py: step-by-step BSP event extraction via CChan
- model_gate.py: dual-model scoring (buy/sell models)
- signal_builder.py: event-to-bar signal mapping
- vectorbt_engine.py: vectorized execution and metrics
- reporter.py: CSV/JSON/HTML output
- engine.py: orchestration and compatibility API
- strategy.py: legacy strategy compatibility base

## Quick Start
Run a single-symbol backtest:

```bash
python Backtest/examples/run_vectorbt_backtest.py \
  --symbols BTCUSDT \
  --begin-time 2025-01-01 \
  --end-time 2025-01-31 \
  --kl-type 15m \
  --signal-threshold 0.55
```

Outputs are written to `result/` by default:
- model_signal_events.csv
- model_signal_bars.csv
- backtest_metrics.json
- xgb_backtest_report.html

## Event Replay Mode (faster iteration)
When tuning execution logic / report rendering, you can skip chan extraction and model scoring,
and directly replay previously saved scored events:

```bash
python Backtest/examples/run_vectorbt_backtest.py \
  --symbols BTCUSDT \
  --begin-time 2025-01-01 \
  --end-time 2025-01-31 \
  --kl-type 15m \
  --event-replay \
  --event-replay-csv result/model_signal_events.csv
```

Optional: re-apply current threshold to CSV probability for quick sensitivity analysis:

```bash
python Backtest/examples/run_vectorbt_backtest.py \
  --symbols BTCUSDT \
  --event-replay \
  --event-replay-csv result/model_signal_events.csv \
  --signal-threshold 0.60 \
  --replay-reapply-threshold
```

## Legacy Compatibility
`Backtest/engine.py` exposes `run_chan_backtest_no_vnpy(...)` and
`Backtest/strategy.py` exposes `ChanStrategyBase` to keep existing imports stable.
