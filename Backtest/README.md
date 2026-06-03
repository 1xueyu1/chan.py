# Backtest Framework

This folder contains the Chan + ML + risk-control backtesting framework.

## Design

- Direct flow: `BarFrame -> RawBSPEvent -> ScoredSignalEvent -> SignalMatrix -> risk layer -> execution engine -> BacktestRunResult -> report`
- Default execution uses the crypto simulator. The legacy vectorbt engine remains available through `BacktestConfig`.
- Daily CLI only exposes parameters that are expected to be tuned often.
- External code should prefer the facade entry (`facade.py`) and typed result objects (`types.py`) over internal orchestration helpers.

## Modules

- `types.py`: cross-layer contracts (`RawBSPEvent`, `ScoredSignalEvent`, `SignalMatrix`, `BacktestRunResult`)
- `data_contract.py`: standard OHLCV BarFrame normalization and validation
- `facade.py`: stable backtest entrypoint for scripts/UI integrations
- `protocols.py`: Protocol definitions for extractor/scorer/risk/execution/reporting boundaries
- `config.py`: backward-compatible runtime config and CLI parser
- `settings.py`: grouped config objects for new code
- `data_loader.py`: load parquet bars
- `chan_signal_extractor.py`: extract BSP events and Chan context features from `CChan`/RustCore
- `ml_filter.py`: score raw Chan events with buy/sell models
- `signal_builder.py`: map events to bar-level entries/exits
- `risk.py`: dynamic size, volatility targeting, drawdown stops, holding limits
- `crypto_engine.py`: exchange-style crypto simulator
- `vectorbt_engine.py`: legacy vectorbt signal engine
- `symbol_runner.py`: single-symbol pipeline
- `metrics.py`: aggregate multi-symbol metrics
- `engine.py`: backward-compatible multi-symbol orchestration
- `reporter.py`: write csv/json/html outputs

## Quick Start

```powershell
python Backtest\examples\run_vectorbt_backtest.py --symbols BTCUSDT --begin-time 2025-01-01 --end-time 2025-01-31 --kl-type 15m --no-bars-csv
```

Programmatic entrypoint:

```python
from Backtest.config import BacktestConfig
from Backtest.facade import run_backtest

result = run_backtest(BacktestConfig(symbols=["BTCUSDT"], begin_time="2025-01-01", end_time="2025-01-31"))
print(result.aggregate_metrics)
```

## Common ML + Risk Run

```powershell
python Backtest\examples\run_vectorbt_backtest.py --symbols BTCUSDT ETHUSDT --begin-time 2025-01-01 --end-time 2026-05-05 --allow-short --trade-exit-mode fixed_tp_sl --fixed-stop-loss-pct 0.02 --fixed-take-profit-pct 0.05 --ml-enabled --ml-buy-model-path result/ml/feature_select_v2/model_buy_top20.pkl --ml-sell-model-path result/ml/feature_select_v2/model_sell_top20.pkl --ml-buy-threshold 0.74 --ml-sell-threshold 0.74 --risk-enabled --risk-base-position-size 0.85 --risk-max-position-size 1.0 --risk-target-annual-vol 1.2 --no-bars-csv
```

## Daily CLI Parameters

- `--symbols`: symbols to backtest.
- `--begin-time` / `--end-time`: backtest time range.
- `--kl-type`: bar interval, for example `15m`.
- `--allow-short`: enable short entries.
- `--trade-exit-mode`: `opposite_signal` or `fixed_tp_sl`.
- `--fixed-stop-loss-pct`: fixed stop-loss percent for `fixed_tp_sl`.
- `--fixed-take-profit-pct`: fixed take-profit percent for `fixed_tp_sl`.
- `--ml-enabled`: enable ML filtering.
- `--ml-buy-model-path` / `--ml-sell-model-path`: buy/sell model paths.
- `--ml-buy-threshold` / `--ml-sell-threshold`: probability thresholds.
- `--risk-enabled`: enable risk layer.
- `--risk-base-position-size`: base notional size as a fraction of equity.
- `--risk-max-position-size`: maximum dynamic size.
- `--risk-target-annual-vol`: volatility target for dynamic sizing.
- `--symbol-workers`: number of symbols processed in parallel.
- `--output-dir`: output directory.
- `--no-bars-csv`: skip large per-bar output.

## Hidden Defaults

These are still available in `BacktestConfig` but intentionally hidden from daily CLI:

- `execution_engine="crypto"`
- `execution_mode="next_bar_open"`
- `initial_cash=100000.0`
- `fee=0.0004`
- `slippage=0.0001`
- `cooldown_bars=0`
- `risk_min_position_size=0.02`
- `risk_use_volatility_target=True`
- `risk_vol_window=96`
- `risk_max_atr_pct=0.12`
- `risk_atr_period=14`
- `risk_symbol_drawdown_stop_pct=0.25`
- `risk_cooldown_bars_after_drawdown=672`
- `risk_max_holding_bars=0`
- `crypto_leverage=1.0`
- `crypto_funding_rate_8h=0.0`
- `crypto_maintenance_margin_rate=0.005`
- `crypto_min_notional=5.0`
- `crypto_qty_step=0.0`
- `crypto_min_qty=0.0`
- `crypto_stop_first=True`
- `skip_symbol_errors=True`

## Outputs

By default in `result/`:

- `signal_events.csv`
- `signal_bars.csv`
- `backtest_metrics.json`
- `executed_trades.csv`
- `portfolio_equity_curve.csv`
- `backtest_report.html`

## Static Backtest Explorer

统一回测浏览页面只生成一份，不随每次回测重复写入各个结果目录。

刷新索引：

```powershell
powershell -ExecutionPolicy Bypass -File Scripts\refresh_backtest_explorer.ps1
```

打开页面：

```text
result\backtest_explorer\index.html
```

该页面会扫描 `result/**/backtest_metrics.json`，并展示每条回测记录的总览、币种排行，以及日/周/月/年分周期收益。
