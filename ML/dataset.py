from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
import time
from typing import Iterable, List, Optional

import pandas as pd

from ChanConfig import CChanConfig
from Backtest.chan_signal_extractor import extract_raw_bsp_events
from Backtest.config import BacktestConfig
from Backtest.data_loader import load_symbol_bars
from RustCore import rust_config_path_from_chan_config
from .features import build_event_features
from .label import LabelConfig, label_events


def _rust_config_path(config: BacktestConfig) -> str | None:
    return rust_config_path_from_chan_config(CChanConfig(dict(config.chan_config)))


def build_symbol_dataset(config: BacktestConfig, symbol: str, label_config: LabelConfig | None = None) -> pd.DataFrame:
    bars = load_symbol_bars(config, symbol)
    events = extract_raw_bsp_events(config, symbol)
    if not events:
        return pd.DataFrame()

    features = build_event_features(bars, events, rust_config_path=_rust_config_path(config))
    labels = label_events(bars, events, label_config)
    out = pd.concat([features.reset_index(drop=True), labels.reset_index(drop=True)], axis=1)
    return out.dropna(subset=["label"]).reset_index(drop=True)


def _build_one(args) -> pd.DataFrame:
    config, symbol, label_config = args
    return build_symbol_dataset(config, symbol, label_config)


def build_dataset(
    config: BacktestConfig,
    symbols: Optional[Iterable[str]] = None,
    label_config: LabelConfig | None = None,
    output_path: str | Path | None = "data/btc_futures_v1/generic_dataset.parquet",
    workers: int | None = None,
    parallel_mode: str = "process",
) -> pd.DataFrame:
    symbol_list = list(symbols or config.normalized_symbols())
    if not symbol_list:
        return pd.DataFrame()

    worker_count = max(1, min(int(workers or config.symbol_workers or 1), len(symbol_list)))
    frames: List[pd.DataFrame] = []

    if worker_count == 1:
        for symbol in symbol_list:
            start = time.perf_counter()
            frame = build_symbol_dataset(config, symbol, label_config)
            frames.append(frame)
            print(f"[ML] built {symbol}: rows={len(frame)} seconds={time.perf_counter() - start:.2f}", flush=True)
    else:
        executor_cls = ThreadPoolExecutor if parallel_mode == "thread" else ProcessPoolExecutor
        with executor_cls(max_workers=worker_count) as pool:
            start_map = {}
            future_map = {
                pool.submit(_build_one, (config, symbol, label_config)): symbol
                for symbol in symbol_list
            }
            start_map.update({future: time.perf_counter() for future in future_map})
            for future in as_completed(future_map):
                symbol = future_map[future]
                try:
                    frame = future.result()
                    frames.append(frame)
                    print(
                        f"[ML] built {symbol}: rows={len(frame)} seconds={time.perf_counter() - start_map[future]:.2f}",
                        flush=True,
                    )
                except Exception:
                    if not config.skip_symbol_errors:
                        raise
                    print(f"[ML][WARN] skip {symbol}: dataset build failed")

    dataset = pd.concat([frame for frame in frames if not frame.empty], ignore_index=True) if frames else pd.DataFrame()
    if not dataset.empty:
        dataset = dataset.sort_values(["exec_time", "symbol", "klu_idx"]).reset_index(drop=True)

    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_parquet(path, index=False)
        print(f"[ML] saved dataset: {path} rows={len(dataset)}")

    return dataset
