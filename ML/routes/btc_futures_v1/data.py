from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


ROUTE_NAME = "btc_futures_v1"
SYMBOL = "BTCUSDT"
TRAIN_SYMBOLS = (
    "BTCUSDT",
    "ETHUSDT",
    "SOLUSDT",
    "SUIUSDT",
    "HYPEUSDT",
    "ADAUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "DOGEUSDT",
    "AVAXUSDT",
    "LINKUSDT",
)
FREQTRADE_PAIR = "BTC/USDT:USDT"
DEFAULT_SOURCE_PATH = Path(r"D:\WorkSpace\czsc_all\data\freqtrade_futures\futures\BTC_USDT_USDT-1m-futures.parquet")
DEFAULT_FUTURES_DATA_DIR = DEFAULT_SOURCE_PATH.parent
DEFAULT_DATASET_PATH = Path("data/btc_futures_v1/btc_futures_v1_dataset.parquet")
DEFAULT_MODEL_DIR = Path("result/ml/btc_futures_v1")
DEFAULT_BACKTEST_DIR = Path("result/btc_futures_v1")
PEER_MARKET_SYMBOLS = ("ETH", "SOL", "ADA", "BNB", "XRP", "DOGE", "AVAX", "LINK")
EXPERIMENTAL_PEER_MARKET_SYMBOLS = ("SUI", "HYPE")


def futures_1m_path(symbol: str, data_dir: str | Path = DEFAULT_FUTURES_DATA_DIR) -> Path:
    clean = str(symbol).upper().replace("/", "_").replace(":", "_")
    if clean.endswith("_USDT_USDT"):
        name = clean
    elif clean.endswith("USDT") and "_" not in clean:
        name = f"{clean[:-4]}_USDT_USDT"
    elif clean.endswith("USDT"):
        name = clean.replace("USDT", "USDT_USDT")
    else:
        name = f"{clean}_USDT_USDT"
    return Path(data_dir) / f"{name}-1m-futures.parquet"


def normalize_symbol(symbol: str) -> str:
    text = str(symbol).upper().strip().replace("/", "").replace(":", "").replace("_", "")
    if text.endswith("USDTUSDT"):
        text = text[: -len("USDT")]
    if not text.endswith("USDT"):
        text = f"{text}USDT"
    return text


def symbol_asset(symbol: str) -> str:
    text = normalize_symbol(symbol)
    return text[:-4] if text.endswith("USDT") else text


def parse_utc(value: Any | None, *, end_of_day: bool = False) -> pd.Timestamp | None:
    if value in (None, ""):
        return None
    text = str(value).strip().replace("/", "-")
    ts = pd.to_datetime(text, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"invalid datetime: {value!r}")
    out = pd.Timestamp(ts)
    if end_of_day and len(text) == 10:
        out = out + pd.Timedelta(days=1) - pd.Timedelta(nanoseconds=1)
    return out


def load_1m_futures_bars(
    source_path: str | Path = DEFAULT_SOURCE_PATH,
    begin_time: Any | None = None,
    end_time: Any | None = None,
) -> pd.DataFrame:
    path = Path(source_path)
    if not path.exists():
        raise FileNotFoundError(f"futures source data not found: {path}")

    frame = pd.read_parquet(path)
    if "date" in frame.columns:
        time_source = frame["date"]
    else:
        time_source = frame.index

    frame = frame.assign(_time=pd.to_datetime(time_source, utc=True, errors="coerce"))
    required = ["open", "high", "low", "close", "volume"]
    missing = [col for col in required if col not in frame.columns]
    if missing:
        raise ValueError(f"source data missing columns: {missing}")

    for col in required:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    frame = (
        frame.dropna(subset=["_time", "open", "high", "low", "close"])
        .drop_duplicates(subset=["_time"], keep="last")
        .sort_values("_time")
        .set_index("_time")
    )
    frame.index.name = "date"

    begin = parse_utc(begin_time)
    end = parse_utc(end_time, end_of_day=True)
    if begin is not None:
        frame = frame.loc[frame.index >= begin]
    if end is not None:
        frame = frame.loc[frame.index <= end]

    return frame[required].copy()


def resample_ohlcv_open_labeled(bars: pd.DataFrame, rule: str = "15min") -> pd.DataFrame:
    if bars.empty:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

    frame = bars.copy()
    frame.index = pd.to_datetime(frame.index, utc=True)
    out = frame.resample(rule, label="left", closed="left", origin="epoch").agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    out = out.dropna(subset=["open", "high", "low", "close"])
    out.index.name = "date"
    return out
