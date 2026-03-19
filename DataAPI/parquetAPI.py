import os
from datetime import datetime

import pandas as pd

from Common.CEnum import AUTYPE, DATA_FIELD, KL_TYPE
from Common.ChanException import CChanException, ErrCode
from Common.CTime import CTime
from Common.func_util import kltype_lt_day
from KLine.KLine_Unit import CKLine_Unit

from .CommonStockAPI import CCommonStockApi


class PARQUET_API(CCommonStockApi):
    FILE_INTERVAL_MAP = {
        KL_TYPE.K_1M: "1m",
        KL_TYPE.K_3M: "3m",
        KL_TYPE.K_5M: "5m",
        KL_TYPE.K_10M: "10m",
        KL_TYPE.K_15M: "15m",
        KL_TYPE.K_30M: "30m",
        KL_TYPE.K_60M: "1h",
        KL_TYPE.K_DAY: "1d",
        KL_TYPE.K_WEEK: "1w",
        KL_TYPE.K_MON: "1mo",
    }

    RESAMPLE_RULE_MAP = {
        KL_TYPE.K_10M: "10min",
        KL_TYPE.K_15M: "15min",
        KL_TYPE.K_30M: "30min",
        KL_TYPE.K_60M: "1h",
        KL_TYPE.K_DAY: "1D",
        KL_TYPE.K_WEEK: "1W",
        KL_TYPE.K_MON: "1MS",
    }

    FALLBACK_INTERVAL = "5m"
    REQUIRED_COLUMNS = ["open_time", "open", "high", "low", "close", "volume"]

    def __init__(self, code, k_type=KL_TYPE.K_DAY,
                 begin_date=None, end_date=None,
                 autype=AUTYPE.NONE):
        super(PARQUET_API, self).__init__(code, k_type, begin_date, end_date, autype)

    def get_kl_data(self):
        df = self._load_dataframe()
        auto = not kltype_lt_day(self.k_type)
        for row in df.itertuples(index=False):
            dt = row.open_time.to_pydatetime()
            item = {
                DATA_FIELD.FIELD_TIME: CTime(
                    dt.year,
                    dt.month,
                    dt.day,
                    dt.hour,
                    dt.minute,
                    dt.second,
                    auto=auto,
                ),
                DATA_FIELD.FIELD_OPEN: float(row.open),
                DATA_FIELD.FIELD_HIGH: float(row.high),
                DATA_FIELD.FIELD_LOW: float(row.low),
                DATA_FIELD.FIELD_CLOSE: float(row.close),
                DATA_FIELD.FIELD_VOLUME: float(row.volume),
            }
            yield CKLine_Unit(item, autofix=True)

    def _load_dataframe(self):
        file_path, need_resample = self._resolve_source_path()

        try:
            df = pd.read_parquet(file_path, columns=self.REQUIRED_COLUMNS)
        except Exception as exc:
            raise CChanException(
                f"failed to read parquet file: {file_path} ({exc})",
                ErrCode.SRC_DATA_FORMAT_ERROR,
            )

        missing = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise CChanException(
                f"parquet missing required columns {missing}: {file_path}",
                ErrCode.SRC_DATA_FORMAT_ERROR,
            )

        df = df[self.REQUIRED_COLUMNS].copy()
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        for col in self.REQUIRED_COLUMNS[1:]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        df = (
            df.dropna(subset=["open_time", "open", "high", "low", "close"])
            .drop_duplicates(subset=["open_time"])
            .sort_values("open_time")
            .reset_index(drop=True)
        )

        if need_resample:
            df = self._resample_from_fallback(df)

        begin_ts = self._parse_bound(self.begin_date, is_end=False)
        end_ts = self._parse_bound(self.end_date, is_end=True)

        if begin_ts is not None:
            df = df[df["open_time"] >= begin_ts]
        if end_ts is not None:
            df = df[df["open_time"] <= end_ts]

        return df.reset_index(drop=True)

    def _resolve_source_path(self):
        if self.k_type not in self.FILE_INTERVAL_MAP:
            raise CChanException(
                f"unsupported parquet k_type: {self.k_type}",
                ErrCode.SRC_DATA_FORMAT_ERROR,
            )

        data_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "data")
        base_symbol = self.code.replace("USDT", "")
        target_interval = self.FILE_INTERVAL_MAP[self.k_type]
        target_path = os.path.join(data_dir, f"{base_symbol}_{target_interval}.parquet")
        if os.path.exists(target_path):
            return target_path, False

        if self.k_type in self.RESAMPLE_RULE_MAP:
            fallback_path = os.path.join(data_dir, f"{base_symbol}_{self.FALLBACK_INTERVAL}.parquet")
            if os.path.exists(fallback_path):
                return fallback_path, True

        raise CChanException(
            f"file not exist: {target_path}",
            ErrCode.SRC_DATA_NOT_FOUND,
        )

    def _resample_from_fallback(self, df: pd.DataFrame):
        rule = self.RESAMPLE_RULE_MAP.get(self.k_type)
        if rule is None:
            raise CChanException(
                f"cannot resample fallback parquet for k_type: {self.k_type}",
                ErrCode.SRC_DATA_FORMAT_ERROR,
            )

        agg = {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
        resampled = (
            df.set_index("open_time")
            .resample(rule, label="left", closed="left")
            .agg(agg)
            .dropna(subset=["open", "high", "low", "close"])
            .reset_index()
        )
        return resampled

    @staticmethod
    def _parse_bound(value, is_end: bool):
        if not value:
            return None

        text = str(value).replace("/", "-").strip()
        formats = [
            "%Y-%m-%d",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d %H:%M:%S",
        ]
        dt = None
        for fmt in formats:
            try:
                dt = datetime.strptime(text, fmt)
                break
            except ValueError:
                continue
        if dt is None:
            raise CChanException(
                f"unknown time column from parquet bound: {value}",
                ErrCode.SRC_DATA_FORMAT_ERROR,
            )

        if len(text) == 10 and is_end:
            dt = dt.replace(hour=23, minute=59, second=59)
        return pd.Timestamp(dt, tz="UTC")

    def SetBasciInfo(self):
        pass

    @classmethod
    def do_init(cls):
        pass

    @classmethod
    def do_close(cls):
        pass