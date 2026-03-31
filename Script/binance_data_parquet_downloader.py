#!/usr/bin/env python3
"""
币安期货多币种 K 线数据下载 + 合并工具

功能：
  1. 自动发现 Binance 全量 USDT 永续合约币种（数字货币）
    2. 使用 Binance 接口估算币种市值，保留市值 >= 5000 万美元的币
  3. 仅下载与输出 15m 周期数据，并清理 data 目录下非 15m 产物
  4. 自动解压 ZIP，去重排序后合并为 Parquet 文件

用法：
  python binance_data_parquet_downloader.py                   # 自动筛币并下载15m
  python binance_data_parquet_downloader.py --symbols BTC ETH # 只处理指定币种
  python binance_data_parquet_downloader.py --start-date 2023-01-01
  python binance_data_parquet_downloader.py --merge-only
  python binance_data_parquet_downloader.py --download-only
"""

import sys
import urllib.request
import urllib.error
import zipfile
import json
import gzip
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser, RawDescriptionHelpFormatter
import time
from typing import Dict, List, Optional, Tuple
from multiprocessing import Pool
import logging

try:
    import pandas as pd
    import pyarrow  # noqa: F401
except ImportError:
    print("缺少依赖，请先安装：pip install pandas pyarrow")
    sys.exit(1)


# ==============================================================
#                   ⚙️  配置区（按需修改）
# ==============================================================

class Config:
    """统一配置管理"""

    # ---------- Binance 数据源 ----------
    BASE_URL = "https://data.binance.vision"
    FUTURES_EXCHANGE_INFO_URL = "https://fapi.binance.com/fapi/v1/exchangeInfo"
    SPOT_PRODUCTS_URL = "https://www.binance.com/bapi/asset/v2/public/asset-service/product/get-products?includeEtf=true"

    # ---------- 要下载的交易对（USDT 永续合约） ----------
    # 为空表示自动拉取 Binance 全量 USDT 永续币种，再按市值阈值筛选。
    SYMBOLS: List[str] = []

    # ---------- 时间框架 ----------
    # 仅保留 15m 数据。
    INTERVALS = ["15m"]

    # 为避免误下载 5m/1m，仅允许 15m。
    SUPPORTED_INTERVALS = ["15m"]

    # ---------- 币种市值筛选 ----------
    MIN_MARKET_CAP_USD = 50_000_000

    # ---------- 数据类型：只下载月度数据 ----------
    DOWNLOAD_MONTHLY = True
    DOWNLOAD_DAILY = False

    # ---------- 日期 ----------
    # 各主流币期货上线时间不同，2020-01-01 是安全的通用起点
    DEFAULT_START_DATE = "2020-01-01"

    # ---------- 目录 ----------
    RAW_DIR = "./data/raw"       # 原始 ZIP/CSV 临时存放目录
    OUTPUT_DIR = "./data"        # Parquet 输出目录

    # ---------- Binance K 线 CSV 列定义（文件本身无表头） ----------
    KLINE_COLUMNS = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "count",
        "taker_buy_volume", "taker_buy_quote_volume", "ignore",
    ]

    # ---------- 网络 ----------
    MAX_RETRIES = 3
    RETRY_DELAY = 1
    TIMEOUT = 30
    MAX_WORKERS = 8

    # ---------- 解压 ----------
    AUTO_EXTRACT = True
    DELETE_ZIP_AFTER_EXTRACT = True

    # ---------- 日志 ----------
    VERBOSE = False
    LOG_FILE = "binance_download.log"
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
    SEPARATOR_LENGTH = 70


# ==============================================================
#                      初始化
# ==============================================================

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

logging.basicConfig(
    level=logging.INFO,
    format=Config.LOG_FORMAT,
    datefmt=Config.LOG_DATE_FORMAT,
    handlers=[
        logging.FileHandler(Config.LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# ==============================================================
#                       统计类
# ==============================================================

class DownloadStats:
    def __init__(self):
        self.total_files = 0
        self.downloaded_files = 0
        self.failed_files = 0
        self.skipped_files = 0
        self.total_size = 0
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None

    def duration(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

    def speed_mb(self) -> float:
        d = self.duration()
        return (self.total_size / 1_048_576) / d if d > 0 else 0.0


# ==============================================================
#                     核心下载器
# ==============================================================

class BinanceMultiDownloader:
    """多币种永续合约 K 线下载 + 合并 Parquet 一体化工具"""

    def __init__(
        self,
        raw_dir: str = None,
        output_dir: str = None,
        extract: bool = None,
        max_workers: int = None,
        verbose: bool = None,
    ):
        self.raw_dir = Path(raw_dir or Config.RAW_DIR)
        self.output_dir = Path(output_dir or Config.OUTPUT_DIR)
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.extract = extract if extract is not None else Config.AUTO_EXTRACT
        self.max_workers = max_workers or Config.MAX_WORKERS
        self.verbose = verbose if verbose is not None else Config.VERBOSE

        self.stats = DownloadStats()

    # ------------------------------------------------------------------ #
    #  币种发现与市值筛选
    # ------------------------------------------------------------------ #

    @staticmethod
    def _fetch_json(url: str) -> dict:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urllib.request.urlopen(req, timeout=Config.TIMEOUT) as resp:
            data = resp.read()
            # 检测gzip压缩
            if data[:2] == b'\x1f\x8b':
                data = gzip.decompress(data)
            return json.loads(data.decode("utf-8"))

    def _discover_futures_symbols(self) -> List[str]:
        payload = self._fetch_json(Config.FUTURES_EXCHANGE_INFO_URL)
        symbols: List[str] = []
        for item in payload.get("symbols", []):
            if item.get("status") != "TRADING":
                continue
            if item.get("contractType") != "PERPETUAL":
                continue
            if item.get("quoteAsset") != "USDT":
                continue
            if item.get("underlyingType") != "COIN":
                continue
            symbol = item.get("symbol")
            if symbol and symbol.endswith("USDT"):
                symbols.append(symbol)
        return sorted(set(symbols))

    def _fetch_base_market_caps(self) -> Dict[str, float]:
        payload = self._fetch_json(Config.SPOT_PRODUCTS_URL)
        market_caps: Dict[str, float] = {}
        for row in payload.get("data", []):
            base = row.get("b")
            cs = row.get("cs")
            close_price = row.get("c")
            if not base or cs in (None, "") or close_price in (None, ""):
                continue
            try:
                cap = float(cs) * float(close_price)
            except (TypeError, ValueError):
                continue
            if cap <= 0:
                continue
            old = market_caps.get(base)
            if old is None or cap > old:
                market_caps[base] = cap
        return market_caps

    def _filter_symbols_by_market_cap(self, symbols: List[str], min_cap_usd: float) -> List[str]:
        market_caps = self._fetch_base_market_caps()
        kept: List[str] = []
        dropped = 0
        for symbol in symbols:
            base = symbol.replace("USDT", "")
            cap = market_caps.get(base, 0.0)
            if cap >= min_cap_usd:
                kept.append(symbol)
            else:
                dropped += 1
        logger.info(
            "市值筛选完成: 候选 %d -> 保留 %d (阈值 >= %.0f USD, 剔除 %d)",
            len(symbols),
            len(kept),
            min_cap_usd,
            dropped,
        )
        return kept

    # ------------------------------------------------------------------ #
    #  URL 构造
    # ------------------------------------------------------------------ #

    @staticmethod
    def _build_url(symbol: str, interval: str, date_str: str, freq: str = "monthly") -> str:
        """
        Binance Vision URL 格式：
        https://data.binance.vision/data/futures/um/{monthly|daily}/klines
            /{SYMBOL}/{interval}/{SYMBOL}-{interval}-{date_str}.zip
        """
        path = f"/data/futures/um/{freq}/klines/{symbol}/{interval}"
        filename = f"{symbol}-{interval}-{date_str}.zip"
        return f"{Config.BASE_URL}{path}/{filename}"

    # ------------------------------------------------------------------ #
    #  月份区间生成
    # ------------------------------------------------------------------ #

    @staticmethod
    def _month_range(start_date: str, end_date: str) -> List[str]:
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
        months: List[str] = []
        cur = start.replace(day=1)
        while cur <= end:
            months.append(cur.strftime("%Y-%m"))
            cur = cur.replace(
                year=cur.year + (1 if cur.month == 12 else 0),
                month=(cur.month % 12) + 1,
            )
        return months

    # ------------------------------------------------------------------ #
    #  单文件下载（幂等 + 重试）
    # ------------------------------------------------------------------ #

    def _download_file(self, url: str, save_path: Path) -> Dict[str, int]:
        """下载单个文件，返回统计信息字典"""
        # ZIP 已存在 -> 跳过
        if save_path.exists():
            return {"skipped": 1, "downloaded": 0, "failed": 0, "size": 0}
        # 解压后的 CSV 已存在 -> 跳过
        if save_path.with_suffix(".csv").exists():
            return {"skipped": 1, "downloaded": 0, "failed": 0, "size": 0}

        save_path.parent.mkdir(parents=True, exist_ok=True)

        for attempt in range(Config.MAX_RETRIES):
            try:
                if self.verbose:
                    print(f"[DL] {save_path.name}", flush=True)

                resp = urllib.request.urlopen(url, timeout=Config.TIMEOUT)
                file_size = int(resp.getheader("content-length", 0))
                tmp = save_path.with_suffix(".tmp")

                with open(tmp, "wb") as f:
                    while True:
                        chunk = resp.read(65536)
                        if not chunk:
                            break
                        f.write(chunk)

                tmp.rename(save_path)

                if self.verbose:
                    print(f"[OK] {save_path.name} ({file_size / 1_048_576:.2f} MB)", flush=True)

                if self.extract:
                    self._extract_zip(save_path)

                return {"skipped": 0, "downloaded": 1, "failed": 0, "size": file_size}

            except urllib.error.HTTPError as e:
                if e.code == 404:
                    # 该月份数据不存在（正常情况，不记为失败）
                    return {"skipped": 1, "downloaded": 0, "failed": 0, "size": 0}
                if self.verbose:
                    print(f"HTTP {e.code}: {url}", flush=True)
            except Exception as e:
                if self.verbose:
                    print(f"下载出错 ({attempt + 1}/{Config.MAX_RETRIES}): {e}", flush=True)

            if attempt < Config.MAX_RETRIES - 1:
                time.sleep(Config.RETRY_DELAY)

        return {"skipped": 0, "downloaded": 0, "failed": 1, "size": 0}

    # ------------------------------------------------------------------ #
    #  解压
    # ------------------------------------------------------------------ #

    def _extract_zip(self, zip_path: Path):
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(zip_path.parent)
            if Config.DELETE_ZIP_AFTER_EXTRACT:
                zip_path.unlink()
                if self.verbose:
                    logger.info(f"[OK] 解压并删除 ZIP: {zip_path.name}")
        except Exception as e:
            logger.warning(f"[!] 解压失败 {zip_path.name}: {e}")

    # ------------------------------------------------------------------ #
    #  工作线程
    # ------------------------------------------------------------------ #

    def _worker_task(self, task: Tuple[str, Path]) -> Dict[str, int]:
        """进程池工作任务：下载单个文件，返回统计信息"""
        url, save_path = task
        return self._download_file(url, save_path)

    # ------------------------------------------------------------------ #
    #  单币种下载
    # ------------------------------------------------------------------ #

    def download_symbol(
        self,
        symbol: str,
        intervals: List[str],
        start_date: str,
        end_date: str,
    ):
        months = self._month_range(start_date, end_date)
        logger.info(
            f"\n[{symbol}] 排队 {len(months)} 个月 x {len(intervals)} 个间隔"
        )
        self.stats.total_files += len(months) * len(intervals)

        if not self.verbose:
            print(f"[{symbol}] 下载进度: ", end="", flush=True)

        tasks: List[Tuple[str, Path]] = []
        for interval in intervals:
            for month in months:
                url = self._build_url(symbol, interval, month, "monthly")
                filename = f"{symbol}-{interval}-{month}.zip"
                save_path = self.raw_dir / symbol / interval / filename
                tasks.append((url, save_path))

        # 使用进程池并行下载
        with Pool(processes=self.max_workers) as pool:
            for result in pool.imap_unordered(self._worker_task, tasks, chunksize=2):
                self.stats.downloaded_files += result.get("downloaded", 0)
                self.stats.failed_files += result.get("failed", 0)
                self.stats.skipped_files += result.get("skipped", 0)
                self.stats.total_size += result.get("size", 0)
                if not self.verbose:
                    sys.stdout.write(".")
                    sys.stdout.flush()

        if not self.verbose:
            print()

    # ------------------------------------------------------------------ #
    #  合并 CSV -> Parquet
    # ------------------------------------------------------------------ #

    def merge_to_parquet(self, symbol: str, interval: str):
        csv_dir = self.raw_dir / symbol / interval
        if not csv_dir.exists():
            logger.warning(f"[{symbol}/{interval}] 原始目录不存在: {csv_dir}")
            return

        csv_files = sorted(csv_dir.glob(f"{symbol}-{interval}-*.csv"))
        if not csv_files:
            logger.warning(f"[{symbol}/{interval}] 未找到 CSV 文件，跳过合并")
            return

        logger.info(f"[{symbol}/{interval}] 合并 {len(csv_files)} 个 CSV 文件...")

        dfs: List[pd.DataFrame] = []
        for f in csv_files:
            try:
                df = pd.read_csv(
                    f,
                    header=None,
                    names=Config.KLINE_COLUMNS,
                    dtype=str,
                )
                # 过滤非数字开头的行（Binance 偶尔会在 CSV 里写入表头）
                df = df[df["open_time"].str.match(r"^\d+$", na=False)]
                if not df.empty:
                    dfs.append(df)
            except Exception as e:
                logger.warning(f"  读取 {f.name} 失败: {e}")

        if not dfs:
            logger.warning(f"[{symbol}/{interval}] 无有效数据，跳过")
            return

        result = pd.concat(dfs, ignore_index=True)

        # 类型转换
        result["open_time"] = result["open_time"].astype("int64")
        result["close_time"] = result["close_time"].astype("int64")
        result["count"] = result["count"].astype("int64")

        float_cols = [
            "open", "high", "low", "close", "volume",
            "quote_volume", "taker_buy_volume", "taker_buy_quote_volume",
        ]
        result[float_cols] = result[float_cols].astype("float64")

        # 去重 + 排序
        result = (
            result.drop_duplicates("open_time")
            .sort_values("open_time")
            .reset_index(drop=True)
        )

        # 删除无意义的 ignore 列
        result = result.drop(columns=["ignore"])

        # 输出路径：data/BTC_15m.parquet
        base = symbol.replace("USDT", "")
        out_path = self.output_dir / f"{base}_{interval}.parquet"
        result.to_parquet(out_path, index=False, engine="pyarrow", compression="snappy")

        size_mb = out_path.stat().st_size / 1_048_576
        logger.info(
            f"[{symbol}/{interval}] -> {out_path}  "
            f"({len(result):,} 行, {size_mb:.1f} MB)"
        )

        # Parquet 写入成功后删除原始 CSV 文件
        deleted = 0
        for f in csv_files:
            try:
                f.unlink()
                deleted += 1
            except Exception as e:
                logger.warning(f"  删除 {f.name} 失败: {e}")
        if deleted:
            logger.info(f"[{symbol}/{interval}] 已删除 {deleted} 个原始 CSV 文件")

    # ------------------------------------------------------------------ #
    #  打印统计
    # ------------------------------------------------------------------ #

    def _print_download_summary(self):
        s = self.stats
        logger.info("\n" + "=" * Config.SEPARATOR_LENGTH)
        logger.info("下载完成")
        logger.info("-" * Config.SEPARATOR_LENGTH)
        logger.info(f"  计划文件数  : {s.total_files}")
        logger.info(f"  成功下载    : {s.downloaded_files}")
        logger.info(f"  跳过（已有）: {s.skipped_files}")
        logger.info(f"  失败        : {s.failed_files}")
        logger.info(f"  总下载量    : {s.total_size / 1_048_576:.1f} MB")
        logger.info(f"  耗时        : {int(s.duration())} 秒")
        if s.speed_mb() > 0:
            logger.info(f"  平均速度    : {s.speed_mb():.2f} MB/s")
        logger.info("=" * Config.SEPARATOR_LENGTH)

    # ------------------------------------------------------------------ #
    #  输出清理（仅保留目标时间框架）
    # ------------------------------------------------------------------ #

    def _cleanup_non_target_intervals(self, intervals: List[str]):
        keep = set(intervals)

        removed_parquet = 0
        for p in self.output_dir.glob("*_*.parquet"):
            stem_parts = p.stem.split("_")
            if not stem_parts:
                continue
            interval = stem_parts[-1]
            if interval not in keep:
                try:
                    p.unlink()
                    removed_parquet += 1
                except Exception as e:
                    logger.warning(f"删除旧 Parquet 失败 {p}: {e}")

        removed_raw_dirs = 0
        if self.raw_dir.exists():
            for symbol_dir in self.raw_dir.iterdir():
                if not symbol_dir.is_dir():
                    continue
                for interval_dir in symbol_dir.iterdir():
                    if not interval_dir.is_dir():
                        continue
                    if interval_dir.name not in keep:
                        try:
                            for f in interval_dir.rglob("*"):
                                if f.is_file():
                                    f.unlink()
                            for d in sorted(interval_dir.rglob("*"), reverse=True):
                                if d.is_dir():
                                    d.rmdir()
                            interval_dir.rmdir()
                            removed_raw_dirs += 1
                        except Exception as e:
                            logger.warning(f"删除旧原始目录失败 {interval_dir}: {e}")

        logger.info(
            "已清理非目标周期数据: 删除 Parquet %d 个, 删除原始周期目录 %d 个",
            removed_parquet,
            removed_raw_dirs,
        )

    # ------------------------------------------------------------------ #
    #  完整流程入口
    # ------------------------------------------------------------------ #

    def run(
        self,
        symbols: Optional[List[str]] = None,
        intervals: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        do_download: bool = True,
        do_merge: bool = True,
    ):
        intervals = intervals or Config.INTERVALS
        start_date = start_date or Config.DEFAULT_START_DATE
        end_date = end_date or datetime.now().strftime("%Y-%m-%d")

        if symbols:
            symbols = sorted(set(symbols))
        else:
            discovered = self._discover_futures_symbols()
            symbols = self._filter_symbols_by_market_cap(discovered, Config.MIN_MARKET_CAP_USD)

        if not symbols:
            raise RuntimeError("市值筛选后无可下载币种，请检查网络或阈值配置")

        self._cleanup_non_target_intervals(intervals)

        logger.info("=" * Config.SEPARATOR_LENGTH)
        logger.info("币安期货多币种 K 线数据工具")
        logger.info("-" * Config.SEPARATOR_LENGTH)
        logger.info(f"  币种数量: {len(symbols)}")
        logger.info(f"  间隔    : {', '.join(intervals)}")
        logger.info(f"  市值阈值: >= {Config.MIN_MARKET_CAP_USD:,} USD")
        logger.info(f"  日期    : {start_date} -> {end_date}")
        logger.info(f"  原始目录: {self.raw_dir.absolute()}")
        logger.info(f"  输出目录: {self.output_dir.absolute()}")
        logger.info("=" * Config.SEPARATOR_LENGTH)

        if self.verbose:
            logger.info("筛选后币种列表: %s", ", ".join(symbols))

        self.stats.start_time = datetime.now()

        for symbol in symbols:
            if do_download:
                self.download_symbol(symbol, intervals, start_date, end_date)

            if do_merge:
                logger.info(f"[{symbol}] 开始合并 CSV -> Parquet ...")
                for interval in intervals:
                    self.merge_to_parquet(symbol, interval)

        self.stats.end_time = datetime.now()
        if do_download:
            self._print_download_summary()
        if do_merge:
            logger.info("\n全部合并完成！")
            logger.info(f"文件位置: {self.output_dir.absolute()}")


# ==============================================================
#                       命令行接口
# ==============================================================

def main():
    parser = ArgumentParser(
        description="币安期货多币种 K 线下载 & 合并工具",
        formatter_class=RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 自动获取全量USDT永续币种，按市值阈值筛选并下载15m
  python binance_data_parquet_downloader.py

  # 仅处理指定币种（可写 BTC 或 BTCUSDT）
  python binance_data_parquet_downloader.py --symbols BTC ETH SOL

  # 从 2023 年开始
  python binance_data_parquet_downloader.py --start-date 2023-01-01

  # 只合并已下载的原始 CSV（跳过网络下载）
  python binance_data_parquet_downloader.py --merge-only
""",
    )

    parser.add_argument(
        "--symbols",
        nargs="+",
        metavar="SYMBOL",
        help="指定币种（支持 BTC 或 BTCUSDT，默认自动发现并按市值筛选）",
    )
    parser.add_argument(
        "--intervals",
        nargs="+",
        choices=Config.SUPPORTED_INTERVALS,
        default=Config.INTERVALS,
        help="时间框架（仅支持15m）",
    )
    parser.add_argument(
        "--start-date",
        default=None,
        help=f"开始日期 YYYY-MM-DD（默认：{Config.DEFAULT_START_DATE}）",
    )
    parser.add_argument(
        "--end-date",
        default=None,
        help="结束日期 YYYY-MM-DD（默认：今天）",
    )
    parser.add_argument(
        "--raw-dir",
        default=Config.RAW_DIR,
        help=f"原始文件下载目录（默认：{Config.RAW_DIR}）",
    )
    parser.add_argument(
        "--output-dir",
        default=Config.OUTPUT_DIR,
        help=f"Parquet 输出目录（默认：{Config.OUTPUT_DIR}）",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=Config.MAX_WORKERS,
        help=f"并发下载进程数（默认：{Config.MAX_WORKERS}）",
    )
    parser.add_argument(
        "--merge-only",
        action="store_true",
        help="只合并已有 CSV，不重新下载",
    )
    parser.add_argument(
        "--download-only",
        action="store_true",
        help="只下载 ZIP/CSV，不合并",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="下载后不自动解压 ZIP（保留原始 ZIP）",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="显示详细日志",
    )

    args = parser.parse_args()

    symbols = None
    if args.symbols:
        symbols = []
        for s in args.symbols:
            token = s.upper().strip()
            if not token:
                continue
            symbols.append(token if token.endswith("USDT") else f"{token}USDT")

    extract = False if args.no_extract else None   # None -> 使用 Config.AUTO_EXTRACT

    downloader = BinanceMultiDownloader(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        extract=extract,
        max_workers=args.workers,
        verbose=args.verbose or Config.VERBOSE,
    )

    try:
        downloader.run(
            symbols=symbols,
            intervals=args.intervals,
            start_date=args.start_date,
            end_date=args.end_date,
            do_download=not args.merge_only,
            do_merge=not args.download_only,
        )
    except KeyboardInterrupt:
        logger.warning("\n用户中断")
        sys.exit(1)
    except Exception as e:
        logger.error(f"发生错误: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
