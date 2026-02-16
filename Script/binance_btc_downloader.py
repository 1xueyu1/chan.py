#!/usr/bin/env python3
"""
币安期货 BTCUSDT 数据下载工具
专门用于下载 BTCUSDT 的多时间框架 K 线数据

支持的时间框架：5m, 15m, 1h, 4h, 1d

用法示例：
    # 下载BTCUSDT所有数据（默认5m,15m,1h,4h,1d）
    python3 binance_btc_downloader.py
    
    # 指定下载目录
    python3 binance_btc_downloader.py --dir /data/binance
    
    # 指定日期范围
    python3 binance_btc_downloader.py --start-date 2024-01-01 --end-date 2024-12-31
    
    # 只下载月数据（更快）
    python3 binance_btc_downloader.py --skip-daily
    
    # 自动解压
    python3 binance_btc_downloader.py --extract
    
    # 显示下载进度和统计信息
    python3 binance_btc_downloader.py --verbose
"""

import os
import sys
import urllib.request
import urllib.error
import zipfile
from pathlib import Path
from datetime import datetime, date, timedelta
from argparse import ArgumentParser
import time
from typing import List, Optional
import threading
from queue import Queue
import logging


# ==================== ⚙️ 配置信息（请在此修改配置） ====================

class Config:
    """统一配置管理类"""
    
    # ============ Binance API 配置 ============
    BASE_URL = 'https://data.binance.vision'  # Binance Vision 数据源
    SYMBOL = "BTCUSDT"  # 下载的交易对
    
    # ============ Daily/Monthly 数据类型选择 ============
    # 【重要配置】选择要下载的数据类型
    # 说明:
    #   - DOWNLOAD_MONTHLY = True  : 下载月度数据（推荐，快速）
    #   - DOWNLOAD_DAILY = True    : 下载日度数据（灵活，但文件数多）
    #   - 两者都为 True             : 同时下载月和日数据
    #   - 两者都为 False            : 无任何数据下载（不建议）
    
    # 是否下载月度数据（按月）
    # ✅ True  - 下载月数据：12个文件/年（快速，推荐）
    # ❌ False - 跳过月数据
    DOWNLOAD_MONTHLY = True
    
    # 是否下载日度数据（按天）
    # ✅ True  - 下载日数据：365个文件/年（灵活，慢）
    # ❌ False - 跳过日数据（推荐）
    DOWNLOAD_DAILY = False
    
    # ============ 下载相关配置 ============
    # 默认下载的时间框架
    DEFAULT_INTERVALS = ["5m", "15m", "1h", "4h", "1d"]
    
    # 所有支持的时间框架
    SUPPORTED_INTERVALS = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1mo']
    
    # 默认开始日期（BTC期货上线时间）
    DEFAULT_START_DATE = "2017-11-08"
    
    # ============ 下载性能配置 ============
    # 最大重试次数
    MAX_RETRIES = 3
    
    # 重试延迟（秒）
    RETRY_DELAY = 1
    
    # 网络超时时间（秒）
    TIMEOUT = 30
    
    # 最大并发下载线程数
    MAX_WORKERS = 10
    
    # ============ 文件保存配置 ============
    # 默认保存目录
    DEFAULT_SAVE_DIR = './btc_data'
    
    # 日志文件名
    LOG_FILE = 'btc_download.log'
    
    # ============ 自动解压配置 ============
    # 是否自动解压（True = 自动解压，False = 不解压）
    # 【重要】改为 True 以启用自动解压功能
    AUTO_EXTRACT = True
    
    # 解压后是否删除原 ZIP 文件（节省空间）
    # ✅ True  - 解压后删除 ZIP（节省��间，但无法重新解压）
    # ❌ False - 保留 ZIP 文件（占用空间，但可重新操作）
    DELETE_ZIP_AFTER_EXTRACT = True
    
    # ============ 显示配置 ============
    # 是否显示详细信息（True/False）
    VERBOSE = False
    
    # ============ UI 配置 ============
    # 符号定义（兼容Windows）
    SYMBOLS = {
        'download': '[DL]',      # ⬇
        'success': '[OK]',       # ✓
        'failed': '[X]',         # ✗
        'warning': '[!]',        # ⚠
    }
    
    # 日志格式
    LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
    LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
    
    # 分隔线长度
    SEPARATOR_LENGTH = 70


# ==================== 修复Windows编码问题 ====================
if sys.platform == 'win32':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')


# ==================== 日志配置 ====================
logging.basicConfig(
    level=logging.INFO,
    format=Config.LOG_FORMAT,
    datefmt=Config.LOG_DATE_FORMAT,
    handlers=[
        logging.FileHandler(Config.LOG_FILE, encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# ==================== 统计类 ====================
class DownloadStats:
    """下载统计信息"""
    
    def __init__(self):
        self.total_files = 0           # 总文件数
        self.downloaded_files = 0      # 已下载文件数
        self.failed_files = 0          # 失败文件数
        self.skipped_files = 0         # 跳过文件数
        self.total_size = 0            # 总下载大小（字节）
        self.start_time = None         # 开始时间
        self.end_time = None           # 结束时间
    
    def get_duration(self) -> float:
        """获取下载耗时（秒）"""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0
    
    def get_speed(self) -> float:
        """获取下载速度（MB/s）"""
        duration = self.get_duration()
        if duration > 0:
            return (self.total_size / (1024 * 1024)) / duration
        return 0


# ==================== BTCUSDT 数据下载器 ====================
class BinanceBTCDownloader:
    """币安 BTCUSDT 数据下载器"""
    
    def __init__(self, save_dir: str = None, extract: bool = None, max_workers: int = None, verbose: bool = None):
        """
        初始化下载器
        
        Args:
            save_dir: 保存目录
            extract: 是否自动解压ZIP文件
            max_workers: 最大并发线程数
            verbose: 是否显示详细信息
        """
        # 使用配置文件中的默认值，或使用传入的参数
        self.save_dir = Path(save_dir or Config.DEFAULT_SAVE_DIR)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 重要：如果 extract 为 None，使用配置中的 AUTO_EXTRACT
        # 如果为真，则无论配置如何都强制解压
        self.extract = extract if extract is not None else Config.AUTO_EXTRACT
        
        self.max_workers = max_workers or Config.MAX_WORKERS
        self.verbose = verbose if verbose is not None else Config.VERBOSE
        
        self.stats = DownloadStats()
        self.download_queue = Queue()
        self.lock = threading.Lock()
        
        logger.info(f"初始化下载器 - 保存目录: {self.save_dir.absolute()}")
        logger.info(f"下载交易对: {Config.SYMBOL}")
        logger.info(f"自动解压: {'启用' if self.extract else '禁用'}")
    
    def construct_url(self,
                     interval: str,
                     date_str: str,
                     frequency: str = "monthly") -> str:
        """
        构建下载URL - 根据官方Binance Vision格式
        
        URL格式:
        - 日数据: https://data.binance.vision/data/futures/um/daily/klines/BTCUSDT/5m/BTCUSDT-5m-2024-01-01.zip
        - 月数据: https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/5m/BTCUSDT-5m-2024-01.zip
        
        Args:
            interval: 时间间隔 (e.g., 5m)
            date_str: 日期字符串 (YYYY-MM 或 YYYY-MM-DD)
            frequency: 频率 (daily/monthly)
            
        Returns:
            完整的下载URL
        """
        path = f"/data/futures/um/{frequency}/klines/{Config.SYMBOL}/{interval}"
        filename = f"{Config.SYMBOL}-{interval}-{date_str}.zip"
        full_url = f"{Config.BASE_URL}{path}/{filename}"
        return full_url
    
    def download_file(self, 
                     url: str, 
                     save_path: Path,
                     retries: int = None) -> bool:
        """
        下载单个文件（带重试机制）
        
        Args:
            url: 下载URL
            save_path: 保存路径
            retries: 重试次数
            
        Returns:
            是否成功
        """
        retries = retries or Config.MAX_RETRIES
        
        # 检查文件是否已存在
        if save_path.exists():
            with self.lock:
                self.stats.skipped_files += 1
            if self.verbose:
                logger.debug(f"跳过已存在文件: {save_path.name}")
            return True
        
        # 确保目录存在
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        for attempt in range(retries):
            try:
                if self.verbose:
                    logger.info(f"{Config.SYMBOLS['download']} 下载: {save_path.name}")
                else:
                    sys.stdout.write('.')
                    sys.stdout.flush()
                
                response = urllib.request.urlopen(url, timeout=Config.TIMEOUT)
                file_size = int(response.getheader('content-length', 0))
                
                # 临时保存为.tmp文件
                tmp_path = save_path.with_suffix('.tmp')
                
                with open(tmp_path, 'wb') as f:
                    downloaded = 0
                    block_size = max(8192, file_size // 100) if file_size else 8192
                    
                    while True:
                        chunk = response.read(block_size)
                        if not chunk:
                            break
                        downloaded += len(chunk)
                        f.write(chunk)
                
                # 下载完成后重命名
                tmp_path.rename(save_path)
                
                with self.lock:
                    self.stats.downloaded_files += 1
                    self.stats.total_size += file_size
                
                if self.verbose:
                    logger.info(f"{Config.SYMBOLS['success']} 完成: {save_path.name} ({file_size/(1024*1024):.2f}MB)")
                
                # 如果需要解压 - 【关键】这里判断 self.extract
                if self.extract:
                    self.extract_file(save_path)
                
                return True
                
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    if self.verbose:
                        logger.warning(f"{Config.SYMBOLS['failed']} 文件不存在: {url}")
                    with self.lock:
                        self.stats.failed_files += 1
                    return False
                if self.verbose:
                    logger.warning(f"{Config.SYMBOLS['warning']} HTTP错误 {e.code}: {url}")
                
            except urllib.error.URLError as e:
                if self.verbose:
                    logger.warning(f"{Config.SYMBOLS['warning']} 网络错误: {e.reason}")
                
            except Exception as e:
                if self.verbose:
                    logger.warning(f"{Config.SYMBOLS['warning']} 下载出错: {e}")
            
            if attempt < retries - 1:
                time.sleep(Config.RETRY_DELAY)
        
        if self.verbose:
            logger.error(f"{Config.SYMBOLS['failed']} 下载失败 (超出重试次数): {save_path.name}")
        with self.lock:
            self.stats.failed_files += 1
        return False
    
    def extract_file(self, zip_path: Path):
        """解压ZIP文件"""
        try:
            extract_to = zip_path.parent
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_to)
            if self.verbose:
                logger.info(f"{Config.SYMBOLS['success']} 已解压: {zip_path.name}")
            
            # 根据配置决定是否删除 ZIP 文件
            if Config.DELETE_ZIP_AFTER_EXTRACT:
                zip_path.unlink()
                if self.verbose:
                    logger.info(f"{Config.SYMBOLS['success']} 已删除: {zip_path.name}")
            
        except Exception as e:
            logger.warning(f"{Config.SYMBOLS['warning']} 解压失败: {e}")
    
    def generate_date_range(self, start_date: str, end_date: str) -> List[str]:
        """
        生成日期范围
        
        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            
        Returns:
            日期列表 (YYYY-MM-DD格式)
        """
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
        
        dates = []
        current = start
        while current <= end:
            dates.append(current.strftime("%Y-%m-%d"))
            current += timedelta(days=1)
        
        return dates
    
    def generate_month_range(self, start_date: str, end_date: str) -> List[str]:
        """
        生成月份范围
        
        Args:
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            
        Returns:
            月份列表 (YYYY-MM格式)
        """
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
        
        months = []
        current = start.replace(day=1)
        
        while current <= end:
            months.append(current.strftime("%Y-%m"))
            
            # 移到下一个月
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
        
        return months
    
    def worker_thread(self):
        """工作线程 - 处理下载队列"""
        while True:
            item = self.download_queue.get()
            if item is None:
                break
            
            url, save_path = item
            self.download_file(url, save_path)
            self.download_queue.task_done()
    
    def download_klines(self,
                       intervals: Optional[List[str]] = None,
                       start_date: Optional[str] = None,
                       end_date: Optional[str] = None,
                       skip_monthly: bool = None,
                       skip_daily: bool = None,
                       use_threading: bool = True):
        """
        下载 BTCUSDT K线数据
        
        Args:
            intervals: 时间间隔列表，None表示使用默认值
            start_date: 开始日期 (YYYY-MM-DD)
            end_date: 结束日期 (YYYY-MM-DD)
            skip_monthly: 是否跳过月数据（None表示使用配置）
            skip_daily: 是否跳过日数据（None表示使用配置）
            use_threading: 是否使用多线程下载
        """
        # 设置默认时间间隔
        if not intervals:
            intervals = Config.DEFAULT_INTERVALS
        
        # 【重要】使用配置中的 Daily/Monthly 设置
        # 如果命令行没有指定，就使用配置文件中的设置
        if skip_monthly is None:
            skip_monthly = not Config.DOWNLOAD_MONTHLY
        if skip_daily is None:
            skip_daily = not Config.DOWNLOAD_DAILY
        
        # 设置日期范围
        if not start_date:
            start_date = Config.DEFAULT_START_DATE
        if not end_date:
            end_date = datetime.now().strftime("%Y-%m-%d")
        
        # 打印下载配置
        self._print_config(intervals, start_date, end_date, skip_monthly, skip_daily)
        
        self.stats.start_time = datetime.now()
        
        try:
            # 启动工作线程
            threads = []
            if use_threading:
                for _ in range(self.max_workers):
                    t = threading.Thread(target=self.worker_thread)
                    t.daemon = True
                    t.start()
                    threads.append(t)
            
            # 下载月数据
            if not skip_monthly:
                logger.info("\n" + "="*Config.SEPARATOR_LENGTH)
                logger.info("[1/2] 开始下载月度K线数据")
                logger.info("="*Config.SEPARATOR_LENGTH)
                self._enqueue_monthly_downloads(intervals, start_date, end_date, use_threading)
                if not use_threading:
                    print()
            
            # 下载日数据
            if not skip_daily:
                logger.info("\n" + "="*Config.SEPARATOR_LENGTH)
                logger.info("[2/2] 开始下载日度K线数据")
                logger.info("="*Config.SEPARATOR_LENGTH)
                self._enqueue_daily_downloads(intervals, start_date, end_date, use_threading)
                if not use_threading:
                    print()
            
            # 等待所有下载完成
            if use_threading:
                self.download_queue.join()
                for _ in range(self.max_workers):
                    self.download_queue.put(None)
                for t in threads:
                    t.join()
            
        except KeyboardInterrupt:
            logger.warning("\n下载被用户中断")
            sys.exit(1)
        finally:
            self.stats.end_time = datetime.now()
            self._print_summary()
    
    def _enqueue_monthly_downloads(self, intervals: List[str], 
                                   start_date: str, end_date: str, use_threading: bool):
        """将月度下载任务加入队列"""
        start = datetime.strptime(start_date, "%Y-%m-%d").date()
        end = datetime.strptime(end_date, "%Y-%m-%d").date()
        months = self.generate_month_range(start_date, end_date)
        
        total_tasks = len(intervals) * len(months)
        self.stats.total_files = total_tasks
        
        logger.info(f"准备下载 {total_tasks} 个文件（{len(intervals)} 个间隔 x {len(months)} 个月份）")
        if months:
            logger.info(f"时间范围: {months[0]} 至 {months[-1]}\n")
        else:
            logger.info("未找到符合条件的月份\n")
            return
        
        if not use_threading:
            print("下载进度: ", end="", flush=True)
        
        for interval in intervals:
            for month_str in months:
                month_date = datetime.strptime(f"{month_str}-01", "%Y-%m-%d").date()
                if month_date < start or month_date > end:
                    continue
                
                url = self.construct_url(interval, month_str, "monthly")
                filename = f"{Config.SYMBOL}-{interval}-{month_str}.zip"
                save_path = self.save_dir / "monthly" / interval / filename
                
                if use_threading:
                    self.download_queue.put((url, save_path))
                else:
                    self.download_file(url, save_path)
    
    def _enqueue_daily_downloads(self, intervals: List[str], 
                                start_date: str, end_date: str, use_threading: bool):
        """将日度下载任务加入队列"""
        dates = self.generate_date_range(start_date, end_date)
        
        total_tasks = len(intervals) * len(dates)
        self.stats.total_files += total_tasks
        
        logger.info(f"准备下载 {total_tasks} 个文件（{len(intervals)} 个间隔 x {len(dates)} 天）")
        if dates:
            logger.info(f"时间范围: {dates[0]} 至 {dates[-1]}\n")
        else:
            logger.info("未找到符合条件的日期\n")
            return
        
        if not use_threading:
            print("下载进度: ", end="", flush=True)
        
        for interval in intervals:
            for date_str in dates:
                url = self.construct_url(interval, date_str, "daily")
                filename = f"{Config.SYMBOL}-{interval}-{date_str}.zip"
                save_path = self.save_dir / "daily" / interval / filename
                
                if use_threading:
                    self.download_queue.put((url, save_path))
                else:
                    self.download_file(url, save_path)
    
    def _print_config(self, intervals: List[str], 
                     start_date: str, end_date: str, skip_monthly: bool, skip_daily: bool):
        """打印下载配置信息"""
        logger.info("\n" + "="*Config.SEPARATOR_LENGTH)
        logger.info("币安期货 BTCUSDT K线数据下载配置")
        logger.info("="*Config.SEPARATOR_LENGTH)
        logger.info(f"交易对: {Config.SYMBOL}")
        logger.info(f"时间间隔: {', '.join(intervals)}")
        logger.info(f"日期范围: {start_date} 至 {end_date}")
        logger.info(f"保存目录: {self.save_dir.absolute()}")
        download_modes = []
        if not skip_monthly:
            download_modes.append("月数据")
        if not skip_daily:
            download_modes.append("日数据")
        logger.info(f"下载模式: {', '.join(download_modes) if download_modes else '无'}")
        logger.info(f"并发线程: {self.max_workers}")
        logger.info(f"自动解压: {'是' if self.extract else '否'}")
        logger.info("="*Config.SEPARATOR_LENGTH + "\n")
    
    def _print_summary(self):
        """打印下载摘要"""
        duration = self.stats.get_duration()
        speed = self.stats.get_speed()
        
        logger.info("\n" + "="*Config.SEPARATOR_LENGTH)
        logger.info("下载统计信息")
        logger.info("="*Config.SEPARATOR_LENGTH)
        logger.info(f"总计划: {self.stats.total_files} 个文件")
        logger.info(f"成功下载: {self.stats.downloaded_files} 个文件")
        logger.info(f"下载失败: {self.stats.failed_files} 个文件")
        logger.info(f"跳过（已存在）: {self.stats.skipped_files} 个文件")
        logger.info(f"总下载大小: {self.stats.total_size/(1024*1024):.2f} MB")
        logger.info(f"耗时: {int(duration)} 秒")
        if speed > 0:
            logger.info(f"平均速度: {speed:.2f} MB/s")
        logger.info(f"文件保存位置: {self.save_dir.absolute()}")
        logger.info("="*Config.SEPARATOR_LENGTH + "\n")


# ==================== 命令行接口 ====================
def main():
    parser = ArgumentParser(
        description="币安期货 BTCUSDT K线数据下载工具",
        epilog="""
示例用法：
  # 使用配置文件中的 Daily/Monthly 设置下载
  python binance_btc_downloader.py
  
  # 指定下载目录
  python binance_btc_downloader.py --dir ./btc_data
  
  # 只下载月数据（推荐，更快）
  python binance_btc_downloader.py --skip-daily
  
  # 只下载日数据（灵活，但慢）
  python binance_btc_downloader.py --skip-monthly
  
  # 指定日期范围
  python binance_btc_downloader.py --start-date 2024-01-01 --end-date 2024-12-31
  
  # 禁用自动解压
  python binance_btc_downloader.py --no-extract
  
  # 只下载特定时间框架
  python binance_btc_downloader.py --intervals 5m 15m 1h
  
  # 增加并发线程数（更快）
  python binance_btc_downloader.py --workers 5
        """
    )
    
    parser.add_argument('--dir', 
                       default=Config.DEFAULT_SAVE_DIR,
                       help=f'保存目录 (默认: {Config.DEFAULT_SAVE_DIR})')
    
    parser.add_argument('-i', '--intervals',
                       nargs='+',
                       choices=Config.SUPPORTED_INTERVALS,
                       help=f'时间间隔 (默认: {" ".join(Config.DEFAULT_INTERVALS)})')
    
    parser.add_argument('--extract',
                       action='store_true',
                       help='下载后自动解压ZIP文件')
    
    parser.add_argument('--no-extract',
                       action='store_true',
                       help='禁用自动解压（默认启用）')
    
    parser.add_argument('--start-date',
                       help=f'开始日期 (YYYY-MM-DD，默认: {Config.DEFAULT_START_DATE})')
    
    parser.add_argument('--end-date',
                       help='结束日期 (YYYY-MM-DD，默认: ���天)')
    
    parser.add_argument('--skip-monthly',
                       action='store_true',
                       help='跳过月数据（仅下载日数据）')
    
    parser.add_argument('--skip-daily',
                       action='store_true',
                       help='跳过日数据（仅下载月数据）')
    
    parser.add_argument('--workers',
                       type=int,
                       default=Config.MAX_WORKERS,
                       help=f'并发下载线程数 (默认: {Config.MAX_WORKERS})')
    
    parser.add_argument('--verbose', '-v',
                       action='store_true',
                       help='显示详细信息')
    
    parser.add_argument('--no-threading',
                       action='store_true',
                       help='禁用多线程（单线程下载，调试用）')
    
    args = parser.parse_args()
    
    # 【重要修复】处理 extract 参数
    extract_flag = None
    if args.no_extract:
        extract_flag = False
    elif args.extract:
        extract_flag = True
    
    # 初始化下载器
    downloader = BinanceBTCDownloader(
        save_dir=args.dir,
        extract=extract_flag,
        max_workers=args.workers,
        verbose=args.verbose or Config.VERBOSE
    )
    
    try:
        # 开始下载
        downloader.download_klines(
            intervals=args.intervals if args.intervals else Config.DEFAULT_INTERVALS,
            start_date=args.start_date,
            end_date=args.end_date,
            skip_monthly=args.skip_monthly if args.skip_monthly else None,
            skip_daily=args.skip_daily if args.skip_daily else None,
            use_threading=not args.no_threading
        )
        
    except KeyboardInterrupt:
        logger.warning("\n程序被用户中断")
        sys.exit(1)
    except Exception as e:
        logger.error(f"发生错误: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()