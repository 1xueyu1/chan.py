from datetime import datetime
import os
import ccxt
from ChanConfig import get_proxy, set_proxy
from Common.CEnum import AUTYPE, DATA_FIELD, KL_TYPE
from Common.CTime import CTime
from Common.func_util import kltype_lt_day, str2float
from KLine.KLine_Unit import CKLine_Unit

from .CommonStockAPI import CCommonStockApi


def GetColumnNameFromFieldList(fileds: str):
    """
    将以逗号分隔的字段名映射为内部的 DATA_FIELD 常量列表。

    参数:
        fileds: 字符串形式的字段列表，例如 "time,open,high,low,close"。

    返回:
        对应的 DATA_FIELD 枚举组成的列表，顺序与输入字段一致。
    """
    _dict = {
        "time": DATA_FIELD.FIELD_TIME,
        "open": DATA_FIELD.FIELD_OPEN,
        "high": DATA_FIELD.FIELD_HIGH,
        "low": DATA_FIELD.FIELD_LOW,
        "close": DATA_FIELD.FIELD_CLOSE,
    }
    return [_dict[x] for x in fileds.split(",")]


class CCXT(CCommonStockApi):
    """
    基于 ccxt 的交易所数据接口实现，用于获取 K 线（OHLCV）数据并转换为项目内部 KLine 单元。

    说明:
    - 使用 `ccxt.binance()` 作为示例交易所（可按需替换）。
    - 支持不同时间粒度的映射（天/周/月/分钟等）。
    """
    is_connect = None

    def __init__(self, code, k_type=KL_TYPE.K_DAY, begin_date=None, end_date=None, autype=AUTYPE.QFQ):
        """初始化 CCXT 数据源实例。

        参数:
            code: 交易对或合约代码（如 'BTC/USDT'）。
            k_type: K 线类型，使用 `KL_TYPE` 枚举。
            begin_date: 起始日期字符串，格式 'YYYY-MM-DD'。
            end_date: 结束日期（未使用，但保留接口一致性）。
            autype: 复权类型（保留接口参数）。
        """
        super(CCXT, self).__init__(code, k_type, begin_date, end_date, autype)

    def get_kl_data(self):
        """
        从交易所拉取 OHLCV 数据并以 `CKLine_Unit` 逐条生成。

        处理流程:
        1. 映射内部字段顺序为 ['time','open','high','low','close']。
        2. 使用 ccxt 拉取指定 `code` 和时间粒度的 OHLCV 数据（自 `begin_date` 起）。
        3. 将时间戳转换为格式化字符串后，构建项目内部的数据字典并包裹为 `CKLine_Unit`。

        注意:
        - ccxt 返回的时间为毫秒时间戳。
        - 目前默认使用 `binance` 交易所实例。
        """
        fields = "time,open,high,low,close"
        # 使用 Clash 本地代理（可通过环境变量覆盖），并设置超时与限速
        # 优先从项目配置读取代理；若未设置则使用环境变量或默认值，并尝试写回配置文件
        cfg = get_proxy()
        if cfg.get('http'):
            proxy_http = cfg.get('http')
            proxy_https = cfg.get('https', proxy_http)
        else:
            proxy_http = os.getenv('HTTP_PROXY', 'http://127.0.0.1:7890')
            proxy_https = os.getenv('HTTPS_PROXY', proxy_http)
            try:
                # 将首次检测到的代理写回配置以便持久化
                set_proxy(proxy_http, proxy_https, persist=True)
            except Exception:
                pass

        exchange = ccxt.binance({
            'enableRateLimit': True,
            'timeout': 30000,
            'proxies': {'http': proxy_http, 'https': proxy_https},
        })
        timeframe = self.__convert_type()
        since_date = exchange.parse8601(f'{self.begin_date}T00:00:00')
        data = exchange.fetch_ohlcv(self.code, timeframe, since=since_date)

        for item in data:
            # item[0] 是毫秒时间戳，需要转换为可读时间字符串
            time_obj = datetime.fromtimestamp(item[0] / 1000)
            time_str = time_obj.strftime('%Y-%m-%d %H:%M:%S')
            item_data = [
                time_str,
                item[1],
                item[2],
                item[3],
                item[4]
            ]
            # 将行数据转换为内部字段名后封装为 CKLine_Unit 并生成
            yield CKLine_Unit(self.create_item_dict(item_data, GetColumnNameFromFieldList(fields)), autofix=True)

    def SetBasciInfo(self):
        """占位方法：保留给父类或外部调用以设置基础信息（目前未实现）。"""
        pass

    @classmethod
    def do_init(cls):
        """类级别的初始化钩子（目前无操作）。"""
        pass

    @classmethod
    def do_close(cls):
        """类级别的关闭/清理钩子（目前无操作）。"""
        pass

    def __convert_type(self):
        """将内部 `KL_TYPE` 映射为 ccxt 支持的 timeframe 字符串。"""
        _dict = {
            KL_TYPE.K_DAY: '1d',
            KL_TYPE.K_WEEK: '1w',
            KL_TYPE.K_MON: '1M',
            KL_TYPE.K_5M: '5m',
            KL_TYPE.K_15M: '15m',
            KL_TYPE.K_30M: '30m',
            KL_TYPE.K_60M: '1h',
        }
        return _dict[self.k_type]

    def parse_time_column(self, inp):
        """
        解析不同格式的时间字符串并返回 `CTime` 对象。

        支持的输入长度与示例格式：
        - 10 长度: 'YYYY-MM-DD' -> 仅日期，时分设为 0
        - 17 长度: 紧凑格式例如 'YYYYMMDDhhmmss'（项目中可能存在的自定义格式）
        - 19 长度: 'YYYY-MM-DD HH:MM:SS' -> 标准带时分秒格式

        返回的 `CTime.auto` 标志根据 k 线粒度是否小于天来决定。
        """
        if len(inp) == 10:
            year = int(inp[:4])
            month = int(inp[5:7])
            day = int(inp[8:10])
            hour = minute = 0
        elif len(inp) == 17:
            year = int(inp[:4])
            month = int(inp[4:6])
            day = int(inp[6:8])
            hour = int(inp[8:10])
            minute = int(inp[10:12])
        elif len(inp) == 19:
            year = int(inp[:4])
            month = int(inp[5:7])
            day = int(inp[8:10])
            hour = int(inp[11:13])
            minute = int(inp[14:16])
        else:
            raise Exception(f"unknown time column from TradingView:{inp}")
        return CTime(year, month, day, hour, minute, auto=not kltype_lt_day(self.k_type))

    def create_item_dict(self, data, column_name):
        """
        将原始行数据转换为带类型的字典。

        - 第一列被视为时间，需要解析为 `CTime`。
        - 其余列尝试转换为浮点数。
        - 最后将 `column_name`（字段枚举列表）与数据 zip 返回字典。
        """
        for i in range(len(data)):
            data[i] = self.parse_time_column(data[i]) if i == 0 else str2float(data[i])
        return dict(zip(column_name, data))


def main():
    """
    简单测试入口：实例化 `CCXT`，拉取 K 线并打印前 5 条。

    说明：需要网络连接且系统已安装 `ccxt`；失败时会打印异常信息。
    """
    code = 'BTC/USDT'
    begin_date = '2021-01-01'
    api = CCXT(code, k_type=KL_TYPE.K_DAY, begin_date=begin_date)
    max_print = 5
    count = 0
    try:
        for item in api.get_kl_data():
            print(item)
            count += 1
            if count >= max_print:
                break
    except Exception as e:
        print('测试运行出错：', e)


if __name__ == '__main__':
    # import requests
    # print(requests.get("https://api.binance.com/api/v3/exchangeInfo", timeout=5).text)

    main()
