"""常用枚举与字段定义。

此模块定义了项目中使用的常量枚举和数据字段名称，注释简洁说明用途。
"""

from enum import Enum, auto
from typing import Literal


# 数据来源枚举（外部数据提供者）
class DATA_SRC(Enum):
    BAO_STOCK = auto()
    CCXT = auto()
    CSV = auto()
    PARQUET = auto()
    AKSHARE = auto()


# K 线类型（常用周期）
class KL_TYPE(Enum):
    K_1S = 1
    K_3S = 2
    K_5S = 3
    K_10S = 4
    K_15S = 5
    K_20S = 6
    K_30S = 7
    K_1M = 8
    K_3M = 9
    K_5M = 10
    K_10M = 11
    K_15M = 12
    K_30M = 13
    K_60M = 14
    K_DAY = 15
    K_WEEK = 16
    K_MON = 17
    K_QUARTER = 18
    K_YEAR = 19


# K 线方向/关系
class KLINE_DIR(Enum):
    UP = auto()
    DOWN = auto()
    COMBINE = auto()
    INCLUDED = auto()


# 波段类型（顶底等）
class FX_TYPE(Enum):
    BOTTOM = auto()
    TOP = auto()
    UNKNOWN = auto()


# 成笔方向（上升/下降）
class BI_DIR(Enum):
    UP = auto()
    DOWN = auto()


# 成笔类型及其标识
class BI_TYPE(Enum):
    UNKNOWN = auto()
    STRICT = auto()
    SUB_VALUE = auto()  # 次高低点成笔
    TIAOKONG_THRED = auto()
    DAHENG = auto()
    TUIBI = auto()
    UNSTRICT = auto()
    TIAOKONG_VALUE = auto()


# BSP 主类别的字面类型
BSP_MAIN_TYPE = Literal['1', '2', '3']


# BSP 类型（字符串编码形式），提供获取主类别的方法
class BSP_TYPE(Enum):
    T1 = '1'
    T1P = '1p'  # 类 1 买
    T2 = '2'
    T2S = '2s'  # 类 2 买
    T3A = '3a'  # 中枢在1类后面
    T3B = '3b'  # 中枢在1类前面

    def main_type(self) -> BSP_MAIN_TYPE:
        """返回 BSP 类型的主类别（第一个字符）。"""
        return self.value[0]  # type: ignore


# 复权类型：前复权、后复权、无复权
class AUTYPE(Enum):
    QFQ = auto()
    HFQ = auto()
    NONE = auto()


# 趋势计算方式
class TREND_TYPE(Enum):
    MEAN = "mean"
    MAX = "max"
    MIN = "min"


# 趋势线侧别（内侧/外侧）
class TREND_LINE_SIDE(Enum):
    INSIDE = auto()
    OUTSIDE = auto()


# 左侧分段方法
class LEFT_SEG_METHOD(Enum):
    ALL = auto()
    PEAK = auto()


# 分型检查方法类型
class FX_CHECK_METHOD(Enum):
    STRICT = auto()
    LOSS = auto()
    HALF = auto()
    TOTALLY = auto()


# 段类型：笔或段
class SEG_TYPE(Enum):
    BI = auto()
    SEG = auto()


# MACD 相关算法枚举
class MACD_ALGO(Enum):
    AREA = auto()
    PEAK = auto()
    FULL_AREA = auto()
    DIFF = auto()
    SLOPE = auto()
    AMP = auto()
    VOLUMN = auto()
    AMOUNT = auto()
    VOLUMN_AVG = auto()
    AMOUNT_AVG = auto()
    TURNRATE_AVG = auto()
    RSI = auto()


# 常用数据字段名集合，统一数据框架字段引用
class DATA_FIELD:
    FIELD_TIME = "time_key"
    FIELD_OPEN = "open"
    FIELD_HIGH = "high"
    FIELD_LOW = "low"
    FIELD_CLOSE = "close"
    FIELD_VOLUME = "volume"  # 成交量
    FIELD_TURNOVER = "turnover"  # 成交额
    FIELD_TURNRATE = "turnover_rate"  # 换手率


# 交易信息相关字段列表（便于取子集）
TRADE_INFO_LST = [DATA_FIELD.FIELD_VOLUME, DATA_FIELD.FIELD_TURNOVER, DATA_FIELD.FIELD_TURNRATE]
