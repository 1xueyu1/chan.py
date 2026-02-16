import os

from Common.CEnum import DATA_FIELD, KL_TYPE
from Common.ChanException import CChanException, ErrCode
from Common.CTime import CTime
from Common.func_util import str2float
from KLine.KLine_Unit import CKLine_Unit

from .CommonStockAPI import CCommonStockApi


# 将 CSV 行数据转换为字段字典
# - 对时间列使用 parse_time_column 解析为 CTime
# - 其他列使用 str2float 转为浮点数
def create_item_dict(data, column_name):
    for i in range(len(data)):
        data[i] = parse_time_column(data[i]) if column_name[i] == DATA_FIELD.FIELD_TIME else str2float(data[i])
    return dict(zip(column_name, data))


# 解析时间列的字符串，支持三种常见格式：
# - 长度为10: 'YYYY-MM-DD'，只包含日期，时分设为0
# - 长度为17: 'YYYYMMDDhhmmss...'（示例: 20210902113000000），按连写格式解析年月日时分
# - 长度为19: 'YYYY-MM-DD hh:mm:ss' 或类似格式，按常见带分隔符的日期时间解析
# 返回 CTime 对象
def parse_time_column(inp):
    # 20210902113000000
    # 2021-09-13
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
        # 未知格式时抛出异常，调用方可捕获并处理
        raise Exception(f"unknown time column from csv:{inp}")
    return CTime(year, month, day, hour, minute)


class CSV_API(CCommonStockApi):
    """
    CSV 数据源 API

    说明:
    - 从工程相对目录中读取以 `{code}_{k_type}.csv` 命名的文件
    - 默认第一行为表头, 若为数据请将 `headers_exist` 设为 False
    - 生成 `CKLine_Unit` 对象以供上层使用
    """

    def __init__(self, code, k_type=KL_TYPE.K_DAY, begin_date=None, end_date=None, autype=None):
        # 是否包含表头行，若 CSV 第一行即为数据则设为 False
        self.headers_exist = True
        # 默认列顺序，时间 + OHLC
        self.columns = [
            DATA_FIELD.FIELD_TIME,
            DATA_FIELD.FIELD_OPEN,
            DATA_FIELD.FIELD_HIGH,
            DATA_FIELD.FIELD_LOW,
            DATA_FIELD.FIELD_CLOSE,
            # 如需成交量等字段，可在此处添加
            # DATA_FIELD.FIELD_VOLUME,
            # DATA_FIELD.FIELD_TURNOVER,
            # DATA_FIELD.FIELD_TURNRATE,
        ]
        # 时间列在 columns 中的索引，用于时间范围过滤
        self.time_column_idx = self.columns.index(DATA_FIELD.FIELD_TIME)
        super(CSV_API, self).__init__(code, k_type, begin_date, end_date, autype)

    def get_kl_data(self):
        """
        逐行读取 CSV 文件并生成 `CKLine_Unit`。

        行为:
        - 根据 `self.k_type` 确定文件名后缀
        - 跳过表头（若 `self.headers_exist` 为 True）
        - 校验每行字段数量与 `self.columns` 一致
        - 根据 begin_date/end_date 进行过滤（字符串比较，需保证格式一致）
        - 使用 `create_item_dict` 将行数据转换为字典，再封装为 `CKLine_Unit`
        """
        cur_path = os.path.dirname(os.path.realpath(__file__))
        k_type = self.k_type.name[2:].lower()
        file_path = f"{cur_path}/../{self.code}_{k_type}.csv"
        if not os.path.exists(file_path):
            raise CChanException(f"file not exist: {file_path}", ErrCode.SRC_DATA_NOT_FOUND)

        for line_number, line in enumerate(open(file_path, 'r')):
            # 跳过表头
            if self.headers_exist and line_number == 0:
                continue
            data = line.strip("\n").split(",")
            # 字段数不匹配视为格式错误
            if len(data) != len(self.columns):
                raise CChanException(f"file format error: {file_path}", ErrCode.SRC_DATA_FORMAT_ERROR)
            # 按时间字符串进行简单过滤（假设时间字符串可直接比较）
            if self.begin_date is not None and data[self.time_column_idx] < self.begin_date:
                continue
            if self.end_date is not None and data[self.time_column_idx] > self.end_date:
                continue
            yield CKLine_Unit(create_item_dict(data, self.columns))

    def SetBasciInfo(self):
        # 占位方法：设置基础信息（按需实现）
        pass

    @classmethod
    def do_init(cls):
        # 类级别初始化（按需实现）
        pass

    @classmethod
    def do_close(cls):
        # 类级别清理（按需实现）
        pass
