import os
from datetime import datetime

from Common.CEnum import AUTYPE, DATA_FIELD, KL_TYPE
from Common.ChanException import CChanException, ErrCode
from Common.CTime import CTime
from Common.func_util import kltype_lt_day, str2float
from KLine.KLine_Unit import CKLine_Unit

from .CommonStockAPI import CCommonStockApi


def GetColumnNameFromFieldList(fields: str):
    _dict = {
        "time": DATA_FIELD.FIELD_TIME,
        "open": DATA_FIELD.FIELD_OPEN,
        "high": DATA_FIELD.FIELD_HIGH,
        "low": DATA_FIELD.FIELD_LOW,
        "close": DATA_FIELD.FIELD_CLOSE,
        "volume": DATA_FIELD.FIELD_VOLUME,
    }
    return [_dict[x] for x in fields.split(",")]


class CSV_API(CCommonStockApi):

    def __init__(self, code, k_type=KL_TYPE.K_DAY,
                 begin_date=None, end_date=None,
                 autype=AUTYPE.QFQ):

        self.headers_exist = False

        super(CSV_API, self).__init__(code, k_type,
                                      begin_date, end_date, autype)

    def get_kl_data(self):

        fields = "time,open,high,low,close,volume"
        column_name = GetColumnNameFromFieldList(fields)

        cur_path = os.path.dirname(os.path.realpath(__file__))
        k_type = self.k_type.name[2:].lower()
        file_path = f"{cur_path}/../btc_data/{self.code}_{k_type}.csv"

        if not os.path.exists(file_path):
            raise CChanException(
                f"file not exist: {file_path}",
                ErrCode.SRC_DATA_NOT_FOUND
            )

        # begin_date 转成 CTime（如果有）
        begin_ctime = None
        end_ctime = None

        if self.begin_date:
            begin_ctime = self.parse_time_column(self.begin_date + " 00:00:00")

        if self.end_date:
            end_ctime = self.parse_time_column(self.end_date + " 23:59:59")

        with open(file_path, "r") as f:

            for line_number, line in enumerate(f):

                if self.headers_exist and line_number == 0:
                    continue

                raw = line.strip().split(",")

                # Binance 12列格式
                if len(raw) < 6:
                    raise CChanException(
                        f"file format error: {file_path}",
                        ErrCode.SRC_DATA_FORMAT_ERROR
                    )

                # 只取前6列
                row = raw[:6]

                # 时间解析
                ktime = self.parse_time_column(row[0])

                # 时间过滤（对象比较）
                if begin_ctime and ktime < begin_ctime:
                    continue
                if end_ctime and ktime > end_ctime:
                    continue

                yield CKLine_Unit(
                    self.create_item_dict(row, column_name),
                    autofix=True
                )

    def parse_time_column(self, inp):
        """
        支持：
        - 13位毫秒时间戳
        - yyyy-mm-dd
        - yyyy-mm-dd HH:MM:SS
        """

        # 毫秒时间戳（Binance CSV）
        if inp.isdigit() and len(inp) == 13:
            dt = datetime.utcfromtimestamp(int(inp) / 1000)
            return CTime(dt.year, dt.month, dt.day,
                         dt.hour, dt.minute,
                         auto=not kltype_lt_day(self.k_type))

        # yyyy-mm-dd
        if len(inp) == 10:
            year = int(inp[:4])
            month = int(inp[5:7])
            day = int(inp[8:10])
            hour = minute = 0

        # yyyy-mm-dd HH:MM:SS
        elif len(inp) == 19:
            year = int(inp[:4])
            month = int(inp[5:7])
            day = int(inp[8:10])
            hour = int(inp[11:13])
            minute = int(inp[14:16])

        else:
            raise Exception(f"unknown time column from CSV:{inp}")

        return CTime(year, month, day,
                     hour, minute,
                     auto=not kltype_lt_day(self.k_type))

    def create_item_dict(self, data, column_name):

        for i in range(len(data)):
            data[i] = self.parse_time_column(data[i]) \
                if i == 0 else str2float(data[i])

        return dict(zip(column_name, data))

    def SetBasciInfo(self):
        pass

    @classmethod
    def do_init(cls):
        pass

    @classmethod
    def do_close(cls):
        pass