"""
示例：演示在多级别 `trigger_load` 场景下如何处理时间对齐。

思路：首次喂入最大级别时同时把所有次级别的历史 K 线一次性喂入，
之后仅按最大级别逐根喂入即可利用框架内置的时间对齐能力，避免手动对齐工作量。
"""

from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE
from DataAPI.BaoStockAPI import CBaoStock

if __name__ == "__main__":
    # 说明：本 demo 展示按日线为主级别、30 分钟为次级别的喂入策略
    code = "sz.000001"
    begin_time = "2023-06-01"
    end_time = None
    data_src = DATA_SRC.BAO_STOCK
    lv_list = [KL_TYPE.K_DAY, KL_TYPE.K_30M]

    config = CChanConfig({
        "trigger_step": True,
        "divergence_rate": 0.8,
        "min_zs_cnt": 1,
    })

    # 初始化 CChan（在首次触发时会把次级别历史 K 线一并喂入）
    chan = CChan(
        code=code,
        begin_time=begin_time,  # 已经没啥用了这一行
        end_time=end_time,  # 已经没啥用了这一行
        data_src=data_src,  # 已经没啥用了这一行
        lv_list=lv_list,
        config=config,
        autype=AUTYPE.QFQ,  # 已经没啥用了这一行
    )
    CBaoStock.do_init()
    data_src_day = CBaoStock(code, k_type=KL_TYPE.K_DAY, begin_date=begin_time, end_date=end_time, autype=AUTYPE.QFQ)
    data_src_30m = CBaoStock(code, k_type=KL_TYPE.K_30M, begin_date=begin_time, end_date=end_time, autype=AUTYPE.QFQ)
    kl_30m_all = list(data_src_30m.get_kl_data())

    for _idx, klu in enumerate(data_src_day.get_kl_data()):
        # 首次将所有次级别历史 K 线一并喂入，框架会自动截取并对齐对应区间
        if _idx == 0:
            chan.trigger_load({KL_TYPE.K_DAY: [klu], KL_TYPE.K_30M: kl_30m_all})
        else:
            # 之后仅按日线逐根喂入即可，时间对齐由框架处理
            chan.trigger_load({KL_TYPE.K_DAY: [klu]})

        if _idx == 4:  # demo 只检查前 4 根日线作为示例
            break
        # 打印当前各级别已加载的 K 线时间以便验证对齐结果
        print("当前所有日线:", [klu.time.to_str() for klu in chan[0].klu_iter()])
        print("当前所有30M K线:", [klu.time.to_str() for klu in chan[1].klu_iter()], "\n")

    CBaoStock.do_close()
