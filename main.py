from Chan import CChan
from ChanConfig import CChanConfig
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE
from Plot.AnimatePlotDriver import CAnimateDriver
from Plot.PlotDriver import CPlotDriver


"""主程序入口（示例）：
切换为从 CCXT 获取 BTC/USDT 合约日线数据并绘图。

说明：保持原有绘图与配置逻辑，仅将数据源与代码改为合约。
"""


if __name__ == "__main__":
    # 交易对与时间范围（BTC 合约示例）
    code = "BTC/USDT"
    begin_time = "2026-01-01"
    end_time = "2026-02-04"

    # 数据来源：使用 ccxt 获取加密货币数据
    data_src = DATA_SRC.CCXT
    # 关注的 K 线粒度列表（此处仅日线）
    lv_list = [KL_TYPE.K_DAY, KL_TYPE.K_60M, KL_TYPE.K_15M]

    # 缠论 配置参数
    config = CChanConfig({
        "bi_strict": True,           # 严格成笔判断
        "trigger_step": False,       # 是否使用动图模式
        "skip_step": 0,
        "divergence_rate": float("inf"),
        "bsp2_follow_1": False,
        "bsp3_follow_1": False,
        "min_zs_cnt": 0,
        "bs1_peak": False,
        "macd_algo": "peak",
        "bs_type": '1,2,3a,1p,2s,3b',
        "print_warning": True,
        "zs_algo": "normal",
    })

    # 绘图开关：按需开启或关闭各类子图
    plot_config = {
        "plot_kline": True,
        "plot_kline_combine": True,
        "plot_bi": True,
        "plot_seg": True,
        "plot_eigen": False,
        "plot_zs": True,
        "plot_macd": False,
        "plot_mean": False,
        "plot_channel": False,
        "plot_bsp": True,
        "plot_extrainfo": False,
        "plot_demark": False,
        "plot_marker": False,
        "plot_rsi": False,
        "plot_kdj": False,
    }

    # 绘图参数：局部绘制参数和图像范围
    plot_para = {
        "seg": {},
        "bi": {},
        "figure": {"x_range": 200},
        "marker": {},
    }

    # 创建主对象并加载数据
    chan = CChan(
        code=code,
        begin_time=begin_time,
        end_time=end_time,
        data_src=data_src,
        lv_list=lv_list,
        config=config,
        autype=AUTYPE.NONE, # 合约数据无需复权
    )

    # 根据是否为动图模式选择静态或动画绘制
    if not config.trigger_step:
        plot_driver = CPlotDriver(chan, plot_config=plot_config, plot_para=plot_para)
        plot_driver.figure.show()          # 在交互环境显示
        plot_driver.save2img("./result/test.png")
    else:
        CAnimateDriver(chan, plot_config=plot_config, plot_para=plot_para)
