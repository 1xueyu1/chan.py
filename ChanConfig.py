"""
ChanConfig.py

本文件包含频道（Chan）模块的配置类与代理配置接口。

主要内容：
- `CChanConfig`：频道运行时的配置容器，负责从外部字典加载配置并校验。
- `ConfigWithCheck`：对传入的配置字典做访问包装，记录已访问的键并在最后检查是否有多余的未识别键。
- 代理管理：`PROXY` 全局变量、`get_proxy()` 和 `set_proxy()` 用于运行时读取/更新代理配置（可持久化写回文件）。

注：本文件尽量不改变原有逻辑，仅补充中文注释以便阅读和维护。
"""

from typing import List

from Bi.BiConfig import CBiConfig
from BuySellPoint.BSPointConfig import CBSPointConfig
from Common.CEnum import TREND_TYPE
from Common.ChanException import CChanException, ErrCode
from Common.func_util import _parse_inf
from Math.BOLL import BollModel
from Math.Demark import CDemarkEngine
from Math.KDJ import KDJ
from Math.MACD import CMACD
from Math.RSI import RSI
from Math.TrendModel import CTrendModel
from Seg.SegConfig import CSegConfig
from ZS.ZSConfig import CZSConfig


class CChanConfig:
    def __init__(self, conf=None):
        """初始化频道配置。

        参数:
            conf: 可选的字典，用于覆盖默认配置。未识别的配置项会在最后触发异常。
        """
        # 如果未传入配置，使用空字典作为默认值
        if conf is None:
            conf = {}
        # 使用包装器以便逐项取值并在最后检查是否有多余项
        conf = ConfigWithCheck(conf)
        self.bi_conf = CBiConfig(
            # 分笔（笔）相关配置，委托给 CBiConfig 处理具体字段
            bi_algo=conf.get("bi_algo", "normal"),
            is_strict=conf.get("bi_strict", True),
            bi_fx_check=conf.get("bi_fx_check", "strict"),
            gap_as_kl=conf.get("gap_as_kl", False),
            bi_end_is_peak=conf.get('bi_end_is_peak', True),
            bi_allow_sub_peak=conf.get("bi_allow_sub_peak", True),
        )
        self.seg_conf = CSegConfig(
            # 段（seg）相关配置
            seg_algo=conf.get("seg_algo", "chan"),
            left_method=conf.get("left_seg_method", "peak"),
        )
        self.zs_conf = CZSConfig(
            # 中枢（zs）相关配置
            need_combine=conf.get("zs_combine", True),
            zs_combine_mode=conf.get("zs_combine_mode", "zs"),
            one_bi_zs=conf.get("one_bi_zs", False),
            zs_algo=conf.get("zs_algo", "normal"),
        )

        # 是否以 step（步）触发内部流程：
        # - 类型: bool
        # - 含义: 若为 True，则在满足触发条件时以更细粒度的 step 进行处理，主要用于调试或逐步回放逻辑。
        # - 默认: False
        self.trigger_step = conf.get("trigger_step", False)

        # 跳过的 step 数量（整数）：用于在回放或处理时跳过前 N 步。
        # - 类型: int
        # - 含义: 常用于恢复执行或定位问题时跳过前面的中间步骤。
        # - 默认: 0
        self.skip_step = conf.get("skip_step", 0)

        # 是否对 KL（K 线）数据进行严格校验
        # - 类型: bool
        # - 含义: 若为 True，会对输入的 K 线数据做时序/完整性检查，发现问题可能抛出或触发自动修正逻辑。
        # - 默认: True
        self.kl_data_check = conf.get("kl_data_check", True)

        # 在 K 线对齐校验中，允许的最大错位次数
        # - 类型: int
        # - 含义: 当多次出现时间戳或索引错位时，累计到该次数将被视为严重问题。
        # - 默认: 2
        self.max_kl_misalgin_cnt = conf.get("max_kl_misalgin_cnt", 2)

        # 在 K 线一致性校验中，允许的最大不一致次数
        # - 类型: int
        # - 含义: 比如数据缺失、重复或价格异常等不一致情况累计到该次数时触发错误处理。
        # - 默认: 5
        self.max_kl_inconsistent_cnt = conf.get("max_kl_inconsistent_cnt", 5)

        # 是否在遇到非法的子层级（sub level）时自动跳过
        # - 类型: bool
        # - 含义: 若为 True，则遇到无法解析或非法的子结构时程序尝试跳过以继续后续分析，
        #   否则会更严格地抛错或停止。
        # - 默认: False
        self.auto_skip_illegal_sub_lv = conf.get("auto_skip_illegal_sub_lv", False)

        # 控制是否打印警告信息（用于调试或监控）
        # - 类型: bool
        # - 含义: 若为 True，会在检测到可恢复的异常或潜在问题时输出警告日志。
        # - 默认: True
        self.print_warning = conf.get("print_warning", True)

        # 控制是否在打印错误时显示时间信息
        # - 类型: bool
        # - 含义: 包含时间戳有助于定位问题发生的具体时刻，便于回溯与对齐。
        # - 默认: True
        self.print_err_time = conf.get("print_err_time", True)

        # 平滑/均值相关的度量周期列表
        # - 类型: List[int]
        # - 含义: 例如 [5, 10] 表示要计算 5 日/10 日均值并把对应的 `CTrendModel(TREND_TYPE.MEAN, T)` 加入指标集合。
        # - 默认: []（空列表，不计算额外均值指标）
        self.mean_metrics: List[int] = conf.get("mean_metrics", [])

        # 趋势相关的周期列表，用于生成 MAX/MIN 类型的趋势模型
        # - 类型: List[int]
        # - 含义: 每个周期会生成 `CTrendModel(TREND_TYPE.MAX, T)` 和 `CTrendModel(TREND_TYPE.MIN, T)`。
        # - 默认: []
        self.trend_metrics: List[int] = conf.get("trend_metrics", [])

        # MACD 指标的参数配置（字典）
        # - keys: 'fast', 'slow', 'signal'
        # - 含义: 分别对应 MACD 计算的快线、慢线与信号线周期
        # - 默认: {'fast':12, 'slow':26, 'signal':9}
        self.macd_config = conf.get("macd", {"fast": 12, "slow": 26, "signal": 9})

        # 是否启用 DeMark 指标计算（布尔）
        self.cal_demark = conf.get("cal_demark", False)
        # 是否启用 RSI 指标计算（布尔）
        self.cal_rsi = conf.get("cal_rsi", False)
        # 是否启用 KDJ 指标计算（布尔）
        self.cal_kdj = conf.get("cal_kdj", False)

        # RSI 与 KDJ 的周期长度
        # - RSI 常见为 14
        # - KDJ 常见为 9
        self.rsi_cycle = conf.get("rsi_cycle", 14)
        self.kdj_cycle = conf.get("kdj_cycle", 9)

        # DeMark 指标的详细参数（通过字典传入），包含 setup/countdown 的偏差与长度等
        # 可用键示例: 'demark_len','setup_bias','countdown_bias','max_countdown','tiaokong_st',
        #           'setup_cmp2close','countdown_cmp2close'
        # 默认值按常用配置给出
        self.demark_config = conf.get("demark", {
            'demark_len': 9,
            'setup_bias': 4,
            'countdown_bias': 2,
            'max_countdown': 13,
            'tiaokong_st': True,
            'setup_cmp2close': True,
            'countdown_cmp2close': True,
        })

        # BOLL 指标的窗口期（整数），用于构建 BollModel
        # - 默认: 20
        self.boll_n = conf.get("boll_n", 20)

        self.set_bsp_config(conf)

        # 所有配置项读取完成后，检查是否存在未被识别或未被消费的配置项
        conf.check()

    def GetMetricModel(self):
        """构建并返回需要计算的指标模型列表。

        返回值是一个模型实例列表，包含 MACD、均值/趋势模型、BOLL、以及可选的 DeMark/RSI/KDJ。
        """
        res: List[CMACD | CTrendModel | BollModel | CDemarkEngine | RSI | KDJ] = [
            # 先添加 MACD 指标模型，参数来源于 self.macd_config
            CMACD(
                fastperiod=self.macd_config['fast'],
                slowperiod=self.macd_config['slow'],
                signalperiod=self.macd_config['signal'],
            )
        ]
        res.extend(CTrendModel(TREND_TYPE.MEAN, mean_T) for mean_T in self.mean_metrics)

        for trend_T in self.trend_metrics:
            res.append(CTrendModel(TREND_TYPE.MAX, trend_T))
            res.append(CTrendModel(TREND_TYPE.MIN, trend_T))
        res.append(BollModel(self.boll_n))
        if self.cal_demark:
            res.append(CDemarkEngine(
                demark_len=self.demark_config['demark_len'],
                setup_bias=self.demark_config['setup_bias'],
                countdown_bias=self.demark_config['countdown_bias'],
                max_countdown=self.demark_config['max_countdown'],
                tiaokong_st=self.demark_config['tiaokong_st'],
                setup_cmp2close=self.demark_config['setup_cmp2close'],
                countdown_cmp2close=self.demark_config['countdown_cmp2close'],
            ))
        if self.cal_rsi:
            res.append(RSI(self.rsi_cycle))
        if self.cal_kdj:
            res.append(KDJ(self.kdj_cycle))
        return res

    def set_bsp_config(self, conf):
        """设置买卖点（Buy/Sell Point）配置。

        逻辑说明：
        - 首先定义一套默认参数 `para_dict`，然后用传入的 `conf` 覆盖默认值得到 `args`。
        - 使用 `CBSPointConfig(**args)` 构建 `bs_point_conf`（用于一般的买卖点判定）。
        - `seg_bs_point_conf` 是用于以段为单位判定的配置，基于相同默认参数创建并针对段分析做少量调整。
        - 最后会遍历 `conf.items()`，根据键的后缀动态设置不同子配置（如 '-buy'、'-sell'、'-seg' 等）。

        注意：遍历时使用了 `exec()` 动态执行 `set` 调用，这是为了方便根据字符串配置映射到不同的子配置上，
        所以传入的配置键必须受信任且在 `args` 或带后缀的形式中被允许，否则会抛异常。
        """
        para_dict = {
            # 默认参数集合，详见业务逻辑使用含义
            "divergence_rate": float("inf"),
            "min_zs_cnt": 1,
            "bsp1_only_multibi_zs": True,
            "max_bs2_rate": 0.9999,
            "macd_algo": "peak",
            "bs1_peak": True,
            "bs_type": "1,1p,2,2s,3a,3b",
            "bsp2_follow_1": True,
            "bsp3_follow_1": True,
            "bsp3_peak": False,
            "bsp2s_follow_2": False,
            "max_bsp2s_lv": None,
            "strict_bsp3": False,
            "bsp3a_max_zs_cnt": 1,
        }
        # 用用户配置覆盖默认值，生成最终参数字典
        args = {para: conf.get(para, default_value) for para, default_value in para_dict.items()}
        # 常规模式下的买卖点配置
        self.bs_point_conf = CBSPointConfig(**args)

        # 段级别的买卖点配置（默认沿用 args，但需要对 macd 算法和 bsp1_only_multibi_zs 做调整）
        self.seg_bs_point_conf = CBSPointConfig(**args)
        self.seg_bs_point_conf.b_conf.set("macd_algo", "slope")
        self.seg_bs_point_conf.s_conf.set("macd_algo", "slope")
        self.seg_bs_point_conf.b_conf.set("bsp1_only_multibi_zs", False)
        self.seg_bs_point_conf.s_conf.set("bsp1_only_multibi_zs", False)

        # 遍历剩余的配置项，根据键名后缀分配到不同子配置
        for k, v in conf.items():
            # 若值为字符串，给它加上引号以便后续 exec 中被视为字符串文本
            if isinstance(v, str):
                v = f'"{v}"'
            # 支持将字符串 "inf" 等解析为 float('inf') 等特殊值
            v = _parse_inf(v)
            # 支持的后缀：-buy、-sell、-segbuy、-segsell、-seg
            if k.endswith("-buy"):
                prop = k.replace("-buy", "")
                exec(f"self.bs_point_conf.b_conf.set('{prop}', {v})")
            elif k.endswith("-sell"):
                prop = k.replace("-sell", "")
                exec(f"self.bs_point_conf.s_conf.set('{prop}', {v})")
            elif k.endswith("-segbuy"):
                prop = k.replace("-segbuy", "")
                exec(f"self.seg_bs_point_conf.b_conf.set('{prop}', {v})")
            elif k.endswith("-segsell"):
                prop = k.replace("-segsell", "")
                exec(f"self.seg_bs_point_conf.s_conf.set('{prop}', {v})")
            elif k.endswith("-seg"):
                prop = k.replace("-seg", "")
                exec(f"self.seg_bs_point_conf.b_conf.set('{prop}', {v})")
                exec(f"self.seg_bs_point_conf.s_conf.set('{prop}', {v})")
            elif k in args:
                # 如果键在默认参数集中，设置到常规模式的 b_conf 和 s_conf
                exec(f"self.bs_point_conf.b_conf.set({k}, {v})")
                exec(f"self.bs_point_conf.s_conf.set({k}, {v})")
            else:
                # 遇到未知参数，抛出配置异常，提示调用者修正
                raise CChanException(f"unknown para = {k}", ErrCode.PARA_ERROR)

        # 最后对各子配置进行类型解析/转换（例如把字符串目标类型解析为内部枚举等）
        self.bs_point_conf.b_conf.parse_target_type()
        self.bs_point_conf.s_conf.parse_target_type()
        self.seg_bs_point_conf.b_conf.parse_target_type()
        self.seg_bs_point_conf.s_conf.parse_target_type()


class ConfigWithCheck:
    def __init__(self, conf):
        """包装一个 dict，用于按需取出键值并在最后检查未识别的配置项。

        设计目的：避免调用者忘记处理某些配置项，减少配置拼写错误导致的静默失效。
        """
        self.conf = conf

    def get(self, k, default_value=None):
        # 读取后立即从原始字典中删除该键，表示该键已被消费
        res = self.conf.get(k, default_value)
        if k in self.conf:
            del self.conf[k]
        return res

    def items(self):
        # 迭代剩余的项，迭代过程中记录已访问键并在结束后删除它们
        visit_keys = set()
        for k, v in self.conf.items():
            yield k, v
            visit_keys.add(k)
        for k in visit_keys:
            del self.conf[k]

    def check(self):
        # 若仍有未消费的键，说明存在未识别的配置项，抛出异常提醒调用者
        if len(self.conf) > 0:
            invalid_key_lst = ",".join(list(self.conf.keys()))
            raise CChanException(f"invalid CChanConfig: {invalid_key_lst}", ErrCode.PARA_ERROR)


# 代理配置，供程序运行时读取/更新
# 默认为 None，首次运行时可由代码写回具体值
PROXY = {'http': 'http://127.0.0.1:7890', 'https': 'http://127.0.0.1:7890'}


def get_proxy():
    """返回当前内存中的代理配置字典，格式 {'http': url, 'https': url}。

    注意：返回的是全局 `PROXY` 对象的引用，调用者若需要本地修改请谨慎。
    """
    return PROXY


def set_proxy(http: str, https: str = None, persist: bool = False):
    """
    更新内存中的代理配置，并可选择持久化到 `ChanConfig.py` 文件中。

    参数:
        http: HTTP 代理 URL（例如 'http://127.0.0.1:7890' 或 'socks5h://127.0.0.1:7891'）。
        https: 可选的 HTTPS 代理 URL，若为 None 则使用与 http 相同的地址。
        persist: 若为 True，则把代理配置写回到 `ChanConfig.py` 源文件顶部（覆盖或追加）。
    """
    # 更新内存中的代理配置
    PROXY['http'] = http
    PROXY['https'] = https or http
    # 如不要求持久化，则仅修改内存并返回
    if not persist:
        return

    # 持久化：将代理配置写回当前源文件，替换已有的 PROXY 定义行或追加到文件末尾
    # 注意：该操作会修改源码文件，请确保程序有写权限且调用方信任此行为
    import io
    import os

    filepath = os.path.realpath(__file__)
    with io.open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    found = False
    for idx, line in enumerate(lines):
        # 寻找以 PROXY = 开头的行并替换（忽略行首空白）
        if line.strip().startswith('PROXY ='):
            lines[idx] = f"PROXY = {{'http': {repr(PROXY['http'])}, 'https': {repr(PROXY['https'])}}}\n"
            found = True
            break

    if not found:
        # 未找到则在文件末尾追加新的 PROXY 定义
        lines.append("\n")
        lines.append(f"PROXY = {{'http': {repr(PROXY['http'])}, 'https': {repr(PROXY['https'])}}}\n")

    with io.open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(lines)
