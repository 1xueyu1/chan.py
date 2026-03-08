"""
Plot/plotly/PlotDriver.py

基于 Plotly 的缠论交互式图表绘制引擎。
功能与 mpl（matplotlib）版本完全对齐，同时充分利用 Plotly 的交互特性提升量化分析体验。

支持绘制的元素：
    K线 ──── 原始K线（蜡烛图/折线）、合并K线
    笔 ──── 确定笔（实线）、不确定笔（虚线）、笔编号、端点值
    线段 ── 线段、线段的线段、趋势线
    特征 ── 笔特征序列、段特征序列
    中枢 ── 笔中枢、段中枢、子中枢、中枢高低点标注
    指标 ── MACD（独立子图）、均线、通道线、布林带
    辅助 ── RSI（副Y轴叠加）、KDJ（副Y轴叠加）、Demark 序列
    标注 ── 买卖点（带箭头）、段买卖点、自定义 Marker

Plotly 特有增强：
    - 十字光标（spike lines）辅助精确定位
    - 鼠标悬停显示 OHLCV 详情
    - 拖拽缩放 & 双击重置
    - 导出为交互式 HTML 或静态图片

用法示例::

    from Plot import get_plot_driver

    CPlotDriver = get_plot_driver("plotly")
    driver = CPlotDriver(chan, plot_config=plot_config, plot_para=plot_para)
    driver.show()                   # 浏览器中打开交互式图表
    driver.save2img("result.html")  # 保存为 HTML（或 .png/.jpg 需安装 kaleido）
"""

import inspect
from typing import Dict, List, Literal, Optional, Tuple, Union

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from Chan import CChan
from Common.CEnum import BI_DIR, FX_TYPE, KL_TYPE, KLINE_DIR, TREND_TYPE
from Common.ChanException import CChanException, ErrCode
from Common.CTime import CTime
from Math.Demark import CDemarkEngine

from ..core.PlotMeta import CChanPlotMeta, CZS_meta


# ════════════════════════════════════════════════════════════════
#  一、配置解析
#     与 mpl 版保持完全一致，支持 dict/str/list 三种传参方式
# ════════════════════════════════════════════════════════════════

def reformat_plot_config(plot_config: Dict[str, bool]) -> Dict[str, bool]:
    """兼容不填写 ``plot_`` 前缀的情况，例如 ``kline`` → ``plot_kline``。"""
    def _fmt(s: str) -> str:
        return s if s.startswith("plot_") else f"plot_{s}"
    return {_fmt(k): v for k, v in plot_config.items()}


def parse_single_lv_plot_config(plot_config: Union[str, dict, list]) -> Dict[str, bool]:
    """
    解析单一级别绘图配置，返回 ``{plot_xxx: True/False}`` 字典。

    支持传入：
        - dict: ``{"kline": True, "bi": True}``
        - str:  ``"kline, bi, seg"``
        - list: ``["kline", "bi", "seg"]``
    """
    if isinstance(plot_config, dict):
        return reformat_plot_config(plot_config)
    elif isinstance(plot_config, str):
        return reformat_plot_config({k.strip().lower(): True for k in plot_config.split(",")})
    elif isinstance(plot_config, list):
        return reformat_plot_config({k.strip().lower(): True for k in plot_config})
    else:
        raise CChanException("plot_config only support list/str/dict", ErrCode.PLOT_ERR)


def parse_plot_config(
    plot_config: Union[str, dict, list],
    lv_list: List[KL_TYPE],
) -> Dict[KL_TYPE, Dict[str, bool]]:
    """
    解析多级别绘图配置。

    支持：
        - 所有级别共用同一配置（key 为 str）
        - 不同级别各自配置（key 为 KL_TYPE）
    """
    if isinstance(plot_config, dict):
        if all(isinstance(k, str) for k in plot_config.keys()):
            # 所有级别共用
            return {lv: parse_single_lv_plot_config(plot_config) for lv in lv_list}
        elif all(isinstance(k, KL_TYPE) for k in plot_config.keys()):
            # 按级别分别配置
            for lv in lv_list:
                assert lv in plot_config, f"plot_config 缺少级别 {lv}"
            return {lv: parse_single_lv_plot_config(plot_config[lv]) for lv in lv_list}
        else:
            raise CChanException("plot_config dict 的 key 必须全部为 str 或全部为 KL_TYPE", ErrCode.PLOT_ERR)
    return {lv: parse_single_lv_plot_config(plot_config) for lv in lv_list}


def cal_x_limit(meta: CChanPlotMeta, x_range: int) -> List[int]:
    """根据 x_range 计算可见区域的 [起始索引, 结束索引]。"""
    total = meta.klu_len
    if x_range and total > x_range:
        return [total - x_range, total - 1]
    return [0, total - 1]


def cal_y_range(meta: CChanPlotMeta, x_begin: int) -> Tuple[float, float]:
    """扫描可见区域内合并K线的最高/最低价，返回 (y_min, y_max)。"""
    y_min = float("inf")
    y_max = float("-inf")
    for klc in meta.klc_list:
        if klc.klu_list[-1].idx < x_begin:
            continue
        y_max = max(y_max, klc.high)
        y_min = min(y_min, klc.low)
    return y_min, y_max


def GetPlotMeta(chan: CChan, figure_config: dict) -> List[CChanPlotMeta]:
    """构建每个级别的绘图元数据；若 ``only_top_lv=True`` 则仅保留最高级别。"""
    metas = [CChanPlotMeta(chan[kl]) for kl in chan.lv_list]
    if figure_config.get("only_top_lv", False):
        metas = [metas[0]]
    return metas


# ════════════════════════════════════════════════════════════════
#  二、颜色工具
#     matplotlib 单字母缩写 → CSS 颜色名
# ════════════════════════════════════════════════════════════════

_COLOR_MAP: Dict[str, str] = {
    "r": "red",
    "g": "green",
    "b": "blue",
    "k": "black",
    "w": "white",
    "y": "yellow",
    "c": "cyan",
    "m": "magenta",
    "orange": "orange",
    "brown": "brown",
    "purple": "purple",
    "pink": "pink",
    "gray": "gray",
    "grey": "grey",
}


def _to_css_color(c: str) -> str:
    """将 matplotlib 缩写颜色（如 ``'r'``）转为 CSS 颜色名（如 ``'red'``）。

    已经是 ``#hex`` 或 ``rgb(...)`` 格式的直接返回。
    """
    if c.startswith("#") or c.startswith("rgb"):
        return c
    return _COLOR_MAP.get(c, c)


def _mpl_linestyle_to_plotly(ls: str) -> str:
    """将 matplotlib 线型名称转换为 plotly dash 值。"""
    mapping = {
        "solid": "solid",
        "dashed": "dash",
        "dashdot": "dashdot",
        "dotted": "dot",
    }
    return mapping.get(ls, "solid")


# ════════════════════════════════════════════════════════════════
#  三、子图布局
# ════════════════════════════════════════════════════════════════

def create_figure(
    plot_macd: Dict[KL_TYPE, bool],
    figure_config: dict,
    lv_lst: List[KL_TYPE],
    code: str = "",
) -> Tuple[go.Figure, Dict[KL_TYPE, List[int]]]:
    """
    根据级别列表和是否绘制 MACD 来创建 Plotly 子图布局。

    返回:
        fig       — Plotly Figure 对象
        rows_dict — ``{KL_TYPE: [主图row, MACD_row(可选)]}``，row 从 1 开始

    所有主图行均启用 ``secondary_y``，用于叠加 RSI / KDJ 等副轴指标。
    MACD 行不需要 secondary_y。
    """
    default_h = 600   # 每级别默认高度（像素）
    macd_h_ratio = figure_config.get("macd_h", 0.3)

    row_heights: List[float] = []
    subplot_titles: List[str] = []
    specs: List[List[dict]] = []  # make_subplots 的 specs 参数

    for lv in lv_lst:
        lv_label = lv.name.split("K_")[1] if "K_" in lv.name else lv.name
        title = f"{code} / {lv_label}" if code else lv_label
        # 主图行 —— 启用 secondary_y（用于 RSI / KDJ）
        row_heights.append(1.0)
        subplot_titles.append(title)
        specs.append([{"secondary_y": True}])
        # MACD 行 —— 不需要 secondary_y
        if plot_macd.get(lv, False):
            row_heights.append(macd_h_ratio)
            subplot_titles.append("MACD")
            specs.append([{"secondary_y": False}])

    total_rows = len(row_heights)
    fig = make_subplots(
        rows=total_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
        row_heights=row_heights,
        subplot_titles=subplot_titles,
        specs=specs,
    )

    # 构建 rows_dict：记录每个级别对应的行号
    rows_dict: Dict[KL_TYPE, List[int]] = {}
    cur_row = 1
    for lv in lv_lst:
        if plot_macd.get(lv, False):
            rows_dict[lv] = [cur_row, cur_row + 1]
            cur_row += 2
        else:
            rows_dict[lv] = [cur_row]
            cur_row += 1

    # 计算总高度
    h = figure_config.get("h", default_h * len(lv_lst))
    w = figure_config.get("w", None)
    layout_kw: dict = {"height": h}
    if w:
        layout_kw["width"] = w
    fig.update_layout(**layout_kw)

    return fig, rows_dict


# ════════════════════════════════════════════════════════════════
#  四、CPlotDriver — 缠论 Plotly 绘图驱动
# ════════════════════════════════════════════════════════════════

class CPlotDriver:
    """
    基于 Plotly 的缠论图表绘制驱动。

    初始化时即完成所有绘制操作，之后通过 :meth:`show` 或 :meth:`save2img` 输出。

    参数:
        chan        — ``CChan`` 缠论计算结果
        plot_config — 绘图开关，支持 dict / str / list
        plot_para   — 绘图参数，按元素类型分组，参见各 ``draw_*`` 方法签名
    """

    def __init__(self, chan: CChan, plot_config: Union[str, dict, list] = "", plot_para=None):
        if plot_para is None:
            plot_para = {}
        figure_config: dict = plot_para.get("figure", {})

        # ---------- 解析配置 ----------
        plot_config = parse_plot_config(plot_config, chan.lv_list)
        plot_metas = GetPlotMeta(chan, figure_config)
        self.lv_lst = chan.lv_list[: len(plot_metas)]
        self.code = chan.code  # 标的代码，用于标题

        x_range = self._calc_real_x_range(figure_config, plot_metas[0])
        plot_macd: Dict[KL_TYPE, bool] = {
            kl: conf.get("plot_macd", False) for kl, conf in plot_config.items()
        }

        # ---------- 创建子图 ----------
        self.figure, rows_dict = create_figure(
            plot_macd, figure_config, self.lv_lst, code=self.code,
        )

        # 多级别子图联动参数
        sseg_begin = 0
        slv_seg_cnt = plot_para.get("seg", {}).get("sub_lv_cnt", None)
        sbi_begin = 0
        slv_bi_cnt = plot_para.get("bi", {}).get("sub_lv_cnt", None)
        srange_begin = 0
        assert slv_seg_cnt is None or slv_bi_cnt is None, \
            "seg.sub_lv_cnt 与 bi.sub_lv_cnt 不能同时设置"

        # ---------- 逐级别绘制 ----------
        for meta, lv in zip(plot_metas, self.lv_lst):
            rows = rows_dict[lv]
            main_row = rows[0]
            macd_row = rows[1] if len(rows) > 1 else None

            # 计算可见范围
            x_limits = cal_x_limit(meta, x_range)
            if lv != self.lv_lst[0]:
                if sseg_begin != 0 or sbi_begin != 0:
                    x_limits[0] = max(sseg_begin, sbi_begin)
                elif srange_begin != 0:
                    x_limits[0] = srange_begin

            self.y_min, self.y_max = cal_y_range(meta, x_limits[0])
            self._x_limits = x_limits

            # 设置 x 轴刻度（日期标签）
            x_tick_num = figure_config.get("x_tick_num", 10)
            self._set_x_tick(meta, main_row, x_limits, x_tick_num)
            if macd_row:
                self._set_x_tick(meta, macd_row, x_limits, x_tick_num)

            # ── 绘制全部元素 ──
            self._draw_all_elements(plot_config[lv], meta, main_row, lv, plot_para, macd_row, x_limits)

            # 计算子级别起始偏移
            if lv != self.lv_lst[-1]:
                if slv_seg_cnt is not None:
                    sseg_begin = meta.sub_last_kseg_start_idx(slv_seg_cnt)
                if slv_bi_cnt is not None:
                    sbi_begin = meta.sub_last_kbi_start_idx(slv_bi_cnt)
                if x_range != 0:
                    srange_begin = meta.sub_range_start_idx(x_range)

            # 设置主图 y 轴范围（primary y）
            self.figure.update_yaxes(
                range=[self.y_min, self.y_max],
                row=main_row, col=1, secondary_y=False,
            )

        # ---------- 全局布局 ----------
        self.figure.update_layout(
            template="plotly_white",
            xaxis_rangeslider_visible=False,
            showlegend=False,
            margin=dict(l=50, r=50, t=50, b=50),
            # 鼠标交互模式：十字光标
            hovermode="x",
            dragmode="zoom",
        )

        # 将 make_subplots 生成的子标题设为红色加粗
        for ann in self.figure.layout.annotations:
            if hasattr(ann, "text") and ann.text:
                ann.update(
                    font=dict(size=14, color="red"),
                )

        # 十字光标 / spike lines（所有轴）
        self.figure.update_xaxes(
            showspikes=True, spikemode="across",
            spikethickness=0.5, spikecolor="grey", spikesnap="cursor",
        )
        self.figure.update_yaxes(
            showspikes=True, spikemode="across",
            spikethickness=0.5, spikecolor="grey", spikesnap="cursor",
        )

        # 网格配置
        grid_cfg = figure_config.get("grid", "xy")
        if grid_cfg:
            self._set_grid(grid_cfg)

    # ────────────────────────────────────────────────────
    #  轴 & 网格 & X 范围计算
    # ────────────────────────────────────────────────────

    def _set_x_tick(self, meta: CChanPlotMeta, row: int, x_limits: List[int], x_tick_num: int):
        """设置 x 轴刻度标签（日期文字），并限定可见范围。"""
        assert x_tick_num > 1
        step = max(1, int((x_limits[1] - x_limits[0]) / float(x_tick_num)))
        tick_vals = list(range(x_limits[0], x_limits[1], step))
        tick_text = [meta.datetick[i] if i < len(meta.datetick) else "" for i in tick_vals]

        self.figure.update_xaxes(
            tickmode="array",
            tickvals=tick_vals,
            ticktext=tick_text,
            tickangle=-20,
            range=[x_limits[0], x_limits[1] + 1],
            row=row, col=1,
        )

    def _set_grid(self, config: str):
        """设置网格显示方式：``"xy"`` / ``"x"`` / ``"y"`` / ``None``。"""
        if config is None:
            return
        if config not in ("xy", "x", "y"):
            raise CChanException(
                f"不支持的 grid 配置: '{config}'，可选: 'xy'/'x'/'y'/None",
                ErrCode.PLOT_ERR,
            )
        show_x = config in ("xy", "x")
        show_y = config in ("xy", "y")
        self.figure.update_xaxes(showgrid=show_x)
        self.figure.update_yaxes(showgrid=show_y)

    def _calc_real_x_range(self, figure_config: dict, meta: CChanPlotMeta) -> int:
        """
        从 figure_config 中计算实际需要显示的 K 线根数。

        互斥的四种指定方式（只能选一种）：
            x_range      — 直接指定根数
            x_bi_cnt     — 最后 N 笔覆盖的根数
            x_seg_cnt    — 最后 N 段覆盖的根数
            x_begin_date — 指定起始日期
        """
        x_range = figure_config.get("x_range", 0)
        bi_cnt = figure_config.get("x_bi_cnt", 0)
        seg_cnt = figure_config.get("x_seg_cnt", 0)
        x_begin_date = figure_config.get("x_begin_date", 0)

        non_zero = sum(1 for v in [x_range, bi_cnt, seg_cnt, x_begin_date] if v)
        assert non_zero <= 1, "x_range / x_bi_cnt / x_seg_cnt / x_begin_date 只能同时指定一项"

        X_LEN = meta.klu_len

        if x_range != 0:
            return x_range
        if bi_cnt != 0:
            return (X_LEN - meta.bi_list[-bi_cnt].begin_x) if len(meta.bi_list) >= bi_cnt else 0
        if seg_cnt != 0:
            return (X_LEN - meta.seg_list[-seg_cnt].begin_x) if len(meta.seg_list) >= seg_cnt else 0
        if x_begin_date != 0:
            cnt = 0
            for dt in meta.datetick[::-1]:
                if dt >= x_begin_date:
                    cnt += 1
                else:
                    break
            return cnt
        return 0  # 默认显示全部

    # ────────────────────────────────────────────────────
    #  元素分发
    # ────────────────────────────────────────────────────

    def _draw_all_elements(
        self,
        plot_config: Dict[str, bool],
        meta: CChanPlotMeta,
        row: int,
        lv: KL_TYPE,
        plot_para: dict,
        macd_row: Optional[int],
        x_limits: List[int],
    ):
        """根据 plot_config 开关依次调用各绘图方法。"""

        # K线
        if plot_config.get("plot_kline", False):
            self.draw_klu(meta, row, **plot_para.get("kl", {}))
        if plot_config.get("plot_kline_combine", False):
            self.draw_klc(meta, row, **plot_para.get("klc", {}))

        # 笔 & 线段
        if plot_config.get("plot_bi", False):
            self.draw_bi(meta, row, lv, **plot_para.get("bi", {}))
        if plot_config.get("plot_seg", False):
            self.draw_seg(meta, row, lv, **plot_para.get("seg", {}))
        if plot_config.get("plot_segseg", False):
            self.draw_segseg(meta, row, **plot_para.get("segseg", {}))

        # 特征序列
        if plot_config.get("plot_eigen", False):
            self.draw_eigen(meta, row, **plot_para.get("eigen", {}))
        if plot_config.get("plot_segeigen", False):
            self.draw_segeigen(meta, row, **plot_para.get("segeigen", {}))

        # 中枢
        if plot_config.get("plot_zs", False):
            self.draw_zs(meta, row, **plot_para.get("zs", {}))
        if plot_config.get("plot_segzs", False):
            self.draw_segzs(meta, row, **plot_para.get("segzs", {}))

        # MACD（独立子图）
        if plot_config.get("plot_macd", False):
            assert macd_row is not None, "plot_macd=True 但未分配 MACD 子图行"
            self.draw_macd(meta, macd_row, x_limits, **plot_para.get("macd", {}))

        # 均线 / 通道 / 布林
        if plot_config.get("plot_mean", False):
            self.draw_mean(meta, row, **plot_para.get("mean", {}))
        if plot_config.get("plot_channel", False):
            self.draw_channel(meta, row, **plot_para.get("channel", {}))
        if plot_config.get("plot_boll", False):
            self.draw_boll(meta, row, **plot_para.get("boll", {}))

        # 买卖点
        if plot_config.get("plot_bsp", False):
            self.draw_bs_point(meta, row, **plot_para.get("bsp", {}))
        if plot_config.get("plot_segbsp", False):
            self.draw_seg_bs_point(meta, row, **plot_para.get("seg_bsp", {}))

        # Demark / Marker
        if plot_config.get("plot_demark", False):
            self.draw_demark(meta, row, **plot_para.get("demark", {}))
        if plot_config.get("plot_marker", False):
            self.draw_marker(meta, row, **plot_para.get("marker", {"markers": {}}))

        # RSI / KDJ —— 叠加在主图的副 Y 轴上（secondary_y）
        if plot_config.get("plot_rsi", False):
            self.draw_rsi(meta, row, **plot_para.get("rsi", {}))
        if plot_config.get("plot_kdj", False):
            self.draw_kdj(meta, row, **plot_para.get("kdj", {}))

    # ────────────────────────────────────────────────────
    #  输出方法
    # ────────────────────────────────────────────────────

    def show(self):
        """在默认浏览器中打开交互式图表。"""
        self.figure.show()

    def _repr_html_(self):
        """Jupyter Notebook 内联显示支持（自动调用）。"""
        return self.figure.to_html(
            full_html=False, include_plotlyjs="cdn",
        )

    def save2img(self, path: str):
        """
        保存图表到文件。

        - ``.html`` → 交互式 HTML（推荐，无额外依赖）
        - ``.png`` / ``.jpg`` / ``.svg`` / ``.pdf`` → 静态图片（需安装 ``kaleido``）
        """
        if path.endswith(".html"):
            self.figure.write_html(path)
        else:
            try:
                self.figure.write_image(path)
            except ValueError as e:
                raise CChanException(
                    f"保存静态图片失败，请安装 kaleido: pip install kaleido\n原始错误: {e}",
                    ErrCode.PLOT_ERR,
                ) from e

    def ShowDrawFuncHelper(self):
        """打印所有 ``draw_*`` 方法的参数签名与默认值，方便查阅。"""
        for name in sorted(dir(self)):
            if not name.startswith("draw_"):
                continue
            show_func_helper(getattr(self, name))

    # ════════════════════════════════════════════════════
    #  五、K线绘制
    # ════════════════════════════════════════════════════

    def draw_klu(
        self, meta: CChanPlotMeta, row: int,
        width=0.4, rugd=True, plot_mode="kl",
    ):
        """
        绘制原始K线单元。

        参数:
            width     — 蜡烛宽度
            rugd      — ``True`` = 红涨绿跌（中国市场习惯）
            plot_mode — ``"kl"`` 蜡烛图 | ``"close"`` / ``"open"`` / ``"high"`` / ``"low"`` 折线
        """
        up_color = "red" if rugd else "green"
        down_color = "green" if rugd else "red"
        x_begin = self._x_limits[0]

        if plot_mode == "kl":
            # ── 蜡烛图模式 ──
            x_data: List[int] = []
            open_d: List[float] = []
            high_d: List[float] = []
            low_d: List[float] = []
            close_d: List[float] = []
            hover_text: List[str] = []

            for kl in meta.klu_iter():
                if kl.idx < x_begin:
                    continue
                x_data.append(kl.idx)
                open_d.append(kl.open)
                high_d.append(kl.high)
                low_d.append(kl.low)
                close_d.append(kl.close)
                # 悬浮提示：日期 + OHLCV
                vol_str = (
                    f"<br>量: {kl.vol}"
                    if hasattr(kl, 'vol') and kl.vol
                    else ""
                )
                hover_text.append(
                    f"时间: {kl.time.to_str()}<br>"
                    f"开: {kl.open}<br>高: {kl.high}<br>"
                    f"低: {kl.low}<br>收: {kl.close}"
                    f"{vol_str}"
                )

            self.figure.add_trace(
                go.Candlestick(
                    x=x_data,
                    open=open_d, high=high_d, low=low_d, close=close_d,
                    increasing_line_color=up_color,
                    decreasing_line_color=down_color,
                    increasing_fillcolor=up_color,
                    decreasing_fillcolor=down_color,
                    text=hover_text,
                    hoverinfo="text",
                    name="K线",
                    showlegend=False,
                ),
                row=row, col=1,
            )
        else:
            # ── 折线模式 ──
            valid_modes = {"close", "open", "high", "low"}
            if plot_mode not in valid_modes:
                raise CChanException(
                    f"未知 plot_mode='{plot_mode}'，可选: kl / {' / '.join(valid_modes)}",
                    ErrCode.PLOT_ERR,
                )
            _x, _y = [], []
            for kl in meta.klu_iter():
                if kl.idx < x_begin:
                    continue
                _x.append(kl.idx)
                _y.append(getattr(kl, plot_mode))
            if _x:
                self.figure.add_trace(
                    go.Scatter(
                        x=_x, y=_y, mode="lines",
                        name=plot_mode, showlegend=False,
                    ),
                    row=row, col=1,
                )

    # ════════════════════════════════════════════════════
    #  六、合并K线
    # ════════════════════════════════════════════════════

    def draw_klc(self, meta: CChanPlotMeta, row: int, width=0.4, plot_single_kl=True):
        """
        绘制合并K线（分型框）。

        颜色规则：
            - 顶分型 → 红色
            - 底分型 → 蓝色
            - 上升/下降方向 → 绿色
        """
        color_map = {
            FX_TYPE.TOP: "red",
            FX_TYPE.BOTTOM: "blue",
            KLINE_DIR.UP: "green",
            KLINE_DIR.DOWN: "green",
        }
        x_begin = self._x_limits[0]

        for klc in meta.klc_list:
            if klc.klu_list[-1].idx + width < x_begin:
                continue
            if klc.end_idx == klc.begin_idx and not plot_single_kl:
                continue
            color = color_map.get(klc.type, "green")
            self.figure.add_shape(
                type="rect",
                x0=klc.begin_idx - width, y0=klc.low,
                x1=klc.end_idx + width, y1=klc.high,
                line=dict(color=color, width=1),
                fillcolor="rgba(0,0,0,0)",
                row=row, col=1,
            )

    # ════════════════════════════════════════════════════
    #  七、笔
    # ════════════════════════════════════════════════════

    def draw_bi(
        self, meta: CChanPlotMeta, row: int, lv,
        color="black",
        show_num=False, num_fontsize=15, num_color="red",
        sub_lv_cnt=None, facecolor="green", alpha=0.1,
        disp_end=False, end_color="black", end_fontsize=10,
    ):
        """
        绘制笔。

        - 确定笔合并为连续折线（实线），减少 trace 数量
        - 不确定笔单独画虚线
        - ``show_num``   在笔中点显示编号
        - ``disp_end``   在端点显示价格
        - ``sub_lv_cnt`` 高亮子级别对齐区域
        """
        x_begin = self._x_limits[0]

        # 确定笔的坐标缓冲（合并为一条折线，避免 trace 过多）
        sure_x: List[int] = []
        sure_y: List[float] = []

        for bi_idx, bi in enumerate(meta.bi_list):
            if bi.end_x < x_begin:
                continue

            if bi.is_sure:
                # 确定笔：累积到缓冲区
                if not sure_x or sure_x[-1] != bi.begin_x:
                    sure_x.append(bi.begin_x)
                    sure_y.append(bi.begin_y)
                sure_x.append(bi.end_x)
                sure_y.append(bi.end_y)
            else:
                # 遇到不确定笔 → 先输出已累积的确定笔折线
                self._flush_line(sure_x, sure_y, color, row)
                sure_x, sure_y = [], []
                # 不确定笔：虚线
                self.figure.add_trace(
                    go.Scatter(
                        x=[bi.begin_x, bi.end_x],
                        y=[bi.begin_y, bi.end_y],
                        mode="lines",
                        line=dict(color=_to_css_color(color), dash="dash"),
                        showlegend=False,
                    ),
                    row=row, col=1,
                )

            # 笔编号
            if show_num and bi.begin_x >= x_begin:
                self.figure.add_annotation(
                    x=(bi.begin_x + bi.end_x) / 2,
                    y=(bi.begin_y + bi.end_y) / 2,
                    text=str(bi.idx),
                    font=dict(size=num_fontsize, color=_to_css_color(num_color)),
                    showarrow=False,
                    row=row, col=1,
                )
            # 端点价格标注
            if disp_end:
                self._bi_text(bi_idx, bi, row, end_fontsize, end_color)

        # 输出最后一段确定笔折线
        self._flush_line(sure_x, sure_y, color, row)

        # 子级别对齐区域高亮
        if sub_lv_cnt is not None and len(self.lv_lst) > 1 and lv != self.lv_lst[-1]:
            if sub_lv_cnt < len(meta.bi_list):
                begin_idx = meta.bi_list[-sub_lv_cnt].begin_x
                self.figure.add_shape(
                    type="rect",
                    x0=begin_idx, y0=self.y_min,
                    x1=self._x_limits[1], y1=self.y_max,
                    fillcolor=_to_css_color(facecolor), opacity=alpha,
                    line=dict(width=0),
                    row=row, col=1,
                )

    def _flush_line(self, xs: List[int], ys: List[float], color: str, row: int):
        """将坐标缓冲输出为一条折线 trace。列表为空时不操作。"""
        if xs:
            self.figure.add_trace(
                go.Scatter(
                    x=xs, y=ys, mode="lines",
                    line=dict(color=_to_css_color(color)),
                    showlegend=False,
                ),
                row=row, col=1,
            )

    # ════════════════════════════════════════════════════
    #  八、线段
    # ════════════════════════════════════════════════════

    def draw_seg(
        self, meta: CChanPlotMeta, row: int, lv,
        width=5, color="g",
        sub_lv_cnt=None, facecolor="green", alpha=0.1,
        disp_end=False, end_color="g", end_fontsize=13,
        plot_trendline=False, trendline_color="r", trendline_width=3,
        show_num=False, num_fontsize=25, num_color="blue",
    ):
        """
        绘制线段。

        - 确定线段: 实线，不确定线段: 虚线
        - ``plot_trendline`` 同时绘制支撑/阻力趋势线
        - ``sub_lv_cnt``     高亮子级别对齐区域
        """
        x_begin = self._x_limits[0]
        css = _to_css_color(color)

        for seg_idx, seg in enumerate(meta.seg_list):
            if seg.end_x < x_begin:
                continue
            dash = None if seg.is_sure else "dash"
            self.figure.add_trace(
                go.Scatter(
                    x=[seg.begin_x, seg.end_x],
                    y=[seg.begin_y, seg.end_y],
                    mode="lines",
                    line=dict(color=css, width=width, dash=dash),
                    showlegend=False,
                ),
                row=row, col=1,
            )
            if disp_end:
                self._bi_text(seg_idx, seg, row, end_fontsize, end_color)

            # 趋势线
            if plot_trendline:
                tl_css = _to_css_color(trendline_color)
                for tl_type in ("support", "resistance"):
                    if seg.tl.get(tl_type):
                        tl = seg.format_tl(seg.tl[tl_type])
                        self.figure.add_trace(
                            go.Scatter(
                                x=[tl[0], tl[2]], y=[tl[1], tl[3]],
                                mode="lines",
                                line=dict(color=tl_css, width=trendline_width),
                                showlegend=False,
                            ),
                            row=row, col=1,
                        )

            # 线段编号
            if show_num and seg.begin_x >= x_begin:
                self.figure.add_annotation(
                    x=(seg.begin_x + seg.end_x) / 2,
                    y=(seg.begin_y + seg.end_y) / 2,
                    text=str(seg.idx),
                    font=dict(size=num_fontsize, color=_to_css_color(num_color)),
                    showarrow=False,
                    row=row, col=1,
                )

        # 子级别对齐区域
        if sub_lv_cnt is not None and len(self.lv_lst) > 1 and lv != self.lv_lst[-1]:
            if sub_lv_cnt < len(meta.seg_list):
                begin_idx = meta.seg_list[-sub_lv_cnt].begin_x
                self.figure.add_shape(
                    type="rect",
                    x0=begin_idx, y0=self.y_min,
                    x1=self._x_limits[1], y1=self.y_max,
                    fillcolor=_to_css_color(facecolor), opacity=alpha,
                    line=dict(width=0),
                    row=row, col=1,
                )

    # ════════════════════════════════════════════════════
    #  九、线段的线段
    # ════════════════════════════════════════════════════

    def draw_segseg(
        self, meta: CChanPlotMeta, row: int,
        width=7, color="brown",
        disp_end=False, end_color="brown", end_fontsize=15,
        show_num=False, num_fontsize=30, num_color="blue",
    ):
        """绘制线段的线段（更高一级的线段结构，线条更粗以区分）。"""
        x_begin = self._x_limits[0]
        css = _to_css_color(color)

        for seg_idx, seg in enumerate(meta.segseg_list):
            if seg.end_x < x_begin:
                continue
            dash = None if seg.is_sure else "dash"
            self.figure.add_trace(
                go.Scatter(
                    x=[seg.begin_x, seg.end_x],
                    y=[seg.begin_y, seg.end_y],
                    mode="lines",
                    line=dict(color=css, width=width, dash=dash),
                    showlegend=False,
                ),
                row=row, col=1,
            )
            if disp_end:
                # 首段额外标注起点
                if seg_idx == 0:
                    va = "top" if seg.dir == BI_DIR.UP else "bottom"
                    self.figure.add_annotation(
                        x=seg.begin_x, y=seg.begin_y,
                        text=f"{seg.begin_y:.2f}",
                        font=dict(size=end_fontsize, color=_to_css_color(end_color)),
                        showarrow=False, yanchor=va,
                        row=row, col=1,
                    )
                # 每段终点
                va = "top" if seg.dir == BI_DIR.DOWN else "bottom"
                self.figure.add_annotation(
                    x=seg.end_x, y=seg.end_y,
                    text=f"{seg.end_y:.2f}",
                    font=dict(size=end_fontsize, color=_to_css_color(end_color)),
                    showarrow=False, yanchor=va,
                    row=row, col=1,
                )
            if show_num and seg.begin_x >= x_begin:
                self.figure.add_annotation(
                    x=(seg.begin_x + seg.end_x) / 2,
                    y=(seg.begin_y + seg.end_y) / 2,
                    text=str(seg.idx),
                    font=dict(size=num_fontsize, color=_to_css_color(num_color)),
                    showarrow=False,
                    row=row, col=1,
                )

    # ════════════════════════════════════════════════════
    #  十、特征序列
    # ════════════════════════════════════════════════════

    def _plot_single_eigen(
        self, eigenfx_meta, row: int,
        color_top: str, color_bottom: str, aplha: float, only_peak: bool,
    ):
        """绘制单个特征序列分型（3 元素矩形组）。"""
        x_begin = self._x_limits[0]
        color = color_top if eigenfx_meta.fx == FX_TYPE.TOP else color_bottom
        css = _to_css_color(color)

        for idx, eigen in enumerate(eigenfx_meta.ele):
            if eigen.begin_x + eigen.w < x_begin:
                continue
            if only_peak and idx != 1:
                continue  # 只画 peak（中间元素）
            self.figure.add_shape(
                type="rect",
                x0=eigen.begin_x, y0=eigen.begin_y,
                x1=eigen.begin_x + eigen.w,
                y1=eigen.begin_y + eigen.h,
                fillcolor=css, opacity=aplha,
                line=dict(width=0),
                row=row, col=1,
            )

    def draw_eigen(
        self, meta: CChanPlotMeta, row: int,
        color_top="r", color_bottom="b", aplha=0.5, only_peak=False,
    ):
        """绘制笔特征序列（笔分型的 3 笔特征矩形）。"""
        for efx in meta.eigenfx_lst:
            self._plot_single_eigen(efx, row, color_top, color_bottom, aplha, only_peak)

    def draw_segeigen(
        self, meta: CChanPlotMeta, row: int,
        color_top="r", color_bottom="b", aplha=0.5, only_peak=False,
    ):
        """绘制段特征序列（段分型的 3 段特征矩形）。"""
        for efx in meta.seg_eigenfx_lst:
            self._plot_single_eigen(efx, row, color_top, color_bottom, aplha, only_peak)

    # ════════════════════════════════════════════════════
    #  十一、中枢
    # ════════════════════════════════════════════════════

    def draw_zs(
        self, meta: CChanPlotMeta, row: int,
        color="orange", linewidth=2, sub_linewidth=0.5,
        show_text=False, fontsize=14, text_color="orange",
        draw_one_bi_zs=False,
    ):
        """
        绘制笔中枢。

        - 矩形框标识中枢区间 [ZD, ZG]
        - ``show_text``       在角落标注高低点价格
        - ``draw_one_bi_zs``  是否绘制单笔中枢
        """
        linewidth = max(linewidth, 2)
        x_begin = self._x_limits[0]
        css = _to_css_color(color)

        for zs in meta.zs_lst:
            if not draw_one_bi_zs and zs.is_onebi_zs:
                continue
            if zs.begin + zs.w < x_begin:
                continue
            dash = "solid" if zs.is_sure else "dash"
            # 主中枢
            self.figure.add_shape(
                type="rect",
                x0=zs.begin, y0=zs.low,
                x1=zs.begin + zs.w, y1=zs.low + zs.h,
                line=dict(color=css, width=linewidth, dash=dash),
                fillcolor="rgba(0,0,0,0)",
                row=row, col=1,
            )
            # 子中枢
            for sub in zs.sub_zs_lst:
                self.figure.add_shape(
                    type="rect",
                    x0=sub.begin, y0=sub.low,
                    x1=sub.begin + sub.w, y1=sub.low + sub.h,
                    line=dict(color=css, width=sub_linewidth, dash=dash),
                    fillcolor="rgba(0,0,0,0)",
                    row=row, col=1,
                )
            # 价格标注
            if show_text:
                self._add_zs_text(zs, row, fontsize, text_color)
                for sub in zs.sub_zs_lst:
                    self._add_zs_text(sub, row, fontsize, text_color)

    def draw_segzs(
        self, meta: CChanPlotMeta, row: int,
        color="red", linewidth=10, sub_linewidth=4,
        show_text=False, fontsize=14, text_color="red",
    ):
        """绘制段中枢（线条更粗，与笔中枢区分；可选标注高低点价格）。"""
        linewidth = max(linewidth, 2)
        x_begin = self._x_limits[0]
        css = _to_css_color(color)

        for zs in meta.segzs_lst:
            if zs.begin + zs.w < x_begin:
                continue
            dash = "solid" if zs.is_sure else "dash"
            self.figure.add_shape(
                type="rect",
                x0=zs.begin, y0=zs.low,
                x1=zs.begin + zs.w, y1=zs.low + zs.h,
                line=dict(color=css, width=linewidth, dash=dash),
                fillcolor="rgba(0,0,0,0)",
                row=row, col=1,
            )
            for sub in zs.sub_zs_lst:
                self.figure.add_shape(
                    type="rect",
                    x0=sub.begin, y0=sub.low,
                    x1=sub.begin + sub.w, y1=sub.low + sub.h,
                    line=dict(color=css, width=sub_linewidth, dash=dash),
                    fillcolor="rgba(0,0,0,0)",
                    row=row, col=1,
                )
            # 价格标注
            if show_text:
                self._add_zs_text(
                    zs, row, fontsize, text_color,
                )
                for sub in zs.sub_zs_lst:
                    self._add_zs_text(
                        sub, row, fontsize, text_color,
                    )

    # ════════════════════════════════════════════════════
    #  十二、MACD（独立子图）
    # ════════════════════════════════════════════════════

    def draw_macd(self, meta: CChanPlotMeta, row: int, x_limits: List[int], width=0.4):
        """
        在独立子图中绘制 MACD。

        - DIF 线: 橙色    - DEA 线: 蓝色    - MACD 柱: 正值红色 / 负值深绿
        """
        macd_lst = [klu.macd for klu in meta.klu_iter()]
        assert macd_lst[0] is not None, "需要先启用 MACD 计算（检查 CChanConfig 中 macd_metric 配置）"

        x_begin = x_limits[0]
        x_idx = list(range(len(macd_lst)))[x_begin:]
        dif = [m.DIF for m in macd_lst[x_begin:]]
        dea = [m.DEA for m in macd_lst[x_begin:]]
        bar = [m.macd for m in macd_lst[x_begin:]]
        bar_colors = ["red" if v >= 0 else "#006400" for v in bar]

        # DIF
        self.figure.add_trace(
            go.Scatter(
                x=x_idx, y=dif, mode="lines",
                line=dict(color="#FFA500", width=1),
                name="DIF", showlegend=False,
            ),
            row=row, col=1,
        )
        # DEA
        self.figure.add_trace(
            go.Scatter(
                x=x_idx, y=dea, mode="lines",
                line=dict(color="#0000FF", width=1),
                name="DEA", showlegend=False,
            ),
            row=row, col=1,
        )
        # MACD 柱
        self.figure.add_trace(
            go.Bar(
                x=x_idx, y=bar,
                marker_color=bar_colors,
                name="MACD", showlegend=False,
                width=width,
            ),
            row=row, col=1,
        )
        # 设置 MACD 子图 y 轴范围（留 5% 边距）
        if bar:
            all_vals = dif + dea + bar
            y_lo, y_hi = min(all_vals), max(all_vals)
            margin = (y_hi - y_lo) * 0.05 + 1e-7
            self.figure.update_yaxes(
                range=[y_lo - margin, y_hi + margin],
                row=row, col=1,
            )

    # ════════════════════════════════════════════════════
    #  十三、均线
    # ════════════════════════════════════════════════════

    def draw_mean(self, meta: CChanPlotMeta, row: int):
        """绘制均线族（颜色自动分配，图例显示周期）。"""
        mean_lst = [klu.trend[TREND_TYPE.MEAN] for klu in meta.klu_iter()]
        Ts = list(mean_lst[0].keys())

        import plotly.colors as pc
        palette = pc.qualitative.Set1 if len(Ts) <= 9 else pc.qualitative.Light24

        for i, T in enumerate(Ts):
            arr = [d[T] for d in mean_lst]
            self.figure.add_trace(
                go.Scatter(
                    x=list(range(len(arr))), y=arr,
                    mode="lines", line=dict(color=palette[i % len(palette)]),
                    name=f"MA{T}", showlegend=True,
                ),
                row=row, col=1,
            )

    # ════════════════════════════════════════════════════
    #  十四、通道线
    # ════════════════════════════════════════════════════

    def draw_channel(
        self, meta: CChanPlotMeta, row: int,
        T=None, top_color="r", bottom_color="b",
        linewidth=3, linestyle="solid",
    ):
        """
        绘制通道线（上轨 + 下轨）。

        参数 ``T`` 为通道周期，默认取配置中最大值。
        """
        max_lst = [klu.trend[TREND_TYPE.MAX] for klu in meta.klu_iter()]
        min_lst = [klu.trend[TREND_TYPE.MIN] for klu in meta.klu_iter()]
        config_T_lst = sorted(max_lst[0].keys())
        if T is None:
            T = config_T_lst[-1]
        elif T not in max_lst[0]:
            raise CChanException(
                f"通道周期 T={T} 未在 CChanConfig.trend_metrics 中配置，可选: {config_T_lst}",
                ErrCode.PLOT_ERR,
            )

        top_arr = [d[T] for d in max_lst]
        bot_arr = [d[T] for d in min_lst]
        x_idx = list(range(len(top_arr)))
        plotly_dash = _mpl_linestyle_to_plotly(linestyle)

        self.figure.add_trace(
            go.Scatter(
                x=x_idx, y=top_arr, mode="lines",
                line=dict(color=_to_css_color(top_color), width=linewidth, dash=plotly_dash),
                name=f"{T}-上轨", showlegend=True,
            ),
            row=row, col=1,
        )
        self.figure.add_trace(
            go.Scatter(
                x=x_idx, y=bot_arr, mode="lines",
                line=dict(color=_to_css_color(bottom_color), width=linewidth, dash=plotly_dash),
                name=f"{T}-下轨", showlegend=True,
            ),
            row=row, col=1,
        )

    # ════════════════════════════════════════════════════
    #  十五、布林带
    # ════════════════════════════════════════════════════

    def draw_boll(
        self, meta: CChanPlotMeta, row: int,
        mid_color="black", up_color="blue",
        down_color="purple",
    ):
        """绘制布林带三轨（MID / UP / DOWN），并自动扩展 y 轴范围。"""
        x_begin = self._x_limits[0]
        ma, up_arr, dn_arr = [], [], []
        try:
            for klu in meta.klu_iter():
                if klu.idx < x_begin:
                    continue
                ma.append(klu.boll.MID)
                up_arr.append(klu.boll.UP)
                dn_arr.append(klu.boll.DOWN)
        except AttributeError as e:
            raise CChanException(
                "无法绘制布林带，请在 CChanConfig 中设置 boll_n",
                ErrCode.PLOT_ERR,
            ) from e

        x_idx = list(range(x_begin, x_begin + len(ma)))
        for arr, c, name in [
            (ma, mid_color, "BOLL-MID"),
            (up_arr, up_color, "BOLL-UP"),
            (dn_arr, down_color, "BOLL-DN"),
        ]:
            self.figure.add_trace(
                go.Scatter(
                    x=x_idx, y=arr, mode="lines",
                    line=dict(color=_to_css_color(c)),
                    name=name, showlegend=False,
                ),
                row=row, col=1,
            )
        # 扩展 y 范围以容纳布林带
        if dn_arr:
            self.y_min = min(self.y_min, min(dn_arr))
            self.y_max = max(self.y_max, max(up_arr))

    # ════════════════════════════════════════════════════
    #  十六、买卖点
    # ════════════════════════════════════════════════════

    def _bsp_common_draw(
        self, bsp_list, row: int,
        buy_color: str, sell_color: str,
        fontsize: int, arrow_l: float, arrow_h: float, arrow_w: float,
    ):
        """
        买卖点通用绘制。

        买点：箭头从下方文字向上指向 K 线低点
        卖点：箭头从上方文字向下指向 K 线高点
        """
        x_begin = self._x_limits[0]
        y_range = self.y_max - self.y_min

        for bsp in bsp_list:
            if bsp.x < x_begin:
                continue
            color = _to_css_color(buy_color) if bsp.is_buy else _to_css_color(sell_color)
            arrow_dir = 1 if bsp.is_buy else -1
            arrow_len = arrow_l * y_range
            text_y = bsp.y - arrow_len * arrow_dir

            self.figure.add_annotation(
                x=bsp.x, y=bsp.y,
                ax=0,
                ay=arrow_len * arrow_dir * 40,  # 像素单位
                text=bsp.desc(),
                font=dict(size=fontsize, color=color),
                arrowcolor=color,
                arrowsize=1.5, arrowwidth=arrow_w, arrowhead=2,
                showarrow=True,
                row=row, col=1,
            )
            # 更新 y 范围确保注解不被裁剪
            self.y_min = min(self.y_min, text_y)
            self.y_max = max(self.y_max, text_y)

    def draw_bs_point(
        self, meta: CChanPlotMeta, row: int,
        buy_color="r", sell_color="g",
        fontsize=15, arrow_l=0.15, arrow_h=0.2, arrow_w=1,
    ):
        """绘制笔买卖点。"""
        self._bsp_common_draw(
            meta.bs_point_lst, row,
            buy_color, sell_color, fontsize, arrow_l, arrow_h, arrow_w,
        )

    def draw_seg_bs_point(
        self, meta: CChanPlotMeta, row: int,
        buy_color="r", sell_color="g",
        fontsize=18, arrow_l=0.2, arrow_h=0.25, arrow_w=1.2,
    ):
        """绘制段买卖点（箭头更大以便区分）。"""
        self._bsp_common_draw(
            meta.seg_bsp_lst, row,
            buy_color, sell_color, fontsize, arrow_l, arrow_h, arrow_w,
        )

    # ════════════════════════════════════════════════════
    #  十七、自定义标记 (Marker)
    # ════════════════════════════════════════════════════

    def draw_marker(
        self, meta: CChanPlotMeta, row: int,
        markers: Dict[
            Union[CTime, str],
            Union[Tuple[str, Literal["up", "down"], str], Tuple[str, Literal["up", "down"]]],
        ],
        arrow_l=0.15, arrow_h_r=0.2, arrow_w=1,
        fontsize=14, default_color="b",
    ):
        """
        绘制自定义标记点。

        markers 格式示例::

            {
                "2022/03/01": ("买入信号", "down"),              # 向下箭头（标记在K线下方）
                "2022/03/02": ("卖出信号", "up", "red"),         # 向上箭头，红色
            }

        ``"down"`` = 标记在K线下方，箭头从文字向上指到低点 — 常用于买入信号
        ``"up"``   = 标记在K线上方，箭头从文字向下指到高点 — 常用于卖出信号
        """
        x_begin, x_end = self._x_limits
        datetick_dict = {date: idx for idx, date in enumerate(meta.datetick)}

        # 处理子级别时间映射（子级别时间戳 → 当前级别 K 线时间戳）
        new_marker: dict = {}
        for klu in meta.klu_iter():
            for date, marker in markers.items():
                date_str = date.to_str() if isinstance(date, CTime) else date
                if klu.include_sub_lv_time(date_str) and klu.time.to_str() != date_str:
                    new_marker[klu.time.to_str()] = marker
        new_marker.update(markers)

        kl_dict = dict(enumerate(meta.klu_iter()))
        y_range = self.y_max - self.y_min
        arrow_len = arrow_l * y_range

        for date, marker in new_marker.items():
            if isinstance(date, CTime):
                date = date.to_str()
            if date not in datetick_dict:
                continue
            x = datetick_dict[date]
            if x < x_begin or x > x_end:
                continue

            if len(marker) == 2:
                color = default_color
                marker_content, position = marker
            else:
                assert len(marker) == 3
                marker_content, position, color = marker
            assert position in ("up", "down"), f"position 必须为 'up' 或 'down'，当前: {position}"

            _dir = -1 if position == "up" else 1
            bench = kl_dict[x].high if position == "up" else kl_dict[x].low
            css = _to_css_color(color)

            self.figure.add_annotation(
                x=x, y=bench,
                ax=0, ay=arrow_len * _dir * 40,
                text=marker_content,
                font=dict(size=fontsize, color=css),
                arrowcolor=css,
                arrowsize=1.5, arrowwidth=arrow_w, arrowhead=2,
                showarrow=True,
                row=row, col=1,
            )

    # ════════════════════════════════════════════════════
    #  十八、RSI（副 Y 轴叠加）
    # ════════════════════════════════════════════════════

    def draw_rsi(self, meta: CChanPlotMeta, row: int, color="b"):
        """
        在主图副 Y 轴（右侧）上叠加 RSI 曲线。

        利用 ``secondary_y`` 将 RSI（0~100）与价格轴完全分离，
        同时绘制 30（超卖）/ 70（超买）水平参考线。
        """
        data = [klu.rsi for klu in meta.klu_iter()]
        x_begin, x_end = self._x_limits
        x_idx = list(range(x_begin, x_end + 1))
        rsi_data = data[x_begin: x_end + 1]

        # RSI 曲线
        self.figure.add_trace(
            go.Scatter(
                x=x_idx, y=rsi_data, mode="lines",
                line=dict(color=_to_css_color(color), width=1),
                name="RSI", showlegend=True,
            ),
            row=row, col=1,
            secondary_y=True,
        )
        # 超卖(30) / 超买(70) 参考线
        for level in (30, 70):
            self.figure.add_trace(
                go.Scatter(
                    x=[x_begin, x_end], y=[level, level],
                    mode="lines",
                    line=dict(color="grey", width=0.5, dash="dot"),
                    showlegend=False,
                ),
                row=row, col=1,
                secondary_y=True,
            )
        # 副 Y 轴配置
        self.figure.update_yaxes(
            range=[0, 100],
            title_text="RSI",
            showgrid=False,
            row=row, col=1,
            secondary_y=True,
        )

    # ════════════════════════════════════════════════════
    #  十九、KDJ（副 Y 轴叠加）
    # ════════════════════════════════════════════════════

    def draw_kdj(
        self, meta: CChanPlotMeta, row: int,
        k_color="orange", d_color="blue", j_color="pink",
    ):
        """
        在主图副 Y 轴（右侧）上叠加 KDJ 三线。

        K/D 通常在 [0, 100]，J 可超出范围。
        利用 ``secondary_y`` 避免与价格轴混淆。
        """
        kdj = [klu.kdj for klu in meta.klu_iter()]
        x_begin, x_end = self._x_limits
        x_idx = list(range(x_begin, x_end + 1))

        for val_fn, c, name in [
            (lambda v: v.k, k_color, "K"),
            (lambda v: v.d, d_color, "D"),
            (lambda v: v.j, j_color, "J"),
        ]:
            self.figure.add_trace(
                go.Scatter(
                    x=x_idx,
                    y=[val_fn(v) for v in kdj[x_begin: x_end + 1]],
                    mode="lines",
                    line=dict(color=_to_css_color(c), width=1),
                    name=name, showlegend=True,
                ),
                row=row, col=1,
                secondary_y=True,
            )
        # 副 Y 轴配置
        self.figure.update_yaxes(
            title_text="KDJ",
            showgrid=False,
            row=row, col=1,
            secondary_y=True,
        )

    # ════════════════════════════════════════════════════
    #  二十、Demark 序列
    # ════════════════════════════════════════════════════

    def draw_demark(
        self, meta: CChanPlotMeta, row: int,
        setup_color="b", countdown_color="r",
        fontsize=12, min_setup=9,
        max_countdown_background="yellow",
        begin_line_color: Optional[str] = "purple",
        begin_line_style="dashed",
    ):
        """
        绘制 Demark 序列（TD Sequential）。

        - Setup   计数: 蓝色数字
        - Countdown 计数: 红色数字（达到 max 时加黄色背景）
        - TDST 起始线: 紫色虚线
        """
        x_begin = self._x_limits[0]
        plot_begin_set: set = set()
        plotly_dash = _mpl_linestyle_to_plotly(begin_line_style)

        # 每根 K 线上注解可能重叠，用偏移量错开
        y_range = self.y_max - self.y_min
        offset_unit = y_range * 0.015  # 每层偏移（数据坐标）

        for klu in meta.klu_iter():
            if klu.idx < x_begin:
                continue
            under_cnt = 0   # 向下累积层数
            upper_cnt = 0   # 向上累积层数

            # ── Setup ──
            for demark_idx in klu.demark.get_setup():
                series = demark_idx["series"]
                if series.idx < min_setup or not series.setup_finished:
                    continue
                # TDST 起始线
                if (
                    begin_line_color is not None
                    and series.TDST_peak is not None
                    and id(series) not in plot_begin_set
                ):
                    end_idx = (
                        series.countdown.kl_list[-1].idx
                        if series.countdown is not None
                        else series.kl_list[-1].idx
                    )
                    self.figure.add_trace(
                        go.Scatter(
                            x=[series.kl_list[CDemarkEngine.SETUP_BIAS].idx, end_idx],
                            y=[series.TDST_peak, series.TDST_peak],
                            mode="lines",
                            line=dict(color=_to_css_color(begin_line_color), dash=plotly_dash),
                            showlegend=False,
                        ),
                        row=row, col=1,
                    )
                    plot_begin_set.add(id(series))

                is_down = demark_idx["dir"] == BI_DIR.DOWN
                if is_down:
                    y_pos = klu.low - under_cnt * offset_unit
                    under_cnt += 1
                else:
                    y_pos = klu.high + upper_cnt * offset_unit
                    upper_cnt += 1
                va = "top" if is_down else "bottom"

                self.figure.add_annotation(
                    x=klu.idx, y=y_pos,
                    text=str(demark_idx["idx"]),
                    font=dict(size=fontsize, color=_to_css_color(setup_color)),
                    showarrow=False, yanchor=va,
                    row=row, col=1,
                )

            # ── Countdown ──
            for demark_idx in klu.demark.get_countdown():
                is_down = demark_idx["dir"] == BI_DIR.DOWN
                if is_down:
                    y_pos = klu.low - under_cnt * offset_unit
                    under_cnt += 1
                else:
                    y_pos = klu.high + upper_cnt * offset_unit
                    upper_cnt += 1
                va = "top" if is_down else "bottom"
                bgcolor = (
                    max_countdown_background
                    if demark_idx["idx"] == CDemarkEngine.MAX_COUNTDOWN
                    else None
                )

                self.figure.add_annotation(
                    x=klu.idx, y=y_pos,
                    text=str(demark_idx["idx"]),
                    font=dict(size=fontsize, color=_to_css_color(countdown_color)),
                    showarrow=False, yanchor=va,
                    bgcolor=bgcolor,
                    row=row, col=1,
                )

    # ────────────────────────────────────────────────────
    #  辅助绘制方法
    # ────────────────────────────────────────────────────

    def _bi_text(self, bi_idx: int, bi, row: int, end_fontsize: int, end_color: str):
        """在笔/段端点标注价格值。"""
        css = _to_css_color(end_color)
        if bi_idx == 0:
            va = "top" if bi.dir == BI_DIR.UP else "bottom"
            self.figure.add_annotation(
                x=bi.begin_x, y=bi.begin_y,
                text=f"{bi.begin_y:.5f}",
                font=dict(size=end_fontsize, color=css),
                showarrow=False, yanchor=va,
                row=row, col=1,
            )
        va = "top" if bi.dir == BI_DIR.DOWN else "bottom"
        self.figure.add_annotation(
            x=bi.end_x, y=bi.end_y,
            text=f"{bi.end_y:.5f}",
            font=dict(size=end_fontsize, color=css),
            showarrow=False, yanchor=va,
            row=row, col=1,
        )

    def _add_zs_text(self, zs_meta: CZS_meta, row: int, fontsize: int, text_color: str):
        """在中枢左下角标注低点、右上角标注高点。"""
        css = _to_css_color(text_color)
        self.figure.add_annotation(
            x=zs_meta.begin, y=zs_meta.low,
            text=f"{zs_meta.low:.2f}",
            font=dict(size=fontsize, color=css),
            showarrow=False, yanchor="top",
            row=row, col=1,
        )
        self.figure.add_annotation(
            x=zs_meta.begin + zs_meta.w,
            y=zs_meta.low + zs_meta.h,
            text=f"{zs_meta.low + zs_meta.h:.2f}",
            font=dict(size=fontsize, color=css),
            showarrow=False, yanchor="bottom",
            row=row, col=1,
        )


# ════════════════════════════════════════════════════════════════
#  工具函数
# ════════════════════════════════════════════════════════════════

def show_func_helper(func):
    """打印函数签名中带默认值的参数，辅助文档编写。"""
    print(f"{func.__name__}:")
    sig = inspect.signature(func)
    for name, param in sig.parameters.items():
        if param.default is inspect.Parameter.empty:
            continue
        if isinstance(param.default, str):
            print(f"\t{name}: '{param.default}'")
        else:
            print(f"\t{name}: {param.default}")
