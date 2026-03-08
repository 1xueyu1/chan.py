"""
Plot/PlotDriver.py — 向后兼容入口。

通过默认绘图引擎（见 ``Plot/config.py`` 中的 ``DEFAULT_ENGINE``）
自动导出 ``CPlotDriver`` 类，使旧代码无需修改即可运行::

    from Plot.PlotDriver import CPlotDriver   # 等价于 get_plot_driver()
"""

from .factory import get_plot_driver

CPlotDriver = get_plot_driver()
