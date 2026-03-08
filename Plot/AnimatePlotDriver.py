"""
Plot/AnimatePlotDriver.py — 向后兼容入口。

提供 ``CAnimateDriver``，在 Jupyter Notebook 中逐步回放缠论绘图。
自动选择当前默认绘图引擎（plotly 或 mpl）。

用法::

    from Plot.AnimatePlotDriver import CAnimateDriver
    CAnimateDriver(chan, plot_config=plot_config, plot_para=plot_para)
"""

from IPython.display import clear_output, display

from Chan import CChan

from .config import DEFAULT_ENGINE
from .factory import get_plot_driver


class CAnimateDriver:
    """
    逐步回放驱动：每读取一根新 K 线即更新一帧图表。

    对 plotly 后端使用 ``display(HTML(...))`` 实现内联刷新；
    对 mpl 后端使用 ``display(fig)`` + ``plt.close()``。
    """

    def __init__(
        self,
        chan: CChan,
        plot_config=None,
        plot_para=None,
    ):
        if plot_config is None:
            plot_config = {}
        if plot_para is None:
            plot_para = {}

        CPlotDriver = get_plot_driver()

        if DEFAULT_ENGINE == "mpl":
            import matplotlib.pyplot as plt
            for _ in chan.step_load():
                driver = CPlotDriver(
                    chan, plot_config, plot_para,
                )
                clear_output(wait=True)
                display(driver.figure)
                plt.close(driver.figure)
        else:
            # plotly / 其他引擎：通过 HTML 内联刷新
            from IPython.display import HTML
            for _ in chan.step_load():
                driver = CPlotDriver(
                    chan, plot_config, plot_para,
                )
                clear_output(wait=True)
                display(HTML(
                    driver.figure.to_html(
                        full_html=False,
                        include_plotlyjs="cdn",
                    )
                ))
