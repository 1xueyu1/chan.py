# Plot/factory.py

from .config import DEFAULT_ENGINE


def get_plot_driver(engine=None):
    engine = engine or DEFAULT_ENGINE

    if engine == "mpl":
        from .mpl.PlotDriver import CPlotDriver
    elif engine == "plotly":
        from .plotly.PlotDriver import CPlotDriver
    else:
        raise ValueError(f"Unknown engine: {engine}")

    return CPlotDriver