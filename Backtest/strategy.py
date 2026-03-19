from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Dict, Optional


class ChanStrategyBase:
    """Compatibility base class for legacy scripts.

    The vectorbt implementation does not rely on callback-driven order execution,
    but this class keeps the old public methods available.
    """

    def __init__(self, chan_config_dict: Optional[Dict[str, Any]] = None):
        self.chan_config_dict = chan_config_dict or {}
        self.position = 0
        self.cash = 0.0
        self.equity_curve = []

        self._backtest_result = None
        self._performance: Dict[str, Any] = {}

    def on_chan_init(self):
        return None

    def on_chan_bsp(self, chan, bsp_list, klu):
        return None

    def on_backtest_end(self):
        return None

    def buy(self, price: float, size: float, comment: str = ""):
        self.position = 1

    def close_position(self, price: float, comment: str = ""):
        self.position = 0

    def set_backtest_result(self, result) -> None:
        self._backtest_result = result
        self._performance = dict(result.aggregate_metrics)

    def get_performance(self) -> Dict[str, Any]:
        return dict(self._performance)

    def generate_report(self, path: str, title: str = "") -> None:
        if not self._backtest_result or not self._backtest_result.artifacts:
            raise RuntimeError("No backtest result is attached")

        src = Path(self._backtest_result.artifacts.report_html)
        dst = Path(path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.resolve() != dst.resolve():
            shutil.copyfile(src, dst)
