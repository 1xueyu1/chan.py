from __future__ import annotations

import numpy as np


def annualized_sharpe(signal_returns: np.ndarray) -> float:
    if signal_returns.size == 0:
        return 0.0
    mean = float(np.mean(signal_returns))
    std = float(np.std(signal_returns, ddof=0))
    if std <= 1e-12:
        return 0.0
    return float(mean / std * np.sqrt(252.0))
