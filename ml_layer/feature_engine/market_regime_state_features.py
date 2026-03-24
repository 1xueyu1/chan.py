from __future__ import annotations

from typing import Dict


def compute_market_regime_state_features(
    signal_context: Dict[str, float],
) -> Dict[str, float]:
    rv5 = float(signal_context.get("realized_vol_5", 0.0))
    vol_regime = float(signal_context.get("vol_regime", 1.0))
    adx_proxy = float(signal_context.get("trend_strength_adx", 0.0))

    if adx_proxy >= 25 and vol_regime < 2:
        market_regime = 1.0
    elif vol_regime >= 2:
        market_regime = 2.0
    else:
        market_regime = 0.0

    return {
        "realized_vol_5": rv5,
        "vol_regime": vol_regime,
        "trend_strength_adx": adx_proxy,
        "rolling_win_rate_20": float(signal_context.get("rolling_win_rate_20", 0.5)),
        "days_since_last_signal": float(signal_context.get("days_since_last_signal", 0.0)),
        "market_regime": market_regime,
    }
