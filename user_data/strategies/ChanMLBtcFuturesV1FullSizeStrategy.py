from __future__ import annotations

import os
from pathlib import Path

from ChanMLRustStrategy import PROJECT_ROOT, ChanMLRustStrategy


class ChanMLBtcFuturesV1FullSizeStrategy(ChanMLRustStrategy):
    can_short = True
    timeframe = "15m"
    startup_candle_count = int(os.getenv("BTC_ML_V1_STARTUP_CANDLES", "2000"))
    allowed_bsp_families = ("2",)

    minimal_roi = {"0": float(os.getenv("BTC_ML_V1_MINIMAL_ROI_PCT", "100.0"))}
    stoploss = -float(os.getenv("BTC_ML_V1_STOP_LOSS_PCT", "0.10"))
    trailing_stop = False
    use_exit_signal = True

    position_sizing_mode = "fullsize"
    fixed_leverage = float(os.getenv("BTC_ML_V1_FULLSIZE_LEVERAGE", "1.0"))

    model_dir = Path(
        os.getenv(
            "BTC_ML_V1_MODEL_DIR",
            str(PROJECT_ROOT / "result/ml/btc_futures_v1_wf_validlabel_tier_sizing_a_boost/test_2026"),
        )
    )
    buy_model_path = Path(os.getenv("BTC_ML_V1_BUY_MODEL_PATH", str(model_dir / "buy_model.pkl")))
    sell_model_path = Path(os.getenv("BTC_ML_V1_SELL_MODEL_PATH", str(model_dir / "sell_model.pkl")))
    threshold_policy_path = Path(os.getenv("BTC_ML_V1_THRESHOLD_POLICY_PATH", str(model_dir / "threshold_policy.json")))

    buy_threshold = float(os.getenv("BTC_ML_V1_BUY_THRESHOLD", "0.76"))
    sell_threshold = float(os.getenv("BTC_ML_V1_SELL_THRESHOLD", "0.74"))
    entry_trend_filter_enabled = False
    long_probability_buffer = 0.0
    short_probability_buffer = 0.0
