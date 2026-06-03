from __future__ import annotations

import logging
import os
import sys
from collections import OrderedDict
from pathlib import Path

import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import IStrategy


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Backtest.chan_signal_extractor import extract_raw_bsp_events_from_bars
from Backtest.config import BacktestConfig
from Backtest.ml_filter import score_raw_events_with_ml
from Backtest.time_utils import event_signal_time
from ChanConfig import CChanConfig
from Common.CEnum import KL_TYPE
from RustCore import rust_config_path_from_chan_config


logger = logging.getLogger(__name__)


class ChanMLRustStrategy(IStrategy):
    INTERFACE_VERSION = 3

    can_short = True
    timeframe = "15m"
    process_only_new_candles = True
    startup_candle_count = int(os.getenv("CHAN_ML_STARTUP_CANDLES", "1500"))

    minimal_roi = {"0": float(os.getenv("CHAN_ML_MINIMAL_ROI_PCT", "0.05"))}
    stoploss = -float(os.getenv("CHAN_ML_STOPLOSS_PCT", "0.02"))
    trailing_stop = False
    use_exit_signal = os.getenv("CHAN_ML_USE_EXIT_SIGNAL", "1").lower() not in {"0", "false", "no", "off"}
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    order_types = {
        "entry": "market",
        "exit": "market",
        "stoploss": "market",
        "stoploss_on_exchange": False,
    }
    order_time_in_force = {"entry": "GTC", "exit": "GTC"}

    buy_model_path = Path(os.getenv("CHAN_ML_BUY_MODEL_PATH", str(PROJECT_ROOT / "result/ml/btc_futures_v1/buy_model.pkl")))
    sell_model_path = Path(os.getenv("CHAN_ML_SELL_MODEL_PATH", str(PROJECT_ROOT / "result/ml/btc_futures_v1/sell_model.pkl")))
    threshold_policy_path = Path(
        os.getenv("CHAN_ML_THRESHOLD_POLICY_PATH", str(PROJECT_ROOT / "result/ml/btc_futures_v1/threshold_policy.json"))
    )
    buy_threshold = float(os.getenv("CHAN_ML_BUY_THRESHOLD", "0.60"))
    sell_threshold = float(os.getenv("CHAN_ML_SELL_THRESHOLD", "0.60"))
    entry_trend_filter_enabled = os.getenv("CHAN_ML_ENABLE_ENTRY_TREND_FILTER", "1").lower() not in {"0", "false", "no", "off"}
    entry_ema_fast = int(os.getenv("CHAN_ML_ENTRY_EMA_FAST", "8"))
    entry_ema_slow = int(os.getenv("CHAN_ML_ENTRY_EMA_SLOW", "34"))
    long_probability_buffer = float(os.getenv("CHAN_ML_LONG_PROBABILITY_BUFFER", "0.02"))
    short_probability_buffer = float(os.getenv("CHAN_ML_SHORT_PROBABILITY_BUFFER", "0.00"))
    long_min_momentum_12 = float(os.getenv("CHAN_ML_LONG_MIN_MOMENTUM_12", "0.0"))
    short_max_momentum_12 = float(os.getenv("CHAN_ML_SHORT_MAX_MOMENTUM_12", "0.0"))
    min_entry_volume_ratio = float(os.getenv("CHAN_ML_MIN_ENTRY_VOLUME_RATIO", "0.0"))
    confidence_high_threshold = float(os.getenv("CHAN_ML_CONFIDENCE_HIGH_THRESHOLD", "0.61"))
    confidence_mid_threshold = float(os.getenv("CHAN_ML_CONFIDENCE_MID_THRESHOLD", "0.55"))
    confidence_low_threshold = float(os.getenv("CHAN_ML_CONFIDENCE_LOW_THRESHOLD", "0.50"))
    confidence_high_stake_scale = float(os.getenv("CHAN_ML_CONFIDENCE_HIGH_STAKE_SCALE", "1.0"))
    confidence_mid_stake_scale = float(os.getenv("CHAN_ML_CONFIDENCE_MID_STAKE_SCALE", "0.15"))
    confidence_low_stake_scale = float(os.getenv("CHAN_ML_CONFIDENCE_LOW_STAKE_SCALE", "0.05"))
    confidence_high_leverage = float(os.getenv("CHAN_ML_CONFIDENCE_HIGH_LEVERAGE", "3.0"))
    confidence_mid_leverage = float(os.getenv("CHAN_ML_CONFIDENCE_MID_LEVERAGE", "1.5"))
    confidence_low_leverage = float(os.getenv("CHAN_ML_CONFIDENCE_LOW_LEVERAGE", "1.0"))
    position_sizing_mode = os.getenv("CHAN_ML_POSITION_SIZING_MODE", "confidence").lower()
    fixed_leverage = float(os.getenv("CHAN_ML_FIXED_LEVERAGE", "1.0"))
    allowed_bsp_families = tuple(
        x.strip()
        for x in os.getenv("CHAN_ML_ALLOWED_BSP_FAMILIES", "").split(",")
        if x.strip()
    )

    @property
    def protections(self) -> list[dict]:
        if os.getenv("CHAN_ML_ENABLE_PROTECTIONS", "1").lower() in {"0", "false", "no", "off"}:
            return []
        return [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": 2,
            },
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 96,
                "trade_limit": 2,
                "stop_duration_candles": 16,
                "only_per_pair": True,
                "only_per_side": True,
                "required_profit": -0.001,
            },
            {
                "method": "LowProfitPairs",
                "lookback_period_candles": 192,
                "trade_limit": 2,
                "stop_duration_candles": 32,
                "required_profit": -0.02,
                "only_per_side": True,
            },
        ]

    plot_config = {
        "main_plot": {},
        "subplots": {
            "Chan ML": {
                "chan_ml_buy_probability": {"color": "green"},
                "chan_ml_sell_probability": {"color": "red"},
            }
        },
    }

    def _empty_signal_columns(self, dataframe: DataFrame) -> DataFrame:
        dataframe["chan_ml_buy_probability"] = 0.0
        dataframe["chan_ml_sell_probability"] = 0.0
        dataframe["chan_ml_buy_threshold"] = float(self.buy_threshold)
        dataframe["chan_ml_sell_threshold"] = float(self.sell_threshold)
        dataframe["chan_ml_buy_quality_score"] = 0.0
        dataframe["chan_ml_sell_quality_score"] = 0.0
        dataframe["chan_ml_market_state"] = ""
        dataframe["chan_ml_threshold_reason"] = ""
        dataframe["chan_ml_confidence_tier"] = ""
        dataframe["chan_ml_structure_tier"] = ""
        dataframe["chan_ml_structure_quality_score"] = 0.0
        dataframe["chan_ml_structure_risk_pct"] = 0.0
        dataframe["chan_ml_stake_multiplier"] = 1.0
        dataframe["chan_ml_risk_size_multiplier"] = 1.0
        dataframe["chan_ml_quality_size_multiplier"] = 1.0
        dataframe["chan_ml_buy_signal"] = 0
        dataframe["chan_ml_sell_signal"] = 0
        dataframe["chan_ml_event_type"] = ""
        return dataframe

    def _confidence_tier(self, probability: float) -> str:
        if float(probability) >= float(self.confidence_high_threshold):
            return "high"
        if float(probability) >= float(self.confidence_mid_threshold):
            return "mid"
        if float(probability) >= float(self.confidence_low_threshold):
            return "low"
        return "none"

    def _confidence_stake_scale(self, tier: str) -> float:
        if tier == "high":
            return float(self.confidence_high_stake_scale)
        if tier == "mid":
            return float(self.confidence_mid_stake_scale)
        if tier == "low":
            return float(self.confidence_low_stake_scale)
        return 0.0

    def _confidence_leverage(self, tier: str) -> float:
        if tier == "high":
            return float(self.confidence_high_leverage)
        if tier == "mid":
            return float(self.confidence_mid_leverage)
        if tier == "low":
            return float(self.confidence_low_leverage)
        return 1.0

    def _config(self) -> BacktestConfig:
        cfg = BacktestConfig(
            symbols=[],
            begin_time="1970-01-01",
            end_time="2099-12-31",
            kl_type=KL_TYPE.K_15M,
            allow_short=True,
            ml_enabled=True,
            ml_buy_model_path=str(self.buy_model_path),
            ml_sell_model_path=str(self.sell_model_path),
            ml_buy_threshold=float(self.buy_threshold),
            ml_sell_threshold=float(self.sell_threshold),
        )
        cfg.chan_config.update(
            {
                "use_rust_core": True,
                "mtf_chan_features": False,
                "trigger_step": True,
                "skip_step": 0,
            }
        )
        allowed = {str(x) for x in getattr(self, "allowed_bsp_families", ()) if str(x)}
        if allowed == {"2"}:
            cfg.chan_config["bs_type"] = "2,2s"
        return cfg

    def _bars_for_model(self, dataframe: DataFrame) -> DataFrame:
        bars = dataframe.copy()
        if "date" in bars.columns:
            index = pd.to_datetime(bars["date"], utc=True, errors="coerce")
        else:
            index = pd.to_datetime(bars.index, utc=True, errors="coerce")
        bars = bars.assign(_time=index).dropna(subset=["_time"]).set_index("_time")
        bars.index.name = "time"
        return bars[["open", "high", "low", "close", "volume"]].copy()

    def _cache_key(self, dataframe: DataFrame, pair: str) -> tuple:
        if dataframe.empty:
            return pair, 0, 0, 0.0
        if "date" in dataframe.columns:
            last_ts = pd.to_datetime(dataframe["date"].iloc[-1], utc=True, errors="coerce")
        else:
            last_ts = pd.to_datetime(dataframe.index[-1], utc=True, errors="coerce")
        last_value = int(last_ts.value) if pd.notna(last_ts) else 0
        return pair, len(dataframe), last_value, float(dataframe["close"].iloc[-1])

    @staticmethod
    def _bsp_family(value: object) -> str:
        text = str(value or "").strip().lower()
        if text.startswith("1"):
            return "1"
        if text.startswith("2"):
            return "2"
        if text.startswith("3"):
            return "3"
        return ""

    def _filter_raw_events_by_bsp_family(self, raw_events: list) -> list:
        allowed = {str(x) for x in getattr(self, "allowed_bsp_families", ()) if str(x)}
        if not allowed:
            return raw_events
        return [
            event
            for event in raw_events
            if self._bsp_family(getattr(event, "bsp_types_str", "") or getattr(event, "bsp_type", "")) in allowed
        ]

    def _score_dataframe(self, dataframe: DataFrame, pair: str) -> DataFrame:
        if not hasattr(self, "_chan_signal_cache"):
            self._chan_signal_cache = OrderedDict()

        key = self._cache_key(dataframe, pair)
        cached = self._chan_signal_cache.get(key)
        if cached is not None:
            self._chan_signal_cache.move_to_end(key)
            return cached.copy(deep=False)

        out = self._empty_signal_columns(dataframe.copy())
        if len(out) < 100:
            return out
        if not self.buy_model_path.exists() or not self.sell_model_path.exists():
            raise FileNotFoundError(f"Chan ML model files not found: {self.buy_model_path}, {self.sell_model_path}")

        cfg = self._config()
        bars = self._bars_for_model(out)
        raw_events = extract_raw_bsp_events_from_bars(cfg, pair, bars)
        raw_events = self._filter_raw_events_by_bsp_family(raw_events)
        if not raw_events:
            self._remember_cache(key, out)
            return out

        rust_config_path = rust_config_path_from_chan_config(CChanConfig(dict(cfg.chan_config)))
        scored = score_raw_events_with_ml(
            bars=bars,
            raw_events=raw_events,
            buy_model_path=str(self.buy_model_path),
            sell_model_path=str(self.sell_model_path),
            both_model_path=None,
            buy_threshold=float(self.buy_threshold),
            sell_threshold=float(self.sell_threshold),
            rust_config_path=rust_config_path,
            threshold_policy_path=(str(self.threshold_policy_path) if self.threshold_policy_path else None),
        )

        if "date" in out.columns:
            date_ns = pd.to_datetime(out["date"], utc=True, errors="coerce").astype("int64")
        else:
            date_ns = pd.to_datetime(out.index, utc=True, errors="coerce").astype("int64")
        pos_by_ns = {int(ns): pos for pos, ns in enumerate(date_ns)}
        columns = {name: out.columns.get_loc(name) for name in out.columns}

        for event in scored:
            pos = pos_by_ns.get(int(event_signal_time(event).value))
            if pos is None:
                continue
            if event.is_buy:
                current = float(out.iat[pos, columns["chan_ml_buy_probability"]])
                if event.probability >= current:
                    out.iat[pos, columns["chan_ml_buy_probability"]] = float(event.probability)
                    out.iat[pos, columns["chan_ml_buy_threshold"]] = float(event.threshold or self.buy_threshold)
                    out.iat[pos, columns["chan_ml_buy_quality_score"]] = float(event.quality_score)
                    out.iat[pos, columns["chan_ml_market_state"]] = str(event.market_state)
                    out.iat[pos, columns["chan_ml_threshold_reason"]] = str(event.threshold_reason)
                    out.iat[pos, columns["chan_ml_confidence_tier"]] = self._confidence_tier(float(event.probability))
                    out.iat[pos, columns["chan_ml_structure_tier"]] = str(getattr(event, "structure_confidence_tier", "") or "")
                    out.iat[pos, columns["chan_ml_structure_quality_score"]] = float(getattr(event, "structure_quality_score", 0.0) or 0.0)
                    out.iat[pos, columns["chan_ml_structure_risk_pct"]] = float(getattr(event, "structure_risk_pct", 0.0) or 0.0)
                    out.iat[pos, columns["chan_ml_stake_multiplier"]] = float(getattr(event, "stake_multiplier", 1.0) or 1.0)
                    out.iat[pos, columns["chan_ml_risk_size_multiplier"]] = float(getattr(event, "risk_size_multiplier", 1.0) or 1.0)
                    out.iat[pos, columns["chan_ml_quality_size_multiplier"]] = float(getattr(event, "quality_size_multiplier", 1.0) or 1.0)
                    out.iat[pos, columns["chan_ml_buy_signal"]] = int(event.signal == 1)
                    out.iat[pos, columns["chan_ml_event_type"]] = str(event.bsp_types_str)
            else:
                current = float(out.iat[pos, columns["chan_ml_sell_probability"]])
                if event.probability >= current:
                    out.iat[pos, columns["chan_ml_sell_probability"]] = float(event.probability)
                    out.iat[pos, columns["chan_ml_sell_threshold"]] = float(event.threshold or self.sell_threshold)
                    out.iat[pos, columns["chan_ml_sell_quality_score"]] = float(event.quality_score)
                    out.iat[pos, columns["chan_ml_market_state"]] = str(event.market_state)
                    out.iat[pos, columns["chan_ml_threshold_reason"]] = str(event.threshold_reason)
                    out.iat[pos, columns["chan_ml_confidence_tier"]] = self._confidence_tier(float(event.probability))
                    out.iat[pos, columns["chan_ml_structure_tier"]] = str(getattr(event, "structure_confidence_tier", "") or "")
                    out.iat[pos, columns["chan_ml_structure_quality_score"]] = float(getattr(event, "structure_quality_score", 0.0) or 0.0)
                    out.iat[pos, columns["chan_ml_structure_risk_pct"]] = float(getattr(event, "structure_risk_pct", 0.0) or 0.0)
                    out.iat[pos, columns["chan_ml_stake_multiplier"]] = float(getattr(event, "stake_multiplier", 1.0) or 1.0)
                    out.iat[pos, columns["chan_ml_risk_size_multiplier"]] = float(getattr(event, "risk_size_multiplier", 1.0) or 1.0)
                    out.iat[pos, columns["chan_ml_quality_size_multiplier"]] = float(getattr(event, "quality_size_multiplier", 1.0) or 1.0)
                    out.iat[pos, columns["chan_ml_sell_signal"]] = int(event.signal == -1)
                    out.iat[pos, columns["chan_ml_event_type"]] = str(event.bsp_types_str)

        self._remember_cache(key, out)
        return out.copy(deep=False)

    def _remember_cache(self, key: tuple, dataframe: DataFrame) -> None:
        self._chan_signal_cache[key] = dataframe.copy(deep=False)
        self._chan_signal_cache.move_to_end(key)
        while len(self._chan_signal_cache) > 64:
            self._chan_signal_cache.popitem(last=False)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        pair = metadata.get("pair", "")
        try:
            return self._score_dataframe(dataframe, pair)
        except Exception:
            logger.exception("Chan ML scoring failed for %s", pair)
            return self._empty_signal_columns(dataframe.copy())

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        long_condition = (
            (dataframe["chan_ml_buy_signal"] == 1)
            & (dataframe["volume"] > 0)
            & (dataframe["chan_ml_buy_probability"] >= (dataframe["chan_ml_buy_threshold"].astype(float) + float(self.long_probability_buffer)))
        )
        short_condition = (
            (dataframe["chan_ml_sell_signal"] == 1)
            & (dataframe["volume"] > 0)
            & (dataframe["chan_ml_sell_probability"] >= (dataframe["chan_ml_sell_threshold"].astype(float) + float(self.short_probability_buffer)))
        )

        if self.entry_trend_filter_enabled:
            close = dataframe["close"]
            volume = dataframe["volume"]
            ema_fast = close.ewm(span=self.entry_ema_fast, adjust=False, min_periods=self.entry_ema_fast).mean()
            ema_slow = close.ewm(span=self.entry_ema_slow, adjust=False, min_periods=self.entry_ema_slow).mean()
            momentum_4 = close.pct_change(4)
            momentum_12 = close.pct_change(12)
            volume_mean = volume.rolling(48, min_periods=1).mean()
            if self.min_entry_volume_ratio > 0:
                volume_ok = volume >= (volume_mean * float(self.min_entry_volume_ratio))
            else:
                volume_ok = volume > 0

            dataframe["chan_ml_entry_ema_fast"] = ema_fast
            dataframe["chan_ml_entry_ema_slow"] = ema_slow
            dataframe["chan_ml_entry_momentum_4"] = momentum_4
            dataframe["chan_ml_entry_momentum_12"] = momentum_12

            long_quality = (
                (close > ema_slow)
                & (ema_fast >= ema_slow)
                & (momentum_12 > float(self.long_min_momentum_12))
                & (momentum_4 > -0.006)
                & volume_ok
            )
            short_quality = (
                (close < ema_slow)
                & (ema_fast <= ema_slow)
                & (momentum_12 < float(self.short_max_momentum_12))
                & (momentum_4 < 0.006)
                & volume_ok
            )
            long_condition &= long_quality.fillna(False)
            short_condition &= short_quality.fillna(False)

        dataframe.loc[long_condition, "enter_long"] = 1
        dataframe.loc[long_condition, "enter_tag"] = (
            "chan_ml_buy_"
            + dataframe.loc[long_condition, "chan_ml_confidence_tier"].replace("", "none").astype(str)
            + ("_trendq" if self.entry_trend_filter_enabled else "")
        )
        dataframe.loc[short_condition, "enter_short"] = 1
        dataframe.loc[short_condition, "enter_tag"] = (
            "chan_ml_sell_"
            + dataframe.loc[short_condition, "chan_ml_confidence_tier"].replace("", "none").astype(str)
            + ("_trendq" if self.entry_trend_filter_enabled else "")
        )
        return dataframe

    def _latest_entry_row(self, pair: str):
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe is not None and not dataframe.empty:
                return dataframe.iloc[-1]
        except Exception:
            logger.debug("Could not resolve latest analyzed row for %s", pair, exc_info=True)
        return None

    def _tier_from_trade_context(self, pair: str, entry_tag: str | None = None) -> str:
        text = str(entry_tag or "").lower()
        for tier in ("high", "mid", "low"):
            if tier in text:
                return tier
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe is not None and not dataframe.empty and "chan_ml_confidence_tier" in dataframe.columns:
                tier = str(dataframe["chan_ml_confidence_tier"].iloc[-1] or "").lower()
                if tier in {"high", "mid", "low"}:
                    return tier
        except Exception:
            logger.debug("Could not resolve confidence tier for %s", pair, exc_info=True)
        return "high"

    def _structure_stake_multiplier(self, pair: str, default: float = 1.0) -> float:
        row = self._latest_entry_row(pair)
        if row is None or "chan_ml_stake_multiplier" not in row.index:
            return float(default)
        try:
            value = float(row.get("chan_ml_stake_multiplier", default))
        except Exception:
            return float(default)
        if value <= 0:
            return float(default)
        return max(0.0, min(1.0, value))

    def custom_stake_amount(
        self,
        pair: str,
        current_time,
        current_rate: float,
        proposed_stake: float,
        min_stake: float | None,
        max_stake: float,
        leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        mode = str(getattr(self, "position_sizing_mode", "confidence")).lower()
        if mode in {"off", "fixed", "none", "full", "fullsize"}:
            stake = float(proposed_stake)
        elif mode in {"structure", "structure_risk", "tier"}:
            stake = float(proposed_stake) * self._structure_stake_multiplier(pair, default=1.0)
        else:
            tier = self._tier_from_trade_context(pair, entry_tag)
            stake = float(proposed_stake) * max(0.0, float(self._confidence_stake_scale(tier)))
        if min_stake is not None and stake > 0:
            stake = max(float(min_stake), stake)
        if max_stake is not None and max_stake > 0:
            stake = min(float(max_stake), stake)
        return float(stake)

    def leverage(
        self,
        pair: str,
        current_time,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str | None,
        side: str,
        **kwargs,
    ) -> float:
        mode = str(getattr(self, "position_sizing_mode", "confidence")).lower()
        if mode in {"off", "fixed", "none", "full", "fullsize", "structure", "structure_risk", "tier"}:
            target = max(1.0, float(getattr(self, "fixed_leverage", 1.0)))
            if max_leverage is not None and max_leverage > 0:
                target = min(float(max_leverage), target)
            return float(target)
        tier = self._tier_from_trade_context(pair, entry_tag)
        target = max(1.0, float(self._confidence_leverage(tier)))
        if max_leverage is not None and max_leverage > 0:
            target = min(float(max_leverage), target)
        return float(target)

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        exit_long = (dataframe["chan_ml_sell_signal"] == 1) & (dataframe["volume"] > 0)
        exit_short = (dataframe["chan_ml_buy_signal"] == 1) & (dataframe["volume"] > 0)

        dataframe.loc[exit_long, "exit_long"] = 1
        dataframe.loc[exit_long, "exit_tag"] = "chan_ml_sell"
        dataframe.loc[exit_short, "exit_short"] = 1
        dataframe.loc[exit_short, "exit_tag"] = "chan_ml_buy"
        return dataframe
