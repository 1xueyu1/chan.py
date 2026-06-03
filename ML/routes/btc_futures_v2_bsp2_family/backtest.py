from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import hashlib

import numpy as np
import pandas as pd

from ML.model import ModelBundle
from ML.shared.structure_position_sizing import bsp2_structure_quality_and_sizing

from .data import (
    DEFAULT_BACKTEST_DIR,
    DEFAULT_DATASET_PATH,
    DEFAULT_FUTURES_DATA_DIR,
    DEFAULT_MODEL_DIR,
    SYMBOL,
    futures_1m_path,
    load_1m_futures_bars,
    normalize_symbol,
    parse_utc,
    resample_ohlcv_open_labeled,
)
from .features import make_chan_config
from Backtest.chan_signal_extractor import extract_raw_bsp_events_from_bars
from .post_management import (
    DEFAULT_L3_MODEL_DIR,
    DEFAULT_POST_MODEL_PATH,
    dynamic_post_exit_for_trade,
    load_l3_hold_models,
    load_post_management_model,
)
from .bundle import bsp_family_for_frame
from .candidate_family import attach_bsp2_candidate_family
from .decision_models import load_bsp2_decision_models

PRIMARY_LABEL_COLUMN = "label_bsp2_family_valid"
STRUCTURE_TARGET_LEVELS = {"third", "t1", "t2", "t3"}
OPPOSITE_BSP_EXIT_MODES = {"opposite_bsp", "third_then_opposite_bsp", "third_then_opposite_no_after_timeout"}
FAMILY_ADAPTIVE_EXIT_MODE = "family_adaptive"
REALTIME_FAMILY_EXIT_MODE = "family_realtime"


def _cache_time_key(value: pd.Timestamp | None) -> str:
    if value is None or pd.isna(value):
        return "none"
    return pd.Timestamp(value).strftime("%Y%m%d%H%M")


def _build_all_bsp_exit_events_cached(
    dataset: pd.DataFrame,
    begin: pd.Timestamp | None,
    end: pd.Timestamp | None,
    *,
    tail_days: int = 7,
    cache_dir: str | Path = "data/btc_futures_v2_bsp2_family/cache/exit_events",
) -> pd.DataFrame:
    symbols = sorted(dataset["symbol"].dropna().astype(str).map(normalize_symbol).unique().tolist()) if "symbol" in dataset else [SYMBOL]
    digest = hashlib.md5(",".join(symbols).encode("utf-8")).hexdigest()[:10]
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / f"all_bsp_exit_events_{_cache_time_key(begin)}_{_cache_time_key(end)}_tail{int(tail_days)}_{digest}.parquet"
    if cache_path.exists():
        out = pd.read_parquet(cache_path)
        if "entry_time" in out.columns:
            out["entry_time"] = pd.to_datetime(out["entry_time"], utc=True, errors="coerce")
        return out
    out = _build_all_bsp_exit_events(dataset, begin, end, tail_days=tail_days)
    out.to_parquet(cache_path, index=False)
    return out


def _build_all_bsp_exit_events(
    dataset: pd.DataFrame,
    begin: pd.Timestamp | None,
    end: pd.Timestamp | None,
    *,
    tail_days: int = 7,
) -> pd.DataFrame:
    """Build same-level all-BSP event pool for rule exits."""
    rows: list[dict[str, Any]] = []
    if dataset.empty or "symbol" not in dataset.columns:
        return pd.DataFrame(columns=["symbol", "entry_time", "is_buy", "bsp_types_str", "entry_price"])
    for symbol_text in sorted(dataset["symbol"].dropna().astype(str).unique()):
        symbol = normalize_symbol(symbol_text)
        source_value = ""
        source_series = dataset.loc[dataset["symbol"].astype(str).eq(symbol_text), "source_path"] if "source_path" in dataset.columns else pd.Series(dtype=str)
        if not source_series.empty:
            source_value = str(source_series.dropna().iloc[0])
        source_path = Path(source_value) if source_value else futures_1m_path(symbol, DEFAULT_FUTURES_DATA_DIR)
        if not source_path.exists():
            source_path = futures_1m_path(symbol, DEFAULT_FUTURES_DATA_DIR)
        load_begin = (begin - pd.Timedelta(days=7)) if begin is not None else None
        load_end = (end + pd.Timedelta(days=max(7, int(tail_days)))) if end is not None else None
        bars_1m = load_1m_futures_bars(source_path, begin_time=load_begin, end_time=load_end)
        bars_15m = resample_ohlcv_open_labeled(bars_1m, "15min")
        raw_events = extract_raw_bsp_events_from_bars(make_chan_config(symbol), symbol, bars_15m)
        for event in raw_events:
            event_time = pd.to_datetime(event.exec_time, utc=True)
            if begin is not None and event_time < begin:
                continue
            if end is not None and event_time > end + pd.Timedelta(days=max(1, int(tail_days))):
                continue
            rows.append(
                {
                    "symbol": symbol,
                    "entry_time": event_time,
                    "is_buy": bool(event.is_buy),
                    "bsp_types_str": str(event.bsp_types_str),
                    "entry_price": float(event.trade_price),
                    "klu_idx": int(event.klu_idx),
                }
            )
    if not rows:
        return pd.DataFrame(columns=["symbol", "entry_time", "is_buy", "bsp_types_str", "entry_price"])
    out = pd.DataFrame(rows)
    out["entry_time"] = pd.to_datetime(out["entry_time"], utc=True, errors="coerce")
    return out.dropna(subset=["entry_time"]).drop_duplicates(subset=["symbol", "entry_time", "is_buy", "bsp_types_str", "klu_idx"]).sort_values(["symbol", "entry_time"]).reset_index(drop=True)


def _profit_factor(returns: pd.Series) -> float:
    wins = returns.loc[returns > 0].sum()
    losses = -returns.loc[returns < 0].sum()
    if losses <= 0:
        return float("inf") if wins > 0 else float("nan")
    return float(wins / losses)


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def _score_dataset(dataset: pd.DataFrame, model_dir: Path) -> pd.DataFrame:
    buy_model = ModelBundle.load(model_dir / "buy_model.pkl")
    sell_model = ModelBundle.load(model_dir / "sell_model.pkl")
    out = dataset.copy()
    for model in (buy_model, sell_model):
        prepare = getattr(model, "prepare_features", None)
        if prepare is not None:
            out = prepare(out)
    out["probability"] = np.nan
    out["threshold"] = np.nan
    out["ml_model_key"] = ""
    buy_mask = out["is_buy"].astype(bool)
    if bool(buy_mask.any()):
        rows = out.loc[buy_mask]
        out.loc[buy_mask, "probability"] = buy_model.predict_proba(rows)
        if hasattr(buy_model, "threshold_for_rows"):
            out.loc[buy_mask, "threshold"] = buy_model.threshold_for_rows(rows)
        else:
            out.loc[buy_mask, "threshold"] = float(getattr(buy_model, "threshold", 0.99))
        if hasattr(buy_model, "predict_model_key"):
            out.loc[buy_mask, "ml_model_key"] = buy_model.predict_model_key(rows)
        else:
            out.loc[buy_mask, "ml_model_key"] = "v1_buy"
    sell_mask = ~buy_mask
    if bool(sell_mask.any()):
        rows = out.loc[sell_mask]
        out.loc[sell_mask, "probability"] = sell_model.predict_proba(rows)
        if hasattr(sell_model, "threshold_for_rows"):
            out.loc[sell_mask, "threshold"] = sell_model.threshold_for_rows(rows)
        else:
            out.loc[sell_mask, "threshold"] = float(getattr(sell_model, "threshold", 0.99))
        if hasattr(sell_model, "predict_model_key"):
            out.loc[sell_mask, "ml_model_key"] = sell_model.predict_model_key(rows)
        else:
            out.loc[sell_mask, "ml_model_key"] = "v1_sell"
    return out


def _finite_float(value: Any, default: float = float("nan")) -> float:
    out = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(out) if np.isfinite(out) else float(default)


def _structure_target_price(row: pd.Series | dict[str, Any], target_level: str) -> float:
    level = str(target_level).lower()
    if level not in STRUCTURE_TARGET_LEVELS:
        raise ValueError(f"structure_target_level must be one of {sorted(STRUCTURE_TARGET_LEVELS)}")
    if level == "third":
        return _finite_float(row.get("label_bsp2_third_move_ref_price", np.nan))

    is_buy = bool(row.get("is_buy"))
    low = _finite_float(row.get("label_bsp2_prev_zs_low", np.nan))
    mid = _finite_float(row.get("label_bsp2_prev_zs_mid", np.nan))
    high = _finite_float(row.get("label_bsp2_prev_zs_high", np.nan))
    if is_buy:
        mapping = {"t1": low, "t2": mid, "t3": high}
    else:
        mapping = {"t1": high, "t2": mid, "t3": low}
    return float(mapping[level])


def _structure_target_exit_for_trade(
    row: pd.Series | dict[str, Any],
    bars_1m: pd.DataFrame,
    *,
    target_level: str = "t1",
    max_holding_minutes: int = 1440,
    cost_rate: float = 0.001,
    require_origin_zs: bool = True,
) -> dict[str, Any]:
    """Replay a pure Chan structure target exit on 1m bars.

    The target and invalidation prices are event-time fields created by the
    label/feature pipeline. Future bars are used only to replay execution.
    Stop is checked before target inside the same 1m bar to keep the replay
    conservative.
    """
    entry_time = pd.to_datetime(row.get("entry_time"), utc=True, errors="coerce")
    if pd.isna(entry_time):
        return {"skip_trade": True, "skip_reason": "missing_entry_time"}
    if require_origin_zs:
        source = str(row.get("label_bsp2_prev_zs_source", "") or "")
        has_source_col = "label_bsp2_prev_zs_source" in row
        has_direct_cols = np.isfinite(_finite_float(row.get("chan_bsp2_origin_zs_low", np.nan))) and np.isfinite(
            _finite_float(row.get("chan_bsp2_origin_zs_high", np.nan))
        )
        if (has_source_col and source != "chan_bsp2_origin_zs") or (not has_source_col and not has_direct_cols):
            return {"skip_trade": True, "skip_reason": "missing_bsp2_origin_zs"}

    bars = bars_1m.sort_index()
    if bars.empty:
        return {"skip_trade": True, "skip_reason": "empty_1m_bars"}
    times = pd.DatetimeIndex(pd.to_datetime(bars.index, utc=True))
    time_ns = times.view("int64")
    entry_pos = int(np.searchsorted(time_ns, pd.Timestamp(entry_time).value, side="left"))
    if entry_pos < 0 or entry_pos >= len(bars):
        return {"skip_trade": True, "skip_reason": "entry_out_of_range"}

    open_ = bars["open"].to_numpy(dtype="float64", copy=False)
    high = bars["high"].to_numpy(dtype="float64", copy=False)
    low = bars["low"].to_numpy(dtype="float64", copy=False)
    close = bars["close"].to_numpy(dtype="float64", copy=False)

    entry_price = _finite_float(row.get("entry_price", np.nan))
    if not np.isfinite(entry_price) or entry_price <= 0:
        entry_price = float(open_[entry_pos])
    if not np.isfinite(entry_price) or entry_price <= 0:
        return {"skip_trade": True, "skip_reason": "invalid_entry_price"}

    target_price = _structure_target_price(row, target_level)
    invalid_price = _finite_float(row.get("label_bsp2_invalid_price", np.nan))
    if not np.isfinite(target_price) or target_price <= 0:
        return {"skip_trade": True, "skip_reason": "missing_structure_target"}
    if not np.isfinite(invalid_price) or invalid_price <= 0:
        return {"skip_trade": True, "skip_reason": "missing_structure_invalid"}

    is_buy = bool(row.get("is_buy"))
    direction = 1.0 if is_buy else -1.0
    target_gross = direction * (target_price - entry_price) / (entry_price + 1e-12)
    invalid_gross = direction * (invalid_price - entry_price) / (entry_price + 1e-12)
    if target_gross <= 0:
        return {
            "skip_trade": True,
            "skip_reason": "target_not_ahead",
            "structure_target_price": float(target_price),
            "structure_invalid_price": float(invalid_price),
            "structure_target_gross_return": float(target_gross),
            "structure_invalid_gross_return": float(invalid_gross),
        }

    future_end = min(len(bars), entry_pos + max(1, int(max_holding_minutes)))
    exit_pos = future_end - 1
    exit_price = float(close[exit_pos])
    exit_reason = "structure_timeout"
    hit_target = False
    hit_invalid = False

    for pos in range(entry_pos, future_end):
        if is_buy:
            if low[pos] <= invalid_price:
                exit_pos = pos
                exit_price = float(invalid_price)
                exit_reason = "structure_invalid"
                hit_invalid = True
                break
            if high[pos] >= target_price:
                exit_pos = pos
                exit_price = float(target_price)
                exit_reason = f"structure_target_{target_level.lower()}"
                hit_target = True
                break
        else:
            if high[pos] >= invalid_price:
                exit_pos = pos
                exit_price = float(invalid_price)
                exit_reason = "structure_invalid"
                hit_invalid = True
                break
            if low[pos] <= target_price:
                exit_pos = pos
                exit_price = float(target_price)
                exit_reason = f"structure_target_{target_level.lower()}"
                hit_target = True
                break

    window = slice(entry_pos, future_end)
    if is_buy:
        mfe = float((np.nanmax(high[window]) - entry_price) / (entry_price + 1e-12))
        mae = float((np.nanmin(low[window]) - entry_price) / (entry_price + 1e-12))
    else:
        mfe = float((entry_price - np.nanmin(low[window])) / (entry_price + 1e-12))
        mae = float((entry_price - np.nanmax(high[window])) / (entry_price + 1e-12))

    gross_return = float(direction * (exit_price - entry_price) / (entry_price + 1e-12))
    return {
        "skip_trade": False,
        "entry_time": times[entry_pos],
        "entry_price": float(entry_price),
        "exit_time": times[exit_pos],
        "exit_price": float(exit_price),
        "exit_reason": exit_reason,
        "gross_return": gross_return,
        "net_return": float(gross_return - float(cost_rate)),
        "mfe": mfe,
        "mae": mae,
        "structure_target_level": str(target_level).lower(),
        "structure_target_price": float(target_price),
        "structure_invalid_price": float(invalid_price),
        "structure_target_gross_return": float(target_gross),
        "structure_invalid_gross_return": float(invalid_gross),
        "structure_hit_target": bool(hit_target),
        "structure_hit_invalid": bool(hit_invalid),
        "structure_holding_minutes": float((times[exit_pos] - times[entry_pos]).total_seconds() / 60.0),
    }


def _invalid_stop_exit_for_trade(
    row: pd.Series | dict[str, Any],
    bars_1m: pd.DataFrame,
    *,
    max_holding_minutes: int = 1440,
    cost_rate: float = 0.001,
) -> dict[str, Any]:
    entry_time = pd.to_datetime(row.get("entry_time"), utc=True, errors="coerce")
    if pd.isna(entry_time):
        return {"skip_trade": True, "skip_reason": "missing_entry_time"}
    bars = bars_1m.sort_index()
    if bars.empty:
        return {"skip_trade": True, "skip_reason": "empty_1m_bars"}
    times = pd.DatetimeIndex(pd.to_datetime(bars.index, utc=True))
    time_ns = times.view("int64")
    entry_pos = int(np.searchsorted(time_ns, pd.Timestamp(entry_time).value, side="left"))
    if entry_pos < 0 or entry_pos >= len(bars):
        return {"skip_trade": True, "skip_reason": "entry_out_of_range"}

    open_ = bars["open"].to_numpy(dtype="float64", copy=False)
    high = bars["high"].to_numpy(dtype="float64", copy=False)
    low = bars["low"].to_numpy(dtype="float64", copy=False)
    close = bars["close"].to_numpy(dtype="float64", copy=False)
    entry_price = _finite_float(row.get("entry_price", np.nan))
    if not np.isfinite(entry_price) or entry_price <= 0:
        entry_price = float(open_[entry_pos])
    invalid_price = _finite_float(row.get("label_bsp2_initial_invalid_price", row.get("label_bsp2_invalid_price", np.nan)))
    if not np.isfinite(invalid_price) or invalid_price <= 0:
        return {"skip_trade": True, "skip_reason": "missing_structure_invalid"}
    is_buy = bool(row.get("is_buy"))
    direction = 1.0 if is_buy else -1.0
    future_end = min(len(bars), entry_pos + max(1, int(max_holding_minutes)))
    exit_pos = future_end - 1
    exit_price = float(close[exit_pos])
    exit_reason = "weak_timeout"
    for pos in range(entry_pos, future_end):
        if is_buy and low[pos] <= invalid_price:
            exit_pos = pos
            exit_price = float(invalid_price)
            exit_reason = "structure_invalid"
            break
        if (not is_buy) and high[pos] >= invalid_price:
            exit_pos = pos
            exit_price = float(invalid_price)
            exit_reason = "structure_invalid"
            break
    window = slice(entry_pos, future_end)
    if is_buy:
        mfe = float((np.nanmax(high[window]) - entry_price) / (entry_price + 1e-12))
        mae = float((np.nanmin(low[window]) - entry_price) / (entry_price + 1e-12))
    else:
        mfe = float((entry_price - np.nanmin(low[window])) / (entry_price + 1e-12))
        mae = float((entry_price - np.nanmax(high[window])) / (entry_price + 1e-12))
    gross_return = float(direction * (exit_price - entry_price) / (entry_price + 1e-12))
    return {
        "skip_trade": False,
        "entry_time": times[entry_pos],
        "entry_price": float(entry_price),
        "exit_time": times[exit_pos],
        "exit_price": float(exit_price),
        "exit_reason": exit_reason,
        "gross_return": gross_return,
        "net_return": float(gross_return - float(cost_rate)),
        "mfe": mfe,
        "mae": mae,
        "structure_invalid_price": float(invalid_price),
        "structure_holding_minutes": float((times[exit_pos] - times[entry_pos]).total_seconds() / 60.0),
    }


def _hit_level(is_buy: bool, high_value: float, low_value: float, price: float) -> bool:
    if not np.isfinite(price) or price <= 0:
        return False
    return bool(high_value >= price) if is_buy else bool(low_value <= price)


def _hit_invalid(is_buy: bool, high_value: float, low_value: float, invalid_price: float) -> bool:
    if not np.isfinite(invalid_price) or invalid_price <= 0:
        return False
    return bool(low_value <= invalid_price) if is_buy else bool(high_value >= invalid_price)


def _family_realtime_exit_for_trade(
    row: pd.Series | dict[str, Any],
    exit_events: pd.DataFrame,
    bars_1m: pd.DataFrame,
    *,
    max_holding_minutes: int = 1440,
    no_timeout_max_holding_minutes: int = 43200,
    cost_rate: float = 0.001,
    use_structure_invalid_stop: bool = True,
    return_target_level: str = "t2",
    no_timeout_before_third: bool = False,
) -> dict[str, Any]:
    """Realtime BSP2-family exit state machine.

    It only uses entry-time structure fields and the replayed future path.
    It never reads label_bsp2_follow_class or label_bsp2_family_exit_mode.
    """
    entry_time = pd.to_datetime(row.get("entry_time"), utc=True, errors="coerce")
    if pd.isna(entry_time):
        return {"skip_trade": True, "skip_reason": "missing_entry_time"}
    bars = bars_1m.sort_index()
    if bars.empty:
        return {"skip_trade": True, "skip_reason": "empty_1m_bars"}

    times = pd.DatetimeIndex(pd.to_datetime(bars.index, utc=True))
    time_ns = times.view("int64")
    entry_pos = int(np.searchsorted(time_ns, pd.Timestamp(entry_time).value, side="left"))
    if entry_pos < 0 or entry_pos >= len(bars):
        return {"skip_trade": True, "skip_reason": "entry_out_of_range"}

    open_ = bars["open"].to_numpy(dtype="float64", copy=False)
    high = bars["high"].to_numpy(dtype="float64", copy=False)
    low = bars["low"].to_numpy(dtype="float64", copy=False)
    close = bars["close"].to_numpy(dtype="float64", copy=False)

    is_buy = bool(row.get("is_buy"))
    direction = 1.0 if is_buy else -1.0
    entry_price = _finite_float(row.get("entry_price", np.nan))
    if not np.isfinite(entry_price) or entry_price <= 0:
        entry_price = float(open_[entry_pos])
    if not np.isfinite(entry_price) or entry_price <= 0:
        return {"skip_trade": True, "skip_reason": "invalid_entry_price"}

    invalid_price = _finite_float(row.get("label_bsp2_invalid_price", row.get("chan_bsp2_invalid_price", np.nan)))
    third_price = _finite_float(row.get("label_bsp2_third_move_ref_price", np.nan))
    return_price = _structure_target_price(row, return_target_level)
    family = str(row.get("bsp2_candidate_family", row.get("bsp2_family", "")) or "").lower()
    bsp_types = str(row.get("bsp_types_str", row.get("bsp_type", "")) or "").lower()
    path_pred = str(row.get("bsp2_path_pred", "") or "").lower()
    weak_prob = _finite_float(row.get("bsp2_path_prob_weak_no_confirm", np.nan))
    invalid_prob = _finite_float(row.get("bsp2_pre_invalid_prob", np.nan))
    entry_has_third_confirm = family == "bsp2_t3_overlap" or "3a" in bsp_types or "3b" in bsp_types
    prefer_trend = bool(entry_has_third_confirm)

    if entry_has_third_confirm:
        replay = _opposite_bsp_exit_for_trade(
            row,
            exit_events,
            bars_1m,
            max_holding_minutes=int(no_timeout_max_holding_minutes),
            cost_rate=float(cost_rate),
            use_structure_invalid_stop=bool(use_structure_invalid_stop),
            require_third_before_opposite=True,
            third_already_confirmed=True,
            no_timeout_after_third=True,
            no_timeout_before_third=bool(no_timeout_before_third),
            no_timeout_max_holding_minutes=int(no_timeout_max_holding_minutes),
            cost_protect_after_third=True,
        )
        replay["adaptive_exit_mode"] = "realtime_entry_third_then_opposite"
        replay["adaptive_follow_class"] = "realtime_third_confirmed"
        return replay

    pre_deadline = entry_time + pd.Timedelta(minutes=max(1, int(max_holding_minutes)))
    if path_pred == "weak_no_confirm" or (np.isfinite(weak_prob) and weak_prob >= 0.55):
        pre_deadline = min(pre_deadline, entry_time + pd.Timedelta(minutes=360))
    elif np.isfinite(invalid_prob) and invalid_prob >= 0.70:
        pre_deadline = min(pre_deadline, entry_time + pd.Timedelta(minutes=720))
    if no_timeout_before_third:
        pre_deadline = entry_time + pd.Timedelta(minutes=max(1, int(no_timeout_max_holding_minutes)))
    pre_end_pos = int(np.searchsorted(time_ns, pd.Timestamp(pre_deadline).value, side="left"))
    pre_end_pos = min(max(entry_pos + 1, pre_end_pos), len(bars) - 1)

    for pos in range(entry_pos, pre_end_pos + 1):
        if use_structure_invalid_stop and _hit_invalid(is_buy, high[pos], low[pos], invalid_price):
            gross_return = float(direction * (float(invalid_price) - entry_price) / (entry_price + 1e-12))
            return {
                "skip_trade": False,
                "entry_time": times[entry_pos],
                "entry_price": float(entry_price),
                "exit_time": times[pos],
                "exit_price": float(invalid_price),
                "exit_reason": "realtime_structure_invalid_before_confirm",
                "gross_return": gross_return,
                "net_return": float(gross_return - float(cost_rate)),
                "third_confirmed": False,
                "adaptive_exit_mode": "realtime_invalid_stop",
                "adaptive_follow_class": "realtime_invalid_before_confirm",
                "structure_invalid_price": float(invalid_price),
                "structure_holding_minutes": float((times[pos] - times[entry_pos]).total_seconds() / 60.0),
            }
        if _hit_level(is_buy, high[pos], low[pos], third_price):
            replay = _opposite_bsp_exit_for_trade(
                row,
                exit_events,
                bars_1m,
                max_holding_minutes=int(no_timeout_max_holding_minutes),
                cost_rate=float(cost_rate),
                use_structure_invalid_stop=bool(use_structure_invalid_stop),
                require_third_before_opposite=True,
                no_timeout_after_third=True,
                no_timeout_before_third=bool(no_timeout_before_third),
                no_timeout_max_holding_minutes=int(no_timeout_max_holding_minutes),
                cost_protect_after_third=True,
            )
            replay["adaptive_exit_mode"] = "realtime_third_then_opposite"
            replay["adaptive_follow_class"] = "realtime_third_confirmed"
            return replay
        if (not prefer_trend) and _hit_level(is_buy, high[pos], low[pos], return_price):
            gross_return = float(direction * (float(return_price) - entry_price) / (entry_price + 1e-12))
            window = slice(entry_pos, pos + 1)
            if is_buy:
                mfe = float((np.nanmax(high[window]) - entry_price) / (entry_price + 1e-12))
                mae = float((np.nanmin(low[window]) - entry_price) / (entry_price + 1e-12))
            else:
                mfe = float((entry_price - np.nanmin(low[window])) / (entry_price + 1e-12))
                mae = float((entry_price - np.nanmax(high[window])) / (entry_price + 1e-12))
            return {
                "skip_trade": False,
                "entry_time": times[entry_pos],
                "entry_price": float(entry_price),
                "exit_time": times[pos],
                "exit_price": float(return_price),
                "exit_reason": f"realtime_return_zs_{return_target_level}",
                "gross_return": gross_return,
                "net_return": float(gross_return - float(cost_rate)),
                "mfe": mfe,
                "mae": mae,
                "third_confirmed": False,
                "adaptive_exit_mode": "realtime_return_zs_target",
                "adaptive_follow_class": "realtime_return_origin_zs",
                "structure_target_level": str(return_target_level).lower(),
                "structure_target_price": float(return_price),
                "structure_invalid_price": float(invalid_price) if np.isfinite(invalid_price) else float("nan"),
                "structure_holding_minutes": float((times[pos] - times[entry_pos]).total_seconds() / 60.0),
            }

    exit_price = float(close[pre_end_pos])
    gross_return = float(direction * (exit_price - entry_price) / (entry_price + 1e-12))
    return {
        "skip_trade": False,
        "entry_time": times[entry_pos],
        "entry_price": float(entry_price),
        "exit_time": times[pre_end_pos],
        "exit_price": exit_price,
        "exit_reason": "realtime_no_confirm_timeout",
        "gross_return": gross_return,
        "net_return": float(gross_return - float(cost_rate)),
        "third_confirmed": False,
        "adaptive_exit_mode": "realtime_timeout",
        "adaptive_follow_class": "realtime_weak_no_confirm",
        "structure_invalid_price": float(invalid_price) if np.isfinite(invalid_price) else float("nan"),
        "structure_holding_minutes": float((times[pre_end_pos] - times[entry_pos]).total_seconds() / 60.0),
    }


def _bsp2_quality_and_sizing(
    row: pd.Series | dict[str, Any],
    *,
    enabled: bool,
    target_risk_pct: float,
    min_stake_multiplier: float,
    tier_a_max_stake: float,
    tier_b_max_stake: float,
    tier_c_max_stake: float,
    tier_a_risk_boost: float,
    tier_b_risk_boost: float,
    tier_c_risk_boost: float,
    cost_rate: float,
) -> dict[str, Any]:
    return bsp2_structure_quality_and_sizing(
        row,
        enabled=enabled,
        target_risk_pct=target_risk_pct,
        min_stake_multiplier=min_stake_multiplier,
        tier_a_max_stake=tier_a_max_stake,
        tier_b_max_stake=tier_b_max_stake,
        tier_c_max_stake=tier_c_max_stake,
        tier_a_risk_boost=tier_a_risk_boost,
        tier_b_risk_boost=tier_b_risk_boost,
        tier_c_risk_boost=tier_c_risk_boost,
        cost_rate=cost_rate,
    )


def _opposite_bsp_exit_for_trade(
    row: pd.Series | dict[str, Any],
    exit_events: pd.DataFrame,
    bars_1m: pd.DataFrame,
    *,
    max_holding_minutes: int = 1440,
    cost_rate: float = 0.001,
    use_structure_invalid_stop: bool = True,
    require_third_before_opposite: bool = False,
    third_already_confirmed: bool = False,
    no_timeout_after_third: bool = False,
    no_timeout_before_third: bool = False,
    no_timeout_max_holding_minutes: int = 43200,
    cost_protect_after_third: bool = False,
) -> dict[str, Any]:
    """Exit on the next same-level opposite BSP event.

    Long positions exit on the next sell BSP; short positions exit on the next
    buy BSP. A structural invalidation stop is checked before the opposite BSP
    to avoid holding through a failed BSP2 structure.
    """
    entry_time = pd.to_datetime(row.get("entry_time"), utc=True, errors="coerce")
    if pd.isna(entry_time):
        return {"skip_trade": True, "skip_reason": "missing_entry_time"}
    bars = bars_1m.sort_index()
    if bars.empty:
        return {"skip_trade": True, "skip_reason": "empty_1m_bars"}

    symbol = normalize_symbol(str(row.get("symbol", SYMBOL) or SYMBOL))
    is_buy = bool(row.get("is_buy"))
    direction = 1.0 if is_buy else -1.0
    max_exit_time = entry_time + pd.Timedelta(minutes=max(1, int(max_holding_minutes)))
    final_exit_time = entry_time + pd.Timedelta(minutes=max(1, int(no_timeout_max_holding_minutes)))
    candidate_deadline = final_exit_time if (no_timeout_after_third or no_timeout_before_third) else max_exit_time
    if exit_events.empty:
        candidates = exit_events
    else:
        event_times = pd.to_datetime(exit_events["entry_time"], utc=True, errors="coerce")
        candidates = exit_events.loc[
            exit_events["symbol"].astype(str).eq(symbol)
            & (exit_events["is_buy"].astype(bool) != is_buy)
            & (event_times > entry_time)
            & (event_times <= candidate_deadline)
        ].sort_values("entry_time")
    opposite_row = candidates.iloc[0] if not candidates.empty else None
    opposite_time = pd.to_datetime(opposite_row["entry_time"], utc=True) if opposite_row is not None else pd.NaT

    times = pd.DatetimeIndex(pd.to_datetime(bars.index, utc=True))
    time_ns = times.view("int64")
    entry_pos = int(np.searchsorted(time_ns, pd.Timestamp(entry_time).value, side="left"))
    if entry_pos < 0 or entry_pos >= len(bars):
        return {"skip_trade": True, "skip_reason": "entry_out_of_range"}
    deadline = opposite_time if pd.notna(opposite_time) else candidate_deadline
    end_pos = int(np.searchsorted(time_ns, pd.Timestamp(deadline).value, side="left"))
    end_pos = min(max(entry_pos + 1, end_pos), len(bars) - 1)

    open_ = bars["open"].to_numpy(dtype="float64", copy=False)
    high = bars["high"].to_numpy(dtype="float64", copy=False)
    low = bars["low"].to_numpy(dtype="float64", copy=False)
    close = bars["close"].to_numpy(dtype="float64", copy=False)
    entry_price = _finite_float(row.get("entry_price", np.nan))
    if not np.isfinite(entry_price) or entry_price <= 0:
        entry_price = float(open_[entry_pos])
    if not np.isfinite(entry_price) or entry_price <= 0:
        return {"skip_trade": True, "skip_reason": "invalid_entry_price"}

    invalid_price = _finite_float(row.get("label_bsp2_invalid_price", np.nan))
    target_price = _finite_float(row.get("label_bsp2_third_move_ref_price", np.nan))
    third_pos: int | None = None
    if require_third_before_opposite:
        if third_already_confirmed:
            third_pos = entry_pos
        elif not np.isfinite(target_price) or target_price <= 0:
            return {"skip_trade": True, "skip_reason": "missing_third_target"}
        if third_pos is None:
            third_deadline = final_exit_time if no_timeout_before_third else max_exit_time
            if pd.notna(opposite_time) and opposite_time < third_deadline:
                third_deadline = opposite_time
            third_end_pos = int(np.searchsorted(time_ns, pd.Timestamp(third_deadline).value, side="left"))
            third_end_pos = min(max(entry_pos + 1, third_end_pos), len(bars) - 1)
            for pos in range(entry_pos, third_end_pos + 1):
                if use_structure_invalid_stop and np.isfinite(invalid_price) and invalid_price > 0:
                    if is_buy and low[pos] <= invalid_price:
                        exit_price = float(invalid_price)
                        exit_time = times[pos]
                        gross_return = float(direction * (exit_price - entry_price) / (entry_price + 1e-12))
                        return {
                            "skip_trade": False,
                            "entry_time": times[entry_pos],
                            "entry_price": float(entry_price),
                            "exit_time": exit_time,
                            "exit_price": exit_price,
                            "exit_reason": "structure_invalid_before_third",
                            "gross_return": gross_return,
                            "net_return": float(gross_return - float(cost_rate)),
                            "opposite_bsp_found": bool(opposite_row is not None),
                            "opposite_bsp_time": opposite_time,
                            "third_confirmed": False,
                        }
                    if (not is_buy) and high[pos] >= invalid_price:
                        exit_price = float(invalid_price)
                        exit_time = times[pos]
                        gross_return = float(direction * (exit_price - entry_price) / (entry_price + 1e-12))
                        return {
                            "skip_trade": False,
                            "entry_time": times[entry_pos],
                            "entry_price": float(entry_price),
                            "exit_time": exit_time,
                            "exit_price": exit_price,
                            "exit_reason": "structure_invalid_before_third",
                            "gross_return": gross_return,
                            "net_return": float(gross_return - float(cost_rate)),
                            "opposite_bsp_found": bool(opposite_row is not None),
                            "opposite_bsp_time": opposite_time,
                            "third_confirmed": False,
                        }
                if is_buy and high[pos] >= target_price:
                    third_pos = pos
                    break
                if (not is_buy) and low[pos] <= target_price:
                    third_pos = pos
                    break
            if third_pos is None:
                exit_time = times[third_end_pos]
                exit_price = float(close[third_end_pos])
                gross_return = float(direction * (exit_price - entry_price) / (entry_price + 1e-12))
                return {
                    "skip_trade": False,
                    "entry_time": times[entry_pos],
                    "entry_price": float(entry_price),
                    "exit_time": exit_time,
                    "exit_price": exit_price,
                    "exit_reason": "third_then_opposite_timeout_before_third",
                    "gross_return": gross_return,
                    "net_return": float(gross_return - float(cost_rate)),
                    "opposite_bsp_found": bool(opposite_row is not None),
                    "opposite_bsp_time": opposite_time,
                    "third_confirmed": False,
                }
        if cost_protect_after_third:
            invalid_price = entry_price * (1.0 + float(cost_rate)) if is_buy else entry_price * (1.0 - float(cost_rate))
        else:
            invalid_price = entry_price
        entry_pos_for_stop = third_pos
        if no_timeout_after_third:
            if exit_events.empty:
                candidates_after = exit_events
            else:
                event_times = pd.to_datetime(exit_events["entry_time"], utc=True, errors="coerce")
                third_time = times[third_pos]
                candidates_after = exit_events.loc[
                    exit_events["symbol"].astype(str).eq(symbol)
                    & (exit_events["is_buy"].astype(bool) != is_buy)
                    & (event_times > third_time)
                    & (event_times <= final_exit_time)
                ].sort_values("entry_time")
            opposite_row = candidates_after.iloc[0] if not candidates_after.empty else None
            opposite_time = pd.to_datetime(opposite_row["entry_time"], utc=True) if opposite_row is not None else pd.NaT
            deadline = opposite_time if pd.notna(opposite_time) else final_exit_time
            end_pos = int(np.searchsorted(time_ns, pd.Timestamp(deadline).value, side="left"))
            end_pos = min(max(entry_pos_for_stop + 1, end_pos), len(bars) - 1)
    else:
        entry_pos_for_stop = entry_pos

    if use_structure_invalid_stop and np.isfinite(invalid_price) and invalid_price > 0:
        for pos in range(entry_pos_for_stop, end_pos + 1):
            if is_buy and low[pos] <= invalid_price:
                exit_price = float(invalid_price)
                exit_time = times[pos]
                gross_return = float(direction * (exit_price - entry_price) / (entry_price + 1e-12))
                return {
                    "skip_trade": False,
                    "entry_time": times[entry_pos],
                    "entry_price": float(entry_price),
                    "exit_time": exit_time,
                    "exit_price": exit_price,
                    "exit_reason": "structure_invalid_before_opposite_bsp",
                    "gross_return": gross_return,
                    "net_return": float(gross_return - float(cost_rate)),
                    "opposite_bsp_found": bool(opposite_row is not None),
                    "opposite_bsp_time": opposite_time,
                    "third_confirmed": bool(third_pos is not None),
                }
            if (not is_buy) and high[pos] >= invalid_price:
                exit_price = float(invalid_price)
                exit_time = times[pos]
                gross_return = float(direction * (exit_price - entry_price) / (entry_price + 1e-12))
                return {
                    "skip_trade": False,
                    "entry_time": times[entry_pos],
                    "entry_price": float(entry_price),
                    "exit_time": exit_time,
                    "exit_price": exit_price,
                    "exit_reason": "structure_invalid_before_opposite_bsp",
                    "gross_return": gross_return,
                    "net_return": float(gross_return - float(cost_rate)),
                    "opposite_bsp_found": bool(opposite_row is not None),
                    "opposite_bsp_time": opposite_time,
                    "third_confirmed": bool(third_pos is not None),
                }

    if opposite_row is not None:
        exit_time = pd.to_datetime(opposite_row["entry_time"], utc=True)
        exit_price = _finite_float(opposite_row.get("entry_price", np.nan))
        if not np.isfinite(exit_price) or exit_price <= 0:
            pos = int(np.searchsorted(time_ns, pd.Timestamp(exit_time).value, side="left"))
            pos = min(max(0, pos), len(bars) - 1)
            exit_price = float(open_[pos])
        exit_reason = "opposite_bsp"
    else:
        exit_time = times[end_pos]
        exit_price = float(close[end_pos])
        exit_reason = "opposite_bsp_open_end" if no_timeout_after_third else "opposite_bsp_timeout"

    window = slice(entry_pos, end_pos + 1)
    if is_buy:
        mfe = float((np.nanmax(high[window]) - entry_price) / (entry_price + 1e-12))
        mae = float((np.nanmin(low[window]) - entry_price) / (entry_price + 1e-12))
    else:
        mfe = float((entry_price - np.nanmin(low[window])) / (entry_price + 1e-12))
        mae = float((entry_price - np.nanmax(high[window])) / (entry_price + 1e-12))
    gross_return = float(direction * (exit_price - entry_price) / (entry_price + 1e-12))
    return {
        "skip_trade": False,
        "entry_time": times[entry_pos],
        "entry_price": float(entry_price),
        "exit_time": exit_time,
        "exit_price": float(exit_price),
        "exit_reason": exit_reason,
        "gross_return": gross_return,
        "net_return": float(gross_return - float(cost_rate)),
        "mfe": mfe,
        "mae": mae,
        "opposite_bsp_found": bool(opposite_row is not None),
        "opposite_bsp_time": opposite_time,
        "third_confirmed": bool(third_pos is not None),
    }


def run_btc_futures_backtest(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    model_dir: str | Path = DEFAULT_MODEL_DIR,
    output_dir: str | Path = DEFAULT_BACKTEST_DIR,
    begin_time: Any | None = "2026-01-01",
    end_time: Any | None = None,
    initial_cash: float = 100000.0,
    stake_fraction: float = 1.0,
    exit_mode: str = REALTIME_FAMILY_EXIT_MODE,
    post_model_path: str | Path = DEFAULT_POST_MODEL_PATH,
    snapshot_minutes: int = 5,
    min_holding_minutes: int = 15,
    emergency_stop_pct: float = 0.012,
    profit_protect_trigger_pct: float = 0.016,
    profit_protect_giveback_ratio: float = 0.35,
    profit_protect_min_giveback_pct: float = 0.006,
    min_model_exit_return: float = -0.006,
    post_exit_cooldown_minutes: int = 0,
    post_model_policy: str = "off",
    use_structure_invalid_stop: bool = True,
    l3_model_dir: str | Path = DEFAULT_L3_MODEL_DIR,
    l3_policy: str = "off",
    l3_long_threshold: float | None = None,
    l3_short_threshold: float | None = None,
    structure_target_level: str = "third",
    structure_target_cost_rate: float = 0.001,
    pure_bsp2_only: bool = False,
    use_ml_filter: bool = False,
    require_origin_zs: bool = False,
    require_origin_zs_for_entry: bool = True,
    enable_structure_risk_sizing: bool = True,
    target_trade_risk_pct: float = 0.01,
    min_stake_multiplier: float = 0.08,
    tier_a_max_stake: float = 1.0,
    tier_b_max_stake: float = 0.55,
    tier_c_max_stake: float = 0.25,
    tier_a_risk_boost: float = 3.0,
    tier_b_risk_boost: float = 1.0,
    tier_c_risk_boost: float = 0.6,
    no_timeout_max_holding_minutes: int = 43200,
    disable_before_third_timeout: bool = False,
    position_scope: str = "symbol",
    use_decision_models: bool = True,
) -> dict[str, Any]:
    if exit_mode not in {"post_management", "fixed_label", "structure_target", "opposite_bsp", "third_then_opposite_bsp", "third_then_opposite_no_after_timeout", FAMILY_ADAPTIVE_EXIT_MODE, REALTIME_FAMILY_EXIT_MODE}:
        raise ValueError("exit_mode must be post_management, fixed_label, structure_target, opposite_bsp, third_then_opposite_bsp, third_then_opposite_no_after_timeout, family_adaptive, or family_realtime")
    position_scope = str(position_scope or "symbol").lower()
    if position_scope not in {"symbol", "global"}:
        raise ValueError("position_scope must be symbol or global")
    structure_target_level = str(structure_target_level).lower()
    if structure_target_level not in STRUCTURE_TARGET_LEVELS:
        raise ValueError(f"structure_target_level must be one of {sorted(STRUCTURE_TARGET_LEVELS)}")
    model_path = Path(model_dir)
    dataset = pd.read_parquet(dataset_path)
    for col in ("exec_time", "entry_time", "exit_time", "signal_available_time"):
        if col in dataset.columns:
            dataset[col] = pd.to_datetime(dataset[col], utc=True, errors="coerce")
    if PRIMARY_LABEL_COLUMN not in dataset.columns:
        raise ValueError(f"dataset missing primary label column: {PRIMARY_LABEL_COLUMN}")
    if exit_mode == "structure_target" and require_origin_zs:
        required_origin_cols = {"chan_bsp2_origin_zs_low", "chan_bsp2_origin_zs_high"}
        missing_origin_cols = sorted(required_origin_cols - set(dataset.columns))
        if missing_origin_cols:
            raise ValueError(
                "structure_target requires rebuilt dataset with actual BSP2 origin Zhongshu fields; "
                f"missing columns: {missing_origin_cols}. Rebuild ML.routes.btc_futures_v2_bsp2_family.features first, "
                "or pass --allow-last-zs-fallback for diagnostic fallback mode."
            )
    dataset = dataset.dropna(subset=[PRIMARY_LABEL_COLUMN, "entry_time", "exit_time"]).sort_values("entry_time").reset_index(drop=True)
    dataset = attach_bsp2_candidate_family(dataset)
    if pure_bsp2_only or exit_mode == "structure_target":
        if "bsp_types_str" not in dataset.columns:
            raise ValueError("pure BSP2 filtering requires column: bsp_types_str")
        dataset = dataset.loc[dataset["bsp_types_str"].astype(str).str.lower().eq("2")].copy()
    else:
        dataset = dataset.loc[dataset["is_bsp2_family_candidate"].astype(float) > 0].copy()
    if require_origin_zs_for_entry:
        origin_cols = {"chan_bsp2_origin_zs_low", "chan_bsp2_origin_zs_high"}
        missing_origin_cols = sorted(origin_cols - set(dataset.columns))
        if missing_origin_cols:
            raise ValueError(f"strict BSP2 backtest requires origin Zhongshu columns, missing: {missing_origin_cols}")
        origin_mask = (
            pd.to_numeric(dataset["chan_bsp2_origin_zs_low"], errors="coerce").notna()
            & pd.to_numeric(dataset["chan_bsp2_origin_zs_high"], errors="coerce").notna()
        )
        dataset = dataset.loc[origin_mask].copy()

    begin = parse_utc(begin_time)
    end = parse_utc(end_time, end_of_day=True)
    if begin is not None:
        dataset = dataset.loc[dataset["entry_time"] >= begin]
    if end is not None:
        dataset = dataset.loc[dataset["entry_time"] <= end]
    if dataset.empty:
        raise ValueError("empty backtest window")

    if use_ml_filter:
        scored = _score_dataset(dataset, model_path)
    else:
        scored = dataset.copy()
        scored["probability"] = np.nan
        scored["threshold"] = np.nan
        scored["ml_model_key"] = "ml_filter_disabled"
    decision_models = load_bsp2_decision_models(model_path) if use_decision_models else None
    if decision_models is not None:
        scored = decision_models.score_frame(scored)
    else:
        scored["bsp2_identity_prob"] = np.nan
        scored["bsp2_pre_invalid_prob"] = np.nan
        scored["bsp2_path_pred"] = ""
        scored["bsp2_path_confidence"] = np.nan
    opposite_exit_events = (
        _build_all_bsp_exit_events_cached(
            dataset,
            begin,
            end,
            tail_days=int(np.ceil(float(no_timeout_max_holding_minutes) / 1440.0)) + 7,
        )
        if exit_mode in OPPOSITE_BSP_EXIT_MODES or exit_mode in {FAMILY_ADAPTIVE_EXIT_MODE, REALTIME_FAMILY_EXIT_MODE}
        else pd.DataFrame(columns=["symbol", "entry_time", "is_buy", "bsp_types_str", "entry_price"])
    )

    qualified_rows = []
    for _, row in scored.iterrows():
        probability = float(row["probability"])
        threshold = float(row["threshold"])
        if use_ml_filter and (not np.isfinite(probability) or not np.isfinite(threshold) or probability < threshold):
            continue
        trade = row.to_dict()
        trade["side"] = "long" if bool(row["is_buy"]) else "short"
        trade["threshold"] = threshold if np.isfinite(threshold) else float("nan")
        trade.update(
            _bsp2_quality_and_sizing(
                row,
                enabled=bool(enable_structure_risk_sizing),
                target_risk_pct=float(target_trade_risk_pct),
                min_stake_multiplier=float(min_stake_multiplier),
                tier_a_max_stake=float(tier_a_max_stake),
                tier_b_max_stake=float(tier_b_max_stake),
                tier_c_max_stake=float(tier_c_max_stake),
                tier_a_risk_boost=float(tier_a_risk_boost),
                tier_b_risk_boost=float(tier_b_risk_boost),
                tier_c_risk_boost=float(tier_c_risk_boost),
                cost_rate=float(structure_target_cost_rate),
            )
        )
        qualified_rows.append(trade)

    qualified = pd.DataFrame(qualified_rows)
    post_bundle = None
    bars_cache: dict[str, pd.DataFrame] = {}
    max_holding_minutes_window = int(
        pd.to_numeric(dataset.get("label_max_holding_minutes", pd.Series([1440])), errors="coerce")
        .dropna()
        .max()
        if "label_max_holding_minutes" in dataset
        else 1440
    )
    if exit_mode == "post_management":
        post_model_file = Path(post_model_path)
        if post_model_policy != "off" and post_model_file.exists():
            post_bundle = load_post_management_model(post_model_file)
            if post_model_policy == "conservative" and hasattr(post_bundle, "thresholds"):
                post_bundle.thresholds.update(
                    {
                        "init_stop": 0.92,
                        "mae_risk": 0.92,
                        "strict_exit": 0.88,
                        "left_mfe_peak": 0.82,
                        "hold_quality": 0.45,
                        "trail_continue": 0.55,
                    }
                )
    l3_models = None
    if exit_mode == "post_management" and str(l3_policy) != "off":
        l3_models = load_l3_hold_models(l3_model_dir)
        if l3_long_threshold is not None and "long" in l3_models:
            l3_models["long"].threshold = float(l3_long_threshold)
        if l3_short_threshold is not None and "short" in l3_models:
            l3_models["short"].threshold = float(l3_short_threshold)

    trades: list[dict[str, Any]] = []
    skipped_trades: list[dict[str, Any]] = []
    equity_rows: list[dict[str, Any]] = [{"time": dataset["entry_time"].min(), "equity": float(initial_cash), "drawdown": 0.0}]
    equity = float(initial_cash)
    min_exit_time = pd.Timestamp.min.tz_localize("UTC")
    last_exit_by_key: dict[str, pd.Timestamp] = {"__global__": min_exit_time}
    window_start = pd.to_datetime(dataset["entry_time"].min(), utc=True)
    window_end = pd.to_datetime(dataset["entry_time"].max(), utc=True)

    if not qualified.empty:
        qualified = qualified.sort_values(["entry_time", "probability"], ascending=[True, False]).reset_index(drop=True)
        for _, row in qualified.iterrows():
            entry_time = pd.to_datetime(row["entry_time"], utc=True)
            exec_time = pd.to_datetime(row["exec_time"], utc=True)
            if entry_time < exec_time:
                raise RuntimeError("lookahead audit failed during backtest")
            symbol_key = normalize_symbol(str(row.get("symbol", SYMBOL) or SYMBOL))
            position_key = "__global__" if position_scope == "global" else symbol_key
            if entry_time <= last_exit_by_key.get(position_key, min_exit_time):
                continue
            leverage = float(row["leverage"])
            multiplier = float(row["stake_multiplier"])
            exposure = max(0.0, float(stake_fraction) * multiplier * leverage)
            trade = dict(row)
            if exit_mode in {"post_management", "structure_target", FAMILY_ADAPTIVE_EXIT_MODE, REALTIME_FAMILY_EXIT_MODE, *OPPOSITE_BSP_EXIT_MODES}:
                symbol = normalize_symbol(str(row.get("symbol", SYMBOL) or SYMBOL))
                if symbol not in bars_cache:
                    source_value = row.get("source_path", "")
                    source_path = Path(str(source_value)) if source_value else futures_1m_path(symbol, DEFAULT_FUTURES_DATA_DIR)
                    if not source_path.exists():
                        source_path = futures_1m_path(symbol, DEFAULT_FUTURES_DATA_DIR)
                    load_start = window_start - pd.Timedelta(minutes=5)
                    tail_minutes = max(max_holding_minutes_window, int(no_timeout_max_holding_minutes) if exit_mode in OPPOSITE_BSP_EXIT_MODES or exit_mode in {FAMILY_ADAPTIVE_EXIT_MODE, REALTIME_FAMILY_EXIT_MODE} else max_holding_minutes_window)
                    load_end = window_end + pd.Timedelta(minutes=tail_minutes + 5)
                    bars_cache[symbol] = load_1m_futures_bars(source_path, begin_time=load_start, end_time=load_end)
                bars_1m = bars_cache.get(symbol, pd.DataFrame())
                if bars_1m.empty:
                    raise RuntimeError(f"{exit_mode} exit requested but 1m bars are unavailable")
            if exit_mode == "post_management":
                trade.update(
                    dynamic_post_exit_for_trade(
                        row,
                        bars_1m,
                        post_bundle,
                        snapshot_minutes=int(snapshot_minutes),
                        min_holding_minutes=int(min_holding_minutes),
                        max_holding_minutes=int(row.get("label_max_holding_minutes", 1440) or 1440),
                        target_pct=float(row.get("label_take_profit_pct", 0.01) or 0.01),
                        emergency_stop_pct=float(emergency_stop_pct),
                        profit_protect_trigger_pct=float(profit_protect_trigger_pct),
                        profit_protect_giveback_ratio=float(profit_protect_giveback_ratio),
                        profit_protect_min_giveback_pct=float(profit_protect_min_giveback_pct),
                        min_model_exit_return=float(min_model_exit_return),
                        use_structure_invalid_stop=bool(use_structure_invalid_stop),
                        l3_models=l3_models,
                        l3_policy=str(l3_policy),
                    )
                )
                exit_time = pd.to_datetime(trade["exit_time"], utc=True)
            elif exit_mode == "structure_target":
                replay = _structure_target_exit_for_trade(
                    row,
                    bars_1m,
                    target_level=structure_target_level,
                    max_holding_minutes=int(row.get("label_max_holding_minutes", 1440) or 1440),
                    cost_rate=float(structure_target_cost_rate),
                    require_origin_zs=bool(require_origin_zs),
                )
                if bool(replay.get("skip_trade", False)):
                    skipped = dict(row)
                    skipped.update(replay)
                    skipped_trades.append(skipped)
                    continue
                trade.update(replay)
                exit_time = pd.to_datetime(trade["exit_time"], utc=True)
            elif exit_mode in OPPOSITE_BSP_EXIT_MODES:
                replay = _opposite_bsp_exit_for_trade(
                    row,
                    opposite_exit_events,
                    bars_1m,
                    max_holding_minutes=int(row.get("label_max_holding_minutes", 1440) or 1440),
                    cost_rate=float(structure_target_cost_rate),
                    use_structure_invalid_stop=bool(use_structure_invalid_stop),
                    require_third_before_opposite=exit_mode in {"third_then_opposite_bsp", "third_then_opposite_no_after_timeout"},
                    no_timeout_after_third=exit_mode == "third_then_opposite_no_after_timeout",
                    no_timeout_before_third=bool(disable_before_third_timeout),
                    no_timeout_max_holding_minutes=int(no_timeout_max_holding_minutes),
                    cost_protect_after_third=exit_mode == "third_then_opposite_no_after_timeout",
                )
                if bool(replay.get("skip_trade", False)):
                    skipped = dict(row)
                    skipped.update(replay)
                    skipped_trades.append(skipped)
                    continue
                trade.update(replay)
                exit_time = pd.to_datetime(trade["exit_time"], utc=True)
            elif exit_mode == FAMILY_ADAPTIVE_EXIT_MODE:
                family_exit_mode = str(row.get("label_bsp2_family_exit_mode", "") or "").lower()
                follow_class = str(row.get("label_bsp2_follow_class", "") or "").lower()
                if family_exit_mode in {"return_zs_target"}:
                    adaptive_level = "t3" if follow_class == "return_origin_zs_full" else "t2"
                    replay = _structure_target_exit_for_trade(
                        row,
                        bars_1m,
                        target_level=adaptive_level,
                        max_holding_minutes=int(row.get("label_max_holding_minutes", 1440) or 1440),
                        cost_rate=float(structure_target_cost_rate),
                        require_origin_zs=False,
                    )
                elif family_exit_mode in {"trend_follow_opposite_bsp"}:
                    replay = _opposite_bsp_exit_for_trade(
                        row,
                        opposite_exit_events,
                        bars_1m,
                        max_holding_minutes=int(no_timeout_max_holding_minutes),
                        cost_rate=float(structure_target_cost_rate),
                        use_structure_invalid_stop=bool(use_structure_invalid_stop),
                        require_third_before_opposite=True,
                        no_timeout_after_third=True,
                        no_timeout_before_third=bool(disable_before_third_timeout),
                        no_timeout_max_holding_minutes=int(no_timeout_max_holding_minutes),
                        cost_protect_after_third=True,
                    )
                elif family_exit_mode in {"higher_level_zs_protect"}:
                    replay = _opposite_bsp_exit_for_trade(
                        row,
                        opposite_exit_events,
                        bars_1m,
                        max_holding_minutes=int(no_timeout_max_holding_minutes),
                        cost_rate=float(structure_target_cost_rate),
                        use_structure_invalid_stop=bool(use_structure_invalid_stop),
                        require_third_before_opposite=True,
                        no_timeout_after_third=True,
                        no_timeout_before_third=bool(disable_before_third_timeout),
                        no_timeout_max_holding_minutes=int(no_timeout_max_holding_minutes),
                        cost_protect_after_third=True,
                    )
                    replay["exit_reference_level"] = "1h"
                else:
                    replay = _invalid_stop_exit_for_trade(
                        row,
                        bars_1m,
                        max_holding_minutes=min(int(row.get("label_max_holding_minutes", 1440) or 1440), 2880),
                        cost_rate=float(structure_target_cost_rate),
                    )
                if bool(replay.get("skip_trade", False)):
                    skipped = dict(row)
                    skipped.update(replay)
                    skipped_trades.append(skipped)
                    continue
                trade.update(replay)
                trade["adaptive_exit_mode"] = family_exit_mode or "invalid_stop"
                trade["adaptive_follow_class"] = follow_class
                exit_time = pd.to_datetime(trade["exit_time"], utc=True)
            elif exit_mode == REALTIME_FAMILY_EXIT_MODE:
                replay = _family_realtime_exit_for_trade(
                    row,
                    opposite_exit_events,
                    bars_1m,
                    max_holding_minutes=int(row.get("label_max_holding_minutes", 1440) or 1440),
                    no_timeout_max_holding_minutes=int(no_timeout_max_holding_minutes),
                    cost_rate=float(structure_target_cost_rate),
                    use_structure_invalid_stop=bool(use_structure_invalid_stop),
                    return_target_level="t2",
                    no_timeout_before_third=bool(disable_before_third_timeout),
                )
                if bool(replay.get("skip_trade", False)):
                    skipped = dict(row)
                    skipped.update(replay)
                    skipped_trades.append(skipped)
                    continue
                trade.update(replay)
                exit_time = pd.to_datetime(trade["exit_time"], utc=True)
            else:
                exit_time = pd.to_datetime(row["exit_time"], utc=True)
            base_return = float(trade["net_return"])
            pnl = equity * exposure * base_return
            equity_before = equity
            equity += pnl
            cooldown = pd.Timedelta(0)
            if exit_mode == "post_management" and str(trade.get("exit_reason", "")).startswith(("post_", "multi_")):
                cooldown = pd.Timedelta(minutes=max(0, int(post_exit_cooldown_minutes)))
            last_exit_by_key[position_key] = exit_time + cooldown
            trade["equity_before"] = float(equity_before)
            trade["exposure"] = float(exposure)
            trade["pnl"] = float(pnl)
            trade["equity_after"] = float(equity)
            trades.append(trade)
            eq = pd.Series([r["equity"] for r in equity_rows] + [equity])
            dd = float(equity / max(eq.max(), equity) - 1.0)
            equity_rows.append({"time": exit_time, "equity": float(equity), "drawdown": dd})

    trades_df = pd.DataFrame(trades)
    equity_df = pd.DataFrame(equity_rows)
    if not equity_df.empty:
        equity_df["time"] = pd.to_datetime(equity_df["time"], utc=True, errors="coerce")
        equity_df = equity_df.sort_values("time")
        equity_series = equity_df.set_index("time")["equity"]
    else:
        equity_series = pd.Series(dtype="float64")

    returns = pd.to_numeric(trades_df["net_return"], errors="coerce") if not trades_df.empty else pd.Series(dtype="float64")
    pnl = pd.to_numeric(trades_df["pnl"], errors="coerce") if not trades_df.empty else pd.Series(dtype="float64")
    wins = int((pnl > 0).sum()) if not trades_df.empty else 0
    backtest_days = max(1.0, float((window_end - window_start).total_seconds() / 86400.0))
    trade_count = int(len(trades_df))
    metrics = {
        "route": "btc_futures_v2_bsp2_family",
        "symbols": sorted(scored["symbol"].dropna().astype(str).unique().tolist()) if "symbol" in scored else [SYMBOL],
        "begin_time": str(begin) if begin is not None else None,
        "end_time": str(end) if end is not None else None,
        "initial_cash": float(initial_cash),
        "final_equity": float(equity),
        "total_return": float(equity / float(initial_cash) - 1.0),
        "trades": trade_count,
        "backtest_days": backtest_days,
        "trades_per_day": float(trade_count / backtest_days),
        "days_per_trade": float(backtest_days / trade_count) if trade_count else float("inf"),
        "candidate_signals": int(len(scored)),
        "qualified_signals": int(len(qualified)) if not qualified.empty else 0,
        "skipped_signals": int(len(skipped_trades)),
        "wins": wins,
        "losses": int(len(trades_df) - wins),
        "win_rate": float(wins / len(trades_df)) if len(trades_df) else float("nan"),
        "avg_signal_return": float(returns.mean()) if len(returns) else float("nan"),
        "avg_trade_pnl": float(pnl.mean()) if len(pnl) else float("nan"),
        "profit_factor": _profit_factor(pnl) if len(pnl) else float("nan"),
        "max_drawdown": _max_drawdown(equity_series),
        "model_design": "BSP2 family quality: expanded BSP2 candidates; family-specific exits; long/short split",
        "exit_mode": str(exit_mode),
        "position_scope": str(position_scope),
        "signal_filter": {
            "pure_bsp2_only": bool(pure_bsp2_only),
            "use_ml_filter": bool(use_ml_filter),
            "require_origin_zs": bool(require_origin_zs) if exit_mode == "structure_target" else False,
            "require_origin_zs_for_entry": bool(require_origin_zs_for_entry),
            "bsp_type_rule": "bsp_types_str == '2'" if pure_bsp2_only else "is_bsp2_family_candidate == 1",
        },
        "decision_models": {
            "enabled": bool(decision_models is not None),
            "requested": bool(use_decision_models),
        },
        "structure_target": {
            "enabled": exit_mode == "structure_target",
            "level": str(structure_target_level),
            "cost_rate": float(structure_target_cost_rate),
            "target_hits": int(trades_df["structure_hit_target"].fillna(False).astype(bool).sum()) if not trades_df.empty and "structure_hit_target" in trades_df else 0,
            "invalid_hits": int(trades_df["structure_hit_invalid"].fillna(False).astype(bool).sum()) if not trades_df.empty and "structure_hit_invalid" in trades_df else 0,
            "timeouts": int((trades_df["exit_reason"].astype(str) == "structure_timeout").sum()) if not trades_df.empty and "exit_reason" in trades_df else 0,
            "skipped": int(len(skipped_trades)),
            "skip_reasons": pd.Series([r.get("skip_reason", "") for r in skipped_trades]).value_counts().to_dict() if skipped_trades else {},
        },
        "opposite_bsp_exit": {
            "enabled": exit_mode in OPPOSITE_BSP_EXIT_MODES or exit_mode in {FAMILY_ADAPTIVE_EXIT_MODE, REALTIME_FAMILY_EXIT_MODE},
            "require_third_before_opposite": exit_mode in {"third_then_opposite_bsp", "third_then_opposite_no_after_timeout"},
            "no_timeout_after_third": exit_mode == "third_then_opposite_no_after_timeout",
            "no_timeout_before_third": bool(disable_before_third_timeout),
            "cost_protect_after_third": exit_mode == "third_then_opposite_no_after_timeout",
            "no_timeout_max_holding_minutes": int(no_timeout_max_holding_minutes),
            "same_level_dataset": "all same-level BSP events: 1/1p/2/2s/3a/3b",
            "event_pool_size": int(len(opposite_exit_events)) if (exit_mode in OPPOSITE_BSP_EXIT_MODES or exit_mode in {FAMILY_ADAPTIVE_EXIT_MODE, REALTIME_FAMILY_EXIT_MODE}) else 0,
            "found": int(trades_df["opposite_bsp_found"].map(bool).sum()) if not trades_df.empty and "opposite_bsp_found" in trades_df else 0,
            "timeouts": int((trades_df["exit_reason"].astype(str) == "opposite_bsp_timeout").sum()) if not trades_df.empty and "exit_reason" in trades_df else 0,
            "open_end": int((trades_df["exit_reason"].astype(str) == "opposite_bsp_open_end").sum()) if not trades_df.empty and "exit_reason" in trades_df else 0,
            "invalid_before_opposite": int((trades_df["exit_reason"].astype(str) == "structure_invalid_before_opposite_bsp").sum()) if not trades_df.empty and "exit_reason" in trades_df else 0,
            "invalid_before_third": int((trades_df["exit_reason"].astype(str) == "structure_invalid_before_third").sum()) if not trades_df.empty and "exit_reason" in trades_df else 0,
            "third_confirmed": int(trades_df["third_confirmed"].fillna(False).astype(bool).sum()) if not trades_df.empty and "third_confirmed" in trades_df else 0,
        },
        "family_adaptive_exit": {
            "enabled": exit_mode in {FAMILY_ADAPTIVE_EXIT_MODE, REALTIME_FAMILY_EXIT_MODE},
            "follow_class_counts": trades_df["adaptive_follow_class"].astype(str).value_counts().to_dict() if not trades_df.empty and "adaptive_follow_class" in trades_df else {},
            "exit_mode_counts": trades_df["adaptive_exit_mode"].astype(str).value_counts().to_dict() if not trades_df.empty and "adaptive_exit_mode" in trades_df else {},
            "candidate_family_counts": trades_df["bsp2_candidate_family"].astype(str).value_counts().to_dict() if not trades_df.empty and "bsp2_candidate_family" in trades_df else {},
        },
        "structure_risk_sizing": {
            "enabled": bool(enable_structure_risk_sizing),
            "target_trade_risk_pct": float(target_trade_risk_pct),
            "min_stake_multiplier": float(min_stake_multiplier),
            "tier_a_max_stake": float(tier_a_max_stake),
            "tier_b_max_stake": float(tier_b_max_stake),
            "tier_c_max_stake": float(tier_c_max_stake),
            "tier_a_risk_boost": float(tier_a_risk_boost),
            "tier_b_risk_boost": float(tier_b_risk_boost),
            "tier_c_risk_boost": float(tier_c_risk_boost),
            "tier_counts": trades_df["confidence_tier"].value_counts().to_dict() if not trades_df.empty and "confidence_tier" in trades_df else {},
            "avg_stake_multiplier": float(pd.to_numeric(trades_df["stake_multiplier"], errors="coerce").mean()) if not trades_df.empty and "stake_multiplier" in trades_df else float("nan"),
        },
        "post_management": {
            "enabled": exit_mode == "post_management",
            "model_path": str(post_model_path) if exit_mode == "post_management" else None,
            "model_loaded": bool(post_bundle is not None) if exit_mode == "post_management" else False,
            "snapshot_minutes": int(snapshot_minutes),
            "min_holding_minutes": int(min_holding_minutes),
            "emergency_stop_pct": float(emergency_stop_pct),
            "profit_protect_trigger_pct": float(profit_protect_trigger_pct),
            "profit_protect_giveback_ratio": float(profit_protect_giveback_ratio),
            "profit_protect_min_giveback_pct": float(profit_protect_min_giveback_pct),
            "min_model_exit_return": float(min_model_exit_return),
            "post_exit_cooldown_minutes": int(post_exit_cooldown_minutes),
            "post_model_policy": str(post_model_policy),
            "use_structure_invalid_stop": bool(use_structure_invalid_stop),
            "l3_policy": str(l3_policy),
            "l3_model_dir": str(l3_model_dir),
            "l3_model_loaded": bool(l3_models is not None),
            "l3_long_threshold": float(getattr(l3_models["long"], "threshold", float("nan"))) if l3_models and "long" in l3_models else None,
            "l3_short_threshold": float(getattr(l3_models["short"], "threshold", float("nan"))) if l3_models and "short" in l3_models else None,
            "fixed_1_to_1_exit_removed": exit_mode == "post_management",
        },
        "lookahead_check": "exec_time is signal availability time; entry_time >= exec_time",
    }

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trades_df.to_csv(out_dir / "executed_trades.csv", index=False)
    if skipped_trades:
        pd.DataFrame(skipped_trades).to_csv(out_dir / "skipped_signals.csv", index=False)
    scored.to_csv(out_dir / "scored_events.csv", index=False)
    equity_df.to_csv(out_dir / "equity_curve.csv", index=False)
    if not trades_df.empty:
        exit_month = pd.to_datetime(trades_df["exit_time"], utc=True).dt.tz_convert(None).dt.to_period("M").astype(str)
        monthly = trades_df.assign(month=exit_month)
        monthly_summary = monthly.groupby("month").agg(
            trades=("pnl", "size"),
            wins=("pnl", lambda s: int((s > 0).sum())),
            pnl=("pnl", "sum"),
            avg_return=("net_return", "mean"),
        )
        monthly_summary["win_rate"] = monthly_summary["wins"] / monthly_summary["trades"]
        monthly_summary.to_csv(out_dir / "monthly_summary.csv")
        if "symbol" in trades_df.columns:
            symbol_summary = trades_df.groupby("symbol").agg(
                trades=("pnl", "size"),
                wins=("pnl", lambda s: int((s > 0).sum())),
                pnl=("pnl", "sum"),
                avg_return=("net_return", "mean"),
            )
            symbol_summary["win_rate"] = symbol_summary["wins"] / symbol_summary["trades"]
            symbol_summary.to_csv(out_dir / "symbol_summary.csv")
    (out_dir / "backtest_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Backtest BTC futures v1 route")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET_PATH))
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--output-dir", default=str(DEFAULT_BACKTEST_DIR))
    parser.add_argument("--begin-time", default="2026-01-01")
    parser.add_argument("--end-time", default="")
    parser.add_argument("--initial-cash", type=float, default=100000.0)
    parser.add_argument("--stake-fraction", type=float, default=1.0)
    parser.add_argument("--exit-mode", default=REALTIME_FAMILY_EXIT_MODE, choices=["post_management", "fixed_label", "structure_target", "opposite_bsp", "third_then_opposite_bsp", "third_then_opposite_no_after_timeout", FAMILY_ADAPTIVE_EXIT_MODE, REALTIME_FAMILY_EXIT_MODE])
    parser.add_argument("--post-model-path", default=str(DEFAULT_POST_MODEL_PATH))
    parser.add_argument("--snapshot-minutes", type=int, default=5)
    parser.add_argument("--min-holding-minutes", type=int, default=15)
    parser.add_argument("--emergency-stop-pct", type=float, default=0.012)
    parser.add_argument("--profit-protect-trigger-pct", type=float, default=0.016)
    parser.add_argument("--profit-protect-giveback-ratio", type=float, default=0.35)
    parser.add_argument("--profit-protect-min-giveback-pct", type=float, default=0.006)
    parser.add_argument("--min-model-exit-return", type=float, default=-0.006)
    parser.add_argument("--post-exit-cooldown-minutes", type=int, default=0)
    parser.add_argument("--post-model-policy", default="off", choices=["off", "conservative", "raw"])
    parser.add_argument("--disable-structure-invalid-stop", action="store_true")
    parser.add_argument("--l3-model-dir", default=str(DEFAULT_L3_MODEL_DIR))
    parser.add_argument("--l3-policy", default="off", choices=["off", "raw", "conservative"])
    parser.add_argument("--l3-long-threshold", type=float, default=None)
    parser.add_argument("--l3-short-threshold", type=float, default=None)
    parser.add_argument("--structure-target-level", default="third", choices=sorted(STRUCTURE_TARGET_LEVELS))
    parser.add_argument("--structure-target-cost-rate", type=float, default=0.001)
    parser.add_argument("--pure-bsp2-only", action="store_true")
    parser.add_argument("--include-bsp2s-family", action="store_true")
    parser.add_argument("--enable-ml-filter", action="store_true")
    parser.add_argument("--disable-ml-filter", action="store_true")
    parser.add_argument("--allow-last-zs-fallback", action="store_true")
    parser.add_argument("--allow-missing-origin-zs-entry", action="store_true")
    parser.add_argument("--enable-structure-risk-sizing", action="store_true")
    parser.add_argument("--disable-structure-risk-sizing", action="store_true")
    parser.add_argument("--target-trade-risk-pct", type=float, default=0.01)
    parser.add_argument("--min-stake-multiplier", type=float, default=0.08)
    parser.add_argument("--tier-a-max-stake", type=float, default=1.0)
    parser.add_argument("--tier-b-max-stake", type=float, default=0.55)
    parser.add_argument("--tier-c-max-stake", type=float, default=0.25)
    parser.add_argument("--tier-a-risk-boost", type=float, default=3.0)
    parser.add_argument("--tier-b-risk-boost", type=float, default=1.0)
    parser.add_argument("--tier-c-risk-boost", type=float, default=0.6)
    parser.add_argument("--no-timeout-max-holding-minutes", type=int, default=43200)
    parser.add_argument("--disable-before-third-timeout", action="store_true")
    parser.add_argument("--position-scope", default="symbol", choices=["symbol", "global"])
    parser.add_argument("--disable-decision-models", action="store_true")
    args = parser.parse_args()

    metrics = run_btc_futures_backtest(
        dataset_path=args.dataset,
        model_dir=args.model_dir,
        output_dir=args.output_dir,
        begin_time=args.begin_time,
        end_time=args.end_time or None,
        initial_cash=args.initial_cash,
        stake_fraction=args.stake_fraction,
        exit_mode=args.exit_mode,
        post_model_path=args.post_model_path,
        snapshot_minutes=args.snapshot_minutes,
        min_holding_minutes=args.min_holding_minutes,
        emergency_stop_pct=args.emergency_stop_pct,
        profit_protect_trigger_pct=args.profit_protect_trigger_pct,
        profit_protect_giveback_ratio=args.profit_protect_giveback_ratio,
        profit_protect_min_giveback_pct=args.profit_protect_min_giveback_pct,
        min_model_exit_return=args.min_model_exit_return,
        post_exit_cooldown_minutes=args.post_exit_cooldown_minutes,
        post_model_policy=args.post_model_policy,
        use_structure_invalid_stop=not args.disable_structure_invalid_stop,
        l3_model_dir=args.l3_model_dir,
        l3_policy=args.l3_policy,
        l3_long_threshold=args.l3_long_threshold,
        l3_short_threshold=args.l3_short_threshold,
        structure_target_level=args.structure_target_level,
        structure_target_cost_rate=args.structure_target_cost_rate,
        pure_bsp2_only=bool(args.pure_bsp2_only),
        use_ml_filter=bool(args.enable_ml_filter and not args.disable_ml_filter),
        require_origin_zs=not args.allow_last_zs_fallback,
        require_origin_zs_for_entry=not args.allow_missing_origin_zs_entry,
        enable_structure_risk_sizing=bool((args.enable_structure_risk_sizing or not args.disable_structure_risk_sizing) and not args.disable_structure_risk_sizing),
        target_trade_risk_pct=args.target_trade_risk_pct,
        min_stake_multiplier=args.min_stake_multiplier,
        tier_a_max_stake=args.tier_a_max_stake,
        tier_b_max_stake=args.tier_b_max_stake,
        tier_c_max_stake=args.tier_c_max_stake,
        tier_a_risk_boost=args.tier_a_risk_boost,
        tier_b_risk_boost=args.tier_b_risk_boost,
        tier_c_risk_boost=args.tier_c_risk_boost,
        no_timeout_max_holding_minutes=args.no_timeout_max_holding_minutes,
        disable_before_third_timeout=bool(args.disable_before_third_timeout),
        position_scope=args.position_scope,
        use_decision_models=not args.disable_decision_models,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

