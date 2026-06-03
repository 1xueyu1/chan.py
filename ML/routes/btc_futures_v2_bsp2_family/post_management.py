from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ML.model import ModelBundle, train_classifier
from ML.routes.btc_futures_v3_beta_edge.exit_management import (
    DEFAULT_MULTI_HEAD_THRESHOLDS,
    MultiHeadExitModelBundle,
    _snapshot_features,
    build_strict_exit_lifecycle_dataset,
    train_multi_head_exit_model,
)
from ML.shared.json_utils import json_safe

from .data import (
    DEFAULT_DATASET_PATH,
    DEFAULT_FUTURES_DATA_DIR,
    DEFAULT_MODEL_DIR,
    DEFAULT_SOURCE_PATH,
    futures_1m_path,
    load_1m_futures_bars,
    normalize_symbol,
)


DEFAULT_POST_DATASET_PATH = Path("data/btc_futures_v2_bsp2_family/btc_futures_v2_bsp2_family_post_lifecycle.parquet")
DEFAULT_POST_MODEL_PATH = DEFAULT_MODEL_DIR / "post_management_multi_head_model.pkl"
DEFAULT_POST_MODEL_DIR = DEFAULT_MODEL_DIR / "post_management"
DEFAULT_L3_DATASET_PATH = Path("data/btc_futures_v2_bsp2_family/btc_futures_v2_bsp2_family_l3_hold_snapshots.parquet")
DEFAULT_L3_MODEL_DIR = DEFAULT_MODEL_DIR / "l3_hold"
DEFAULT_L3_LONG_MODEL_PATH = DEFAULT_L3_MODEL_DIR / "long_hold_model.pkl"
DEFAULT_L3_SHORT_MODEL_PATH = DEFAULT_L3_MODEL_DIR / "short_hold_model.pkl"

L3_FEATURE_COLUMNS = [
    "current_return",
    "mfe_so_far",
    "mae_so_far",
    "drawdown_from_mfe",
    "holding_minutes",
    "return_per_minute",
    "mfe_mae_ratio",
    "dist_to_invalid",
    "dist_to_zs_low",
    "dist_to_zs_mid",
    "dist_to_zs_high",
    "inside_prev_zs",
    "target_level_reached",
    "post_ret_5m",
    "post_ret_15m",
    "post_ret_30m",
    "post_trend_efficiency",
    "post_volume_ratio",
    "post_range_pct",
    "entry_probability",
    "entry_threshold_gap",
    "asset_is_btc",
    "asset_is_eth",
    "asset_is_sol",
    "asset_is_sui",
    "asset_is_hype",
    "asset_is_ada",
]


def _safe_num(value: Any, default: float = np.nan) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if np.isfinite(out) else float(default)


def _signed_distance(direction: float, price: float, ref: Any) -> float:
    ref_price = _safe_num(ref)
    if not np.isfinite(ref_price) or not np.isfinite(price) or price <= 0:
        return float("nan")
    return float(direction * (ref_price - price) / (price + 1e-12))


def _l3_snapshot_features(
    row: dict[str, Any] | pd.Series,
    bars: pd.DataFrame,
    start_pos: int,
    pos: int,
    *,
    cost_rate: float = 0.001,
) -> dict[str, Any]:
    item = row.to_dict() if isinstance(row, pd.Series) else dict(row)
    entry_time = pd.to_datetime(item["entry_time"], utc=True)
    entry_price = float(item["entry_price"])
    is_buy = bool(item.get("is_buy", str(item.get("side", "")).lower() == "long"))
    direction = 1.0 if is_buy else -1.0
    now = pd.Timestamp(bars.index[pos])
    close = float(bars["close"].iloc[pos])
    high_so_far = float(bars["high"].iloc[start_pos : pos + 1].max())
    low_so_far = float(bars["low"].iloc[start_pos : pos + 1].min())
    current_return = float(direction * (close - entry_price) / (entry_price + 1e-12) - cost_rate)
    if direction > 0:
        mfe = float((high_so_far - entry_price) / (entry_price + 1e-12))
        mae = float((low_so_far - entry_price) / (entry_price + 1e-12))
    else:
        mfe = float((entry_price - low_so_far) / (entry_price + 1e-12))
        mae = float((entry_price - high_so_far) / (entry_price + 1e-12))
    drawdown = max(0.0, mfe - (current_return + cost_rate))
    holding_minutes = max(1.0, float((now - entry_time).total_seconds() / 60.0))
    hist = bars.iloc[start_pos : pos + 1]
    first_close = float(hist["close"].iloc[0])
    denom = abs(float(hist["close"].iloc[-1]) - first_close)
    path = float(hist["close"].diff().abs().sum())
    trend_eff = float(denom / (path + 1e-12))
    vol_now = float(hist["volume"].tail(5).mean()) if "volume" in hist else 0.0
    vol_base = float(hist["volume"].head(max(1, min(len(hist), 20))).mean()) if "volume" in hist else 0.0
    zs_low = _safe_num(item.get("label_bsp2_prev_zs_low"))
    zs_mid = _safe_num(item.get("label_bsp2_prev_zs_mid"))
    zs_high = _safe_num(item.get("label_bsp2_prev_zs_high"))
    target_level = 0.0
    if direction > 0:
        if np.isfinite(zs_low) and high_so_far >= zs_low:
            target_level = max(target_level, 1.0)
        if np.isfinite(zs_mid) and high_so_far >= zs_mid:
            target_level = max(target_level, 2.0)
        if np.isfinite(zs_high) and high_so_far >= zs_high:
            target_level = max(target_level, 3.0)
    else:
        if np.isfinite(zs_high) and low_so_far <= zs_high:
            target_level = max(target_level, 1.0)
        if np.isfinite(zs_mid) and low_so_far <= zs_mid:
            target_level = max(target_level, 2.0)
        if np.isfinite(zs_low) and low_so_far <= zs_low:
            target_level = max(target_level, 3.0)
    def ret_back(minutes: int) -> float:
        back_pos = max(start_pos, pos - int(minutes))
        prev = float(bars["close"].iloc[back_pos])
        return float(direction * (close - prev) / (prev + 1e-12))

    symbol = str(item.get("symbol", "")).upper()
    asset = symbol.replace("USDT", "").lower()
    return {
        "symbol": symbol,
        "is_buy": bool(is_buy),
        "snapshot_time": now,
        "entry_time": entry_time,
        "current_return": current_return,
        "mfe_so_far": mfe,
        "mae_so_far": mae,
        "drawdown_from_mfe": drawdown,
        "holding_minutes": holding_minutes,
        "return_per_minute": current_return / holding_minutes,
        "mfe_mae_ratio": mfe / (abs(mae) + 1e-12),
        "dist_to_invalid": _signed_distance(-direction, close, item.get("label_bsp2_invalid_price")),
        "dist_to_zs_low": _signed_distance(direction, close, zs_low),
        "dist_to_zs_mid": _signed_distance(direction, close, zs_mid),
        "dist_to_zs_high": _signed_distance(direction, close, zs_high),
        "inside_prev_zs": float(np.isfinite(zs_low) and np.isfinite(zs_high) and zs_low <= close <= zs_high),
        "target_level_reached": target_level,
        "post_ret_5m": ret_back(5),
        "post_ret_15m": ret_back(15),
        "post_ret_30m": ret_back(30),
        "post_trend_efficiency": trend_eff,
        "post_volume_ratio": vol_now / (vol_base + 1e-12),
        "post_range_pct": float((hist["high"].max() - hist["low"].min()) / (entry_price + 1e-12)),
        "entry_probability": _safe_num(item.get("probability", np.nan), 0.5),
        "entry_threshold_gap": _safe_num(item.get("probability", np.nan), 0.5) - _safe_num(item.get("threshold", np.nan), 0.5),
        "asset_is_btc": float(asset == "btc"),
        "asset_is_eth": float(asset == "eth"),
        "asset_is_sol": float(asset == "sol"),
        "asset_is_sui": float(asset == "sui"),
        "asset_is_hype": float(asset == "hype"),
        "asset_is_ada": float(asset == "ada"),
    }


def build_post_management_dataset(
    trades_path: str | Path = "result/btc_futures_v2_bsp2_family/executed_trades.csv",
    source_path: str | Path = DEFAULT_SOURCE_PATH,
    output_path: str | Path = DEFAULT_POST_DATASET_PATH,
    snapshot_minutes: int = 5,
    min_holding_minutes: int = 15,
    force: bool = False,
) -> pd.DataFrame:
    """Build lifecycle snapshots for the v1 post-entry management model.

    The snapshot features use only bars from entry_time through snapshot_time.
    The labels use the later trade outcome, so this dataset is for supervised
    training only and must not be used as live features.
    """
    dataset = build_strict_exit_lifecycle_dataset(
        trades_path=trades_path,
        source_path=source_path,
        output_path=output_path,
        snapshot_minutes=snapshot_minutes,
        min_holding_minutes=min_holding_minutes,
        force=force,
    )
    meta_path = Path(output_path).with_name(Path(output_path).stem + "_v1_meta.json")
    meta = {
        "route": "btc_futures_v2_bsp2_family",
        "rows": int(len(dataset)),
        "trades": int(dataset["trade_idx"].nunique()) if "trade_idx" in dataset else 0,
        "snapshot_minutes": int(snapshot_minutes),
        "min_holding_minutes": int(min_holding_minutes),
        "lookahead_rule": "features use entry_time..snapshot_time only; labels use future outcome for training",
    }
    meta_path.write_text(json.dumps(json_safe(meta), ensure_ascii=False, indent=2), encoding="utf-8")
    return dataset


def train_post_management_model(
    dataset_path: str | Path = DEFAULT_POST_DATASET_PATH,
    model_path: str | Path = DEFAULT_POST_MODEL_PATH,
    train_end: Any = "2025-01-01",
    valid_start: Any = "2025-01-01",
    min_holding_minutes: int = 15,
) -> dict[str, Any]:
    metrics = train_multi_head_exit_model(
        dataset_path=dataset_path,
        model_path=model_path,
        train_end=train_end,
        valid_start=valid_start,
        min_holding_minutes=min_holding_minutes,
    )
    metrics["route"] = "btc_futures_v2_bsp2_family"
    metrics["exit_design"] = "post-entry multi-head management; no fixed 1:1 TP exit in replay"
    Path(model_path).with_name("post_management_multi_head_metrics.json").write_text(
        json.dumps(json_safe(metrics), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return metrics


def load_post_management_model(model_path: str | Path = DEFAULT_POST_MODEL_PATH) -> MultiHeadExitModelBundle:
    return MultiHeadExitModelBundle.load(model_path)


def build_l3_hold_dataset(
    dataset_path: str | Path = DEFAULT_DATASET_PATH,
    output_path: str | Path = DEFAULT_L3_DATASET_PATH,
    *,
    begin_time: Any | None = None,
    end_time: Any | None = None,
    snapshot_minutes: int = 5,
    min_holding_minutes: int = 15,
    max_snapshot_minutes: int = 360,
    min_extra_profit: float = 0.004,
    max_extra_risk: float = 0.012,
    force: bool = False,
) -> pd.DataFrame:
    output = Path(output_path)
    if output.exists() and not force:
        return pd.read_parquet(output)

    entries = pd.read_parquet(dataset_path)
    for col in ("entry_time", "exec_time"):
        if col in entries:
            entries[col] = pd.to_datetime(entries[col], utc=True, errors="coerce")
    entries = entries.dropna(subset=["entry_time", "entry_price"]).copy()
    if begin_time is not None:
        entries = entries.loc[entries["entry_time"] >= pd.to_datetime(begin_time, utc=True)]
    if end_time is not None:
        entries = entries.loc[entries["entry_time"] < pd.to_datetime(end_time, utc=True)]
    if "bsp_types_str" in entries:
        entries = entries.loc[entries["bsp_types_str"].astype(str).str.lower().str.startswith("2")]

    rows: list[dict[str, Any]] = []
    bars_cache: dict[str, pd.DataFrame] = {}
    for _, entry in entries.iterrows():
        symbol = normalize_symbol(str(entry.get("symbol", "BTCUSDT") or "BTCUSDT"))
        if symbol not in bars_cache:
            source_path = futures_1m_path(symbol, DEFAULT_FUTURES_DATA_DIR)
            if not source_path.exists() and symbol == "BTCUSDT":
                source_path = DEFAULT_SOURCE_PATH
            bars_cache[symbol] = load_1m_futures_bars(source_path)
        bars = bars_cache[symbol]
        if bars.empty:
            continue
        entry_time = pd.to_datetime(entry["entry_time"], utc=True)
        start_pos, _ = _bar_at_or_before(bars, entry_time)
        if start_pos is None:
            continue
        entry_price = float(entry["entry_price"])
        is_buy = bool(entry.get("is_buy", True))
        direction = 1.0 if is_buy else -1.0
        max_hold = min(int(max_snapshot_minutes), int(entry.get("label_max_holding_minutes", max_snapshot_minutes) or max_snapshot_minutes))
        end_time = entry_time + pd.Timedelta(minutes=max_hold)
        end_pos = int(np.searchsorted(bars.index.view("int64"), end_time.value, side="right") - 1)
        end_pos = min(max(start_pos + 1, end_pos), len(bars) - 1)
        if end_pos <= start_pos:
            continue
        for pos in range(start_pos + int(min_holding_minutes), end_pos + 1, int(snapshot_minutes)):
            if pos <= start_pos or pos >= len(bars):
                continue
            feat = _l3_snapshot_features(entry, bars, start_pos, pos)
            close_now = float(bars["close"].iloc[pos])
            current_net = float(direction * (close_now - entry_price) / (entry_price + 1e-12) - 0.001)
            future = bars.iloc[pos + 1 : end_pos + 1]
            if future.empty:
                continue
            if direction > 0:
                future_best = float((future["high"].max() - entry_price) / (entry_price + 1e-12) - 0.001)
                future_worst = float((future["low"].min() - close_now) / (close_now + 1e-12))
            else:
                future_best = float((entry_price - future["low"].min()) / (entry_price + 1e-12) - 0.001)
                future_worst = float((close_now - future["high"].max()) / (close_now + 1e-12))
            label = float((future_best - current_net >= float(min_extra_profit)) and (future_worst >= -abs(float(max_extra_risk))))
            feat.update(
                {
                    "label": label,
                    "future_best_net_return": future_best,
                    "future_extra_return": future_best - current_net,
                    "future_worst_drawdown": future_worst,
                }
            )
            rows.append(feat)

    result = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    output.with_name(output.stem + "_meta.json").write_text(
        json.dumps(
            json_safe(
                {
                    "route": "btc_futures_v2_bsp2_family",
                    "dataset": "l3_hold_snapshots",
                    "rows": int(len(result)),
                    "snapshot_minutes": int(snapshot_minutes),
                    "min_holding_minutes": int(min_holding_minutes),
                    "max_snapshot_minutes": int(max_snapshot_minutes),
                    "label": "continue holding if future extra return clears risk-adjusted hurdle",
                }
            ),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return result


def train_l3_hold_models(
    dataset_path: str | Path = DEFAULT_L3_DATASET_PATH,
    model_dir: str | Path = DEFAULT_L3_MODEL_DIR,
    *,
    valid_start: Any = "2025-01-01",
    model_kind: str = "auto",
    long_threshold: float = 0.42,
    short_threshold: float = 0.42,
) -> dict[str, Any]:
    data = pd.read_parquet(dataset_path)
    data["snapshot_time"] = pd.to_datetime(data["snapshot_time"], utc=True, errors="coerce")
    data = data.dropna(subset=["snapshot_time", "label"]).copy()
    valid_ts = pd.to_datetime(valid_start, utc=True)
    train = data.loc[data["snapshot_time"] < valid_ts].copy()
    valid = data.loc[data["snapshot_time"] >= valid_ts].copy()
    out_dir = Path(model_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics: dict[str, Any] = {"route": "btc_futures_v2_bsp2_family", "model": "l3_hold", "rows": int(len(data)), "sides": {}}
    for side, is_buy, threshold in [("long", True, long_threshold), ("short", False, short_threshold)]:
        side_train = train.loc[train["is_buy"].astype(bool) == is_buy].copy()
        side_valid = valid.loc[valid["is_buy"].astype(bool) == is_buy].copy()
        bundle = train_classifier(
            side_train,
            feature_columns=L3_FEATURE_COLUMNS,
            model_kind=model_kind,
            threshold=float(threshold),
            side="both",
            random_state=4300 + (0 if is_buy else 1),
        )
        path = out_dir / f"{side}_hold_model.pkl"
        bundle.save(path)
        valid_prob = bundle.predict_proba(side_valid) if not side_valid.empty else np.array([], dtype="float64")
        metrics["sides"][side] = {
            "model_path": str(path),
            "threshold": float(threshold),
            "train_rows": int(len(side_train)),
            "valid_rows": int(len(side_valid)),
            "train_positive_rate": float(side_train["label"].mean()) if len(side_train) else float("nan"),
            "valid_positive_rate": float(side_valid["label"].mean()) if len(side_valid) else float("nan"),
            "valid_exit_rate": float((valid_prob < float(threshold)).mean()) if len(valid_prob) else float("nan"),
        }
    (out_dir / "metrics.json").write_text(json.dumps(json_safe(metrics), ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def load_l3_hold_models(model_dir: str | Path = DEFAULT_L3_MODEL_DIR) -> dict[str, ModelBundle]:
    root = Path(model_dir)
    return {
        "long": ModelBundle.load(root / "long_hold_model.pkl"),
        "short": ModelBundle.load(root / "short_hold_model.pkl"),
    }


def _bar_at_or_before(bars: pd.DataFrame, ts: pd.Timestamp) -> tuple[int, pd.Timestamp] | tuple[None, None]:
    pos = int(np.searchsorted(bars.index.view("int64"), pd.Timestamp(ts).value, side="right") - 1)
    if pos < 0 or pos >= len(bars):
        return None, None
    return pos, pd.Timestamp(bars.index[pos])


def dynamic_post_exit_for_trade(
    trade: dict[str, Any] | pd.Series,
    bars_1m: pd.DataFrame,
    bundle: MultiHeadExitModelBundle | None,
    *,
    snapshot_minutes: int = 5,
    min_holding_minutes: int = 15,
    max_holding_minutes: int = 1440,
    target_pct: float = 0.01,
    cost_rate: float = 0.001,
    emergency_stop_pct: float = 0.012,
    profit_protect_trigger_pct: float = 0.016,
    profit_protect_giveback_ratio: float = 0.35,
    profit_protect_min_giveback_pct: float = 0.006,
    min_model_exit_return: float = -0.006,
    multi_policy: str = "profit_protect",
    use_structure_invalid_stop: bool = True,
    l3_models: dict[str, ModelBundle] | None = None,
    l3_policy: str = "off",
) -> dict[str, Any]:
    """Replay one trade with dynamic post-entry exit management.

    This intentionally does not use fixed TP/SL as the terminal exit. The
    terminal fallback is time_stop, while earlier exits come from hard risk,
    profit protection, or the multi-head post-management model.
    """
    row = trade.to_dict() if isinstance(trade, pd.Series) else dict(trade)
    entry_time = pd.to_datetime(row["entry_time"], utc=True)
    entry_price = float(row["entry_price"])
    is_buy = bool(row.get("is_buy", str(row.get("side", "")).lower() == "long"))
    direction = 1.0 if is_buy else -1.0
    structure_invalid_price = pd.to_numeric(
        pd.Series([row.get("label_bsp2_invalid_price", np.nan)]),
        errors="coerce",
    ).iloc[0]
    structure_invalid_enabled = bool(use_structure_invalid_stop) and np.isfinite(structure_invalid_price)
    max_hold = max(1, int(row.get("label_max_holding_minutes", max_holding_minutes) or max_holding_minutes))
    target = max(float(row.get("label_take_profit_pct", target_pct) or target_pct), 1e-6)
    start_pos, actual_entry = _bar_at_or_before(bars_1m, entry_time)
    if start_pos is None:
        raise ValueError(f"entry_time is outside bars: {entry_time}")

    end_time = entry_time + pd.Timedelta(minutes=max_hold)
    end_pos = int(np.searchsorted(bars_1m.index.view("int64"), end_time.value, side="right") - 1)
    end_pos = min(max(start_pos, end_pos), len(bars_1m) - 1)
    if end_pos <= start_pos:
        end_pos = min(len(bars_1m) - 1, start_pos + 1)

    chosen_pos = end_pos
    chosen_reason = "time_stop"
    chosen_exit_price: float | None = None
    chosen_model_probs: dict[str, float] = {}
    l3_hold_probability: float | None = None
    next_snapshot = entry_time + pd.Timedelta(minutes=max(int(min_holding_minutes), int(snapshot_minutes)))

    for pos in range(start_pos + 1, end_pos + 1):
        ts = pd.Timestamp(bars_1m.index[pos])
        high_price = float(bars_1m["high"].iloc[pos])
        low_price = float(bars_1m["low"].iloc[pos])
        close_price = float(bars_1m["close"].iloc[pos])
        gross = direction * (close_price - entry_price) / (entry_price + 1e-12)
        net = gross - float(cost_rate)
        hist = bars_1m.iloc[start_pos : pos + 1]
        if direction > 0:
            mfe_so_far = max(0.0, float((hist["high"].max() - entry_price) / (entry_price + 1e-12)))
        else:
            mfe_so_far = max(0.0, float((entry_price - hist["low"].min()) / (entry_price + 1e-12)))
        drawdown_from_mfe = max(0.0, mfe_so_far - gross)

        if structure_invalid_enabled:
            if (direction > 0 and low_price <= float(structure_invalid_price)) or (
                direction < 0 and high_price >= float(structure_invalid_price)
            ):
                chosen_pos = pos
                chosen_reason = "structure_invalid_stop"
                chosen_exit_price = float(structure_invalid_price)
                break

        snapshot_due = ts >= next_snapshot
        if snapshot_due:
            if l3_models is not None and str(l3_policy) != "off":
                side_key = "long" if is_buy else "short"
                l3_model = l3_models.get(side_key)
                if l3_model is not None:
                    feat = pd.DataFrame([_l3_snapshot_features(row, bars_1m, start_pos, pos, cost_rate=cost_rate)])
                    l3_hold_probability = float(l3_model.predict_proba(feat)[0])
                    threshold = float(getattr(l3_model, "threshold", 0.5))
                    if str(l3_policy) == "conservative" and net < 0:
                        threshold = max(0.35, threshold - 0.05)
                    if l3_hold_probability < threshold:
                        chosen_pos = pos
                        chosen_reason = "l3_strict_exit"
                        break

        if net <= -abs(float(emergency_stop_pct)):
            chosen_pos = pos
            chosen_reason = "post_emergency_stop"
            break
        if (
            mfe_so_far >= max(float(profit_protect_trigger_pct), target * 1.20)
            and drawdown_from_mfe >= max(float(profit_protect_min_giveback_pct), mfe_so_far * float(profit_protect_giveback_ratio))
        ):
            chosen_pos = pos
            chosen_reason = "post_profit_protect"
            break

        if bundle is not None and snapshot_due:
            feat = _snapshot_features(
                bars=bars_1m,
                direction=direction,
                entry_price=entry_price,
                entry_time=entry_time,
                snapshot_time=ts,
                target_pct=target,
                max_holding_minutes=max_hold,
            )
            frame = pd.DataFrame([feat])
            should_exit, reason, probs = bundle.should_exit(
                frame,
                current_exit_return=float(net),
                min_model_exit_return=float(min_model_exit_return),
                policy=str(multi_policy),
            )
            if should_exit:
                chosen_pos = pos
                chosen_reason = reason
                chosen_model_probs = {str(k): float(v) for k, v in probs.items()}
                break
        if snapshot_due:
            next_snapshot = ts + pd.Timedelta(minutes=int(snapshot_minutes))

    exit_time = pd.Timestamp(bars_1m.index[chosen_pos])
    exit_price = float(chosen_exit_price) if chosen_exit_price is not None else float(bars_1m["close"].iloc[chosen_pos])
    window = bars_1m.iloc[start_pos : chosen_pos + 1]
    if direction > 0:
        mfe = float((window["high"].max() - entry_price) / (entry_price + 1e-12))
        mae = float((window["low"].min() - entry_price) / (entry_price + 1e-12))
    else:
        mfe = float((entry_price - window["low"].min()) / (entry_price + 1e-12))
        mae = float((entry_price - window["high"].max()) / (entry_price + 1e-12))
    gross_return = float(direction * (exit_price - entry_price) / (entry_price + 1e-12))
    net_return = float(gross_return - float(cost_rate))
    out = {
        "exit_time": exit_time,
        "exit_price": exit_price,
        "exit_reason": chosen_reason,
        "gross_return": gross_return,
        "net_return": net_return,
        "mfe": mfe,
        "mae": mae,
        "holding_minutes": float((exit_time - entry_time).total_seconds() / 60.0),
        "post_model_used": bool(chosen_reason.startswith("multi_")),
        "l3_hold_probability": float(l3_hold_probability) if l3_hold_probability is not None else np.nan,
        "l3_policy": str(l3_policy),
    }
    for head, prob in chosen_model_probs.items():
        out[f"{head}_probability"] = float(prob)
    return out


def load_bars_for_trades(
    trades: pd.DataFrame,
    source_path: str | Path = DEFAULT_SOURCE_PATH,
    max_holding_minutes: int = 1440,
) -> pd.DataFrame:
    if trades.empty:
        return pd.DataFrame()
    entry = pd.to_datetime(trades["entry_time"], utc=True, errors="coerce")
    begin = entry.min() - pd.Timedelta(minutes=5)
    end = entry.max() + pd.Timedelta(minutes=int(max_holding_minutes) + 5)
    return load_1m_futures_bars(source_path, begin_time=begin, end_time=end)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="BTC futures v1 post-entry management")
    parser.add_argument("--action", choices=["build", "train", "all", "build-l3", "train-l3", "l3-all"], default="all")
    parser.add_argument("--trades", default="result/btc_futures_v2_bsp2_family/executed_trades.csv")
    parser.add_argument("--dataset", default=str(DEFAULT_POST_DATASET_PATH))
    parser.add_argument("--model", default=str(DEFAULT_POST_MODEL_PATH))
    parser.add_argument("--l3-dataset", default=str(DEFAULT_L3_DATASET_PATH))
    parser.add_argument("--l3-model-dir", default=str(DEFAULT_L3_MODEL_DIR))
    parser.add_argument("--snapshot-minutes", type=int, default=5)
    parser.add_argument("--min-holding-minutes", type=int, default=15)
    parser.add_argument("--max-snapshot-minutes", type=int, default=360)
    parser.add_argument("--begin-time", default="")
    parser.add_argument("--end-time", default="")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    result: dict[str, Any] = {}
    if args.action in {"build", "all"}:
        ds = build_post_management_dataset(
            trades_path=args.trades,
            output_path=args.dataset,
            snapshot_minutes=args.snapshot_minutes,
            min_holding_minutes=args.min_holding_minutes,
            force=args.force,
        )
        result["build"] = {"rows": int(len(ds)), "trades": int(ds["trade_idx"].nunique())}
    if args.action in {"train", "all"}:
        result["train"] = train_post_management_model(
            dataset_path=args.dataset,
            model_path=args.model,
            min_holding_minutes=args.min_holding_minutes,
        )
    if args.action in {"build-l3", "l3-all"}:
        ds = build_l3_hold_dataset(
            output_path=args.l3_dataset,
            begin_time=args.begin_time or None,
            end_time=args.end_time or None,
            snapshot_minutes=args.snapshot_minutes,
            min_holding_minutes=args.min_holding_minutes,
            max_snapshot_minutes=args.max_snapshot_minutes,
            force=args.force,
        )
        result["build_l3"] = {"rows": int(len(ds))}
    if args.action in {"train-l3", "l3-all"}:
        result["train_l3"] = train_l3_hold_models(dataset_path=args.l3_dataset, model_dir=args.l3_model_dir)
    print(json.dumps(json_safe(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

