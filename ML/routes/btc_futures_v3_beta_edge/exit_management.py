from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ML.routes.btc_futures_v1.data import DEFAULT_SOURCE_PATH, load_1m_futures_bars, parse_utc
from ML.shared.json_utils import json_safe

from .dataset import DEFAULT_BACKTEST_DIR, DEFAULT_MODEL_DIR, ROUTE_NAME


DEFAULT_EXIT_DATASET_PATH = Path("data/btc_futures_v3_beta_edge/btc_futures_v3_beta_exit_lifecycle.parquet")
DEFAULT_EXIT_MODEL_PATH = DEFAULT_MODEL_DIR / "strict_exit_model.pkl"
DEFAULT_MULTI_HEAD_MODEL_PATH = DEFAULT_MODEL_DIR / "multi_head_exit_model.pkl"
DEFAULT_EXIT_BACKTEST_DIR = DEFAULT_BACKTEST_DIR / "ml_strict_exit"
DEFAULT_TRAIN_TRADES_PATH = DEFAULT_BACKTEST_DIR / "frequency_floor_2024_2025_scan" / "executed_trades.csv"
DEFAULT_TEST_TRADES_PATH = DEFAULT_BACKTEST_DIR / "frequency_floor_2day" / "executed_trades.csv"


EXIT_FEATURE_COLUMNS = [
    "life_elapsed_minutes",
    "life_elapsed_ratio",
    "life_unrealized_return",
    "life_unrealized_vs_target",
    "life_mfe_so_far",
    "life_mae_so_far",
    "life_mfe_vs_target",
    "life_mae_vs_target",
    "life_drawdown_from_mfe",
    "life_drawdown_from_mfe_vs_target",
    "life_distance_to_tp",
    "life_distance_to_sl",
    "life_favorable_ratio",
    "life_adverse_ratio",
    "life_last_ret_1",
    "life_last_ret_3",
    "life_last_ret_5",
    "life_last_ret_15",
    "life_ret_since_entry",
    "life_ret_since_entry_vs_target",
    "life_realized_vol_15",
    "life_range_pct_15",
    "life_volume_ratio_5_30",
    "life_trend_efficiency_15",
    "life_minutes_since_mfe",
    "life_minutes_since_mae",
    "life_recovery_from_mae",
    "life_recovery_from_mae_vs_target",
    "life_profit_lock_score",
    "life_init_stop_risk_score",
]

MULTI_HEAD_TARGETS = {
    "init_stop": "label_init_stop",
    "hold_quality": "label_hold_quality",
    "mae_risk": "label_mae_risk",
    "left_mfe_peak": "label_left_mfe_peak",
    "trail_continue": "label_trail_continue",
    "strict_exit": "label_strict_exit",
}

DEFAULT_MULTI_HEAD_THRESHOLDS = {
    "init_stop": 0.58,
    "hold_quality": 0.55,
    "mae_risk": 0.62,
    "left_mfe_peak": 0.58,
    "trail_continue": 0.55,
    "strict_exit": 0.50,
}


@dataclass
class StrictExitModelBundle:
    estimator: object
    feature_columns: list[str]
    fill_values: dict[str, float]
    threshold: float
    min_holding_minutes: int = 15
    route: str = ROUTE_NAME

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        x = frame.reindex(columns=self.feature_columns)
        x = x.replace([np.inf, -np.inf], np.nan)
        return x.fillna(self.fill_values).fillna(0.0)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        if frame.empty:
            return np.empty(0, dtype="float64")
        x = self.transform(frame)
        if hasattr(self.estimator, "predict_proba"):
            return np.asarray(self.estimator.predict_proba(x)[:, 1], dtype="float64")
        score = np.asarray(self.estimator.decision_function(x), dtype="float64")
        return 1.0 / (1.0 + np.exp(-score))

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str | Path) -> "StrictExitModelBundle":
        with Path(path).open("rb") as fh:
            return pickle.load(fh)


@dataclass
class MultiHeadExitModelBundle:
    estimators: dict[str, object]
    feature_columns: list[str]
    fill_values: dict[str, float]
    thresholds: dict[str, float]
    min_holding_minutes: int = 15
    route: str = ROUTE_NAME

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        x = frame.reindex(columns=self.feature_columns)
        x = x.replace([np.inf, -np.inf], np.nan)
        return x.fillna(self.fill_values).fillna(0.0)

    def predict_components(self, frame: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=frame.index)
        if frame.empty:
            for head in self.estimators:
                out[f"{head}_probability"] = pd.Series(dtype="float64")
            return out
        x = self.transform(frame)
        for head, estimator in self.estimators.items():
            if hasattr(estimator, "predict_proba"):
                prob = np.asarray(estimator.predict_proba(x)[:, 1], dtype="float64")
            else:
                score = np.asarray(estimator.decision_function(x), dtype="float64")
                prob = 1.0 / (1.0 + np.exp(-score))
            out[f"{head}_probability"] = prob
        return out

    def should_exit(
        self,
        frame: pd.DataFrame,
        *,
        current_exit_return: float,
        min_model_exit_return: float,
        policy: str = "balanced",
    ) -> tuple[bool, str, dict[str, float]]:
        comp = self.predict_components(frame)
        if comp.empty:
            return False, "empty_components", {}
        probs = {col.replace("_probability", ""): float(comp[col].iloc[0]) for col in comp.columns}
        th = {**DEFAULT_MULTI_HEAD_THRESHOLDS, **{str(k): float(v) for k, v in self.thresholds.items()}}
        row = frame.iloc[0]
        mfe_vs = float(pd.to_numeric(row.get("life_mfe_vs_target", 0.0), errors="coerce") or 0.0)
        mae_vs = float(pd.to_numeric(row.get("life_mae_vs_target", 0.0), errors="coerce") or 0.0)
        drawdown_vs = float(pd.to_numeric(row.get("life_drawdown_from_mfe_vs_target", 0.0), errors="coerce") or 0.0)
        elapsed = float(pd.to_numeric(row.get("life_elapsed_minutes", 0.0), errors="coerce") or 0.0)
        unrealized_vs = float(pd.to_numeric(row.get("life_unrealized_vs_target", 0.0), errors="coerce") or 0.0)
        init_bad = probs.get("init_stop", 0.0) >= th["init_stop"]
        weak_hold = probs.get("hold_quality", 0.0) < th["hold_quality"]
        mae_bad = probs.get("mae_risk", 0.0) >= th["mae_risk"]
        left_peak = probs.get("left_mfe_peak", 0.0) >= th["left_mfe_peak"]
        strict_bad = probs.get("strict_exit", 0.0) >= th["strict_exit"]
        trail_ok = probs.get("trail_continue", 0.0) >= th["trail_continue"]

        strong_clean_trend = mfe_vs >= 1.50 and mae_vs <= 0.85 and drawdown_vs <= 0.65
        profit_fade = (
            current_exit_return >= float(min_model_exit_return)
            and mfe_vs >= 0.70
            and drawdown_vs >= 0.35
            and unrealized_vs <= 0.35
            and not strong_clean_trend
        )
        severe_init_bad = elapsed <= 120.0 and mae_vs >= 0.90 and mfe_vs <= 0.45

        if str(policy).lower() == "profit_protect":
            init_loss_cut_floor = -0.003
            if severe_init_bad and weak_hold and current_exit_return >= init_loss_cut_floor:
                return True, "multi_rule_init_bad", probs
            if current_exit_return < float(min_model_exit_return):
                return False, "below_min_exit_return", probs
            if profit_fade and not trail_ok:
                return True, "multi_profit_protect", probs
            if init_bad and weak_hold and mfe_vs <= 0.75:
                return True, "multi_init_stop", probs
            if mae_bad and weak_hold and not strong_clean_trend:
                return True, "multi_mae_risk", probs
            if left_peak and not trail_ok and not strong_clean_trend:
                return True, "multi_left_peak", probs
            if strict_bad and weak_hold and not trail_ok and mfe_vs <= 1.20 and mae_vs >= 0.70:
                return True, "multi_strict_exit", probs
            return False, "multi_hold", probs

        if current_exit_return < float(min_model_exit_return):
            return False, "below_min_exit_return", probs
        if init_bad and weak_hold:
            return True, "multi_init_stop", probs
        if mae_bad and weak_hold:
            return True, "multi_mae_risk", probs
        if left_peak and not trail_ok:
            return True, "multi_left_peak", probs
        if strict_bad and weak_hold and not trail_ok:
            return True, "multi_strict_exit", probs
        return False, "multi_hold", probs

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: str | Path) -> "MultiHeadExitModelBundle":
        with Path(path).open("rb") as fh:
            return pickle.load(fh)


def _profit_factor(values: pd.Series) -> float:
    values = pd.to_numeric(values, errors="coerce").dropna()
    wins = float(values.loc[values > 0].sum())
    losses = float(-values.loc[values < 0].sum())
    if losses <= 0:
        return float("inf") if wins > 0 else float("nan")
    return float(wins / losses)


def _max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    return float((equity / equity.cummax() - 1.0).min())


def _signed_returns(direction: float, prices: pd.Series, windows: tuple[int, ...]) -> dict[int, float]:
    out: dict[int, float] = {}
    clean = pd.to_numeric(prices, errors="coerce").dropna()
    for window in windows:
        tail = clean.tail(max(2, int(window)))
        if len(tail) < 2:
            out[window] = 0.0
        else:
            out[window] = float(direction * (float(tail.iloc[-1]) - float(tail.iloc[0])) / (float(tail.iloc[0]) + 1e-12))
    return out


def _trend_efficiency(values: pd.Series) -> float:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if len(clean) < 2:
        return 0.0
    net = abs(float(clean.iloc[-1]) - float(clean.iloc[0]))
    path = float(clean.diff().abs().sum())
    return float(net / (path + 1e-12))


def _snapshot_features(
    *,
    bars: pd.DataFrame,
    direction: float,
    entry_price: float,
    entry_time: pd.Timestamp,
    snapshot_time: pd.Timestamp,
    target_pct: float,
    max_holding_minutes: float,
) -> dict[str, float]:
    hist = bars.loc[(bars.index >= entry_time) & (bars.index <= snapshot_time)]
    if hist.empty:
        return {name: 0.0 for name in EXIT_FEATURE_COLUMNS}

    close = pd.to_numeric(hist["close"], errors="coerce")
    high = pd.to_numeric(hist["high"], errors="coerce")
    low = pd.to_numeric(hist["low"], errors="coerce")
    volume = pd.to_numeric(hist["volume"], errors="coerce")
    current_price = float(close.iloc[-1])
    elapsed = max(0.0, (snapshot_time - entry_time).total_seconds() / 60.0)
    target = max(float(target_pct), 1e-6)
    unrealized = float(direction * (current_price - entry_price) / (entry_price + 1e-12))

    if direction > 0:
        favorable_path = (high - entry_price) / (entry_price + 1e-12)
        adverse_path = (low - entry_price) / (entry_price + 1e-12)
    else:
        favorable_path = (entry_price - low) / (entry_price + 1e-12)
        adverse_path = (entry_price - high) / (entry_price + 1e-12)

    favorable_path = pd.to_numeric(favorable_path, errors="coerce").fillna(0.0)
    adverse_path = pd.to_numeric(adverse_path, errors="coerce").fillna(0.0)
    mfe = float(max(0.0, favorable_path.max()))
    mae = float(min(0.0, adverse_path.min()))
    mfe_pos = int(favorable_path.to_numpy(dtype="float64").argmax()) if len(favorable_path) else 0
    mae_pos = int(adverse_path.to_numpy(dtype="float64").argmin()) if len(adverse_path) else 0
    bars_since_mfe = max(0, len(hist) - 1 - mfe_pos)
    bars_since_mae = max(0, len(hist) - 1 - mae_pos)

    signed = _signed_returns(direction, close, (1, 3, 5, 15))
    ret = close.pct_change().dropna()
    realized_vol_15 = float(ret.tail(15).std()) if len(ret.tail(15)) else 0.0
    tail_15 = hist.tail(15)
    range_pct_15 = float((tail_15["high"].max() - tail_15["low"].min()) / (current_price + 1e-12)) if len(tail_15) else 0.0
    vol_5 = float(volume.tail(5).mean()) if len(volume) else 0.0
    vol_30 = float(volume.tail(30).mean()) if len(volume) else 0.0
    volume_ratio = float(vol_5 / (vol_30 + 1e-12))
    body = direction * (hist["close"] - hist["open"])
    favorable_ratio = float((body > 0).mean()) if len(body) else 0.0
    adverse_ratio = float((body < 0).mean()) if len(body) else 0.0
    drawdown_from_mfe = max(0.0, mfe - unrealized)
    recovery_from_mae = max(0.0, unrealized - mae)
    profit_lock_score = float(max(0.0, mfe / target) * max(0.0, 1.0 - drawdown_from_mfe / (target + 1e-12)))
    init_stop_risk = float(
        0.35 * max(0.0, -unrealized / target)
        + 0.30 * max(0.0, -mae / target)
        + 0.20 * adverse_ratio
        + 0.15 * max(0.0, -signed.get(5, 0.0) / target)
    )

    return {
        "life_elapsed_minutes": float(elapsed),
        "life_elapsed_ratio": float(elapsed / max(float(max_holding_minutes), 1.0)),
        "life_unrealized_return": float(unrealized),
        "life_unrealized_vs_target": float(unrealized / target),
        "life_mfe_so_far": float(mfe),
        "life_mae_so_far": float(mae),
        "life_mfe_vs_target": float(mfe / target),
        "life_mae_vs_target": float(abs(mae) / target),
        "life_drawdown_from_mfe": float(drawdown_from_mfe),
        "life_drawdown_from_mfe_vs_target": float(drawdown_from_mfe / target),
        "life_distance_to_tp": float((target - unrealized) / target),
        "life_distance_to_sl": float((target + unrealized) / target),
        "life_favorable_ratio": float(favorable_ratio),
        "life_adverse_ratio": float(adverse_ratio),
        "life_last_ret_1": float(signed.get(1, 0.0)),
        "life_last_ret_3": float(signed.get(3, 0.0)),
        "life_last_ret_5": float(signed.get(5, 0.0)),
        "life_last_ret_15": float(signed.get(15, 0.0)),
        "life_ret_since_entry": float(unrealized),
        "life_ret_since_entry_vs_target": float(unrealized / target),
        "life_realized_vol_15": float(realized_vol_15),
        "life_range_pct_15": float(range_pct_15),
        "life_volume_ratio_5_30": float(volume_ratio),
        "life_trend_efficiency_15": float(_trend_efficiency(close.tail(15))),
        "life_minutes_since_mfe": float(bars_since_mfe),
        "life_minutes_since_mae": float(bars_since_mae),
        "life_recovery_from_mae": float(recovery_from_mae),
        "life_recovery_from_mae_vs_target": float(recovery_from_mae / target),
        "life_profit_lock_score": float(profit_lock_score),
        "life_init_stop_risk_score": float(init_stop_risk),
    }


def _load_trades(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"trade file not found: {p}")
    trades = pd.read_csv(p)
    for col in ("entry_time", "exit_time"):
        trades[col] = pd.to_datetime(trades[col], utc=True, errors="coerce")
    numeric = [
        "entry_price",
        "exit_price",
        "net_return",
        "mfe",
        "mae",
        "label_take_profit_pct",
        "label_stop_loss_pct",
        "label_max_holding_minutes",
        "exposure",
    ]
    for col in numeric:
        if col in trades.columns:
            trades[col] = pd.to_numeric(trades[col], errors="coerce")
    trades = trades.dropna(subset=["entry_time", "exit_time", "entry_price"]).sort_values("entry_time").reset_index(drop=True)
    if "side" not in trades.columns:
        trades["side"] = np.where(trades["is_buy"].astype(bool), "long", "short")
    return trades


def build_strict_exit_lifecycle_dataset(
    trades_path: str | Path = DEFAULT_TRAIN_TRADES_PATH,
    source_path: str | Path = DEFAULT_SOURCE_PATH,
    output_path: str | Path = DEFAULT_EXIT_DATASET_PATH,
    snapshot_minutes: int = 5,
    min_holding_minutes: int = 15,
    force: bool = False,
) -> pd.DataFrame:
    output = Path(output_path)
    if output.exists() and not force:
        return pd.read_parquet(output)

    trades = _load_trades(trades_path)
    if trades.empty:
        raise ValueError("empty trade file")
    begin = trades["entry_time"].min() - pd.Timedelta(minutes=5)
    end = trades["exit_time"].max() + pd.Timedelta(minutes=5)
    bars = load_1m_futures_bars(source_path, begin_time=begin, end_time=end)

    rows: list[dict[str, Any]] = []
    step = max(1, int(snapshot_minutes))
    min_hold = max(0, int(min_holding_minutes))
    for trade_idx, trade in trades.reset_index(drop=True).iterrows():
        entry_time = pd.Timestamp(trade["entry_time"])
        exit_time = pd.Timestamp(trade["exit_time"])
        entry_price = float(trade["entry_price"])
        direction = 1.0 if str(trade.get("side", "")).lower() == "long" or bool(trade.get("is_buy", False)) else -1.0
        target_pct = float(trade.get("label_take_profit_pct", trade.get("target_pct", 0.01)) or 0.01)
        max_holding = float(trade.get("label_max_holding_minutes", 1440) or 1440)
        fixed_net = float(trade.get("net_return", np.nan))
        exit_reason = str(trade.get("exit_reason", ""))
        if not np.isfinite(entry_price) or entry_price <= 0 or pd.isna(entry_time) or pd.isna(exit_time):
            continue
        snapshot_time = entry_time + pd.Timedelta(minutes=min_hold)
        while snapshot_time < exit_time:
            if snapshot_time not in bars.index:
                pos = int(np.searchsorted(bars.index.view("int64"), snapshot_time.value, side="right") - 1)
                if pos < 0:
                    snapshot_time += pd.Timedelta(minutes=step)
                    continue
                actual_snapshot = pd.Timestamp(bars.index[pos])
            else:
                actual_snapshot = snapshot_time
            feat = _snapshot_features(
                bars=bars,
                direction=direction,
                entry_price=entry_price,
                entry_time=entry_time,
                snapshot_time=actual_snapshot,
                target_pct=target_pct,
                max_holding_minutes=max_holding,
            )
            current_exit_return = float(feat["life_unrealized_return"] - 0.001)
            bad_continue = (
                str(exit_reason) == "stop_loss"
                or (np.isfinite(fixed_net) and fixed_net < 0.0)
                or (feat["life_mae_vs_target"] >= 0.85 and feat["life_mfe_vs_target"] < 0.35)
            )
            protect_profit = (
                feat["life_mfe_vs_target"] >= 0.65
                and feat["life_drawdown_from_mfe_vs_target"] >= 0.35
                and (not np.isfinite(fixed_net) or fixed_net <= feat["life_unrealized_return"])
            )
            exit_better_than_hold = (
                np.isfinite(fixed_net)
                and current_exit_return >= fixed_net + max(0.0015, target_pct * 0.15)
                and (
                    fixed_net < 0.0
                    or protect_profit
                    or feat["life_drawdown_from_mfe_vs_target"] >= 0.55
                )
            )
            future_upside = float(fixed_net - current_exit_return) if np.isfinite(fixed_net) else float("nan")
            init_stop = (
                feat["life_elapsed_minutes"] <= 120.0
                and feat["life_mae_vs_target"] >= 0.45
                and feat["life_mfe_vs_target"] < 0.35
                and (str(exit_reason) == "stop_loss" or (np.isfinite(fixed_net) and fixed_net < 0.0))
            )
            mae_risk = (
                feat["life_mae_vs_target"] >= 0.75
                and feat["life_recovery_from_mae_vs_target"] < 0.35
                and (not np.isfinite(fixed_net) or fixed_net < target_pct * 0.25)
            )
            left_mfe_peak = (
                feat["life_mfe_vs_target"] >= 0.65
                and feat["life_drawdown_from_mfe_vs_target"] >= 0.35
                and (not np.isfinite(fixed_net) or current_exit_return >= fixed_net + max(0.001, target_pct * 0.10))
            )
            trail_continue = (
                feat["life_mfe_vs_target"] >= 0.45
                and feat["life_drawdown_from_mfe_vs_target"] < 0.35
                and np.isfinite(future_upside)
                and future_upside > max(0.0015, target_pct * 0.20)
            )
            hold_quality = (
                np.isfinite(future_upside)
                and future_upside > max(0.001, target_pct * 0.15)
                and feat["life_mae_vs_target"] < 0.95
                and not init_stop
            )
            row = {
                "trade_idx": int(trade_idx),
                "symbol": str(trade.get("symbol", "BTCUSDT")),
                "entry_time": entry_time,
                "snapshot_time": actual_snapshot,
                "original_exit_time": exit_time,
                "side": "long" if direction > 0 else "short",
                "entry_price": float(entry_price),
                "original_exit_price": float(trade.get("exit_price", np.nan)),
                "original_exit_reason": exit_reason,
                "original_net_return": fixed_net,
                "target_pct": target_pct,
                "confidence_tier": str(trade.get("confidence_tier", "")),
                "probability": float(trade.get("probability", np.nan)),
                "current_exit_return": float(current_exit_return),
                "future_upside_from_snapshot": float(future_upside),
                "label_bad_continue": float(bad_continue),
                "label_protect_profit": float(protect_profit),
                "label_exit_better_than_hold": float(exit_better_than_hold),
                "label_init_stop": float(init_stop),
                "label_hold_quality": float(hold_quality),
                "label_mae_risk": float(mae_risk),
                "label_left_mfe_peak": float(left_mfe_peak),
                "label_trail_continue": float(trail_continue),
                "label_strict_exit": float(exit_better_than_hold),
            }
            row.update(feat)
            rows.append(row)
            snapshot_time += pd.Timedelta(minutes=step)

    dataset = pd.DataFrame(rows)
    if dataset.empty:
        raise ValueError("no lifecycle snapshots generated")
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(output, index=False)
    meta = {
        "route": ROUTE_NAME,
        "trades_path": str(trades_path),
        "source_path": str(source_path),
        "rows": int(len(dataset)),
        "trades": int(dataset["trade_idx"].nunique()),
        "positive_rate": float(dataset["label_strict_exit"].mean()),
        "snapshot_minutes": int(snapshot_minutes),
        "min_holding_minutes": int(min_holding_minutes),
        "lookahead_rule": "features use bars from entry_time through snapshot_time only; labels use future fixed-exit outcome",
    }
    output.with_name(output.stem + "_meta.json").write_text(json.dumps(json_safe(meta), ensure_ascii=False, indent=2), encoding="utf-8")
    return dataset


def _classification_metrics(y: pd.Series, p: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {"rows": int(len(y)), "positive_rate": float(pd.Series(y).mean()) if len(y) else float("nan")}
    if len(pd.Series(y).dropna().unique()) < 2:
        out.update({"auc": float("nan"), "ap": float("nan"), "brier": float("nan")})
        return out
    try:
        from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

        yy = pd.Series(y).astype(int).to_numpy()
        out["auc"] = float(roc_auc_score(yy, p))
        out["ap"] = float(average_precision_score(yy, p))
        out["brier"] = float(brier_score_loss(yy, p))
    except Exception:
        out.update({"auc": float("nan"), "ap": float("nan"), "brier": float("nan")})
    return out


def train_strict_exit_model(
    dataset_path: str | Path = DEFAULT_EXIT_DATASET_PATH,
    model_path: str | Path = DEFAULT_EXIT_MODEL_PATH,
    train_end: Any = "2025-01-01",
    valid_start: Any = "2025-01-01",
    threshold_grid: tuple[float, ...] = tuple(np.round(np.arange(0.50, 0.91, 0.03), 2)),
    min_holding_minutes: int = 15,
) -> dict[str, Any]:
    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        build_strict_exit_lifecycle_dataset(output_path=dataset_file)
    data = pd.read_parquet(dataset_file)
    data["snapshot_time"] = pd.to_datetime(data["snapshot_time"], utc=True, errors="coerce")
    data = data.dropna(subset=["label_strict_exit", "snapshot_time"]).sort_values("snapshot_time").reset_index(drop=True)
    train_end_ts = parse_utc(train_end)
    valid_start_ts = parse_utc(valid_start)
    train = data.loc[data["snapshot_time"] < train_end_ts].copy()
    valid = data.loc[data["snapshot_time"] >= valid_start_ts].copy()
    if train.empty or valid.empty:
        raise ValueError("empty train or validation split for strict exit model")

    x_train = train.reindex(columns=EXIT_FEATURE_COLUMNS).replace([np.inf, -np.inf], np.nan)
    fill_values = x_train.median(numeric_only=True).fillna(0.0).to_dict()
    x_train = x_train.fillna(fill_values).fillna(0.0)
    y_train = train["label_strict_exit"].astype(int)

    from sklearn.ensemble import HistGradientBoostingClassifier

    estimator = HistGradientBoostingClassifier(
        max_iter=180,
        learning_rate=0.045,
        max_leaf_nodes=15,
        min_samples_leaf=80,
        l2_regularization=0.05,
        random_state=20260529,
    )
    estimator.fit(x_train, y_train)

    bundle = StrictExitModelBundle(
        estimator=estimator,
        feature_columns=list(EXIT_FEATURE_COLUMNS),
        fill_values={str(k): float(v) for k, v in fill_values.items()},
        threshold=0.65,
        min_holding_minutes=int(min_holding_minutes),
    )
    train_prob = bundle.predict_proba(train)
    valid_prob = bundle.predict_proba(valid)
    threshold_rows = []
    y_valid = valid["label_strict_exit"].astype(int)
    for threshold in threshold_grid:
        pred = valid_prob >= float(threshold)
        selected = int(pred.sum())
        precision = float(y_valid[pred].mean()) if selected else float("nan")
        recall = float((pred & (y_valid == 1)).sum() / max(int((y_valid == 1).sum()), 1))
        threshold_rows.append({"threshold": float(threshold), "selected": selected, "precision": precision, "recall": recall})
    threshold_frame = pd.DataFrame(threshold_rows)
    eligible = threshold_frame.loc[threshold_frame["selected"] >= max(20, int(len(valid) * 0.03))].copy()
    if eligible.empty:
        best_threshold = 0.65
    else:
        eligible["score"] = eligible["precision"].fillna(0.0) * 0.70 + eligible["recall"].fillna(0.0) * 0.30
        best_threshold = float(eligible.sort_values(["score", "precision"], ascending=[False, False]).iloc[0]["threshold"])
    bundle.threshold = best_threshold
    bundle.save(model_path)
    threshold_frame.to_csv(Path(model_path).with_name("strict_exit_threshold_scan.csv"), index=False)

    metrics = {
        "route": ROUTE_NAME,
        "model": "strict_exit_model",
        "dataset_path": str(dataset_path),
        "model_path": str(model_path),
        "feature_columns": list(EXIT_FEATURE_COLUMNS),
        "threshold": float(best_threshold),
        "train": _classification_metrics(y_train, train_prob),
        "valid": _classification_metrics(y_valid, valid_prob),
        "label_definition": "strict_exit=current exit is materially better than holding to original fixed exit, focused on loss cutting/profit protection",
    }
    Path(model_path).with_name("strict_exit_model_metrics.json").write_text(
        json.dumps(json_safe(metrics), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics


def _fit_exit_head(train: pd.DataFrame, target: str, random_state: int) -> object:
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier

    y = train[target].astype(int)
    if int(y.nunique()) < 2:
        estimator = DummyClassifier(strategy="constant", constant=int(y.iloc[0]) if len(y) else 0)
    else:
        estimator = HistGradientBoostingClassifier(
            max_iter=160,
            learning_rate=0.045,
            max_leaf_nodes=15,
            min_samples_leaf=80,
            l2_regularization=0.05,
            random_state=int(random_state),
        )
    return estimator


def train_multi_head_exit_model(
    dataset_path: str | Path = DEFAULT_EXIT_DATASET_PATH,
    model_path: str | Path = DEFAULT_MULTI_HEAD_MODEL_PATH,
    train_end: Any = "2025-01-01",
    valid_start: Any = "2025-01-01",
    min_holding_minutes: int = 15,
) -> dict[str, Any]:
    dataset_file = Path(dataset_path)
    if not dataset_file.exists():
        build_strict_exit_lifecycle_dataset(output_path=dataset_file)
    data = pd.read_parquet(dataset_file)
    data["snapshot_time"] = pd.to_datetime(data["snapshot_time"], utc=True, errors="coerce")
    missing = [target for target in MULTI_HEAD_TARGETS.values() if target not in data.columns]
    if missing:
        data = build_strict_exit_lifecycle_dataset(output_path=dataset_file, force=True)
        data["snapshot_time"] = pd.to_datetime(data["snapshot_time"], utc=True, errors="coerce")
    data = data.dropna(subset=["snapshot_time"]).sort_values("snapshot_time").reset_index(drop=True)
    train_end_ts = parse_utc(train_end)
    valid_start_ts = parse_utc(valid_start)
    train = data.loc[data["snapshot_time"] < train_end_ts].copy()
    valid = data.loc[data["snapshot_time"] >= valid_start_ts].copy()
    if train.empty or valid.empty:
        raise ValueError("empty train or validation split for multi-head exit model")

    x_train = train.reindex(columns=EXIT_FEATURE_COLUMNS).replace([np.inf, -np.inf], np.nan)
    fill_values = x_train.median(numeric_only=True).fillna(0.0).to_dict()
    x_train = x_train.fillna(fill_values).fillna(0.0)

    estimators: dict[str, object] = {}
    metrics: dict[str, Any] = {
        "route": ROUTE_NAME,
        "model": "multi_head_exit_model",
        "dataset_path": str(dataset_path),
        "model_path": str(model_path),
        "feature_columns": list(EXIT_FEATURE_COLUMNS),
        "heads": {},
    }
    for idx, (head, target) in enumerate(MULTI_HEAD_TARGETS.items()):
        train[target] = pd.to_numeric(train[target], errors="coerce").fillna(0).astype(int)
        valid[target] = pd.to_numeric(valid[target], errors="coerce").fillna(0).astype(int)
        estimator = _fit_exit_head(train, target, random_state=20260530 + idx)
        estimator.fit(x_train, train[target].astype(int))
        estimators[head] = estimator

    bundle = MultiHeadExitModelBundle(
        estimators=estimators,
        feature_columns=list(EXIT_FEATURE_COLUMNS),
        fill_values={str(k): float(v) for k, v in fill_values.items()},
        thresholds=dict(DEFAULT_MULTI_HEAD_THRESHOLDS),
        min_holding_minutes=int(min_holding_minutes),
    )
    train_components = bundle.predict_components(train)
    valid_components = bundle.predict_components(valid)
    for head, target in MULTI_HEAD_TARGETS.items():
        metrics["heads"][head] = {
            "target": target,
            "threshold": float(bundle.thresholds.get(head, np.nan)),
            "train": _classification_metrics(train[target].astype(int), train_components[f"{head}_probability"].to_numpy(dtype="float64")),
            "valid": _classification_metrics(valid[target].astype(int), valid_components[f"{head}_probability"].to_numpy(dtype="float64")),
        }
    bundle.save(model_path)
    Path(model_path).with_name("multi_head_exit_model_metrics.json").write_text(
        json.dumps(json_safe(metrics), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics


def classify_trade_lifecycle_groups(
    trades_path: str | Path = DEFAULT_TEST_TRADES_PATH,
    output_dir: str | Path = DEFAULT_EXIT_BACKTEST_DIR / "trade_groups",
) -> dict[str, Any]:
    trades = _load_trades(trades_path).copy()
    if trades.empty:
        raise ValueError("empty trades for lifecycle group diagnostics")
    target = pd.to_numeric(trades.get("label_take_profit_pct", 0.01), errors="coerce").replace(0, np.nan).fillna(0.01)
    net = pd.to_numeric(trades.get("net_return", np.nan), errors="coerce")
    mfe = pd.to_numeric(trades.get("mfe", np.nan), errors="coerce").fillna(0.0)
    mae = pd.to_numeric(trades.get("mae", np.nan), errors="coerce").fillna(0.0).abs()
    holding = pd.to_numeric(trades.get("holding_minutes", np.nan), errors="coerce")
    exit_reason = trades.get("exit_reason", "").astype(str)

    mfe_vs = mfe / (target + 1e-12)
    mae_vs = mae / (target + 1e-12)
    group = pd.Series("noisy_survivor", index=trades.index, dtype="object")
    group.loc[(exit_reason == "stop_loss") & ((holding <= 120.0) | ((mae_vs >= 0.85) & (mfe_vs < 0.35)))] = "init_bad"
    group.loc[(mfe_vs < 0.35) & (group == "noisy_survivor")] = "no_edge"
    group.loc[(mfe_vs >= 0.65) & (net <= 0.0)] = "good_then_fade"
    group.loc[(net > 0.0) & (mfe_vs >= 0.85) & (mae_vs < 0.65)] = "clean_trend"
    group.loc[(net > 0.0) & (mfe_vs < 0.85) & (group == "noisy_survivor")] = "small_win"
    group.loc[(net <= 0.0) & (group == "noisy_survivor")] = "small_loss"

    trades["lifecycle_group"] = group
    trades["mfe_vs_target"] = mfe_vs
    trades["mae_vs_target"] = mae_vs
    summary = trades.groupby("lifecycle_group", dropna=False).agg(
        trades=("net_return", "size"),
        wins=("net_return", lambda s: int((pd.to_numeric(s, errors="coerce") > 0).sum())),
        avg_return=("net_return", "mean"),
        total_return=("net_return", "sum"),
        avg_mfe_vs_target=("mfe_vs_target", "mean"),
        avg_mae_vs_target=("mae_vs_target", "mean"),
        avg_holding_minutes=("holding_minutes", "mean"),
    )
    summary["win_rate"] = summary["wins"] / summary["trades"]
    summary = summary.sort_values(["total_return", "trades"], ascending=[True, False])
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out / "trade_lifecycle_groups.csv", index=False)
    summary.to_csv(out / "trade_lifecycle_group_summary.csv")
    result = {
        "route": ROUTE_NAME,
        "trades_path": str(trades_path),
        "output_dir": str(output_dir),
        "groups": json_safe(summary.reset_index().to_dict("records")),
    }
    (out / "trade_lifecycle_group_summary.json").write_text(json.dumps(json_safe(result), ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def _simulate_fixed_or_ml_exit(
    trades: pd.DataFrame,
    bars: pd.DataFrame,
    bundle: StrictExitModelBundle | MultiHeadExitModelBundle | None,
    *,
    initial_cash: float,
    snapshot_minutes: int,
    cost_rate: float,
    min_model_exit_return: float,
    use_model: bool,
    model_mode: str = "strict",
    multi_policy: str = "balanced",
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    equity = float(initial_cash)
    trade_rows: list[dict[str, Any]] = []
    equity_rows = []
    for trade_idx, trade in trades.reset_index(drop=True).iterrows():
        direction = 1.0 if str(trade.get("side", "")).lower() == "long" or bool(trade.get("is_buy", False)) else -1.0
        entry_time = pd.Timestamp(trade["entry_time"])
        exit_time = pd.Timestamp(trade["exit_time"])
        entry_price = float(trade["entry_price"])
        target_pct = float(trade.get("label_take_profit_pct", trade.get("target_pct", 0.01)) or 0.01)
        max_holding = float(trade.get("label_max_holding_minutes", 1440) or 1440)
        exposure = float(trade.get("exposure", 1.0) or 1.0)
        chosen_exit_time = exit_time
        chosen_exit_price = float(trade.get("exit_price", np.nan))
        chosen_reason = str(trade.get("exit_reason", "fixed_exit"))
        exit_probability = float("nan")
        exit_head_probabilities: dict[str, float] = {}

        if use_model and bundle is not None:
            snapshot_time = entry_time + pd.Timedelta(minutes=max(int(bundle.min_holding_minutes), int(snapshot_minutes)))
            while snapshot_time < exit_time:
                pos = int(np.searchsorted(bars.index.view("int64"), snapshot_time.value, side="right") - 1)
                if pos < 0:
                    snapshot_time += pd.Timedelta(minutes=snapshot_minutes)
                    continue
                actual_snapshot = pd.Timestamp(bars.index[pos])
                feat = _snapshot_features(
                    bars=bars,
                    direction=direction,
                    entry_price=entry_price,
                    entry_time=entry_time,
                    snapshot_time=actual_snapshot,
                    target_pct=target_pct,
                    max_holding_minutes=max_holding,
                )
                frame = pd.DataFrame([feat])
                current_exit_return = float(feat["life_unrealized_return"] - float(cost_rate))
                if isinstance(bundle, MultiHeadExitModelBundle) or model_mode == "multi":
                    should_exit, reason, probs = bundle.should_exit(
                        frame,
                        current_exit_return=current_exit_return,
                        min_model_exit_return=float(min_model_exit_return),
                        policy=str(multi_policy),
                    )
                    if should_exit:
                        chosen_exit_time = actual_snapshot
                        chosen_exit_price = float(bars["close"].iloc[pos])
                        chosen_reason = reason
                        exit_probability = float(max(probs.values())) if probs else float("nan")
                        exit_head_probabilities = probs
                        break
                else:
                    prob = float(bundle.predict_proba(frame)[0])
                    if prob >= float(bundle.threshold) and current_exit_return >= float(min_model_exit_return):
                        chosen_exit_time = actual_snapshot
                        chosen_exit_price = float(bars["close"].iloc[pos])
                        chosen_reason = "ml_strict_exit"
                        exit_probability = prob
                        break
                snapshot_time += pd.Timedelta(minutes=snapshot_minutes)

        if not np.isfinite(chosen_exit_price) or chosen_exit_price <= 0:
            chosen_exit_price = float(trade.get("exit_price", entry_price))
        net_return = float(direction * (chosen_exit_price - entry_price) / (entry_price + 1e-12) - float(cost_rate))
        equity_before = equity
        pnl = equity * exposure * net_return
        equity += pnl
        row = trade.to_dict()
        row.update(
            {
                "original_exit_time": trade["exit_time"],
                "original_exit_price": float(trade.get("exit_price", np.nan)),
                "original_exit_reason": str(trade.get("exit_reason", "")),
                "original_net_return": float(trade.get("net_return", np.nan)),
                "exit_time": chosen_exit_time,
                "exit_price": float(chosen_exit_price),
                "exit_reason": chosen_reason,
                "exit_model_probability": exit_probability,
                "net_return": float(net_return),
                "equity_before": float(equity_before),
                "pnl": float(pnl),
                "equity_after": float(equity),
                "ml_exit_used": bool(str(chosen_reason).startswith("ml_") or str(chosen_reason).startswith("multi_")),
            }
        )
        for head, prob in exit_head_probabilities.items():
            row[f"{head}_probability"] = float(prob)
        trade_rows.append(row)
        equity_rows.append({"time": chosen_exit_time, "equity": float(equity)})
    trades_df = pd.DataFrame(trade_rows)
    equity_df = pd.DataFrame(equity_rows)
    pnl = pd.to_numeric(trades_df["pnl"], errors="coerce") if not trades_df.empty else pd.Series(dtype="float64")
    returns = pd.to_numeric(trades_df["net_return"], errors="coerce") if not trades_df.empty else pd.Series(dtype="float64")
    wins = int((returns > 0).sum()) if len(returns) else 0
    metrics = {
        "initial_cash": float(initial_cash),
        "final_equity": float(equity),
        "total_return": float(equity / float(initial_cash) - 1.0),
        "trades": int(len(trades_df)),
        "wins": wins,
        "losses": int(len(trades_df) - wins),
        "win_rate": float(wins / len(trades_df)) if len(trades_df) else float("nan"),
        "avg_return": float(returns.mean()) if len(returns) else float("nan"),
        "profit_factor": _profit_factor(pnl) if len(pnl) else float("nan"),
        "max_drawdown": _max_drawdown(equity_df.set_index("time")["equity"]) if not equity_df.empty else 0.0,
        "ml_exit_count": int(trades_df["ml_exit_used"].sum()) if not trades_df.empty else 0,
    }
    return trades_df, equity_df, metrics


def backtest_strict_exit_model(
    trades_path: str | Path = DEFAULT_TEST_TRADES_PATH,
    source_path: str | Path = DEFAULT_SOURCE_PATH,
    model_path: str | Path = DEFAULT_EXIT_MODEL_PATH,
    output_dir: str | Path = DEFAULT_EXIT_BACKTEST_DIR,
    begin_time: Any | None = None,
    end_time: Any | None = None,
    initial_cash: float = 100000.0,
    snapshot_minutes: int = 5,
    cost_rate: float = 0.001,
    min_model_exit_return: float = 0.0,
    exit_threshold: float | None = None,
    model_mode: str = "strict",
    multi_policy: str = "balanced",
) -> dict[str, Any]:
    trades = _load_trades(trades_path)
    begin = parse_utc(begin_time)
    end = parse_utc(end_time, end_of_day=True)
    if begin is not None:
        trades = trades.loc[trades["entry_time"] >= begin]
    if end is not None:
        trades = trades.loc[trades["entry_time"] <= end]
    if trades.empty:
        raise ValueError("empty trades for strict exit backtest")
    bars = load_1m_futures_bars(
        source_path,
        begin_time=trades["entry_time"].min() - pd.Timedelta(minutes=5),
        end_time=trades["exit_time"].max() + pd.Timedelta(minutes=5),
    )
    if model_mode == "multi":
        bundle: StrictExitModelBundle | MultiHeadExitModelBundle = MultiHeadExitModelBundle.load(model_path)
    else:
        bundle = StrictExitModelBundle.load(model_path)
    if exit_threshold is not None and isinstance(bundle, StrictExitModelBundle):
        bundle.threshold = float(exit_threshold)
    fixed_trades, fixed_equity, fixed_metrics = _simulate_fixed_or_ml_exit(
        trades,
        bars,
        None,
        initial_cash=initial_cash,
        snapshot_minutes=snapshot_minutes,
        cost_rate=cost_rate,
        min_model_exit_return=min_model_exit_return,
        use_model=False,
        model_mode=model_mode,
        multi_policy=multi_policy,
    )
    ml_trades, ml_equity, ml_metrics = _simulate_fixed_or_ml_exit(
        trades,
        bars,
        bundle,
        initial_cash=initial_cash,
        snapshot_minutes=snapshot_minutes,
        cost_rate=cost_rate,
        min_model_exit_return=min_model_exit_return,
        use_model=True,
        model_mode=model_mode,
        multi_policy=multi_policy,
    )
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    fixed_trades.to_csv(out / "fixed_exit_replay_trades.csv", index=False)
    fixed_equity.to_csv(out / "fixed_exit_replay_equity.csv", index=False)
    trade_file = "ml_multi_head_exit_trades.csv" if model_mode == "multi" else "ml_strict_exit_trades.csv"
    equity_file = "ml_multi_head_exit_equity.csv" if model_mode == "multi" else "ml_strict_exit_equity.csv"
    ml_trades.to_csv(out / trade_file, index=False)
    ml_equity.to_csv(out / equity_file, index=False)
    if not ml_trades.empty:
        by_reason = ml_trades.groupby("exit_reason", dropna=False).agg(
            trades=("net_return", "size"),
            win_rate=("net_return", lambda s: float((pd.to_numeric(s, errors="coerce") > 0).mean())),
            avg_return=("net_return", "mean"),
            pnl=("pnl", "sum"),
        )
        by_reason.to_csv(out / "exit_reason_summary.csv")
    metrics = {
        "route": ROUTE_NAME,
        "model_mode": str(model_mode),
        "multi_policy": str(multi_policy) if model_mode == "multi" else None,
        "model_path": str(model_path),
        "exit_threshold": float(bundle.threshold) if isinstance(bundle, StrictExitModelBundle) else None,
        "multi_head_thresholds": dict(bundle.thresholds) if isinstance(bundle, MultiHeadExitModelBundle) else None,
        "trades_path": str(trades_path),
        "min_model_exit_return": float(min_model_exit_return),
        "fixed_exit_replay": fixed_metrics,
        "ml_exit": ml_metrics,
        "delta": {
            "win_rate": float(ml_metrics["win_rate"] - fixed_metrics["win_rate"]),
            "total_return": float(ml_metrics["total_return"] - fixed_metrics["total_return"]),
            "profit_factor": float(ml_metrics["profit_factor"] - fixed_metrics["profit_factor"])
            if np.isfinite(ml_metrics["profit_factor"]) and np.isfinite(fixed_metrics["profit_factor"])
            else float("nan"),
            "max_drawdown": float(ml_metrics["max_drawdown"] - fixed_metrics["max_drawdown"]),
        },
        "lookahead_rule": "exit model features use entry_time..snapshot_time only; replay exits at snapshot close",
    }
    (out / "strict_exit_backtest_metrics.json").write_text(json.dumps(json_safe(metrics), ensure_ascii=False, indent=2), encoding="utf-8")
    return metrics


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="BTC futures v3 beta ML exit management")
    parser.add_argument(
        "--action",
        choices=["build", "train", "train-multi", "backtest", "backtest-multi", "diagnose", "all", "multi-all"],
        default="all",
    )
    parser.add_argument("--trades", default=str(DEFAULT_TRAIN_TRADES_PATH))
    parser.add_argument("--test-trades", default=str(DEFAULT_TEST_TRADES_PATH))
    parser.add_argument("--source", default=str(DEFAULT_SOURCE_PATH))
    parser.add_argument("--dataset", default=str(DEFAULT_EXIT_DATASET_PATH))
    parser.add_argument("--model", default=str(DEFAULT_EXIT_MODEL_PATH))
    parser.add_argument("--multi-model", default=str(DEFAULT_MULTI_HEAD_MODEL_PATH))
    parser.add_argument("--output-dir", default=str(DEFAULT_EXIT_BACKTEST_DIR))
    parser.add_argument("--snapshot-minutes", type=int, default=5)
    parser.add_argument("--min-holding-minutes", type=int, default=15)
    parser.add_argument("--min-model-exit-return", type=float, default=0.0)
    parser.add_argument("--exit-threshold", type=float, default=None)
    parser.add_argument("--multi-policy", default="balanced", choices=["balanced", "profit_protect"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    result: dict[str, Any] = {}
    if args.action in {"build", "all", "multi-all"}:
        ds = build_strict_exit_lifecycle_dataset(
            trades_path=args.trades,
            source_path=args.source,
            output_path=args.dataset,
            snapshot_minutes=args.snapshot_minutes,
            min_holding_minutes=args.min_holding_minutes,
            force=args.force,
        )
        result["build"] = {"rows": int(len(ds)), "trades": int(ds["trade_idx"].nunique())}
    if args.action in {"train", "all"}:
        result["train"] = train_strict_exit_model(
            dataset_path=args.dataset,
            model_path=args.model,
            min_holding_minutes=args.min_holding_minutes,
        )
    if args.action in {"train-multi", "multi-all"}:
        result["train_multi"] = train_multi_head_exit_model(
            dataset_path=args.dataset,
            model_path=args.multi_model,
            min_holding_minutes=args.min_holding_minutes,
        )
    if args.action in {"backtest", "all"}:
        result["backtest"] = backtest_strict_exit_model(
            trades_path=args.test_trades,
            source_path=args.source,
            model_path=args.model,
            output_dir=args.output_dir,
            snapshot_minutes=args.snapshot_minutes,
            min_model_exit_return=args.min_model_exit_return,
            exit_threshold=args.exit_threshold,
            model_mode="strict",
        )
    if args.action in {"backtest-multi", "multi-all"}:
        result["backtest_multi"] = backtest_strict_exit_model(
            trades_path=args.test_trades,
            source_path=args.source,
            model_path=args.multi_model,
            output_dir=args.output_dir,
            snapshot_minutes=args.snapshot_minutes,
            min_model_exit_return=args.min_model_exit_return,
            model_mode="multi",
            multi_policy=args.multi_policy,
        )
    if args.action in {"diagnose", "multi-all"}:
        result["diagnose"] = classify_trade_lifecycle_groups(
            trades_path=args.test_trades,
            output_dir=str(Path(args.output_dir) / "trade_groups"),
        )
    print(json.dumps(json_safe(result), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
