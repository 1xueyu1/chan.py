from __future__ import annotations

import argparse
import json
import os
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from Backtest.chan_signal_extractor import extract_raw_bsp_events
from Backtest.config import BacktestConfig
from Backtest.data_loader import load_symbol_bars
from Backtest.feature_adapter import enrich_raw_events_with_feature_engine
from Backtest.model_gate import DualModelGate
from Backtest.types import RawBSPEvent
from Debug.model_predict_analyzer_common import (
    Sample,
    analyze_one_sample,
    build_feature_vector,
    load_artifacts,
)


@dataclass
class TradeLegMatch:
    symbol: str
    leg: str
    expected_signal: int
    trade_time: str
    matched_events: int
    selected_event: Optional[Dict[str, object]]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze decision chain for executed backtest trades only"
    )
    parser.add_argument(
        "--run-dir",
        default="Debug/runs/full_fixed_verify_20260323",
        help="Run root directory like Debug/runs/<run_id>",
    )
    parser.add_argument(
        "--symbols",
        nargs="*",
        default=[],
        help="Optional symbol filter, e.g. BTCUSDT ETHUSDT",
    )
    parser.add_argument(
        "--max-trades",
        type=int,
        default=0,
        help="Limit number of executed trades to analyze, 0 means all",
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="Output path, default: <run-dir>/backtest/trade_decision_chain.json",
    )
    return parser.parse_args()


def _normalize_symbol(symbol: str) -> str:
    s = str(symbol).upper().replace("/", "")
    if not s.endswith("USDT"):
        s = f"{s}USDT"
    return s


def _parse_trade_rows_from_detail_html(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"detail html not found: {path}")

    text = path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(
        r"const\s+tradeRows\s*=\s*(\[.*?\]);\s*let\s+symbolFilter",
        text,
        flags=re.S,
    )
    if not m:
        raise RuntimeError("Cannot locate tradeRows JSON in detail html")

    payload = m.group(1)
    rows = json.loads(payload)
    if not isinstance(rows, list):
        raise RuntimeError("Parsed tradeRows is not a list")
    return rows


def _load_run_context(run_dir: Path) -> Dict[str, object]:
    manifest_path = run_dir / "run_manifest.json"
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    backtest_cfg = manifest.get("backtest_config", {})
    backtest_paths = manifest.get("paths", {})
    artifacts = manifest.get("artifacts", {})

    train_dir = Path(backtest_paths.get("train", run_dir / "train"))
    if not train_dir.is_absolute():
        train_dir = (Path.cwd() / train_dir).resolve()

    detail_html = artifacts.get("backtest_report_detail", "")
    if detail_html:
        detail_html_path = Path(detail_html)
        if not detail_html_path.is_absolute():
            detail_html_path = (Path.cwd() / detail_html_path).resolve()
    else:
        detail_html_path = (run_dir / "backtest" / "xgb_backtest_report_detail.html").resolve()

    signal_events_csv = run_dir / "backtest" / "model_signal_events.csv"
    signal_bars_csv = run_dir / "backtest" / "model_signal_bars.csv"

    return {
        "manifest": manifest,
        "backtest_cfg": backtest_cfg,
        "train_dir": train_dir,
        "detail_html": detail_html_path,
        "events_csv": signal_events_csv.resolve(),
        "bars_csv": signal_bars_csv.resolve(),
    }


def _resolve_target_loc(index: pd.DatetimeIndex, ts: pd.Timestamp, execution_mode: str) -> Optional[int]:
    loc = int(index.searchsorted(ts, side="left"))
    if loc >= len(index):
        return None
    if execution_mode == "next_bar_open":
        loc += 1
        if loc >= len(index):
            return None
    return loc


def _event_key(
    symbol: str,
    exec_time: pd.Timestamp,
    is_buy: bool,
    bsp_time: str,
    bsp_type: str,
    bsp_types_str: str,
) -> Tuple[str, str, int, str, str, str]:
    return (
        _normalize_symbol(symbol),
        pd.Timestamp(exec_time).tz_convert("UTC").strftime("%Y-%m-%d %H:%M:%S"),
        int(bool(is_buy)),
        str(bsp_time),
        str(bsp_type),
        str(bsp_types_str),
    )


def _event_key_from_raw(ev: RawBSPEvent) -> Tuple[str, str, int, str, str, str]:
    ts = pd.Timestamp(ev.exec_time)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return (
        _normalize_symbol(ev.symbol),
        ts.strftime("%Y-%m-%d %H:%M:%S"),
        int(bool(ev.is_buy)),
        str(ev.bsp_time),
        str(ev.bsp_type),
        str(ev.bsp_types_str),
    )


def _sample_from_raw(ev: RawBSPEvent) -> Sample:
    return Sample(
        symbol=ev.symbol,
        bsp_time=ev.bsp_time,
        trade_price=float(ev.trade_price),
        is_buy=bool(ev.is_buy),
        bsp_type=str(ev.bsp_type),
        bsp_types_str=str(ev.bsp_types_str),
        feature_map={str(k): float(v) for k, v in ev.feature_map.items()},
    )


def _load_raw_event_index(
    cfg: BacktestConfig,
    symbol: str,
) -> Dict[Tuple[str, str, int, str, str, str], List[RawBSPEvent]]:
    bars = load_symbol_bars(cfg, symbol)
    raw_events = extract_raw_bsp_events(cfg, symbol)
    raw_events = enrich_raw_events_with_feature_engine(
        symbol=symbol,
        bars=bars,
        raw_events=raw_events,
    )

    idx: Dict[Tuple[str, str, int, str, str, str], List[RawBSPEvent]] = {}
    for ev in raw_events:
        k = _event_key_from_raw(ev)
        idx.setdefault(k, []).append(ev)
    return idx


def _compute_symbol_framework_details(
    gate: DualModelGate,
    cfg: BacktestConfig,
    raw_events: List[RawBSPEvent],
) -> Dict[Tuple[str, str, int, str, str, str], Dict[str, object]]:
    if not raw_events:
        return {}

    buy_idx: List[int] = []
    sell_idx: List[int] = []
    buy_events: List[RawBSPEvent] = []
    sell_events: List[RawBSPEvent] = []
    for i, ev in enumerate(raw_events):
        if bool(ev.is_buy):
            buy_idx.append(i)
            buy_events.append(ev)
        else:
            sell_idx.append(i)
            sell_events.append(ev)

    buy_primary = gate._predict_batch(buy_events, True)
    sell_primary = gate._predict_batch(sell_events, False)

    primary_probs = np.zeros((len(raw_events), 3), dtype=np.float32)
    for i, ridx in enumerate(buy_idx):
        primary_probs[ridx] = buy_primary[i]
    for i, ridx in enumerate(sell_idx):
        primary_probs[ridx] = sell_primary[i]

    X = np.full((len(raw_events), len(gate.meta_buy)), np.nan, dtype=np.float32)
    for i, ev in enumerate(raw_events):
        row, _, _ = gate._vectorize(ev, bool(ev.is_buy))
        X[i] = row

    direction = gate._direction_from_primary(primary_probs)
    probs = gate._meta_prob_batch(X, primary_probs, direction)
    threshold = float(cfg.signal_threshold + cfg.signal_margin)

    strict_qualified = (probs >= threshold) & (direction != 0)
    fallback_applied = not bool(np.any(strict_qualified))

    final_probs = probs.copy()
    final_direction = direction.copy()
    final_qualified = strict_qualified.copy()
    fallback_quantile = None
    fallback_threshold = None

    if fallback_applied:
        primary_conf = np.max(primary_probs[:, [0, 2]], axis=1)
        q = max(0.0, min(1.0, 1.0 - gate._EMPTY_SIGNAL_TARGET_RATE))
        adaptive_thr = float(np.quantile(primary_conf, q))
        final_qualified = primary_conf >= adaptive_thr
        final_probs = primary_conf.astype(np.float32)
        final_direction = np.array(
            [1 if bool(ev.is_buy) else -1 for ev in raw_events],
            dtype=np.int32,
        )
        fallback_quantile = q
        fallback_threshold = adaptive_thr

    out: Dict[Tuple[str, str, int, str, str, str], Dict[str, object]] = {}
    for i, ev in enumerate(raw_events):
        k = _event_key_from_raw(ev)
        out[k] = {
            "fallback_applied_for_symbol": bool(fallback_applied),
            "threshold": threshold,
            "strict": {
                "probability": float(probs[i]),
                "direction": int(direction[i]),
                "qualified": bool(strict_qualified[i]),
                "signal": int(direction[i]) if bool(strict_qualified[i]) else 0,
                "primary_probs": {
                    "class_0_sl": float(primary_probs[i, 0]),
                    "class_1_timeout": float(primary_probs[i, 1]),
                    "class_2_pt": float(primary_probs[i, 2]),
                },
            },
            "final": {
                "probability": float(final_probs[i]),
                "direction": int(final_direction[i]),
                "qualified": bool(final_qualified[i]),
                "signal": int(final_direction[i]) if bool(final_qualified[i]) else 0,
            },
            "fallback": {
                "quantile": fallback_quantile,
                "adaptive_threshold": fallback_threshold,
            },
        }
    return out


def _pick_best_candidate(df: pd.DataFrame) -> Optional[pd.Series]:
    if df.empty:
        return None
    order = df.sort_values(["probability"], ascending=False, kind="mergesort")
    return order.iloc[0]


def _as_iso_text(ts_text: str) -> str:
    ts = pd.Timestamp(ts_text, tz="UTC")
    return ts.strftime("%Y-%m-%d %H:%M:%S")


def _to_event_record(row: pd.Series) -> Dict[str, object]:
    return {
        "symbol": str(row["symbol"]),
        "exec_time": pd.Timestamp(row["exec_time"]).strftime("%Y-%m-%d %H:%M:%S"),
        "exec_target_time": pd.Timestamp(row["exec_target_time"]).strftime("%Y-%m-%d %H:%M:%S"),
        "bsp_time": str(row["bsp_time"]),
        "is_buy": bool(row["is_buy"]),
        "bsp_type": str(row["bsp_type"]),
        "bsp_types_str": str(row["bsp_types_str"]),
        "probability": float(row["probability"]),
        "qualified": bool(row["qualified"]),
        "signal": int(row["signal"]),
        "trade_price": float(row["trade_price"]),
    }


def _iter_symbol_filters(rows: Iterable[Dict[str, object]], symbols: List[str]) -> List[Dict[str, object]]:
    if not symbols:
        return list(rows)
    sset = {_normalize_symbol(s) for s in symbols}
    return [r for r in rows if _normalize_symbol(str(r.get("symbol", ""))) in sset]


def main() -> None:
    ns = _parse_args()

    run_dir = Path(ns.run_dir).resolve()
    ctx = _load_run_context(run_dir)

    detail_rows = _parse_trade_rows_from_detail_html(ctx["detail_html"])
    detail_rows = _iter_symbol_filters(detail_rows, ns.symbols)
    if ns.max_trades > 0:
        detail_rows = detail_rows[: int(ns.max_trades)]

    if not detail_rows:
        raise RuntimeError("No executed trades matched the filters")

    backtest_cfg = ctx["backtest_cfg"]
    execution_mode = str(backtest_cfg.get("execution_mode", "next_bar_open"))
    threshold = float(backtest_cfg.get("signal_threshold", 0.55))

    events_df = pd.read_csv(ctx["events_csv"])
    bars_df = pd.read_csv(ctx["bars_csv"])

    for col in ["symbol", "bsp_time", "bsp_type", "bsp_types_str"]:
        events_df[col] = events_df[col].astype(str)
    events_df["symbol"] = events_df["symbol"].map(_normalize_symbol)
    events_df["is_buy"] = events_df["is_buy"].astype(bool)
    events_df["qualified"] = events_df["qualified"].astype(bool)
    events_df["signal"] = events_df["signal"].astype(int)
    events_df["probability"] = pd.to_numeric(events_df["probability"], errors="coerce").fillna(0.0)
    events_df["trade_price"] = pd.to_numeric(events_df["trade_price"], errors="coerce").fillna(0.0)
    events_df["exec_time"] = pd.to_datetime(events_df["exec_time"], utc=True, errors="coerce")
    events_df = events_df.dropna(subset=["exec_time"]).copy()

    bars_df["symbol"] = bars_df["symbol"].map(_normalize_symbol)
    bars_df["time"] = pd.to_datetime(bars_df["time"], utc=True, errors="coerce")
    bars_df = bars_df.dropna(subset=["time"]).copy()

    bar_index_by_symbol: Dict[str, pd.DatetimeIndex] = {}
    for symbol, grp in bars_df.groupby("symbol", sort=False):
        idx = pd.DatetimeIndex(grp.sort_values("time")["time"])
        bar_index_by_symbol[symbol] = idx

    exec_target_list: List[pd.Timestamp] = []
    for row in events_df.itertuples(index=False):
        idx = bar_index_by_symbol.get(getattr(row, "symbol"))
        if idx is None or len(idx) == 0:
            exec_target_list.append(pd.NaT)
            continue
        loc = _resolve_target_loc(idx, getattr(row, "exec_time"), execution_mode)
        if loc is None:
            exec_target_list.append(pd.NaT)
        else:
            exec_target_list.append(idx[loc])
    events_df["exec_target_time"] = exec_target_list
    events_df = events_df.dropna(subset=["exec_target_time"]).copy()

    train_dir = Path(ctx["train_dir"]).resolve()
    buy_model, buy_meta, buy_fnames, buy_meta_model = load_artifacts(train_dir, "buy")
    sell_model, sell_meta, sell_fnames, sell_meta_model = load_artifacts(train_dir, "sell")

    manifest = ctx["manifest"]
    cfg = BacktestConfig(
        symbols=list((manifest.get("backtest_config", {}) or {}).get("symbols", [])) or ["BTCUSDT"],
        begin_time=str((manifest.get("train_config", {}).get("time_range", {}) or {}).get("begin", "2020-01-01")),
        end_time=str((manifest.get("train_config", {}).get("time_range", {}) or {}).get("end", "2026-12-31")),
        signal_threshold=threshold,
        execution_mode=execution_mode,
        allow_short=bool(backtest_cfg.get("allow_short", False)),
    )

    cfg.model_buy_path = str((train_dir / "model_buy.json").resolve())
    cfg.model_sell_path = str((train_dir / "model_sell.json").resolve())
    cfg.meta_buy_path = str((train_dir / "meta_buy.json").resolve())
    cfg.meta_sell_path = str((train_dir / "meta_sell.json").resolve())
    cfg.meta_model_path = str((train_dir / "meta_model.pkl").resolve())
    cfg.chan_config = dict(cfg.chan_config)

    gate = DualModelGate(cfg)

    raw_index_cache: Dict[str, Dict[Tuple[str, str, int, str, str, str], List[RawBSPEvent]]] = {}
    raw_list_cache: Dict[str, List[RawBSPEvent]] = {}
    framework_cache: Dict[str, Dict[Tuple[str, str, int, str, str, str], Dict[str, object]]] = {}

    out_trades: List[Dict[str, object]] = []
    leg_match_stats: List[TradeLegMatch] = []

    for i, tr in enumerate(detail_rows, start=1):
        symbol = _normalize_symbol(str(tr.get("symbol", "")))
        direction = str(tr.get("direction", "long")).lower()
        entry_time = _as_iso_text(str(tr.get("entry_time")))
        exit_time = _as_iso_text(str(tr.get("exit_time")))

        if direction == "short":
            entry_signal = -1
            exit_signal = 1
        else:
            entry_signal = 1
            exit_signal = -1

        e_ts = pd.Timestamp(entry_time, tz="UTC")
        x_ts = pd.Timestamp(exit_time, tz="UTC")

        sym_events = events_df[events_df["symbol"] == symbol]

        entry_cands = sym_events[
            (sym_events["exec_target_time"] == e_ts)
            & (sym_events["signal"] == entry_signal)
        ]
        exit_cands = sym_events[
            (sym_events["exec_target_time"] == x_ts)
            & (sym_events["signal"] == exit_signal)
        ]

        entry_pick = _pick_best_candidate(entry_cands)
        exit_pick = _pick_best_candidate(exit_cands)

        def _leg_payload(leg_name: str, pick: Optional[pd.Series], expected: int, cands: pd.DataFrame):
            if pick is None:
                leg_match_stats.append(
                    TradeLegMatch(
                        symbol=symbol,
                        leg=leg_name,
                        expected_signal=int(expected),
                        trade_time=entry_time if leg_name == "entry" else exit_time,
                        matched_events=int(len(cands)),
                        selected_event=None,
                    )
                )
                return {
                    "match": {
                        "matched_events": int(len(cands)),
                        "selected": None,
                    },
                    "decision_chain": None,
                    "warning": "No matched signal event for this trade leg",
                }

            selected = _to_event_record(pick)
            leg_match_stats.append(
                TradeLegMatch(
                    symbol=symbol,
                    leg=leg_name,
                    expected_signal=int(expected),
                    trade_time=entry_time if leg_name == "entry" else exit_time,
                    matched_events=int(len(cands)),
                    selected_event=selected,
                )
            )

            if symbol not in raw_index_cache:
                bars = load_symbol_bars(cfg, symbol)
                raw_events = extract_raw_bsp_events(cfg, symbol)
                raw_events = enrich_raw_events_with_feature_engine(
                    symbol=symbol,
                    bars=bars,
                    raw_events=raw_events,
                )

                idx_map: Dict[Tuple[str, str, int, str, str, str], List[RawBSPEvent]] = {}
                for _ev in raw_events:
                    _k = _event_key_from_raw(_ev)
                    idx_map.setdefault(_k, []).append(_ev)

                raw_index_cache[symbol] = idx_map
                raw_list_cache[symbol] = raw_events
                framework_cache[symbol] = _compute_symbol_framework_details(
                    gate=gate,
                    cfg=cfg,
                    raw_events=raw_events,
                )
            raw_idx = raw_index_cache[symbol]

            key = _event_key(
                symbol=selected["symbol"],
                exec_time=pd.Timestamp(selected["exec_time"], tz="UTC"),
                is_buy=bool(selected["is_buy"]),
                bsp_time=str(selected["bsp_time"]),
                bsp_type=str(selected["bsp_type"]),
                bsp_types_str=str(selected["bsp_types_str"]),
            )
            raw_cands = raw_idx.get(key, [])

            if not raw_cands:
                return {
                    "match": {
                        "matched_events": int(len(cands)),
                        "selected": selected,
                    },
                    "decision_chain": None,
                    "warning": "Matched scored event but raw feature event not found",
                }

            raw_ev = raw_cands[0]
            sample = _sample_from_raw(raw_ev)

            if bool(sample.is_buy):
                X = build_feature_vector(sample, buy_meta)
                res = analyze_one_sample(
                    booster=buy_model,
                    meta_model=buy_meta_model,
                    fnames=buy_fnames,
                    X=X,
                    threshold=threshold,
                )
                side = "buy"
            else:
                X = build_feature_vector(sample, sell_meta)
                res = analyze_one_sample(
                    booster=sell_model,
                    meta_model=sell_meta_model,
                    fnames=sell_fnames,
                    X=X,
                    threshold=threshold,
                )
                side = "sell"

            res["model_side"] = side
            res["csv_probability"] = float(selected.get("probability", 0.0))
            res["csv_signal"] = int(selected.get("signal", 0))
            res["csv_qualified"] = bool(selected.get("qualified", False))

            framework_detail = framework_cache.get(symbol, {}).get(key)

            return {
                "match": {
                    "matched_events": int(len(cands)),
                    "selected": selected,
                },
                "decision_chain_model": res,
                "decision_chain_framework": framework_detail,
                "warning": None,
            }

        entry_payload = _leg_payload("entry", entry_pick, entry_signal, entry_cands)
        exit_payload = _leg_payload("exit", exit_pick, exit_signal, exit_cands)

        out_trades.append(
            {
                "trade_no": i,
                "symbol": symbol,
                "direction": direction,
                "entry_time": entry_time,
                "exit_time": exit_time,
                "holding_bars": int(tr.get("holding_bars", 0) or 0),
                "entry_price": float(tr.get("entry_price", np.nan)),
                "exit_price": float(tr.get("exit_price", np.nan)),
                "pnl": float(tr.get("pnl", np.nan)),
                "return_pct": float(tr.get("return_pct", np.nan)),
                "entry_leg": entry_payload,
                "exit_leg": exit_payload,
            }
        )

    match_ok = sum(1 for x in leg_match_stats if x.selected_event is not None)
    match_total = len(leg_match_stats)

    summary = {
        "run_dir": str(run_dir),
        "train_dir": str(train_dir),
        "execution_mode": execution_mode,
        "signal_threshold": threshold,
        "requested_symbols": ns.symbols,
        "analyzed_trades": len(out_trades),
        "leg_match_rate": (float(match_ok) / float(match_total)) if match_total else 0.0,
        "matched_legs": int(match_ok),
        "total_legs": int(match_total),
    }

    payload = {
        "summary": summary,
        "trades": out_trades,
    }

    output_json = (
        Path(ns.output_json).resolve()
        if ns.output_json
        else (run_dir / "backtest" / "trade_decision_chain.json").resolve()
    )
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print("=" * 72)
    print("Executed Trade Decision Chain Analysis")
    print(f"Run Dir        : {run_dir}")
    print(f"Output         : {output_json}")
    print(f"Analyzed Trades: {len(out_trades)}")
    print(f"Leg Match      : {match_ok}/{match_total} ({summary['leg_match_rate']:.2%})")
    print("=" * 72)


if __name__ == "__main__":
    main()
