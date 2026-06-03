from __future__ import annotations

import math
import json
import os
import sqlite3
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Backtest.config import DEFAULT_CHAN_CONFIG
from ChanConfig import CChanConfig
from RustCore import RustChanEngine, rust_config_path_from_chan_config


DATA_ROOT = Path(os.getenv("CHAN_TV_DATA_ROOT", PROJECT_ROOT / "data"))
DEFAULT_SYMBOL = "BTCUSDT"

BACKTEST_FILES = {
    "v1_bsp2_fullsize": PROJECT_ROOT / "result/btc_futures_v1_wf_validlabel_no_after_timeout_fullsize/test_2026/executed_trades.csv",
    "v1_bsp2_structure_sizing": PROJECT_ROOT / "result/btc_futures_v1_wf_validlabel_tier_sizing_a_boost/test_2026/executed_trades.csv",
}

BACKTEST_RESULT_ROOTS = {
    "v1_bsp2_fullsize": PROJECT_ROOT / "result/btc_futures_v1_wf_validlabel_no_after_timeout_fullsize",
    "v1_bsp2_structure_sizing": PROJECT_ROOT / "result/btc_futures_v1_wf_validlabel_tier_sizing_a_boost",
}

MODEL_REGISTRY_PATH = PROJECT_ROOT / "ML/MODEL_REGISTRY.md"
MODEL_ARTIFACT_DIRS = {
    "btc_futures_v1": PROJECT_ROOT / "result/ml/btc_futures_v1_wf_validlabel_tier_sizing_a_boost/test_2026",
}
MODEL_BACKTEST_DIRS = {
    "btc_futures_v1": [
        PROJECT_ROOT / "result/btc_futures_v1_wf_validlabel_no_after_timeout_fullsize",
        PROJECT_ROOT / "result/btc_futures_v1_wf_validlabel_tier_sizing_a_boost",
    ],
}

DRYRUN_BOTS = {
    "fullsize": {
        "name": "不定仓",
        "strategy": "ChanMLBtcFuturesV1FullSizeStrategy",
        "db_path": PROJECT_ROOT / "user_data/chan_v1_fullsize.dryrun.sqlite",
        "api_port": 8091,
    },
    "structure_sizing": {
        "name": "结构风险定仓",
        "strategy": "ChanMLBtcFuturesV1StructureSizingStrategy",
        "db_path": PROJECT_ROOT / "user_data/chan_v1_structure_sizing.dryrun.sqlite",
        "api_port": 8092,
    },
}

FREQ_RULES = {
    "1m": "1min",
    "3m": "3min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1h",
    "2h": "2h",
    "4h": "4h",
    "1d": "1d",
}


app = FastAPI(title="Chan TradingView local adapter")


def _json_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return None
        return value
    if isinstance(value, pd.Timestamp):
        return int(value.timestamp())
    return value


def _finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return float(default)
    return out if math.isfinite(out) else float(default)


def _parse_time(value: str | None) -> pd.Timestamp | None:
    if not value:
        return None
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    if ts.tzinfo is None:
        ts = ts.tz_localize("Asia/Shanghai")
    return ts.tz_convert("UTC")


def _empty_chan(symbol: str, freq: str) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "freq": freq,
        "bis": [],
        "segs": [],
        "zs": [],
        "segzs": [],
        "bsps": [],
        "seg_bsps": [],
    }


def _normalize_symbol(symbol: str | None) -> str:
    text = (symbol or DEFAULT_SYMBOL).upper().strip()
    if ":" in text:
        left, right = text.split(":", 1)
        if "/" in left and right in {"USDT", "USD", "PERP"}:
            text = left
        elif left in {"BINANCE", "BYBIT", "OKX", "GATE", "HUOBI"}:
            text = right
        else:
            text = right
    text = text.replace("-", "_").replace("/", "_")
    if text.endswith("_PERP"):
        text = text[:-5]
    if text.endswith("_FUTURES"):
        text = text[:-8]
    text = text.replace("_USDT_USDT", "USDT")
    text = text.replace("_USDT", "USDT")
    text = "".join(ch for ch in text if ch.isalnum())
    if not text.endswith("USDT"):
        text = f"{text}USDT"
    return text


def _symbol_file(symbol: str) -> Path:
    normalized = _normalize_symbol(symbol)
    base = normalized[:-4]
    return DATA_ROOT / f"{base}_USDT_USDT-1m-futures.parquet"


def _available_symbols() -> list[str]:
    if not DATA_ROOT.exists():
        return [DEFAULT_SYMBOL]
    symbols = []
    for path in sorted(DATA_ROOT.glob("*_USDT_USDT-1m-futures.parquet")):
        base = path.name.split("_USDT_USDT-", 1)[0]
        symbols.append(f"{base.replace('_', '')}USDT")
    return symbols or [DEFAULT_SYMBOL]


def _time_by_klu_idx(data: pd.DataFrame, idx: Any) -> int | None:
    try:
        i = int(idx)
    except Exception:
        return None
    if i < 0 or i >= len(data):
        return None
    return int(pd.Timestamp(data.iloc[i]["time"]).timestamp())


def _direction(value: Any) -> str:
    return "UP" if str(value).lower().startswith("up") else "DOWN"


def _feature_pairs_to_dict(features: Any) -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(features, dict):
        iterator = features.items()
    else:
        iterator = features or []
    for item in iterator:
        try:
            key, value = item
        except Exception:
            continue
        value = _finite_float(value, float("nan"))
        if math.isfinite(value):
            out[str(key)] = value
    return out


def _normalize_bsp_type(value: Any) -> str:
    text = str(value).strip().lower().replace("'", "p")
    mapping = {
        "1": "T1",
        "1p": "T1P",
        "2": "T2",
        "2s": "T2S",
        "3a": "T3A",
        "3b": "T3B",
    }
    return mapping.get(text, text.upper())


def _format_bis(items: list[dict[str, Any]], data: pd.DataFrame) -> list[dict[str, Any]]:
    out = []
    for item in items:
        t0 = _time_by_klu_idx(data, item.get("begin_klu_idx"))
        t1 = _time_by_klu_idx(data, item.get("end_klu_idx"))
        if t0 is None or t1 is None:
            continue
        out.append(
            {
                "idx": int(item.get("idx", len(out))),
                "dir": _direction(item.get("direction")),
                "t0": t0,
                "p0": _finite_float(item.get("begin_val")),
                "t1": t1,
                "p1": _finite_float(item.get("end_val")),
                "sure": bool(item.get("is_sure", False)),
                "seg_idx": _json_value(item.get("parent_seg_idx", item.get("seg_idx"))),
            }
        )
    return out


def _format_segs(items: list[dict[str, Any]], data: pd.DataFrame) -> list[dict[str, Any]]:
    out = []
    for item in items:
        t0 = _time_by_klu_idx(data, item.get("begin_klu_idx"))
        t1 = _time_by_klu_idx(data, item.get("end_klu_idx"))
        if t0 is None or t1 is None:
            continue
        out.append(
            {
                "id": int(item.get("idx", len(out))),
                "dir": _direction(item.get("direction")),
                "t0": t0,
                "p0": _finite_float(item.get("begin_val")),
                "t1": t1,
                "p1": _finite_float(item.get("end_val")),
                "sure": bool(item.get("is_sure", False)),
                "zs_count": int(item.get("multi_bi_zs_cnt", 0) or 0),
                "element_count": int(item.get("element_cnt", 0) or 0),
            }
        )
    return out


def _format_zs(items: list[dict[str, Any]], data: pd.DataFrame) -> list[dict[str, Any]]:
    out = []
    for idx, item in enumerate(items):
        t0 = _time_by_klu_idx(data, item.get("begin_klu_id"))
        t1 = _time_by_klu_idx(data, item.get("end_klu_id"))
        if t0 is None or t1 is None:
            continue
        out.append(
            {
                "idx": idx,
                "dir": _direction(item.get("direction")),
                "t0": t0,
                "t1": t1,
                "low": _finite_float(item.get("low")),
                "high": _finite_float(item.get("high")),
                "peak_low": _finite_float(item.get("peak_low", item.get("low"))),
                "peak_high": _finite_float(item.get("peak_high", item.get("high"))),
                "is_sure": bool(item.get("is_sure", False)),
                "sub_zs_count": int(item.get("sub_zs_count", 0) or 0),
                "element_count": int(len(item.get("element_list") or [])),
            }
        )
    return out


def _format_bsps(items: list[dict[str, Any]], data: pd.DataFrame) -> list[dict[str, Any]]:
    out = []
    for item in items:
        t = _time_by_klu_idx(data, item.get("klu_idx"))
        if t is None:
            raw_time = item.get("klu_time")
            try:
                t = int(float(raw_time) / 1000) if raw_time is not None else None
            except Exception:
                t = None
        if t is None:
            continue
        out.append(
            {
                "element_idx": int(item.get("element", -1) or -1),
                "klu_idx": int(item.get("klu_idx", -1) or -1),
                "t": int(t),
                "is_buy": bool(item.get("is_buy", False)),
                "is_segbsp": bool(item.get("is_segbsp", False)),
                "is_target": bool(item.get("is_target", True)),
                "types": [_normalize_bsp_type(x) for x in (item.get("types") or [])],
                "features": _feature_pairs_to_dict(item.get("features")),
            }
        )
    return out


_bars_cache: dict[tuple[str, str], pd.DataFrame] = {}
_chan_cache: dict[tuple[str, str, str, str], dict[str, Any]] = {}
_trade_cache: dict[str, pd.DataFrame] = {}


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _parse_registry_table() -> dict[str, dict[str, Any]]:
    if not MODEL_REGISTRY_PATH.exists():
        return {}
    text = MODEL_REGISTRY_PATH.read_text(encoding="utf-8", errors="ignore")
    rows: dict[str, dict[str, Any]] = {}
    in_table = False
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("| 路线 |"):
            in_table = True
            continue
        if not in_table:
            continue
        if not line.startswith("|"):
            if rows:
                break
            continue
        if set(line.replace("|", "").strip()) <= {"-", ":"}:
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 5:
            continue
        route = cells[0].strip("` ")
        rows[route] = {
            "route": route,
            "goal": cells[1],
            "label": cells[2],
            "features": cells[3],
            "status": cells[4],
        }
    return rows


def _extract_registry_sections(routes: set[str]) -> dict[str, str]:
    if not MODEL_REGISTRY_PATH.exists():
        return {}
    lines = MODEL_REGISTRY_PATH.read_text(encoding="utf-8", errors="ignore").splitlines()
    sections: dict[str, str] = {}
    for i, line in enumerate(lines):
        stripped = line.strip()
        for route in routes:
            if stripped.startswith("##") and f"`{route}`" in stripped:
                body: list[str] = []
                for nxt in lines[i + 1 :]:
                    if nxt.startswith("## "):
                        break
                    body.append(nxt)
                sections[route] = "\n".join(body).strip()
    return sections


def _metric_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    keys = (
        "trades",
        "win_rate",
        "total_return",
        "profit_factor",
        "max_drawdown",
        "qualified_candidates",
        "candidate_events",
    )
    out = {k: _json_value(metrics.get(k)) for k in keys if k in metrics}
    return out


def _find_backtest_metrics(route: str) -> list[dict[str, Any]]:
    base = PROJECT_ROOT / "result"
    candidates = []
    for root in MODEL_BACKTEST_DIRS.get(route, []):
        candidates.extend(root.glob("**/backtest_metrics.json"))
    for path in base.glob(f"{route}/**/backtest_metrics.json"):
        candidates.append(path)
    direct = base / route / "backtest_metrics.json"
    if direct.exists():
        candidates.append(direct)
    seen: set[Path] = set()
    out: list[dict[str, Any]] = []
    for path in sorted(candidates, key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True):
        if path in seen:
            continue
        seen.add(path)
        payload = _read_json(path)
        out.append(
            {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "updated_at": int(path.stat().st_mtime),
                "metrics": _metric_summary(payload),
            }
        )
        if len(out) >= 5:
            break
    return out


def _model_artifact_info(route: str) -> dict[str, Any]:
    model_dir = MODEL_ARTIFACT_DIRS.get(route, PROJECT_ROOT / "result/ml" / route)
    metrics = _read_json(model_dir / "metrics.json")
    files = []
    if model_dir.exists():
        for path in sorted(model_dir.iterdir()):
            if path.is_file():
                files.append(
                    {
                        "name": path.name,
                        "path": str(path.relative_to(PROJECT_ROOT)),
                        "size": path.stat().st_size,
                        "updated_at": int(path.stat().st_mtime),
                    }
                )
    return {
        "model_dir": str(model_dir.relative_to(PROJECT_ROOT)) if model_dir.exists() else "",
        "has_artifact": model_dir.exists(),
        "updated_at": int(model_dir.stat().st_mtime) if model_dir.exists() else None,
        "files": files[:12],
        "metrics": metrics,
    }


def _sqlite_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    try:
        return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    except Exception:
        return set()


def _first_existing(cols: set[str], names: tuple[str, ...]) -> str | None:
    for name in names:
        if name in cols:
            return name
    return None


def _row_value(row: sqlite3.Row, key: str | None, default: Any = None) -> Any:
    if not key:
        return default
    try:
        return row[key]
    except Exception:
        return default


def _read_dryrun_db(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "exists": False,
            "open_trades": [],
            "recent_closed": [],
            "summary": {"total_trades": 0, "open_trades": 0, "closed_trades": 0},
        }
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        tables = {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        if "trades" not in tables:
            return {
                "exists": True,
                "tables": sorted(tables),
                "open_trades": [],
                "recent_closed": [],
                "summary": {"total_trades": 0, "open_trades": 0, "closed_trades": 0},
            }
        cols = _sqlite_columns(conn, "trades")
        pair_col = _first_existing(cols, ("pair", "symbol"))
        open_date_col = _first_existing(cols, ("open_date", "open_date_utc"))
        close_date_col = _first_existing(cols, ("close_date", "close_date_utc"))
        open_rate_col = _first_existing(cols, ("open_rate",))
        close_rate_col = _first_existing(cols, ("close_rate",))
        stake_col = _first_existing(cols, ("stake_amount", "stake_amount_filled"))
        amount_col = _first_existing(cols, ("amount",))
        profit_col = _first_existing(cols, ("close_profit", "close_profit_abs", "realized_profit"))
        profit_abs_col = _first_existing(cols, ("close_profit_abs", "profit_abs", "realized_profit"))
        is_open_col = _first_existing(cols, ("is_open",))
        enter_tag_col = _first_existing(cols, ("enter_tag", "buy_tag"))
        exit_reason_col = _first_existing(cols, ("exit_reason", "sell_reason"))
        side_col = _first_existing(cols, ("trading_mode", "is_short"))

        def norm_trade(row: sqlite3.Row) -> dict[str, Any]:
            is_short = _row_value(row, "is_short", None) if "is_short" in cols else None
            side = "short" if bool(is_short) else "long"
            return {
                "id": _row_value(row, "id"),
                "pair": _row_value(row, pair_col, ""),
                "side": side,
                "is_open": bool(_row_value(row, is_open_col, 0)),
                "open_date": _row_value(row, open_date_col, ""),
                "close_date": _row_value(row, close_date_col, ""),
                "open_rate": _json_value(_row_value(row, open_rate_col)),
                "close_rate": _json_value(_row_value(row, close_rate_col)),
                "stake_amount": _json_value(_row_value(row, stake_col)),
                "amount": _json_value(_row_value(row, amount_col)),
                "profit_ratio": _json_value(_row_value(row, profit_col)),
                "profit_abs": _json_value(_row_value(row, profit_abs_col)),
                "enter_tag": _row_value(row, enter_tag_col, ""),
                "exit_reason": _row_value(row, exit_reason_col, ""),
                "raw_side": _row_value(row, side_col, ""),
            }

        order_col = close_date_col or open_date_col or "id"
        open_rows = conn.execute(
            f"SELECT * FROM trades WHERE {is_open_col}=1 ORDER BY {open_date_col or 'id'} DESC LIMIT 50"
            if is_open_col
            else "SELECT * FROM trades ORDER BY id DESC LIMIT 50"
        ).fetchall()
        closed_where = f"WHERE {is_open_col}=0" if is_open_col else ""
        closed_rows = conn.execute(f"SELECT * FROM trades {closed_where} ORDER BY {order_col} DESC LIMIT 50").fetchall()
        total = int(conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0])
        open_count = int(conn.execute(f"SELECT COUNT(*) FROM trades WHERE {is_open_col}=1").fetchone()[0]) if is_open_col else 0
        closed_count = int(conn.execute(f"SELECT COUNT(*) FROM trades WHERE {is_open_col}=0").fetchone()[0]) if is_open_col else total
        wins = 0
        total_profit_abs = 0.0
        total_profit_ratio = 0.0
        if profit_abs_col:
            for row in conn.execute(f"SELECT {profit_abs_col} FROM trades {closed_where}").fetchall():
                val = _finite_float(row[0], 0.0)
                wins += int(val > 0)
                total_profit_abs += float(val)
        if profit_col:
            for row in conn.execute(f"SELECT {profit_col} FROM trades {closed_where}").fetchall():
                total_profit_ratio += _finite_float(row[0], 0.0)
        return {
            "exists": True,
            "updated_at": int(path.stat().st_mtime),
            "summary": {
                "total_trades": total,
                "open_trades": open_count,
                "closed_trades": closed_count,
                "wins": wins,
                "win_rate": (wins / closed_count if closed_count else None),
                "total_profit_abs": total_profit_abs,
                "total_profit_ratio_sum": total_profit_ratio,
            },
            "open_trades": [norm_trade(row) for row in open_rows],
            "recent_closed": [norm_trade(row) for row in closed_rows],
        }
    finally:
        conn.close()


def _load_1m(symbol: str) -> pd.DataFrame:
    symbol = _normalize_symbol(symbol)
    key = (symbol, "1m")
    if key not in _bars_cache:
        path = _symbol_file(symbol)
        if not path.exists():
            raise HTTPException(status_code=404, detail=f"missing data file for {symbol}: {path}")
        data = pd.read_parquet(path)
        data = data.rename(
            columns={"date": "time", "open": "o", "high": "h", "low": "l", "close": "c", "volume": "v"}
        )
        data["time"] = pd.to_datetime(data["time"], utc=True, errors="coerce")
        data = data.dropna(subset=["time", "o", "h", "l", "c"]).sort_values("time")
        _bars_cache[key] = data[["time", "o", "h", "l", "c", "v"]].reset_index(drop=True)
    return _bars_cache[key]


def _bars_for_freq(symbol: str, freq: str) -> pd.DataFrame:
    symbol = _normalize_symbol(symbol)
    if freq not in FREQ_RULES:
        raise HTTPException(status_code=400, detail=f"unsupported freq: {freq}")
    key = (symbol, freq)
    if key in _bars_cache:
        return _bars_cache[key]
    base = _load_1m(symbol)
    if freq == "1m":
        return base
    rule = FREQ_RULES[freq]
    resampled = (
        base.set_index("time")
        .resample(rule, label="left", closed="left")
        .agg({"o": "first", "h": "max", "l": "min", "c": "last", "v": "sum"})
        .dropna(subset=["o", "h", "l", "c"])
        .reset_index()
    )
    _bars_cache[key] = resampled
    return resampled


def _format_bars(frame: pd.DataFrame) -> list[dict[str, Any]]:
    out = []
    for row in frame.itertuples(index=False):
        out.append(
            {
                "t": int(row.time.timestamp()),
                "o": float(row.o),
                "h": float(row.h),
                "l": float(row.l),
                "c": float(row.c),
                "v": float(row.v) if pd.notna(row.v) else 0.0,
            }
        )
    return out


def _chan_slice(symbol: str, freq: str, start: pd.Timestamp | None, end: pd.Timestamp | None) -> dict[str, Any]:
    symbol = _normalize_symbol(symbol)
    data = _bars_for_freq(symbol, freq)
    if end is None:
        end = data["time"].max()
    if start is None:
        start_key = ""
        calc_data = data.loc[data["time"] <= end].tail(5000).reset_index(drop=True)
    else:
        start_key = str(int(start.timestamp()))
        end_pos = int(data["time"].searchsorted(end, side="right"))
        start_pos = max(0, int(data["time"].searchsorted(start, side="left")) - 2500)
        calc_data = data.iloc[start_pos:end_pos].reset_index(drop=True)
    end_key = str(int(pd.Timestamp(end).timestamp()))
    cache_key = (symbol, freq, start_key, end_key)
    if cache_key in _chan_cache:
        return _chan_cache[cache_key]
    if calc_data.empty:
        return _empty_chan(symbol, freq)

    chan_cfg = dict(DEFAULT_CHAN_CONFIG)
    chan_cfg.update({"use_rust_core": True, "trigger_step": True, "skip_step": 0})
    engine = RustChanEngine(freq=freq, config_path=rust_config_path_from_chan_config(CChanConfig(chan_cfg)))
    for row in calc_data.itertuples(index=False):
        engine.push_bar(row.time, row.o, row.h, row.l, row.c, row.v)
        engine.cal_seg_and_zs()
    snapshot = engine.snapshot()
    payload = {
        "symbol": symbol,
        "freq": freq,
        "bis": _format_bis(snapshot.get("bis") or [], calc_data),
        "segs": _format_segs(snapshot.get("segs") or [], calc_data),
        "zs": _format_zs(snapshot.get("bzs") or [], calc_data),
        "segzs": _format_zs(snapshot.get("szs") or [], calc_data),
        "bsps": _format_bsps(snapshot.get("bi_bsp") or [], calc_data),
        "seg_bsps": _format_bsps(snapshot.get("seg_bsp") or [], calc_data),
    }
    if start is not None:
        start_ts = int(start.timestamp())
        end_ts = int(pd.Timestamp(end).timestamp())
        for key in ("bis", "segs", "zs", "segzs"):
            payload[key] = [
                item for item in payload[key]
                if int(item.get("t1", item.get("t", 0))) >= start_ts and int(item.get("t0", item.get("t", 0))) <= end_ts
            ]
        for key in ("bsps", "seg_bsps"):
            payload[key] = [item for item in payload[key] if start_ts <= int(item["t"]) <= end_ts]
    _chan_cache[cache_key] = payload
    return payload


def _load_trades(strategy: str) -> pd.DataFrame:
    if strategy not in BACKTEST_FILES:
        raise HTTPException(status_code=404, detail=f"unknown strategy: {strategy}")
    if strategy in _trade_cache:
        return _trade_cache[strategy]
    path = BACKTEST_FILES[strategy]
    if not path.exists():
        data = pd.DataFrame(
            columns=["entry_time", "exit_time", "entry_price", "exit_price", "side", "pnl", "exit_reason"]
        )
    else:
        data = pd.read_csv(path)
    for col in ("entry_time", "exit_time", "exec_time"):
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], utc=True, errors="coerce")
    for col in ("entry_price", "exit_price", "pnl", "net_return"):
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.dropna(subset=["entry_time", "exit_time", "entry_price", "exit_price"]).sort_values("entry_time")
    _trade_cache[strategy] = data.reset_index(drop=True)
    return _trade_cache[strategy]


def _backtest_result_files() -> list[tuple[str, str, Path]]:
    items: list[tuple[str, str, Path]] = []
    for strategy, root in BACKTEST_RESULT_ROOTS.items():
        if not root.exists():
            continue
        for path in sorted(root.glob("test_*/executed_trades.csv")):
            result_id = path.relative_to(PROJECT_ROOT).as_posix()
            period = path.parent.name.replace("test_", "")
            items.append((result_id, strategy, path))
    return items


def _backtest_result_path(result_id: str) -> tuple[str, Path]:
    normalized = result_id.replace("\\", "/").strip("/")
    for known_id, strategy, path in _backtest_result_files():
        if known_id == normalized:
            return strategy, path
    raise HTTPException(status_code=404, detail=f"unknown backtest result: {result_id}")


def _read_metric_for_trade_file(path: Path) -> dict[str, Any]:
    payload = _read_json(path.parent / "backtest_metrics.json")
    return _metric_summary(payload)


def _load_result_trades(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    data = pd.read_csv(path)
    for col in ("entry_time", "exit_time", "exec_time", "actual_exec_time"):
        if col in data.columns:
            data[col] = pd.to_datetime(data[col], utc=True, errors="coerce")
    for col in ("entry_price", "exit_price", "pnl", "gross_return", "net_return", "probability", "stake_multiplier", "leverage"):
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")
    return data.dropna(subset=["entry_time", "exit_time", "entry_price", "exit_price"]).reset_index(drop=True)


def _trade_side(row: pd.Series) -> str:
    side_text = str(row.get("side", "")).lower()
    if side_text in {"long", "short"}:
        return side_text
    is_buy = row.get("is_buy", None)
    if pd.notna(is_buy):
        return "long" if bool(is_buy) else "short"
    return "long"


def _format_result_trade(idx: int, row: pd.Series) -> dict[str, Any]:
    side = _trade_side(row)
    entry_time = pd.Timestamp(row["entry_time"])
    exit_time = pd.Timestamp(row["exit_time"])
    net_return = _finite_float(row.get("net_return"), float("nan"))
    pnl = _finite_float(row.get("pnl"), float("nan"))
    symbol = _normalize_symbol(str(row.get("symbol", DEFAULT_SYMBOL)))
    return {
        "id": int(idx),
        "symbol": symbol,
        "side": side,
        "entry_time": int(entry_time.timestamp()),
        "exit_time": int(exit_time.timestamp()),
        "entry_price": _finite_float(row.get("entry_price")),
        "exit_price": _finite_float(row.get("exit_price")),
        "pnl": _json_value(pnl),
        "gross_return": _json_value(_finite_float(row.get("gross_return"), float("nan"))),
        "net_return": _json_value(net_return),
        "exit_reason": str(row.get("exit_reason", "")),
        "probability": _json_value(_finite_float(row.get("probability"), float("nan"))),
        "stake_multiplier": _json_value(_finite_float(row.get("stake_multiplier"), float("nan"))),
        "leverage": _json_value(_finite_float(row.get("leverage"), float("nan"))),
        "bsp_types": str(row.get("bsp_types_str", row.get("bsp_type", ""))),
        "structure_tier": str(row.get("structure_confidence_tier", "")),
        "is_win": bool((net_return if math.isfinite(net_return) else pnl) > 0),
    }


def _fill_from_trade(row: pd.Series, action: str) -> dict[str, Any]:
    side_text = str(row.get("side", "")).lower()
    is_long = bool(row.get("is_buy", False)) or side_text == "long"
    if action == "OPEN":
        side = "BUY" if is_long else "SELL"
        t = row["entry_time"]
        price = row["entry_price"]
        realized = 0.0
        reason = "OPEN"
    else:
        side = "SELL" if is_long else "BUY"
        t = row["exit_time"]
        price = row["exit_price"]
        realized = float(row.get("pnl", 0.0) or 0.0)
        reason = str(row.get("exit_reason", "CLOSE"))
    return {
        "t": int(pd.Timestamp(t).timestamp()),
        "side": side,
        "action": action,
        "qty": 1,
        "price": float(price),
        "fee": 0.0,
        "realized_pnl": realized,
        "reason": reason,
    }


class BacktestStartBody(BaseModel):
    exchange: str = "BINANCE"
    symbol: str
    freq: str
    at: str
    strategy: str
    params: dict[str, Any] | None = None
    init_cash: float = 100000.0
    slippage_ticks: float | None = None
    margin_rate: float | None = None
    fee_rate: float | None = None
    fee_fixed: float | None = None


@dataclass
class Session:
    exchange: str
    symbol: str
    freq: str
    strategy: str
    init_cash: float
    cursor_time: pd.Timestamp
    cursor_idx: int
    fills: list[dict[str, Any]] = field(default_factory=list)
    closed_trade_indices: set[int] = field(default_factory=set)


SESSIONS: dict[str, Session] = {}


@app.get("/health")
def health() -> dict[str, Any]:
    return {"ok": True, "data_root": str(DATA_ROOT), "symbols": _available_symbols()}


@app.get("/symbols")
def symbols(q: str = "", exchange: str = "") -> dict[str, Any]:
    if exchange and exchange.upper() not in {"BINANCE", ""}:
        return {"symbols": []}
    items = []
    query = q.lower().strip()
    for symbol in _available_symbols():
        item = {"exchange": "BINANCE", "symbol": symbol, "ticker": f"BINANCE:{symbol}"}
        text = f"{item['exchange']} {item['symbol']} {item['ticker']}".lower()
        if query and query not in text:
            continue
        items.append(item)
    return {"symbols": items}


@app.get("/contract/{symbol}")
def contract(symbol: str) -> dict[str, Any]:
    return {
        "symbol": symbol.upper(),
        "exchange": "BINANCE",
        "multiplier": 1,
        "price_tick": 0.1,
        "margin_rate": 0.05,
        "fee_rate": 0.0004,
        "fee_fixed": 0.0,
        "intraday_fee_mult": 1.0,
    }


@app.get("/bars")
def bars(
    symbol: str,
    freq: str,
    exchange: str = "BINANCE",
    from_: str | None = Query(default=None, alias="from"),
    to: str | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    del exchange
    normalized = _normalize_symbol(symbol)
    data = _bars_for_freq(normalized, freq)
    start = _parse_time(from_)
    end = _parse_time(to)
    if end is None:
        end = data["time"].max()
    mask = data["time"] <= end
    if start is not None:
        mask &= data["time"] >= start
    subset = data.loc[mask]
    if limit:
        subset = subset.tail(max(1, min(int(limit), 5000)))
    return {"symbol": normalized, "freq": freq, "n": int(len(subset)), "bars": _format_bars(subset)}


@app.post("/chan/build")
def chan_build(symbol: str, freq: str, exchange: str = "BINANCE", force: bool = False) -> dict[str, Any]:
    del exchange, force
    normalized = _normalize_symbol(symbol)
    data = _bars_for_freq(normalized, freq)
    payload = _chan_slice(normalized, freq, None, None)
    return {
        "exchange": "BINANCE",
        "symbol": normalized,
        "freq": freq,
        "bars": int(len(data)),
        "bis": int(len(payload["bis"])),
        "segs": int(len(payload["segs"])),
        "zs": int(len(payload["zs"])),
        "segzs": int(len(payload["segzs"])),
        "bsps": int(len(payload["bsps"])),
        "csv_mtime": int(_symbol_file(normalized).stat().st_mtime),
    }


@app.get("/chan")
def chan(
    symbol: str,
    freq: str,
    exchange: str = "BINANCE",
    from_: str | None = Query(default=None, alias="from"),
    to: str | None = None,
) -> dict[str, Any]:
    del exchange
    start = _parse_time(from_)
    end = _parse_time(to)
    return _chan_slice(symbol, freq, start, end)


@app.post("/replay/start")
def replay_start(symbol: str, freq: str, at: str, exchange: str = "BINANCE") -> dict[str, Any]:
    cutoff = _parse_time(at)
    if cutoff is None:
        raise HTTPException(status_code=400, detail="invalid replay start time")
    symbol = _normalize_symbol(symbol)
    data = _bars_for_freq(symbol, freq)
    cursor = int(data["time"].searchsorted(cutoff, side="right") - 1)
    sid = uuid.uuid4().hex
    SESSIONS[sid] = Session(exchange, symbol, freq, "replay", 0.0, cutoff, max(cursor, 0))
    return {
        "session_id": sid,
        "exchange": exchange,
        "symbol": symbol,
        "freq": freq,
        "remaining": max(0, int(len(data) - max(cursor + 1, 0))),
        "counts": {},
    }


@app.post("/replay/{session_id}/step")
def replay_step(session_id: str, n: int = 1) -> dict[str, Any]:
    sess = SESSIONS.get(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="session not found")
    data = _bars_for_freq(sess.symbol, sess.freq)
    start = sess.cursor_idx + 1
    end = min(len(data), start + max(1, int(n)))
    new = data.iloc[start:end]
    if not new.empty:
        sess.cursor_idx = end - 1
        sess.cursor_time = pd.Timestamp(new.iloc[-1]["time"])
    return {
        "session_id": session_id,
        "pushed": int(len(new)),
        "remaining": max(0, int(len(data) - end)),
        "bars": _format_bars(new),
        "counts": {},
        "delta": {},
        "new": {},
    }


@app.get("/replay/{session_id}/chan")
def replay_chan(
    session_id: str,
    from_: str | None = Query(default=None, alias="from"),
    to: str | None = None,
) -> dict[str, Any]:
    sess = SESSIONS.get(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="session not found")
    start = _parse_time(from_)
    end = _parse_time(to)
    return _chan_slice(sess.symbol, sess.freq, start, end)


@app.delete("/replay/{session_id}")
def replay_stop(session_id: str) -> dict[str, bool]:
    SESSIONS.pop(session_id, None)
    return {"ok": True}


@app.get("/backtest/strategies")
def backtest_strategies() -> dict[str, Any]:
    return {
        "strategies": [
            {"name": "v1_bsp2_fullsize", "label": "二买/二卖结构模型：不定仓", "params": []},
            {"name": "v1_bsp2_structure_sizing", "label": "二买/二卖结构模型：结构风险定仓", "params": []},
        ]
    }


@app.get("/backtest/results")
def backtest_results() -> dict[str, Any]:
    items = []
    for result_id, strategy, path in _backtest_result_files():
        trades = _load_result_trades(path)
        symbols = sorted({_normalize_symbol(str(s)) for s in trades.get("symbol", pd.Series(dtype=str)).dropna().unique()})
        period = path.parent.name.replace("test_", "")
        label_prefix = "不定仓" if strategy == "v1_bsp2_fullsize" else "结构风险定仓"
        items.append(
            {
                "id": result_id,
                "strategy": strategy,
                "label": f"{label_prefix} / {period}",
                "period": period,
                "path": result_id,
                "updated_at": int(path.stat().st_mtime),
                "trade_count": int(len(trades)),
                "symbols": symbols,
                "metrics": _read_metric_for_trade_file(path),
            }
        )
    return {"results": items, "count": len(items)}


@app.get("/backtest/result-trades")
def backtest_result_trades(
    result_id: str = Query(alias="id"),
    symbol: str | None = None,
) -> dict[str, Any]:
    strategy, path = _backtest_result_path(result_id)
    trades = _load_result_trades(path)
    if symbol:
        normalized = _normalize_symbol(symbol)
        trades = trades.loc[trades["symbol"].map(lambda x: _normalize_symbol(str(x))) == normalized]
    trades = trades.sort_values("entry_time").reset_index(drop=True)
    return {
        "id": result_id,
        "strategy": strategy,
        "label": path.parent.name,
        "path": result_id,
        "metrics": _read_metric_for_trade_file(path),
        "trades": [_format_result_trade(i, row) for i, row in trades.iterrows()],
        "count": int(len(trades)),
    }


@app.get("/models")
def models() -> dict[str, Any]:
    registry = _parse_registry_table()
    ml_root = PROJECT_ROOT / "result/ml"
    hidden_artifact_names = {path.parent.name for path in MODEL_ARTIFACT_DIRS.values()}
    artifact_routes = {
        p.name
        for p in ml_root.iterdir()
        if p.is_dir() and p.name not in hidden_artifact_names
    } if ml_root.exists() else set()
    routes = set(registry) | artifact_routes
    sections = _extract_registry_sections(routes)
    items: list[dict[str, Any]] = []
    for route in sorted(routes):
        reg = registry.get(route, {})
        artifact = _model_artifact_info(route)
        items.append(
            {
                "route": route,
                "goal": reg.get("goal", ""),
                "label": reg.get("label", ""),
                "features": reg.get("features", ""),
                "status": reg.get("status", "仅有产物目录" if artifact["has_artifact"] else ""),
                "description": sections.get(route, ""),
                "artifact": artifact,
                "backtests": _find_backtest_metrics(route),
            }
        )
    return {
        "registry_path": str(MODEL_REGISTRY_PATH.relative_to(PROJECT_ROOT)),
        "models": items,
        "count": len(items),
    }


@app.get("/dryrun/bots")
def dryrun_bots() -> dict[str, Any]:
    items = []
    for bot_id, info in DRYRUN_BOTS.items():
        db_path = Path(info["db_path"])
        payload = _read_dryrun_db(db_path)
        items.append(
            {
                "id": bot_id,
                "name": info["name"],
                "strategy": info["strategy"],
                "api_port": info["api_port"],
                "db_path": str(db_path.relative_to(PROJECT_ROOT)),
                **payload,
            }
        )
    return {"bots": items, "count": len(items)}


@app.post("/backtest/start")
def backtest_start(body: BacktestStartBody) -> dict[str, Any]:
    cutoff = _parse_time(body.at)
    if cutoff is None:
        raise HTTPException(status_code=400, detail="invalid backtest start time")
    symbol = _normalize_symbol(body.symbol)
    data = _bars_for_freq(symbol, body.freq)
    cursor = int(data["time"].searchsorted(cutoff, side="right") - 1)
    _load_trades(body.strategy)
    sid = uuid.uuid4().hex
    SESSIONS[sid] = Session(
        exchange=body.exchange,
        symbol=symbol,
        freq=body.freq,
        strategy=body.strategy,
        init_cash=float(body.init_cash),
        cursor_time=cutoff,
        cursor_idx=max(cursor, 0),
    )
    return {
        "session_id": sid,
        "exchange": body.exchange,
        "symbol": body.symbol,
        "freq": body.freq,
        "strategy": body.strategy,
        "backfilled": max(0, cursor + 1),
        "remaining": max(0, int(len(data) - max(cursor + 1, 0))),
        "init_cash": float(body.init_cash),
    }


@app.post("/backtest/{session_id}/step")
def backtest_step(session_id: str, n: int = 1) -> dict[str, Any]:
    sess = SESSIONS.get(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="session not found")
    data = _bars_for_freq(sess.symbol, sess.freq)
    old_time = sess.cursor_time
    start = sess.cursor_idx + 1
    end = min(len(data), start + max(1, int(n)))
    new_bars = data.iloc[start:end]
    if not new_bars.empty:
        sess.cursor_idx = end - 1
        sess.cursor_time = pd.Timestamp(new_bars.iloc[-1]["time"])
    new_fills: list[dict[str, Any]] = []
    trades = _load_trades(sess.strategy)
    if not trades.empty and sess.cursor_time > old_time:
        window = (trades["entry_time"] > old_time) & (trades["entry_time"] <= sess.cursor_time)
        for idx, row in trades.loc[window].iterrows():
            new_fills.append(_fill_from_trade(row, "OPEN"))
        window = (trades["exit_time"] > old_time) & (trades["exit_time"] <= sess.cursor_time)
        for idx, row in trades.loc[window].iterrows():
            new_fills.append(_fill_from_trade(row, "CLOSE"))
            sess.closed_trade_indices.add(int(idx))
        new_fills.sort(key=lambda x: x["t"])
        sess.fills.extend(new_fills)
    state = _state(sess)
    return {
        "session_id": session_id,
        "pushed": int(len(new_bars)),
        "remaining": max(0, int(len(data) - end)),
        "bars": _format_bars(new_bars),
        "new_fills": new_fills,
        "new_equity": [
            {
                "t": int(sess.cursor_time.timestamp()),
                "equity": state["equity"],
                "cash": state["equity"],
                "floating_pnl": 0.0,
                "position_qty": 0,
            }
        ],
    }


def _state(sess: Session) -> dict[str, Any]:
    trades = _load_trades(sess.strategy)
    if trades.empty or not sess.closed_trade_indices:
        closed = trades.iloc[0:0]
    else:
        closed = trades.loc[sorted(sess.closed_trade_indices)]
    pnl = float(pd.to_numeric(closed.get("pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())
    wins = int((pd.to_numeric(closed.get("pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0) > 0).sum())
    n_trades = int(len(closed))
    equity = float(sess.init_cash + pnl)
    returns = pd.to_numeric(closed.get("net_return", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    sharpe = 0.0
    if len(returns) > 1 and float(returns.std(ddof=0)) > 0:
        sharpe = float(returns.mean() / returns.std(ddof=0) * math.sqrt(len(returns)))
    equity_curve = sess.init_cash + pd.to_numeric(closed.get("pnl", pd.Series(dtype=float)), errors="coerce").fillna(0.0).cumsum()
    max_dd = 0.0
    if not equity_curve.empty:
        running_max = equity_curve.cummax()
        drawdown = equity_curve / running_max - 1.0
        max_dd = float(drawdown.min() * 100.0)
    return {
        "session_id": "",
        "cursor": int(sess.cursor_idx),
        "remaining": max(0, int(len(_bars_for_freq(sess.symbol, sess.freq)) - sess.cursor_idx - 1)),
        "fills_count": int(len(sess.fills)),
        "n_trades": n_trades,
        "initial_cash": float(sess.init_cash),
        "equity": equity,
        "total_pnl": pnl,
        "total_return_pct": float(pnl / sess.init_cash * 100.0) if sess.init_cash else 0.0,
        "max_drawdown_pct": max_dd,
        "win_rate_pct": float(wins / n_trades * 100.0) if n_trades else 0.0,
        "sharpe": sharpe,
        "position_qty": 0,
    }


@app.get("/backtest/{session_id}/state")
def backtest_state(session_id: str) -> dict[str, Any]:
    sess = SESSIONS.get(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="session not found")
    state = _state(sess)
    state["session_id"] = session_id
    return state


@app.get("/backtest/{session_id}/chan")
def backtest_chan(
    session_id: str,
    from_: str | None = Query(default=None, alias="from"),
    to: str | None = None,
) -> dict[str, Any]:
    sess = SESSIONS.get(session_id)
    if sess is None:
        raise HTTPException(status_code=404, detail="session not found")
    start = _parse_time(from_)
    end = _parse_time(to)
    return _chan_slice(sess.symbol, sess.freq, start, end)


@app.delete("/backtest/{session_id}")
def backtest_stop(session_id: str) -> dict[str, bool]:
    SESSIONS.pop(session_id, None)
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("server:app", host="127.0.0.1", port=3000, reload=False)
