from __future__ import annotations

import json
import math
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .types import BacktestArtifacts, ScoredSignalEvent, SymbolBacktestResult


PERIOD_RETURN_LABELS = {
    "daily": "D",
    "weekly": "W",
    "monthly": "M",
    "yearly": "Y",
}


def ensure_output_dir(output_dir: str) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def events_to_dataframe(events: List[ScoredSignalEvent]) -> pd.DataFrame:
    rows = []
    for ev in events:
        gate_probability = getattr(ev, "gate_probability", float("nan"))
        if gate_probability is None:
            gate_probability = float("nan")
        rows.append(
            {
                "exec_time": pd.Timestamp(ev.exec_time).strftime("%Y-%m-%d %H:%M:%S"),
                "actual_exec_time": (
                    pd.Timestamp(ev.actual_exec_time).strftime("%Y-%m-%d %H:%M:%S")
                    if getattr(ev, "actual_exec_time", None) is not None
                    else ""
                ),
                "bsp_time": str(ev.bsp_time),
                "is_buy": bool(ev.is_buy),
                "bsp_type": str(ev.bsp_type),
                "bsp_types_str": str(ev.bsp_types_str),
                "probability": float(ev.probability),
                "qualified": bool(ev.qualified),
                "signal": int(ev.signal),
                "threshold": float(getattr(ev, "threshold", 0.0) or 0.0),
                "market_state": str(getattr(ev, "market_state", "") or ""),
                "quality_score": float(getattr(ev, "quality_score", 0.0) or 0.0),
                "gate_probability": float(gate_probability),
                "threshold_reason": str(getattr(ev, "threshold_reason", "") or ""),
                "trade_price": float(ev.trade_price),
                "symbol": str(ev.symbol),
            }
        )
    return pd.DataFrame(rows)


def bars_to_dataframe(item: SymbolBacktestResult) -> pd.DataFrame:
    frame = item.bars.copy()
    frame["time"] = pd.to_datetime(frame.index, utc=True).strftime("%Y-%m-%d %H:%M:%S")
    frame["symbol"] = item.symbol
    frame["signal"] = item.signal_matrix.signal.reindex(frame.index).fillna(0).astype(int).values
    if item.signal_matrix.size is not None:
        frame["risk_size"] = item.signal_matrix.size.reindex(frame.index).fillna(0.0).astype(float).values
    if item.signal_matrix.risk_blocked_entries is not None:
        frame["risk_blocked_entry"] = item.signal_matrix.risk_blocked_entries.reindex(frame.index).fillna(False).astype(bool).values
    if item.signal_matrix.risk_forced_exits is not None:
        frame["risk_forced_exit"] = item.signal_matrix.risk_forced_exits.reindex(frame.index).fillna(False).astype(bool).values
    return frame.reset_index(drop=True)


def portfolio_equity_to_dataframe(
    per_symbol: List[SymbolBacktestResult],
    portfolio_equity: pd.Series,
    portfolio_drawdown: pd.Series,
) -> pd.DataFrame:
    cols: Dict[str, pd.Series] = {
        "portfolio_equity": portfolio_equity.astype(float),
        "portfolio_drawdown": portfolio_drawdown.astype(float),
    }
    for item in per_symbol:
        cols[f"equity_{item.symbol}"] = item.equity_curve.reindex(portfolio_equity.index).ffill().astype(float)

    frame = pd.DataFrame(cols, index=portfolio_equity.index).reset_index()
    idx_col = frame.columns[0]
    if idx_col != "time":
        frame = frame.rename(columns={idx_col: "time"})
    frame["time"] = pd.to_datetime(frame["time"], utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")
    return frame


def per_symbol_metrics_to_dataframe(per_symbol: List[SymbolBacktestResult]) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for item in per_symbol:
        rows.append({"symbol": item.symbol, **item.metrics})
    return pd.DataFrame(rows)


def _normalise_equity_series(equity: pd.Series) -> pd.Series:
    if equity is None or len(equity) == 0:
        return pd.Series(dtype="float64", index=pd.DatetimeIndex([], name="time"))
    idx = pd.to_datetime(equity.index, utc=True, errors="coerce")
    out = pd.Series(pd.to_numeric(equity, errors="coerce").to_numpy(dtype="float64"), index=idx)
    out = out.dropna().sort_index()
    out = out[~out.index.duplicated(keep="last")]
    out.index.name = "time"
    return out


def _period_label(index: pd.DatetimeIndex, period: str) -> pd.Index:
    naive = index.tz_convert("UTC").tz_localize(None)
    rule = PERIOD_RETURN_LABELS[period]
    return naive.to_period(rule).astype(str)


def _trade_count_for_window(trades: pd.DataFrame, symbol: str, start: pd.Timestamp, end: pd.Timestamp) -> int:
    if trades is None or trades.empty:
        return 0
    frame = trades
    if "symbol" in frame.columns:
        frame = frame[frame["symbol"].astype(str) == str(symbol)]
    time_col = "exit_time" if "exit_time" in frame.columns else ("entry_time" if "entry_time" in frame.columns else None)
    if time_col is None or frame.empty:
        return 0
    times = pd.to_datetime(frame[time_col], utc=True, errors="coerce")
    return int(((times >= start) & (times <= end)).sum())


def period_returns_from_equity(
    equity_by_symbol: Dict[str, pd.Series],
    period: str,
    trades: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    if period not in PERIOD_RETURN_LABELS:
        raise ValueError(f"unsupported period: {period}")

    rows: List[Dict[str, Any]] = []
    trades_df = trades if trades is not None else pd.DataFrame()
    for symbol, equity in equity_by_symbol.items():
        series = _normalise_equity_series(equity)
        if series.empty:
            continue
        labels = pd.Series(_period_label(pd.DatetimeIndex(series.index), period), index=series.index)
        for label, group in series.groupby(labels, sort=True):
            if group.empty:
                continue
            start_equity = float(group.iloc[0])
            end_equity = float(group.iloc[-1])
            return_pct = ((end_equity / start_equity) - 1.0) * 100.0 if start_equity else 0.0
            drawdown = group / group.cummax() - 1.0
            start_time = pd.Timestamp(group.index[0])
            end_time = pd.Timestamp(group.index[-1])
            rows.append(
                {
                    "symbol": symbol,
                    "period_type": period,
                    "period": str(label),
                    "start_time": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "end_time": end_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "start_equity": start_equity,
                    "end_equity": end_equity,
                    "return_pct": float(return_pct),
                    "max_drawdown_pct": float(drawdown.min() * 100.0) if len(drawdown) else 0.0,
                    "closed_trades": _trade_count_for_window(trades_df, symbol, start_time, end_time),
                }
            )
    return pd.DataFrame(rows)


def per_symbol_period_returns_to_dataframe(per_symbol: List[SymbolBacktestResult], period: str) -> pd.DataFrame:
    equity_by_symbol = {item.symbol: item.equity_curve for item in per_symbol}
    trade_rows: List[Dict[str, Any]] = []
    for item in per_symbol:
        for trade in item.closed_trades:
            row = dict(trade)
            row["symbol"] = item.symbol
            trade_rows.append(row)
    return period_returns_from_equity(equity_by_symbol, period=period, trades=pd.DataFrame(trade_rows))


def save_metrics_json(path: Path, metrics: Dict[str, Any]) -> None:
    def _sanitize(value: Any) -> Any:
        if isinstance(value, dict):
            return {k: _sanitize(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_sanitize(v) for v in value]
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value

    with open(path, "w", encoding="utf-8") as f:
        json.dump(_sanitize(metrics), f, indent=2, ensure_ascii=False)


def _render_metrics_table(metrics: Dict[str, float]) -> str:
    rows = []
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            text = f"{float(value):,.6f}"
        else:
            text = str(value)
        rows.append(f"<tr><td>{escape(str(key))}</td><td>{escape(text)}</td></tr>")
    return "".join(rows)


def _render_html_report(
    metrics: Dict[str, float],
    per_symbol: List[SymbolBacktestResult],
    report_params: Optional[Dict[str, Any]],
    monthly_returns: Optional[pd.DataFrame] = None,
) -> str:
    symbol_rows = []
    for item in per_symbol:
        symbol_rows.append(
            "<tr>"
            f"<td>{escape(item.symbol)}</td>"
            f"<td>{escape(str(item.metrics.get('total_return_pct', 0.0)))}</td>"
            f"<td>{escape(str(item.metrics.get('max_drawdown_pct', 0.0)))}</td>"
            f"<td>{escape(str(item.metrics.get('total_trades', 0)))}</td>"
            "</tr>"
        )

    param_rows = []
    for k, v in (report_params or {}).items():
        param_rows.append(f"<tr><td>{escape(str(k))}</td><td>{escape(str(v))}</td></tr>")

    monthly_rows = []
    if monthly_returns is not None and not monthly_returns.empty:
        preview = monthly_returns.sort_values(["period", "symbol"]).tail(240)
        for row in preview.itertuples(index=False):
            monthly_rows.append(
                "<tr>"
                f"<td>{escape(str(row.symbol))}</td>"
                f"<td>{escape(str(row.period))}</td>"
                f"<td>{float(row.return_pct):,.4f}</td>"
                f"<td>{float(row.max_drawdown_pct):,.4f}</td>"
                f"<td>{int(row.closed_trades)}</td>"
                "</tr>"
            )

    return (
        "<!doctype html><html><head><meta charset='utf-8'><title>Backtest Report</title>"
        "<style>"
        "body{font-family:Arial,sans-serif;margin:24px;background:#f8fafc;color:#0f172a;}"
        "h1,h2{margin:0 0 12px;}"
        ".panel{background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:16px;margin-bottom:16px;}"
        "table{width:100%;border-collapse:collapse;}"
        "th,td{border-bottom:1px solid #e2e8f0;padding:8px;text-align:left;}"
        "th{background:#f1f5f9;}"
        "</style></head><body>"
        "<div class='panel'><h1>Chan Rule-Based Backtest Report</h1>"
        f"<p>Generated at: {escape(pd.Timestamp.now(tz='UTC').strftime('%Y-%m-%d %H:%M:%S UTC'))}</p></div>"
        "<div class='panel'><h2>Aggregate Metrics</h2><table><tr><th>Metric</th><th>Value</th></tr>"
        f"{_render_metrics_table(metrics)}</table></div>"
        "<div class='panel'><h2>Per Symbol</h2><table><tr><th>Symbol</th><th>Total Return %</th><th>Max Drawdown %</th><th>Total Trades</th></tr>"
        f"{''.join(symbol_rows)}</table></div>"
        "<div class='panel'><h2>Monthly Returns</h2><table><tr><th>Symbol</th><th>Month</th><th>Return %</th><th>Max Drawdown %</th><th>Trades</th></tr>"
        f"{''.join(monthly_rows)}</table></div>"
        "<div class='panel'><h2>Parameters</h2><table><tr><th>Key</th><th>Value</th></tr>"
        f"{''.join(param_rows)}</table></div>"
        "</body></html>"
    )


def write_outputs(
    output_dir: str,
    aggregate_metrics: Dict[str, float],
    per_symbol: List[SymbolBacktestResult],
    save_events_csv: bool,
    save_bars_csv: bool,
    save_metrics_json_flag: bool,
    save_html_report_flag: bool,
    save_trades_csv_flag: bool = True,
    save_equity_csv_flag: bool = True,
    report_params: Optional[Dict[str, Any]] = None,
) -> BacktestArtifacts:
    out = ensure_output_dir(output_dir)

    events_path = out / "signal_events.csv"
    bars_path = out / "signal_bars.csv"
    metrics_path = out / "backtest_metrics.json"
    report_path = out / "backtest_report.html"
    trades_path = out / "executed_trades.csv"
    equity_path = out / "portfolio_equity_curve.csv"
    per_symbol_metrics_path = out / "per_symbol_metrics.csv"
    period_return_paths = {
        period: out / f"per_symbol_period_returns_{period}.csv"
        for period in PERIOD_RETURN_LABELS
    }

    all_events: List[ScoredSignalEvent] = []
    for item in per_symbol:
        all_events.extend(item.signal_events)

    if save_events_csv:
        events_to_dataframe(all_events).to_csv(events_path, index=False, encoding="utf-8")

    if save_bars_csv:
        bars_df = pd.concat([bars_to_dataframe(item) for item in per_symbol], ignore_index=True)
        bars_df.to_csv(bars_path, index=False, encoding="utf-8")

    if save_metrics_json_flag:
        payload = {
            "aggregate": aggregate_metrics,
            "per_symbol": [{"symbol": item.symbol, **item.metrics} for item in per_symbol],
        }
        save_metrics_json(metrics_path, payload)

    per_symbol_metrics_to_dataframe(per_symbol).to_csv(per_symbol_metrics_path, index=False, encoding="utf-8")

    if save_trades_csv_flag:
        trade_rows: List[Dict[str, Any]] = []
        for item in per_symbol:
            for trade in item.closed_trades:
                row = dict(trade)
                row["symbol"] = item.symbol
                trade_rows.append(row)
        pd.DataFrame(trade_rows).to_csv(trades_path, index=False, encoding="utf-8")

    if save_equity_csv_flag:
        if len(per_symbol) == 1:
            eq = per_symbol[0].equity_curve
            dd = per_symbol[0].drawdown_curve
        else:
            eq = pd.concat([item.equity_curve.rename(item.symbol) for item in per_symbol], axis=1).ffill().mean(axis=1)
            dd = eq / eq.cummax() - 1.0
        portfolio_equity_to_dataframe(per_symbol, eq, dd).to_csv(equity_path, index=False, encoding="utf-8")

    monthly_returns = None
    for period, path in period_return_paths.items():
        period_df = per_symbol_period_returns_to_dataframe(per_symbol, period)
        period_df.to_csv(path, index=False, encoding="utf-8")
        if period == "monthly":
            monthly_returns = period_df

    if save_html_report_flag:
        html = _render_html_report(aggregate_metrics, per_symbol, report_params, monthly_returns=monthly_returns)
        report_path.write_text(html, encoding="utf-8")

    return BacktestArtifacts(
        events_csv=(events_path if save_events_csv else None),
        bars_csv=(bars_path if save_bars_csv else None),
        metrics_json=(metrics_path if save_metrics_json_flag else None),
        report_html=(report_path if save_html_report_flag else None),
        trades_csv=(trades_path if save_trades_csv_flag else None),
        equity_csv=(equity_path if save_equity_csv_flag else None),
        per_symbol_metrics_csv=per_symbol_metrics_path,
        period_returns_csvs=period_return_paths,
    )
