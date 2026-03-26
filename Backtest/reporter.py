from __future__ import annotations

# flake8: noqa: E501

import json
import math
from html import escape
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd

from .types import BacktestArtifacts, ScoredSignalEvent, SymbolBacktestResult


_METRIC_META: Dict[str, Dict[str, str]] = {
    "total_return_pct": {"label": "总收益率", "format": "pct", "desc": "回测区间内组合净值的整体涨跌幅"},
    "annualized_return_pct": {"label": "年化收益率", "format": "pct", "desc": "按时间长度折算后的年化收益"},
    "max_drawdown_pct": {"label": "最大回撤", "format": "pct", "desc": "从高点回落的最大幅度，越小越稳健"},
    "sharpe": {"label": "夏普比率", "format": "ratio", "desc": "单位波动下的超额收益能力"},
    "sortino": {"label": "索提诺比率", "format": "ratio", "desc": "仅考虑下行风险时的收益风险比"},
    "calmar": {"label": "卡玛比率", "format": "ratio", "desc": "年化收益与最大回撤的比值"},
    "win_rate_pct": {"label": "胜率", "format": "pct", "desc": "盈利交易占总交易的比例"},
    "profit_factor": {"label": "盈亏比", "format": "ratio", "desc": "总盈利与总亏损绝对值的比值"},
    "total_trades": {"label": "闭环成交数", "format": "int", "desc": "回测期间已完成平仓的完整交易事件数量（开仓→平仓）"},
    "avg_trade_return_pct": {"label": "单笔平均收益", "format": "pct", "desc": "每笔交易平均收益率"},
    "exposure_time_pct": {"label": "持仓时间占比", "format": "pct", "desc": "处于非空仓状态的时间占比"},
    "start_cash": {"label": "初始资金", "format": "money", "desc": "回测开始时投入资金"},
    "end_equity": {"label": "期末权益", "format": "money", "desc": "回测结束时组合总权益"},
}

_KPI_ORDER = [
    "total_return_pct",
    "annualized_return_pct",
    "max_drawdown_pct",
    "sharpe",
    "win_rate_pct",
    "profit_factor",
    "total_trades",
    "avg_trade_return_pct",
    "exposure_time_pct",
    "start_cash",
    "end_equity",
]

_PARAM_META: Dict[str, Dict[str, str]] = {
    "symbols": {"label": "回测标的", "format": "list"},
    "begin_time": {"label": "开始时间", "format": "str"},
    "end_time": {"label": "结束时间", "format": "str"},
    "kl_type": {"label": "K线级别", "format": "str"},
    "execution_mode": {"label": "成交模式", "format": "execution_mode"},
    "event_source": {"label": "事件来源模式", "format": "str"},
    "event_replay_csv_path": {"label": "复用事件CSV", "format": "str"},
    "replay_reapply_threshold": {"label": "复用时重算阈值", "format": "bool"},
    "allow_short": {"label": "是否允许做空", "format": "bool"},
    "signal_threshold": {"label": "信号阈值", "format": "float4"},
    "initial_cash": {"label": "初始资金", "format": "money"},
    "fee": {"label": "手续费率", "format": "float6"},
    "slippage": {"label": "滑点", "format": "float6"},
    "conflict_policy": {"label": "冲突处理", "format": "str"},
    "output_dir": {"label": "输出目录", "format": "str"},
    "model_buy_path": {"label": "买模型路径", "format": "str"},
    "model_sell_path": {"label": "卖模型路径", "format": "str"},
    "meta_buy_path": {"label": "买模型元信息路径", "format": "str"},
    "meta_sell_path": {"label": "卖模型元信息路径", "format": "str"},
    "meta_model_path": {"label": "Meta模型路径", "format": "str"},
}

_PARAM_ORDER = [
    "symbols",
    "begin_time",
    "end_time",
    "kl_type",
    "execution_mode",
    "event_source",
    "event_replay_csv_path",
    "replay_reapply_threshold",
    "allow_short",
    "signal_threshold",
    "initial_cash",
    "fee",
    "slippage",
    "conflict_policy",
    "output_dir",
    "model_buy_path",
    "model_sell_path",
    "meta_buy_path",
    "meta_sell_path",
    "meta_model_path",
]


# Limit heavy payload in detail HTML to avoid slow/blocked rendering on large backtests.
_DETAIL_REPORT_MAX_PLOT_POINTS = 4000
# 0 means no capping: keep complete executed trade events in detail report.
_DETAIL_REPORT_MAX_TRADE_EVENTS = 0


def _is_invalid_number(value: object) -> bool:
    if not isinstance(value, (int, float)):
        return True
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return True
    return False


def _metric_label(metric_key: str) -> str:
    info = _METRIC_META.get(metric_key)
    if info:
        return info["label"]
    return metric_key.replace("_", " ").title()


def _format_metric_value(metric_key: str, value: object) -> str:
    if _is_invalid_number(value):
        return "--"

    meta = _METRIC_META.get(metric_key, {})
    fmt = meta.get("format", "float")
    numeric_value = float(value)

    if fmt == "pct":
        return f"{numeric_value:,.2f}%"
    if fmt == "money":
        return f"{numeric_value:,.2f}"
    if fmt == "int":
        return f"{int(round(numeric_value))}"
    if fmt == "ratio":
        return f"{numeric_value:,.3f}"
    return f"{numeric_value:,.6f}"


def _metric_desc(metric_key: str) -> str:
    info = _METRIC_META.get(metric_key)
    if info:
        return info["desc"]
    return "-"


def _classify_insights(metrics: Dict[str, float]) -> List[str]:
    insights: List[str] = []

    total_return = metrics.get("total_return_pct")
    max_drawdown = metrics.get("max_drawdown_pct")
    sharpe = metrics.get("sharpe")
    win_rate = metrics.get("win_rate_pct")
    profit_factor = metrics.get("profit_factor")

    if isinstance(total_return, (int, float)) and not _is_invalid_number(total_return):
        if total_return >= 20:
            insights.append("收益表现较强，策略在样本区间内具备较好的进攻性。")
        elif total_return >= 0:
            insights.append("策略实现正收益，但增益幅度中等，建议继续优化入场效率。")
        else:
            insights.append("策略区间收益为负，需重点检查信号阈值和交易成本假设。")

    if isinstance(max_drawdown, (int, float)) and not _is_invalid_number(max_drawdown):
        if max_drawdown <= -30:
            insights.append("最大回撤偏大，建议收紧风控参数并评估减仓机制。")
        elif max_drawdown <= -15:
            insights.append("回撤处于可接受但偏高区间，需关注连续亏损阶段。")
        else:
            insights.append("回撤控制相对稳健，资金曲线波动较温和。")

    if isinstance(sharpe, (int, float)) and not _is_invalid_number(sharpe):
        if sharpe >= 1.5:
            insights.append("风险调整后收益优秀，夏普比率表现亮眼。")
        elif sharpe >= 1.0:
            insights.append("风险调整后收益良好，策略具备一定稳定性。")
        elif sharpe > 0:
            insights.append("策略有正向收益，但单位风险收益偏低。")
        else:
            insights.append("夏普比率不理想，收益与波动匹配度不足。")

    if isinstance(win_rate, (int, float)) and not _is_invalid_number(win_rate):
        if win_rate < 40:
            insights.append("胜率偏低，建议结合盈亏比评估是否属于低胜率高赔率结构。")
        elif win_rate >= 55:
            insights.append("胜率较高，信号方向性较好。")

    if isinstance(profit_factor, (int, float)) and not _is_invalid_number(profit_factor):
        if profit_factor >= 1.5:
            insights.append("盈亏比健康，亏损覆盖能力较强。")
        elif profit_factor < 1.0:
            insights.append("盈亏比低于 1，策略长期可持续性存在风险。")

    if not insights:
        insights.append("当前样本不足以形成稳定结论，建议扩大回测区间并增加标的验证。")

    return insights


def _format_param_value(param_key: str, value: Any) -> str:
    if value is None:
        return "--"

    meta = _PARAM_META.get(param_key, {})
    fmt = meta.get("format", "str")

    if fmt == "list":
        if isinstance(value, (list, tuple, set)):
            return ", ".join(str(item) for item in value)
        return str(value)

    if fmt == "bool":
        return "是" if bool(value) else "否"

    if fmt == "execution_mode":
        if str(value) == "next_bar_open":
            return "下一根K线开盘成交"
        if str(value) == "close":
            return "当前K线收盘成交"
        return str(value)

    if fmt == "money":
        try:
            return f"{float(value):,.2f}"
        except Exception:
            return str(value)

    if fmt == "float4":
        try:
            return f"{float(value):.4f}"
        except Exception:
            return str(value)

    if fmt == "float6":
        try:
            return f"{float(value):.6f}"
        except Exception:
            return str(value)

    return str(value)


def _render_params_table(report_params: Optional[Dict[str, Any]]) -> str:
    if not report_params:
        return ""

    ordered_keys = [k for k in _PARAM_ORDER if k in report_params]
    ordered_keys.extend(k for k in report_params if k not in ordered_keys)

    rows = []
    for key in ordered_keys:
        meta = _PARAM_META.get(key, {})
        label = meta.get("label", key)
        value_text = _format_param_value(key, report_params.get(key))
        rows.append(
            "<tr>"
            f"<td>{escape(label)}</td>"
            f"<td>{escape(value_text)}</td>"
            "</tr>"
        )

    return (
        "<section class='panel'>"
        "<h2>回测参数</h2>"
        "<table class='data-table params-table'>"
        "<tr><th>参数</th><th>取值</th></tr>"
        + "".join(rows)
        + "</table></section>"
    )


def _downsample_series(series: pd.Series, max_points: int) -> pd.Series:
    if max_points <= 0 or len(series) <= max_points:
        return series
    step = int(math.ceil(len(series) / max_points))
    sampled = series.iloc[::step]
    if sampled.index[-1] != series.index[-1]:
        sampled = pd.concat([sampled, series.iloc[[-1]]])
    return sampled


def _cap_trade_events(trade_events: Optional[List[ScoredSignalEvent]], max_events: int) -> tuple[Optional[List[ScoredSignalEvent]], bool]:
    if not trade_events or max_events <= 0:
        return trade_events, False
    if len(trade_events) <= max_events:
        return trade_events, False
    # Keep newest events for readability and to reduce HTML payload.
    return trade_events[-max_events:], True


def _cap_table_rows(rows: Optional[List[Dict[str, Any]]], max_rows: int) -> tuple[Optional[List[Dict[str, Any]]], bool]:
    if not rows or max_rows <= 0:
        return rows, False
    if len(rows) <= max_rows:
        return rows, False
    # Keep newest rows for interactive browsing.
    return rows[-max_rows:], True


def _build_executed_trade_rows(
    per_symbol: List[SymbolBacktestResult],
    initial_cash: float,
    execution_mode: str,
) -> List[Dict[str, Any]]:
    del initial_cash, execution_mode
    rows: List[Dict[str, Any]] = []

    for item in per_symbol:
        for trade in item.closed_trades:
            rows.append(
                {
                    "symbol": str(trade.get("symbol", item.symbol)),
                    "direction": str(trade.get("direction", "long")),
                    "entry_time": str(trade.get("entry_time", "")),
                    "exit_time": str(trade.get("exit_time", "")),
                    "holding_bars": int(trade.get("holding_bars", 0)),
                    "entry_price": trade.get("entry_price"),
                    "exit_price": trade.get("exit_price"),
                    "pnl": trade.get("pnl"),
                    "return_pct": trade.get("return_pct"),
                    "fees": trade.get("fees"),
                    "equity_before": trade.get("equity_before"),
                    "equity_after": trade.get("equity_after"),
                    "equity_delta": trade.get("equity_delta"),
                    "equity_delta_pct": trade.get("equity_delta_pct"),
                }
            )

    rows.sort(key=lambda x: (x["entry_time"], x["symbol"]))
    return rows


def _normalize_trade_events(trade_events: Optional[List[ScoredSignalEvent]]) -> List[Dict[str, Any]]:
    if not trade_events:
        return []

    rows: List[Dict[str, Any]] = []
    for ev in sorted(trade_events, key=lambda x: x.exec_time):
        direction = "买点" if ev.is_buy else "卖点"
        if ev.signal == 1:
            action = "开多"
        elif ev.signal == -1:
            action = "平多"
        elif ev.signal == 0:
            action = "观望"
        else:
            action = str(ev.signal)

        rows.append(
            {
                "exec_time": ev.exec_time.strftime("%Y-%m-%d %H:%M:%S"),
                "symbol": ev.symbol,
                "direction": direction,
                "action": action,
                "bsp_type": ev.bsp_type,
                "bsp_types_str": ev.bsp_types_str,
                "probability": round(float(ev.probability), 6),
                "qualified": "是" if ev.qualified else "否",
                "trade_price": round(float(ev.trade_price), 8),
            }
        )

    return rows


def _risk_assessment(metrics: Dict[str, float]) -> Dict[str, str]:
    score = 50.0

    total_return = metrics.get("total_return_pct")
    drawdown = metrics.get("max_drawdown_pct")
    sharpe = metrics.get("sharpe")
    profit_factor = metrics.get("profit_factor")

    if isinstance(total_return, (int, float)) and not _is_invalid_number(total_return):
        if total_return >= 30:
            score += 20
        elif total_return >= 10:
            score += 12
        elif total_return < 0:
            score -= 12

    if isinstance(drawdown, (int, float)) and not _is_invalid_number(drawdown):
        if drawdown >= -10:
            score += 15
        elif drawdown >= -20:
            score += 6
        elif drawdown <= -35:
            score -= 18
        elif drawdown <= -25:
            score -= 10

    if isinstance(sharpe, (int, float)) and not _is_invalid_number(sharpe):
        if sharpe >= 1.5:
            score += 12
        elif sharpe >= 1.0:
            score += 6
        elif sharpe < 0:
            score -= 10

    if isinstance(profit_factor, (int, float)) and not _is_invalid_number(profit_factor):
        if profit_factor >= 1.6:
            score += 10
        elif profit_factor >= 1.2:
            score += 5
        elif profit_factor < 1.0:
            score -= 10

    score = max(0.0, min(score, 100.0))

    if score >= 80:
        return {
            "level": "低风险（稳健）",
            "css": "risk-low",
            "summary": "策略收益与回撤匹配较好，稳定性较强，可作为核心策略候选。",
        }
    if score >= 65:
        return {
            "level": "中低风险（偏稳健）",
            "css": "risk-mid-low",
            "summary": "策略具备较好的风险收益比，建议继续做跨周期和跨标的验证。",
        }
    if score >= 50:
        return {
            "level": "中等风险（均衡）",
            "css": "risk-mid",
            "summary": "策略可用但仍有波动压力，建议在仓位和止损参数上进一步优化。",
        }
    if score >= 35:
        return {
            "level": "中高风险（进攻）",
            "css": "risk-mid-high",
            "summary": "策略更偏进攻型，回撤容忍度要求高，建议配合风控开关使用。",
        }
    return {
        "level": "高风险（谨慎）",
        "css": "risk-high",
        "summary": "策略在当前样本下风险偏高，建议先降杠杆并重新校准信号与成本参数。",
    }


def _render_trade_table(trade_rows: Optional[List[Dict[str, Any]]]) -> str:
    if not trade_rows:
        return (
            "<section class='panel'>"
            "<h2>交易明细</h2>"
            "<p class='hint'>暂无交易明细数据。</p>"
            "</section>"
        )

    symbols = sorted({str(row.get("symbol", "")) for row in trade_rows if row.get("symbol")})

    trades_json = json.dumps(trade_rows, ensure_ascii=False)
    symbol_options = "".join(
        f"<option value='{escape(sym)}'>{escape(sym)}</option>" for sym in symbols
    )

    return (
        "<section class='panel'>"
        "<h2>交易明细（分页）</h2>"
        f"<p class='hint'>总完整交易事件：{len(trade_rows)}（开仓→平仓，按币种筛选）</p>"
        "<div class='trade-table-wrap'>"
        "<table class='data-table trade-table' id='trade-table'>"
        "<thead><tr>"
        "<th>开仓时间</th><th>平仓时间</th><th>标的</th><th>方向</th><th>持有Bars</th>"
        "<th>开仓价</th><th>平仓价</th><th>PnL</th><th>收益率</th><th>手续费</th>"
        "<th>资金(前)</th><th>资金(后)</th><th>资金变化</th><th>变化率</th>"
        "</tr></thead><tbody id='trade-table-body'></tbody></table></div>"
        "<div class='pager'>"
        "<label for='symbol-filter'>币种</label>"
        f"<select id='symbol-filter'><option value='ALL'>全部</option>{symbol_options}</select>"
        "<button id='page-prev' type='button'>上一页</button>"
        "<span id='page-info'>第 1 / 1 页</span>"
        "<button id='page-next' type='button'>下一页</button>"
        "<label for='page-size'>每页</label>"
        "<select id='page-size'><option value='20'>20</option><option value='50'>50</option><option value='100'>100</option></select>"
        "</div></section>"
        "<script>"
        f"const tradeRows = {trades_json};"
        "let symbolFilter = 'ALL';"
        "let currentPage = 1;"
        "let pageSize = 20;"
        "const bodyEl = document.getElementById('trade-table-body');"
        "const infoEl = document.getElementById('page-info');"
        "const prevEl = document.getElementById('page-prev');"
        "const nextEl = document.getElementById('page-next');"
        "const sizeEl = document.getElementById('page-size');"
        "const symbolEl = document.getElementById('symbol-filter');"
        "function fmtNum(v, d){ if(v === null || v === undefined || Number.isNaN(Number(v))) return '--'; return Number(v).toFixed(d); }"
        "function renderTradeRows(){"
        "const filtered = symbolFilter === 'ALL' ? tradeRows : tradeRows.filter(function(r){ return r.symbol === symbolFilter; });"
        "const totalPages = Math.max(1, Math.ceil(filtered.length / pageSize));"
        "if(currentPage > totalPages){ currentPage = totalPages; }"
        "const start = (currentPage - 1) * pageSize;"
        "const end = Math.min(start + pageSize, filtered.length);"
        "const sliced = filtered.slice(start, end);"
        "bodyEl.innerHTML = sliced.map(function(row){"
        "const dirClass = String(row.direction).toLowerCase() === 'long' ? 'txt-up' : 'txt-down';"
        "const pnlClass = Number(row.pnl) >= 0 ? 'txt-up' : 'txt-down';"
        "const deltaClass = Number(row.equity_delta) >= 0 ? 'txt-up' : 'txt-down';"
        "return '<tr>'"
        "+ '<td>' + row.entry_time + '</td>'"
        "+ '<td>' + row.exit_time + '</td>'"
        "+ '<td>' + row.symbol + '</td>'"
        "+ '<td class=' + String.fromCharCode(34) + dirClass + String.fromCharCode(34) + '>' + row.direction + '</td>'"
        "+ '<td>' + String(row.holding_bars) + '</td>'"
        "+ '<td>' + fmtNum(row.entry_price, 6) + '</td>'"
        "+ '<td>' + fmtNum(row.exit_price, 6) + '</td>'"
        "+ '<td class=' + String.fromCharCode(34) + pnlClass + String.fromCharCode(34) + '>' + fmtNum(row.pnl, 2) + '</td>'"
        "+ '<td class=' + String.fromCharCode(34) + pnlClass + String.fromCharCode(34) + '>' + fmtNum(row.return_pct, 3) + '%' + '</td>'"
        "+ '<td>' + fmtNum(row.fees, 2) + '</td>'"
        "+ '<td>' + fmtNum(row.equity_before, 2) + '</td>'"
        "+ '<td>' + fmtNum(row.equity_after, 2) + '</td>'"
        "+ '<td class=' + String.fromCharCode(34) + deltaClass + String.fromCharCode(34) + '>' + fmtNum(row.equity_delta, 2) + '</td>'"
        "+ '<td class=' + String.fromCharCode(34) + deltaClass + String.fromCharCode(34) + '>' + fmtNum(row.equity_delta_pct, 3) + '%' + '</td>'"
        "+ '</tr>';"
        "}).join('');"
        "infoEl.textContent = '第 ' + currentPage + ' / ' + totalPages + ' 页（记录 ' + filtered.length + '）';"
        "prevEl.disabled = currentPage <= 1;"
        "nextEl.disabled = currentPage >= totalPages;"
        "}"
        "prevEl.addEventListener('click', function(){ if(currentPage > 1){ currentPage -= 1; renderTradeRows(); } });"
        "nextEl.addEventListener('click', function(){ const filtered = symbolFilter === 'ALL' ? tradeRows : tradeRows.filter(function(r){ return r.symbol === symbolFilter; }); const totalPages = Math.max(1, Math.ceil(filtered.length / pageSize)); if(currentPage < totalPages){ currentPage += 1; renderTradeRows(); } });"
        "sizeEl.addEventListener('change', function(){ pageSize = Number(sizeEl.value || 20); currentPage = 1; renderTradeRows(); });"
        "symbolEl.addEventListener('change', function(){ symbolFilter = symbolEl.value || 'ALL'; currentPage = 1; renderTradeRows(); });"
        "renderTradeRows();"
        "</script>"
    )


def ensure_output_dir(output_dir: str) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    return out


def events_to_dataframe(events: Iterable[ScoredSignalEvent]) -> pd.DataFrame:
    rows = []
    for ev in events:
        rows.append(
            {
                "exec_time": ev.exec_time.strftime("%Y-%m-%d %H:%M:%S"),
                "bsp_time": ev.bsp_time,
                "is_buy": ev.is_buy,
                "bsp_type": ev.bsp_type,
                "bsp_types_str": ev.bsp_types_str,
                "probability": ev.probability,
                "qualified": ev.qualified,
                "signal": ev.signal,
                "trade_price": ev.trade_price,
                "symbol": ev.symbol,
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "exec_time",
            "bsp_time",
            "is_buy",
            "bsp_type",
            "bsp_types_str",
            "probability",
            "qualified",
            "signal",
            "trade_price",
            "symbol",
        ],
    )


def bars_to_dataframe(symbol_result: SymbolBacktestResult) -> pd.DataFrame:
    bars = symbol_result.bars.copy()
    bars["signal"] = symbol_result.signal_matrix.signal
    bars["position"] = symbol_result.signal_matrix.position
    bars["symbol"] = symbol_result.symbol
    bars = bars.reset_index().rename(columns={"time": "time"})
    bars["time"] = pd.to_datetime(bars["time"], utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")
    return bars[["time", "open", "high", "low", "close", "volume", "signal", "position", "symbol"]]


def save_metrics_json(path: Path, metrics: Dict[str, float]) -> None:
    def _sanitize(value):
        if isinstance(value, dict):
            return {k: _sanitize(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_sanitize(v) for v in value]
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return None
        return value

    with open(path, "w", encoding="utf-8") as f:
        json.dump(_sanitize(metrics), f, indent=2, ensure_ascii=False)


def _render_html_report(
    report_title: str,
    metrics: Dict[str, float],
    equity_curve: pd.Series,
    drawdown_curve: pd.Series,
    per_symbol_metrics: Optional[List[Dict[str, object]]] = None,
    trade_events: Optional[List[ScoredSignalEvent]] = None,
    executed_trade_rows: Optional[List[Dict[str, Any]]] = None,
    report_params: Optional[Dict[str, Any]] = None,
    include_plotly_chart: bool = True,
    include_trade_table: bool = True,
) -> str:
    chart_was_downsampled = False
    trade_events_were_capped = False

    if include_plotly_chart:
        sampled_equity = _downsample_series(equity_curve, _DETAIL_REPORT_MAX_PLOT_POINTS)
        if len(sampled_equity) < len(equity_curve):
            chart_was_downsampled = True
        sampled_drawdown = drawdown_curve.reindex(sampled_equity.index).ffill().fillna(0.0)
    else:
        sampled_equity = equity_curve
        sampled_drawdown = drawdown_curve

    if include_trade_table:
        trade_events, trade_events_were_capped = _cap_trade_events(
            trade_events, _DETAIL_REPORT_MAX_TRADE_EVENTS
        )
        executed_trade_rows, table_rows_were_capped = _cap_table_rows(
            executed_trade_rows, _DETAIL_REPORT_MAX_TRADE_EVENTS
        )
        trade_events_were_capped = trade_events_were_capped or table_rows_were_capped

    if include_plotly_chart:
        try:
            import plotly.graph_objects as go
            from plotly.subplots import make_subplots

            period_returns = sampled_equity.pct_change().fillna(0.0) * 100.0
            period_return_colors = ["#159895" if x >= 0 else "#c44536" for x in period_returns.values]

            fig = make_subplots(
                rows=3,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.06,
                subplot_titles=("资金曲线", "回撤曲线", "单周期收益率（%）"),
            )
            fig.add_trace(
                go.Scatter(
                    x=sampled_equity.index,
                    y=sampled_equity.values,
                    mode="lines",
                    name="资金曲线",
                    line={"color": "#1f6f8b", "width": 2.2},
                ),
                row=1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=sampled_drawdown.index,
                    y=sampled_drawdown.values * 100.0,
                    mode="lines",
                    fill="tozeroy",
                    name="回撤",
                    line={"color": "#b23a48", "width": 2.0},
                ),
                row=2,
                col=1,
            )
            fig.add_trace(
                go.Bar(
                    x=period_returns.index,
                    y=period_returns.values,
                    name="单周期收益率",
                    marker={"color": period_return_colors},
                ),
                row=3,
                col=1,
            )
            fig.update_layout(
                height=1020,
                template="plotly_white",
                title={"text": report_title, "x": 0.02, "xanchor": "left"},
                margin={"l": 40, "r": 24, "t": 80, "b": 36},
                showlegend=False,
            )
            fig.update_yaxes(title_text="权益", row=1, col=1)
            fig.update_yaxes(title_text="回撤(%)", row=2, col=1)
            fig.update_yaxes(title_text="收益率(%)", row=3, col=1)
            fig_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
        except Exception:
            fig_html = "<p class='hint'>未检测到 Plotly，已跳过图表绘制。</p>"
    else:
        fig_html = "<p class='hint'>当前为 summary 报告，已跳过图表与交易明细以降低体积与生成耗时。</p>"

    kpi_cards: List[str] = []
    for key in _KPI_ORDER:
        if key not in metrics:
            continue
        value_text = _format_metric_value(key, metrics[key])
        desc = _metric_desc(key)
        card_class = "kpi-card"
        if key == "total_return_pct" and isinstance(metrics[key], (int, float)) and not _is_invalid_number(metrics[key]):
            card_class += " kpi-up" if float(metrics[key]) >= 0 else " kpi-down"
        kpi_cards.append(
            "<div class='{}'>".format(card_class)
            + f"<div class='kpi-name'>{escape(_metric_label(key))}</div>"
            + f"<div class='kpi-value'>{escape(value_text)}</div>"
            + f"<div class='kpi-desc'>{escape(desc)}</div>"
            + "</div>"
        )

    ordered_keys = [k for k in _KPI_ORDER if k in metrics] + [k for k in metrics if k not in _KPI_ORDER]
    metric_rows = []
    for key in ordered_keys:
        value_text = _format_metric_value(key, metrics.get(key))
        metric_rows.append(
            "<tr>"
            f"<td>{escape(_metric_label(key))}</td>"
            f"<td>{escape(value_text)}</td>"
            f"<td>{escape(_metric_desc(key))}</td>"
            "</tr>"
        )

    insight_lines = "".join(f"<li>{escape(item)}</li>" for item in _classify_insights(metrics))
    risk = _risk_assessment(metrics)

    report_time = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M:%S UTC")
    symbol_count = len(per_symbol_metrics or [])
    scope_text = "单标的" if symbol_count <= 1 else "多标的组合"

    symbol_table = ""
    if per_symbol_metrics:
        header = "".join(["<th>交易标的</th>", "<th>总收益率</th>", "<th>最大回撤</th>", "<th>闭环成交数</th>"])
        rows = []
        for row in per_symbol_metrics:
            row_return = row.get("total_return_pct", float("nan"))
            row_return_class = "txt-up" if isinstance(row_return, (int, float)) and not _is_invalid_number(row_return) and row_return >= 0 else "txt-down"
            rows.append(
                "<tr>"
                f"<td>{escape(str(row.get('symbol', '')))}</td>"
                f"<td class='{row_return_class}'>{escape(_format_metric_value('total_return_pct', row_return))}</td>"
                f"<td>{escape(_format_metric_value('max_drawdown_pct', row.get('max_drawdown_pct', float('nan'))))}</td>"
                f"<td>{escape(_format_metric_value('total_trades', row.get('total_trades', 0)))}</td>"
                "</tr>"
            )
        symbol_table = (
            "<section class='panel'>"
            "<h2>分标的表现</h2>"
            "<table class='data-table'>"
            f"<tr>{header}</tr>"
            + "".join(rows)
            + "</table>"
            + "</section>"
        )

    params_table = _render_params_table(report_params)
    trade_table = _render_trade_table(executed_trade_rows) if include_trade_table else ""

    detail_data_hints = ""
    if chart_was_downsampled or trade_events_were_capped:
        hint_parts: List[str] = []
        if chart_was_downsampled:
            hint_parts.append(f"图表已降采样至约 {_DETAIL_REPORT_MAX_PLOT_POINTS} 点")
        if trade_events_were_capped:
            hint_parts.append(f"交易明细仅保留最近 {_DETAIL_REPORT_MAX_TRADE_EVENTS} 条")
        detail_data_hints = (
            "<section class='panel'>"
            "<h2>性能说明</h2>"
            f"<p class='hint'>{escape('；'.join(hint_parts))}，以提升详细报告生成与浏览速度。</p>"
            "</section>"
        )

    return (
        "<!doctype html>"
        "<html><head><meta charset='utf-8'><title>回测报告</title>"
        "<meta name='viewport' content='width=device-width, initial-scale=1'>"
        "<style>"
        ":root{--bg:#f5f7fb;--panel:#ffffff;--line:#e6e9f0;--title:#1b263b;--text:#334155;--muted:#6b7280;--up:#1d9a6c;--down:#c44536;}"
        "body{margin:0;padding:0;background:linear-gradient(180deg,#f8fafc 0%,#eef3fb 60%,#f7f9fc 100%);font-family:'Microsoft YaHei','PingFang SC','Noto Sans SC',sans-serif;color:var(--text);}"
        ".page{max-width:1240px;margin:0 auto;padding:28px 18px 42px;}"
        ".hero{background:radial-gradient(circle at 20% 10%,#ffffff 0%,#f4f7fd 45%,#edf2fb 100%);border:1px solid var(--line);border-radius:16px;padding:20px 24px;box-shadow:0 8px 24px rgba(20,28,45,.06);}"
        ".hero h1{margin:0 0 8px;font-size:28px;color:var(--title);letter-spacing:.5px;}"
        ".hero p{margin:4px 0;font-size:14px;color:var(--muted);}"
        ".kpi-grid{margin-top:16px;display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px;}"
        ".kpi-card{background:var(--panel);border:1px solid var(--line);border-radius:12px;padding:12px 14px;box-shadow:0 3px 10px rgba(50,58,77,.05);}"
        ".kpi-up{border-left:4px solid var(--up);}"
        ".kpi-down{border-left:4px solid var(--down);}"
        ".kpi-name{font-size:13px;color:var(--muted);margin-bottom:6px;}"
        ".kpi-value{font-size:24px;font-weight:700;color:var(--title);line-height:1.2;}"
        ".kpi-desc{margin-top:5px;font-size:12px;color:var(--muted);min-height:30px;}"
        ".layout{display:grid;grid-template-columns:1.2fr .8fr;gap:14px;margin-top:14px;}"
        ".panel{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:14px 16px;margin-top:14px;box-shadow:0 4px 14px rgba(40,48,67,.04);}"
        ".panel h2{margin:0 0 12px;color:var(--title);font-size:18px;}"
        ".summary-box{border:1px dashed var(--line);border-radius:10px;padding:10px 12px;background:#fafcff;margin-top:10px;}"
        ".summary-title{font-weight:700;color:var(--title);margin-bottom:6px;}"
        ".risk-badge{display:inline-block;border-radius:999px;padding:4px 10px;font-size:12px;font-weight:700;margin-bottom:8px;}"
        ".risk-low{background:#def5eb;color:#0f8c61;}"
        ".risk-mid-low{background:#e7f6ff;color:#1b6ca8;}"
        ".risk-mid{background:#fff3d9;color:#9f6a00;}"
        ".risk-mid-high{background:#ffe9d8;color:#b85b14;}"
        ".risk-high{background:#ffe1e1;color:#b42318;}"
        ".insights{margin:0;padding-left:18px;line-height:1.7;}"
        ".insights li{margin:4px 0;}"
        ".params-table th:first-child,.params-table td:first-child{width:36%;}"
        ".trade-table-wrap{overflow-x:auto;}"
        ".trade-table th,.trade-table td{white-space:nowrap;}"
        ".pager{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-top:12px;}"
        ".pager button{border:1px solid var(--line);background:#fff;border-radius:8px;padding:5px 10px;cursor:pointer;color:#334155;}"
        ".pager button:disabled{opacity:.45;cursor:not-allowed;}"
        ".pager select{border:1px solid var(--line);border-radius:8px;padding:4px 8px;background:#fff;}"
        ".data-table{width:100%;border-collapse:collapse;font-size:14px;}"
        ".data-table th,.data-table td{border-bottom:1px solid var(--line);padding:9px 8px;text-align:left;}"
        ".data-table th{background:#f8fafc;color:#344054;font-weight:600;}"
        ".txt-up{color:var(--up);font-weight:600;}"
        ".txt-down{color:var(--down);font-weight:600;}"
        ".hint{color:var(--muted);font-size:14px;}"
        "@media (max-width:960px){.layout{grid-template-columns:1fr;}.hero h1{font-size:24px;}}"
        "</style></head><body><div class='page'>"
        "<section class='hero'>"
        f"<h1>{escape(report_title)}</h1>"
        f"<p>回测范围：{escape(scope_text)} | 标的数量：{symbol_count} | 引擎：vectorbt</p>"
        f"<p>报告生成时间：{escape(report_time)}</p>"
        "<div class='kpi-grid'>"
        + "".join(kpi_cards)
        + "</div>"
        + "</section>"
        + "<div class='layout'>"
        + "<section class='panel'><h2>策略解读</h2>"
        + f"<div class='risk-badge {escape(risk['css'])}'>风险等级：{escape(risk['level'])}</div>"
        + "<div class='summary-box'>"
        + "<div class='summary-title'>中文总评</div>"
        + f"<div>{escape(risk['summary'])}</div>"
        + "</div><ul class='insights'>"
        + insight_lines
        + "</ul></section>"
        + "<section class='panel'><h2>指标释义</h2><table class='data-table'><tr><th>指标</th><th>值</th><th>含义</th></tr>"
        + "".join(metric_rows)
        + "</table></section></div>"
        + params_table
        + detail_data_hints
        + symbol_table
        + "<section class='panel'><h2>资金与风险曲线</h2>"
        + fig_html
        + "</section>"
        + trade_table
        + "</div></body></html>"
    )


def write_outputs(
    output_dir: str,
    aggregate_metrics: Dict[str, float],
    per_symbol: List[SymbolBacktestResult],
    save_events_csv: bool,
    save_bars_csv: bool,
    save_metrics_json_flag: bool,
    save_html_report_flag: bool,
    save_html_detail_report_flag: bool,
    report_params: Optional[Dict[str, Any]] = None,
) -> BacktestArtifacts:
    out = ensure_output_dir(output_dir)

    events_path = out / "model_signal_events.csv"
    bars_path = out / "model_signal_bars.csv"
    metrics_path = out / "backtest_metrics.json"
    report_path = out / "xgb_backtest_report.html"
    report_detail_path = out / "xgb_backtest_report_detail.html"

    all_events = []
    for item in per_symbol:
        all_events.extend(item.signal_events)

    if save_events_csv:
        events_df = events_to_dataframe(all_events)
        events_df.to_csv(events_path, index=False, encoding="utf-8")

    if save_bars_csv:
        bars_df = pd.concat([bars_to_dataframe(item) for item in per_symbol], axis=0, ignore_index=True)
        bars_df.to_csv(bars_path, index=False, encoding="utf-8")

    if save_metrics_json_flag:
        payload = {
            "aggregate": aggregate_metrics,
            "per_symbol": [
                {
                    "symbol": item.symbol,
                    **item.metrics,
                }
                for item in per_symbol
            ],
        }
        save_metrics_json(metrics_path, payload)

    if save_html_report_flag or save_html_detail_report_flag:
        if len(per_symbol) == 1:
            eq = per_symbol[0].equity_curve
            dd = per_symbol[0].drawdown_curve
        else:
            eq = pd.concat([item.equity_curve.rename(item.symbol) for item in per_symbol], axis=1).ffill().mean(axis=1)
            dd = eq / eq.cummax() - 1.0

        per_symbol_metrics = [
            {
                "symbol": item.symbol,
                "total_return_pct": float(item.metrics.get("total_return_pct", float("nan"))),
                "max_drawdown_pct": float(item.metrics.get("max_drawdown_pct", float("nan"))),
                "total_trades": int(item.metrics.get("total_trades", 0)),
            }
            for item in per_symbol
        ]
        if save_html_report_flag:
            summary_html = _render_html_report(
                report_title="Chan 策略回测报告（Summary）",
                metrics=aggregate_metrics,
                equity_curve=eq,
                drawdown_curve=dd,
                per_symbol_metrics=per_symbol_metrics,
                trade_events=None,
                report_params=report_params,
                include_plotly_chart=False,
                include_trade_table=False,
            )
            report_path.write_text(summary_html, encoding="utf-8")

        if save_html_detail_report_flag:
            exec_mode = "next_bar_open"
            if report_params and isinstance(report_params.get("execution_mode"), str):
                exec_mode = str(report_params.get("execution_mode"))
            initial_cash = float(aggregate_metrics.get("start_cash", 0.0) or 0.0)
            executed_trade_rows = _build_executed_trade_rows(
                per_symbol=per_symbol,
                initial_cash=initial_cash,
                execution_mode=exec_mode,
            )
            detail_html = _render_html_report(
                report_title="Chan 策略回测报告（Detail, XGBoost + SHAP）",
                metrics=aggregate_metrics,
                equity_curve=eq,
                drawdown_curve=dd,
                per_symbol_metrics=per_symbol_metrics,
                trade_events=all_events,
                executed_trade_rows=executed_trade_rows,
                report_params=report_params,
                include_plotly_chart=True,
                include_trade_table=True,
            )
            report_detail_path.write_text(detail_html, encoding="utf-8")

    return BacktestArtifacts(
        events_csv=events_path,
        bars_csv=bars_path,
        metrics_json=metrics_path,
        report_html=report_path,
        report_detail_html=(report_detail_path if save_html_detail_report_flag else None),
    )
