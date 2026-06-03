from __future__ import annotations

import argparse
import csv
import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any


def _float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    if out != out:
        return default
    return out


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _run_name(run_dir: Path, result_root: Path) -> str:
    rel = run_dir.relative_to(result_root).as_posix()
    parts = rel.split("/")
    if len(parts) >= 2:
        return f"{parts[0]} / {parts[-1]}"
    return rel


def _period_rows(run_dir: Path, period: str, symbol: str) -> list[dict[str, Any]]:
    path = run_dir / f"per_symbol_period_returns_{period}.csv"
    rows = _read_csv(path)
    if not rows:
        return []
    out = []
    for row in rows:
        row_symbol = str(row.get("symbol") or row.get("pair") or "").upper()
        if row_symbol and row_symbol != symbol.upper():
            continue
        out.append(row)
    return out


def _build_run(metrics_path: Path, result_root: Path) -> dict[str, Any]:
    run_dir = metrics_path.parent
    metrics = _read_json(metrics_path)
    aggregate = metrics.get("aggregate") or metrics
    symbols = metrics.get("per_symbol") or _read_csv(run_dir / "per_symbol_metrics.csv")
    normalised_symbols = []
    for item in symbols:
        symbol = str(item.get("symbol") or item.get("pair") or "").upper()
        if not symbol:
            continue
        row = {
            "symbol": symbol,
            "total_return_pct": _float(item.get("total_return_pct")),
            "annualized_return_pct": _float(item.get("annualized_return_pct")),
            "max_drawdown_pct": _float(item.get("max_drawdown_pct")),
            "win_rate_pct": _float(item.get("win_rate_pct")),
            "profit_factor": _float(item.get("profit_factor")),
            "total_trades": _float(item.get("total_trades")),
            "avg_trade_return_pct": _float(item.get("avg_trade_return_pct")),
            "exposure_time_pct": _float(item.get("exposure_time_pct")),
            "daily": _period_rows(run_dir, "daily", symbol),
            "weekly": _period_rows(run_dir, "weekly", symbol),
            "monthly": _period_rows(run_dir, "monthly", symbol),
            "yearly": _period_rows(run_dir, "yearly", symbol),
        }
        normalised_symbols.append(row)
    normalised_symbols.sort(key=lambda x: x["total_return_pct"], reverse=True)
    stat = metrics_path.stat()
    return {
        "id": run_dir.relative_to(result_root).as_posix(),
        "name": _run_name(run_dir, result_root),
        "path": run_dir.relative_to(result_root).as_posix(),
        "updated_at": datetime.fromtimestamp(stat.st_mtime).strftime("%Y-%m-%d %H:%M:%S"),
        "aggregate": {
            "total_return_pct": _float(aggregate.get("total_return_pct")),
            "annualized_return_pct": _float(aggregate.get("annualized_return_pct")),
            "max_drawdown_pct": _float(aggregate.get("max_drawdown_pct")),
            "win_rate_pct": _float(aggregate.get("win_rate_pct")),
            "profit_factor": _float(aggregate.get("profit_factor")),
            "total_trades": _float(aggregate.get("total_trades")),
            "avg_trade_return_pct": _float(aggregate.get("avg_trade_return_pct")),
            "exposure_time_pct": _float(aggregate.get("exposure_time_pct")),
            "start_cash": _float(aggregate.get("start_cash")),
            "end_equity": _float(aggregate.get("end_equity")),
        },
        "symbols": normalised_symbols,
    }


def _json_script(data: dict[str, Any]) -> str:
    return json.dumps(data, ensure_ascii=False).replace("</", "<\\/")


def _html_page(data: dict[str, Any]) -> str:
    payload = _json_script(data)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Chan 回测浏览器</title>
  <style>
    :root {{ color-scheme: light; --bg:#f6f7f9; --panel:#fff; --line:#d9dee7; --text:#172033; --muted:#667085; --good:#087443; --bad:#b42318; --accent:#1d4ed8; }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; font-family: "Microsoft YaHei", "Segoe UI", Arial, sans-serif; background:var(--bg); color:var(--text); }}
    header {{ padding:18px 24px; border-bottom:1px solid var(--line); background:var(--panel); display:flex; align-items:center; justify-content:space-between; gap:16px; }}
    h1 {{ margin:0; font-size:22px; }}
    .muted {{ color:var(--muted); }}
    .wrap {{ display:grid; grid-template-columns: 360px 1fr; min-height:calc(100vh - 70px); }}
    aside {{ border-right:1px solid var(--line); padding:16px; background:#fbfcfe; overflow:auto; }}
    main {{ padding:18px; overflow:auto; }}
    input, select {{ width:100%; height:36px; border:1px solid var(--line); border-radius:6px; padding:0 10px; background:#fff; color:var(--text); }}
    .controls {{ display:grid; gap:10px; margin-bottom:14px; }}
    .run {{ border:1px solid var(--line); border-radius:8px; padding:12px; background:var(--panel); cursor:pointer; margin-bottom:10px; }}
    .run.active {{ border-color:var(--accent); box-shadow:0 0 0 2px rgba(29,78,216,.10); }}
    .run-title {{ font-weight:700; margin-bottom:6px; }}
    .run-meta {{ display:flex; flex-wrap:wrap; gap:8px; font-size:12px; color:var(--muted); }}
    .cards {{ display:grid; grid-template-columns: repeat(6, minmax(120px, 1fr)); gap:10px; margin-bottom:14px; }}
    .card {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:12px; }}
    .label {{ color:var(--muted); font-size:12px; margin-bottom:7px; }}
    .value {{ font-size:20px; font-weight:700; }}
    .pos {{ color:var(--good); }}
    .neg {{ color:var(--bad); }}
    .grid {{ display:grid; grid-template-columns: minmax(520px, 1.2fr) minmax(360px, .8fr); gap:14px; align-items:start; }}
    .panel {{ background:var(--panel); border:1px solid var(--line); border-radius:8px; padding:14px; }}
    .panel h2 {{ margin:0 0 12px; font-size:16px; }}
    table {{ width:100%; border-collapse:collapse; font-size:13px; }}
    th, td {{ padding:8px 9px; border-bottom:1px solid #edf0f5; text-align:right; white-space:nowrap; }}
    th:first-child, td:first-child {{ text-align:left; }}
    tbody tr {{ cursor:pointer; }}
    tbody tr:hover, tbody tr.active {{ background:#f1f5ff; }}
    .detail-head {{ display:flex; justify-content:space-between; align-items:center; gap:12px; margin-bottom:12px; }}
    .detail-head h2 {{ margin:0; }}
    .tabs {{ display:flex; gap:8px; margin:12px 0; }}
    .tabs button {{ border:1px solid var(--line); background:#fff; border-radius:6px; padding:7px 10px; cursor:pointer; }}
    .tabs button.active {{ border-color:var(--accent); color:var(--accent); }}
    .mini {{ display:grid; grid-template-columns: repeat(2, 1fr); gap:8px; margin-bottom:12px; }}
    .mini .card {{ padding:10px; }}
    .empty {{ padding:20px; text-align:center; color:var(--muted); }}
    @media (max-width: 1100px) {{ .wrap, .grid {{ grid-template-columns:1fr; }} aside {{ border-right:none; border-bottom:1px solid var(--line); }} .cards {{ grid-template-columns: repeat(2, 1fr); }} }}
  </style>
</head>
<body>
  <header>
    <div><h1>Chan 回测浏览器</h1><div class="muted">可查看每条回测记录、20 个币种收益、分日/周/月/年表现</div></div>
    <div class="muted">生成时间：{html.escape(str(data.get("generated_at", "")))}</div>
  </header>
  <div class="wrap">
    <aside>
      <div class="controls">
        <input id="search" placeholder="搜索路线 / 路径 / 币种">
        <select id="sort">
          <option value="time">按时间</option>
          <option value="return">按收益</option>
          <option value="trades">按交易数</option>
          <option value="drawdown">按回撤</option>
        </select>
      </div>
      <div id="runs"></div>
    </aside>
    <main>
      <section class="cards" id="cards"></section>
      <section class="grid">
        <div class="panel">
          <h2>币种收益排名</h2>
          <table id="symbols"></table>
        </div>
        <div class="panel">
          <div class="detail-head">
            <h2 id="symbol-title">币种详情</h2>
            <select id="symbol-select"></select>
          </div>
          <div class="mini" id="symbol-cards"></div>
          <div class="tabs">
            <button data-period="monthly" class="active">月</button>
            <button data-period="weekly">周</button>
            <button data-period="daily">日</button>
            <button data-period="yearly">年</button>
          </div>
          <table id="periods"></table>
        </div>
      </section>
    </main>
  </div>
  <script>
    window.DATA = {payload};
    const fmt = (v, d=2) => `${{Number(v||0)>0?'+':''}}${{Number(v||0).toFixed(d)}}%`;
    const intf = (v) => Math.round(Number(v||0)).toLocaleString('zh-CN');
    const cls = (v) => Number(v||0) > 0 ? 'pos' : (Number(v||0) < 0 ? 'neg' : '');
    let state = {{ runId: '', symbol: '', period: 'monthly', search: '', sort: 'time' }};
    const runs = window.DATA.runs || [];
    function filteredRuns() {{
      const q = state.search.trim().toLowerCase();
      let out = runs.filter(r => !q || `${{r.name}} ${{r.path}} ${{(r.symbols||[]).map(s=>s.symbol).join(' ')}}`.toLowerCase().includes(q));
      out.sort((a,b) => {{
        if (state.sort === 'return') return Number(b.aggregate.total_return_pct||0) - Number(a.aggregate.total_return_pct||0);
        if (state.sort === 'trades') return Number(b.aggregate.total_trades||0) - Number(a.aggregate.total_trades||0);
        if (state.sort === 'drawdown') return Number(b.aggregate.max_drawdown_pct||0) - Number(a.aggregate.max_drawdown_pct||0);
        return String(b.updated_at).localeCompare(String(a.updated_at));
      }});
      return out;
    }}
    function currentRun() {{
      const fs = filteredRuns();
      return runs.find(r => r.id === state.runId) || fs[0] || runs[0];
    }}
    function currentSymbol(run) {{
      return (run?.symbols||[]).find(s => s.symbol === state.symbol) || (run?.symbols||[])[0];
    }}
    function renderRuns() {{
      const fs = filteredRuns();
      if (!state.runId && fs[0]) state.runId = fs[0].id;
      document.getElementById('runs').innerHTML = fs.map(r => `<div class="run ${{r.id===state.runId?'active':''}}" data-id="${{r.id}}">
        <div class="run-title">${{r.name}}</div>
        <div class="run-meta"><span class="${{cls(r.aggregate.total_return_pct)}}">${{fmt(r.aggregate.total_return_pct)}}</span><span>${{intf(r.aggregate.total_trades)}} 笔</span><span>胜率 ${{fmt(r.aggregate.win_rate_pct)}}</span><span>${{r.updated_at}}</span></div>
      </div>`).join('') || '<div class="empty">没有回测记录</div>';
      document.querySelectorAll('.run').forEach(el => el.onclick = () => {{ state.runId = el.dataset.id; state.symbol = ''; render(); }});
    }}
    function renderCards(run) {{
      const a = run?.aggregate || {{}};
      const rows = [['总收益', fmt(a.total_return_pct), cls(a.total_return_pct)], ['年化', fmt(a.annualized_return_pct), cls(a.annualized_return_pct)], ['最大回撤', fmt(a.max_drawdown_pct), cls(a.max_drawdown_pct)], ['交易数', intf(a.total_trades), ''], ['胜率', fmt(a.win_rate_pct), ''], ['PF', Number(a.profit_factor||0).toFixed(3), '']];
      document.getElementById('cards').innerHTML = rows.map(([k,v,c]) => `<div class="card"><div class="label">${{k}}</div><div class="value ${{c}}">${{v}}</div></div>`).join('');
    }}
    function renderSymbols(run) {{
      const syms = (run?.symbols || []).slice().sort((a,b) => Number(b.total_return_pct||0)-Number(a.total_return_pct||0));
      if (!state.symbol && syms[0]) state.symbol = syms[0].symbol;
      document.getElementById('symbol-select').innerHTML = syms.map(s => `<option value="${{s.symbol}}" ${{s.symbol===state.symbol?'selected':''}}>${{s.symbol}}</option>`).join('');
      document.getElementById('symbols').innerHTML = `<thead><tr><th>币种</th><th>收益</th><th>交易</th><th>胜率</th><th>回撤</th><th>PF</th><th>暴露</th></tr></thead><tbody>${{syms.map(s => `<tr class="${{s.symbol===state.symbol?'active':''}}" data-symbol="${{s.symbol}}"><td>${{s.symbol}}</td><td class="${{cls(s.total_return_pct)}}">${{fmt(s.total_return_pct)}}</td><td>${{intf(s.total_trades)}}</td><td>${{fmt(s.win_rate_pct)}}</td><td class="${{cls(s.max_drawdown_pct)}}">${{fmt(s.max_drawdown_pct)}}</td><td>${{Number(s.profit_factor||0).toFixed(3)}}</td><td>${{fmt(s.exposure_time_pct)}}</td></tr>`).join('')}}</tbody>`;
      document.querySelectorAll('#symbols tbody tr').forEach(el => el.onclick = () => {{ state.symbol = el.dataset.symbol; renderDetails(); }});
      document.getElementById('symbol-select').onchange = e => {{ state.symbol = e.target.value; renderDetails(); }};
    }}
    function periodLabel(row, period) {{
      return row.period || row.date || row.day || row.week || row.month || row.year || row.time || '';
    }}
    function periodReturn(row) {{
      return Number(row.return_pct_on_wallet ?? row.total_return_pct ?? row.return_pct ?? row.profit_pct ?? 0);
    }}
    function periodTrades(row) {{
      return Number(row.trades ?? row.total_trades ?? 0);
    }}
    function renderDetails() {{
      const run = currentRun();
      const s = currentSymbol(run);
      if (!s) return;
      document.getElementById('symbol-title').textContent = `${{s.symbol}} 详情`;
      const rows = [['收益', fmt(s.total_return_pct), cls(s.total_return_pct)], ['交易数', intf(s.total_trades), ''], ['胜率', fmt(s.win_rate_pct), ''], ['最大回撤', fmt(s.max_drawdown_pct), cls(s.max_drawdown_pct)]];
      document.getElementById('symbol-cards').innerHTML = rows.map(([k,v,c]) => `<div class="card"><div class="label">${{k}}</div><div class="value ${{c}}">${{v}}</div></div>`).join('');
      const pr = s[state.period] || [];
      document.getElementById('periods').innerHTML = pr.length ? `<thead><tr><th>周期</th><th>收益</th><th>交易数</th></tr></thead><tbody>${{pr.slice(-80).map(r => `<tr><td>${{periodLabel(r,state.period)}}</td><td class="${{cls(periodReturn(r))}}">${{fmt(periodReturn(r))}}</td><td>${{intf(periodTrades(r))}}</td></tr>`).join('')}}</tbody>` : '<tbody><tr><td class="empty">暂无分周期数据</td></tr></tbody>';
    }}
    function render() {{
      renderRuns();
      const run = currentRun();
      renderCards(run);
      renderSymbols(run);
      renderDetails();
    }}
    document.getElementById('search').oninput = e => {{ state.search = e.target.value; state.runId=''; render(); }};
    document.getElementById('sort').onchange = e => {{ state.sort = e.target.value; state.runId=''; render(); }};
    document.querySelectorAll('.tabs button').forEach(btn => btn.onclick = () => {{ document.querySelectorAll('.tabs button').forEach(b=>b.classList.remove('active')); btn.classList.add('active'); state.period = btn.dataset.period; renderDetails(); }});
    render();
  </script>
</body>
</html>
"""


def build(result_root: Path, output_dir: Path) -> None:
    metrics_paths = sorted(result_root.rglob("backtest_metrics.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    runs = []
    for metrics_path in metrics_paths:
        try:
            runs.append(_build_run(metrics_path, result_root))
        except Exception as exc:
            print(f"skip {metrics_path}: {exc}")
    output_dir.mkdir(parents=True, exist_ok=True)
    data = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "runs": runs,
    }
    (output_dir / "index.html").write_text(_html_page(data), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build static Chan backtest explorer")
    parser.add_argument("--result-root", default="result")
    parser.add_argument("--output-dir", default="result/backtest_explorer")
    args = parser.parse_args()
    build(Path(args.result_root), Path(args.output_dir))


if __name__ == "__main__":
    main()
