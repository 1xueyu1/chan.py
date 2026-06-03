const rawData = window.CHAN_BACKTEST_EXPLORER_DATA || { runs: [], generated_at: "" };

const state = {
  runs: rawData.runs || [],
  filtered: [],
  selectedRunId: "",
  selectedSymbol: "",
  period: "monthly",
  search: "",
  sort: "return",
};

const els = {
  generatedAt: document.getElementById("generated-at"),
  runSelect: document.getElementById("run-select"),
  runSearch: document.getElementById("run-search"),
  sortSelect: document.getElementById("sort-select"),
  runList: document.getElementById("run-list"),
  metricStrip: document.getElementById("metric-strip"),
  symbolCount: document.getElementById("symbol-count"),
  symbolSelect: document.getElementById("symbol-select"),
  symbolTable: document.getElementById("symbol-table"),
  symbolTitle: document.getElementById("symbol-title"),
  symbolSubtitle: document.getElementById("symbol-subtitle"),
  symbolSummary: document.getElementById("symbol-summary"),
  yearBars: document.getElementById("year-bars"),
  periodChart: document.getElementById("period-chart"),
  periodTable: document.getElementById("period-table"),
  tabs: Array.from(document.querySelectorAll(".tab")),
};

function num(v) {
  const n = Number(v || 0);
  return Number.isFinite(n) ? n : 0;
}

function fmtPct(v, digits = 2) {
  const n = num(v);
  return `${n > 0 ? "+" : ""}${n.toFixed(digits)}%`;
}

function fmtMoney(v, digits = 0) {
  const n = num(v);
  return `${n > 0 ? "+" : ""}${n.toLocaleString("zh-CN", { maximumFractionDigits: digits, minimumFractionDigits: digits })}`;
}

function fmtInt(v) {
  return Math.round(num(v)).toLocaleString("zh-CN");
}

function cls(v) {
  const n = num(v);
  if (n > 0) return "positive";
  if (n < 0) return "negative";
  return "muted";
}

function currentRun() {
  return state.runs.find((run) => run.id === state.selectedRunId) || state.filtered[0] || state.runs[0];
}

function currentSymbols(run) {
  return (run?.strategy?.pairs || []).slice().sort((a, b) => num(b.total_return_pct) - num(a.total_return_pct));
}

function currentSymbol(run) {
  const symbols = currentSymbols(run);
  return symbols.find((item) => item.pair === state.selectedSymbol) || symbols[0];
}

function applyFilters() {
  const q = state.search.trim().toLowerCase();
  state.filtered = state.runs.filter((run) => {
    if (!q) return true;
    return `${run.name} ${run.path} ${run.route}`.toLowerCase().includes(q);
  });
  state.filtered.sort((a, b) => {
    if (state.sort === "drawdown") return num(b.aggregate?.max_drawdown_pct) - num(a.aggregate?.max_drawdown_pct);
    if (state.sort === "trades") return num(b.aggregate?.total_trades) - num(a.aggregate?.total_trades);
    if (state.sort === "time") return String(b.updated_at).localeCompare(String(a.updated_at));
    return num(b.aggregate?.total_return_pct) - num(a.aggregate?.total_return_pct);
  });
  if (!state.filtered.some((run) => run.id === state.selectedRunId)) {
    state.selectedRunId = state.filtered[0]?.id || "";
    state.selectedSymbol = "";
  }
}

function renderRunSelect() {
  els.runSelect.innerHTML = state.runs.map((run) => (
    `<option value="${run.id}" ${run.id === state.selectedRunId ? "selected" : ""}>${run.name}</option>`
  )).join("");
}

function renderRunList() {
  if (!state.filtered.length) {
    els.runList.innerHTML = '<div class="empty">没有匹配的回测记录</div>';
    return;
  }
  els.runList.innerHTML = state.filtered.map((run) => {
    const agg = run.aggregate || {};
    return `
      <article class="run-card ${run.id === state.selectedRunId ? "active" : ""}" data-run="${run.id}">
        <div class="run-card-top">
          <div>
            <div class="run-name">${run.name}</div>
            <div class="run-path">${run.path}</div>
          </div>
          <strong class="${cls(agg.total_return_pct)}">${fmtPct(agg.total_return_pct)}</strong>
        </div>
        <div class="run-meta">
          <span>回撤 ${fmtPct(agg.max_drawdown_pct)}</span>
          <span>${fmtInt(agg.total_trades)} 笔</span>
          <span>${run.updated_at || ""}</span>
        </div>
      </article>
    `;
  }).join("");
  els.runList.querySelectorAll(".run-card").forEach((node) => {
    node.addEventListener("click", () => {
      state.selectedRunId = node.dataset.run || "";
      state.selectedSymbol = "";
      render();
    });
  });
}

function renderMetrics(run) {
  const agg = run?.aggregate || {};
  const cards = [
    ["总收益", fmtPct(agg.total_return_pct), cls(agg.total_return_pct)],
    ["年化收益", fmtPct(agg.annualized_return_pct), cls(agg.annualized_return_pct)],
    ["最大回撤", fmtPct(agg.max_drawdown_pct), cls(agg.max_drawdown_pct)],
    ["交易次数", fmtInt(agg.total_trades), ""],
    ["胜率", fmtPct(agg.win_rate_pct), ""],
    ["Profit Factor", num(agg.profit_factor).toFixed(3), ""],
  ];
  els.metricStrip.innerHTML = cards.map(([label, value, klass]) => `
    <div class="metric">
      <span class="label">${label}</span>
      <span class="value ${klass}">${value}</span>
    </div>
  `).join("");
}

function renderSymbolControls(run) {
  const symbols = currentSymbols(run);
  els.symbolCount.textContent = `${fmtInt(symbols.length)} 个币种`;
  if (!state.selectedSymbol && symbols.length) state.selectedSymbol = symbols[0].pair;
  els.symbolSelect.innerHTML = symbols.map((item) => (
    `<option value="${item.pair}" ${item.pair === state.selectedSymbol ? "selected" : ""}>${item.pair}</option>`
  )).join("");
}

function renderSymbolTable(run) {
  const symbols = currentSymbols(run);
  if (!symbols.length) {
    els.symbolTable.innerHTML = '<tbody><tr><td class="empty">暂无币种数据</td></tr></tbody>';
    return;
  }
  const rows = symbols.map((item, index) => `
    <tr class="${item.pair === state.selectedSymbol ? "active" : ""}" data-symbol="${item.pair}">
      <td>#${index + 1} ${item.pair}</td>
      <td class="${cls(item.total_return_pct)}">${fmtPct(item.total_return_pct)}</td>
      <td>${fmtPct(item.max_drawdown_pct)}</td>
      <td>${fmtInt(item.total_trades)}</td>
      <td>${fmtPct(num(item.winrate) * 100)}</td>
      <td>${num(item.profit_factor).toFixed(3)}</td>
      <td>${fmtPct(item.exposure_time_pct)}</td>
    </tr>
  `).join("");
  els.symbolTable.innerHTML = `
    <thead><tr><th>币种</th><th>收益</th><th>回撤</th><th>交易</th><th>胜率</th><th>PF</th><th>暴露</th></tr></thead>
    <tbody>${rows}</tbody>
  `;
  els.symbolTable.querySelectorAll("tbody tr").forEach((row) => {
    row.addEventListener("click", () => {
      state.selectedSymbol = row.dataset.symbol || "";
      renderDetailsOnly();
    });
  });
}

function renderSymbolSummary(run) {
  const item = currentSymbol(run);
  if (!item) {
    els.symbolTitle.textContent = "币种详情";
    els.symbolSubtitle.textContent = "";
    els.symbolSummary.innerHTML = '<div class="empty">暂无数据</div>';
    els.yearBars.innerHTML = "";
    return;
  }
  els.symbolTitle.textContent = item.pair;
  els.symbolSubtitle.textContent = run.name;
  const cards = [
    ["累计收益", fmtPct(item.total_return_pct), cls(item.total_return_pct)],
    ["累计盈利", fmtMoney(item.total_profit_abs, 0), cls(item.total_profit_abs)],
    ["最大回撤", fmtPct(item.max_drawdown_pct), cls(item.max_drawdown_pct)],
    ["交易次数", fmtInt(item.total_trades), ""],
    ["胜率", fmtPct(num(item.winrate) * 100), ""],
    ["赢 / 亏", `${fmtInt(item.win_count)} / ${fmtInt(item.loss_count)}`, ""],
  ];
  els.symbolSummary.innerHTML = cards.map(([k, v, klass]) => `
    <div class="summary-item"><span class="k">${k}</span><span class="v ${klass}">${v}</span></div>
  `).join("");
  renderBars(els.yearBars, item.years || [], "year", "return_pct_on_wallet", "trades");
}

function renderBars(target, rows, labelKey, valueKey, tradeKey) {
  if (!rows.length) {
    target.innerHTML = '<div class="empty">暂无图表数据</div>';
    return;
  }
  const shown = rows.slice(-32);
  const maxAbs = Math.max(...shown.map((r) => Math.abs(num(r[valueKey]))), 1);
  target.innerHTML = shown.map((r) => {
    const value = num(r[valueKey]);
    const width = Math.max(1, Math.round((Math.abs(value) / maxAbs) * 48));
    const sign = value >= 0 ? "pos" : "neg";
    const style = value >= 0 ? `left:50%;width:${width}%` : `right:50%;width:${width}%`;
    return `
      <div class="bar-row">
        <span>${r[labelKey]}</span>
        <span class="bar-track"><span class="bar-fill ${sign}" style="${style}"></span></span>
        <span class="bar-meta ${cls(value)}">${fmtPct(value)} · ${fmtInt(r[tradeKey])} 笔</span>
      </div>
    `;
  }).join("");
}

function periodRows(run) {
  const all = run?.periods?.[state.period] || [];
  const symbol = state.selectedSymbol || currentSymbol(run)?.pair;
  return all.filter((row) => row.symbol === symbol);
}

function renderPeriod(run) {
  const rows = periodRows(run);
  renderBars(els.periodChart, rows, "period", "return_pct", "closed_trades");
  if (!rows.length) {
    els.periodTable.innerHTML = '<tbody><tr><td class="empty">暂无周期收益</td></tr></tbody>';
    return;
  }
  const sorted = rows.slice().sort((a, b) => String(b.period).localeCompare(String(a.period)));
  els.periodTable.innerHTML = `
    <thead><tr><th>周期</th><th>收益</th><th>回撤</th><th>交易数</th><th>开始权益</th><th>结束权益</th><th>区间</th></tr></thead>
    <tbody>
      ${sorted.map((r) => `
        <tr>
          <td>${r.period}</td>
          <td class="${cls(r.return_pct)}">${fmtPct(r.return_pct)}</td>
          <td>${fmtPct(r.max_drawdown_pct)}</td>
          <td>${fmtInt(r.closed_trades)}</td>
          <td>${fmtMoney(r.start_equity, 0)}</td>
          <td>${fmtMoney(r.end_equity, 0)}</td>
          <td class="muted">${r.start_time || ""} 至 ${r.end_time || ""}</td>
        </tr>
      `).join("")}
    </tbody>
  `;
}

function renderDetailsOnly() {
  const run = currentRun();
  renderSymbolControls(run);
  renderSymbolTable(run);
  renderSymbolSummary(run);
  renderPeriod(run);
}

function render() {
  applyFilters();
  const run = currentRun();
  if (!run) {
    els.runList.innerHTML = '<div class="empty">没有回测记录，请先刷新 data.js</div>';
    return;
  }
  state.selectedRunId = run.id;
  renderRunSelect();
  renderRunList();
  renderMetrics(run);
  renderDetailsOnly();
}

els.generatedAt.textContent = `索引生成：${rawData.generated_at || "-"} · ${fmtInt(rawData.runs_count || 0)} 条记录`;
els.runSelect.addEventListener("change", (event) => {
  state.selectedRunId = event.target.value;
  state.selectedSymbol = "";
  render();
});
els.runSearch.addEventListener("input", (event) => {
  state.search = event.target.value || "";
  render();
});
els.sortSelect.addEventListener("change", (event) => {
  state.sort = event.target.value || "return";
  render();
});
els.symbolSelect.addEventListener("change", (event) => {
  state.selectedSymbol = event.target.value;
  renderDetailsOnly();
});
els.tabs.forEach((tab) => {
  tab.addEventListener("click", () => {
    state.period = tab.dataset.period || "monthly";
    els.tabs.forEach((item) => item.classList.toggle("active", item === tab));
    renderPeriod(currentRun());
  });
});

state.selectedRunId = state.runs[0]?.id || "";
render();
