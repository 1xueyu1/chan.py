// 回测面板：右侧抽屉式 UI，集成策略选择、参数表单、控制按钮、绩效指标显示。
//
// 数据流：
//   1. 用户填表 → 点开始 → POST /backtest/start → 拿到 session_id
//   2. enterBacktest(sid, cutoffTs)，调 chartActions.resetData() 让 TV 重拉 K（自动截断）
//   3. 用户点 step / 自动播放 → POST /backtest/{id}/step → advanceStream(bars) 喂入 K
//      → chartActions.refreshChan() 刷 chan + 累积 fills 在面板显示
//   4. 定期 /state 拉绩效指标
//   5. 点停止 → DELETE /backtest/{id} + exitStream + chartActions.resetData() 切回历史

import { useEffect, useRef, useState } from 'react'
import { api } from '../api/client'
import type {
  BacktestFill, BacktestResultInfo, BacktestResultTrade,
  BacktestState, BacktestStrategy, ContractSpec,
} from '../api/types'
import { advanceStream, enterBacktest, exitStream, streamState } from './streamState'

const PANEL_WIDTH = 300

export interface ChartActions {
  resetData: () => void
  refreshChan: () => void
  /** 喂入成交 fills，画箭头 */
  appendFills: (fills: BacktestFill[]) => void
  /** 清掉之前画的所有 fill 箭头 */
  clearFills: () => void
  showResultTrades: (payload: {
    resultId: string
    label: string
    trades: BacktestResultTrade[]
  }) => void
  clearResultTrades: () => void
}

interface Props {
  open: boolean
  onClose: () => void
  theme: 'light' | 'dark'
  /** 当前图表的合约/周期，传给 backtest start */
  exchange: string
  symbol: string
  freq: string
  chartActions: ChartActions
}

function parseAtToUnix(at: string): number {
  const cleaned = at.trim()
  const padded = cleaned.includes(' ') ? cleaned : `${cleaned} 00:00:00`
  const d = new Date(padded.replace(' ', 'T') + '+08:00')
  return Math.floor(d.getTime() / 1000)
}

export function BacktestPanel({
  open, onClose, theme, exchange, symbol, freq, chartActions,
}: Props) {
  // 配色
  const C = theme === 'dark' ? {
    bg: '#131722', text: '#d1d4dc', sub: '#787b86',
    border: '#1e222d', hover: 'rgba(255,255,255,0.04)',
    inputBg: '#1e222d', inputBorder: '#2a2e39',
    accent: '#2962ff', success: '#089981', danger: '#ef4444',
  } : {
    bg: '#ffffff', text: '#131722', sub: '#787b86',
    border: '#e0e3eb', hover: 'rgba(0,0,0,0.04)',
    inputBg: '#ffffff', inputBorder: '#e0e3eb',
    accent: '#2962ff', success: '#089981', danger: '#d32f2f',
  }

  // 配置 state
  const [strategies, setStrategies] = useState<BacktestStrategy[]>([])
  const [selStrategy, setSelStrategy] = useState<string>('')
  const [results, setResults] = useState<BacktestResultInfo[]>([])
  const [selResult, setSelResult] = useState<string>('')
  const [resultSymbol, setResultSymbol] = useState<string>('__all__')
  const [loadingResult, setLoadingResult] = useState(false)
  const [params, setParams] = useState<Record<string, any>>({})
  const [startDate, setStartDate] = useState<string>('')
  const [initCash, setInitCash] = useState<string>('1000000')
  // 合约规格（覆盖项）：留空走后端 spec 默认。marginPct 以「百分比」展示（0.08 → 8）。
  const [marginPct, setMarginPct] = useState<string>('')
  const [feeRate, setFeeRate] = useState<string>('')
  const [feeFixed, setFeeFixed] = useState<string>('')
  const [slippageTicks, setSlippageTicks] = useState<string>('1')
  const [spec, setSpec] = useState<ContractSpec | null>(null)

  // 运行时 state
  const [active, setActive] = useState(false)
  const [isAutoPlaying, setIsAutoPlaying] = useState(false)
  const [autoSpeed, setAutoSpeed] = useState<1 | 5 | 20>(1)  // 倍率：每秒 N 根 K
  const [remaining, setRemaining] = useState(0)
  const [metrics, setMetrics] = useState<BacktestState | null>(null)
  const [error, setError] = useState<string | null>(null)
  const autoTimer = useRef<number | null>(null)

  // 拉策略列表
  useEffect(() => {
    if (!open) return
    Promise.all([api.backtestStrategies(), api.backtestResults()])
      .then(([r, br]) => {
        setResults(br.results)
        if (br.results.length && !selResult) setSelResult(br.results[0].id)
        setStrategies(r.strategies)
        if (r.strategies.length && !selStrategy) {
          setSelStrategy(r.strategies[0].name)
          // 初始化默认 params
          const defaults: Record<string, any> = {}
          for (const p of r.strategies[0].params) defaults[p.name] = p.default
          setParams(defaults)
        }
      })
      .catch((e) => setError(`策略列表加载失败: ${e}`))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open])

  // 面板打开 / 合约变化时拉 spec 预填保证金/手续费
  useEffect(() => {
    if (!open) return
    api.contract(symbol)
      .then((s) => {
        setSpec(s)
        setMarginPct((s.margin_rate * 100).toFixed(2))
        setFeeRate(s.fee_rate ? String(s.fee_rate) : '')
        setFeeFixed(s.fee_fixed ? String(s.fee_fixed) : '')
      })
      .catch(() => { /* 未登记品种走后端默认 */ })
  }, [open, symbol])

  // 切换策略时重置 params 为该策略的默认值
  function onStrategyChange(name: string) {
    setSelStrategy(name)
    const s = strategies.find((x) => x.name === name)
    if (!s) return
    const defaults: Record<string, any> = {}
    for (const p of s.params) defaults[p.name] = p.default
    setParams(defaults)
  }

  function setParam(name: string, v: any) {
    setParams((prev) => ({ ...prev, [name]: v }))
  }

  // 拉最新绩效
  async function refreshState(sid: string) {
    try {
      const st = await api.backtestState(sid)
      setMetrics(st)
      setRemaining(st.remaining)
      if (st.remaining === 0) stopAuto()
    } catch (e) {
      console.warn('refresh backtest state failed', e)
    }
  }

  async function handleStart() {
    if (!startDate.trim() || !selStrategy) {
      setError('请填写起始日期并选择策略')
      return
    }
    setError(null)
    try {
      const toNum = (s: string): number | undefined => {
        const t = s.trim()
        if (!t) return undefined
        const n = Number(t)
        return Number.isFinite(n) ? n : undefined
      }
      const mp = toNum(marginPct)
      const resp = await api.backtestStart({
        exchange, symbol, freq, at: startDate, strategy: selStrategy,
        params, init_cash: Number(initCash) || 1_000_000,
        slippage_ticks: toNum(slippageTicks) ?? 1.0,
        margin_rate: mp == null ? undefined : mp / 100,  // UI 是百分比，传后端要 0-1
        fee_rate: toNum(feeRate),
        fee_fixed: toNum(feeFixed),
      })
      enterBacktest(resp.session_id, parseAtToUnix(startDate))
      chartActions.clearFills()
      // 强制 TV 重拉 K（自动按 cutoffTs 截断到 backfill 范围）
      streamState.resetCache?.()
      chartActions.resetData()
      chartActions.refreshChan()
      setActive(true)
      setRemaining(resp.remaining)
      await refreshState(resp.session_id)
    } catch (e) {
      setError(`启动失败: ${e}`)
    }
  }

  async function handleStep(n: number) {
    const sid = streamState.sessionId
    if (!sid || streamState.mode !== 'backtest') return
    try {
      const resp = await api.backtestStep(sid, n)
      advanceStream(resp.bars)
      if (resp.new_fills.length) chartActions.appendFills(resp.new_fills)
      chartActions.refreshChan()
      await refreshState(sid)
    } catch (e) {
      setError(`步进失败: ${e}`)
    }
  }

  // 按倍率启动/切换自动播放：1x = 每秒 1 根，5x = 每秒 5 根，20x = 每秒 20 根
  function playAt(speed: 1 | 5 | 20) {
    if (autoTimer.current) {
      window.clearInterval(autoTimer.current)
      autoTimer.current = null
    }
    setAutoSpeed(speed)
    setIsAutoPlaying(true)
    const intervalMs = Math.max(20, Math.floor(1000 / speed))
    autoTimer.current = window.setInterval(() => {
      void handleStep(1)
    }, intervalMs)
  }
  function stopAuto() {
    if (autoTimer.current) {
      window.clearInterval(autoTimer.current)
      autoTimer.current = null
    }
    setIsAutoPlaying(false)
  }

  async function handleStop() {
    stopAuto()
    const sid = streamState.sessionId
    if (sid && streamState.mode === 'backtest') {
      try { await api.backtestStop(sid) } catch {}
    }
    exitStream()
    chartActions.clearFills()
    setActive(false)
    setRemaining(0)
    setMetrics(null)
    streamState.resetCache?.()
    chartActions.resetData()
    chartActions.refreshChan()
  }

  async function handleLoadResult() {
    if (!selResult) {
      setError('请选择回测结果')
      return
    }
    setError(null)
    setLoadingResult(true)
    try {
      const resp = await api.backtestResultTrades({
        id: selResult,
        symbol: resultSymbol === '__all__' ? undefined : resultSymbol,
      })
      const result = results.find((x) => x.id === selResult)
      chartActions.clearResultTrades()
      chartActions.showResultTrades({
        resultId: resp.id,
        label: result?.label || resp.label,
        trades: resp.trades,
      })
    } catch (e) {
      setError(`加载回测结果失败: ${e}`)
    } finally {
      setLoadingResult(false)
    }
  }

  // 卸载时清定时器
  useEffect(() => () => {
    if (autoTimer.current) window.clearInterval(autoTimer.current)
  }, [])

  if (!open) return null

  const curStrategy = strategies.find((s) => s.name === selStrategy)

  return (
    <div style={{
      width: PANEL_WIDTH, flexShrink: 0, height: '100%',
      background: C.bg, color: C.text, borderLeft: `1px solid ${C.border}`,
      display: 'flex', flexDirection: 'column', overflow: 'hidden',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    }}>
      {/* Header */}
      <div style={{
        padding: '12px 12px 8px',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        borderBottom: `1px solid ${C.border}`,
      }}>
        <span style={{ fontSize: 15, fontWeight: 600 }}>回测</span>
        <button onClick={onClose} style={{
          width: 24, height: 24, border: 'none', background: 'transparent',
          cursor: 'pointer', color: C.text, fontSize: 16, lineHeight: 1, padding: 0,
        }} title="关闭">×</button>
      </div>

      <div style={{ flex: 1, overflowY: 'auto', padding: '12px' }}>
        {error && (
          <div style={{ color: C.danger, fontSize: 12, marginBottom: 8 }}>{error}</div>
        )}

        {!active ? (
          <>
            <div style={{
              marginBottom: 14, paddingBottom: 12,
              borderBottom: `1px solid ${C.border}`,
            }}>
              <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8 }}>历史回测结果</div>
              <Field label="结果" colors={C}>
                <select
                  value={selResult}
                  onChange={(e) => {
                    setSelResult(e.target.value)
                    setResultSymbol('__all__')
                  }}
                  style={inputStyle(C)}
                >
                  {results.map((r) => (
                    <option key={r.id} value={r.id}>
                      {r.label} · {r.trade_count}笔
                    </option>
                  ))}
                </select>
              </Field>
              <Field label="币种" colors={C}>
                <select
                  value={resultSymbol}
                  onChange={(e) => setResultSymbol(e.target.value)}
                  style={inputStyle(C)}
                >
                  <option value="__all__">全部币种</option>
                  {(results.find((r) => r.id === selResult)?.symbols ?? []).map((s) => (
                    <option key={s} value={s}>{s}</option>
                  ))}
                </select>
              </Field>
              <div style={{ display: 'flex', gap: 8 }}>
                <button
                  onClick={handleLoadResult}
                  disabled={loadingResult || !selResult}
                  style={{ ...btnPrimary(C), flex: 1 }}
                >
                  {loadingResult ? '加载中...' : '展示交易'}
                </button>
                <button onClick={chartActions.clearResultTrades} style={btn(C)}>
                  清除
                </button>
              </div>
            </div>

            <Field label="合约" colors={C}>
              <span style={{ fontSize: 13 }}>{exchange} / {symbol} · {freq}</span>
            </Field>

            <Field label="起始日期" colors={C}>
              <input
                type="text"
                value={startDate}
                onChange={(e) => setStartDate(e.target.value)}
                placeholder="2024-01-01 或 2024-01-01 09:00:00"
                style={inputStyle(C)}
              />
            </Field>

            <Field label="初始资金" colors={C}>
              <input
                type="number"
                value={initCash}
                onChange={(e) => setInitCash(e.target.value)}
                style={inputStyle(C)}
              />
            </Field>

            <Field label="策略" colors={C}>
              <select
                value={selStrategy}
                onChange={(e) => onStrategyChange(e.target.value)}
                style={inputStyle(C)}
              >
                {strategies.map((s) => (
                  <option key={s.name} value={s.name}>{s.label}</option>
                ))}
              </select>
            </Field>

            {curStrategy?.params.map((p) => (
              <Field key={p.name} label={p.label || p.name} colors={C} help={p.help}>
                {p.type === 'bool' ? (
                  <label style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
                    <input
                      type="checkbox"
                      checked={!!params[p.name]}
                      onChange={(e) => setParam(p.name, e.target.checked)}
                    />
                    <span style={{ fontSize: 12, color: C.sub }}>{params[p.name] ? '开' : '关'}</span>
                  </label>
                ) : (
                  <input
                    type={p.type === 'int' || p.type === 'float' ? 'number' : 'text'}
                    value={params[p.name] ?? ''}
                    onChange={(e) => {
                      let v: any = e.target.value
                      if (p.type === 'int') v = v === '' ? '' : parseInt(v, 10)
                      else if (p.type === 'float') v = v === '' ? '' : parseFloat(v)
                      setParam(p.name, v)
                    }}
                    style={inputStyle(C)}
                  />
                )}
              </Field>
            ))}

            {/* 合约成本参数：保证金占比 / 费率 / 固定手续费 / 滑点。spec 默认值由后端
                /contract/{symbol} 拉取并预填；用户编辑后会覆盖。 */}
            <div style={{
              marginTop: 14, paddingTop: 10, borderTop: `1px solid ${C.border}`,
              fontSize: 11, color: C.sub, marginBottom: 6,
            }}>
              交易成本（覆盖合约默认）
              {spec && (
                <span style={{ float: 'right' }}>
                  乘数 {spec.multiplier} · tick {spec.price_tick}
                </span>
              )}
            </div>
            <Field label="保证金占比 (%)" colors={C}
                   help={spec ? `合约默认 ${(spec.margin_rate * 100).toFixed(2)}%` : undefined}>
              <input
                type="number" step="0.01" min="0" max="100"
                value={marginPct}
                onChange={(e) => setMarginPct(e.target.value)}
                placeholder="如 8 表示 8%"
                style={inputStyle(C)}
              />
            </Field>
            <Field label="手续费率" colors={C}
                   help={'按成交金额收取，如 0.0001 = 万一；与「每手固定」二选一'}>
              <input
                type="number" step="0.00001" min="0"
                value={feeRate}
                onChange={(e) => setFeeRate(e.target.value)}
                placeholder="0.0001"
                style={inputStyle(C)}
              />
            </Field>
            <Field label="每手固定手续费 (¥)" colors={C}
                   help="若 > 0 则忽略手续费率，按每手定额收取">
              <input
                type="number" step="0.1" min="0"
                value={feeFixed}
                onChange={(e) => setFeeFixed(e.target.value)}
                placeholder="0 表示走费率"
                style={inputStyle(C)}
              />
            </Field>
            <Field label="滑点 (tick 数)" colors={C}
                   help="每次成交按 tick 数偏移：买价上滑、卖价下滑">
              <input
                type="number" step="0.5" min="0"
                value={slippageTicks}
                onChange={(e) => setSlippageTicks(e.target.value)}
                style={inputStyle(C)}
              />
            </Field>

            <button
              onClick={handleStart}
              style={{
                ...btnPrimary(C), width: '100%', marginTop: 12,
              }}
            >开始回测</button>
          </>
        ) : (
          <>
            {/* 控制：手动单步 + 速度倍率（1x/5x/20x，每秒 N 根 K）+ 停止 */}
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, marginBottom: 12 }}>
              <button
                onClick={() => { stopAuto(); void handleStep(1) }}
                style={btn(C)}
                title="手动下一根"
              >⏭</button>
              {([1, 5, 20] as const).map((s) => {
                const sel = isAutoPlaying && autoSpeed === s
                return (
                  <button
                    key={s}
                    onClick={() => playAt(s)}
                    style={sel ? btnActive(C) : btn(C)}
                    title={`每秒 ${s} 根 K`}
                  >{s}x</button>
                )
              })}
              {isAutoPlaying && (
                <button onClick={stopAuto} style={btn(C)} title="暂停">⏸</button>
              )}
              <button onClick={handleStop} style={{ ...btn(C), color: C.danger, marginLeft: 'auto' }}>停止</button>
            </div>

            <div style={{ fontSize: 12, color: C.sub, marginBottom: 8 }}>剩余 {remaining} 根</div>

            {/* 绩效指标 */}
            {metrics && <Metrics m={metrics} colors={C} />}
          </>
        )}
      </div>
    </div>
  )
}

// ---------- 子组件 ---------- //

type Colors = {
  bg: string; text: string; sub: string; border: string; hover: string
  inputBg: string; inputBorder: string
  accent: string; success: string; danger: string
}

function Field({
  label, children, colors, help,
}: { label: string; children: React.ReactNode; colors: Colors; help?: string }) {
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ fontSize: 11, color: colors.sub, marginBottom: 4 }}>{label}</div>
      {children}
      {help && <div style={{ fontSize: 10, color: colors.sub, marginTop: 2 }}>{help}</div>}
    </div>
  )
}

function inputStyle(c: Colors): React.CSSProperties {
  return {
    width: '100%', boxSizing: 'border-box',
    padding: '5px 8px', height: 28,
    background: c.inputBg, color: c.text,
    border: `1px solid ${c.inputBorder}`, borderRadius: 4,
    fontSize: 12, outline: 'none',
  }
}

function btn(c: Colors): React.CSSProperties {
  return {
    padding: '4px 10px', height: 26,
    background: 'transparent', color: c.text,
    border: `1px solid ${c.inputBorder}`, borderRadius: 4,
    fontSize: 12, cursor: 'pointer',
  }
}

// 当前速度档位高亮（蓝底白字），表明「自动播放正在以此速率运行」
function btnActive(c: Colors): React.CSSProperties {
  return {
    padding: '4px 10px', height: 26,
    background: c.accent, color: '#ffffff',
    border: `1px solid ${c.accent}`, borderRadius: 4,
    fontSize: 12, cursor: 'pointer',
  }
}

function btnPrimary(c: Colors): React.CSSProperties {
  return {
    padding: '6px 12px', height: 32,
    background: c.accent, color: '#ffffff',
    border: 'none', borderRadius: 4,
    fontSize: 13, fontWeight: 500, cursor: 'pointer',
  }
}

function Metrics({ m, colors }: { m: BacktestState; colors: Colors }) {
  const pnlColor = m.total_pnl > 0 ? colors.danger /* 中国习惯 涨红 */
                  : m.total_pnl < 0 ? colors.success : colors.text
  const rows: { label: string; value: string; color?: string }[] = [
    { label: '当前权益', value: m.equity.toLocaleString('zh-CN', { maximumFractionDigits: 2 }) },
    { label: '总盈亏', value: `${m.total_pnl >= 0 ? '+' : ''}${m.total_pnl.toFixed(2)}`, color: pnlColor },
    { label: '收益率', value: `${m.total_return_pct.toFixed(2)}%`, color: pnlColor },
    { label: '最大回撤', value: `${m.max_drawdown_pct.toFixed(2)}%`, color: colors.success },
    { label: '胜率', value: `${m.win_rate_pct.toFixed(1)}%` },
    { label: '夏普', value: m.sharpe.toFixed(2) },
    { label: '交易数', value: String(m.n_trades) },
    { label: '成交数', value: String(m.fills_count) },
    { label: '当前持仓', value: String(m.position_qty) },
  ]
  return (
    <div style={{ paddingTop: 8, borderTop: `1px solid ${colors.border}` }}>
      <div style={{ fontSize: 12, fontWeight: 500, marginBottom: 8 }}>绩效</div>
      {rows.map((r) => (
        <div key={r.label} style={{
          display: 'flex', justifyContent: 'space-between',
          padding: '3px 0', fontSize: 12,
        }}>
          <span style={{ color: colors.sub }}>{r.label}</span>
          <span style={{ color: r.color ?? colors.text, fontVariantNumeric: 'tabular-nums' }}>
            {r.value}
          </span>
        </div>
      ))}
    </div>
  )
}
