import { useEffect, useMemo, useState } from 'react'
import { api } from '../api/client'
import type { DryRunBot, DryRunBotsResp, DryRunTrade } from '../api/types'

const PANEL_WIDTH = 460

interface Props {
  open: boolean
  onClose: () => void
  theme: 'light' | 'dark'
}

function palette(theme: 'light' | 'dark') {
  return theme === 'dark'
    ? { bg: '#131722', panel: '#1e222d', text: '#d1d4dc', sub: '#787b86', border: '#2a2e39', good: '#089981', bad: '#ef4444' }
    : { bg: '#fff', panel: '#f8f9fd', text: '#131722', sub: '#787b86', border: '#e0e3eb', good: '#089981', bad: '#d32f2f' }
}

function pct(v: number | null | undefined) {
  if (v === null || v === undefined || Number.isNaN(v)) return '-'
  return `${(Number(v) * 100).toFixed(2)}%`
}

function money(v: number | null | undefined) {
  if (v === null || v === undefined || Number.isNaN(v)) return '-'
  return Number(v).toFixed(2)
}

function timeText(v: string | undefined) {
  if (!v) return '-'
  const d = new Date(v)
  if (Number.isNaN(d.getTime())) return String(v)
  return d.toLocaleString('zh-CN', { hour12: false })
}

function TradeRows({ rows, C }: { rows: DryRunTrade[]; C: ReturnType<typeof palette> }) {
  if (!rows.length) return <div style={{ color: C.sub, fontSize: 12, padding: '10px 0' }}>暂无记录</div>
  return (
    <div style={{ display: 'grid', gap: 6 }}>
      {rows.slice(0, 12).map((t) => {
        const profit = Number(t.profit_abs ?? 0)
        return (
          <div key={`${t.id}-${t.close_date || t.open_date}`} style={{ border: `1px solid ${C.border}`, borderRadius: 6, padding: 8 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: 8 }}>
              <b>{t.pair}</b>
              <span style={{ color: t.side === 'long' ? C.bad : C.good }}>{t.side === 'long' ? '多' : '空'}</span>
            </div>
            <div style={{ color: C.sub, fontSize: 12, marginTop: 4 }}>
              开仓 {timeText(t.open_date)} · {money(t.open_rate)} · 仓位 {money(t.stake_amount)}
            </div>
            {!t.is_open && (
              <div style={{ color: profit >= 0 ? C.good : C.bad, fontSize: 12, marginTop: 4 }}>
                平仓 {timeText(t.close_date)} · {money(t.close_rate)} · 盈亏 {money(t.profit_abs)} / {pct(t.profit_ratio)}
              </div>
            )}
            {(t.enter_tag || t.exit_reason) && (
              <div style={{ color: C.sub, fontSize: 12, marginTop: 4, wordBreak: 'break-all' }}>
                {t.enter_tag || '-'} {t.exit_reason ? `→ ${t.exit_reason}` : ''}
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}

function BotCard({ bot, C }: { bot: DryRunBot; C: ReturnType<typeof palette> }) {
  const s = bot.summary
  return (
    <section style={{ background: C.panel, border: `1px solid ${C.border}`, borderRadius: 8, padding: 12 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, alignItems: 'baseline' }}>
        <h3 style={{ margin: 0, fontSize: 16 }}>{bot.name}</h3>
        <span style={{ color: bot.exists ? C.good : C.bad, fontSize: 12 }}>{bot.exists ? '运行库已生成' : '未生成数据库'}</span>
      </div>
      <div style={{ color: C.sub, fontSize: 12, marginTop: 4 }}>{bot.strategy} · API :{bot.api_port}</div>
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 8, marginTop: 12 }}>
        <Metric label="总笔数" value={s.total_trades} C={C} />
        <Metric label="持仓" value={s.open_trades} C={C} />
        <Metric label="胜率" value={pct(s.win_rate)} C={C} />
        <Metric label="盈亏" value={money(s.total_profit_abs)} C={C} color={Number(s.total_profit_abs ?? 0) >= 0 ? C.good : C.bad} />
      </div>
      <h4 style={{ margin: '14px 0 6px', fontSize: 13 }}>当前持仓</h4>
      <TradeRows rows={bot.open_trades} C={C} />
      <h4 style={{ margin: '14px 0 6px', fontSize: 13 }}>最近平仓</h4>
      <TradeRows rows={bot.recent_closed} C={C} />
    </section>
  )
}

function Metric({ label, value, C, color }: { label: string; value: any; C: ReturnType<typeof palette>; color?: string }) {
  return (
    <div style={{ border: `1px solid ${C.border}`, borderRadius: 6, padding: 8 }}>
      <div style={{ color: C.sub, fontSize: 11 }}>{label}</div>
      <div style={{ color: color ?? C.text, fontSize: 15, marginTop: 3 }}>{value ?? '-'}</div>
    </div>
  )
}

export function DryRunPanel({ open, onClose, theme }: Props) {
  const C = palette(theme)
  const [payload, setPayload] = useState<DryRunBotsResp | null>(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  function refresh() {
    setLoading(true)
    setError('')
    api.dryrunBots()
      .then(setPayload)
      .catch((e) => setError(String(e)))
      .finally(() => setLoading(false))
  }

  useEffect(() => {
    if (!open) return
    refresh()
    const timer = window.setInterval(refresh, 10000)
    return () => window.clearInterval(timer)
  }, [open])

  const bots = useMemo(() => payload?.bots ?? [], [payload])

  if (!open) return null
  return (
    <aside style={{ width: PANEL_WIDTH, height: '100%', background: C.bg, color: C.text, borderLeft: `1px solid ${C.border}`, overflow: 'auto' }}>
      <div style={{ position: 'sticky', top: 0, zIndex: 2, background: C.bg, borderBottom: `1px solid ${C.border}`, padding: 12, display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div>
          <div style={{ fontSize: 16, fontWeight: 700 }}>模拟实盘</div>
          <div style={{ color: C.sub, fontSize: 12 }}>不定仓 vs 结构风险定仓</div>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <button onClick={refresh} style={{ border: `1px solid ${C.border}`, background: C.panel, color: C.text, borderRadius: 4, padding: '5px 8px', cursor: 'pointer' }}>{loading ? '刷新中' : '刷新'}</button>
          <button onClick={onClose} style={{ border: `1px solid ${C.border}`, background: C.panel, color: C.text, borderRadius: 4, padding: '5px 8px', cursor: 'pointer' }}>关闭</button>
        </div>
      </div>
      <div style={{ padding: 12, display: 'grid', gap: 12 }}>
        {error && <div style={{ color: C.bad, fontSize: 13 }}>{error}</div>}
        {bots.map((bot) => <BotCard key={bot.id} bot={bot} C={C} />)}
        {!bots.length && !error && <div style={{ color: C.sub, fontSize: 13 }}>暂无 bot 配置</div>}
      </div>
    </aside>
  )
}
