import { useEffect, useMemo, useState } from 'react'
import { api } from '../api/client'
import type { ModelInfo, ModelsResp } from '../api/types'

const PANEL_WIDTH = 420

interface Props {
  open: boolean
  onClose: () => void
  theme: 'light' | 'dark'
}

type Colors = {
  bg: string
  panel: string
  text: string
  sub: string
  border: string
  hover: string
  inputBg: string
  accent: string
  success: string
  danger: string
}

function colors(theme: 'light' | 'dark'): Colors {
  return theme === 'dark'
    ? {
        bg: '#131722',
        panel: '#1e222d',
        text: '#d1d4dc',
        sub: '#787b86',
        border: '#2a2e39',
        hover: 'rgba(255,255,255,0.04)',
        inputBg: '#161a24',
        accent: '#2962ff',
        success: '#089981',
        danger: '#ef4444',
      }
    : {
        bg: '#ffffff',
        panel: '#f8f9fd',
        text: '#131722',
        sub: '#787b86',
        border: '#e0e3eb',
        hover: 'rgba(0,0,0,0.04)',
        inputBg: '#ffffff',
        accent: '#2962ff',
        success: '#089981',
        danger: '#d32f2f',
      }
}

function fmtTime(ts: number | null | undefined): string {
  if (!ts) return '-'
  return new Date(ts * 1000).toLocaleString('zh-CN', { hour12: false })
}

function fmtMetric(key: string, value: any): string {
  if (value === null || value === undefined || value === '') return '-'
  if (typeof value === 'boolean') return value ? '是' : '否'
  if (typeof value !== 'number') return String(value)
  if (key.includes('rate') || key.includes('return') || key.includes('drawdown')) {
    return `${(value * 100).toFixed(2)}%`
  }
  if (key.includes('factor')) return value.toFixed(2)
  if (Math.abs(value) >= 100) return value.toFixed(0)
  return value.toFixed(4)
}

function metricLabel(key: string): string {
  const map: Record<string, string> = {
    trades: '交易数',
    win_rate: '胜率',
    total_return: '收益',
    profit_factor: 'PF',
    max_drawdown: '最大回撤',
    qualified_candidates: '合格候选',
    candidate_events: '候选事件',
  }
  return map[key] ?? key
}

function firstParagraph(text: string): string {
  const cleaned = text
    .replace(/```[\s\S]*?```/g, '')
    .split('\n')
    .map((x) => x.trim())
    .filter((x) => x && !x.startsWith('|') && !x.startsWith('- ') && !x.startsWith('#'))
  return cleaned.slice(0, 3).join('\n')
}

export function ModelPanel({ open, onClose, theme }: Props) {
  const C = colors(theme)
  const [payload, setPayload] = useState<ModelsResp | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [query, setQuery] = useState('')
  const [selected, setSelected] = useState<string>('')

  function refresh() {
    setError(null)
    api.models()
      .then((resp) => {
        setPayload(resp)
        if (!selected && resp.models.length) setSelected(resp.models[0].route)
      })
      .catch((e) => setError(String(e)))
  }

  useEffect(() => {
    if (open) refresh()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open])

  const models = useMemo(() => {
    const q = query.trim().toLowerCase()
    const all = payload?.models ?? []
    if (!q) return all
    return all.filter((m) =>
      [m.route, m.goal, m.label, m.features, m.status].some((x) => String(x).toLowerCase().includes(q)),
    )
  }, [payload, query])

  const current: ModelInfo | null = useMemo(() => {
    return models.find((m) => m.route === selected) ?? models[0] ?? null
  }, [models, selected])

  if (!open) return null

  return (
    <div style={{
      width: PANEL_WIDTH,
      flexShrink: 0,
      height: '100%',
      background: C.bg,
      color: C.text,
      borderLeft: `1px solid ${C.border}`,
      display: 'flex',
      flexDirection: 'column',
      overflow: 'hidden',
      fontFamily: '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
    }}>
      <div style={{
        padding: '12px 12px 8px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        borderBottom: `1px solid ${C.border}`,
      }}>
        <div>
          <div style={{ fontSize: 15, fontWeight: 600 }}>模型</div>
          <div style={{ fontSize: 11, color: C.sub, marginTop: 2 }}>
            {payload ? `${payload.count} 个模型 · ${payload.registry_path}` : '读取中'}
          </div>
        </div>
        <div style={{ display: 'flex', gap: 6 }}>
          <button onClick={refresh} style={iconBtn(C)} title="刷新">↻</button>
          <button onClick={onClose} style={iconBtn(C)} title="关闭">×</button>
        </div>
      </div>

      <div style={{ padding: 12, borderBottom: `1px solid ${C.border}` }}>
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="搜索路线、目标、状态"
          style={{
            width: '100%',
            boxSizing: 'border-box',
            height: 30,
            padding: '5px 8px',
            border: `1px solid ${C.border}`,
            borderRadius: 4,
            background: C.inputBg,
            color: C.text,
            outline: 'none',
            fontSize: 12,
          }}
        />
      </div>

      {error && <div style={{ color: C.danger, fontSize: 12, padding: 12 }}>{error}</div>}

      <div style={{ display: 'flex', minHeight: 0, flex: 1 }}>
        <div style={{
          width: 158,
          flexShrink: 0,
          borderRight: `1px solid ${C.border}`,
          overflowY: 'auto',
          padding: 8,
        }}>
          {models.map((m) => {
            const active = current?.route === m.route
            return (
              <button
                key={m.route}
                onClick={() => setSelected(m.route)}
                style={{
                  width: '100%',
                  textAlign: 'left',
                  border: `1px solid ${active ? C.accent : 'transparent'}`,
                  background: active ? (theme === 'dark' ? '#17233f' : '#eef4ff') : 'transparent',
                  color: C.text,
                  borderRadius: 6,
                  padding: '7px 8px',
                  marginBottom: 6,
                  cursor: 'pointer',
                }}
              >
                <div style={{ fontSize: 12, fontWeight: 600, wordBreak: 'break-word' }}>{m.route}</div>
                <div style={{ color: C.sub, fontSize: 10, marginTop: 3, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                  {m.status || (m.artifact.has_artifact ? '有模型产物' : '未训练')}
                </div>
              </button>
            )
          })}
        </div>

        <div style={{ flex: 1, minWidth: 0, overflowY: 'auto', padding: 12 }}>
          {current ? (
            <>
              <div style={{ fontSize: 16, fontWeight: 700, wordBreak: 'break-word' }}>{current.route}</div>
              <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', marginTop: 8 }}>
                <Badge text={current.status || '未登记状态'} color={C.accent} />
                <Badge text={current.artifact.has_artifact ? '有模型产物' : '无模型产物'} color={current.artifact.has_artifact ? C.success : C.sub} />
              </div>

              <Section title="模型介绍">
                <InfoRow label="目标" value={current.goal || '-'} colors={C} />
                <InfoRow label="标签" value={current.label || '-'} colors={C} />
                <InfoRow label="特点" value={current.features || '-'} colors={C} />
                {current.description && (
                  <pre style={preStyle(C)}>{firstParagraph(current.description)}</pre>
                )}
              </Section>

              <Section title="产物">
                <InfoRow label="目录" value={current.artifact.model_dir || '-'} colors={C} />
                <InfoRow label="更新时间" value={fmtTime(current.artifact.updated_at)} colors={C} />
                {current.artifact.files.length > 0 && (
                  <div style={{ marginTop: 8 }}>
                    {current.artifact.files.map((f) => (
                      <div key={f.path} style={{
                        display: 'flex',
                        justifyContent: 'space-between',
                        gap: 8,
                        padding: '4px 0',
                        borderTop: `1px solid ${C.border}`,
                        fontSize: 11,
                      }}>
                        <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{f.name}</span>
                        <span style={{ color: C.sub, flexShrink: 0 }}>{Math.max(1, Math.round(f.size / 1024))} KB</span>
                      </div>
                    ))}
                  </div>
                )}
              </Section>

              <Section title="最近回测">
                {current.backtests.length === 0 ? (
                  <div style={{ color: C.sub, fontSize: 12 }}>暂无 backtest_metrics.json</div>
                ) : current.backtests.map((bt) => (
                  <div key={bt.path} style={{
                    padding: 8,
                    border: `1px solid ${C.border}`,
                    borderRadius: 6,
                    marginBottom: 8,
                    background: C.panel,
                  }}>
                    <div style={{ fontSize: 11, color: C.sub, marginBottom: 6 }}>{bt.path}</div>
                    <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6 }}>
                      {Object.entries(bt.metrics).map(([k, v]) => (
                        <InfoRow key={k} label={metricLabel(k)} value={fmtMetric(k, v)} colors={C} compact />
                      ))}
                    </div>
                  </div>
                ))}
              </Section>
            </>
          ) : (
            <div style={{ color: C.sub, fontSize: 12 }}>没有模型记录</div>
          )}
        </div>
      </div>
    </div>
  )
}

function Badge({ text, color }: { text: string; color: string }) {
  return (
    <span style={{
      display: 'inline-flex',
      alignItems: 'center',
      height: 20,
      padding: '0 7px',
      borderRadius: 4,
      border: `1px solid ${color}`,
      color,
      fontSize: 11,
      maxWidth: '100%',
    }}>
      {text}
    </span>
  )
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div style={{ marginTop: 16 }}>
      <div style={{ fontSize: 12, fontWeight: 700, marginBottom: 8 }}>{title}</div>
      <div>{children}</div>
    </div>
  )
}

function InfoRow({ label, value, colors, compact }: { label: string; value: string; colors: Colors; compact?: boolean }) {
  return (
    <div style={{
      display: 'flex',
      justifyContent: 'space-between',
      gap: 8,
      fontSize: compact ? 11 : 12,
      padding: compact ? 0 : '3px 0',
    }}>
      <span style={{ color: colors.sub, flexShrink: 0 }}>{label}</span>
      <span style={{ color: colors.text, textAlign: 'right', wordBreak: 'break-word' }}>{value}</span>
    </div>
  )
}

function iconBtn(c: Colors): React.CSSProperties {
  return {
    width: 26,
    height: 26,
    border: `1px solid ${c.border}`,
    borderRadius: 4,
    background: 'transparent',
    color: c.text,
    cursor: 'pointer',
    fontSize: 15,
    lineHeight: 1,
  }
}

function preStyle(c: Colors): React.CSSProperties {
  return {
    whiteSpace: 'pre-wrap',
    wordBreak: 'break-word',
    margin: '8px 0 0',
    padding: 8,
    border: `1px solid ${c.border}`,
    borderRadius: 6,
    background: c.panel,
    color: c.text,
    fontSize: 11,
    lineHeight: 1.5,
    fontFamily: 'inherit',
  }
}
