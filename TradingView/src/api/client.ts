// 后端 HTTP 客户端。dev 时走 Vite 代理 (/api → :3000)。

import type {
  BacktestResultTradesResp, BacktestResultsResp,
  BacktestStartResp, BacktestState, BacktestStepResp, BacktestStrategy,
  Bar, ChanBuildInfo, ChanSlice, ContractSpec, DryRunBotsResp, ModelsResp,
  ReplayStartResp, ReplayStepResp,
} from './types'

const BASE = '/api'

export interface BarsQuery {
  exchange?: string
  symbol: string
  freq: string
  from?: string
  to?: string
  limit?: number
}

export interface ChanQuery {
  exchange?: string
  symbol: string
  freq: string
  from?: string
  to?: string
}

function qs(params: Record<string, string | number | boolean | undefined | null>): string {
  const u = new URLSearchParams()
  for (const [k, v] of Object.entries(params)) {
    if (v === undefined || v === null || v === '') continue
    u.set(k, String(v))
  }
  const s = u.toString()
  return s ? `?${s}` : ''
}

async function http<T>(path: string, init?: RequestInit): Promise<T> {
  const r = await fetch(`${BASE}${path}`, init)
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}: ${path}`)
  return r.json() as Promise<T>
}

export const api = {
  health: () => http<{ ok: boolean; data_root: string }>('/health'),

  bars: (q: BarsQuery) =>
    http<{ symbol: string; freq: string; n: number; bars: Bar[] }>(
      `/bars${qs({ ...q, exchange: q.exchange ?? 'SHFE' })}`,
    ),

  /** 触发缠论全量计算（首次 ~5-17s，落 state.bin + snapshot.pkl；之后秒回）。 */
  chanBuild: (q: { exchange?: string; symbol: string; freq: string; force?: boolean }) =>
    http<ChanBuildInfo>(`/chan/build${qs({ ...q, exchange: q.exchange ?? 'SHFE' })}`, {
      method: 'POST',
    }),

  /** 按 [from, to] 切片返回笔/段/中枢/BSP（结构）。 */
  chan: (q: ChanQuery) =>
    http<ChanSlice>(`/chan${qs({ ...q, exchange: q.exchange ?? 'SHFE' })}`),

  /** 创建回放 session：从 at 时刻开始，剩余 bar 由 step 逐根喂入。 */
  replayStart: (q: { exchange?: string; symbol: string; freq: string; at: string }) =>
    http<ReplayStartResp>(`/replay/start${qs({ ...q, exchange: q.exchange ?? 'SHFE' })}`, {
      method: 'POST',
    }),

  /** 步进 N 根。返回新推入的 K 线 + 增量结构。 */
  replayStep: (sessionId: string, n: number = 1) =>
    http<ReplayStepResp>(`/replay/${sessionId}/step${qs({ n })}`, { method: 'POST' }),

  /** 释放 session。 */
  replayStop: (sessionId: string) =>
    http<{ ok: boolean }>(`/replay/${sessionId}`, { method: 'DELETE' }),

  /** 从回放 session 自己的 MultiChan 取切片（chan 跟着 step 演化，无未来知识）。 */
  replayChan: (sessionId: string, q: { from?: string; to?: string }) =>
    http<ChanSlice>(`/replay/${sessionId}/chan${qs(q)}`),

  // ---------- 合约规格 ---------- //
  contract: (symbol: string) =>
    http<ContractSpec>(`/contract/${encodeURIComponent(symbol)}`),

  models: () =>
    http<ModelsResp>('/models'),

  dryrunBots: () =>
    http<DryRunBotsResp>('/dryrun/bots'),

  // ---------- 回测 ---------- //
  backtestStrategies: () =>
    http<{ strategies: BacktestStrategy[] }>('/backtest/strategies'),

  backtestResults: () =>
    http<BacktestResultsResp>('/backtest/results'),

  backtestResultTrades: (q: { id: string; symbol?: string }) =>
    http<BacktestResultTradesResp>(`/backtest/result-trades${qs(q)}`),

  backtestStart: (body: {
    exchange?: string; symbol: string; freq: string; at: string
    strategy: string; params?: Record<string, any>
    init_cash?: number; slippage_ticks?: number
    // 覆盖合约 spec；不传走默认
    margin_rate?: number; fee_rate?: number; fee_fixed?: number
  }) =>
    http<BacktestStartResp>(`/backtest/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ exchange: body.exchange ?? 'SHFE', ...body }),
    }),

  backtestStep: (sessionId: string, n: number = 1) =>
    http<BacktestStepResp>(`/backtest/${sessionId}/step${qs({ n })}`, { method: 'POST' }),

  backtestState: (sessionId: string) =>
    http<BacktestState>(`/backtest/${sessionId}/state`),

  backtestChan: (sessionId: string, q: { from?: string; to?: string }) =>
    http<ChanSlice>(`/backtest/${sessionId}/chan${qs(q)}`),

  backtestStop: (sessionId: string) =>
    http<{ ok: boolean }>(`/backtest/${sessionId}`, { method: 'DELETE' }),
}
