// 流式状态：Datafeed 和 TradingViewChart 之间共享 cutoff 时间戳和 onTick 回调。
// 同时支持「回放」「回测」两种模式 —— 它们的数据流完全一致：
//   - getBars 时按 cutoffTs 截断（不返回起点之后的 K）
//   - step 后通过 pushBar 推新 bar 进 TV
//   - 同 symbol 强制重拉数据要先调 resetCache + chart.resetData
// 唯一不同：chan 数据各自走 /replay/{id}/chan 或 /backtest/{id}/chan，
// 由 TradingViewChart 通过 mode 字段分发。

export type StreamMode = 'replay' | 'backtest' | null

export const streamState = {
  mode: null as StreamMode,
  /** 当前会话 id（mode 为 null 时也是 null）。 */
  sessionId: null as string | null,
  /** 截止 unix 秒，bars/chan 都不超过此时间。step 推进时这个值会随新 bar 增大。 */
  cutoffTs: 0,
  /** TV subscribeBars 注册的 onTick：调它把新 bar 实时喂进图表。 */
  pushBar: null as ((bar: { time: number; open: number; high: number; low: number; close: number; volume: number }) => void) | null,
  /** TV subscribeBars 注册的 onResetCacheNeededCallback：清 TV bar 缓存。 */
  resetCache: null as (() => void) | null,
}

/** 是否在流式模式（回放或回测均算）。 */
export function isStreamActive(): boolean {
  return streamState.mode !== null
}

/** 进入回放模式。 */
export function enterReplay(sessionId: string, cutoffTs: number): void {
  streamState.mode = 'replay'
  streamState.sessionId = sessionId
  streamState.cutoffTs = cutoffTs
  // 不清 pushBar/resetCache：它们由 subscribeBars 维护，跟模式无关
}

/** 进入回测模式。 */
export function enterBacktest(sessionId: string, cutoffTs: number): void {
  streamState.mode = 'backtest'
  streamState.sessionId = sessionId
  streamState.cutoffTs = cutoffTs
}

/** 推进：把新 bar 数组挨个喂给 TV，并把 cutoffTs 提升到最后一根。 */
export function advanceStream(newBars: { t: number; o: number; h: number; l: number; c: number; v: number }[]): void {
  for (const b of newBars) {
    if (b.t > streamState.cutoffTs) streamState.cutoffTs = b.t
    streamState.pushBar?.({
      time: b.t * 1000,
      open: b.o, high: b.h, low: b.l, close: b.c, volume: b.v,
    })
  }
}

/** 退出流式模式（回放或回测停止时调用）。 */
export function exitStream(): void {
  streamState.mode = null
  streamState.sessionId = null
  streamState.cutoffTs = 0
  // pushBar/resetCache 不清，subscribeBars 还活着；切回历史模式后下次拉数据正常走
}
