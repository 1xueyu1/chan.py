// TradingView resolution 字符串 ↔ 后端 freq 字符串。

export const FREQS = ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '1d'] as const
export type Freq = typeof FREQS[number]

export function freqToResolution(freq: Freq): string {
  const map: Record<Freq, string> = {
    '1m': '1', '3m': '3', '5m': '5', '15m': '15', '30m': '30',
    '1h': '60', '2h': '120', '4h': '240', '1d': 'D',
  }
  return map[freq]
}

export function resolutionToFreq(res: string): Freq {
  const map: Record<string, Freq> = {
    '1': '1m', '3': '3m', '5': '5m', '15': '15m', '30': '30m',
    '60': '1h', '120': '2h', '240': '4h', 'D': '1d', '1D': '1d',
  }
  const f = map[res]
  if (!f) throw new Error(`Unknown TradingView resolution: ${res}`)
  return f
}
