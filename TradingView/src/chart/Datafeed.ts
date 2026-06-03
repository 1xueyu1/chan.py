// TradingView Custom Datafeed：从后端 /bars 拉数据。
//
// Symbol 命名约定：`<EXCHANGE>:<SYMBOL>` 例如 "SHFE:RB"。
// TradingView resolution 例如 "5"（5min）。

import { api } from '../api/client'
import { resolutionToFreq } from './resolution'
import { isStreamActive, streamState } from './streamState'
import { symbolNameCn } from './symbolNames'

interface SymbolInfo {
  name: string
  ticker: string
  description: string
  type: string
  session: string
  timezone: string
  exchange: string
  minmov: number
  pricescale: number
  has_intraday: boolean
  visible_plots_set: string
  supported_resolutions: string[]
  volume_precision: number
  data_status: string
}

const SUPPORTED_RESOLUTIONS = ['1', '3', '5', '15', '30', '60', '120', '240', 'D']

export function makeDatafeed() {
  return {
    onReady: (cb: (cfg: any) => void) => {
      setTimeout(
        () =>
          cb({
            supported_resolutions: SUPPORTED_RESOLUTIONS,
            supports_marks: false,
            supports_timescale_marks: false,
            supports_time: true,
            exchanges: [
              { value: 'BINANCE', name: 'Binance Futures', desc: 'BINANCE' },
              { value: 'SHFE', name: '上期所', desc: 'SHFE' },
              { value: 'DCE', name: '大商所', desc: 'DCE' },
              { value: 'CZCE', name: '郑商所', desc: 'CZCE' },
              { value: 'INE', name: '上海能源', desc: 'INE' },
              { value: 'GFEX', name: '广期所', desc: 'GFEX' },
              { value: 'CFFEX', name: '中金所', desc: 'CFFEX' },
            ],
            symbols_types: [{ name: 'Futures', value: 'futures' }],
          }),
        0,
      )
    },

    searchSymbols: async (
      userInput: string,
      exchange: string,
      _symbolType: string,
      onResult: (items: any[]) => void,
    ) => {
      try {
        const params = new URLSearchParams()
        if (userInput) params.set('q', userInput)
        if (exchange) params.set('exchange', exchange)
        const r = await fetch(`/api/symbols?${params}`)
        const data = await r.json() as { symbols: Array<{
          exchange: string; symbol: string; ticker: string
        }> }
        onResult(
          (data.symbols || []).map((s) => ({
            symbol: s.symbol,
            full_name: s.ticker,
            description: `${symbolNameCn(s.symbol)} 主力连续（${s.exchange}/${s.symbol}）`,
            exchange: s.exchange,
            ticker: s.ticker,
            type: 'futures',
          })),
        )
      } catch (e) {
        console.warn('searchSymbols failed', e)
        onResult([])
      }
    },

    resolveSymbol: (
      symbolName: string,
      onResolve: (info: SymbolInfo) => void,
      onError: (err: string) => void,
    ) => {
      // symbolName 形如 "SHFE:RB"
      const [exchange, symbol] = symbolName.includes(':')
        ? symbolName.split(':')
        : ['BINANCE', symbolName]
      if (!exchange || !symbol) {
        onError(`invalid symbol: ${symbolName}`)
        return
      }
      setTimeout(() => {
        const cn = symbolNameCn(symbol)
        onResolve({
          // name 显示在左上角 badge（小标签）。给中文便于识别。
          name: cn,
          ticker: `${exchange}:${symbol}`,
          // description 显示在图表顶部「合约名称区」。
          description: `${cn} 主力连续`,
          type: 'futures',
          // 国内期货周末不开盘，写 '24x7' 会让 TV 在周末也保留空位 → 日线 filled_area
          // 中枢/段中枢在每个周末断一格。'0000-0000:23456' 表示 24h × 周一到周五。
          // 数字 1=周日 2=周一 ... 7=周六。
          session: exchange === 'BINANCE' ? '24x7' : '0000-0000:23456',
          timezone: 'Asia/Shanghai',
          exchange,
          minmov: 1,
          // 国内期货大多是整数价格变动（RB/HC=1, CU=10, AU=0.02 等），
          // 这里给 1（无小数）足以满足 RB；后面可按品种区分
          pricescale: symbol.toUpperCase().includes('BTC') ? 10 : 1,
          has_intraday: true,
          visible_plots_set: 'ohlcv',
          supported_resolutions: SUPPORTED_RESOLUTIONS,
          volume_precision: 0,
          // 'streaming' 是 TV 默认；'endofday' 会让 TV 认为仅日线，影响 intraday 显示
          data_status: 'streaming',
        })
      }, 0)
    },

    getBars: async (
      symbolInfo: SymbolInfo,
      resolution: string,
      periodParams: { from: number; to: number; countBack: number; firstDataRequest: boolean },
      onResult: (bars: any[], meta: { noData: boolean }) => void,
      onError: (err: string) => void,
    ) => {
      try {
        const freq = resolutionToFreq(resolution)
        // 用 countBack + to 而不是 from + to：
        // - TV 的 from 是按"线性时间" countBack*interval 算的，但期货有夜盘/休市 gap，
        //   严格按 from..to 切片会返回比 countBack 少很多的 bars，导致首屏左侧大片空白。
        // - 直接让后端从 to 倒推 countBack 根（跳过 gap），首屏永远能塞满 viewport。
        // 后端用 symbol 找 CSV 文件，必须是原始代号。symbolInfo.name 现在是中文（用于
        // badge 显示），所以从 ticker（始终 'EXCH:CODE' 形式）解析出代号。
        const code = symbolInfo.ticker.includes(':')
          ? symbolInfo.ticker.split(':')[1]
          : symbolInfo.ticker
        // 回放/回测模式下，把 to 截断到 cutoffTs（取较小者），保证只返回游标之前的 K。
        const effectiveTo = isStreamActive()
          ? Math.min(periodParams.to, streamState.cutoffTs)
          : periodParams.to
        const resp = await api.bars({
          exchange: symbolInfo.exchange,
          symbol: code,
          freq,
          to: unixToIsoLocal(effectiveTo),
          limit: periodParams.countBack,
        })
        const bars = resp.bars.map((b) => ({
          time: b.t * 1000, // TradingView 要毫秒
          open: b.o, high: b.h, low: b.l, close: b.c, volume: b.v,
        }))
        onResult(bars, { noData: bars.length === 0 })
      } catch (e) {
        onError(String(e))
      }
    },

    subscribeBars: (
      _symbolInfo: SymbolInfo,
      _resolution: string,
      onTick: (bar: { time: number; open: number; high: number; low: number; close: number; volume: number }) => void,
      _subscriberUID: string,
      onResetCacheNeededCallback?: () => void,
    ) => {
      // 回放/回测模式用 onTick 把新 bar 实时推给 TV（advanceStream 里调用）。
      // onResetCacheNeededCallback 让我们通知 TV 清空它的 bar 缓存，配合 chart.resetData()
      // 实现「同 symbol 强制重拉数据」—— 否则 TV 看 symbol 没变就不会再调 getBars。
      streamState.pushBar = onTick
      streamState.resetCache = onResetCacheNeededCallback ?? null
    },
    unsubscribeBars: () => {
      streamState.pushBar = null
      streamState.resetCache = null
    },
  }
}

function unixToIsoLocal(unix: number): string {
  const d = new Date(unix * 1000)
  // 用本地（不是 UTC）格式输出 YYYY-MM-DD HH:MM:SS，给 backend 当北京时间用
  const pad = (n: number) => String(n).padStart(2, '0')
  return (
    `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ` +
    `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`
  )
}
