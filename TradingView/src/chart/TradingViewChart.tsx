import { useEffect, useRef, useState } from 'react'
import { createPortal } from 'react-dom'

import { api } from '../api/client'
import type { BacktestFill, BacktestResultTrade } from '../api/types'
import { BacktestPanel, type ChartActions } from './BacktestPanel'
import { DryRunPanel } from './DryRunPanel'
import { ModelPanel } from './ModelPanel'
import { buildChanLookup, chanLookupHolder, commitChanLookup, makeChanIndicator, saveChanStyles } from './chanIndicator'
import { advanceStream, enterReplay, exitStream, isStreamActive, streamState } from './streamState'
import { makeDatafeed } from './Datafeed'
import { freqToResolution, resolutionToFreq, type Freq } from './resolution'

const TV_SCRIPT = '/charting_library/charting_library.js'
const CHAN_AFTER_DATA_LOADED_MS = 10
// 必须跟 chanIndicator.metainfo.description 完全一致
const CHAN_STUDY_NAME = 'Chan 缠论'

let scriptPromise: Promise<void> | null = null

function loadTradingView(): Promise<void> {
  if (typeof window !== 'undefined' && window.TradingView) return Promise.resolve()
  if (scriptPromise) return scriptPromise
  scriptPromise = new Promise((resolve, reject) => {
    const s = document.createElement('script')
    s.src = TV_SCRIPT
    s.async = true
    s.onload = () => resolve()
    s.onerror = () => reject(new Error(`无法加载 ${TV_SCRIPT}`))
    document.head.appendChild(s)
  })
  return scriptPromise
}

export interface ChartProps {
  exchange: string
  symbol: string
  freq: Freq
  /** TV 内部切了周期后回调（用户点左下角时间段按钮时），让上层同步 ControlPanel */
  onFreqChange?: (freq: Freq) => void
}

export function TradingViewChart(props: ChartProps) {
  const containerRef = useRef<HTMLDivElement | null>(null)
  const widgetRef = useRef<any>(null)
  const subRef = useRef<{ unsubscribe?: () => void } | null>(null)
  const rangeSubRef = useRef<{ unsubscribe?: () => void } | null>(null)
  const intervalSubRef = useRef<{ unsubscribe?: () => void } | null>(null)
  const debounceTimer = useRef<number | null>(null)
  const rangeDebounceTimer = useRef<number | null>(null)
  const propsRef = useRef(props)
  propsRef.current = props
  const reloadGen = useRef(0)
  // 上次成功提交的 (symbol, freq, range)。下次 refresh 若三者完全一致就跳过。
  const lastCommitted = useRef<{ key: string; from: number; to: number } | null>(null)
  // 串行化：同时只允许一个 refresh 跑。inFlight 期间到来的 trigger 全部合并到 pending，
  // 当前 refresh 结束后若 pending=true 再跑一轮。否则 setInputValues 引发的 onDataLoaded
  // 会让多个 refresh 互相 stale-abort，永远没人 commit。
  const refreshInFlight = useRef(false)
  const refreshPending = useRef(false)

  // ---------- 回放 UI state ---------- //
  const [replayDate, setReplayDate] = useState('')
  const [replayUiActive, setReplayUiActive] = useState(false)
  const [isAutoPlaying, setIsAutoPlaying] = useState(false)
  const [replayRemaining, setReplayRemaining] = useState(0)
  const [replayPanelOpen, setReplayPanelOpen] = useState(false)
  const [replayHover, setReplayHover] = useState(false)
  const [themeHover, setThemeHover] = useState(false)
  const autoPlayTimer = useRef<number | null>(null)
  // TV 顶部工具栏插槽（widget.createButton 创建）。React Portal 把「回放」开关按钮渲染
  // 进去，点击展开/收起底部控制面板。这样 UI 出现在 TV 自带的顶栏里，不占用图表区。
  const [headerEl, setHeaderEl] = useState<HTMLElement | null>(null)
  const [themeBtnEl, setThemeBtnEl] = useState<HTMLElement | null>(null)
  const [backtestBtnEl, setBacktestBtnEl] = useState<HTMLElement | null>(null)
  const [modelBtnEl, setModelBtnEl] = useState<HTMLElement | null>(null)
  const [dryrunBtnEl, setDryrunBtnEl] = useState<HTMLElement | null>(null)
  const [backtestPanelOpen, setBacktestPanelOpen] = useState(false)
  const [backtestHover, setBacktestHover] = useState(false)
  const [modelPanelOpen, setModelPanelOpen] = useState(false)
  const [modelHover, setModelHover] = useState(false)
  const [dryrunPanelOpen, setDryrunPanelOpen] = useState(false)
  const [dryrunHover, setDryrunHover] = useState(false)
  // 回测成交点 shape ids，用于追加和清理箭头
  const backtestFillShapeIds = useRef<any[]>([])
  const resultShapeIds = useRef<any[]>([])
  const [resultPanel, setResultPanel] = useState<{
    resultId: string
    label: string
    trades: BacktestResultTrade[]
    selectedId: number | null
  } | null>(null)
  const resultPanelRef = useRef<typeof resultPanel>(null)
  // 主题：从 localStorage 读，默认 light；切换时通过 widget.changeTheme 实时生效
  const [theme, setTheme] = useState<'light' | 'dark'>(() => {
    try { return (localStorage.getItem('chart-theme') as 'light' | 'dark') ?? 'light' }
    catch { return 'light' }
  })
  const themeRef = useRef(theme)
  themeRef.current = theme
  resultPanelRef.current = resultPanel

  useEffect(() => {
    let cancelled = false
    ;(async () => {
      await loadTradingView()
      if (cancelled || !containerRef.current) return

      const widget = new window.TradingView.widget({
        container: containerRef.current,
        library_path: '/charting_library/',
        datafeed: makeDatafeed(),
        symbol: `${propsRef.current.exchange}:${propsRef.current.symbol}`,
        interval: freqToResolution(propsRef.current.freq),
        fullscreen: false,
        autosize: true,
        theme: themeRef.current,
        timezone: 'Asia/Shanghai',
        locale: 'zh',
        client_id: 'chan-core',
        user_id: 'local',
        custom_indicators_getter: () => Promise.resolve([makeChanIndicator()]),
        // 用户的指标/周期收藏走 items_favoriting 单独的 localStorage 键，能存；
        // 但整张图表的布局（pane 高度等）一定要 disable, 否则会反复覆盖我们的 setHeight。
        disabled_features: [
          'header_compare',
          'header_saveload',
          'study_templates',
          'create_volume_indicator_by_default',
          'save_chart_properties_to_local_storage',
        ],
        enabled_features: [
          'hide_left_toolbar_by_default',
          'items_favoriting',  // 启用周期/指标收藏功能
        ],
        favorites: {
          intervals: ['1', '5', '15', '30', '60', '240', 'D'],
          chartTypes: ['Candles'],
        },
        // MACD 默认只画图，不显示价格刻度上的值标签 + 状态行（顶部）的值
        // display 是 bitmask：1=Pane 2=DataWindow 4=PriceScale 8=StatusLine → 3 = 只 1+2
        studies_overrides: {
          'macd.macd.display': 3,
          'macd.signal.display': 3,
          'macd.histogram.display': 3,
        },
        overrides: {
          'paneProperties.background': themeRef.current === 'light' ? '#ffffff' : '#131722',
          'paneProperties.backgroundType': 'solid',
          // 中国习惯：涨红跌绿；绿色用 TV 默认 #089981
          'mainSeriesProperties.candleStyle.upColor': '#ef4444',
          'mainSeriesProperties.candleStyle.downColor': '#089981',
          'mainSeriesProperties.candleStyle.borderUpColor': '#ef4444',
          'mainSeriesProperties.candleStyle.borderDownColor': '#089981',
          'mainSeriesProperties.candleStyle.wickUpColor': '#ef4444',
          'mainSeriesProperties.candleStyle.wickDownColor': '#089981',
          
        },
      })
      widgetRef.current = widget
      // 顶部工具栏插槽：headerReady 后 createButton 拿到原生 HTMLElement，用 React
      // Portal 把回放/主题按钮渲染进去。align 只支持 left/right，没有 middle —
      // 选 'left' 让插槽出现在左侧元素组的末尾（周期/指标按钮之后），视觉上更靠中间。
      widget.headerReady?.().then(() => {
        const makeSlot = (align: 'left' | 'right'): HTMLElement | null => {
          try {
            const el = widget.createButton({ align }) as HTMLElement
            el.style.padding = '0'
            el.style.border = 'none'
            el.style.background = 'transparent'
            return el
          } catch (e) {
            console.warn('createButton failed', e)
            return null
          }
        }
        // 回放/回测放左侧（周期/指标按钮之后），主题按钮放右侧（设置/搜索那侧）
        const replaySlot = makeSlot('left')
        const backtestSlot = makeSlot('left')
        const modelSlot = makeSlot('left')
        const dryrunSlot = makeSlot('left')
        const themeSlot = makeSlot('right')
        if (replaySlot) setHeaderEl(replaySlot)
        if (backtestSlot) setBacktestBtnEl(backtestSlot)
        if (modelSlot) setModelBtnEl(modelSlot)
        if (dryrunSlot) setDryrunBtnEl(dryrunSlot)
        if (themeSlot) setThemeBtnEl(themeSlot)
      })
      widget.onChartReady(async () => {
        const chart = widget.activeChart?.()
        if (!chart) return
        try {
          await chart.createStudy('MACD', false, false)
        } catch (e) {
          console.warn('createStudy MACD failed', e)
        }
        // MACD 副图大小：setHeight 单位是"伸展因子"（按 pane 比例分配总高度）。
        // 默认 pane[0]≈534、pane[1]≈267 是 2:1。设 100 → 大约 5:1，MACD 占 ~17%。
        // 调大调小就改这个数：30=很小，50=小，100=中，150=大，200=接近 1/3。
        try {
          const panes = chart.getPanes()
          if (panes.length >= 2) {
            panes[panes.length - 1].setHeight(150)
          }
        } catch (e) {
          console.warn('resize MACD pane failed', e)
        }
        // K 线默认宽度：bar spacing 是每根 bar 占的像素数。TV 默认 ~6，调大让 K 线
        // 更宽更易看。10=略宽，14=明显宽，20=很宽（首屏 bar 数会变少）。
        try {
          chart.getTimeScale?.()?.setBarSpacing?.(2)
        } catch (e) {
          console.warn('setBarSpacing failed', e)
        }
        attachDataLoadedListener(chart)
        attachIntervalListener(chart)
        attachVisibleRangeListener(chart)
        // 监听 chan study 的样式变化，把当前 getStyleValues() 持久化到 localStorage；
        // 下次 makeChanIndicator() 时把保存值合并进 metainfo defaults，跨刷新保留。
        try {
          widget.subscribe?.('study_properties_changed', (entityId: any) => {
            const studies: any[] = chart.getAllStudies?.() ?? []
            if (!studies.some((s) => s.name === CHAN_STUDY_NAME && s.id === entityId)) return
            const api = chart.getStudyById?.(entityId)
            const sv = api?.getStyleValues?.()
            if (sv) saveChanStyles(sv)
          })
        } catch (e) {
          console.warn('subscribe study_properties_changed failed', e)
        }
        void refreshChanStudy(chart)
      })
    })()
    return () => {
      cancelled = true
      try {
        subRef.current?.unsubscribe?.()
      } catch {}
      try {
        rangeSubRef.current?.unsubscribe?.()
      } catch {}
      try {
        intervalSubRef.current?.unsubscribe?.()
      } catch {}
      try {
        widgetRef.current?.remove?.()
      } catch {}
      widgetRef.current = null
      if (debounceTimer.current) {
        window.clearTimeout(debounceTimer.current)
        debounceTimer.current = null
      }
      if (rangeDebounceTimer.current) {
        window.clearTimeout(rangeDebounceTimer.current)
        rangeDebounceTimer.current = null
      }
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // 主题切换：localStorage 持久化 + 刷新页面让 chan 指标按新主题重建默认色（makeChanIndicator
  // 是在 widget 创建时调用的，主题切换不会重新读默认色）。chan 样式 localStorage 按主题分开存，
  // 切回原主题时颜色不会被另一主题污染。
  useEffect(() => {
    const prev = (() => {
      try { return localStorage.getItem('chart-theme') } catch { return null }
    })()
    try { localStorage.setItem('chart-theme', theme) } catch {}
    document.body.style.background = theme === 'light' ? '#ffffff' : '#0e1116'
    document.body.style.color = theme === 'light' ? '#131722' : '#d1d4dc'
    // 初次进入页面（prev === theme）不刷新；用户真的切换主题（prev !== theme）才刷新
    if (prev && prev !== theme) {
      window.location.reload()
      return
    }
    const w = widgetRef.current
    if (w?.changeTheme) {
      void w.changeTheme(theme).then(() => {
        try {
          w.applyOverrides?.({
            'paneProperties.background': theme === 'light' ? '#ffffff' : '#131722',
          })
        } catch {}
      })
    }
  }, [theme])

  useEffect(() => {
    const w = widgetRef.current
    if (!w) return
    try {
      w.setSymbol(
        `${props.exchange}:${props.symbol}`,
        freqToResolution(props.freq),
        () => {
          const chart = w.activeChart?.()
          if (!chart) return
          attachDataLoadedListener(chart)
          attachIntervalListener(chart)
          attachVisibleRangeListener(chart)
          void refreshChanStudy(chart)
        },
      )
    } catch {
      // widget 还没 ready，无视
    }
  }, [props.exchange, props.symbol, props.freq])

  function attachVisibleRangeListener(chart: any) {
    try {
      rangeSubRef.current?.unsubscribe?.()
    } catch {}
    rangeSubRef.current = null
    try {
      // 用户往回拖（visible range 变了但没新数据），onDataLoaded 不会触发；
      // 这里兜底：visible range 变化 → debounce 200ms → 重新拉 chan 覆盖新视区
      const sub = chart.onVisibleRangeChanged().subscribe(null, () => {
        if (rangeDebounceTimer.current) window.clearTimeout(rangeDebounceTimer.current)
        rangeDebounceTimer.current = window.setTimeout(() => {
          void refreshChanStudy(chart)
        }, 200)
      })
      rangeSubRef.current = sub ?? null
    } catch (e) {
      console.warn('onVisibleRangeChanged subscribe failed', e)
    }
  }

  function attachIntervalListener(chart: any) {
    try {
      intervalSubRef.current?.unsubscribe?.()
    } catch {}
    intervalSubRef.current = null
    try {
      const sub = chart.onIntervalChanged().subscribe(null, (resolution: any) => {
        try {
          const f = resolutionToFreq(String(resolution))
          if (f !== propsRef.current.freq) {
            propsRef.current.onFreqChange?.(f)
          }
        } catch {
          // 未知 resolution（如 user 设置奇怪的 tick 周期），忽略
        }
      })
      intervalSubRef.current = sub ?? null
    } catch (e) {
      console.warn('onIntervalChanged subscribe failed', e)
    }
  }

  function attachDataLoadedListener(chart: any) {
    try {
      subRef.current?.unsubscribe?.()
    } catch {}
    subRef.current = null
    try {
      const sub = chart.onDataLoaded().subscribe(null, () => {
        if (debounceTimer.current) window.clearTimeout(debounceTimer.current)
        debounceTimer.current = window.setTimeout(() => {
          void refreshChanStudy(chart)
        }, CHAN_AFTER_DATA_LOADED_MS)
      })
      subRef.current = sub ?? null
    } catch (e) {
      console.warn('onDataLoaded subscribe failed', e)
    }
  }

  async function refreshChanStudy(chart: any) {
    // 串行化：同时只允许一个 refresh 跑。期间所有外部 trigger 合并成一个 pending，
    // 当前 refresh 结束后若 pending=true 再跑一轮（range 没变会被下面的 dedupe 跳过）。
    if (refreshInFlight.current) {
      refreshPending.current = true
      return
    }
    refreshInFlight.current = true
    try {
      do {
        refreshPending.current = false
        await doRefreshOnce(chart)
      } while (refreshPending.current)
    } finally {
      refreshInFlight.current = false
    }
  }

  async function doRefreshOnce(chart: any) {
    ++reloadGen.current
    let range: { from: number; to: number } | null = null
    try {
      range = chart.getVisibleRange()
    } catch {}
    if (!range || !range.from || !range.to) return

    // 关键：symbol/freq 都从 TV widget 实时取，不依赖 React props。
    // 用户可能通过 TV 顶部搜索框换 symbol，或左下角按钮切 freq。
    // 用 symbolExt().ticker（始终 'EXCH:CODE'）而不是 chart.symbol()，后者返回的是
    // resolveSymbol 里设置的 name 字段，现在 name 已经是中文（如「螺纹钢」），split 不出 code。
    let exchange = propsRef.current.exchange
    let symbol = propsRef.current.symbol
    try {
      const tk = String(chart.symbolExt?.()?.ticker ?? '')
      if (tk.includes(':')) {
        const [ex, sy] = tk.split(':')
        exchange = ex
        symbol = sy
      } else if (tk) {
        symbol = tk
      }
    } catch {}
    let freq: Freq
    try {
      const res = String(chart.resolution())
      freq = resolutionToFreq(res)
    } catch {
      freq = propsRef.current.freq
    }
    // 回放模式：把可视区右端截到 cutoffTs，chan 只取游标之前的结构。
    const effectiveTo = isStreamActive()
      ? Math.min(range.to, streamState.cutoffTs)
      : range.to
    // 去重：若 (symbol,freq,range,cutoff) 跟上次成功提交一致，跳过 —— 否则
    // setInputValues 引发的 onDataLoaded 会无限重入 refresh。
    const key = `${exchange}:${symbol}:${freq}:${streamState.cutoffTs}`
    const lc = lastCommitted.current
    if (lc && lc.key === key && lc.from === range.from && lc.to === effectiveTo) return
    const fromIso = unixToIso(range.from)
    const toIso = unixToIso(effectiveTo)
    try {
      // 先拿 chan，再根据"所有结构 [t0,t1] 的覆盖范围"扩大 /bars 请求范围
      // 避免笔/段延伸到可视区外时，bar-index 插值斜率算错（笔/段形变）。
      // 回放/回测走各自的 /{mode}/{id}/chan：chan 仅基于已 push 的 bar 计算，
      // 最新一段未确认时 sure=false 显示虚线。普通模式走 /chan（全 csv 切片，秒回）。
      const snap = await (() => {
        const sid = streamState.sessionId
        if (sid && streamState.mode === 'replay') return api.replayChan(sid, { from: fromIso, to: toIso })
        if (sid && streamState.mode === 'backtest') return api.backtestChan(sid, { from: fromIso, to: toIso })
        return api.chan({ exchange, symbol, freq, from: fromIso, to: toIso })
      })()

      let minT = range.from, maxT = range.to
      const extend = (t0: number, t1: number) => {
        if (t0 < minT) minT = t0
        if (t1 > maxT) maxT = t1
      }
      for (const x of snap.bis) extend(x.t0, x.t1)
      for (const x of snap.segs) extend(x.t0, x.t1)
      for (const x of snap.zs) extend(x.t0, x.t1)
      for (const x of snap.segzs) extend(x.t0, x.t1)

      const barsResp = await api.bars({
        exchange, symbol, freq,
        from: unixToIso(minT), to: unixToIso(maxT),
      })
      // 累积合并：同 (exchange,symbol,freq) 多次 refresh 不再覆盖，让用户拖拽过的
      // 历史区间 BSP 标签持续保留；切 symbol/freq 时按 key 整个替换。
      commitChanLookup(buildChanLookup(snap, barsResp.bars), key)
      lastCommitted.current = { key, from: range.from, to: range.to }
      await ensureAndPokeChanStudy(chart)
    } catch (e) {
      console.error('chan study refresh failed', e)
    }
  }

  // ---------- 回放 handlers ---------- //
  function parseAtToUnix(at: string): number {
    // at 可能是 'YYYY-MM-DD' 或 'YYYY-MM-DD HH:MM:SS'，按北京时间解析
    const cleaned = at.trim()
    const padded = cleaned.includes(' ') ? cleaned : `${cleaned} 00:00:00`
    const d = new Date(padded.replace(' ', 'T') + '+08:00')
    return Math.floor(d.getTime() / 1000)
  }

  async function resolveCurrentSymbolFreq(): Promise<{ exchange: string; symbol: string; freq: Freq } | null> {
    const chart = widgetRef.current?.activeChart?.()
    if (!chart) return null
    let exchange = propsRef.current.exchange
    let symbol = propsRef.current.symbol
    let freq = propsRef.current.freq
    try {
      const tk = String(chart.symbolExt?.()?.ticker ?? '')
      if (tk.includes(':')) {
        const [ex, sy] = tk.split(':')
        exchange = ex
        symbol = sy
      }
    } catch {}
    try { freq = resolutionToFreq(String(chart.resolution())) } catch {}
    return { exchange, symbol, freq }
  }

  // 给定 freq 估算「200 根 bar 大致是多少秒」，用来给 setVisibleRange 算左边界
  function approxSecondsPerNBars(freq: Freq, n: number): number {
    const sec: Record<string, number> = {
      '1m': 60, '3m': 180, '5m': 300, '10m': 600, '15m': 900, '30m': 1800,
      '1h': 3600, '2h': 7200, '4h': 14400, '1d': 86400,
    }
    return (sec[freq] ?? 60) * n
  }

  async function handleReplayStart() {
    if (!replayDate.trim()) return
    const w = widgetRef.current
    const chart = w?.activeChart?.()
    const ctx = await resolveCurrentSymbolFreq()
    if (!w || !chart || !ctx) return
    try {
      const resp = await api.replayStart({
        exchange: ctx.exchange, symbol: ctx.symbol, freq: ctx.freq, at: replayDate,
      })
      const cutoffUnix = parseAtToUnix(replayDate)
      enterReplay(resp.session_id, cutoffUnix)
      setReplayUiActive(true)
      setReplayRemaining(resp.remaining)
      // 同 symbol 强制 TV 重拉数据：先通知 TV 清缓存，再 resetData() 触发 getBars
      // （getBars 里会读 streamState.cutoffTs，自动截断到回放起点之前）。
      streamState.resetCache?.()
      chart.resetData?.()
      lastCommitted.current = null
      // 滚动到 cutoff 附近：左侧 200 根历史作上下文，cutoff 留在右侧约 4/5 位置
      const back = approxSecondsPerNBars(ctx.freq, 200)
      const ahead = approxSecondsPerNBars(ctx.freq, 50)
      try {
        void chart.setVisibleRange?.({ from: cutoffUnix - back, to: cutoffUnix + ahead })
      } catch (e) {
        console.warn('setVisibleRange failed', e)
      }
      void refreshChanStudy(chart)
    } catch (e) {
      console.error('replay start failed', e)
    }
  }

  async function handleReplayStep(n: number) {
    const sid = streamState.sessionId
    if (!sid) return
    try {
      const resp = await api.replayStep(sid, n)
      advanceStream(resp.bars)
      setReplayRemaining(resp.remaining)
      const chart = widgetRef.current?.activeChart?.()
      if (chart) {
        lastCommitted.current = null
        void refreshChanStudy(chart)
      }
      if (resp.remaining === 0) stopAutoPlay()
    } catch (e) {
      console.error('replay step failed', e)
    }
  }

  function startAutoPlay() {
    if (autoPlayTimer.current) return
    setIsAutoPlaying(true)
    autoPlayTimer.current = window.setInterval(() => {
      void handleReplayStep(1)
    }, 500)
  }

  function stopAutoPlay() {
    if (autoPlayTimer.current) {
      window.clearInterval(autoPlayTimer.current)
      autoPlayTimer.current = null
    }
    setIsAutoPlaying(false)
  }

  async function handleReplayStop() {
    stopAutoPlay()
    const sid = streamState.sessionId
    if (sid) { try { await api.replayStop(sid) } catch {} }
    exitStream()
    setReplayUiActive(false)
    setReplayRemaining(0)
    // 切回历史模式：同 symbol 强制重拉数据
    const w = widgetRef.current
    const chart = w?.activeChart?.()
    if (chart) {
      streamState.resetCache?.()
      chart.resetData?.()
      lastCommitted.current = null
      void refreshChanStudy(chart)
    }
  }

  // 卸载时清自动播放定时器
  useEffect(() => () => {
    if (autoPlayTimer.current) window.clearInterval(autoPlayTimer.current)
  }, [])

  // 关闭面板：如果正在回放就停回放，否则只是收起 UI
  async function closeReplayPanel() {
    if (replayUiActive) {
      await handleReplayStop()
    }
    setReplayPanelOpen(false)
  }

  function clearResultTradeShapes() {
    const chart = widgetRef.current?.activeChart?.()
    if (chart) {
      for (const id of resultShapeIds.current) {
        try { chart.removeEntity?.(id) } catch {}
      }
    }
    resultShapeIds.current = []
  }

  function rememberResultShape(id: any) {
    if (id && typeof id.then === 'function') {
      id.then((eid: any) => {
        if (eid != null) resultShapeIds.current.push(eid)
      })
    } else if (id != null) {
      resultShapeIds.current.push(id)
    }
  }

  function drawResultTrade(trade: BacktestResultTrade, highlighted = false) {
    const chart = widgetRef.current?.activeChart?.()
    if (!chart?.createShape) return
    const isLong = trade.side === 'long'
    const win = trade.is_win
    const entryColor = isLong ? '#d32f2f' : '#089981'
    const exitColor = win ? '#2962ff' : '#f59e0b'
    const lineColor = win ? '#089981' : '#ef4444'
    try {
      rememberResultShape(chart.createShape(
        { time: trade.entry_time, price: trade.entry_price },
        {
          shape: isLong ? 'arrow_up' : 'arrow_down',
          text: isLong ? '多' : '空',
          overrides: { color: entryColor, arrowColor: entryColor, textColor: entryColor },
          lock: true, disableSelection: false, disableSave: true, disableUndo: true,
        },
      ))
      rememberResultShape(chart.createShape(
        { time: trade.exit_time, price: trade.exit_price },
        {
          shape: isLong ? 'arrow_down' : 'arrow_up',
          text: win ? '盈' : '亏',
          overrides: { color: exitColor, arrowColor: exitColor, textColor: exitColor },
          lock: true, disableSelection: false, disableSave: true, disableUndo: true,
        },
      ))
      if (chart.createMultipointShape) {
        rememberResultShape(chart.createMultipointShape(
          [
            { time: trade.entry_time, price: trade.entry_price },
            { time: trade.exit_time, price: trade.exit_price },
          ],
          {
            shape: 'trend_line',
            overrides: {
              linecolor: lineColor,
              linewidth: highlighted ? 3 : 1,
              linestyle: highlighted ? 0 : 2,
            },
            lock: true, disableSelection: true, disableSave: true, disableUndo: true,
          },
        ))
      }
    } catch (e) {
      console.warn('draw result trade failed', e)
    }
  }

  function renderResultTrades(trades: BacktestResultTrade[], highlightedId: number | null = null) {
    clearResultTradeShapes()
    for (const trade of trades) drawResultTrade(trade, trade.id === highlightedId)
  }

  function resultTradesForSymbol(trades: BacktestResultTrade[], symbol: string) {
    return trades.filter((t) => t.symbol.toUpperCase() === symbol.toUpperCase())
  }

  function resultTradesInWindow(trades: BacktestResultTrade[], from: number, to: number) {
    return trades.filter((t) => t.exit_time >= from && t.entry_time <= to)
  }

  function resetChartAndChan(chart: any) {
    chanLookupHolder.current = null
    chanLookupHolder.key = null
    lastCommitted.current = null
    streamState.resetCache?.()
    try { chart?.resetData?.() } catch {}
  }

  function applyTradeRange(
    chart: any,
    from: number,
    to: number,
    trades: BacktestResultTrade[],
    highlightedId: number | null,
  ) {
    if (!chart) return
    try { chart.setVisibleRange?.({ from, to }) } catch {}
    renderResultTrades(resultTradesInWindow(trades, from, to), highlightedId)
    lastCommitted.current = null
    void refreshChanStudy(chart)
  }

  function scheduleTradeRange(
    from: number,
    to: number,
    trades: BacktestResultTrade[],
    highlightedId: number | null,
  ) {
    const run = () => {
      const chart = widgetRef.current?.activeChart?.()
      applyTradeRange(chart, from, to, trades, highlightedId)
    }
    run()
    window.setTimeout(run, 250)
    window.setTimeout(run, 800)
  }

  function focusResultTrade(trade: BacktestResultTrade) {
    const w = widgetRef.current
    const chart = w?.activeChart?.()
    if (!w || !chart) return
    setResultPanel((prev) => prev ? { ...prev, selectedId: trade.id } : prev)
    const switchAndFocus = () => {
      const activeChart = w?.activeChart?.()
      if (!activeChart) return
      const pad = Math.max(3600, Math.floor((trade.exit_time - trade.entry_time) * 0.35))
      const from = trade.entry_time - pad
      const to = trade.exit_time + pad
      resetChartAndChan(activeChart)
      const trades = resultPanelRef.current?.trades ?? []
      scheduleTradeRange(from, to, resultTradesForSymbol(trades, trade.symbol), trade.id)
    }
    try {
      const current = String(chart.symbolExt?.()?.ticker ?? '')
      const target = `${propsRef.current.exchange}:${trade.symbol}`
      if (!current.endsWith(`:${trade.symbol}`) && current !== trade.symbol) {
        w.setSymbol(target, freqToResolution(propsRef.current.freq), switchAndFocus)
      } else {
        switchAndFocus()
      }
    } catch {
      switchAndFocus()
    }
  }

  // 按主题计算样式：light 浅灰底深字，dark 深灰底浅字
  const btn: React.CSSProperties = theme === 'light'
    ? { padding: '2px 8px', background: '#f0f3fa', color: '#131722',
        border: '1px solid #e0e3eb', borderRadius: 3, cursor: 'pointer',
        fontSize: 12, height: 22 }
    : { padding: '2px 8px', background: '#2a2e39', color: '#d1d4dc',
        border: '1px solid #363a45', borderRadius: 3, cursor: 'pointer',
        fontSize: 12, height: 22 }
  const panelBg = theme === 'light' ? '#ffffff' : '#1e222d'
  const panelBorder = theme === 'light' ? '#e0e3eb' : '#363a45'
  const inputBg = theme === 'light' ? '#ffffff' : '#131722'
  const textColor = theme === 'light' ? '#131722' : '#d1d4dc'

  // 顶栏回放按钮：无背景、无边框，跟 TV 自带的图标按钮风格一致；
  // 激活时（面板展开 / 回放中）字色高亮成蓝色而不是改背景
  const replayIconColor = replayPanelOpen || replayUiActive
    ? '#2962ff'
    : (theme === 'light' ? '#131722' : '#d1d4dc')
  const headerButton = (
    <button
      onClick={() => setReplayPanelOpen((v) => !v)}
      onMouseEnter={() => setReplayHover(true)}
      onMouseLeave={() => setReplayHover(false)}
      style={{
        padding: '6px 8px',
        background: replayHover ? 'rgba(127,127,127,0.15)' : 'transparent',
        border: 'none', borderRadius: 4,
        cursor: 'pointer', fontSize: 14, lineHeight: 1, height: 34,
        display: 'inline-flex', alignItems: 'center', gap: 4,
        color: replayIconColor,
        transition: 'background 120ms ease',
      }}
      title="回放"
    >
      <svg
        width="16" height="16" viewBox="0 0 24 24" fill="none"
        stroke="currentColor" strokeWidth="2"
        strokeLinecap="round" strokeLinejoin="round"
      >
        <polygon points="19 20 9 12 19 4 19 20" />
        <line x1="5" y1="19" x2="5" y2="5" />
      </svg>
      回放
    </button>
  )

  // 回测按钮：图标 + 文字，类似回放
  const backtestIconColor = backtestPanelOpen
    ? '#2962ff'
    : (theme === 'light' ? '#131722' : '#d1d4dc')
  const backtestButton = (
    <button
      onClick={() => {
        setBacktestPanelOpen((v) => !v)
        setModelPanelOpen(false)
        setDryrunPanelOpen(false)
      }}
      onMouseEnter={() => setBacktestHover(true)}
      onMouseLeave={() => setBacktestHover(false)}
      style={{
        padding: '6px 8px',
        background: backtestHover ? 'rgba(127,127,127,0.15)' : 'transparent',
        border: 'none', borderRadius: 4,
        cursor: 'pointer', fontSize: 14, lineHeight: 1, height: 34,
        display: 'inline-flex', alignItems: 'center', gap: 4,
        color: backtestIconColor,
        transition: 'background 120ms ease',
      }}
      title="回测"
    >
      <svg
        width="16" height="16" viewBox="0 0 24 24" fill="none"
        stroke="currentColor" strokeWidth="2"
        strokeLinecap="round" strokeLinejoin="round"
      >
        <path d="M3 3v18h18" />
        <path d="M7 14l4-4 4 4 5-5" />
      </svg>
      回测
    </button>
  )

  const modelIconColor = modelPanelOpen
    ? '#2962ff'
    : (theme === 'light' ? '#131722' : '#d1d4dc')
  const modelButton = (
    <button
      onClick={() => {
        setModelPanelOpen((v) => !v)
        setBacktestPanelOpen(false)
        setDryrunPanelOpen(false)
      }}
      onMouseEnter={() => setModelHover(true)}
      onMouseLeave={() => setModelHover(false)}
      style={{
        padding: '6px 8px',
        background: modelHover ? 'rgba(127,127,127,0.15)' : 'transparent',
        border: 'none', borderRadius: 4,
        cursor: 'pointer', fontSize: 14, lineHeight: 1, height: 34,
        display: 'inline-flex', alignItems: 'center', gap: 4,
        color: modelIconColor,
        transition: 'background 120ms ease',
      }}
      title="模型"
    >
      <svg
        width="16" height="16" viewBox="0 0 24 24" fill="none"
        stroke="currentColor" strokeWidth="2"
        strokeLinecap="round" strokeLinejoin="round"
      >
        <rect x="4" y="4" width="6" height="6" rx="1" />
        <rect x="14" y="4" width="6" height="6" rx="1" />
        <rect x="4" y="14" width="6" height="6" rx="1" />
        <path d="M14 17h6" />
        <path d="M17 14v6" />
      </svg>
      模型
    </button>
  )

  const dryrunIconColor = dryrunPanelOpen
    ? '#2962ff'
    : (theme === 'light' ? '#131722' : '#d1d4dc')
  const dryrunButton = (
    <button
      onClick={() => {
        setDryrunPanelOpen((v) => !v)
        setBacktestPanelOpen(false)
        setModelPanelOpen(false)
      }}
      onMouseEnter={() => setDryrunHover(true)}
      onMouseLeave={() => setDryrunHover(false)}
      style={{
        padding: '6px 8px',
        background: dryrunHover ? 'rgba(127,127,127,0.15)' : 'transparent',
        border: 'none', borderRadius: 4,
        cursor: 'pointer', fontSize: 14, lineHeight: 1, height: 34,
        display: 'inline-flex', alignItems: 'center', gap: 4,
        color: dryrunIconColor,
        transition: 'background 120ms ease',
      }}
      title="模拟实盘"
    >
      <svg
        width="16" height="16" viewBox="0 0 24 24" fill="none"
        stroke="currentColor" strokeWidth="2"
        strokeLinecap="round" strokeLinejoin="round"
      >
        <path d="M4 19V5" />
        <path d="M4 19h16" />
        <path d="M8 15l3-3 3 2 5-7" />
      </svg>
      实盘
    </button>
  )

  // 回测面板对接图表的「动作」集合
  const chartActions: ChartActions = {
    resetData: () => {
      try { widgetRef.current?.activeChart?.()?.resetData?.() } catch {}
    },
    refreshChan: () => {
      const chart = widgetRef.current?.activeChart?.()
      if (!chart) return
      lastCommitted.current = null
      void refreshChanStudy(chart)
    },
    appendFills: (fills: BacktestFill[]) => {
      const chart = widgetRef.current?.activeChart?.()
      if (!chart || !chart.createShape) return
      for (const f of fills) {
        if (f.reason && f.reason.startsWith('ROLLOVER')) continue
        // 多空 + 开平 4 类组合：
        //   BUY  OPEN  → 红色上箭头放在 K 下方（做多入场）
        //   SELL CLOSE → 红色下箭头放在 K 上方（平多出场）
        //   SELL OPEN  → 绿色下箭头放在 K 上方（做空入场）
        //   BUY  CLOSE → 绿色上箭头放在 K 下方（平空出场）
        const isLongSide = f.side === 'BUY'  // 此 fill 的方向是 BUY 还是 SELL
        const isOpen = f.action === 'OPEN'
        // 开仓在 K 旁边（多在下 / 空在上）；平仓也跟着开仓位置（避免重叠混乱）
        const below = (isLongSide && isOpen) || (!isLongSide && !isOpen)
        const shape = below ? 'arrow_up' : 'arrow_down'
        // 颜色：入场红，出场绿（涨红跌绿的延伸：开仓属于「博取上涨」的红，平仓收益落袋为绿）
        const color = isOpen ? '#d32f2f' : '#089981'
        try {
          const id = chart.createShape(
            { time: f.t, price: f.price },
            {
              shape,
              text: '',
              overrides: { color, arrowColor: color },
              lock: true,
              disableSelection: true,
              disableSave: true,
              disableUndo: true,
            },
          )
          if (id && typeof (id as any).then === 'function') {
            (id as Promise<any>).then((eid) => {
              if (eid != null) backtestFillShapeIds.current.push(eid)
            })
          } else if (id != null) {
            backtestFillShapeIds.current.push(id)
          }
        } catch {
          // 单个箭头画失败不影响其他
        }
      }
    },
    clearFills: () => {
      const chart = widgetRef.current?.activeChart?.()
      if (!chart) return
      for (const id of backtestFillShapeIds.current) {
        try { chart.removeEntity?.(id) } catch {}
      }
      backtestFillShapeIds.current = []
    },
    showResultTrades: ({ resultId, label, trades }) => {
      exitStream()
      setResultPanel({ resultId, label, trades, selectedId: null })
      if (trades.length) {
        const first = trades[0]
        const w = widgetRef.current
        const target = `${propsRef.current.exchange}:${first.symbol}`
        const pad = Math.max(3600, Math.floor((first.exit_time - first.entry_time) * 0.6))
        const from = first.entry_time - pad
        const to = first.exit_time + pad
        const visibleTrades = resultTradesInWindow(resultTradesForSymbol(trades, first.symbol), from, to)
        try {
          w?.setSymbol?.(target, freqToResolution(propsRef.current.freq), () => {
            const chart = w?.activeChart?.()
            resetChartAndChan(chart)
            scheduleTradeRange(from, to, visibleTrades, null)
          })
        } catch {
          const chart = w?.activeChart?.()
          resetChartAndChan(chart)
          scheduleTradeRange(from, to, visibleTrades, null)
        }
      }
    },
    clearResultTrades: () => {
      clearResultTradeShapes()
      setResultPanel(null)
    },
  }

  // 主题切换按钮：light 显示月亮（点击切到 dark），dark 显示太阳。
  // 无背景填充、无边框，跟 TV 顶栏自带的设置/全屏图标风格一致。
  const themeButton = (
    <button
      onClick={() => setTheme(theme === 'light' ? 'dark' : 'light')}
      onMouseEnter={() => setThemeHover(true)}
      onMouseLeave={() => setThemeHover(false)}
      style={{
        padding: '6px 6px',
        background: themeHover ? 'rgba(127,127,127,0.15)' : 'transparent',
        border: 'none', borderRadius: 4,
        cursor: 'pointer', fontSize: 18, lineHeight: 1, height: 34,
        color: theme === 'light' ? '#131722' : '#d1d4dc',
        transition: 'background 120ms ease',
      }}
      title={theme === 'light' ? '切换到深色主题' : '切换到浅色主题'}
    >
      {theme === 'light' ? '🌙' : '🎨'}
    </button>
  )

  const bottomPanel = (
    <div style={{
      position: 'absolute', left: '50%', transform: 'translateX(-50%)',
      bottom: 48,
      display: 'flex', alignItems: 'center', gap: 8,
      padding: '6px 10px',
      background: panelBg, border: `1px solid ${panelBorder}`, borderRadius: 6,
      color: textColor, fontSize: 13, boxShadow: '0 2px 8px rgba(0,0,0,0.15)',
      zIndex: 10,
    }}>
      {!replayUiActive ? (
        <>
          <span>📅</span>
          <input
            type="text"
            value={replayDate}
            onChange={(e) => setReplayDate(e.target.value)}
            placeholder="2024-01-01 或 2024-01-01 09:00:00"
            style={{
              width: 220, padding: '3px 8px', background: inputBg,
              color: textColor, border: `1px solid ${panelBorder}`, borderRadius: 3,
              fontSize: 12, height: 24,
            }}
          />
          <button onClick={handleReplayStart} style={btn}>开始回放</button>
        </>
      ) : (
        <>
          <button onClick={() => handleReplayStep(1)} disabled={isAutoPlaying} style={btn}>⏭ 下一根</button>
          <button onClick={() => handleReplayStep(5)} disabled={isAutoPlaying} style={btn}>下 5 根</button>
          <button onClick={() => handleReplayStep(20)} disabled={isAutoPlaying} style={btn}>下 20 根</button>
          {!isAutoPlaying ? (
            <button onClick={startAutoPlay} style={btn}>▶ 自动</button>
          ) : (
            <button onClick={stopAutoPlay} style={btn}>⏸ 暂停</button>
          )}
          <span style={{ opacity: 0.7, fontSize: 12, marginLeft: 4 }}>剩余 {replayRemaining}</span>
        </>
      )}
      <button
        onClick={closeReplayPanel}
        style={{ ...btn, marginLeft: 8 }}
        title={replayUiActive ? '停止回放并关闭' : '关闭'}
      >
        ✕
      </button>
    </div>
  )

  const fmtTradeTime = (sec: number) => {
    const d = new Date(sec * 1000)
    const pad = (n: number) => String(n).padStart(2, '0')
    return `${pad(d.getMonth() + 1)}-${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`
  }
  const fmtPct = (v: number | null) => {
    if (v == null || !Number.isFinite(v)) return '-'
    return `${v >= 0 ? '+' : ''}${(v * 100).toFixed(2)}%`
  }

  const resultTradeTable = resultPanel ? (
    <div style={{
      height: 220,
      borderTop: `1px solid ${panelBorder}`,
      background: panelBg,
      color: textColor,
      display: 'flex',
      flexDirection: 'column',
      fontSize: 12,
    }}>
      <div style={{
        height: 34,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 10px',
        borderBottom: `1px solid ${panelBorder}`,
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <strong>{resultPanel.label}</strong>
          <span style={{ opacity: 0.7 }}>{resultPanel.trades.length} 笔交易</span>
        </div>
        <button
          onClick={() => chartActions.clearResultTrades()}
          style={{ ...btn, height: 24 }}
        >
          关闭
        </button>
      </div>
      <div style={{ flex: 1, overflow: 'auto' }}>
        <table style={{ width: '100%', borderCollapse: 'collapse', fontVariantNumeric: 'tabular-nums' }}>
          <thead style={{ position: 'sticky', top: 0, background: panelBg, zIndex: 1 }}>
            <tr>
              {['#', '币种', '方向', '入场', '出场', '入场价', '出场价', '收益', '概率', '仓位', '退出原因'].map((h) => (
                <th key={h} style={{
                  textAlign: h === '退出原因' ? 'left' : 'right',
                  padding: '6px 8px',
                  borderBottom: `1px solid ${panelBorder}`,
                  color: theme === 'light' ? '#787b86' : '#9aa0aa',
                  fontWeight: 500,
                  whiteSpace: 'nowrap',
                }}>{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {resultPanel.trades.map((t, i) => {
              const selected = resultPanel.selectedId === t.id
              const ret = t.net_return ?? t.gross_return
              const retColor = (ret ?? 0) >= 0 ? '#d32f2f' : '#089981'
              return (
                <tr
                  key={`${t.symbol}-${t.id}`}
                  onClick={() => focusResultTrade(t)}
                  style={{
                    cursor: 'pointer',
                    background: selected ? 'rgba(41,98,255,0.14)' : 'transparent',
                  }}
                >
                  <td style={tdStyle(panelBorder)}>{i + 1}</td>
                  <td style={tdStyle(panelBorder)}>{t.symbol}</td>
                  <td style={{ ...tdStyle(panelBorder), color: t.side === 'long' ? '#d32f2f' : '#089981' }}>
                    {t.side === 'long' ? '多' : '空'}
                  </td>
                  <td style={tdStyle(panelBorder)}>{fmtTradeTime(t.entry_time)}</td>
                  <td style={tdStyle(panelBorder)}>{fmtTradeTime(t.exit_time)}</td>
                  <td style={tdStyle(panelBorder)}>{t.entry_price.toFixed(4)}</td>
                  <td style={tdStyle(panelBorder)}>{t.exit_price.toFixed(4)}</td>
                  <td style={{ ...tdStyle(panelBorder), color: retColor }}>{fmtPct(ret)}</td>
                  <td style={tdStyle(panelBorder)}>{t.probability == null ? '-' : t.probability.toFixed(3)}</td>
                  <td style={tdStyle(panelBorder)}>{t.stake_multiplier == null ? '-' : t.stake_multiplier.toFixed(2)}</td>
                  <td style={{ ...tdStyle(panelBorder), textAlign: 'left', maxWidth: 260, overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {t.exit_reason || '-'}
                  </td>
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </div>
  ) : null

  return (
    <>
      {headerEl && createPortal(headerButton, headerEl)}
      {backtestBtnEl && createPortal(backtestButton, backtestBtnEl)}
      {modelBtnEl && createPortal(modelButton, modelBtnEl)}
      {dryrunBtnEl && createPortal(dryrunButton, dryrunBtnEl)}
      {themeBtnEl && createPortal(themeButton, themeBtnEl)}
      <div style={{ width: '100%', height: '100%', display: 'flex' }}>
        <div style={{ flex: 1, minWidth: 0, height: '100%', display: 'flex', flexDirection: 'column' }}>
          <div style={{ position: 'relative', flex: 1, minHeight: 0 }}>
            <div ref={containerRef} style={{ width: '100%', height: '100%' }} />
            {replayPanelOpen && bottomPanel}
          </div>
          {resultTradeTable}
        </div>
        <BacktestPanel
          open={backtestPanelOpen}
          onClose={() => setBacktestPanelOpen(false)}
          theme={theme}
          exchange={props.exchange}
          symbol={props.symbol}
          freq={props.freq}
          chartActions={chartActions}
        />
        <ModelPanel
          open={modelPanelOpen}
          onClose={() => setModelPanelOpen(false)}
          theme={theme}
        />
        <DryRunPanel
          open={dryrunPanelOpen}
          onClose={() => setDryrunPanelOpen(false)}
          theme={theme}
        />
      </div>
    </>
  )
}

// 关键设计：chan study 创建一次后就一直保留，不再 remove+create。
//
// 之前用 remove+create 重建 study 的方式，用户在指标设置对话框里改的颜色/线宽全丢，
// 又试过 getStyleValues + applyOverrides 二次贴回，结果 TV 的 StudyPropertiesOverrider
// 不认自定义指标的 plot id（控制台报 'has no plot or input bi' 等），样式仍被重置。
//
// 现改用 setInputValues 触发 TV 重跑 main()：chanIndicator 加了个隐藏的 epoch input，
// 每次 chanLookupHolder 更新后 bump epoch，TV 会按新 input 重新对所有 bar 调 main()，
// main() 读最新的 chanLookupHolder.current 拿到新值。study 对象本身不动，样式自然保留。
let chanEpoch = 0

function tdStyle(borderColor: string): React.CSSProperties {
  return {
    textAlign: 'right',
    padding: '5px 8px',
    borderBottom: `1px solid ${borderColor}`,
    whiteSpace: 'nowrap',
  }
}

async function ensureAndPokeChanStudy(chart: any) {
  try {
    const studies: any[] = chart.getAllStudies?.() ?? []
    let existingId: any = null
    for (const s of studies) {
      if (s.name === CHAN_STUDY_NAME) {
        existingId = s.id
        break
      }
    }
    chanEpoch += 1
    if (existingId) {
      try {
        const api = chart.getStudyById?.(existingId)
        api?.setInputValues?.([{ id: 'epoch', value: chanEpoch }])
      } catch (err) {
        console.warn('setInputValues Chan failed', err)
      }
      return
    }
    const id = await chart.createStudy(CHAN_STUDY_NAME, false, false, { epoch: chanEpoch })
    if (!id) {
      console.warn('createStudy Chan returned no id（指标未注册成功）')
      return
    }
    // 把整个 chan study 提到 K 线上方，避免笔/段/中枢被 candle 遮住。
    try {
      chart.getStudyById?.(id)?.bringToFront?.()
    } catch (err) {
      console.warn('bringToFront Chan failed', err)
    }
  } catch (e) {
    console.error('ensureAndPokeChanStudy 抛错', e)
  }
}

function unixToIso(unix: number): string {
  const d = new Date(unix * 1000)
  const pad = (n: number) => String(n).padStart(2, '0')
  return (
    `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ` +
    `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`
  )
}
