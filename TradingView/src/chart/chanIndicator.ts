// TradingView 自定义指标：缠论·笔 / 段 / 中枢 / 段中枢 / 笔BSP / 段BSP。
//
// 所有结构都在指标里：
// - 笔/段（实+虚 4 plot）：line + line_with_breaks，按 bar 索引线性插值连成直线
// - 中枢 / 段中枢：上下沿 line plot + filled_area 填中间
// - BSP：chars plot 在 K 上下方显示文字标签（'B'/'S'/'段B'/'段S'）
//
// chan 数据从 chanLookupHolder 注入，上层切 symbol/freq 后换 holder.current 并重建 study。

import type { Bsp, ChanSlice } from '../api/types'

// 完全按 chan_core 原版 BSP 子类型打标：T1 / T1P / T2 / T2S / T3A / T3B
// 一个 BSP 的 types 可能同时包含多个子类（如 T1+T2），此时全部打上对应标签（重叠也接受）。
export const BSP_SUBTYPES = ['T1', 'T1P', 'T2', 'T2S', 'T3A', 'T3B'] as const
export type BspSubtype = typeof BSP_SUBTYPES[number]

type BspPlotKey =
  | `bi_bsp_buy_${BspSubtype}` | `bi_bsp_sell_${BspSubtype}`
  | `seg_bsp_buy_${BspSubtype}` | `seg_bsp_sell_${BspSubtype}`

// BarEntry：8 个线/中枢字段 + 24 个 BSP plot 字段（6 子类 × 买卖 × 笔/段）。
// 每个字段是 number?，非 undefined 表示该 bar 在该 plot 上有标签 / 数值。
type BarEntry = {
  bi?: number; bi_pending?: number
  seg?: number; seg_pending?: number
  zs_top?: number; zs_bottom?: number
  zs_pending_top?: number; zs_pending_bottom?: number
  segzs_top?: number; segzs_bottom?: number
} & Partial<Record<BspPlotKey, number>>

export interface ChanLookup {
  byTimeMs: Map<number, BarEntry>
}

// chanLookupHolder.key 标识当前 lookup 属于哪个 (exchange,symbol,freq) 组合。
// 同 key 时多次 refresh 累积合并（用户拖拽到新区域 → 旧区 BSP 标签仍保留）；
// key 不同时（切 symbol/freq）整个清空。
export const chanLookupHolder: {
  current: ChanLookup | null
  key: string | null
} = { current: null, key: null }

/** 把 fresh lookup 累积/替换进 chanLookupHolder。同 key 累积，不同 key 替换。 */
export function commitChanLookup(fresh: ChanLookup, key: string): void {
  if (chanLookupHolder.key !== key || !chanLookupHolder.current) {
    chanLookupHolder.current = fresh
    chanLookupHolder.key = key
    return
  }
  // 同 key：合并 fresh 的 BarEntry 到现有 byTimeMs，让历史区间的 BSP 标签持续保留。
  // 同一 bar 在两次 fetch 中都有数据时，新值覆盖旧值（chan 边界可能微调）。
  const target = chanLookupHolder.current.byTimeMs
  for (const [t, e] of fresh.byTimeMs) {
    const existed = target.get(t)
    if (existed) Object.assign(existed, e)
    else target.set(t, e)
  }
}

function lowerBound(arr: number[], v: number): number {
  let lo = 0, hi = arr.length
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (arr[mid] < v) lo = mid + 1
    else hi = mid
  }
  return lo
}

function upperBound(arr: number[], v: number): number {
  let lo = 0, hi = arr.length
  while (lo < hi) {
    const mid = (lo + hi) >> 1
    if (arr[mid] <= v) lo = mid + 1
    else hi = mid
  }
  return lo
}

export function buildChanLookup(
  slice: ChanSlice,
  bars: { t: number; h: number; l: number }[],
): ChanLookup {
  const byTimeMs = new Map<number, BarEntry>()
  const ensure = (sec: number): BarEntry => {
    const key = sec * 1000
    let e = byTimeMs.get(key)
    if (!e) {
      e = {}
      byTimeMs.set(key, e)
    }
    return e
  }

  const sortedBarTimes = bars.map((b) => b.t).sort((a, b) => a - b)

  // 笔 / 段：按 bar 索引线性插值
  const fillLine = (
    t0: number, t1: number, p0: number, p1: number,
    field: 'bi' | 'bi_pending' | 'seg' | 'seg_pending',
  ) => {
    const lo = lowerBound(sortedBarTimes, t0)
    const hi = upperBound(sortedBarTimes, t1)
    const span = hi - 1 - lo
    if (span <= 0) {
      if (lo < sortedBarTimes.length) ensure(sortedBarTimes[lo])[field] = p0
      return
    }
    const step = (p1 - p0) / span
    for (let i = lo; i < hi; i++) {
      ensure(sortedBarTimes[i])[field] = p0 + step * (i - lo)
    }
  }

  for (const b of slice.bis) {
    fillLine(b.t0, b.t1, b.p0, b.p1, b.sure ? 'bi' : 'bi_pending')
  }
  for (const s of slice.segs) {
    fillLine(s.t0, s.t1, s.p0, s.p1, s.sure ? 'seg' : 'seg_pending')
  }

  // 中枢 / 段中枢：[t0, t1] 全段 K 都写 top/bottom；filled_area 自动填出矩形
  const fillZs = (
    t0: number, t1: number, high: number, low: number,
    topField: 'zs_top' | 'zs_pending_top' | 'segzs_top',
    botField: 'zs_bottom' | 'zs_pending_bottom' | 'segzs_bottom',
  ) => {
    const lo = lowerBound(sortedBarTimes, t0)
    const hi = upperBound(sortedBarTimes, t1)
    for (let i = lo; i < hi; i++) {
      const e = ensure(sortedBarTimes[i])
      e[topField] = high
      e[botField] = low
    }
  }
  // 笔中枢按 is_sure 路由：确认的走实色填充 (zs_top/bottom + zs_fill)，
  // 未确认的走虚线上下沿 (zs_pending_top/bottom，dashed line plot，没有 fill；
  // 左右垂直边 TV line plot 画不了，所以只有上下两条水平虚线)
  // 虚中枢的右边界向后延伸到当前最新一根 K，表示中枢仍在形成中
  const lastBarTime = sortedBarTimes[sortedBarTimes.length - 1] ?? 0
  for (const z of slice.zs) {
    if (z.is_sure) {
      fillZs(z.t0, z.t1, z.high, z.low, 'zs_top', 'zs_bottom')
    } else {
      const t1Extended = Math.max(z.t1, lastBarTime)
      fillZs(z.t0, t1Extended, z.high, z.low, 'zs_pending_top', 'zs_pending_bottom')
    }
  }
  for (const z of slice.segzs) fillZs(z.t0, z.t1, z.high, z.low, 'segzs_top', 'segzs_bottom')

  // bar 时间 → high/low 索引，给 BSP 标签算 Y 坐标用
  const barByTime = new Map<number, { h: number; l: number }>()
  for (const b of bars) barByTime.set(b.t, { h: b.h, l: b.l })

  // BSP：完全按原版子类型 T1/T1P/T2/T2S/T3A/T3B 打标，多 type 全部展示。
  // 标签 Y 坐标：配合 plot location: 'Absolute'，plot value 就是实际 Y。
  // 买/卖偏移系数不对称 —— 因为 TV chars 在 Absolute 下 char 体是从 Y 锚点「向上」延伸的：
  //   卖点 Y 在 K 上方 → char 朝外延伸，offSell=0 也能贴上沿；
  //   买点 Y 在 K 下方 → char 反而朝 K 延伸，必须给 offBuy 留出 char 自身高度，
  //   否则 char 主体会盖在 K 体上。
  // 同侧笔+段还要错开：段比笔再外推 0.002 价格比例。
  const placeBsps = (list: Bsp[], prefix: 'bi_bsp' | 'seg_bsp') => {
    const offSell = prefix === 'seg_bsp' ? 0.002 : 0.000
    const offBuy = prefix === 'seg_bsp' ? 0.004 : 0.002
    for (const bsp of list) {
      const bar = barByTime.get(bsp.t)
      if (!bar) continue
      const dir = bsp.is_buy ? 'buy' : 'sell'
      const y = dir === 'buy'
        ? bar.l * (1 - offBuy)    // 买点放 K 下方（大偏移，避开 char 自身高度）
        : bar.h * (1 + offSell)   // 卖点放 K 上方（小偏移即可）
      const e = ensure(bsp.t)
      for (const raw of bsp.types ?? []) {
        const t = String(raw).toUpperCase()
        if (!(BSP_SUBTYPES as readonly string[]).includes(t)) continue
        const key = `${prefix}_${dir}_${t as BspSubtype}` as BspPlotKey
        e[key] = y
      }
    }
  }
  placeBsps(slice.bsps, 'bi_bsp')
  placeBsps(slice.seg_bsps ?? [], 'seg_bsp')

  return { byTimeMs }
}

// 持久化：用户在指标设置对话框改的颜色保存到 localStorage，下次 makeChanIndicator
// 时把保存的值合并进 metainfo.defaults，新建 chan study 直接拿到用户的颜色。
// 按主题分 key（light/dark 互不影响）。不做颜色传播 —— 保存什么读什么。
function currentTheme(): 'light' | 'dark' {
  try { return (localStorage.getItem('chart-theme') as 'light' | 'dark') ?? 'light' }
  catch { return 'light' }
}

function chanStyleKey(): string {
  return `chan-indicator-styles-v3-${currentTheme()}`
}

function loadSavedChanStyles(): {
  styles?: Record<string, any>
  filledAreasStyle?: Record<string, any>
} | null {
  try {
    const raw = localStorage.getItem(chanStyleKey())
    return raw ? JSON.parse(raw) : null
  } catch {
    return null
  }
}

// 24 个 BSP plot id —— 按方向分桶，方便传播
const BUY_BSP_IDS = BSP_SUBTYPES.flatMap((t) => [
  `bi_bsp_buy_${t}`, `seg_bsp_buy_${t}`,
])
const SELL_BSP_IDS = BSP_SUBTYPES.flatMap((t) => [
  `bi_bsp_sell_${t}`, `seg_bsp_sell_${t}`,
])

/** TradingViewChart 在 study_properties_changed 时调用，把 getStyleValues() 写入 localStorage。
 *
 *  传播逻辑：设置对话框只暴露「买点」（绑 bi_bsp_buy_T1）和「卖点」（绑 bi_bsp_sell_T1）2 项，
 *  用户改的是这俩的 color。这里把它俩的颜色复制到所有 12 个同方向 plot（含段、含其他子类），
 *  text 颜色同步。下次刷新 makeChanIndicator 读 localStorage 时，24 个 BSP 用户都看到一致颜色。 */
export function saveChanStyles(sv: any): void {
  try {
    const styles: Record<string, any> = { ...(sv?.styles ?? {}) }
    const buyColor = styles.bi_bsp_buy_T1?.color
    const sellColor = styles.bi_bsp_sell_T1?.color
    if (buyColor) {
      for (const id of BUY_BSP_IDS) {
        styles[id] = { ...(styles[id] ?? {}), color: buyColor, textColor: buyColor }
      }
    }
    if (sellColor) {
      for (const id of SELL_BSP_IDS) {
        styles[id] = { ...(styles[id] ?? {}), color: sellColor, textColor: sellColor }
      }
    }
    localStorage.setItem(chanStyleKey(), JSON.stringify({
      styles,
      filledAreasStyle: sv?.filledAreasStyle ?? {},
    }))
  } catch {}
}

export function makeChanIndicator(): any {
  const saved = loadSavedChanStyles()
  const isDark = currentTheme() === 'dark'
  // 按主题选笔/段基础色：暗色主题用白笔亮蓝段，亮色主题用黑笔深蓝段
  const biColor = isDark ? '#ffffff' : '#000000'
  const segColor = isDark ? '#42a5f5' : '#1565c0'
  // 笔虚中枢上下沿：暗色主题白虚线，亮色主题黑虚线
  const zsPendingColor = isDark ? '#ffffff' : '#000000'
  // 把 localStorage 里保存的字段叠在 base 之上 —— 用户改过的字段覆盖默认值
  const mergeStyle = (id: string, base: Record<string, any>) => ({
    ...base,
    ...(saved?.styles?.[id] ?? {}),
  })
  const mergeFill = (id: string, base: Record<string, any>) => ({
    ...base,
    ...(saved?.filledAreasStyle?.[id] ?? {}),
  })
  return {
    name: '缠论',
    metainfo: {
      _metainfoVersion: 53,
      id: 'Chan@tv-basicstudies-1',
      scriptIdPart: '',
      description: 'Chan 缠论',
      shortDescription: '缠论',
      is_hidden_study: false,
      isCustomIndicator: true,
      is_price_study: true,
      linkedToSeries: true,
      format: { type: 'inherit' },
      plots: [
        { id: 'bi', type: 'line' },
        { id: 'bi_pending', type: 'line' },
        { id: 'seg', type: 'line' },
        { id: 'seg_pending', type: 'line' },
        { id: 'zs_top', type: 'line' },
        { id: 'zs_bottom', type: 'line' },
        // 笔中枢（虚）：is_sure=false 的笔中枢用 dashed line plot 画水平上下沿，没有 fill
        { id: 'zs_pending_top', type: 'line' },
        { id: 'zs_pending_bottom', type: 'line' },
        { id: 'segzs_top', type: 'line' },
        { id: 'segzs_bottom', type: 'line' },
        // 24 个 BSP chars plot：6 子类 × 买/卖 × 笔/段。
        // 多 type 的 BSP 会同时在多个 plot 上打标（视觉上可能重叠，少见情况可接受）。
        ...BSP_SUBTYPES.flatMap((t) => [
          { id: `bi_bsp_buy_${t}`, type: 'chars' },
          { id: `bi_bsp_sell_${t}`, type: 'chars' },
          { id: `seg_bsp_buy_${t}`, type: 'chars' },
          { id: `seg_bsp_sell_${t}`, type: 'chars' },
        ]),
      ],
      filledAreas: [
        { id: 'zs_fill', objAId: 'zs_top', objBId: 'zs_bottom', type: 'plot_plot',
          title: '中枢', isHidden: false },
        { id: 'segzs_fill', objAId: 'segzs_top', objBId: 'segzs_bottom', type: 'plot_plot',
          title: '段中枢', isHidden: false },
      ],
      defaults: {
        // 每个 style 都过一遍 mergeStyle/mergeFill —— 把 localStorage 里上次保存的同名字段
        // 覆盖到内置默认值上，让用户编辑过的颜色/线宽跨浏览器刷新保留。
        // 颜色按白色主题选：笔黑、段蓝、BSP 深色，保证在白底可见。
        styles: {
          bi: mergeStyle('bi', { linestyle: 0, linewidth: 1, plottype: 0, trackPrice: false,
                transparency: 0, visible: true, color: biColor, display: 3 }),
          bi_pending: mergeStyle('bi_pending', { linestyle: 2, linewidth: 1, plottype: 0, trackPrice: false,
                        transparency: 0, visible: true, color: biColor, display: 3 }),
          seg: mergeStyle('seg', { linestyle: 0, linewidth: 3, plottype: 0, trackPrice: false,
                 transparency: 0, visible: true, color: segColor, display: 3 }),
          seg_pending: mergeStyle('seg_pending', { linestyle: 2, linewidth: 3, plottype: 0, trackPrice: false,
                         transparency: 0, visible: true, color: segColor, display: 3 }),
          // 中枢/段中枢上下沿：本身不画线，只作为 filled_area 的边界
          zs_top: mergeStyle('zs_top', { linestyle: 0, linewidth: 0, plottype: 0, trackPrice: false,
                    transparency: 100, visible: false, color: '#e4eaf1', display: 0 }),
          zs_bottom: mergeStyle('zs_bottom', { linestyle: 0, linewidth: 0, plottype: 0, trackPrice: false,
                       transparency: 100, visible: false, color: '#1565c0', display: 0 }),
          segzs_top: mergeStyle('segzs_top', { linestyle: 0, linewidth: 0, plottype: 0, trackPrice: false,
                       transparency: 100, visible: false, color: '#ef6c00', display: 0 }),
          segzs_bottom: mergeStyle('segzs_bottom', { linestyle: 0, linewidth: 0, plottype: 0, trackPrice: false,
                          transparency: 100, visible: false, color: '#ef6c00', display: 0 }),
          // 笔中枢（虚）上下沿：未确认的笔中枢用 dashed line 画虚线（linestyle: 2）
          // 颜色跟主题：暗色用白线、亮色用黑线（zsPendingColor）
          // transparency: 0（= 不透明度 100%）用 spread 强制覆盖 localStorage 里旧的值，
          // 否则之前保存过的半透明设置会污染。
          zs_pending_top: { ...mergeStyle('zs_pending_top', { linestyle: 2, linewidth: 1, plottype: 0, trackPrice: false,
                              visible: true, color: zsPendingColor, display: 3 }), transparency: 0 },
          zs_pending_bottom: { ...mergeStyle('zs_pending_bottom', { linestyle: 2, linewidth: 1, plottype: 0, trackPrice: false,
                                 visible: true, color: zsPendingColor, display: 3 }), transparency: 0 },
          // BSP chars plot：12 个 plot 对应 12 种文案。char 给空格占位（TV 仍认 chars plot
          // 结构），实际标签由 styles.<plotId>.text 提供「T1/T1P/T2/T2S/T3A/T3B...」多字符文案。
          // 24 个 BSP plot 默认：买红色 BelowBar，卖绿色 AboveBar。颜色由 char 默认 + styles
          // text 提供。每个子类独立 plot，可在设置对话框单独调颜色。
          // 位置策略：全部用 location: 'Absolute' —— plot value 就是 Y 坐标。
          // 笔买/卖紧贴 K 上下沿，段买/卖在同侧再外推一段，避免「同根 K 上笔+段同方向」时重叠。
          // 实际 Y 在 placeBsps 里按 bar.h/bar.l ± 偏移系数计算。
          // location 用 spread 强制写在 mergeStyle 之后，覆盖 localStorage 里旧的 location；
          // 其他字段（color/size/text）仍尊重 saved 里的用户改动。
          ...Object.fromEntries(BSP_SUBTYPES.flatMap((t) => [
            [`bi_bsp_buy_${t}`, { ...mergeStyle(`bi_bsp_buy_${t}`,
              { char: ' ', visible: true, size: 'large',
                color: '#d32f2f', display: 3 }), location: 'Absolute' }],
            [`bi_bsp_sell_${t}`, { ...mergeStyle(`bi_bsp_sell_${t}`,
              { char: ' ', visible: true, size: 'large',
                color: '#2e7d32', display: 3 }), location: 'Absolute' }],
            [`seg_bsp_buy_${t}`, { ...mergeStyle(`seg_bsp_buy_${t}`,
              { char: ' ', visible: true, size: 'large',
                color: '#d32f2f', display: 3 }), location: 'Absolute' }],
            [`seg_bsp_sell_${t}`, { ...mergeStyle(`seg_bsp_sell_${t}`,
              { char: ' ', visible: true, size: 'large',
                color: '#2e7d32', display: 3 }), location: 'Absolute' }],
          ])),
        },
        filledAreasStyle: {
          zs_fill: mergeFill('zs_fill', { color: '#f1d96a', visible: true, transparency: 75 }),
          segzs_fill: mergeFill('segzs_fill', { color: '#6361f7', visible: true, transparency: 75 }),
        },
        precision: 2,
        // epoch 是个隐藏的整数 input：上层在 chanLookupHolder 更新后通过 setInputValues
        // 改 epoch，逼 TV 重新跑 main() 重算所有 bar。这样 study 对象不需要被 remove+create
        // 重建，用户改的颜色/线宽等样式自然保留。
        inputs: { epoch: 0 },
      },
      styles: {
        bi: { title: '笔', histogramBase: 0 },
        bi_pending: { title: '笔（虚）', histogramBase: 0 },
        seg: { title: '段', histogramBase: 0 },
        seg_pending: { title: '段（虚）', histogramBase: 0 },
        // 上下沿在设置对话框里也隐藏（仅作为 filled_area 边界，用户无需配置）
        zs_top: { title: '中枢上沿', histogramBase: 0, isHidden: true },
        zs_bottom: { title: '中枢下沿', histogramBase: 0, isHidden: true },
        zs_pending_top: { title: '中枢（虚）上沿', histogramBase: 0, isHidden: true },
        zs_pending_bottom: { title: '中枢（虚）下沿', histogramBase: 0, isHidden: true },
        segzs_top: { title: '段中枢上沿', histogramBase: 0, isHidden: true },
        segzs_bottom: { title: '段中枢下沿', histogramBase: 0, isHidden: true },
        // 24 个 BSP plot 的标签：text 字段提供多字符渲染文案。子类型保留 chan_core 原版
        // 后缀（1/1P/2/2S/3A/3B），但前缀用 B（买）/ S（卖）替换 T 来区分方向；段加「段」前缀。
        // 设置对话框只暴露 2 项 —— 「买点」绑到 bi_bsp_buy_T1、「卖点」绑到 bi_bsp_sell_T1；
        // 其他 22 个 plot 全部 isHidden 隐藏。改色后由 saveChanStyles 传播到所有 buy/sell
        // 同方向 plot，刷新后所有标签同步换色。
        ...Object.fromEntries(BSP_SUBTYPES.flatMap((t) => {
          const suffix = t.slice(1)  // T1 → '1'，T1P → '1P'，T2S → '2S' 等
          const buyText = `B${suffix}`
          const sellText = `S${suffix}`
          const isFirst = t === 'T1'  // 用 T1 作为颜色源头暴露给设置对话框
          return [
            [`bi_bsp_buy_${t}`, {
              title: isFirst ? '买点' : `B${suffix} 买点 (内部)`,
              isHidden: !isFirst, text: buyText,
            }],
            [`bi_bsp_sell_${t}`, {
              title: isFirst ? '卖点' : `S${suffix} 卖点 (内部)`,
              isHidden: !isFirst, text: sellText,
            }],
            [`seg_bsp_buy_${t}`, { title: `段 ${buyText} 买点 (内部)`, isHidden: true, text: `段${buyText}` }],
            [`seg_bsp_sell_${t}`, { title: `段 ${sellText} 卖点 (内部)`, isHidden: true, text: `段${sellText}` }],
          ]
        })),
      },
      inputs: [
        { id: 'epoch', name: 'epoch', type: 'integer', defval: 0, isHidden: true },
      ],
    },
    constructor: function (this: any) {
      this.init = function (this: any, context: any) {
        this._context = context
      }
      this.main = function (this: any, context: any) {
        this._context = context
        // 10 个线/中枢 plot + 24 个 BSP chars plot（6 子类 × 买/卖 × 笔/段） = 34
        // 顺序必须跟 plots 数组完全一致：先线/中枢 10 个，然后按 BSP_SUBTYPES 顺序展开
        // 每个 subtype 的 [bi_buy, bi_sell, seg_buy, seg_sell] 共 4 个 plot。
        const NANS = new Array(34).fill(NaN)
        const lookup = chanLookupHolder.current
        if (!lookup) return NANS
        const t = context.symbol.time
        const e = lookup.byTimeMs.get(t)
        if (!e) return NANS
        const out: number[] = [
          e.bi ?? NaN, e.bi_pending ?? NaN,
          e.seg ?? NaN, e.seg_pending ?? NaN,
          e.zs_top ?? NaN, e.zs_bottom ?? NaN,
          e.zs_pending_top ?? NaN, e.zs_pending_bottom ?? NaN,
          e.segzs_top ?? NaN, e.segzs_bottom ?? NaN,
        ]
        for (const sub of BSP_SUBTYPES) {
          out.push(
            (e as any)[`bi_bsp_buy_${sub}`] ?? NaN,
            (e as any)[`bi_bsp_sell_${sub}`] ?? NaN,
            (e as any)[`seg_bsp_buy_${sub}`] ?? NaN,
            (e as any)[`seg_bsp_sell_${sub}`] ?? NaN,
          )
        }
        return out
      }
    },
  }
}
