// TradingView Charting Library 通过 <script src=".../charting_library.js"> 注入到 window。
// 这里给 TS 一个 ambient 声明，让 window.TradingView 能用。无 export → 自动被识别为全局脚本。

interface Window {
  TradingView: any
}
