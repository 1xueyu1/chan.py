import { TradingViewChart } from './chart/TradingViewChart'
import './App.css'

export default function App() {
  return (
    <div className="app">
      <TradingViewChart
        exchange="BINANCE"
        symbol="BTCUSDT"
        freq="15m"
      />
    </div>
  )
}
