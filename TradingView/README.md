# Chan TradingView 页面

这是本地 TradingView Charting Library 展示页，默认展示：

```text
BINANCE:BTCUSDT / 15m
```

本地接口：

```text
TradingView\server.py
```

它当前负责：

- 从 `D:\WorkSpace\czsc_all\data\freqtrade_futures\futures\BTC_USDT_USDT-1m-futures.parquet` 读取 BTC 合约 1m 数据。
- 按 TV 周期请求聚合为 1m、3m、5m、15m、30m、1h、2h、4h、1d。
- 暴露当前二类买卖点结构模型的回测结果，支持在页面右侧“回测”面板中选择策略并逐步播放成交点。
- 暴露两个 freqtrade dry-run bot 的只读交易状态，支持在页面右侧“实盘”面板中统一查看。

当前接入的回测结果：

| 策略名 | 数据文件 |
|---|---|
| `二买/二卖结构模型：不定仓` | `result\btc_futures_v1_wf_validlabel_no_after_timeout_fullsize\test_2026\executed_trades.csv` |
| `二买/二卖结构模型：结构风险定仓` | `result\btc_futures_v1_wf_validlabel_tier_sizing_a_boost\test_2026\executed_trades.csv` |

当前接入的模拟实盘：

| 策略名 | 数据库 | API |
|---|---|---|
| `不定仓` | `user_data\chan_v1_fullsize.dryrun.sqlite` | `127.0.0.1:8091` |
| `结构风险定仓` | `user_data\chan_v1_structure_sizing.dryrun.sqlite` | `127.0.0.1:8092` |

启动后端：

```powershell
cd D:\WorkSpace\python\chan.py\TradingView
python -m uvicorn server:app --host 127.0.0.1 --port 3000
```

启动前端：

```powershell
cd D:\WorkSpace\python\chan.py\TradingView
npm run dev -- --host 127.0.0.1
```

访问：

```text
http://127.0.0.1:8080
```

说明：

- `/api/*` 由 Vite 代理到 `127.0.0.1:3000`。
- 当前 `chan` 接口先返回空结构，页面会正常显示 K 线、回测成交点和模拟实盘交易状态；后续再接真实笔、线段、中枢、买卖点结构。
