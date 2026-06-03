# BTC v3 Beta Edge 严格出场模型

本文档记录从聊天记录中提炼出的“仓后管理模型 / 严出模型”的当前实现。

## 目标

入场模型只负责找到缠论候选交易。出场模型负责在持仓后每根 bar 重新判断：

```text
继续持有，还是提前退出。
```

第一版优先处理聊天记录中提到的 `init stop` 和“严出”问题：

```text
允许小亏小赢，但尽量避免继续持有明显变差的单。
```

## 当前实现文件

```text
ML\routes\btc_futures_v3_beta_edge\exit_management.py
Scripts\btc_futures_v3_beta_strict_exit.ps1
```

输出文件：

```text
data\btc_futures_v3_beta_edge\btc_futures_v3_beta_exit_lifecycle.parquet
result\ml\btc_futures_v3_beta_edge\strict_exit_model.pkl
result\ml\btc_futures_v3_beta_edge\strict_exit_model_metrics.json
result\btc_futures_v3_beta_edge\ml_strict_exit_winrate_055\
```

## 生命周期样本

样本不是“一笔交易一行”，而是：

```text
一笔交易入场后，每 5 分钟生成一条持仓快照。
```

每条快照只使用：

```text
entry_time -> snapshot_time
```

之间已经发生的 1m K 线数据，不使用快照之后的数据作为特征。

主要特征包括：

- 当前浮盈浮亏。
- 当前 MFE / MAE。
- 距离止盈 / 止损还有多远。
- 从最大浮盈回撤了多少。
- 最近 1/3/5/15 根 1m 的顺向收益。
- 持仓后的顺向 bar 比例和逆向 bar 比例。
- 近期波动、量能比、趋势效率。
- 距离 MFE / MAE 出现已经过去多久。
- init stop 风险分数。
- profit lock 分数。

## 标签定义

第一版没有简单把所有最终亏损单都标记为“应该退出”，因为这样会过度严出。

当前标签是：

```text
当前退出收益 明显优于 继续拿到原固定 TP/SL 出场收益
```

并且主要聚焦：

- 原本会亏损的交易。
- 已经有一定浮盈但出现明显回撤的交易。
- 当前退出能够减少亏损或保护利润的交易。

## 当前训练结果

训练窗口来自：

```text
result\btc_futures_v3_beta_edge\frequency_floor_2024_2025_scan\executed_trades.csv
```

生成结果：

```text
生命周期样本：29205 行
覆盖交易：786 笔
```

模型验证结果：

| 数据段 | AUC | AP | Brier |
|---|---:|---:|---:|
| 训练 | 0.966 | 0.959 | 0.089 |
| 验证 | 0.662 | 0.544 | 0.240 |

验证集有一定排序能力，但不强，因此当前只能作为实验模型。

## 2026 回测结果

基准是频率底仓版 2026 交易：

```text
result\btc_futures_v3_beta_edge\frequency_floor_2day\executed_trades.csv
```

固定 TP/SL 重放：

| 指标 | 数值 |
|---|---:|
| 交易数 | 78 |
| 胜率 | 47.44% |
| 收益 | 0.19% |
| PF | 1.06 |
| 最大回撤 | -1.58% |

严出模型，目标偏胜率，参数：

```text
exit_threshold = 0.40
min_model_exit_return = 0.0
```

结果：

| 指标 | 数值 |
|---|---:|
| 交易数 | 78 |
| ML 提前退出 | 32 |
| 胜率 | 57.69% |
| 收益 | -1.11% |
| PF | 0.54 |
| 最大回撤 | -1.55% |

结论：

```text
严格出场模型可以把胜率推到 55% 以上，
但当前版本是用更早的小赢/微赢退出换来的，收益和 PF 变差。
```

这说明模型框架有效，但如果目标是“胜率 + 收益”同时提升，不能只做严出，还需要增加 trail / 大浮盈追踪模型，否则会把本来可能继续扩大的盈利过早截断。

## 常用命令

完整构建、训练、回测：

```powershell
Scripts\btc_futures_v3_beta_strict_exit.ps1 -Action all -Force
```

只跑 55% 胜率目标回测：

```powershell
Scripts\btc_futures_v3_beta_strict_exit.ps1 `
  -Action backtest `
  -OutputDir result\btc_futures_v3_beta_edge\ml_strict_exit_winrate_055 `
  -ExitThreshold 0.40 `
  -MinModelExitReturn 0.0 `
  -SnapshotMinutes 5
```

更保守、偏收益/PF 的回测：

```powershell
Scripts\btc_futures_v3_beta_strict_exit.ps1 `
  -Action backtest `
  -OutputDir result\btc_futures_v3_beta_edge\ml_strict_exit_conservative `
  -ExitThreshold 0.70 `
  -MinModelExitReturn 0.0 `
  -SnapshotMinutes 5
```

## 下一步

## 多头组合版

根据 2026-05-30 聊天记录，仓后管理已经扩展为多头组合模型：

```text
init_stop_head
hold_quality_head
mae_risk_head
left_mfe_peak_head
trail_continue_head
strict_exit_head
```

实现文件仍为：

```text
ML\routes\btc_futures_v3_beta_edge\exit_management.py
```

模型文件：

```text
result\ml\btc_futures_v3_beta_edge\multi_head_exit_model.pkl
result\ml\btc_futures_v3_beta_edge\multi_head_exit_model_metrics.json
```

组合规则不是单个模型直接决定退出，而是：

```text
init_stop 高 AND hold_quality 低 -> 退出
mae_risk 高 AND hold_quality 低 -> 退出
left_mfe_peak 高 AND trail_continue 低 -> 退出
strict_exit 高 AND hold_quality 低 AND trail_continue 低 -> 退出
否则继续持有或交给固定 TP/SL
```

当前支持两种多头执行策略：

| 策略 | 含义 |
|---|---|
| `balanced` | 默认策略，按多头模型组合判断，胜率优先但不过度加入手工利润保护 |
| `profit_protect` | 更强调 MFE 后回撤保护，同时允许 init_bad 小亏止血 |

2026 频率底仓交易结果：

| 模式 | 交易数 | 胜率 | 收益 | PF | ML 提前退出 |
|---|---:|---:|---:|---:|---:|
| 固定 TP/SL | 78 | 47.44% | 0.19% | 1.06 | 0 |
| 单头严出 5m | 78 | 57.69% | -1.11% | 0.54 | 32 |
| 多头组合 5m balanced | 78 | 55.13% | -1.02% | 0.58 | 26 |
| 多头组合 5m profit_protect | 78 | 55.13% | -1.11% | 0.57 | 26 |

多头组合相比单头严出，少退出 6 笔，收益和 PF 略好，但仍然没有达到“胜率、收益、PF 同时改善”。当前默认建议使用 `balanced` 作为后续迭代基线。

## 交易分群诊断

已新增倒推式交易分群诊断：

```powershell
Scripts\btc_futures_v3_beta_strict_exit.ps1 `
  -Action diagnose `
  -OutputDir result\btc_futures_v3_beta_edge\ml_multi_head_exit_5m
```

多头回测：

```powershell
Scripts\btc_futures_v3_beta_strict_exit.ps1 `
  -Action backtest-multi `
  -OutputDir result\btc_futures_v3_beta_edge\ml_multi_head_exit_5m_balanced `
  -MultiPolicy balanced `
  -SnapshotMinutes 5
```

利润保护策略回测：

```powershell
Scripts\btc_futures_v3_beta_strict_exit.ps1 `
  -Action backtest-multi `
  -OutputDir result\btc_futures_v3_beta_edge\ml_multi_head_exit_5m_profit_protect `
  -MultiPolicy profit_protect `
  -SnapshotMinutes 5
```

输出：

```text
result\btc_futures_v3_beta_edge\ml_multi_head_exit_5m\trade_groups\
```

2026 交易分群结果：

| 类型 | 笔数 | 胜率 | 总收益贡献 | 解释 |
|---|---:|---:|---:|---|
| `good_then_fade` | 25 | 0.00% | -20.84% | 有较大 MFE，但最终回撤为亏损 |
| `init_bad` | 14 | 0.00% | -13.72% | 入场后很快变坏 |
| `small_loss` | 2 | 0.00% | -2.10% | 小样本亏损 |
| `clean_trend` | 15 | 100.00% | 12.79% | 干净趋势盈利 |
| `noisy_survivor` | 22 | 100.00% | 15.31% | 有回撤但最终盈利 |

这个诊断说明：当前最大问题不是没有 MFE，而是 `good_then_fade` 太多。也就是说，很多单其实曾经给过利润，但没有在回撤前保护住。

## 下一步

当前多头模型已经比单头严出更符合聊天记录里的“组合拳”思路，但还缺少一个真正有效的收益保护层。

下一步应该拆成三个模型：

```text
init_stop_model：处理进场初期明显坏单。
strict_exit_model：处理持仓中明显转弱的单。
trail_model：处理大 MFE 后继续持有还是落袋。
```

从交易分群看，优先级应该调整为：

```text
第一优先级：good_then_fade 的 profit protection / left peak 模型。
第二优先级：init_bad 的提前退出。
第三优先级：clean_trend/noisy_survivor 的误杀保护。
```

最终目标不是单纯把胜率做高，而是：

```text
胜率 >= 55%
PF > 1
收益为正
最大回撤不扩大
```
