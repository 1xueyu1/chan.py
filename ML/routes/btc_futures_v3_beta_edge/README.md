# BTC 合约 v3 Beta Edge 路线

## 当前定位

这条路线现在只保留最小可解释主线：

```text
BTC 1m 合约数据
-> 15m 缠论二类买卖点
-> 一个买卖点只生成一条可执行样本
-> 标签只使用 label_tp_first
-> 买入/卖出方向分别训练
-> 阈值层只决定是否放行
-> 决策层负责仓位、杠杆和最终入场
```

旧版的样本展开层已经从主链路移除，不再把一个信号展开成 `0/5/15/30/60` 多个候选入场。这样训练样本数、回测候选数和实盘信号数可以保持同一口径。

## 数据和标签

基础事件来自：

```text
data\btc_futures_v3_alpha\btc_futures_v3_alpha_dataset.parquet
```

当前 beta 数据集输出：

```text
data\btc_futures_v3_beta_edge\btc_futures_v3_beta_edge_dataset.parquet
```

每个缠论买卖点只保留一条样本：

- `exec_time`：15m 信号 K 线时间。
- `entry_time`：信号 K 线收完后，第一根可交易 1m K 线时间。
- `exit_time`：固定 1:1 止盈止损或最大持仓后的退出时间。
- `label_tp_first`：入场后是否先碰到止盈。

当前标签层只保留：

```text
label_tp_first
```

含义很直接：在 1:1 盈亏比、同口径手续费和滑点下，这个买卖点入场后，是不是先到止盈。

## 模型层

每个方向单独训练一个模型：

- `buy_model.pkl`：只处理二类买点做多。
- `sell_model.pkl`：只处理二类卖点做空。

模型目标只有一个：

```text
P(label_tp_first = 1)
```

也就是模型输出的 `probability` 可以理解为“这个二类买卖点先到止盈的估计概率”。旧版的 candidate/timing/tradeability 三层融合不再作为当前主链路使用。

## 阈值层

阈值层不是训练模型，它是模型训练完成后的放行规则。

流程是：

1. 在训练集和验证集训练模型。
2. 在验证年和确认年上扫描阈值，例如 `0.45, 0.46, ..., 0.95`。
3. 每个阈值都会统计：
   - 放行多少笔交易。
   - 胜率是否达到要求。
   - 平均净收益是否大于 0。
   - Profit Factor 是否达到要求。
   - 交易频率是否过低。
4. 只有验证年和确认年同时通过的阈值，才会写入 `threshold_policy.json`。
5. 如果没有阈值通过，就把该方向阈值设为 `0.99`，等价于禁用该方向。

所以阈值的含义是：

```text
概率 >= 阈值，才允许进入决策层。
概率 < 阈值，只记录信号，不交易。
```

阈值文件：

```text
result\ml\btc_futures_v3_beta_edge_bsp2_only\threshold_policy.json
```

关键字段：

- `default_thresholds.buy`：默认做多阈值。
- `default_thresholds.sell`：默认做空阈值。
- `symbol_thresholds.BTCUSDT`：BTC 单币种覆盖阈值。
- `validation`：生成该阈值时使用的训练、验证、确认切分参数。

## 决策层

决策层不再重新判断标签，它只使用模型分数和阈值做执行控制。

离线回测中的顺序：

```text
读取样本
-> 按方向调用 buy/sell 模型
-> 得到 probability
-> 读取对应 threshold
-> probability >= threshold 的信号进入 qualified
-> 同一时间只允许一笔持仓
-> 按 fixed 1:1 出场结果计算收益
```

Freqtrade 实盘/模拟盘中的顺序：

```text
15m 新 K 线完成
-> 提取二类买卖点
-> 构造当前时刻可用特征
-> 调用模型得到 probability
-> 使用 threshold_policy 或模型内阈值
-> 趋势/成交量等执行过滤
-> 根据信心层决定仓位和杠杆
-> 下单
```

当前信心层用于仓位和杠杆：

- `high`：高置信度，使用较大仓位和杠杆。
- `mid`：中等置信度，使用较小仓位和杠杆。
- `low`：低置信度，只做很小仓位观察。
- `none`：不交易。

这层的作用不是提高模型概率，而是控制风险暴露。真正决定“这个信号值不值得做”的，仍然是模型概率和阈值。

## 常用命令

重建数据集：

```powershell
python -m ML.routes.btc_futures_v3_beta_edge.dataset --force
```

训练二类买卖点模型：

```powershell
python -m ML.routes.btc_futures_v3_beta_edge.train_bsp_split
```

回测：

```powershell
python -m ML.routes.btc_futures_v3_beta_edge.backtest `
  --model-dir result\ml\btc_futures_v3_beta_edge_bsp2_only `
  --output-dir result\btc_futures_v3_beta_edge_bsp2_only\fixed_exit `
  --begin-time 2026-01-01 `
  --bsp-families 2
```

## 当前需要重点观察

- `candidate_rows` 是否接近真实二类信号数，而不是信号数乘以多个 delay。
- `qualified_candidates` 是否由阈值决定，而不是样本展开决定。
- `win_rate` 是否和 `label_tp_first` 的定义一致。
- Freqtrade 触发的信号时间是否和离线 `entry_time` 口径一致。
