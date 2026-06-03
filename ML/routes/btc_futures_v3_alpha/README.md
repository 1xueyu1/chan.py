# BTC 合约 v3 Alpha 路线

## 目标

`btc_futures_v3_alpha` 是在 `btc_futures_v2_stable` 基础上的新路线，不覆盖 v1/v2。它解决的问题不是“放松阈值多交易”，而是先把 BTC 缠论买卖点拆成更细的结构池，再观察哪些结构池在严格验证下真的有优势。

核心目标：

- 保持修复后的同口径标签和回测链路。
- 保持 1:1 止盈止损思想，默认沿用 v2 的 ATR 自适应 1:1 标签。
- 强化缠论结构特征：中枢边界、区间套、背驰嵌套、多周期共振/冲突、笔/线段/中枢成熟度。
- 训练买/卖全局模型，同时尝试结构池分模型。
- 每个结构池必须在验证集上通过成本线，才允许进入回测或模拟实盘。

## 数据来源

默认读取：

```text
data\btc_futures_v2_stable\btc_futures_v2_stable_dataset.parquet
```

如果 v2 数据不存在，构建脚本会尝试先生成 v2 数据。v2 的底层 1m BTC 合约数据仍来自：

```text
D:\WorkSpace\czsc_all\data\freqtrade_futures\futures\BTC_USDT_USDT-1m-futures.parquet
```

v3 输出：

```text
data\btc_futures_v3_alpha\btc_futures_v3_alpha_dataset.parquet
result\ml\btc_futures_v3_alpha\
result\btc_futures_v3_alpha\
```

## 新增结构特征

主要新增 `v3_` 前缀特征：

- `v3_htf_consensus_score`：多周期结构方向共振。
- `v3_htf_conflict_score`：多周期方向冲突。
- `v3_boundary_score`：信号是否靠近大级别中枢有效边界。
- `v3_maturity_score`：笔、线段、中枢的综合成熟度。
- `v3_style_*`：趋势延续、震荡反抽、中枢离开、嵌套背驰。
- `v3_alpha_structure_score`：结构质量综合分。
- `v3_structure_pool`：结构池标签。
- `v3_pool_is_*`：结构池 one-hot 特征。
- `v3_market_*` 和 `v3_session_*`：市场状态和交易时段辅助特征。

这些特征只使用信号当时已经可见的 15m、1h、4h、1d 缠论结构和 K 线状态，不读取 `label`、`net_return`、`exit_time` 等未来结果。

## 结构池

当前固定结构池：

- `nested_boundary`：小级别背驰嵌套在大级别中枢边界。
- `rebound_boundary`：中枢边界反抽。
- `rebound_mid`：中枢内部反抽，但边界优势不强。
- `zs_departure_resonant`：中枢离开段且多周期共振。
- `zs_departure_weak`：中枢离开段但结构共振不足。
- `trend_continuation_resonant`：趋势延续且多周期一致。
- `trend_continuation_mixed`：趋势延续但周期状态混合。
- `boundary_probe`：靠近边界但风格不够明确。
- `direction_conflict`：多周期方向冲突。
- `mixed_low_edge`：结构信息不足或优势不明显。

## 训练逻辑

训练分三层：

1. 买/卖方向分别训练全局模型。
2. 每个结构池尝试训练分池模型。
3. 在验证集上比较全局模型和分池模型，谁通过成本线并且更好，就使用谁。

验证约束默认：

- 默认验证区间：`2024-01-01` 到 `2026-01-01`，也就是 2024-2025 双年验证。
- 最低验证胜率：`0.60`
- 最低 Profit Factor：`1.20`
- 买/卖全局模型最低验证 AUC：`0.54`
- 结构池分模型最低验证 AUC：`0.54`
- 平均净收益必须大于 0
- 结构池样本太少、单类标签、或验证不稳定时禁用该池

禁用的结构池不会在回测或模拟实盘中交易。

## 常用命令

构建数据：

```powershell
Scripts\btc_futures_v3_build_dataset.ps1
```

结构池诊断：

```powershell
Scripts\btc_futures_v3_diagnostics.ps1
```

训练：

```powershell
Scripts\btc_futures_v3_train.ps1
```

回测：

```powershell
Scripts\btc_futures_v3_backtest.ps1
```

年度 walk-forward：

```powershell
Scripts\btc_futures_v3_walk_forward.ps1
```

一键跑数据、诊断、训练和普通回测：

```powershell
Scripts\btc_futures_v3_run_all.ps1
```

## Freqtrade 策略入口

```text
user_data\strategies\ChanMLBtcFuturesV3AlphaStrategy.py
```

它继承当前 Rust 缠论信号链路，并使用：

```text
result\ml\btc_futures_v3_alpha\buy_model.pkl
result\ml\btc_futures_v3_alpha\sell_model.pkl
result\ml\btc_futures_v3_alpha\threshold_policy.json
```

在线推理时，v3 模型包会自动补齐 `v3_` 派生特征，避免离线训练和模拟实盘特征不一致。

## 当前严格版结果

本轮默认严格版已经完成一次完整链路验证：

- 数据集：14207 条 BTC 缠论买卖点事件。
- 普通 2026 回测：候选信号 1069 条，合格信号 0 条，实际交易 0 笔。
- 年度 walk-forward：2024、2025、2026 三个测试年全部 0 笔交易。
- 买入模型验证 AUC：约 0.534，低于默认最低 `0.54`。
- 卖出模型验证 AUC：约 0.530，低于默认最低 `0.54`。

这不是执行失败，而是严格过滤后的结论：当前这批结构特征还没有证明具备稳定排序能力。早期不加 AUC 门槛时，验证集里能挑出少数结构池，但 2026 回测会亏损；因此现在默认策略会禁用这些“阈值碰巧通过但排序能力不足”的池。

## 无未来函数约束

- 缠论信号来自 15m K 线，必须在该 K 线收盘后才可用。
- 标签和回测入场从 `exec_time + 15min` 之后的 1m open 开始。
- 数据构建、训练、回测都会检查：

```text
entry_time >= exec_time + 15min
```
