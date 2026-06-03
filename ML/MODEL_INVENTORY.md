# 项目模型全量盘点

更新时间：2026-06-04

本文件用于整理当前项目中存在过或仍保留代码的数据集、模型路线和回测结果。它不是推荐使用清单；推荐使用清单以 [MODEL_REGISTRY.md](MODEL_REGISTRY.md) 为准。

## 总览

| 路线 | 代码目录 | 数据集 | 模型产物 | 当前状态 | 说明 |
|---|---|---|---|---|---|
| `btc_futures_v1` | 有 | 有 | 有历史 walk-forward 产物 | 保留基准 | 标准二类买卖点基准路线，交易少但对照价值高 |
| `btc_futures_v2_bsp2_family` | 有 | 有 | 有正式基准产物 | 当前主线 | `decision_realtime_v2` 是当前正式基准 |
| `btc_futures_v2_stable` | 有 | 有 | 未发现正式模型产物目录 | 历史实验 | 稳定性/严格验证路线，交易频率过低 |
| `btc_futures_v3_alpha` | 有 | 有 | 未发现正式模型产物目录 | 历史实验 | 结构池/稳定特征路线，严格过滤后无交易 |
| `btc_futures_v3_beta_edge` | 有 | 有 | 未发现正式模型产物目录 | 历史实验 | 旧 1:1 `label_tp_first` 和阈值路线，不作为当前主线 |

## 1. `btc_futures_v1`

### 定位

历史基准路线，主要处理标准二类买卖点。当前不作为主线继续开发，但仍用于对照“扩展二类买卖点族群”是否真的带来增量。

### 数据情况

```text
data/btc_futures_v1/btc_futures_v1_dataset.parquet
data/btc_futures_v1/btc_futures_v1_dataset_meta.json
```

元信息：

```text
rows: 43516
symbols: 11
primary_label: label_bsp2_chan_entry_quality
lookahead_check: exec_time is signal availability time; entry_time >= exec_time
```

另外仍存在一些历史中间数据：

```text
btc_futures_v1_l3_hold_snapshots_meta.json    rows: 1174008
btc_futures_v1_post_lifecycle_meta.json       rows: 66591
btc_futures_v1_post_lifecycle_v1_meta.json    rows: 66591
```

这些属于早期仓后管理/生命周期实验产物，不作为当前正式模型依据。

### 主要结果

历史结果目录：

```text
result/btc_futures_v1_wf_validlabel_no_after_timeout_fullsize
result/btc_futures_v1_wf_validlabel_tier_sizing_a_boost
result/ml/btc_futures_v1_wf_validlabel_tier_sizing_a_boost
```

`tier_sizing_a_boost` walk-forward 摘要：

| 年份 | 交易数 | 胜率 | PF | 最大回撤 |
|---:|---:|---:|---:|---:|
| 2023 | 164 | 27.44% | 2.262 | -7.69% |
| 2024 | 209 | 42.58% | 3.952 | -8.21% |
| 2025 | 138 | 39.13% | 3.889 | -4.14% |
| 2026 | 58 | 39.66% | 4.226 | -2.97% |

评价：收益质量不错，但交易频率太低，不满足当前“提高交易机会覆盖率”的方向。

## 2. `btc_futures_v2_bsp2_family`

### 定位

当前正式主线。它从标准二类买卖点扩展为二类买卖点族群，目标是在保留频率的情况下用二级决策模型和结构风险定仓处理低质量信号。

### 数据情况

```text
data/btc_futures_v2_bsp2_family/btc_futures_v2_bsp2_family_dataset.parquet
data/btc_futures_v2_bsp2_family/btc_futures_v2_bsp2_family_dataset_meta.json
```

元信息：

```text
rows: 43516
symbols: 11
primary_label: label_bsp2_family_valid
lookahead_check: exec_time is signal availability time; entry_time >= exec_time
```

### 当前正式基准

```text
版本：decision_realtime_v2
结果目录：result/btc_futures_v2_bsp2_family_wf_decision_realtime_v2
模型目录：result/ml/btc_futures_v2_bsp2_family_wf_decision_realtime_v2
```

walk-forward：

| 年份 | 交易数 | 胜率 | 收益 | PF | 最大回撤 |
|---:|---:|---:|---:|---:|---:|
| 2023 | 2041 | 30.92% | 13.50% | 1.126 | -8.84% |
| 2024 | 2651 | 33.23% | 72.75% | 1.343 | -9.00% |
| 2025 | 2974 | 31.54% | 38.67% | 1.212 | -12.13% |
| 2026 | 1071 | 31.84% | 20.60% | 1.433 | -8.34% |

汇总：

```text
总交易数：8737
加权胜率：31.94%
年度复合收益：227.91%
最差年度收益：13.50%
最差年度回撤：-12.13%
```

### 历史结果目录

以下目录是历史对照或调试结果，不作为当前正式基准：

```text
result/btc_futures_v2_bsp2_family
result/btc_futures_v2_bsp2_family_no_ml_filter
result/btc_futures_v2_bsp2_family_realtime_2023_riskcap
result/btc_futures_v2_bsp2_family_realtime_smoke
result/btc_futures_v2_bsp2_family_symbol_scope
result/btc_futures_v2_bsp2_family_symbol_scope_no_ml_filter
result/btc_futures_v2_bsp2_family_symbol_scope_no_ml_filter_risk_sizing
result/btc_futures_v2_bsp2_family_wf
result/btc_futures_v2_bsp2_family_wf_decision
result/btc_futures_v2_bsp2_family_wf_realtime
result/btc_futures_v2_bsp2_family_wf_realtime_riskcap
```

评价：这些结果可以用于复盘演进，但后续实验统一以 `wf_decision_realtime_v2` 为基准。

## 3. `btc_futures_v2_stable`

### 定位

稳定性路线，目标是通过 ATR 自适应 1:1 标签、稳定特征筛选、成本感知阈值和市场状态过滤，换取更稳定的验证表现。

### 数据情况

```text
data/btc_futures_v2_stable/btc_futures_v2_stable_dataset.parquet
data/btc_futures_v2_stable/btc_futures_v2_stable_dataset_meta.json
```

元信息：

```text
rows: 14207
label_rate: 0.4891
target_mode: atr
max_holding_minutes: 1440
lookahead_check: entry_time >= exec_time + 15min
```

### 当前状态

代码和数据保留，但未发现 `result/ml/btc_futures_v2_stable*` 的正式模型产物目录。

路线结论：严格过滤后交易频率极低，不符合当前项目“二类买卖点族群 + 高频覆盖 + 结构风险定仓”的主线方向。

## 4. `btc_futures_v3_alpha`

### 定位

结构池和稳定特征增强路线。目标是把缠论买卖点拆成更细结构池，再观察哪些结构池在严格验证下有优势。

### 数据情况

```text
data/btc_futures_v3_alpha/btc_futures_v3_alpha_dataset.parquet
data/btc_futures_v3_alpha/btc_futures_v3_alpha_dataset_meta.json
```

元信息：

```text
rows: 14207
label_rate: 0.4891
lookahead_check: entry_time >= exec_time + 15min
```

### 当前状态

代码和数据保留，但未发现 `result/ml/btc_futures_v3_alpha*` 的正式模型产物目录。

路线结论：严格版曾出现候选信号多但合格信号为 0 的情况，不作为当前主线。

## 5. `btc_futures_v3_beta_edge`

### 定位

旧的 1:1 `label_tp_first` 和阈值路线。它把每个买卖点压成一条样本，买入/卖出方向分别训练，模型目标是：

```text
P(label_tp_first = 1)
```

### 数据情况

```text
data/btc_futures_v3_beta_edge/btc_futures_v3_beta_edge_dataset.parquet
data/btc_futures_v3_beta_edge/btc_futures_v3_beta_edge_dataset_meta.json
```

元信息：

```text
rows: 14207
label_tp_first_rate: 0.4891
lookahead_check: one sample per BSP event; entry_time >= exec_time + 15min; beta features use pre-entry 1m bars only
```

另有仓后生命周期数据：

```text
data/btc_futures_v3_beta_edge/btc_futures_v3_beta_exit_lifecycle_meta.json
rows: 29205
```

### 当前状态

代码和数据保留，但未发现当前命名下的正式模型产物目录。该路线属于旧 1:1 固定标签思路，不作为当前主线。

## 当前建议

### 保留

- `btc_futures_v2_bsp2_family`：当前主线。
- `btc_futures_v1`：历史基准和对照。

### 暂不继续

- `btc_futures_v2_stable`
- `btc_futures_v3_alpha`
- `btc_futures_v3_beta_edge`

这些路线可以保留代码作为参考，但后续不要继续在它们上扩散新实验，除非先写新的实验 spec 并明确为什么要重启。

### 可清理对象

如果后续要进一步瘦身，可以清理以下类别：

1. `result/*smoke*`
2. 非 `wf_decision_realtime_v2` 的 v2 临时回测结果
3. `v2_stable/v3_alpha/v3_beta_edge` 的数据集和脚本

但当前本次只做盘点，不删除。
