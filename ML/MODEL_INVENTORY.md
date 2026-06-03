# 项目模型全量盘点

更新时间：2026-06-04

当前项目只保留二类买卖点相关路线。旧实验路线已经移除，后续研究集中在二类买卖点结构识别、结构风险定仓和实时出场状态机上。

## 总览

| 路线 | 代码目录 | 数据集 | 模型产物 | 当前状态 | 说明 |
|---|---|---|---|---|---|
| `btc_futures_v1` | 有 | 有 | 有历史 walk-forward 产物 | 保留基准 | 标准二类买卖点基准路线，交易少但对照价值高 |
| `btc_futures_v2_bsp2_family` | 有 | 有 | 有正式基准产物 | 当前主线 | `decision_realtime_v2` 是当前正式基准 |

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

以下目录是二类买卖点族群路线内部的历史对照或调试结果，不作为当前正式基准：

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

后续实验统一以 `wf_decision_realtime_v2` 为基准。

## 当前共享模块

旧实验路线删除后，仍被当前二类买卖点路线使用的通用能力迁移到了 shared：

```text
ML/shared/structure_features.py
ML/shared/post_exit_management.py
```

这两个模块只服务当前二类买卖点路线，不代表独立模型路线。
