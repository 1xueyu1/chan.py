# 模型登记表

当前项目只保留二类买卖点相关路线。旧实验路线已经从代码、脚本、数据目录和当前登记中移除。

完整盘点见 [MODEL_INVENTORY.md](MODEL_INVENTORY.md)。

## 当前正式状态

| 路线 | 类型 | 信号范围 | 主标签 | 出场/决策 | 状态 |
|---|---|---|---|---|---|
| `btc_futures_v2_bsp2_family` | 当前正式主线 | 15m 二类买卖点族群 | `label_bsp2_family_valid` | `family_realtime` + 二级决策模型 + 结构风险定仓 | 正式基准：`decision_realtime_v2` |
| `btc_futures_v1` | 保留基准 | 标准二买/二卖 | `label_bsp2_chan_entry_quality` | 第三段确认后等待反向买卖点，含结构保护 | 历史基准，仅用于对照 |

## 当前基准

```text
基准版本：decision_realtime_v2
模型路线：btc_futures_v2_bsp2_family
结果目录：result/btc_futures_v2_bsp2_family_wf_decision_realtime_v2
模型目录：result/ml/btc_futures_v2_bsp2_family_wf_decision_realtime_v2
```

最新 walk-forward：

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

## 当前保留能力

| 能力 | 位置 | 说明 |
|---|---|---|
| 二类买卖点族群候选 | `ML/routes/btc_futures_v2_bsp2_family/candidate_family.py` | 当前主线候选生成 |
| 二类买卖点标签 | `ML/routes/btc_futures_v2_bsp2_family/labels.py` | 当前主线训练标签 |
| 实时出场状态机 | `ML/routes/btc_futures_v2_bsp2_family/backtest.py` | 当前主线回测和决策口径 |
| 结构特征增强 | `ML/shared/structure_features.py` | 当前主线复用特征 |
| 仓后管理能力 | `ML/shared/post_exit_management.py` | 当前主线复用仓后管理 |

## 后续规则

1. 新实验必须围绕二类买卖点路线展开。
2. 新实验必须先写 `.claude/experiments/*.md`。
3. 实验成功后才允许进入本登记表。
4. 失败实验删除代码和临时结果，只在 `ML/EXPERIMENT_LOG.md` 留结论。
5. 当前所有新实验默认对照 `decision_realtime_v2`。
