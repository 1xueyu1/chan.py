# btc_futures_v2_bsp2_family：二类买卖点族群路线

这条路线是在 `btc_futures_v1` 基础上继续迭代的正式分支。当前已回到 `decision_realtime_v2` 口径：保留已经验证有效的第三段确认后 15m `opposite_bsp` 出场路径，只用二级决策模型和结构风险定仓压制确认前失效风险。

完整工作回顾见 [WORK_REVIEW.md](WORK_REVIEW.md)。

## 当前口径

- 入场信号来自 15m 缠论二类买卖点族群。
- 入场时间使用信号可用时间，不额外推迟 15m。
- 默认不使用旧概率硬过滤，保留合格候选。
- 默认使用 `family_realtime` 实时状态机出场，不读取未来走势标签。
- 使用二级决策模型输出结构身份概率、确认前失效概率、后续路径概率。
- 使用结构风险定仓，而不是直接删除低质量信号。
- 已删除 `taxonomy_v1` 和 `hybrid_v1` 的分类/出场改造。

## 候选族群

当前候选来自 `candidate_family.py`：

```text
standard_bsp2                  标准二买/二卖
subclass_bsp2s                 类二买/类二卖
bsp2_t3_overlap                二类与三类买卖点重合或近似重合
bsp2_after_bsp1                一类买卖点后的二类确认形态
bsp2_near_origin_zs_boundary   原中枢边界附近的二类确认
```

## 标签设计

主标签：

```text
label_bsp2_family_valid
```

辅助标签：

```text
label_bsp2_identity_strict
label_pre_confirm_invalid
label_bsp2_realtime_path
label_bsp2_realtime_path_id
```

实时路径：

```text
invalid_before_confirm
return_origin_zs
third_confirmed
higher_level_expansion
bsp2_t3_overlap_trend
weak_no_confirm
```

这些标签用于训练二级决策模型，但正式回测出场不直接读取未来标签。

## 实时出场状态机

`family_realtime` 的处理顺序：

1. 如果信号本身是 `2,3a`、`2,3b` 或 `bsp2_t3_overlap`，则视为入场时第三段已经确认。
2. 普通二类族群先观察是否触及结构失效位。
3. 如果先回到原中枢目标位，则按回中枢目标出场，默认目标为 T2。
4. 如果先走出第三段确认，则移动保护止损，之后等待 15m 同级别任意反向买卖点出场。
5. 如果限定时间内既没有确认也没有回中枢，则按弱确认超时出场。

## 结构风险定仓

结构风险定仓位于 `ML/shared/structure_position_sizing.py`。

主要输入：

```text
bsp2_identity_prob
bsp2_pre_invalid_prob
bsp2_path_prob_invalid_before_confirm
bsp2_path_prob_weak_no_confirm
structure_risk_pct
多周期冲突、回撤比例、分型支撑距离、中枢边界对齐等缠论特征
```

当前原则：

- A 类信号可以使用较大仓位，但要求失效概率低、身份概率高、多周期冲突低、结构失效距离可控。
- B 类信号使用中等仓位。
- C 类信号保留交易，但压低仓位。
- 如果二级模型明确提示确认前失效或弱确认，则仓位上限低于普通最小仓位。

## 当前正式 Walk-Forward

输出目录：

```text
result/btc_futures_v2_bsp2_family_wf_decision_realtime_v2
```

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

## 主要结论

- 当前主线保留第三段确认后的 15m `opposite_bsp` 高收益路径。
- 最大拖累仍然是 `realtime_invalid_before_confirm`。
- 后续优化应只针对确认前失效和弱确认亏损，不再改动第三段确认后的盈利出场路径。

## 常用命令

刷新标签：

```powershell
python -m ML.routes.btc_futures_v2_bsp2_family.features --refresh-labels-only --force --workers 4
```

年度 walk-forward：

```powershell
python -m ML.routes.btc_futures_v2_bsp2_family.walk_forward --output-root result/btc_futures_v2_bsp2_family_wf_decision_realtime_v2 --model-root result/ml/btc_futures_v2_bsp2_family_wf_decision_realtime_v2
```
