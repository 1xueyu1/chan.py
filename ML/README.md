# 机器学习路线

当前 ML 目录采用“基准版本 + 实验治理”的管理方式。正式主线是：

```text
当前基准：decision_realtime_v2
当前路线：btc_futures_v2_bsp2_family
核心信号：15m 二类买卖点族群
```

## 关键文档

| 文档 | 用途 |
|---|---|
| `ML/MODEL_REGISTRY.md` | 当前保留模型登记 |
| `ML/BASELINE.md` | `decision_realtime_v2` 基准指标 |
| `ML/EXPERIMENT_LOG.md` | 正式实验记录 |
| `.claude/spec.md` | 项目级 spec 和主线约束 |
| `.claude/review_checklist.md` | 每次实验和代码审查清单 |
| `docs/backtest_rules.md` | 回测口径说明 |
| `docs/chan_theory_mapping.md` | 缠论理论到代码映射 |

## 当前正式路线

```text
ML/routes/btc_futures_v2_bsp2_family/
```

核心原则：

- 保留二类买卖点族群候选，提高交易频率。
- 保留第三段确认后的 15m `opposite_bsp` 出场路径。
- 使用二级决策模型和结构风险定仓压制低质量信号。
- 后续优化优先针对 `realtime_invalid_before_confirm` 和 `weak_no_confirm`。
- 不再保留 `taxonomy_v1`、`hybrid_v1` 等失败或过时实验路线。

## 常用命令

刷新标签：

```powershell
python -m ML.routes.btc_futures_v2_bsp2_family.features --refresh-labels-only --force --workers 4
```

年度 walk-forward：

```powershell
python -m ML.routes.btc_futures_v2_bsp2_family.walk_forward --output-root result/btc_futures_v2_bsp2_family_wf_decision_realtime_v2 --model-root result/ml/btc_futures_v2_bsp2_family_wf_decision_realtime_v2
```

编译检查：

```powershell
python -m compileall .\ML\routes\btc_futures_v2_bsp2_family .\ML\shared
```

## 新实验流程

每个新实验先写 spec，再改代码：

1. 在 `.claude/experiments/` 创建实验文档。
2. 明确基准、目标、约束、允许改动范围。
3. 实现最小改动。
4. 跑编译检查和 walk-forward。
5. 与 `ML/BASELINE.md` 对照。
6. 更新 `ML/EXPERIMENT_LOG.md`。
7. 成功则保留并提交，失败则删除实验代码和临时结果。
