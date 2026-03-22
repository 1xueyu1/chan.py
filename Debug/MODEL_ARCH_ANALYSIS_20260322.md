# 模型架构合理性与问题诊断（基于最近实验）

生成时间：2026-03-22

## 1. 使用的实验样本

- 训练产物：
  - `Debug/full_v5_off/metrics_buy.json`
  - `Debug/full_v5_off/metrics_sell.json`
  - `Debug/tmp_v4/metrics_buy.json`
  - `Debug/tmp_v4/metrics_sell.json`
- 回测产物：
  - `result/full_symbols_1m_after_speedup/backtest_metrics.json`
  - `result/tmp_v4_bt/backtest_metrics.json`
  - `result/smoke_after_speedup/backtest_metrics.json`
  - `result/parallel_symbol_smoke/backtest_metrics.json`

## 2. 架构合理性结论

### 2.1 合理的部分

1. 数据切分架构合理：
- 训练使用“时间切分 + 币种切分（holdout）”的双重泛化评估，能避免纯时间内过拟合。

2. 双模型设计合理：
- 买点、卖点分模训练，符合方向异质性，避免单模型方向冲突。

3. 线上离线特征对齐思路合理：
- 使用 meta 索引映射对齐特征，工程稳定性较好。

4. 性能工程方向正确：
- 已实现 symbol 并行回测、门控批量预测、step 内矩阵预分配，性能瓶颈处理方向正确。

### 2.2 不合理或风险较高的部分

1. 目标定义偏弱：
- 当前标签阈值（固定收益阈值）更像“短期收益达标”，未直接对接交易成本后净收益与风险约束。

2. 概率校准不足：
- 训练 AUC 可用（full_v5_off 买/卖约 0.756/0.759），但回测端收益大幅为负，说明概率到交易决策映射存在失真。

3. 阈值与成本耦合未建模：
- 使用固定 signal_threshold（0.55）在不同币种、波动状态下不稳定。

4. 类别不平衡较强：
- tmp_v4 卖点模型出现 AUC=0.5、Recall=0，属于明显退化样本；说明小样本版本不具备可迁移性。

## 3. 关键问题定位

### 问题 A：训练指标与回测收益断层

- full_v5_off 训练表现：
  - 买点 AUC ~0.756，卖点 AUC ~0.759（可用）
- 但 full_symbols_1m_after_speedup 回测聚合：
  - total_return_pct = -20.52%
  - max_drawdown_pct = -22.12%
  - profit_factor = 0.139

解释：
- 分类任务目标与交易目标（净收益、回撤、成本）不一致。

### 问题 B：卖点退化风险

- tmp_v4 卖点：AUC=0.5，Precision/Recall/F1=0
- 说明在小样本与高不平衡环境下，卖点模型很容易塌缩到全负预测。

### 问题 C：跨币种泛化仍弱

- 回测多币种多数标的为负收益（且夏普普遍显著负值），说明当前特征 + 阈值方案在跨币种上不稳。

## 4. 专业解决方案（按优先级）

## P0（必须先做）

1. 重新定义训练标签为“成本后净收益达标”
- 标签构造中加入 fee + slippage + 最小持仓周期约束；
- 把目标从“毛收益阈值”改为“净收益阈值”。

2. 阈值校准改为分方向、分币种、分波动分位
- 不再使用全局 0.55；
- 在验证集做网格阈值搜索，以净收益/回撤作为目标选阈值。

3. 概率校准
- 对 buy/sell 分别做 Platt/Isotonic 校准（验证集）；
- 回测使用校准后概率决策。

## P1（高收益改进）

1. 回测层加入交易过滤器
- 冷却窗口（避免连续同向噪声信号）；
- 最小置信度差（buy_prob - sell_prob 或方向 margin）；
- 波动率过滤（极端高波动阶段降频）。

2. 损失函数与采样优化
- 保持 scale_pos_weight 的同时，增加 focal-like reweight 或 hard negative 采样；
- 卖点模型优先保证 Recall 下限。

3. 分市场状态建模
- 增加 regime 特征（波动分位、趋势状态、成交量状态）；
- 或按 regime 训练轻量子模型。

## P2（中长期）

1. 元标签（Meta-Labeling）
- 先做方向候选，再用二阶段模型判定“是否执行”。

2. 组合级目标优化
- 训练后阈值选择直接以组合层 Calmar / Sortino 最优为目标。

## 5. 版本管理与备份建议

已实现：
- `backup/` 统一版本备份目录；
- 使用 `backup/archive_version.py` 归档代码 + 训练/预测/回测产物。

建议规范：
- 目录命名：`framework_<框架名>_<日期或标签>`；
- 每次实验都保留 `manifest.json` + `docs/version_notes.md`；
- 避免覆盖历史目录，全部走新增版本目录。

## 6. 下一步执行清单（可直接落地）

1. 新增“阈值自动搜索脚本”（目标：净收益与回撤约束）。
2. 训练标签改成成本后净收益标签。
3. 新增概率校准文件输出（buy/sell 各一份）。
4. 回测引擎接入“分方向动态阈值 + 冷却窗口”。
