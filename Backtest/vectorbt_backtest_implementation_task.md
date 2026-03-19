# Chan 项目 vectorbt 回测框架重构任务文档（可直接交给 AI 执行）

## 1. 文档目的
本任务文档用于指导 AI 在当前仓库内，完整实现一套基于 vectorbt 的专业化回测框架，并与现有缠论信号生成、XGBoost 买卖点质量过滤模型、Parquet 数据体系无缝衔接。

目标不是做一个独立 demo，而是构建可复用、可扩展、可验证的回测架构，能够支撑后续策略研究迭代。

---

## 2. 项目现状与关键事实（必须先读）

### 2.1 当前真实可用能力
1. 缠论引擎可用：
   - CChan 支持 step_load 逐步计算。
   - 可在每一帧获取最新 BSP（买卖点）。
2. 数据源可用：
   - 已接入 DATA_SRC.PARQUET。
   - 支持从 data 目录读取 BASE_interval.parquet。
   - 缺少目标周期文件时，可从 5m 自动重采样到 15m/30m/1h/1d/1w/1mo。
3. 模型训练与推理可用：
   - 已有双模型（买点模型 + 卖点模型）训练脚本。
   - 推理逻辑已可按 BSP 方向切换模型。
4. 结果产物约定已存在：
   - result/model_signal_events.csv
   - result/model_signal_bars.csv
   - result/xgb_backtest_report.html
   - result/baseline_backtest_report.html
   - result/comparison_report.html

### 2.2 当前明显缺口
1. 代码内引用了 Backtest.engine 与 Backtest.strategy，但当前工作区没有 Backtest 源码目录。
2. 现有回测逻辑分散在 Debug 脚本中，缺少统一框架层和标准接口。
3. 缺少向量化回测主引擎（vectorbt）与缠论事件流之间的规范适配层。

### 2.3 关键文件参考（实现时必须保持兼容）
- Chan.py
- ChanConfig.py
- Common/CEnum.py
- DataAPI/parquetAPI.py
- ChanModel/feature_center.py
- Debug/xgboost_shap_train.py
- Debug/xgboost_shap_predict.py
- Debug/strategy_xgb.py
- Script/binance_data_parquet_downloader.py

---

## 3. 总体目标

### 3.1 功能目标
构建一个 vectorbt 回测框架，实现以下能力：
1. 从 Parquet 读取并标准化 K 线数据。
2. 通过 CChan.step_load 逐帧提取 BSP 信号事件。
3. 将 BSP 信号通过 XGBoost 双模型打分，得到可交易信号。
4. 将事件流转换为 vectorbt 可消费的向量化 entries/exits。
5. 支持 long-only 与 long-short 两种交易模式。
6. 输出标准化结果文件（事件表、bars 表、指标 JSON、HTML 报告）。
7. 提供 CLI 入口脚本，支持单币种与多币种回测。
8. 保证无未来函数（lookahead bias）。

### 3.2 工程目标
1. 模块化：数据层、信号层、模型层、执行层、报告层解耦。
2. 可测试：至少具备核心单元测试与端到端烟雾测试。
3. 可复现：相同输入参数下输出稳定。
4. 可扩展：后续可以添加资金管理、组合优化、多策略并行。

### 3.3 非目标（本轮不做）
1. 不重写缠论核心计算算法。
2. 不重写模型训练框架。
3. 不引入实盘下单逻辑。
4. 不做高频撮合级别微观结构模拟。

---

## 4. 设计原则与约束
1. Python 版本按项目现状保持 3.11。
2. 保持 DATA_SRC.PARQUET 为默认回测数据源。
3. 交易执行必须明确时点：默认使用 next_bar_open 执行，避免同 bar 信号同 bar 成交导致潜在未来函数。
4. 保持结果文件命名兼容，便于沿用既有分析流程。
5. 所有新增模块必须有清晰类型注解与最小必要注释。
6. 所有时间索引统一为 UTC-aware DatetimeIndex。

---

## 5. 目标架构（分层）

### 5.1 L0 配置层
职责：统一读取回测参数。

建议文件：
- Backtest/config.py

核心内容：
- BacktestConfig dataclass
- 参数校验与默认值
- CLI 参数映射

### 5.2 L1 数据层
职责：将 Parquet K 线加载为规范 DataFrame。

建议文件：
- Backtest/data_loader.py

输入：
- symbol
- interval
- begin_time
- end_time

输出标准：
DataFrame（index=DatetimeIndex[UTC], columns 至少包含）
- open
- high
- low
- close
- volume

### 5.3 L2 缠论事件层
职责：逐帧调用 CChan，抽取 BSP 事件，形成事件表。

建议文件：
- Backtest/chan_signal_extractor.py

输出事件表字段：
- exec_time
- bsp_time
- is_buy
- bsp_type
- bsp_types_str
- trade_price
- klu_idx
- symbol

注意：
- 同一 bsp.klu.idx 仅处理一次（去重）。
- 事件时间与执行时间需分离。

### 5.4 L3 模型打分层
职责：加载买卖双模型，对事件打分并产出交易信号。

建议文件：
- Backtest/model_gate.py

输入：
- 事件对象 + BSP 特征
- model_buy/model_sell + meta_buy/meta_sell

输出：
- probability
- qualified
- signal（1, 0, -1）

规则：
- is_buy=True 且 probability >= threshold -> signal=1
- is_buy=False 且 probability >= threshold -> signal=-1
- 否则 signal=0

### 5.5 L4 向量化执行层（vectorbt 核心）
职责：把离散事件投影到 bar 级布尔矩阵，运行 vectorbt 回测。

建议文件：
- Backtest/vectorbt_engine.py

关键步骤：
1. 将事件映射到 bar 时间轴。
2. 构建 entries/exits 或 long_entries/long_exits/short_entries/short_exits。
3. 根据 execution_mode 进行信号位移：
   - next_bar_open：signal 向后 shift(1)，使用 open 作为 price。
   - close：不位移，使用 close 作为 price（仅研究模式）。
4. 调用 vectorbt.Portfolio.from_signals。
5. 输出 Portfolio 对象与关键统计。

### 5.6 L5 报告层
职责：产出 CSV/JSON/HTML 报告并兼容历史产物命名。

建议文件：
- Backtest/reporter.py

输出建议：
- result/model_signal_events.csv
- result/model_signal_bars.csv
- result/backtest_metrics.json
- result/xgb_backtest_report.html
- result/comparison_report.html（可选）

---

## 6. 目录与文件级实施清单（必须执行）

### 6.1 新增目录结构
Backtest/
- __init__.py
- config.py
- types.py
- data_loader.py
- chan_signal_extractor.py
- model_gate.py
- signal_builder.py
- vectorbt_engine.py
- reporter.py
- engine.py
- strategy.py
- examples/
  - run_vectorbt_backtest.py
- tests/
  - test_data_loader.py
  - test_signal_builder.py
  - test_vectorbt_engine.py
  - test_smoke_e2e.py

说明：
- engine.py 与 strategy.py 作为兼容层，供 Debug/strategy_xgb.py 继续使用原入口名。
- 兼容层内部转发到 vectorbt 新引擎。

### 6.2 需要改造的现有文件
1. Debug/strategy_xgb.py
   - 迁移到新框架调用方式。
   - 保持配置常量与模型路径定义不变。
   - 允许开关：long_only / long_short。
2. Debug/xgboost_shap_predict.py（可选轻改）
   - 抽取可复用的单事件打分逻辑到 Backtest/model_gate.py。
3. Script/requirements.txt 或新增 requirements_backtest.txt
   - 增加 vectorbt 及其依赖。

---

## 7. 接口契约（AI 实现时必须遵守）

### 7.1 BacktestConfig
字段建议：
- symbols: list[str]
- begin_time: str
- end_time: str
- kl_type: str 或 KL_TYPE
- data_src: DATA_SRC（默认 PARQUET）
- initial_cash: float
- fee: float
- slippage: float
- signal_threshold: float
- allow_short: bool
- execution_mode: str（next_bar_open 或 close）
- output_dir: str
- save_events_csv: bool
- save_bars_csv: bool
- save_html_report: bool
- random_seed: int

### 7.2 信号事件对象
字段必须包含：
- exec_time: pd.Timestamp
- bsp_time: str
- symbol: str
- is_buy: bool
- bsp_type: str
- bsp_types_str: str
- probability: float
- qualified: bool
- signal: int
- trade_price: float

### 7.3 bars 表对象
字段必须包含：
- time
- open
- high
- low
- close
- volume
- signal
- position

position 规则：
- long_only：0 或 1
- long_short：-1, 0, 1

### 7.4 兼容入口
Backtest/engine.py 暴露函数：
- run_chan_backtest_no_vnpy(...)

Backtest/strategy.py 暴露基类：
- ChanStrategyBase

注意：
- 即使内部不再是事件驱动撮合，也必须保留这个 API 名称，减少已有脚本改造范围。

---

## 8. 核心算法细则（必须按此实现）

### 8.1 信号去重
同一 symbol 下，若 BSP 的 klu_idx 已处理，则跳过，避免重复触发。

### 8.2 信号冲突处理
同一 bar 若出现买卖冲突：
1. 默认优先平仓信号（exit 优先）。
2. 再处理开仓信号（entry）。
3. 冲突规则必须可配置（future: conflict_policy）。

### 8.3 执行时点
默认 next_bar_open：
1. 事件在 t 生成。
2. 订单在 t+1 的 open 执行。
3. 若 t+1 不存在（最后一根），丢弃该事件并记录 warning。

### 8.4 资金与仓位
long_only 基线：
- 买点合格信号触发满仓开多（target 100%）
- 卖点合格信号触发平多（target 0%）

long_short 扩展：
- 买点合格：若空仓则开多，若有空先平空后开多
- 卖点合格：若空仓则开空，若有多先平多后开空

### 8.5 费用模型
至少支持：
- 单边手续费 fee（按成交额比例）
- 滑点 slippage（按成交价比例）

---

## 9. 输出规范（必须落地）

### 9.1 model_signal_events.csv
字段顺序建议：
exec_time,bsp_time,is_buy,bsp_type,bsp_types_str,probability,qualified,signal,trade_price,symbol

### 9.2 model_signal_bars.csv
字段顺序建议：
time,open,high,low,close,volume,signal,position,symbol

### 9.3 backtest_metrics.json
至少包含：
- total_return_pct
- annualized_return_pct
- max_drawdown_pct
- sharpe
- sortino
- calmar
- win_rate_pct
- profit_factor
- total_trades
- avg_trade_return_pct
- exposure_time_pct
- start_cash
- end_equity

### 9.4 HTML 报告
必须包含：
1. 核心 KPI 卡片
2. 权益曲线
3. 回撤曲线
4. 按月收益热力图（可选）
5. 交易分布与信号统计

---

## 10. 多币种支持方案

### 10.1 第一阶段（必做）
多币种独立回测：
- 每个 symbol 单独跑一套 vectorbt Portfolio
- 汇总组合报告（等权聚合）

### 10.2 第二阶段（可做）
统一资金池回测：
- 使用多列 close 与信号矩阵
- 支持 cash_sharing 与 group_by 配置

---

## 11. 测试与验收（必须执行）

### 11.1 单元测试
1. test_data_loader.py
   - 能正确读取 parquet
   - 时间过滤正确
   - 缺列时抛异常
2. test_signal_builder.py
   - 去重正确
   - next_bar_open 位移正确
   - 冲突处理正确
3. test_vectorbt_engine.py
   - 在已知信号序列下，交易次数与持仓状态符合预期

### 11.2 端到端烟雾测试
test_smoke_e2e.py：
- symbol: BTCUSDT
- interval: 15m
- 日期范围: 2025-01-01 ~ 2025-01-31
- 输出文件全部生成
- 指标 JSON 可解析
- HTML 报告可打开

### 11.3 无未来函数校验
必须做至少一条断言：
- next_bar_open 模式下，任何成交时间 > 信号生成时间。

### 11.4 验收标准（Definition of Done）
满足以下全部条件才算完成：
1. Debug/strategy_xgb.py 可直接运行且调用 vectorbt 新框架。
2. 输出 CSV/JSON/HTML 文件完整。
3. 单测通过，烟雾测试通过。
4. 关键指标可复现。
5. 无未来函数风险。

---

## 12. 迁移步骤建议（执行顺序）

1. 创建 Backtest 目录与基础 dataclass/types。
2. 实现 data_loader 与 chan_signal_extractor。
3. 抽离 model_gate（从现有推理逻辑复用）。
4. 实现 signal_builder（事件到向量化信号）。
5. 实现 vectorbt_engine。
6. 实现 reporter。
7. 实现兼容层 engine.py 与 strategy.py。
8. 改造 Debug/strategy_xgb.py 接入新引擎。
9. 补充测试与样例脚本。
10. 运行烟雾回测并生成报告。

---

## 13. 关键风险与规避

1. 风险：同 bar 信号与成交导致未来函数。
   - 规避：默认 next_bar_open + 信号位移。
2. 风险：BSP 重绘导致历史信号与实时不一致。
   - 规避：严格使用 step_load 每帧快照事件，不在末端回看重算。
3. 风险：多币种时间轴不齐。
   - 规避：先做单币种独立回测，再做组合层按时间对齐。
4. 风险：模型 meta 与特征错位。
   - 规避：严格按 meta 索引构造向量，缺失填 NaN。
5. 风险：当前仓库缺 Backtest 源码导致兼容断裂。
   - 规避：本任务必须补齐 Backtest/engine.py 与 Backtest/strategy.py。

---

## 14. 需要 AI 在实现后提交的产物清单

### 14.1 代码文件
- Backtest 全套新增文件
- Debug/strategy_xgb.py 改造版本
- 如有需要：requirements_backtest.txt

### 14.2 文档文件
- Backtest/README.md
  - 如何运行
  - 参数说明
  - 输出说明

### 14.3 运行产物
- result/model_signal_events.csv
- result/model_signal_bars.csv
- result/backtest_metrics.json
- result/xgb_backtest_report.html

### 14.4 验证记录
- 单测输出摘要
- 烟雾测试命令与关键日志

---

## 15. 建议命令（供 AI 执行）

1. 安装依赖
- pip install vectorbt pandas numpy plotly numba

2. 运行单币种回测
- python Backtest/examples/run_vectorbt_backtest.py --symbols BTCUSDT --begin-time 2025-01-01 --end-time 2025-01-31 --kl-type 15m --signal-threshold 0.55 --allow-short false

3. 运行多币种回测
- python Backtest/examples/run_vectorbt_backtest.py --symbols BTCUSDT ETHUSDT BNBUSDT SOLUSDT ADAUSDT --begin-time 2025-01-01 --end-time 2025-03-01 --kl-type 15m --signal-threshold 0.55 --allow-short false

4. 运行测试
- pytest Backtest/tests -q

---

## 16. 交付质量要求
1. 代码可读性高，命名语义化。
2. 关键逻辑有简洁注释。
3. 不引入与任务无关的大规模重构。
4. 不改动缠论核心模块行为。
5. 所有路径默认兼容当前仓库结构。

---

## 17. 可选增强（本轮后续）
1. 接入 vectorbt 参数扫描（阈值、手续费、滑点）自动寻优。
2. 引入 walk-forward 回测流程。
3. 增加组合层资金分配策略（等权、风险平价、波动率目标）。
4. 加入交易时段过滤、冷静期、最大持仓条数等风控组件。
5. 增加与训练指标联动的一体化报告。

---

## 18. 最终执行要求（给 AI 的硬约束）
1. 先补齐 Backtest 目录再改 Debug 脚本。
2. 每完成一个模块先跑最小可用验证再进入下一个模块。
3. 任何时间相关逻辑优先保证无未来函数。
4. 最后统一跑端到端并检查结果文件齐全。

本任务文档到此结束。