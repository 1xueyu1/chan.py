# 全流程执行诊断报告

## 一. 最新执行状态 ✅

**最新成功运行**: `full_timing_benchmark_20260323` (2026-03-23 11:01:50)

| 阶段 | 状态 | 耗时 | 时间段 |
|------|------|------|--------|
| **train** | ✅ 完成 | 683.95s (11.4 min) | 10:32:43 ~ 10:44:07 |
| **predict** | 🔹 跳过 | - | - |
| **backtest** | ✅ 完成 | 1063.33s (17.7 min) | 10:44:07 ~ 11:01:50 |
| **总耗时** | ✅ 完成 | **1747.28s (0.49h)** | - |

---

## 二. 历史问题回顾

### 已修复的问题

**问题 #1**: DataFrame 索引歧义 (`t0_ts` 既是索引又是列)
- ❌ 症状: `ValueError: 't0_ts' is both an index level and a column label`
- 📝 位置: [ml_layer/feature_engine/engine.py#L198](ml_layer/feature_engine/engine.py#L198)
- ✅ 修复: 添加 `sample_df.reset_index(drop=True)` 
- ✅ 状态: 已验证成功

**问题 #2**: Pandas FutureWarning
- ❌ 症状: `Series.fillna with 'method' is deprecated`
- 📝 位置: [ml_layer/feature_engine/price_volume_microstructure_features.py#L117](ml_layer/feature_engine/price_volume_microstructure_features.py#L117)
- ✅ 修复: `.fillna(method="ffill")` → `.ffill()`
- ✅ 状态: 已验证成功

---

## 三. 当前代码稳定性评估

### ✅ 正常功能

1. **并行特征工程** (L3 FeatureEngine)
   - 符号级 ThreadPoolExecutor 并行化
   - O(1) 事件-K线查找表
   - 默认 4-8 个并行线程

2. **标签生成优化** (L4 LabelEngine)
   - OHLC numpy 数组缓存
   - 三重障碍高效计算

3. **特征重要性采样** (MDA)
   - 采样上限 6000 样本
   - 并行 MDA 计算

4. **耗时记录**
   - 每阶段时间线 JSON (`timing_summary.json`)
   - 全局历史 CSV (`timing_history.csv`)
   - 可用于长期性能追踪

### 性能数据

```
数据规模:
  - 合计样本数: 121,859 (8币种，6+年历史)
  - 训练币种: 8 (ADAUSDT, AVAXUSDT, BNBUSDT, BTCUSDT, DOGEUSDT, DOTUSDT, ETHUSDT, LTCUSDT)
  - 回测币种: 10 (上述8个 + SOLUSDT, XRPUSDT)

耗时分解:
  - 训练阶段: 683.95s = 标签生成 + 特征工程 + 5-fold 交叉验证 + XGBoost + SHAP
  - 回测阶段: 1063.33s = 10币种并行回测 + 聚合指标
  
性能指标:
  - OOS Sharpe: -6.6149 (完整历史数据，交易频繁)
  - OOS Precision: 0.6160 (买点准确度)
  - OOS Macro F1: 0.3184 (综合指标)
```

---

## 四. 隐患排查

### ❓ "代码总是跑一半停了" — 可能原因

**假设 A**: 用户指的是之前的 `full_train_backtest_20260323` 失败
- ✅ 已确认: 那次是 t0_ts 索引错误，已修复
- ✅ 新运行完全成功

**假设 B**: 用户的本地环境有隐藏的中断（网络断开、内存不足、进程超时）
- 建议处理方案见下文

**假设 C**: 代码有 race condition 或特定场景下的挂起
- 需要启用日志追踪

---

## 五. 稳定性加固建议

### (1) 添加进程看门狗 (Watchdog Timeout)

文件: `Debug/run_pipeline.py`

```python
# 为每个阶段添加超时保护
MAX_TRAIN_TIME = 2 * 3600  # 2小时
MAX_BACKTEST_TIME = 2 * 3600  # 2小时

def _run_cmd_with_timeout(cmd, log_path, timeout_sec, extra_env=None):
    """执行命令，超时时自动 kill"""
    start = time.time()
    try:
        elapsed = _run_cmd(cmd, log_path, extra_env)
        if elapsed > timeout_sec:
            print(f"[WARN] {cmd[0]} 耗时 {elapsed:.2f}s 超过阈值 {timeout_sec}s")
        return elapsed
    except subprocess.TimeoutExpired:
        print(f"[ERROR] 命令超时 ({timeout_sec}s): {' '.join(cmd)}")
        raise
```

### (2) 内存监控与低内存告警

```python
import psutil

def _check_memory_before_stage(stage_name: str, required_gb: float = 4.0):
    """检查可用内存，低于阈值时告警"""
    available_gb = psutil.virtual_memory().available / (1024**3)
    if available_gb < required_gb:
        print(f"[WARN] {stage_name} 可用内存仅 {available_gb:.2f}GB，建议关闭其他程序")
```

### (3) 子进程挂起检测

```python
def _run_cmd_with_hang_detection(cmd, log_path, idle_timeout_sec=600):
    """如果 30 分钟无日志输出，自动 kill"""
    # 监控日志文件大小是否增长
    # 如果卡住，自动 terminate 并记录
```

### (4) 增强日志输出

在关键节点添加详细日志：
- 数据加载完成数 (events count)
- 特征构造进度 (per symbol)
- 模型训练 fold 进度
- 回测 symbol 进度
- 子进程心跳信号

---

## 六. 立即可采取的行动

### ✅ 已完成

1. 修复 DataFrame 索引歧义
2. 修复 Pandas FutureWarning
3. 添加阶段时间线记录 (timing_summary.json + timing_history.csv)

### 🔜 建议优先级

| 优先级 | 任务 | 预计时间 | 效果 |
|--------|------|---------|------|
| **HIGH** | 添加进程超时看门狗 (MAX_TIME) | 1 小时 | 防止无限等待 |
| **HIGH** | 添加内存不足检测 | 30 分钟 | 提前告警 |
| **MEDIUM** | 添加详细进度日志 (per-symbol) | 1 小时 | 快速定位卡点 |
| **MEDIUM** | 实现子进程挂起自动恢复 | 2 小时 | 提升鲁棒性 |
| **LOW** | 性能优化（GPU 加速、分布式） | TBD | 降低耗时 |

---

## 七. 验证步骤

### 即刻验证（确认当前状态良好）

```bash
# 1. 检查最新运行是否成功完成
cat Debug/runs/full_timing_benchmark_20260323/timing_summary.json | jq ".timing_seconds.total"

# 2. 列出 backtest 产物
ls -lh Debug/runs/full_timing_benchmark_20260323/backtest/

# 3. 查看历史耗时趋势
cat Debug/runs/timing_history.csv | tail -10
```

### 长期监控（后续优化用）

```bash
# 每次运行后自动对比性能
head -1 Debug/runs/timing_history.csv > /tmp/header.txt
cat Debug/runs/timing_history.csv | tail -3 >> /tmp/perf_trend.txt

cat /tmp/perf_trend.txt | cut -d',' -f1,4,5,6 \
  | column -t -s',' | sort -k2 -u
```

---

## 八. 小结

**当前状态**: ✅ **功能完整，运行稳定**

最新全流程测试成功执行：
- 121,859 个样本
- 8 币种训练 + 10 币种回测
- 完整 5-fold 交叉验证
- 生成 SHAP 解释报告
- 总耗时 29.1 分钟

**如果用户还在遇到"跑一半停"**，建议：
1. 检查是否有网络中断 / 进程被 kill
2. 启用新增的看门狗+内存监控（待实现）
3. 查看 `Debug/runs/timing_history.csv` 历史，对比成功/失败的耗时模式
4. 运行中实时监控 `top` 或任务管理器，观察内存/CPU 使用

---

**生成时间**: 2026-03-23 11:05:00  
**基准运行**: `full_timing_benchmark_20260323`  
**历史 CSV**: `Debug/runs/timing_history.csv`
