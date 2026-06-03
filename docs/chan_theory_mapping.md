# 缠论理论到代码的映射

## 当前核心结构

当前主线围绕二类买卖点族群：

```text
standard_bsp2
subclass_bsp2s
bsp2_t3_overlap
bsp2_after_bsp1
bsp2_near_origin_zs_boundary
```

代码位置：

```text
ML/routes/btc_futures_v2_bsp2_family/candidate_family.py
```

## 标签映射

主标签：

```text
label_bsp2_family_valid
```

含义：二类买卖点族群信号后，结构是否能按照当前定义走出有效路径。

辅助标签：

```text
label_bsp2_identity_strict
label_pre_confirm_invalid
label_bsp2_realtime_path
label_bsp2_realtime_path_id
```

代码位置：

```text
ML/routes/btc_futures_v2_bsp2_family/labels.py
```

## 实时路径分类

```text
invalid_before_confirm
return_origin_zs
third_confirmed
higher_level_expansion
bsp2_t3_overlap_trend
weak_no_confirm
```

这些分类用于训练二级决策模型和交易事件复盘。正式回测不能直接读取未来分类结果。

## 出场映射

`family_realtime` 状态机位于：

```text
ML/routes/btc_futures_v2_bsp2_family/backtest.py
```

核心理论映射：

- 结构失效：二类买卖点后未能维持结构，触及失效位。
- 回原中枢：二类买卖点后回到原中枢目标区域。
- 第三段确认：二类买卖点后走出第三段，随后等待同级别反向买卖点出场。
- 弱确认：限定时间内没有结构确认，也没有回中枢。

## 需要谨慎的地方

- 缠论理论分类可以用于设计标签，但回测必须只使用当时可见信息。
- 多周期和区间套特征必须对齐到当前时间点。
- Rust 加速不能改变 Python 缠论元素计算结果。
- 任何对买卖点底层识别的修改都必须增加一致性检查。
