# btc_futures_v1：缠论二类买卖点结构模型

当前主线按缠论原文重新收敛为“纯二类买卖点”模型。模型不再把二买/二卖简单理解为“触达旧中枢某个固定目标价”，而是学习二类买卖点之后的结构演化质量。

## 数据范围

原始数据使用本地 Binance U 本位合约 1m parquet：

```text
D:\WorkSpace\czsc_all\data\freqtrade_futures\futures
```

当前参与训练的币种：

```text
BTCUSDT / ETHUSDT / SOLUSDT / SUIUSDT / HYPEUSDT / ADAUSDT /
BNBUSDT / XRPUSDT / DOGEUSDT / AVAXUSDT / LINKUSDT
```

链路：

```text
1m 数据 -> 15m 缠论买卖点 -> 纯二类买卖点 -> 原文口径结构标签 -> 多空模型训练
```

无未来函数约束：

```text
exec_time 是信号可用时间；entry_time >= exec_time
```

## 标签设计

主训练标签：

```text
label_bsp2_valid
```

语义：

```text
二类买卖点之后，在真实结构失效前，出现第三段同向次级别走势。
二买对应：未先跌破一买结构失效位，随后向上突破一买后的第一段反弹高点。
二卖对应：未先突破一卖结构失效位，随后向下跌破一卖后的第一段回抽低点。
```

这个主标签不再把“回到原中枢”当作二类买卖点成立的必要条件。回原中枢、形成更大级别中枢扩张、二三买/卖合一，均作为二类买卖点之后的走势分类和强弱诊断字段保留。

`label_bsp2_chan_entry_quality` 仍作为诊断字段保留，但当前正式训练优先使用 `label_bsp2_valid`，因为它更直接地区分“第三段兑现前是否先结构失效”，更贴合当前要解决的坏二买/坏二卖问题。

保留的诊断字段：

```text
label_bsp2_valid
label_bsp2_origin_zs_available
label_bsp2_constructive_overlap
label_bsp2_strong_2_3_combo
label_bsp2_weak_or_invalid
label_bsp2_strength_class
label_bsp2_confirm_time
label_bsp2_confirm_price
label_bsp2_confirm_reason
label_bsp2_return_prev_zs
label_bsp2_return_prev_zs_level
label_bsp2_break_first_rebound
label_bsp2_third_move
label_bsp2_third_move_ref_price
label_bsp2_third_move_ref_source
label_bsp2_to_bsp3
label_bsp2_position_vs_origin_zs
label_bsp2_follow_path
```

结构失效位使用底层缠论元素给出的真实失效价：

```text
label_bsp2_invalid_price
label_bsp2_invalid_bi_idx
label_bsp2_invalid_source
```

二类买卖点对应的原中枢优先来自底层 BSP 特征：

```text
chan_bsp2_origin_zs_low
chan_bsp2_origin_zs_mid
chan_bsp2_origin_zs_high
```

如果没有真实原中枢，则记录为 `last_zs_fallback`，正式结构兑现回测默认不使用这类样本。

结构目标回放默认使用：

```text
--structure-target-level third
```

即用 `label_bsp2_third_move_ref_price` 作为二类买卖点第三段同向走势的兑现目标。`t1/t2/t3` 仍保留为原中枢 low/mid/high 的诊断口径，不再作为当前主标签的默认出场目标。

时间口径：

```text
15m K 线数据使用开盘时间标记。
exec_time 记录为当前链路发现新二类买卖点信号的事件时间。
标签和回测不再额外推迟 15m，入场直接使用 exec_time 对应的下一根 1m bar。
```

## 模型拆分

当前只训练纯二类买卖点：

```text
bsp_types_str == "2"
```

默认不混入 `2s`、`2,3b` 等扩展/混合信号。模型文件：

```text
buy_bsp2_model.pkl
sell_bsp2_model.pkl
buy_model.pkl
sell_model.pkl
```

`buy_model.pkl` 和 `sell_model.pkl` 是兼容入口。

## 特征设计

保留并扩展缠论结构特征：

```text
chan_ctx_ / chan_mtf_ / chan_bi_ / chan_zs_ / chan_fx_ / chan_bsp_
cap_chan_ / v3_
```

多周期和区间套相关特征：

```text
tech_5m_
tech_15m_
tech_1h_
tech_4h_
tech_1d_
```

覆盖 MACD、均线、布林带、ATR、波动状态、多周期共振和冲突等。

## 重要说明

旧字段 `label_bsp2_entry_quality` 仍可能存在于历史数据中，但不再作为主训练标签。正式训练应使用 `label_bsp2_valid`，并在重建数据集后再训练模型。

## 当前推荐回测口径

当前更稳的口径是：

```text
--model-kind chan_xgb
--primary-label-column label_bsp2_valid
--exit-mode third_then_opposite_no_after_timeout
--enable-structure-risk-sizing
--target-trade-risk-pct 0.01
--min-stake-multiplier 0.15
--tier-a-max-stake 1.0
--tier-b-max-stake 0.55
--tier-c-max-stake 0.25
--tier-a-risk-boost 3.0
--tier-b-risk-boost 1.0
--tier-c-risk-boost 0.6
```

含义是：第三段前保留超时保护，第三段兑现后不再因为时间到期退出，而是等待同级别任意反向买卖点，或结构/成本保护触发。结构风险定仓不再把所有信号平均压低，而是让 A 类信号使用更大仓位，B/C 类继续低仓位。
