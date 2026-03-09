# feature_center.py
# ============================================================
# 所有机器学习特征统一管理文件
# 使用方式：
#   from ChanModel.feature_center import build_features
#   feat_dict = build_features(klu, history, chan, bsp=last_bsp)
#   cfeatures.add_feat(feat_dict)
#
# 特征分类:
#   1. 收益率 & 动量 (ReturnFeatures)
#   2. 波动率 (VolatilityFeatures)      [已精简]
#   3. 趋势 & 均线 (TrendFeatures)
#   4. 成交量 (VolumeFeatures)
#   5. K线形态 (CandleFeatures)
#   6. RSI (RSIFeatures)
#   7. MACD (MACDFeatures)
#   8. KDJ (KDJFeatures)                [已精简]
#   9. 布林带 (BollingerFeatures)
#  10. ATR (ATRFeatures)
#  11. CCI (CCIFeatures)                [已精简]
#  12. 缠论结构 (ChanStructureFeatures)
#  13. 缠论笔MACD深度 (BiMACDFeatures)      [新增]
#  14. 缠论中枢深化 (ZSAdvancedFeatures)     [新增]
#  15. 缠论线段深化 (SegAdvancedFeatures)     [新增]
#  16. BSP关联特征 (BSPContextFeatures)       [新增]
#  17. 多K线结构 (MultiBarFeatures)           [新增]
#  18. 均线斜率 & EMA (EMAFeatures)           [新增]
# ============================================================

import numpy as np


# ============================================================
# 工具函数
# ============================================================

def _close_array(history):
    """合并K线收盘价 = 最后一个子K线单元的收盘价"""
    return np.array([k[-1].close for k in history])


def _open_array(history):
    """合并K线开盘价 = 第一个子K线单元的开盘价"""
    return np.array([k[0].open for k in history])


def _high_array(history):
    """合并K线最高价 = 所有子K线单元最高价的最大值"""
    return np.array([max(unit.high for unit in k.lst) for k in history])


def _low_array(history):
    """合并K线最低价 = 所有子K线单元最低价的最小值"""
    return np.array([min(unit.low for unit in k.lst) for k in history])


def _volume_array(history):
    """合并K线的成交量 = 所有子K线单元成交量之和"""
    return np.array([sum(unit.vol for unit in k.lst) for k in history])


def _ema(arr, period):
    """指数移动平均 (EMA)"""
    out = np.zeros_like(arr, dtype=float)
    alpha = 2.0 / (period + 1)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out


# ============================================================
# 1️⃣ 收益率 & 动量类
# ============================================================

class ReturnFeatures:

    @staticmethod
    def compute(history):
        if len(history) < 20:
            return {}

        close = _close_array(history)

        features = {
            "ret_1": close[-1] / close[-2] - 1,
            # "ret_3": close[-1] / close[-4] - 1,       # [精简] 与ret_1/ret_5高度相关
            "ret_5": close[-1] / close[-6] - 1,
            # "ret_10": close[-1] / close[-11] - 1,     # [精简] 与ret_5高度相关
            "momentum_5": close[-1] - close[-6],
            # "momentum_10": close[-1] - close[-11],    # [精简] 与momentum_5高度相关
        }

        # 价格加速度 (二阶差分，正值=加速上涨)
        features["price_acceleration"] = close[-1] - 2 * close[-2] + close[-3]

        # 三根K线动量方向: 连涨=1, 连跌=-1, 其他=0
        features["momentum_3bar"] = (
            1 if (close[-1] > close[-2] > close[-3]) else
            (-1 if (close[-1] < close[-2] < close[-3]) else 0)
        )

        return features


# ============================================================
# 2️⃣ 波动率类
# ============================================================

class VolatilityFeatures:

    @staticmethod
    def compute(history):
        if len(history) < 20:
            return {}

        close = _close_array(history)
        ret = np.diff(close) / (close[:-1] + 1e-9)

        return {
            # "vol_std_5": float(np.std(ret[-5:])),    # [精简] 与vol_std_10高相关
            # "vol_std_10": float(np.std(ret[-10:])),  # [精简] 与bar_range/boll_bandwidth重叠
            # "vol_std_20": float(np.std(ret[-20:])),  # [精简] 与vol_std_10高相关
        }


# ============================================================
# 3️⃣ 趋势 & 均线类
# ============================================================

class TrendFeatures:

    @staticmethod
    def compute(history):
        if len(history) < 30:
            return {}

        close = _close_array(history)
        price = close[-1]

        ma5 = float(np.mean(close[-5:]))
        ma10 = float(np.mean(close[-10:]))
        ma20 = float(np.mean(close[-20:]))

        features = {
            # 均线差值
            "ma5_ma20_diff": (ma5 - ma20) / (ma20 + 1e-9),
            # "price_ma20_ratio": price / (ma20 + 1e-9),  # [精简] ≈ price_ma20_dist + 1
            # "price_pos_20": (                         # [精简] 与boll_position信息重叠
            #     (price - np.min(close[-20:])) /
            #     (np.max(close[-20:]) - np.min(close[-20:]) + 1e-9)
            # ),
        }

        # ---- 价格相对各MA的归一化距离 ----
        # features["price_ma5_dist"] = (price - ma5) / (ma5 + 1e-9)  # [精简] 与ma20/ma60_dist高度相关
        # features["price_ma10_dist"] = (price - ma10) / (ma10 + 1e-9)  # [精简] 与ma5/ma20_dist高相关
        features["price_ma20_dist"] = (price - ma20) / (ma20 + 1e-9)

        # ---- 均线方向 (当前值 vs 前一期) ----
        # prev_ma5 = float(np.mean(close[-6:-1]))    # [精简] SHAP极低
        # prev_ma10 = float(np.mean(close[-11:-1]))  # [精简]
        # prev_ma20 = float(np.mean(close[-21:-1]))  # [精简] SHAP极低
        # features["ma5_direction"] = 1 if ma5 > prev_ma5 else -1    # [精简] SHAP极低
        # features["ma10_direction"] = 1 if ma10 > prev_ma10 else -1  # [精简]
        # features["ma20_direction"] = 1 if ma20 > prev_ma20 else -1  # [精简] SHAP极低

        # ---- 均线交叉 (短期 vs 长期) ----
        # features["ma5_ma10_cross"] = 1 if ma5 > ma10 else (-1 if ma5 < ma10 else 0)    # [精简] SHAP低
        # features["ma10_ma20_cross"] = 1 if ma10 > ma20 else (-1 if ma10 < ma20 else 0)  # [精简] SHAP低

        # ---- MA宽度 (衡量均线发散/收敛) ----
        # features["ma_width"] = (                    # [精简] 与boll_bandwidth重叠
        #     (max(ma5, ma10, ma20) - min(ma5, ma10, ma20)) / (price + 1e-9)
        # )

        # ---- 需要 60 根以上的扩展特征 ----
        if len(history) >= 60:
            ma60 = float(np.mean(close[-60:]))
            features["price_ma60_dist"] = (price - ma60) / (ma60 + 1e-9)
            features["ma20_ma60_cross"] = 1 if ma20 > ma60 else (-1 if ma20 < ma60 else 0)

            # 多级MA一致性趋势: [精简] SHAP极低
            # all_up = int(ma5 > ma10 > ma20 > ma60)
            # all_down = int(ma5 < ma10 < ma20 < ma60)
            # features["ma_trend_aligned"] = 1 if all_up else (-1 if all_down else 0)

            # 价格在近60根中的百分位位置
            # features["price_pos_60"] = (                    # [精简] 与price_pos_20高相关
            #     (price - np.min(close[-60:])) /
            #     (np.max(close[-60:]) - np.min(close[-60:]) + 1e-9)
            # )

        return features


# ============================================================
# 4️⃣ 成交量类
# ============================================================

class VolumeFeatures:

    @staticmethod
    def compute(history):
        if len(history) < 20:
            return {}

        vol = _volume_array(history)
        close = _close_array(history)

        # vol_ma5 = float(np.mean(vol[-5:]))  # [精简] 仅用于vol_ma5_direction
        vol_ma20 = float(np.mean(vol[-20:]))

        features = {
            "vol_ratio_20": vol[-1] / (vol_ma20 + 1e-9),
            "vol_change": vol[-1] / (vol[-2] + 1e-9) - 1,
            # 成交量标准差 (与 VolatilityFeatures.vol_std_20 收益率std 区分)
            "vol_amount_std_20": float(np.std(vol[-20:])),
        }

        # 成交量MA方向
        # prev_vol_ma5 = float(np.mean(vol[-6:-1]))                       # [精简] SHAP极低
        # features["vol_ma5_direction"] = 1 if vol_ma5 > prev_vol_ma5 else -1  # [精简] SHAP极低

        # 放量/缩量状态
        # features["vol_enlarge"] = 1 if vol[-1] > 1.5 * vol_ma20 else 0  # [精简] vol_ratio_20的阈值派生
        # features["vol_shrink"] = 1 if vol[-1] < 0.5 * vol_ma20 else 0   # [精简] vol_ratio_20的阈值派生

        # 量价配合度
        # price_up = close[-1] > close[-2]                                  # [精简] SHAP极低
        # vol_up = vol[-1] > vol[-2]                                        # [精简] SHAP极低
        # features["vol_price_sync"] = 1 if (price_up == vol_up) else 0    # [精简] SHAP极低

        return features


# ============================================================
# 5️⃣ K线形态类
# ============================================================

class CandleFeatures:

    @staticmethod
    def compute(klu):
        body = abs(klu.close - klu.open)
        full_range = klu.high - klu.low
        upper = klu.high - max(klu.close, klu.open)
        lower = min(klu.close, klu.open) - klu.low

        features = {
            "body_ratio": body / (klu.open + 1e-9),
            "upper_shadow_ratio": upper / (body + 1e-9),
            "lower_shadow_ratio": lower / (body + 1e-9),
            "is_bull": 1 if klu.close > klu.open else 0,
        }

        # K线振幅 = (最高-最低) / 开盘
        features["bar_range"] = full_range / (klu.open + 1e-9)

        # 实体在K线中的位置 (0=下部, 1=上部)
        if full_range > 0:
            features["bar_body_position"] = (
                (min(klu.close, klu.open) - klu.low) / full_range
            )
        else:
            features["bar_body_position"] = 0.5

        # 蜡烛线强度 = 实体 / 全长 (越大实体越饱满)
        features["candle_strength"] = body / (full_range + 1e-9)

        return features

    @staticmethod
    def compute_triple(history):
        """三根K线组合形态特征"""
        if len(history) < 3:
            return {}

        close = _close_array(history)
        open_arr = _open_array(history)
        high = _high_array(history)
        low = _low_array(history)

        c1, c2, c3 = close[-3], close[-2], close[-1]
        o1, o2, o3 = open_arr[-3], open_arr[-2], open_arr[-1]
        h1, h2, h3 = high[-3], high[-2], high[-1]
        l1, l2, l3 = low[-3], low[-2], low[-1]

        features = {}

        # 三连阳 / 三连阴
        features["triple_up"] = 1 if (c1 > o1 and c2 > o2 and c3 > o3) else 0
        features["triple_down"] = 1 if (c1 < o1 and c2 < o2 and c3 < o3) else 0

        # 递进新高 / 递进新低
        features["triple_new_high"] = 1 if (h3 > h2 > h1) else 0
        features["triple_new_low"] = 1 if (l3 < l2 < l1) else 0

        # 吞没形态 (最后两根): 看涨吞没=1, 看跌吞没=-1, 无=0
        bull_engulf = (c2 < o2) and (c3 > o3) and (c3 > o2) and (o3 < c2)
        bear_engulf = (c2 > o2) and (c3 < o3) and (o3 > c2) and (c3 < o2)
        features["engulfing"] = 1 if bull_engulf else (-1 if bear_engulf else 0)

        # 锤子线 (最后一根): 下影线 >= 2倍实体，上影线极小
        body3 = abs(c3 - o3)
        upper3 = h3 - max(c3, o3)
        lower3 = min(c3, o3) - l3
        if body3 > 0:
            features["hammer"] = 1 if (lower3 >= 2 * body3 and upper3 <= 0.3 * body3) else 0
            features["shooting_star"] = 1 if (upper3 >= 2 * body3 and lower3 <= 0.3 * body3) else 0
        else:
            features["hammer"] = 0
            features["shooting_star"] = 0

        return features


# ============================================================
# 6️⃣ RSI
# ============================================================

class RSIFeatures:

    @staticmethod
    def compute(history, period=14):
        if len(history) < period + 1:
            return {}

        close = _close_array(history)
        delta = np.diff(close)

        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)

        avg_gain = float(np.mean(gain[-period:]))
        avg_loss = float(np.mean(loss[-period:])) + 1e-9

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return {
            "rsi_14": rsi,
            # "rsi_zone": 1 if rsi > 70 else (-1 if rsi < 30 else 0),  # [精简] rsi_14的阈值编码，树模型自动学
        }


# ============================================================
# 7️⃣ MACD
# ============================================================

class MACDFeatures:

    @staticmethod
    def compute(history):
        if len(history) < 35:
            return {}

        close = _close_array(history)

        ema12 = _ema(close, 12)
        ema26 = _ema(close, 26)

        macd_line = ema12 - ema26
        signal = _ema(macd_line, 9)
        hist = macd_line - signal

        features = {
            # "macd": macd_line[-1],           # [精简] macd_hist = macd - signal，线性冗余
            # "macd_signal": signal[-1],       # [精简] 同上
            "macd_hist": hist[-1],
        }

        # 金叉=1 (柱状图由负转正), 死叉=-1 (由正转负), 其他=0
        if hist[-1] > 0 and hist[-2] <= 0:
            features["macd_cross"] = 1
        elif hist[-1] < 0 and hist[-2] >= 0:
            features["macd_cross"] = -1
        else:
            features["macd_cross"] = 0

        # 柱状图是否在扩大 (动能增强=1, 衰减=-1)
        features["macd_hist_expand"] = 1 if abs(hist[-1]) > abs(hist[-2]) else -1

        return features


# ============================================================
# 8️⃣ KDJ
# ============================================================

class KDJFeatures:

    @staticmethod
    def compute(history, fastk_period=9, slowk_period=3, slowd_period=3):
        min_len = fastk_period + max(slowk_period, slowd_period)
        if len(history) < min_len:
            return {}

        close = _close_array(history)
        high = _high_array(history)
        low = _low_array(history)

        n = len(close)

        # RSV: Raw Stochastic Value
        rsv = np.full(n, 50.0)
        for i in range(fastk_period - 1, n):
            highest = np.max(high[i - fastk_period + 1: i + 1])
            lowest = np.min(low[i - fastk_period + 1: i + 1])
            rng = highest - lowest
            if rng > 0:
                rsv[i] = (close[i] - lowest) / rng * 100

        # K = SMA(RSV, slowk_period, 1) = (K_prev*(N-1) + RSV) / N
        k = np.full(n, 50.0)
        for i in range(fastk_period, n):
            k[i] = (k[i - 1] * (slowk_period - 1) + rsv[i]) / slowk_period

        # D = SMA(K, slowd_period, 1)
        d = np.full(n, 50.0)
        for i in range(fastk_period, n):
            d[i] = (d[i - 1] * (slowd_period - 1) + k[i]) / slowd_period

        # J = 3K - 2D
        # j = 3 * k - 2 * d  # [精简] J未使用

        features = {
            "kdj_k": k[-1],
            # "kdj_d": d[-1],    # [精简] D是K的平滑，高度相关
            # "kdj_j": j[-1],    # [精简] J=3K-2D，与K线性相关
        }

        # 区间: [精简] 树模型自动从kdj_k学阈值
        # if k[-1] > 80 or d[-1] > 80:
        #     features["kdj_zone"] = 1
        # elif k[-1] < 20 or d[-1] < 20:
        #     features["kdj_zone"] = -1
        # else:
        #     features["kdj_zone"] = 0

        # K/D交叉: 金叉=1, 死叉=-1, 无=0
        if n >= fastk_period + 2:
            if k[-1] > d[-1] and k[-2] <= d[-2]:
                features["kdj_cross"] = 1
            elif k[-1] < d[-1] and k[-2] >= d[-2]:
                features["kdj_cross"] = -1
            else:
                features["kdj_cross"] = 0
        else:
            features["kdj_cross"] = 0

        # KDJ力度 = |K - D|
        features["kdj_power"] = abs(k[-1] - d[-1])

        return features


# ============================================================
# 9️⃣ 布林带 (Bollinger Bands)
# ============================================================

class BollingerFeatures:

    @staticmethod
    def compute(history, period=20, num_std=2):
        if len(history) < period:
            return {}

        close = _close_array(history)
        price = close[-1]

        ma = float(np.mean(close[-period:]))
        std = float(np.std(close[-period:]))

        upper = ma + num_std * std
        lower = ma - num_std * std
        bandwidth = upper - lower

        features = {}

        # 价格在布林带中的位置 (0=下轨, 1=上轨)
        features["boll_position"] = (price - lower) / (bandwidth + 1e-9)

        # 布林带宽度 (归一化)
        features["boll_bandwidth"] = bandwidth / (ma + 1e-9)

        # [精简] boll_zone => boll_position的阈值编码，树模型自动学
        # features["boll_zone"] = 1 if price > upper else (-1 if price < lower else 0)

        # [精简] boll_squeeze => boll_bandwidth的派生，且计算开销大
        # if len(close) >= 2 * period:
        #     prev_bws = []
        #     for offset in range(period):
        #         end_idx = len(close) - period + offset
        #         _ma = float(np.mean(close[end_idx - period: end_idx]))
        #         _std = float(np.std(close[end_idx - period: end_idx]))
        #         prev_bws.append(2 * num_std * _std / (_ma + 1e-9))
        #     avg_bw = float(np.mean(prev_bws))
        #     features["boll_squeeze"] = 1 if features["boll_bandwidth"] < 0.5 * avg_bw else 0
        # else:
        #     features["boll_squeeze"] = 0

        return features


# ============================================================
# 🔟 ATR (Average True Range)
# ============================================================

class ATRFeatures:

    @staticmethod
    def compute(history, period=14):
        if len(history) < period + 1:
            return {}

        close = _close_array(history)
        high = _high_array(history)
        low = _low_array(history)

        n = len(close)

        # True Range = max(H-L, |H-C_prev|, |L-C_prev|)
        tr = np.zeros(n)
        tr[0] = high[0] - low[0]
        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )

        atr = float(np.mean(tr[-period:]))

        features = {
            # "atr_value": atr,                        # [精简] 与atr_ratio仅差一个价格因子
            # 相对波动率 = ATR / 当前价格
            "atr_ratio": atr / (close[-1] + 1e-9),
        }

        # ATR趋势: 当前ATR vs 前一段ATR, 扩大=1, 缩小=-1
        if n >= 2 * period:
            prev_atr = float(np.mean(tr[-2 * period: -period]))
            features["atr_expanding"] = 1 if atr > prev_atr else -1
        else:
            features["atr_expanding"] = 0

        return features


# ============================================================
# 1️⃣1️⃣ CCI (Commodity Channel Index)
# ============================================================

class CCIFeatures:

    @staticmethod
    def compute(history, period=14):
        if len(history) < period:
            return {}

        close = _close_array(history)
        high = _high_array(history)
        low = _low_array(history)

        # Typical Price = (H + L + C) / 3
        tp = (high + low + close) / 3.0
        tp_recent = tp[-period:]

        ma_tp = float(np.mean(tp_recent))
        mean_dev = float(np.mean(np.abs(tp_recent - ma_tp))) + 1e-9

        cci = (tp[-1] - ma_tp) / (0.015 * mean_dev)

        return {
            "cci_value": cci,
            # "cci_zone": 1 if cci > 100 else (-1 if cci < -100 else 0),  # [精简] cci_value的阈值编码
        }


# ============================================================
# 1️⃣2️⃣ 缠论结构特征
# ============================================================

class ChanStructureFeatures:

    @staticmethod
    def compute(chan):
        if not hasattr(chan, "bi_list") or len(chan.bi_list) < 2:
            return {}

        bi_list = chan.bi_list
        last_bi = bi_list[-1]
        prev_bi = bi_list[-2]

        # ==========================
        # 1️⃣ 当前笔基础特征
        # ==========================

        start_price = last_bi.get_begin_val()
        end_price = last_bi.get_end_val()

        bi_return = (end_price - start_price) / (start_price + 1e-9)
        bi_is_up = 1 if last_bi.is_up() else 0
        bi_length = last_bi.get_klc_cnt()         # 合并K线数量
        # bi_klu_cnt = last_bi.get_klu_cnt()       # [精简] 与bi_length高相关
        # bi_amp = last_bi.amp()                   # [精简] 与bi_return高相关(仅差归一化)

        # ==========================
        # 2️⃣ 与前一笔对比（结构演化）
        # ==========================

        prev_return = (
            (prev_bi.get_end_val() - prev_bi.get_begin_val()) /
            (prev_bi.get_begin_val() + 1e-9)
        )
        prev_length = prev_bi.get_klc_cnt()
        # prev_amp = prev_bi.amp()  # [精简] 仅用于bi_amp_ratio

        bi_return_ratio = bi_return / (prev_return + 1e-9)
        bi_length_ratio = bi_length / (prev_length + 1e-9)
        # bi_amp_ratio = bi_amp / (prev_amp + 1e-9)  # [精简] 与bi_return_ratio高相关

        # 是否结构突破
        is_break_high = 0
        is_break_low = 0
        if last_bi.is_up():
            is_break_high = 1 if end_price > prev_bi.get_end_val() else 0
        else:
            is_break_low = 1 if end_price < prev_bi.get_end_val() else 0

        # ==========================
        # 3️⃣ 连续趋势统计
        # ==========================

        consecutive_count = 1
        for i in range(len(bi_list) - 2, -1, -1):
            if bi_list[i].is_up() == last_bi.is_up():
                consecutive_count += 1
            else:
                break

        # ==========================
        # 4️⃣ 背驰检测（力度衰减）
        # ==========================

        # strength_decay = 1 if abs(bi_return) < abs(prev_return) else 0  # [精简] 被divergence_signal包含

        price_new_extreme = 0
        if last_bi.is_up():
            price_new_extreme = 1 if end_price > prev_bi.get_end_val() else 0
        else:
            price_new_extreme = 1 if end_price < prev_bi.get_end_val() else 0

        # 价格创新高/低且力度衰减 → 背驰信号
        _strength_decay = 1 if abs(bi_return) < abs(prev_return) else 0
        divergence_signal = 1 if (price_new_extreme and _strength_decay) else 0

        # ==========================
        # 5️⃣ 笔的MACD度量 (利用CBi内置方法)
        # ==========================

        bi_macd_area = 0.0
        try:
            bi_macd_area = last_bi.Cal_MACD_area()
        except Exception:
            pass
        # bi_macd_slope: [精简] 与bi_macd_area高相关(slope≈area/length)

        # ==========================
        # 6️⃣ 中枢相关
        # ==========================

        is_in_zhongshu = 0
        distance_to_zhongshu_center = 0.0
        zs_bi_count = 0
        zs_peak_range = 0.0

        if hasattr(chan, "zs_list") and len(chan.zs_list) > 0:
            last_zs = chan.zs_list[-1]

            zs_high = last_zs.high
            zs_low = last_zs.low
            zs_center = (zs_high + zs_low) / 2
            current_price = end_price

            if zs_low <= current_price <= zs_high:
                is_in_zhongshu = 1

            # zhongshu_width: [精简] 与zs_peak_range高相关，保留peak_range

            distance_to_zhongshu_center = (
                (current_price - zs_center) / (zs_center + 1e-9)
            )

            # 中枢内笔数
            if hasattr(last_zs, "bi_lst"):
                zs_bi_count = len(last_zs.bi_lst)

            # 中枢峰谷振幅 (peak_high - peak_low) / center
            if hasattr(last_zs, "peak_high") and hasattr(last_zs, "peak_low"):
                zs_peak_range = (
                    (last_zs.peak_high - last_zs.peak_low) / (zs_center + 1e-9)
                )

        # ==========================
        # 7️⃣ 线段特征
        # ==========================

        seg_direction = 0
        seg_bi_count = 0
        seg_klu_slope = 0.0
        seg_zs_count = 0

        if hasattr(chan, "seg_list") and len(chan.seg_list) > 0:
            last_seg = chan.seg_list[-1]

            try:
                seg_direction = 1 if last_seg.is_up() else -1
            except Exception:
                pass
            try:
                seg_bi_count = last_seg.cal_bi_cnt()
            except Exception:
                pass
            # seg_amp: [精简] 与seg_klu_slope高相关(slope=amp/klu_cnt)
            try:
                seg_klu_slope = last_seg.cal_klu_slope()
            except Exception:
                pass
            try:
                seg_zs_count = last_seg.get_multi_bi_zs_cnt()
            except Exception:
                pass

        # ==========================
        # 8️⃣ 输出
        # ==========================

        return {
            # 当前笔
            "bi_return": bi_return,
            "bi_is_up": bi_is_up,
            "bi_length": bi_length,
            # "bi_klu_cnt": bi_klu_cnt,       # [精简]
            # "bi_amp": bi_amp,               # [精简]

            # 笔间关系
            "bi_return_ratio": bi_return_ratio,
            "bi_length_ratio": bi_length_ratio,
            # "bi_amp_ratio": bi_amp_ratio,   # [精简]

            # 趋势持续
            "bi_consecutive_count": consecutive_count,

            # 突破
            "is_break_high": is_break_high,
            "is_break_low": is_break_low,

            # 背驰
            # "strength_decay": strength_decay,   # [精简]
            "divergence_signal": divergence_signal,

            # 笔MACD度量
            "bi_macd_area": bi_macd_area,
            # "bi_macd_slope": bi_macd_slope,     # [精简]

            # 中枢
            "is_in_zhongshu": is_in_zhongshu,
            # "zhongshu_width": zhongshu_width,   # [精简]
            "distance_to_zhongshu_center": distance_to_zhongshu_center,
            "zs_bi_count": zs_bi_count,
            "zs_peak_range": zs_peak_range,

            # 线段
            "seg_direction": seg_direction,
            "seg_bi_count": seg_bi_count,
            # "seg_amp": seg_amp,               # [精简]
            "seg_klu_slope": seg_klu_slope,
            "seg_zs_count": seg_zs_count,
        }


# ============================================================
# 1️⃣3️⃣ 缠论笔MACD深度特征 (利用CBi内置MACD系列方法)
# ============================================================

class BiMACDFeatures:
    """
    利用笔对象的 Cal_MACD_peak / Cal_MACD_half / Cal_MACD_diff 等方法
    从多维度衡量笔的MACD力度，是背驰判断的核心依据。
    """

    @staticmethod
    def compute(chan):
        if not hasattr(chan, "bi_list") or len(chan.bi_list) < 3:
            return {}

        bi_list = chan.bi_list
        last_bi = bi_list[-1]
        prev_bi = bi_list[-2]

        features = {}

        # ---- 当前笔 MACD 多维度量 ----
        try:
            features["bi_macd_peak"] = last_bi.Cal_MACD_peak()
        except Exception:
            features["bi_macd_peak"] = 0.0

        try:
            features["bi_macd_half_obverse"] = last_bi.Cal_MACD_half_obverse()
        except Exception:
            features["bi_macd_half_obverse"] = 0.0

        try:
            features["bi_macd_half_reverse"] = last_bi.Cal_MACD_half_reverse()
        except Exception:
            features["bi_macd_half_reverse"] = 0.0

        try:
            features["bi_macd_diff"] = last_bi.Cal_MACD_diff()
        except Exception:
            features["bi_macd_diff"] = 0.0

        # ---- 当前笔 vs 前笔 MACD 力度比 ----
        try:
            prev_area = prev_bi.Cal_MACD_area()
            cur_area = last_bi.Cal_MACD_area()
            features["bi_macd_area_ratio"] = cur_area / (prev_area + 1e-9)
        except Exception:
            features["bi_macd_area_ratio"] = 1.0

        try:
            prev_peak = prev_bi.Cal_MACD_peak()
            cur_peak = last_bi.Cal_MACD_peak()
            features["bi_macd_peak_ratio"] = cur_peak / (prev_peak + 1e-9)
        except Exception:
            features["bi_macd_peak_ratio"] = 1.0

        # ---- MACD 半面积比 (正反向) ----
        # 正向半面积 / 全面积: 反映笔内MACD能量分布
        try:
            full = last_bi.Cal_MACD_area()
            half_ob = last_bi.Cal_MACD_half_obverse()
            features["bi_macd_half_ratio"] = half_ob / (full + 1e-9)
        except Exception:
            features["bi_macd_half_ratio"] = 0.5

        # ---- 前前笔对比 (同向相邻两笔力度衰减 → 核心背驰) ----
        if len(bi_list) >= 4:
            # 同向前笔 = bi_list[-3] (与 last_bi 方向相同)
            same_dir_bi = bi_list[-3]
            try:
                same_area = same_dir_bi.Cal_MACD_area()
                cur_area = last_bi.Cal_MACD_area()
                features["bi_macd_same_dir_ratio"] = cur_area / (same_area + 1e-9)
            except Exception:
                features["bi_macd_same_dir_ratio"] = 1.0

            try:
                features["bi_amp_same_dir_ratio"] = last_bi.amp() / (same_dir_bi.amp() + 1e-9)
            except Exception:
                features["bi_amp_same_dir_ratio"] = 1.0

        return features


# ============================================================
# 1️⃣4️⃣ 缠论中枢深化特征
# ============================================================

class ZSAdvancedFeatures:
    """
    中枢的多维特征: 中枢宽度、振荡强度、进出笔力度、
    多中枢趋势、中枢级别演化等。
    """

    @staticmethod
    def compute(chan):
        if not hasattr(chan, "zs_list") or len(chan.zs_list) == 0:
            return {}

        features = {}
        zs_list = chan.zs_list
        last_zs = zs_list[-1]

        bi_list = chan.bi_list if hasattr(chan, "bi_list") else []
        last_bi = bi_list[-1] if len(bi_list) > 0 else None
        current_price = last_bi.get_end_val() if last_bi else 0.0

        zs_center = (last_zs.high + last_zs.low) / 2
        zs_width = last_zs.high - last_zs.low

        # ---- 中枢基础归一化 ----
        features["zs_width_ratio"] = zs_width / (zs_center + 1e-9)

        # 中枢振荡强度: peak范围 / 中枢宽度
        if hasattr(last_zs, "peak_high") and hasattr(last_zs, "peak_low"):
            peak_range = last_zs.peak_high - last_zs.peak_low
            features["zs_oscillation_intensity"] = peak_range / (zs_width + 1e-9)
        else:
            features["zs_oscillation_intensity"] = 1.0

        # ---- 进出中枢笔力度比 ----
        if last_zs.bi_in is not None and last_zs.bi_out is not None:
            try:
                in_amp = last_zs.bi_in.amp()
                out_amp = last_zs.bi_out.amp()
                features["zs_out_in_amp_ratio"] = out_amp / (in_amp + 1e-9)
            except Exception:
                features["zs_out_in_amp_ratio"] = 1.0

            try:
                in_area = last_zs.bi_in.Cal_MACD_area()
                out_area = last_zs.bi_out.Cal_MACD_area()
                features["zs_out_in_macd_ratio"] = out_area / (in_area + 1e-9)
            except Exception:
                features["zs_out_in_macd_ratio"] = 1.0
        else:
            features["zs_out_in_amp_ratio"] = 0.0
            features["zs_out_in_macd_ratio"] = 0.0

        # ---- 价格突破中枢的程度 ----
        if current_price > last_zs.high:
            features["zs_break_degree"] = (current_price - last_zs.high) / (zs_width + 1e-9)
        elif current_price < last_zs.low:
            features["zs_break_degree"] = (last_zs.low - current_price) / (zs_width + 1e-9)
        else:
            features["zs_break_degree"] = 0.0

        # ---- 中枢延伸/复杂度 (子中枢数) ----
        if hasattr(last_zs, "sub_zs_lst"):
            features["zs_sub_count"] = len(last_zs.sub_zs_lst)
        else:
            features["zs_sub_count"] = 0

        # ---- 多中枢趋势特征 ----
        if len(zs_list) >= 2:
            prev_zs = zs_list[-2]
            prev_center = (prev_zs.high + prev_zs.low) / 2
            # 中枢中心升降
            features["zs_center_shift"] = (zs_center - prev_center) / (prev_center + 1e-9)
            # 中枢宽度变化 (收敛=正, 扩张=负)
            prev_width = prev_zs.high - prev_zs.low
            features["zs_width_change"] = (prev_width - zs_width) / (prev_width + 1e-9)
            # 两中枢是否有重叠 → 盘整 vs 趋势
            overlap = min(last_zs.high, prev_zs.high) - max(last_zs.low, prev_zs.low)
            features["zs_overlap_ratio"] = max(overlap, 0) / (zs_width + 1e-9)
        else:
            features["zs_center_shift"] = 0.0
            features["zs_width_change"] = 0.0
            features["zs_overlap_ratio"] = 0.0

        # 全局中枢数量
        features["zs_total_count"] = len(zs_list)

        return features


# ============================================================
# 1️⃣5️⃣ 缠论线段深化特征
# ============================================================

class SegAdvancedFeatures:
    """
    线段的深层特征: 线段内中枢结构、线段间演化、趋势线指标。
    """

    @staticmethod
    def compute(chan):
        if not hasattr(chan, "seg_list") or len(chan.seg_list) == 0:
            return {}

        features = {}
        seg_list = chan.seg_list
        last_seg = seg_list[-1]

        bi_list = chan.bi_list if hasattr(chan, "bi_list") else []
        current_price = bi_list[-1].get_end_val() if len(bi_list) > 0 else 0.0

        # ---- 线段振幅 ----
        try:
            features["seg_amp"] = last_seg.cal_amp()
        except Exception:
            features["seg_amp"] = 0.0

        # ---- 线段内第一/最后中枢特征 ----
        try:
            first_zs = last_seg.get_first_multi_bi_zs()
            final_zs = last_seg.get_final_multi_bi_zs()

            if first_zs is not None and final_zs is not None:
                fc = (first_zs.high + first_zs.low) / 2
                lc = (final_zs.high + final_zs.low) / 2
                features["seg_zs_center_trend"] = (lc - fc) / (fc + 1e-9)

                fw = first_zs.high - first_zs.low
                lw = final_zs.high - final_zs.low
                features["seg_zs_width_trend"] = (lw - fw) / (fw + 1e-9)
            else:
                features["seg_zs_center_trend"] = 0.0
                features["seg_zs_width_trend"] = 0.0
        except Exception:
            features["seg_zs_center_trend"] = 0.0
            features["seg_zs_width_trend"] = 0.0

        # ---- 线段 vs 前线段 ----
        if len(seg_list) >= 2:
            prev_seg = seg_list[-2]
            try:
                features["seg_amp_ratio"] = last_seg.amp() / (prev_seg.amp() + 1e-9)
            except Exception:
                features["seg_amp_ratio"] = 1.0

            try:
                features["seg_bi_cnt_ratio"] = (
                    last_seg.cal_bi_cnt() / (prev_seg.cal_bi_cnt() + 1e-9)
                )
            except Exception:
                features["seg_bi_cnt_ratio"] = 1.0

            try:
                features["seg_klu_cnt_ratio"] = (
                    last_seg.get_klu_cnt() / (prev_seg.get_klu_cnt() + 1e-9)
                )
            except Exception:
                features["seg_klu_cnt_ratio"] = 1.0
        else:
            features["seg_amp_ratio"] = 0.0
            features["seg_bi_cnt_ratio"] = 0.0
            features["seg_klu_cnt_ratio"] = 0.0

        # ---- 当前价格在线段中的位置 ----
        try:
            seg_high = last_seg._high()
            seg_low = last_seg._low()
            seg_range = seg_high - seg_low
            features["price_in_seg_position"] = (
                (current_price - seg_low) / (seg_range + 1e-9)
            )
        except Exception:
            features["price_in_seg_position"] = 0.5

        # ---- 趋势线特征 ----
        try:
            if last_seg.support_trend_line is not None and last_seg.support_trend_line.line is not None:
                features["seg_support_slope"] = last_seg.support_trend_line.line.slope
            else:
                features["seg_support_slope"] = 0.0
        except Exception:
            features["seg_support_slope"] = 0.0

        try:
            if last_seg.resistance_trend_line is not None and last_seg.resistance_trend_line.line is not None:
                features["seg_resistance_slope"] = last_seg.resistance_trend_line.line.slope
            else:
                features["seg_resistance_slope"] = 0.0
        except Exception:
            features["seg_resistance_slope"] = 0.0

        # 趋势线收敛度 (支撑斜率 - 阻力斜率)
        sup_slope = features.get("seg_support_slope", 0.0)
        res_slope = features.get("seg_resistance_slope", 0.0)
        features["seg_trendline_converge"] = sup_slope - res_slope

        return features


# ============================================================
# 1️⃣6️⃣ BSP 关联上下文特征
# ============================================================

class BSPContextFeatures:
    """
    利用买卖点对象本身的结构信息:
    - BSP 所在笔的 MACD 多维度量
    - BSP 关联的 1 类买卖点距离
    - BSP 所在位置的缠论力度
    """

    @staticmethod
    def compute(bsp, chan):
        if bsp is None:
            return {}

        features = {}
        bi = bsp.bi  # BSP 关联的笔

        # ---- BSP 笔的 MACD 丰富度量 ----
        try:
            features["bsp_bi_macd_peak"] = bi.Cal_MACD_peak()
        except Exception:
            features["bsp_bi_macd_peak"] = 0.0

        try:
            features["bsp_bi_macd_half_ob"] = bi.Cal_MACD_half_obverse()
        except Exception:
            features["bsp_bi_macd_half_ob"] = 0.0

        try:
            features["bsp_bi_macd_half_rev"] = bi.Cal_MACD_half_reverse()
        except Exception:
            features["bsp_bi_macd_half_rev"] = 0.0

        # 笔的MACD斜率 (price变化率/klu长度)
        try:
            features["bsp_bi_slope"] = bi.Cal_MACD_slope()
        except Exception:
            features["bsp_bi_slope"] = 0.0

        # ---- BSP 笔的成交量指标 ----
        try:
            features["bsp_bi_volume_avg"] = bi.Cal_MACD_trade_metric("volume", cal_avg=True)
        except Exception:
            features["bsp_bi_volume_avg"] = 0.0

        # ---- BSP 笔长度 (KLU精度) ----
        try:
            features["bsp_bi_klu_cnt"] = bi.get_klu_cnt()
        except Exception:
            features["bsp_bi_klu_cnt"] = 0

        # ---- 与关联的 1 类买卖点距离 ----
        if bsp.relate_bsp1 is not None:
            try:
                bsp1_price = bsp.relate_bsp1.klu.close
                bsp_price = bsp.klu.close
                features["bsp_to_bsp1_dist"] = (bsp_price - bsp1_price) / (bsp1_price + 1e-9)
            except Exception:
                features["bsp_to_bsp1_dist"] = 0.0
        else:
            features["bsp_to_bsp1_dist"] = 0.0

        # ---- BSP 笔前后笔的关系 ----
        if bi.pre is not None:
            try:
                features["bsp_pre_bi_amp"] = bi.pre.amp()
            except Exception:
                features["bsp_pre_bi_amp"] = 0.0
            try:
                features["bsp_pre_bi_macd_area"] = bi.pre.Cal_MACD_area()
            except Exception:
                features["bsp_pre_bi_macd_area"] = 0.0
        else:
            features["bsp_pre_bi_amp"] = 0.0
            features["bsp_pre_bi_macd_area"] = 0.0

        return features


# ============================================================
# 1️⃣7️⃣ 多K线结构特征
# ============================================================

class MultiBarFeatures:
    """
    多根K线的结构性特征: 缺口、连续同向、上下影线统计等。
    """

    @staticmethod
    def compute(history):
        if len(history) < 10:
            return {}

        close = _close_array(history)
        high = _high_array(history)
        low = _low_array(history)
        open_arr = _open_array(history)

        features = {}

        # ---- 近 N 根收益率分布特征 ----
        ret_10 = np.diff(close[-11:]) / (close[-11:-1] + 1e-9)
        features["ret_skew_10"] = float(_skewness(ret_10))
        features["ret_kurtosis_10"] = float(_kurtosis(ret_10))

        # ---- 缺口检测 (最近5根K线中是否有跳空) ----
        gap_count = 0
        for i in range(-5, -1):
            if low[i + 1] > high[i]:      # 向上跳空
                gap_count += 1
            elif high[i + 1] < low[i]:     # 向下跳空
                gap_count -= 1
        features["recent_gap_net"] = gap_count

        # ---- 最近 N 根中阳线占比 ----
        recent_bulls = sum(1 for i in range(-5, 0) if close[i] > open_arr[i])
        features["bull_ratio_5"] = recent_bulls / 5.0

        # ---- K线实体递增/递减趋势 ----
        bodies = [abs(close[i] - open_arr[i]) for i in range(-5, 0)]
        if len(bodies) >= 2:
            features["body_trend_5"] = (bodies[-1] - bodies[0]) / (bodies[0] + 1e-9)
        else:
            features["body_trend_5"] = 0.0

        # ---- 高低点收敛度 (最近10根的high-low逐渐减小=收敛) ----
        ranges_5 = [high[i] - low[i] for i in range(-5, 0)]
        ranges_prev_5 = [high[i] - low[i] for i in range(-10, -5)]
        avg_cur = np.mean(ranges_5)
        avg_prev = np.mean(ranges_prev_5)
        features["range_contraction"] = (avg_prev - avg_cur) / (avg_prev + 1e-9)

        # ---- 价格相对近20根高低点位置 ----
        h20 = np.max(high[-20:])
        l20 = np.min(low[-20:])
        features["price_range_20_pos"] = (close[-1] - l20) / (h20 - l20 + 1e-9)

        # ---- 距离近20根最高点/最低点的K线数 ----
        idx_high = np.argmax(high[-20:])
        idx_low = np.argmin(low[-20:])
        features["bars_since_high_20"] = 19 - idx_high
        features["bars_since_low_20"] = 19 - idx_low

        return features


# ============================================================
# 1️⃣8️⃣ EMA 距离 & 均线斜率特征
# ============================================================

class EMAFeatures:
    """
    补充 EMA 系列特征:
    - EMA 距离 (比 SMA 更灵敏)
    - 均线斜率 (方向强度)
    - DIF/DEA 距离 (MACD零轴距离)
    """

    @staticmethod
    def compute(history):
        if len(history) < 30:
            return {}

        close = _close_array(history)
        price = close[-1]
        features = {}

        # ---- EMA 距离 ----
        ema12 = _ema(close, 12)
        ema26 = _ema(close, 26)

        features["price_ema12_dist"] = (price - ema12[-1]) / (ema12[-1] + 1e-9)
        features["price_ema26_dist"] = (price - ema26[-1]) / (ema26[-1] + 1e-9)

        # ---- EMA 斜率 (变化率) ----
        features["ema12_slope"] = (ema12[-1] - ema12[-2]) / (ema12[-2] + 1e-9)
        features["ema26_slope"] = (ema26[-1] - ema26[-2]) / (ema26[-2] + 1e-9)

        # ---- DIF/DEA 位置 (MACD零轴关系) ----
        dif = ema12 - ema26
        dea = _ema(dif, 9)

        features["dif_position"] = dif[-1] / (price + 1e-9)
        features["dea_position"] = dea[-1] / (price + 1e-9)
        features["dif_dea_gap"] = (dif[-1] - dea[-1]) / (price + 1e-9)

        # ---- DIF 斜率 (DIF加速/减速) ----
        features["dif_slope"] = dif[-1] - dif[-2]

        # ---- EMA 带宽 (短周期 vs 长周期 EMA 距离) ----
        features["ema_band_width"] = (ema12[-1] - ema26[-1]) / (price + 1e-9)

        return features


# ============================================================
# 辅助统计函数
# ============================================================

def _skewness(arr):
    """偏度 (无需 scipy)"""
    n = len(arr)
    if n < 3:
        return 0.0
    m = np.mean(arr)
    s = np.std(arr)
    if s < 1e-12:
        return 0.0
    return float(np.mean(((arr - m) / s) ** 3))


def _kurtosis(arr):
    """超额峰度 (无需 scipy)"""
    n = len(arr)
    if n < 4:
        return 0.0
    m = np.mean(arr)
    s = np.std(arr)
    if s < 1e-12:
        return 0.0
    return float(np.mean(((arr - m) / s) ** 4) - 3.0)


# ============================================================
# 🎯 总入口函数（策略里只调用这个）
# ============================================================

def build_features(klu, history, chan, bsp=None):
    """
    构建全量 ML 特征字典

    Parameters
    ----------
    klu : CKLine_Unit
        当前（开仓）K线单元
    history : list[CKLine]
        合并K线历史列表 (通常为 cur_lv_chan.lst)
    chan : CKLine_List
        当前级别的缠论分析对象 (含 bi_list / zs_list / seg_list)
    bsp : CBS_Point, optional
        当前买卖点对象 (含关联笔、类型、特征)

    Returns
    -------
    dict[str, float]
        特征名 → 特征值
    """
    feat = {}

    # 价量基础
    feat.update(ReturnFeatures.compute(history))
    # feat.update(VolatilityFeatures.compute(history))  # [精简] vol_std_10与bar_range/boll_bandwidth重叠
    feat.update(TrendFeatures.compute(history))
    feat.update(VolumeFeatures.compute(history))

    # K线形态
    feat.update(CandleFeatures.compute(klu))
    # feat.update(CandleFeatures.compute_triple(history))  # [精简] 稀疏二值特征，SHAP贡献极低

    # 技术指标
    feat.update(RSIFeatures.compute(history))
    feat.update(MACDFeatures.compute(history))
    # feat.update(KDJFeatures.compute(history))    # [精简] kdj_k/cross/power与rsi_14重叠
    feat.update(BollingerFeatures.compute(history))
    feat.update(ATRFeatures.compute(history))
    # feat.update(CCIFeatures.compute(history))    # [精简] cci_value与rsi_14重叠

    # 缠论结构 (原始)
    feat.update(ChanStructureFeatures.compute(chan))

    # 缠论扩展 [新增]
    feat.update(BiMACDFeatures.compute(chan))
    feat.update(ZSAdvancedFeatures.compute(chan))
    feat.update(SegAdvancedFeatures.compute(chan))

    # BSP 关联 [新增]
    feat.update(BSPContextFeatures.compute(bsp, chan))

    # 多K线结构 [新增]
    feat.update(MultiBarFeatures.compute(history))

    # EMA & 均线斜率 [新增]
    feat.update(EMAFeatures.compute(history))

    return feat