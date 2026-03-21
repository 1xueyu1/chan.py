# feature_center.py
# flake8: noqa: E501
# ============================================================
# 所有机器学习特征统一管理文件
# 使用方式：
#   from ChanModel.feature_center import build_features
#   feat_dict = build_features(klu, history, chan)
#   cfeatures.add_feat(feat_dict)
#
# 特征分类:
#   1. 收益率 & 动量 (ReturnFeatures)
#   2. 波动率 (VolatilityFeatures)
#   3. 趋势 & 均线 (TrendFeatures)
#   4. 成交量 (VolumeFeatures)
#   5. K线形态 (CandleFeatures)
#   6. RSI (RSIFeatures)
#   7. MACD (MACDFeatures)
#   8. KDJ (KDJFeatures)
#   9. 布林带 (BollingerFeatures)
#  10. ATR (ATRFeatures)
#  11. CCI (CCIFeatures)
#  12. 缠论结构 (ChanStructureFeatures)
# ============================================================

import numpy as np


# 限制技术指标计算窗口，避免在长历史上重复全量扫描导致性能退化。
TECH_HISTORY_WINDOW = 180


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
            "momentum_5": close[-1] - close[-6],
        }

        # 价格加速度 (二阶差分，正值=加速上涨)
        features["price_acceleration"] = close[-1] - 2 * close[-2] + close[-3]

        return features


# ============================================================
# 2️⃣ 波动率类
# ============================================================

class VolatilityFeatures:

    @staticmethod
    def compute(history):
        # 历史验证中该类特征稳定低贡献，保留类接口以兼容旧调用。
        return {}


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

        ma20 = float(np.mean(close[-20:]))

        features = {
            # 价格在近20根中的百分位位置 (0=最低, 1=最高)
            "price_pos_20": (
                (price - np.min(close[-20:])) /
                (np.max(close[-20:]) - np.min(close[-20:]) + 1e-9)
            ),
            "price_ma20_dist": (price - ma20) / (ma20 + 1e-9),
        }

        # ---- 需要 60 根以上的扩展特征 ----
        if len(history) >= 60:
            ma60 = float(np.mean(close[-60:]))
            features["price_ma60_dist"] = (price - ma60) / (ma60 + 1e-9)
            features["ma20_ma60_cross"] = 1 if ma20 > ma60 else (-1 if ma20 < ma60 else 0)

            # 多级MA一致性趋势: 全部多头排列=1, 全部空头排列=-1, 其他=0
            ma5 = float(np.mean(close[-5:]))
            ma10 = float(np.mean(close[-10:]))
            all_up = int(ma5 > ma10 > ma20 > ma60)
            all_down = int(ma5 < ma10 < ma20 < ma60)
            features["ma_trend_aligned"] = 1 if all_up else (-1 if all_down else 0)

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
        vol_ma20 = float(np.mean(vol[-20:]))

        features = {
            "vol_ratio_20": vol[-1] / (vol_ma20 + 1e-9),
            "vol_change": vol[-1] / (vol[-2] + 1e-9) - 1,
            # 成交量标准差 (与 VolatilityFeatures.vol_std_20 收益率std 区分)
            "vol_amount_std_20": float(np.std(vol[-20:])),
        }

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
            "upper_shadow_ratio": upper / (body + 1e-9),
            "lower_shadow_ratio": lower / (body + 1e-9),
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

        return {
            "macd_hist": hist[-1],
        }


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

        return {
            "kdj_k": k[-1],
        }


# ============================================================
# 9️⃣ 布林带 (Bollinger Bands)
# ============================================================

class BollingerFeatures:

    @staticmethod
    def compute(history, period=20, num_std=2):
        if len(history) < period:
            return {}

        close = _close_array(history)

        ma = float(np.mean(close[-period:]))
        std = float(np.std(close[-period:]))

        upper = ma + num_std * std
        lower = ma - num_std * std
        bandwidth = upper - lower

        return {
            "boll_bandwidth": bandwidth / (ma + 1e-9),
        }


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

        features = {}

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
        bi_length = last_bi.get_klc_cnt()         # 合并K线数量

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

        # ==========================
        # 3️⃣ 笔的MACD度量 (利用CBi内置方法)
        # ==========================

        bi_macd_area = 0.0
        try:
            bi_macd_area = last_bi.Cal_MACD_area()
        except Exception:
            pass
        # bi_macd_slope: [精简] 与bi_macd_area高相关(slope≈area/length)

        # ==========================
        # 4️⃣ 中枢相关
        # ==========================

        distance_to_zhongshu_center = 0.0
        zs_bi_count = 0
        zs_peak_range = 0.0

        if hasattr(chan, "zs_list") and len(chan.zs_list) > 0:
            last_zs = chan.zs_list[-1]

            zs_high = last_zs.high
            zs_low = last_zs.low
            zs_center = (zs_high + zs_low) / 2
            current_price = end_price

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
        # 5️⃣ 线段特征
        # ==========================

        seg_direction = 0
        seg_bi_count = 0

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
        # ==========================
        # 6️⃣ 输出
        # ==========================

        return {
            # 当前笔
            "bi_length": bi_length,

            # 笔间关系
            "bi_return_ratio": bi_return_ratio,
            "bi_length_ratio": bi_length_ratio,

            # 笔MACD度量
            "bi_macd_area": bi_macd_area,

            # 中枢
            "distance_to_zhongshu_center": distance_to_zhongshu_center,
            "zs_bi_count": zs_bi_count,
            "zs_peak_range": zs_peak_range,

            # 线段
            "seg_direction": seg_direction,
            "seg_bi_count": seg_bi_count,
        }


# ============================================================
# 🎯 总入口函数（策略里只调用这个）
# ============================================================

def build_features(klu, history, chan):
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

    Returns
    -------
    dict[str, float]
        特征名 → 特征值
    """
    feat = {}
    history_tail = history[-TECH_HISTORY_WINDOW:] if len(history) > TECH_HISTORY_WINDOW else history

    # 价量基础
    feat.update(ReturnFeatures.compute(history_tail))
    feat.update(TrendFeatures.compute(history_tail))
    feat.update(VolumeFeatures.compute(history_tail))

    # K线形态
    feat.update(CandleFeatures.compute(klu))
    # feat.update(CandleFeatures.compute_triple(history))  # [精简] 稀疏二值特征，SHAP贡献极低

    # 技术指标
    feat.update(RSIFeatures.compute(history_tail))
    feat.update(MACDFeatures.compute(history_tail))
    feat.update(KDJFeatures.compute(history_tail))
    feat.update(BollingerFeatures.compute(history_tail))
    feat.update(ATRFeatures.compute(history_tail))
    feat.update(CCIFeatures.compute(history_tail))

    # 缠论结构
    feat.update(ChanStructureFeatures.compute(chan))

    return feat