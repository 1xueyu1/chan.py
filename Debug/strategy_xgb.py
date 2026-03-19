"""
XGBoost 质量过滤 × 缠论买卖点回测策略
============================================================
使用训练好的双模型（买点/卖点质量评估）对买卖信号进行过滤：
  - 检测到买点时，用买点模型预测是否为合格买点
  - 检测到卖点时，用卖点模型预测是否为合格卖点
  - 仅在模型预测质量概率 >= SIGNAL_THRESHOLD 时执行交易

配套文件（需提前训练生成）:
  Debug/model_buy.json    Debug/meta_buy.json
  Debug/model_sell.json   Debug/meta_sell.json

运行：
  cd D:\\WorkSpace\\python\\chan.py
  python Debug/strategy_xgb.py
"""

import json
import os
import sys

# 确保项目根目录在 sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import xgboost as xgb

from Backtest.strategy import ChanStrategyBase
from Backtest.engine import run_chan_backtest_no_vnpy
from ChanModel.feature_center import build_features
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE


# ============================================================
# 配置
# ============================================================

CODE = "BTCUSDT"

# 与训练集时间段不重叠的样本外测试区间
BACKTEST_BEGIN = "2023-01-01"
BACKTEST_END   = "2026-01-01"

INITIAL_CASH = 100_000.0

# 模型合格概率阈值：概率 >= 此值才执行交易
SIGNAL_THRESHOLD = 0.55

# 模型文件路径（相对项目根目录）
_DEBUG = os.path.join(os.path.dirname(__file__))
MODEL_BUY_PATH  = os.path.join(_DEBUG, "model_buy.json")
MODEL_SELL_PATH = os.path.join(_DEBUG, "model_sell.json")
META_BUY_PATH   = os.path.join(_DEBUG, "meta_buy.json")
META_SELL_PATH  = os.path.join(_DEBUG, "meta_sell.json")

# 缠论配置（必须与训练时保持一致）
CHAN_CONFIG = {
    "bi_strict": True,
    "skip_step": 0,
    "divergence_rate": float("inf"),
    "bsp2_follow_1": False,
    "bsp3_follow_1": False,
    "min_zs_cnt": 0,
    "bs1_peak": False,
    "macd_algo": "peak",
    "bs_type": "1,2,3a,1p,2s,3b",
    "print_warning": False,
    "zs_algo": "normal",
}


# ============================================================
# XGBoost 预测辅助
# ============================================================

def _predict_bsp_quality(bsp, bsp_main_type, model, meta, feature_names):
    """
    对单个 BSP 预测"合格"概率。

    Parameters
    ----------
    bsp           : CBS_Point，已补充 feature_center 特征
    bsp_main_type : str，'1'/'2'/'3'
    model         : xgb.Booster
    meta          : dict[str, int]，特征名 → 列索引
    feature_names : list[str]，按列索引排列的特征名

    Returns
    -------
    float: 模型预测的"合格"概率 [0, 1]
    """
    feature_arr = np.full(len(meta), np.nan)

    # BSP 类型独热编码（meta 中可能没有 bsp_type_1，按实际情况填充）
    for t in ('1', '2', '3'):
        key = f"bsp_type_{t}"
        if key in meta:
            feature_arr[meta[key]] = 1.0 if bsp_main_type == t else 0.0

    # feature_center 特征 + BSP 内置特征
    for feat_name, feat_value in bsp.features.items():
        if feat_name in meta:
            feature_arr[meta[feat_name]] = feat_value

    dmat = xgb.DMatrix(
        feature_arr.reshape(1, -1),
        feature_names=feature_names,
        missing=np.nan,
    )
    return float(model.predict(dmat)[0])


# ============================================================
# 策略类
# ============================================================

class XGBChanStrategy(ChanStrategyBase):
    """
    XGBoost 质量过滤的缠论买卖点策略（做多）。

    买卖规则：
      出现买点 AND 买点模型概率 >= threshold → 全仓买入（空仓时）
      出现卖点 AND 卖点模型概率 >= threshold → 全仓平多（有仓位时）

    特征构建与训练脚本完全一致，确保推理/训练一致性。
    缺失特征（因 feature_center.py 精简）由 XGBoost 以 NaN 处理。
    """

    def __init__(self, signal_threshold: float = SIGNAL_THRESHOLD):
        super().__init__(chan_config_dict=CHAN_CONFIG)
        self.signal_threshold = signal_threshold

        # 模型（on_chan_init 中加载）
        self._model_buy: xgb.Booster = None
        self._model_sell: xgb.Booster = None
        self._meta_buy: dict = None
        self._meta_sell: dict = None
        self._fnames_buy: list = None
        self._fnames_sell: list = None

        # 统计
        self.n_buy_signals = 0
        self.n_sell_signals = 0
        self.n_buy_traded = 0
        self.n_sell_traded = 0

    # ----------------------------------------------------------
    # 初始化：加载模型
    # ----------------------------------------------------------

    def on_chan_init(self):
        """加载买点模型和卖点模型。"""
        # 买点模型
        self._model_buy = xgb.Booster()
        self._model_buy.load_model(MODEL_BUY_PATH)
        with open(META_BUY_PATH, "r") as f:
            self._meta_buy = json.load(f)
        self._fnames_buy = [""] * len(self._meta_buy)
        for name, idx in self._meta_buy.items():
            self._fnames_buy[idx] = name

        # 卖点模型
        self._model_sell = xgb.Booster()
        self._model_sell.load_model(MODEL_SELL_PATH)
        with open(META_SELL_PATH, "r") as f:
            self._meta_sell = json.load(f)
        self._fnames_sell = [""] * len(self._meta_sell)
        for name, idx in self._meta_sell.items():
            self._fnames_sell[idx] = name

        print(f"  [XGB] 买点模型已加载  特征数={len(self._meta_buy)}")
        print(f"  [XGB] 卖点模型已加载  特征数={len(self._meta_sell)}")
        print(f"  [XGB] 信号阈值={self.signal_threshold:.0%}")

    # ----------------------------------------------------------
    # 买卖点回调：特征构建 → 模型预测 → 下单
    # ----------------------------------------------------------

    def on_chan_bsp(self, chan, bsp_list, klu):
        """
        处理新买卖点：
          1. 用 build_features() 为 BSP 补充技术特征
          2. 调用对应模型预测合格概率
          3. 超过阈值才执行交易
        """
        cur_lv_chan = chan[0]

        for bsp in bsp_list:
            # 补充 feature_center 特征（内置 BSP 特征已由框架填充）
            extra = build_features(
                klu=klu,
                history=cur_lv_chan.lst,
                chan=cur_lv_chan,
            )
            bsp.features.add_feat(extra)

            bsp_main_type = bsp.type[0].value[0]   # '1' / '2' / '3'

            if bsp.is_buy:
                self.n_buy_signals += 1
                prob = _predict_bsp_quality(
                    bsp, bsp_main_type,
                    self._model_buy, self._meta_buy, self._fnames_buy,
                )
                qualified = prob >= self.signal_threshold
                self._log_signal("BUY", klu, bsp_main_type, prob, qualified)

                if qualified and self.position == 0:
                    self.buy(
                        klu.close,
                        self.cash / klu.close,
                        comment=f"XGB买入 T{bsp_main_type} p={prob:.2%}",
                    )
                    self.n_buy_traded += 1

            else:
                self.n_sell_signals += 1
                prob = _predict_bsp_quality(
                    bsp, bsp_main_type,
                    self._model_sell, self._meta_sell, self._fnames_sell,
                )
                qualified = prob >= self.signal_threshold
                self._log_signal("SELL", klu, bsp_main_type, prob, qualified)

                if qualified and self.position > 0:
                    self.close_position(
                        klu.close,
                        comment=f"XGB卖出 T{bsp_main_type} p={prob:.2%}",
                    )
                    self.n_sell_traded += 1

    def on_backtest_end(self):
        """回测结束时强制平掉剩余仓位。"""
        if self.position > 0 and self.equity_curve:
            last_price = self.equity_curve[-1]["price"]
            self.close_position(last_price, comment="回测结束平仓")

    # ----------------------------------------------------------
    # 辅助：打印信号日志
    # ----------------------------------------------------------

    @staticmethod
    def _log_signal(direction, klu, bsp_type, prob, qualified):
        tag = "✓" if qualified else "✗"
        action = ("→ 买入" if direction == "BUY" else "→ 卖出") if qualified else ""
        print(
            f"  {tag} {direction:4s} {klu.time}  T{bsp_type}  "
            f"prob={prob:.2%}  {action}"
        )

    # ----------------------------------------------------------
    # 绩效打印扩展
    # ----------------------------------------------------------

    def print_signal_stats(self):
        print(f"\n{'─' * 50}")
        print(f"  信号统计")
        print(f"{'─' * 50}")
        print(f"  买点信号: {self.n_buy_signals}  →  实际买入: {self.n_buy_traded}")
        print(f"  卖点信号: {self.n_sell_signals}  →  实际卖出: {self.n_sell_traded}")
        total = self.n_buy_signals + self.n_sell_signals
        traded = self.n_buy_traded + self.n_sell_traded
        if total > 0:
            print(f"  信号过滤率: {1 - traded / total:.1%}  "
                  f"(阈值={self.signal_threshold:.0%})")


# ============================================================
# Baseline 对照策略：全量缠论信号（不过滤）
# ============================================================

class ChanBaselineStrategy(ChanStrategyBase):
    """
    对照策略：不使用 XGBoost 过滤，所有 1/2/3 类买卖点直接执行。
    用于与 XGBChanStrategy 比较，量化 XGBoost 过滤的价值。
    """

    def __init__(self):
        super().__init__(chan_config_dict={**CHAN_CONFIG, "print_warning": False})

    def on_chan_bsp(self, chan, bsp_list, klu):
        for bsp in bsp_list:
            if bsp.is_buy and self.position == 0:
                self.buy(klu.close, self.cash / klu.close,
                         comment=f"原始买入 T{bsp.type[0].value[0]}")
            elif not bsp.is_buy and self.position > 0:
                self.close_position(klu.close,
                                    comment=f"原始卖出 T{bsp.type[0].value[0]}")

    def on_backtest_end(self):
        if self.position > 0 and self.equity_curve:
            self.close_position(self.equity_curve[-1]["price"], comment="回测结束平仓")


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    import os
    os.makedirs("result", exist_ok=True)

    print("\n" + "★" * 60)
    print("   XGBoost 质量过滤 × 缠论买卖点回测")
    print(f"   标的: {CODE}  区间: {BACKTEST_BEGIN} ~ {BACKTEST_END}")
    print(f"   初始资金: {INITIAL_CASH:,.0f}  信号阈值: {SIGNAL_THRESHOLD:.0%}")
    print("★" * 60)

    # ── 主策略：XGBoost 过滤 ──────────────────────────────────
    print("\n" + "▓" * 60)
    print("  [策略A] XGBoost 质量过滤策略")
    print("▓" * 60)

    xgb_strategy = XGBChanStrategy(signal_threshold=SIGNAL_THRESHOLD)
    _ = run_chan_backtest_no_vnpy(
        strategy_instance=xgb_strategy,
        code=CODE,
        begin_time=BACKTEST_BEGIN,
        end_time=BACKTEST_END,
        kl_type=KL_TYPE.K_15M,
        initial_cash=INITIAL_CASH,
        data_src=DATA_SRC.PARQUET,
        autype=AUTYPE.NONE,
        chan_config_override=CHAN_CONFIG,
    )
    xgb_strategy.print_signal_stats()

    # ── 对照策略：全量缠论信号 ───────────────────────────────
    print("\n" + "▓" * 60)
    print("  [策略B] 全量缠论信号（基准对照）")
    print("▓" * 60)

    baseline = ChanBaselineStrategy()
    _ = run_chan_backtest_no_vnpy(
        strategy_instance=baseline,
        code=CODE,
        begin_time=BACKTEST_BEGIN,
        end_time=BACKTEST_END,
        kl_type=KL_TYPE.K_15M,
        initial_cash=INITIAL_CASH,
        data_src=DATA_SRC.PARQUET,
        autype=AUTYPE.NONE,
        chan_config_override=CHAN_CONFIG,
        signal_threshold=0.0,
    )

    # ── 对比摘要 ──────────────────────────────────────────────
    xgb_perf  = xgb_strategy.get_performance()
    base_perf = baseline.get_performance()

    print(f"\n{'═' * 60}")
    print(f"{'策略对比摘要':^56}")
    print(f"{'═' * 60}")
    print(f"  {'指标':<20} {'XGBoost过滤':>16} {'全量信号(基准)':>16}")
    print(f"  {'─' * 54}")

    metrics = [
        ("总收益率",        "total_return_pct"),
        ("年化收益率",      "annualized_return_pct"),
        ("最大回撤",        "max_drawdown_pct"),
        ("胜率",            "win_rate_pct"),
        ("盈亏比",          "profit_factor"),
        ("总交易次数",      "total_trades"),
        ("总盈亏",          "total_pnl"),
    ]
    for label, key in metrics:
        xv = xgb_perf.get(key, "N/A")
        bv = base_perf.get(key, "N/A")
        if isinstance(xv, float):
            xv_str = f"{xv:.4f}"
            bv_str = f"{bv:.4f}" if isinstance(bv, float) else str(bv)
        else:
            xv_str = str(xv)
            bv_str = str(bv)
        print(f"  {label:<20} {xv_str:>16} {bv_str:>16}")

    print(f"{'═' * 60}")

    # ── 生成 HTML 报告 ────────────────────────────────────────
    xgb_report_path  = "result/xgb_backtest_report.html"
    base_report_path = "result/baseline_backtest_report.html"

    xgb_strategy.generate_report(
        path=xgb_report_path,
        title=f"XGBoost 缠论策略 · {CODE} · {BACKTEST_BEGIN}~{BACKTEST_END}",
    )
    baseline.generate_report(
        path=base_report_path,
        title=f"全量信号基准策略 · {CODE} · {BACKTEST_BEGIN}~{BACKTEST_END}",
    )

    print(f"\n  [XGB策略报告]  {xgb_report_path}")
    print(f"  [基准策略报告] {base_report_path}")
    print(f"\n{'★' * 60}")
