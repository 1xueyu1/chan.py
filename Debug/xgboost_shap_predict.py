"""
XGBoost + SHAP V3 实时推理与单样本解释 — 双模型 (买点/卖点质量评估)
============================================================
配套 xgboost_shap_train.py V3:
  - 双模型: 买点使用买点模型, 卖点使用卖点模型
  - 支持 BSP 类型分类 (1/2/3类独热编码)
  - 使用过滤后的特征集 (与训练一致)
  - 预测含义: 该买卖点是否“合格” (值得操作)
============================================================
"""

import json
import os
import time
from typing import Dict, List

import numpy as np
import xgboost as xgb

from BuySellPoint.BS_Point import CBS_Point
from Chan import CChan
from ChanConfig import CChanConfig
from ChanModel.Features import CFeatures
from ChanModel.feature_center import build_features
from ChanModel.shap_analyzer import SHAPAnalyzer, SHAPResult
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE
from Common.CTime import CTime


# ============================================================
# 预测结果类
# ============================================================

class PredictionResult:
    """单个 BSP 的预测结果"""

    def __init__(self, bsp_time, is_buy, probability, shap_explanation,
                 feature_values, bsp_main_type, bsp_types_str):
        self.bsp_time = bsp_time
        self.is_buy = is_buy
        self.probability = probability
        self.confidence = self._calc_confidence(probability)
        self.shap_explanation = shap_explanation
        self.feature_values = feature_values
        self.bsp_main_type = bsp_main_type
        self.bsp_types_str = bsp_types_str
        self.qualified = probability >= SIGNAL_THRESHOLD
        self.action = self._get_action()

    def _get_action(self):
        """根据买卖点方向和合格概率给出交易动作建议"""
        if not self.qualified:
            return "观望"
        if self.is_buy:
            return "买入/平空"  # 合格买点 → 开多或平空
        else:
            return "卖出/平多"  # 合格卖点 → 开空或平多

    @staticmethod
    def _calc_confidence(prob):
        if prob >= 0.8:
            return "极高"
        elif prob >= 0.6:
            return "较高"
        elif prob >= 0.4:
            return "中等"
        elif prob >= 0.2:
            return "较低"
        else:
            return "极低"


# ============================================================
# 配置区
# ============================================================

CODE = "BTCUSDT"
BEGIN_TIME = "2025-01-01"
END_TIME = "2026-02-07"
DATA_SRC_TYPE = DATA_SRC.CSV
LV_LIST = [KL_TYPE.K_15M]

MODEL_BUY_PATH = "Debug/model_buy.json"
MODEL_SELL_PATH = "Debug/model_sell.json"
META_BUY_PATH = "Debug/meta_buy.json"
META_SELL_PATH = "Debug/meta_sell.json"

# 缠论配置 (必须与训练一致!)
CHAN_CONFIG = {
    "trigger_step": True,
    "bi_strict": True,
    "skip_step": 0,
    "divergence_rate": float("inf"),
    "bsp2_follow_1": False,
    "bsp3_follow_1": False,
    "min_zs_cnt": 0,
    "bs1_peak": False,
    "macd_algo": "peak",
    "bs_type": "1,2,3a,1p,2s,3b",
    "print_warning": True,
    "zs_algo": "normal",
}

OUTPUT_DIR = "Debug"
PREDICT_REPORT_PATH = os.path.join(OUTPUT_DIR, "shap_predict_report.html")

# 置信度阈值
SIGNAL_THRESHOLD = 0.55


# ============================================================
# 预测函数
# ============================================================

def predict_with_shap(model, analyzer, last_bsp, meta, feature_names,
                      bsp_main_type):
    """
    对单个 BSP 进行 XGBoost 预测 + SHAP 解释 (V3: 无 is_buy_signal)

    Parameters
    ----------
    bsp_main_type : str  ('1', '2', '3')
    """
    missing = np.nan
    n_features = len(meta)
    feature_arr = np.full(n_features, missing)

    # BSP 类型独热
    if "bsp_type_1" in meta:
        feature_arr[meta["bsp_type_1"]] = 1.0 if bsp_main_type == '1' else 0.0
    if "bsp_type_2" in meta:
        feature_arr[meta["bsp_type_2"]] = 1.0 if bsp_main_type == '2' else 0.0
    if "bsp_type_3" in meta:
        feature_arr[meta["bsp_type_3"]] = 1.0 if bsp_main_type == '3' else 0.0

    # feature_center + 内置特征
    for feat_name, feat_value in last_bsp.features.items():
        if feat_name in meta:
            feature_arr[meta[feat_name]] = feat_value

    # 预测
    feature_2d = feature_arr.reshape(1, -1)
    dtest = xgb.DMatrix(feature_2d, feature_names=feature_names, missing=np.nan)
    prob = float(model.predict(dtest)[0])

    # SHAP 解释
    result = analyzer.analyze(feature_2d)
    explanation = analyzer.explain_single(result, idx=0)

    return prob, explanation, feature_arr


# ============================================================
# 报告生成
# ============================================================

def generate_predict_report(predictions, output_path):
    """生成推理结果 HTML 报告"""
    import datetime
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    n_total = len(predictions)
    n_buy = sum(1 for p in predictions if p.is_buy)
    n_sell = n_total - n_buy
    n_qualified = sum(1 for p in predictions if p.qualified)
    n_buy_qualified = sum(1 for p in predictions if p.is_buy and p.qualified)
    n_sell_qualified = sum(1 for p in predictions if not p.is_buy and p.qualified)
    avg_prob = np.mean([p.probability for p in predictions]) if predictions else 0

    # 按 BSP 类型统计
    type_stats = {}
    for t in ['1', '2', '3']:
        sub = [p for p in predictions if p.bsp_main_type == t]
        if sub:
            type_stats[t] = {
                "count": len(sub),
                "avg_prob": np.mean([p.probability for p in sub]),
                "qualified": sum(1 for p in sub if p.qualified),
            }

    type_cards = ""
    for t, stats in type_stats.items():
        type_cards += (
            f'<div class="metric-card">'
            f'<div class="value">{stats["count"]}</div>'
            f'<div class="label">{t}类信号 (合格:{stats["qualified"]})</div>'
            f'</div>'
        )

    # 信号表格
    rows = ""
    for i, pred in enumerate(predictions):
        direction = "买入" if pred.is_buy else "卖出"
        dir_class = "tag-buy" if pred.is_buy else "tag-sell"
        type_label = f"{pred.bsp_main_type}类"

        prob_color = "#28a745" if pred.probability >= 0.6 else (
            "#dc3545" if pred.probability < 0.4 else "#fd7e14"
        )

        top_pos_str = "<br>".join(
            [f"<span style='color:#dc3545'>▲ {n}: {s:+.4f}</span>"
             for n, s, v in pred.shap_explanation.get("top_positive", [])[:3]]
        )
        top_neg_str = "<br>".join(
            [f"<span style='color:#28a745'>▼ {n}: {s:+.4f}</span>"
             for n, s, v in pred.shap_explanation.get("top_negative", [])[:3]]
        )

        row_class = ' class="signal-strong"' if pred.probability >= SIGNAL_THRESHOLD else ''

        action_class = 'tag-buy' if pred.is_buy and pred.qualified else (
            'tag-sell' if not pred.is_buy and pred.qualified else ''
        )
        action_label = pred.action

        rows += f"""
        <tr{row_class}>
            <td>{i + 1}</td>
            <td><strong>{pred.bsp_time}</strong></td>
            <td><span class="tag {dir_class}">{direction}</span></td>
            <td><span class="tag">{type_label}</span></td>
            <td style="color:{prob_color};font-weight:bold;">{pred.probability:.2%}</td>
            <td>{pred.confidence}</td>
            <td><span class="tag {action_class}">{action_label}</span></td>
            <td>{top_pos_str}</td>
            <td>{top_neg_str}</td>
        </tr>"""

    html = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>XGBoost + SHAP V3 推理报告</title>
<style>
    * {{ margin: 0; padding: 0; box-sizing: border-box; }}
    body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
           background: #f8f9fa; color: #333; line-height: 1.6; }}
    .container {{ max-width: 1600px; margin: 0 auto; padding: 20px; }}
    .header {{ background: linear-gradient(135deg, #0f3460 0%, #16213e 50%, #1a1a2e 100%);
              color: white; padding: 30px; border-radius: 12px; margin-bottom: 24px;
              text-align: center; }}
    .header h1 {{ font-size: 1.8em; margin-bottom: 8px; }}
    .header .subtitle {{ opacity: 0.8; }}
    .section {{ background: white; border-radius: 10px; padding: 24px;
               margin-bottom: 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }}
    .section h2 {{ font-size: 1.3em; color: #1a1a2e; border-bottom: 3px solid #0f3460;
                   padding-bottom: 8px; margin-bottom: 16px; }}
    .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
                    gap: 12px; margin: 16px 0; }}
    .metric-card {{ background: #f0f4ff; border-radius: 8px; padding: 14px; text-align: center;
                   border: 1px solid #d0d8f0; }}
    .metric-card .value {{ font-size: 1.6em; font-weight: bold; color: #0f3460; }}
    .metric-card .label {{ font-size: 0.85em; color: #666; margin-top: 4px; }}
    .tag {{ display: inline-block; padding: 2px 8px; border-radius: 4px;
           font-size: 0.85em; font-weight: 600; }}
    .tag-buy {{ background: #d4edda; color: #155724; }}
    .tag-sell {{ background: #f8d7da; color: #721c24; }}
    table {{ width: 100%; border-collapse: collapse; }}
    th {{ background: #f0f4ff; padding: 10px 8px; text-align: left;
         border-bottom: 2px solid #d0d8f0; font-weight: 600; font-size: 0.9em; }}
    td {{ padding: 8px; border-bottom: 1px solid #eee; font-size: 0.88em; vertical-align: top; }}
    tr:hover {{ background: #f8f9ff; }}
    .signal-strong {{ background: #d4edda; }}
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>XGBoost + SHAP V3 推理报告 (双模型: 买点/卖点质量评估)</h1>
        <div class="subtitle">{CODE} · {BEGIN_TIME} ~ {END_TIME} · 生成: {now} · 含BSP类型分类</div>
    </div>

    <div class="section">
        <h2>信号概览</h2>
        <div class="metrics-grid">
            <div class="metric-card"><div class="value">{n_total}</div><div class="label">总信号数</div></div>
            <div class="metric-card"><div class="value">{n_buy}</div><div class="label">买点信号</div></div>
            <div class="metric-card"><div class="value">{n_sell}</div><div class="label">卖点信号</div></div>
            <div class="metric-card"><div class="value">{n_qualified}</div><div class="label">合格信号 (≥{SIGNAL_THRESHOLD:.0%})</div></div>
            <div class="metric-card"><div class="value">{n_buy_qualified}</div><div class="label">合格买点 (买入/平空)</div></div>
            <div class="metric-card"><div class="value">{n_sell_qualified}</div><div class="label">合格卖点 (卖出/平多)</div></div>
            <div class="metric-card"><div class="value">{avg_prob:.1%}</div><div class="label">平均合格概率</div></div>
            {type_cards}
        </div>
    </div>

    <div class="section">
        <h2>详细信号列表</h2>
        <table>
            <tr>
                <th>#</th>
                <th>时间</th>
                <th>方向</th>
                <th>BSP类型</th>
                <th>合格概率</th>
                <th>置信度</th>
                <th>建议操作</th>
                <th>正向贡献 (Top-3)</th>
                <th>负向贡献 (Top-3)</th>
            </tr>
            {rows}
        </table>
    </div>

    <div class="section" style="text-align:center; color:#999; font-size:0.85em;">
        Powered by XGBoost + SHAP V3 · 买点模型: {MODEL_BUY_PATH} · 卖点模型: {MODEL_SELL_PATH}
    </div>
</div>
</body>
</html>'''

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"[Report] 推理报告: {output_path}")


# ============================================================
# 主流程
# ============================================================

if __name__ == "__main__":
    start_time = time.time()

    print("\n" + "★" * 60)
    print("   XGBoost + SHAP V3 实时推理 (双模型: 买点/卖点质量评估)")
    print("★" * 60)

    # 1. 加载买点模型+Meta
    print(f"\n[1] 加载买点模型: {MODEL_BUY_PATH}")
    model_buy = xgb.Booster()
    model_buy.load_model(MODEL_BUY_PATH)

    with open(META_BUY_PATH, "r") as f:
        meta_buy: Dict[str, int] = json.load(f)
    fnames_buy = [""] * len(meta_buy)
    for name, idx in meta_buy.items():
        fnames_buy[idx] = name
    print(f"    买点特征数: {len(meta_buy)}")

    analyzer_buy = SHAPAnalyzer(model=model_buy, feature_names=fnames_buy)

    # 加载卖点模型+Meta
    print(f"    加载卖点模型: {MODEL_SELL_PATH}")
    model_sell = xgb.Booster()
    model_sell.load_model(MODEL_SELL_PATH)

    with open(META_SELL_PATH, "r") as f:
        meta_sell: Dict[str, int] = json.load(f)
    fnames_sell = [""] * len(meta_sell)
    for name, idx in meta_sell.items():
        fnames_sell[idx] = name
    print(f"    卖点特征数: {len(meta_sell)}")

    analyzer_sell = SHAPAnalyzer(model=model_sell, feature_names=fnames_sell)

    # 2. 缠论初始化
    print(f"\n[2] 缠论分析: {CODE} {BEGIN_TIME}~{END_TIME}")
    config = CChanConfig(CHAN_CONFIG)
    chan = CChan(
        code=CODE, begin_time=BEGIN_TIME, end_time=END_TIME,
        data_src=DATA_SRC_TYPE, lv_list=LV_LIST, config=config, autype=AUTYPE.QFQ,
    )

    # 3. 逐步推理
    print("\n[3] 逐步推理 ...")
    treated_bsp_idx = set()
    predictions: List[PredictionResult] = []
    step_count = 0

    for chan_snapshot in chan.step_load():
        step_count += 1

        last_klu = chan_snapshot[0][-1][-1]
        bsp_list = chan_snapshot.get_latest_bsp()
        if not bsp_list:
            continue

        last_bsp = bsp_list[0]
        cur_lv_chan = chan_snapshot[0]

        if last_bsp.klu.idx in treated_bsp_idx:
            continue
        if cur_lv_chan[-2].idx != last_bsp.klu.klc.idx:
            continue

        # BSP 类型
        bsp_main_type = last_bsp.type[0].value[0]
        bsp_types_str = last_bsp.type2str()

        # 补充特征
        extra_feat = build_features(
            klu=last_klu,
            history=cur_lv_chan.lst,
            chan=cur_lv_chan,
            bsp=last_bsp,
        )
        last_bsp.features.add_feat(extra_feat)

        # 根据方向选择对应模型
        if last_bsp.is_buy:
            model_use = model_buy
            analyzer_use = analyzer_buy
            meta_use = meta_buy
            fnames_use = fnames_buy
        else:
            model_use = model_sell
            analyzer_use = analyzer_sell
            meta_use = meta_sell
            fnames_use = fnames_sell

        # 预测
        prob, explanation, feature_arr = predict_with_shap(
            model_use, analyzer_use, last_bsp, meta_use, fnames_use,
            bsp_main_type=bsp_main_type,
        )

        result = PredictionResult(
            bsp_time=str(last_bsp.klu.time),
            is_buy=last_bsp.is_buy,
            probability=prob,
            shap_explanation=explanation,
            feature_values={n: float(v) for n, v in last_bsp.features.items()},
            bsp_main_type=bsp_main_type,
            bsp_types_str=bsp_types_str,
        )
        predictions.append(result)

        # 控制台输出
        direction = "🟢买" if last_bsp.is_buy else "🔴卖"
        model_tag = "买点模型" if last_bsp.is_buy else "卖点模型"
        qualified = "✓合格" if result.qualified else "✗不合格"
        type_tag = f"T{bsp_main_type}"
        action_str = f"→{result.action}" if result.qualified else ""
        print(
            f"  {result.bsp_time} | {direction} {type_tag} [{model_tag}] | "
            f"P={prob:.2%} [{result.confidence}] {qualified} {action_str} | "
            f"Top+: {explanation['top_positive'][0][0] if explanation['top_positive'] else 'N/A'} "
            f"Top-: {explanation['top_negative'][-1][0] if explanation['top_negative'] else 'N/A'}"
        )

        treated_bsp_idx.add(last_bsp.klu.idx)

    # 4. 报告
    print("\n[4] 生成报告 ...")
    generate_predict_report(predictions, PREDICT_REPORT_PATH)

    # 5. 汇总
    total_time = time.time() - start_time
    n_qualified = sum(1 for p in predictions if p.qualified)
    n_buy_q = sum(1 for p in predictions if p.is_buy and p.qualified)
    n_sell_q = sum(1 for p in predictions if not p.is_buy and p.qualified)

    # 按类型统计
    from collections import Counter
    type_counter = Counter(p.bsp_main_type for p in predictions)

    print(f"\n{'★' * 60}")
    print("   推理完成!")
    print(f"   总信号: {len(predictions)}  合格: {n_qualified}  耗时: {total_time:.1f}s")
    print(f"   买点: {sum(1 for p in predictions if p.is_buy)} "
          f"(合格:{n_buy_q} → 买入/平空)")
    print(f"   卖点: {sum(1 for p in predictions if not p.is_buy)} "
          f"(合格:{n_sell_q} → 卖出/平多)")
    print(f"   按类型: {dict(type_counter)}")
    print(f"   报告: {PREDICT_REPORT_PATH}")
    print(f"{'★' * 60}")
