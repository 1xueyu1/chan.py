# shap_analyzer.py
# ============================================================
# XGBoost + SHAP 可解释性分析模块
#
# 提供:
#   - SHAPAnalyzer: 基于 TreeExplainer 的 SHAP 分析器
#   - 全局特征重要性 (summary / bar / dependence)
#   - 单样本决策解释 (force / waterfall / decision)
#   - 交互式 HTML 报告生成 (Plotly + 内联 base64 图片)
#
# 依赖: xgboost, shap, numpy, pandas, matplotlib, plotly
# ============================================================

from __future__ import annotations

import base64
import io
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")  # 无头后端，避免弹窗

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xgboost as xgb

try:
    import shap
except ImportError:
    raise ImportError("请先安装 shap: pip install shap")


# ============================================================
# 工具函数
# ============================================================

def _fig_to_base64(fig: plt.Figure, dpi: int = 120) -> str:
    """将 matplotlib Figure 编码为 base64 PNG 字符串"""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return b64


def _make_img_tag(b64: str, alt: str = "", width: str = "100%") -> str:
    return f'<img src="data:image/png;base64,{b64}" alt="{alt}" style="width:{width};max-width:1200px;">'


# ============================================================
# 数据结构
# ============================================================

@dataclass
class SHAPResult:
    """SHAP 分析结果容器"""
    shap_values: np.ndarray          # (n_samples, n_features) SHAP 值
    base_value: float                # 期望值 (base value)
    feature_names: List[str]         # 特征名列表
    X: np.ndarray                    # 原始特征矩阵
    y: Optional[np.ndarray] = None   # 标签

    # 全局重要性（按 mean |SHAP| 排名）
    importance_df: pd.DataFrame = field(default_factory=pd.DataFrame)

    def compute_importance(self) -> pd.DataFrame:
        """计算全局特征重要性排名"""
        mean_abs = np.mean(np.abs(self.shap_values), axis=0)
        df = pd.DataFrame({
            "feature": self.feature_names,
            "mean_abs_shap": mean_abs,
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        df["rank"] = range(1, len(df) + 1)
        self.importance_df = df
        return df


# ============================================================
# 核心分析器
# ============================================================

class SHAPAnalyzer:
    """
    XGBoost + SHAP 可解释性分析器

    用法::

        analyzer = SHAPAnalyzer(model, feature_names)
        result = analyzer.analyze(X_train, y_train)

        # 生成 HTML 报告
        analyzer.generate_report(result, "shap_report.html")

        # 单样本解释
        explanation = analyzer.explain_single(result, idx=0)
    """

    def __init__(self, model: xgb.Booster, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer: Optional[shap.TreeExplainer] = None

    # ----------------------------------------------------------
    # 核心: 计算 SHAP 值
    # ----------------------------------------------------------

    def analyze(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
    ) -> SHAPResult:
        """
        对特征矩阵 X 计算 SHAP 值

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, optional, shape (n_samples,)

        Returns
        -------
        SHAPResult
        """
        # 构建 DMatrix 供 TreeExplainer 使用
        dmat = xgb.DMatrix(X, feature_names=self.feature_names)

        # 创建 TreeExplainer（对于二分类，输出 log-odds 空间的 SHAP 值）
        self.explainer = shap.TreeExplainer(self.model)
        shap_values = self.explainer.shap_values(dmat)

        # shap_values 可能是 list（多分类）或 ndarray（二分类/多分类）
        if isinstance(shap_values, list):
            # 多分类 list：默认取最后一类（训练脚本中约定为 PT(+1)）
            shap_values = shap_values[-1]

        shap_values = np.array(shap_values)
        if shap_values.ndim == 3:
            n_feat = len(self.feature_names)
            # 常见布局1: (n_samples, n_features, n_classes)
            if shap_values.shape[1] == n_feat:
                shap_values = shap_values[:, :, -1]
            # 常见布局2: (n_classes, n_samples, n_features)
            elif shap_values.shape[2] == n_feat and shap_values.shape[0] <= 10:
                shap_values = shap_values[-1, :, :]
            # 常见布局3: (n_samples, n_classes, n_features)
            elif shap_values.shape[2] == n_feat and shap_values.shape[1] <= 10:
                shap_values = shap_values[:, -1, :]
            else:
                raise ValueError(
                    f"Unsupported SHAP 3D shape: {shap_values.shape}"
                )

        expected_value = self.explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            base_value = float(expected_value[-1])
        else:
            base_value = float(expected_value)

        result = SHAPResult(
            shap_values=np.array(shap_values),
            base_value=base_value,
            feature_names=self.feature_names,
            X=X,
            y=y,
        )
        result.compute_importance()
        return result

    # ----------------------------------------------------------
    # 绘图: 全局特征重要性
    # ----------------------------------------------------------

    def plot_summary_bar(self, result: SHAPResult, max_display: int = 20) -> str:
        """SHAP 全局重要性柱状图 → base64 PNG"""
        fig, ax = plt.subplots(figsize=(10, max(6, max_display * 0.3)))
        shap.summary_plot(
            result.shap_values, result.X,
            feature_names=result.feature_names,
            plot_type="bar", max_display=max_display,
            show=False,
        )
        fig = plt.gcf()
        fig.set_size_inches(10, max(6, max_display * 0.3))
        return _fig_to_base64(fig)

    def plot_summary_dot(self, result: SHAPResult, max_display: int = 20) -> str:
        """SHAP Summary Dot Plot (蜂群图) → base64 PNG"""
        fig, ax = plt.subplots(figsize=(10, max(6, max_display * 0.3)))
        shap.summary_plot(
            result.shap_values, result.X,
            feature_names=result.feature_names,
            plot_type="dot", max_display=max_display,
            show=False,
        )
        fig = plt.gcf()
        fig.set_size_inches(10, max(6, max_display * 0.3))
        return _fig_to_base64(fig)

    def plot_dependence(
        self, result: SHAPResult, feature: str,
        interaction_feature: Optional[str] = "auto",
    ) -> str:
        """SHAP Dependence Plot → base64 PNG"""
        fig, ax = plt.subplots(figsize=(8, 5))
        shap.dependence_plot(
            feature, result.shap_values, result.X,
            feature_names=result.feature_names,
            interaction_index=interaction_feature,
            ax=ax, show=False,
        )
        return _fig_to_base64(fig)

    # ----------------------------------------------------------
    # 绘图: 单样本解释
    # ----------------------------------------------------------

    def plot_waterfall(self, result: SHAPResult, idx: int = 0) -> str:
        """SHAP Waterfall Plot (瀑布图) 单样本 → base64 PNG"""
        explanation = shap.Explanation(
            values=result.shap_values[idx],
            base_values=result.base_value,
            data=result.X[idx],
            feature_names=result.feature_names,
        )
        fig = plt.figure(figsize=(10, 6))
        shap.plots.waterfall(explanation, show=False)
        fig = plt.gcf()
        return _fig_to_base64(fig)

    def plot_force_single(self, result: SHAPResult, idx: int = 0) -> str:
        """SHAP Force Plot (力图) 单样本 → HTML 字符串"""
        explanation = shap.Explanation(
            values=result.shap_values[idx],
            base_values=result.base_value,
            data=result.X[idx],
            feature_names=result.feature_names,
        )
        # 使用 matplotlib 方式生成 force plot
        fig = plt.figure(figsize=(16, 3))
        shap.plots.force(explanation, matplotlib=True, show=False)
        fig = plt.gcf()
        return _fig_to_base64(fig)

    # ----------------------------------------------------------
    # 绘图: Plotly 交互式
    # ----------------------------------------------------------

    def plotly_importance_bar(
        self, result: SHAPResult, max_display: int = 20,
    ) -> go.Figure:
        """Plotly 交互式特征重要性柱状图"""
        df = result.importance_df.head(max_display).iloc[::-1]

        fig = go.Figure(go.Bar(
            x=df["mean_abs_shap"],
            y=df["feature"],
            orientation="h",
            marker_color="#1f77b4",
        ))
        fig.update_layout(
            title=dict(text="SHAP 全局特征重要性 (Mean |SHAP|)", font=dict(size=16)),
            xaxis_title="Mean |SHAP value|",
            yaxis_title="Feature",
            height=max(400, max_display * 28),
            margin=dict(l=180),
            template="plotly_white",
        )
        return fig

    def plotly_shap_scatter(
        self, result: SHAPResult, feature: str,
    ) -> go.Figure:
        """Plotly 交互式 SHAP 值 vs 特征值散点图"""
        if feature not in result.feature_names:
            raise ValueError(f"特征 '{feature}' 不在特征列表中")

        idx = result.feature_names.index(feature)
        x_vals = result.X[:, idx]
        y_vals = result.shap_values[:, idx]

        colors = result.y if result.y is not None else y_vals

        fig = go.Figure(go.Scatter(
            x=x_vals, y=y_vals,
            mode="markers",
            marker=dict(
                size=5, color=colors,
                colorscale="RdBu_r", showscale=True,
                colorbar=dict(title="Label" if result.y is not None else "SHAP"),
            ),
            text=[f"SHAP={v:.4f}<br>{feature}={x:.4f}"
                  for x, v in zip(x_vals, y_vals)],
            hoverinfo="text",
        ))
        fig.update_layout(
            title=dict(text=f"SHAP Dependence: {feature}", font=dict(size=14)),
            xaxis_title=feature,
            yaxis_title="SHAP value",
            height=400,
            template="plotly_white",
        )
        return fig

    def plotly_shap_heatmap(
        self, result: SHAPResult, max_display: int = 20,
    ) -> go.Figure:
        """Plotly 热力图: 样本 × 特征的 SHAP 值"""
        top_features = result.importance_df.head(max_display)["feature"].tolist()
        indices = [result.feature_names.index(f) for f in top_features]

        heatmap_data = result.shap_values[:, indices]

        # 按 SHAP 值总和排序样本
        order = np.argsort(heatmap_data.sum(axis=1))
        heatmap_data = heatmap_data[order]

        fig = go.Figure(go.Heatmap(
            z=heatmap_data,
            x=top_features,
            y=list(range(len(order))),
            colorscale="RdBu_r",
            zmid=0,
            colorbar=dict(title="SHAP"),
        ))
        fig.update_layout(
            title=dict(text="SHAP 值热力图 (样本 × 特征)", font=dict(size=14)),
            xaxis_title="Feature",
            yaxis_title="Sample (sorted by SHAP sum)",
            height=max(500, len(order) * 0.3),
            template="plotly_white",
        )
        return fig

    # ----------------------------------------------------------
    # 单样本解释（字典输出）
    # ----------------------------------------------------------

    def explain_single(self, result: SHAPResult, idx: int) -> Dict:
        """
        解释单个样本的预测

        Returns
        -------
        dict with keys:
            prediction, base_value, top_positive, top_negative, shap_dict
        """
        sv = result.shap_values[idx]
        fv = result.X[idx]

        # 预估预测概率 sigmoid(base + sum_shap)
        logit = result.base_value + sv.sum()
        prob = 1.0 / (1.0 + np.exp(-logit))

        # 排序: 影响最大的正 / 负特征
        pairs = list(zip(result.feature_names, sv, fv))
        pairs_sorted = sorted(pairs, key=lambda t: t[1], reverse=True)

        top_pos = [(n, float(s), float(v)) for n, s, v in pairs_sorted if s > 0][:5]
        top_neg = [(n, float(s), float(v)) for n, s, v in pairs_sorted if s < 0][-5:]

        return {
            "prediction_prob": float(prob),
            "base_value": float(result.base_value),
            "sum_shap": float(sv.sum()),
            "top_positive": top_pos,  # [(name, shap, value), ...]
            "top_negative": top_neg,
            "shap_dict": {n: float(s) for n, s in zip(result.feature_names, sv)},
        }

    # ----------------------------------------------------------
    # HTML 报告生成
    # ----------------------------------------------------------

    def generate_report(
        self,
        result: SHAPResult,
        output_path: str = "shap_report.html",
        model_metrics: Optional[Dict] = None,
        top_dependence: int = 5,
        sample_indices: Optional[List[int]] = None,
    ) -> str:
        """
        生成完整的 SHAP 分析 HTML 报告

        Parameters
        ----------
        result : SHAPResult
        output_path : str
        model_metrics : dict, optional  模型评估指标 (accuracy, auc, ...)
        top_dependence : int  Top-N 特征显示 Dependence Plot
        sample_indices : list[int], optional  指定要展示的样本索引

        Returns
        -------
        str : 输出文件路径
        """
        html_parts = [self._report_header()]

        # 1. 模型概览
        html_parts.append(self._section_model_overview(result, model_metrics))

        # 2. 模型作用与训练结果分析
        html_parts.append(self._section_model_role_and_analysis(model_metrics))

        # 3. 全局特征重要性 (Plotly 交互)
        html_parts.append(self._section_global_importance(result))

        # 4. SHAP Summary (matplotlib)
        html_parts.append(self._section_summary_plots(result))

        # 5. Top-N Dependence Plots (Plotly)
        html_parts.append(self._section_dependence_plots(result, top_dependence))

        # 6. SHAP 热力图 (Plotly)
        html_parts.append(self._section_heatmap(result))

        # 7. 单样本解释 (Waterfall + Force)
        if sample_indices is None:
            # 选择预测最高 / 最低 / 中间各一个
            probs = self._predict_probs(result)
            sample_indices = self._auto_select_samples(probs)
        html_parts.append(self._section_sample_explanations(result, sample_indices))

        # 8. 特征重要性表格
        html_parts.append(self._section_importance_table(result))

        html_parts.append(self._report_footer())

        html = "\n".join(html_parts)

        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html)

        print(f"[SHAP Report] 已保存到: {output_path}")
        return output_path

    # ----------------------------------------------------------
    # 报告内部方法
    # ----------------------------------------------------------

    def _predict_probs(self, result: SHAPResult) -> np.ndarray:
        """从 SHAP 值反算预测概率"""
        logits = result.base_value + result.shap_values.sum(axis=1)
        return 1.0 / (1.0 + np.exp(-logits))

    def _auto_select_samples(self, probs: np.ndarray) -> List[int]:
        """自动选择有代表性的样本"""
        indices = []
        indices.append(int(np.argmax(probs)))   # 最高概率
        indices.append(int(np.argmin(probs)))   # 最低概率
        mid_idx = int(np.argmin(np.abs(probs - 0.5)))  # 最接近 0.5
        if mid_idx not in indices:
            indices.append(mid_idx)
        return indices

    def _report_header(self) -> str:
        return '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>XGBoost + SHAP 可解释性分析报告</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>
    * { margin: 0; padding: 0; box-sizing: border-box; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
           background: #f8f9fa; color: #333; line-height: 1.6; }
    .container { max-width: 1400px; margin: 0 auto; padding: 20px; }
    .header { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
              color: white; padding: 40px; border-radius: 12px; margin-bottom: 30px;
              text-align: center; }
    .header h1 { font-size: 2em; margin-bottom: 10px; }
    .header .subtitle { opacity: 0.8; font-size: 1.1em; }
    .section { background: white; border-radius: 10px; padding: 30px;
               margin-bottom: 24px; box-shadow: 0 2px 8px rgba(0,0,0,0.08); }
    .section h2 { font-size: 1.4em; color: #1a1a2e; border-bottom: 3px solid #0f3460;
                   padding-bottom: 10px; margin-bottom: 20px; }
    .section h3 { font-size: 1.1em; color: #16213e; margin: 16px 0 12px; }
    .metrics-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
                    gap: 16px; margin: 20px 0; }
    .metric-card { background: #f0f4ff; border-radius: 8px; padding: 16px; text-align: center;
                   border: 1px solid #d0d8f0; }
    .metric-card .value { font-size: 1.8em; font-weight: bold; color: #0f3460; }
    .metric-card .label { font-size: 0.9em; color: #666; margin-top: 4px; }
    .plot-container { margin: 16px 0; text-align: center; }
    .plot-container img { border-radius: 8px; box-shadow: 0 2px 6px rgba(0,0,0,0.1); }
    .explanation-card { background: #fafbfc; border: 1px solid #e0e4e8; border-radius: 8px;
                        padding: 20px; margin: 16px 0; }
    .explanation-card .prob { font-size: 1.4em; font-weight: bold; }
    .explanation-card .prob.high { color: #28a745; }
    .explanation-card .prob.low { color: #dc3545; }
    .explanation-card .prob.mid { color: #fd7e14; }
    .feat-table { width: 100%; border-collapse: collapse; margin-top: 12px; }
    .feat-table th { background: #f0f4ff; padding: 10px 12px; text-align: left;
                     border-bottom: 2px solid #d0d8f0; font-weight: 600; }
    .feat-table td { padding: 8px 12px; border-bottom: 1px solid #eee; }
    .feat-table tr:hover { background: #f8f9ff; }
    .positive { color: #dc3545; }
    .negative { color: #28a745; }
    .tag { display: inline-block; padding: 2px 8px; border-radius: 4px;
           font-size: 0.85em; font-weight: 600; }
    .tag-buy { background: #d4edda; color: #155724; }
    .tag-sell { background: #f8d7da; color: #721c24; }
    .two-col { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
    @media (max-width: 900px) { .two-col { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>XGBoost + SHAP 可解释性分析报告</h1>
        <div class="subtitle">缠论买卖点预测模型 · 特征归因分析</div>
    </div>
'''

    def _report_footer(self) -> str:
        import datetime
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f'''
    <div class="section" style="text-align:center; color:#999; font-size:0.9em;">
        报告生成时间: {now} &nbsp;|&nbsp; Powered by XGBoost + SHAP + Plotly
    </div>
</div>
</body>
</html>'''

    def _section_model_overview(
        self, result: SHAPResult, metrics: Optional[Dict],
    ) -> str:
        n_samples = len(result.X)
        n_features = len(result.feature_names)

        if result.y is not None:
            n_pos = int(np.sum(result.y == 1))
            n_neg = int(np.sum(result.y == 0))
            label_info = f"正样本: {n_pos} | 负样本: {n_neg} | 正例比: {n_pos/n_samples:.1%}"
        else:
            label_info = "N/A"

        cards = f'''
        <div class="metric-card"><div class="value">{n_samples}</div><div class="label">样本数量</div></div>
        <div class="metric-card"><div class="value">{n_features}</div><div class="label">特征数量</div></div>
        '''

        metric_name_map = {
            "oos_sharpe": "样本外夏普比率（OOS Sharpe）",
            "oos_precision": "样本外精确率（OOS Precision）",
            "oos_macro_f1": "样本外宏平均F1（OOS Macro-F1）",
            "primary_auc_ovr_macro": "主模型AUC（OVR宏平均）",
            "primary_logloss": "主模型对数损失（LogLoss）",
            "primary_signal_coverage": "信号覆盖率（Signal Coverage）",
            "primary_signal_precision": "信号精确率（Signal Precision）",
            "label_pt_ratio": "标签PT占比（Label PT Ratio）",
            "label_timeout_ratio": "标签Timeout占比（Label Timeout Ratio）",
            "label_sl_ratio": "标签SL占比（Label SL Ratio）",
        }

        if metrics:
            for name, val in metrics.items():
                label = metric_name_map.get(name, name)
                if isinstance(val, float):
                    cards += f'<div class="metric-card"><div class="value">{val:.4f}</div><div class="label">{label}</div></div>'
                else:
                    cards += f'<div class="metric-card"><div class="value">{val}</div><div class="label">{label}</div></div>'

        return f'''
    <div class="section">
        <h2>1. 模型概览</h2>
        <p>标签分布: {label_info}</p>
        <div class="metrics-grid">{cards}</div>
    </div>'''

    def _section_model_role_and_analysis(self, metrics: Optional[Dict]) -> str:
        metrics = metrics or {}

        signal_coverage = float(metrics.get("primary_signal_coverage", 0.0) or 0.0)
        signal_precision = float(metrics.get("primary_signal_precision", 0.0) or 0.0)
        oos_precision = float(metrics.get("oos_precision", 0.0) or 0.0)
        oos_macro_f1 = float(metrics.get("oos_macro_f1", 0.0) or 0.0)
        oos_sharpe = float(metrics.get("oos_sharpe", 0.0) or 0.0)
        auc_macro = float(metrics.get("primary_auc_ovr_macro", 0.0) or 0.0)
        logloss = float(metrics.get("primary_logloss", 0.0) or 0.0)

        if oos_sharpe > 0:
            sharpe_comment = "风险调整后收益为正，模型在当前样本区间具备可交易性。"
        elif oos_sharpe > -0.5:
            sharpe_comment = "风险收益接近盈亏平衡，建议结合阈值与交易成本进一步调优。"
        else:
            sharpe_comment = "风险调整后收益偏弱，建议优先优化标签定义、阈值与过滤逻辑。"

        if auc_macro >= 0.70:
            auc_comment = "AUC 辨别能力较好，类别区分有效。"
        elif auc_macro >= 0.60:
            auc_comment = "AUC 处于中等水平，可通过特征工程进一步提升。"
        else:
            auc_comment = "AUC 偏低，模型区分能力有限，建议检查样本与标签质量。"

        if signal_precision >= 0.70:
            precision_comment = "信号质量较高，误报相对可控。"
        elif signal_precision >= 0.55:
            precision_comment = "信号质量中等，需继续配合阈值优化。"
        else:
            precision_comment = "信号误报偏多，建议优化特征与阈值。"

        if signal_coverage >= 0.40:
            coverage_comment = "覆盖率较高，交易机会更充足。"
        elif signal_coverage >= 0.20:
            coverage_comment = "覆盖率中等，交易频率与质量较平衡。"
        else:
            coverage_comment = "覆盖率偏低，可能错失部分机会。"

        if logloss <= 0.60:
            logloss_comment = "概率输出质量较好。"
        elif logloss <= 0.80:
            logloss_comment = "概率输出中等，可继续优化校准。"
        else:
            logloss_comment = "概率输出偏弱，建议检查样本分布与模型稳定性。"

        return f'''
    <div class="section">
        <h2>2. 模型定位与训练结果分析</h2>
        <p><strong>框架定位：</strong>Primary 模型是第一层方向模型，负责把样本划分为 SL / TIMEOUT / PT 三类，输出后续 Meta 过滤所需的先验概率。它决定了策略的“候选信号池质量”和覆盖率上限。</p>
        <p><strong>本次结果解读：</strong></p>
        <ul style="padding-left:18px; margin:10px 0;">
            <li>样本外精确率={oos_precision:.4f}，样本外宏平均F1={oos_macro_f1:.4f}，反映分类稳定性与有效信号命中率。</li>
            <li>信号覆盖率={signal_coverage:.4f}（{coverage_comment}），信号精确率={signal_precision:.4f}（{precision_comment}）。</li>
            <li>主模型AUC（OVR宏平均）={auc_macro:.4f}，主模型LogLoss={logloss:.4f}。{auc_comment} {logloss_comment}</li>
            <li>样本外夏普比率={oos_sharpe:.4f}。{sharpe_comment}</li>
        </ul>

        <h3>指标释义（Primary层）</h3>
        <ul style="padding-left:18px; margin:10px 0;">
            <li><strong>样本外精确率（OOS Precision）：</strong>在未见样本中，模型给出的有效信号有多少是真有效，直接影响实盘信号可信度。</li>
            <li><strong>样本外宏平均F1（OOS Macro-F1）：</strong>综合精确率与召回率，并对各类更均衡，适合评估多分类稳定性。</li>
            <li><strong>信号覆盖率（Signal Coverage）：</strong>模型最终给出交易信号的比例，决定策略交易频率。</li>
            <li><strong>信号精确率（Signal Precision）：</strong>被触发信号中的真实有效比例，决定“每笔交易质量”。</li>
            <li><strong>AUC（OVR宏平均）：</strong>模型区分不同类别的能力，越高说明排序和识别能力越强。</li>
            <li><strong>LogLoss：</strong>概率预测误差，越低越好；对“高置信但错误”惩罚更强。</li>
            <li><strong>样本外夏普比率（OOS Sharpe）：</strong>风险调整后的收益表现，是交易策略可用性的核心指标。</li>
        </ul>

        <p><strong>优化建议：</strong>若追求更稳健收益，优先联动优化三项：1) 标签质量（PT/SL/Timeout 配置）；2) Meta 阈值；3) 特征子集（基于 MDI+MDA+SHAP 交集）。</p>
    </div>'''

    def _section_global_importance(self, result: SHAPResult) -> str:
        """Plotly 交互式全局重要性"""
        fig = self.plotly_importance_bar(result)
        plotly_html = fig.to_html(
            full_html=False, include_plotlyjs=False, validate=False,
        )
        return f'''
    <div class="section">
        <h2>3. 全局特征重要性 (交互式)</h2>
        <p>基于 Mean |SHAP value| 的全局特征重要性排名，反映每个特征对模型预测的平均贡献。</p>
        <div class="plot-container">{plotly_html}</div>
    </div>'''

    def _section_summary_plots(self, result: SHAPResult) -> str:
        """Summary bar + dot plots"""
        bar_b64 = self.plot_summary_bar(result)
        dot_b64 = self.plot_summary_dot(result)
        return f'''
    <div class="section">
        <h2>4. SHAP Summary 图</h2>
        <div class="two-col">
            <div>
                <h3>特征重要性柱状图</h3>
                <div class="plot-container">
                    {_make_img_tag(bar_b64, "SHAP Bar Plot")}
                </div>
            </div>
            <div>
                <h3>蜂群图 (Beeswarm)</h3>
                <p>每个点代表一个样本。颜色表示特征值的高/低，水平位置代表 SHAP 值。</p>
                <div class="plot-container">
                    {_make_img_tag(dot_b64, "SHAP Dot Plot")}
                </div>
            </div>
        </div>
    </div>'''

    def _section_dependence_plots(self, result: SHAPResult, top_n: int) -> str:
        """Top-N 特征的 Dependence Plot"""
        top_feats = result.importance_df.head(top_n)["feature"].tolist()
        plots_html = ""
        for feat in top_feats:
            try:
                fig = self.plotly_shap_scatter(result, feat)
                plot_div = fig.to_html(
                    full_html=False, include_plotlyjs=False, validate=False,
                )
                plots_html += f'<div style="margin-bottom:16px;">{plot_div}</div>'
            except Exception as e:
                plots_html += f'<p style="color:red;">绘制 {feat} 失败: {e}</p>'

        return f'''
    <div class="section">
        <h2>5. SHAP Dependence Plots (Top-{top_n} 特征)</h2>
        <p>展示特征值与其 SHAP 贡献的关系，揭示模型如何利用每个特征做预测。</p>
        {plots_html}
    </div>'''

    def _section_heatmap(self, result: SHAPResult) -> str:
        """SHAP 热力图"""
        fig = self.plotly_shap_heatmap(result, max_display=15)
        plotly_html = fig.to_html(
            full_html=False, include_plotlyjs=False, validate=False,
        )
        return f'''
    <div class="section">
        <h2>6. SHAP 值热力图</h2>
        <p>每行代表一个样本，每列代表一个特征。颜色越红/蓝，SHAP 贡献越正/负。</p>
        <div class="plot-container">{plotly_html}</div>
    </div>'''

    def _section_sample_explanations(
        self, result: SHAPResult, sample_indices: List[int],
    ) -> str:
        """单样本解释 (Waterfall + Force + 文字)"""
        cards = ""

        for i, idx in enumerate(sample_indices):
            exp = self.explain_single(result, idx)
            prob = exp["prediction_prob"]

            # 颜色标记
            prob_class = "high" if prob > 0.6 else ("low" if prob < 0.4 else "mid")
            label_text = f"实际: {'✓ 有效' if result.y is not None and result.y[idx] == 1 else '✗ 无效'}" if result.y is not None else ""

            # Waterfall
            try:
                wf_b64 = self.plot_waterfall(result, idx)
                wf_html = _make_img_tag(wf_b64, f"Waterfall {idx}", "90%")
            except Exception:
                wf_html = "<p>Waterfall 图生成失败</p>"

            # Force
            try:
                force_b64 = self.plot_force_single(result, idx)
                force_html = _make_img_tag(force_b64, f"Force {idx}", "100%")
            except Exception:
                force_html = "<p>Force 图生成失败</p>"

            # Top 特征表
            rows = ""
            for name, shap_val, feat_val in exp["top_positive"]:
                rows += f'<tr><td>{name}</td><td>{feat_val:.4f}</td><td class="positive">+{shap_val:.4f}</td></tr>'
            for name, shap_val, feat_val in exp["top_negative"]:
                rows += f'<tr><td>{name}</td><td>{feat_val:.4f}</td><td class="negative">{shap_val:.4f}</td></tr>'

            cards += f'''
            <div class="explanation-card">
                <h3>样本 #{idx} — 预测概率: <span class="prob {prob_class}">{prob:.2%}</span> &nbsp; {label_text}</h3>
                <p>Base Value: {exp["base_value"]:.4f} &nbsp;|&nbsp; SHAP Sum: {exp["sum_shap"]:.4f}</p>

                <div class="two-col" style="margin-top:12px;">
                    <div>
                        <h3>Waterfall Plot</h3>
                        <div class="plot-container">{wf_html}</div>
                    </div>
                    <div>
                        <h3>Force Plot</h3>
                        <div class="plot-container">{force_html}</div>
                    </div>
                </div>

                <h3>关键特征贡献</h3>
                <table class="feat-table">
                    <tr><th>特征</th><th>特征值</th><th>SHAP 贡献</th></tr>
                    {rows}
                </table>
            </div>'''

        return f'''
    <div class="section">
        <h2>7. 单样本决策解释</h2>
        <p>选取代表性样本（最高置信 / 最低置信 / 中间），展示模型的决策过程。</p>
        {cards}
    </div>'''

    def _section_importance_table(self, result: SHAPResult) -> str:
        """特征重要性完整表格"""
        rows = ""
        for _, row in result.importance_df.iterrows():
            rows += f'''<tr>
                <td>{int(row["rank"])}</td>
                <td><strong>{row["feature"]}</strong></td>
                <td>{row["mean_abs_shap"]:.6f}</td>
            </tr>'''

        return f'''
    <div class="section">
        <h2>8. 特征重要性完整排名</h2>
        <table class="feat-table">
            <tr><th>#</th><th>特征名称</th><th>Mean |SHAP|</th></tr>
            {rows}
        </table>
    </div>'''
