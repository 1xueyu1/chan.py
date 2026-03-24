from __future__ import annotations

# flake8: noqa: E402, E501

import base64
import io
import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    img = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img


def _safe_metric(fn, default: float = 0.0) -> float:
    try:
        return float(fn())
    except Exception:
        return float(default)


def _threshold_rows(meta_y: np.ndarray, probs: np.ndarray) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    for th in [0.30, 0.40, 0.50, 0.55, 0.60, 0.70]:
        pred = (probs >= th).astype(int)
        keep_rate = float(np.mean(pred))
        rows.append(
            {
                "threshold": float(th),
                "keep_rate": keep_rate,
                "precision": _safe_metric(lambda: precision_score(meta_y, pred, zero_division=0)),
                "recall": _safe_metric(lambda: recall_score(meta_y, pred, zero_division=0)),
                "f1": _safe_metric(lambda: f1_score(meta_y, pred, zero_division=0)),
                "accuracy": _safe_metric(lambda: accuracy_score(meta_y, pred)),
            }
        )
    return rows


def _describe_metric_level(name: str, value: float, positive_rate: float = 0.0) -> str:
    """根据经验阈值给出中文解释，便于在报告中直观解读指标表现。"""
    if name == "roc_auc":
        if value >= 0.75:
            return "区分能力较强，模型能较好地区分可执行与不可执行信号。"
        if value >= 0.65:
            return "区分能力中等，仍有优化空间。"
        return "区分能力偏弱，建议优先优化特征与标签定义。"

    if name == "average_precision":
        baseline = max(positive_rate, 1e-6)
        if value >= baseline * 2.0:
            return f"明显高于随机基线（正例率约 {positive_rate:.2%}），排序质量较好。"
        if value >= baseline * 1.2:
            return f"略高于随机基线（正例率约 {positive_rate:.2%}），有一定筛选价值。"
        return f"接近随机基线（正例率约 {positive_rate:.2%}），建议增强可分性。"

    if name == "precision":
        if value >= 0.70:
            return "高概率通过的信号质量较高，误报率相对可控。"
        if value >= 0.55:
            return "信号质量中等，建议结合阈值和交易成本继续调优。"
        return "误报偏多，建议提高阈值或优化特征。"

    if name == "recall":
        if value >= 0.60:
            return "召回较高，保留了较多有效信号。"
        if value >= 0.40:
            return "召回中等，需平衡漏报与交易频率。"
        return "召回偏低，可能漏掉较多有效机会。"

    if name == "f1":
        if value >= 0.65:
            return "精确率与召回率平衡较好。"
        if value >= 0.50:
            return "整体平衡一般，可针对阈值和样本结构优化。"
        return "综合平衡偏弱，建议先提升可分性。"

    if name == "accuracy":
        if value >= 0.65:
            return "整体分类正确率较好。"
        if value >= 0.55:
            return "整体分类正确率中等。"
        return "整体正确率偏低。"

    if name == "brier_score":
        if value <= 0.18:
            return "概率校准较好，预测置信度较可靠。"
        if value <= 0.24:
            return "概率校准中等，建议配合校准方法优化。"
        return "概率校准偏弱，模型可能过于自信或保守。"

    if name == "log_loss":
        if value <= 0.60:
            return "概率预测质量较好。"
        if value <= 0.80:
            return "概率预测质量中等。"
        return "概率预测质量偏弱，建议检查特征与阈值设置。"

    return ""


def generate_meta_model_visual_report(
    output_path: str,
    meta_model,
    meta_X: np.ndarray,
    meta_y: np.ndarray,
    feature_names: List[str],
    threshold: float = 0.5,
    context_metrics: Optional[Dict[str, object]] = None,
) -> str:
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    if meta_X.size == 0 or len(meta_y) == 0:
        out.write_text("<html><body><h2>Meta模型报告</h2><p>无可用样本。</p></body></html>", encoding="utf-8")
        return str(out)

    if not getattr(meta_model, "enabled", False):
        out.write_text("<html><body><h2>Meta模型报告</h2><p>Meta模型未启用（样本不足或单类标签）。</p></body></html>", encoding="utf-8")
        return str(out)

    probs = meta_model.predict_proba(meta_X)
    pred = (probs >= threshold).astype(int)

    uniq = len(np.unique(meta_y))
    auc = _safe_metric(lambda: roc_auc_score(meta_y, probs)) if uniq > 1 else 0.0
    ap = _safe_metric(lambda: average_precision_score(meta_y, probs)) if uniq > 1 else 0.0
    precision = _safe_metric(lambda: precision_score(meta_y, pred, zero_division=0))
    recall = _safe_metric(lambda: recall_score(meta_y, pred, zero_division=0))
    f1 = _safe_metric(lambda: f1_score(meta_y, pred, zero_division=0))
    acc = _safe_metric(lambda: accuracy_score(meta_y, pred))
    brier = _safe_metric(lambda: brier_score_loss(meta_y, probs))
    ll = _safe_metric(lambda: log_loss(meta_y, np.clip(probs, 1e-8, 1 - 1e-8)))
    cm = confusion_matrix(meta_y, pred).tolist()

    # ROC
    fig1 = plt.figure(figsize=(6, 5))
    if uniq > 1:
        fpr, tpr, _ = roc_curve(meta_y, probs)
        plt.plot(fpr, tpr, label=f"AUC={auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Meta ROC Curve")
    plt.legend(loc="lower right")
    roc_b64 = _fig_to_b64(fig1)

    # PR
    fig2 = plt.figure(figsize=(6, 5))
    if uniq > 1:
        prc, rec, _ = precision_recall_curve(meta_y, probs)
        plt.plot(rec, prc, label=f"AP={ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Meta Precision-Recall Curve")
    plt.legend(loc="lower left")
    pr_b64 = _fig_to_b64(fig2)

    # Coefficients
    coefs = meta_model.model.coef_[0]
    coef_pairs = sorted(
        [(feature_names[i], float(coefs[i])) for i in range(min(len(feature_names), len(coefs)))],
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:30]
    names = [x[0] for x in coef_pairs][::-1]
    vals = [x[1] for x in coef_pairs][::-1]
    colors = ["#1f77b4" if v >= 0 else "#d62728" for v in vals]

    fig3 = plt.figure(figsize=(11, 9))
    plt.barh(names, vals, color=colors)
    plt.axvline(0.0, color="black", linewidth=1)
    plt.title("Meta Coefficients (Top 30 by |coef|)")
    coef_b64 = _fig_to_b64(fig3)

    # Probability distribution by correctness
    correct_mask = (pred == meta_y)
    fig4 = plt.figure(figsize=(8, 5))
    plt.hist(probs[correct_mask], bins=30, alpha=0.7, color="#2ca02c", label="Correct")
    plt.hist(probs[~correct_mask], bins=30, alpha=0.6, color="#d62728", label="Wrong")
    plt.axvline(float(threshold), color="black", linestyle="--", label=f"Threshold={threshold:.2f}")
    plt.title("Meta Probability Distribution (Correct vs Wrong)")
    plt.xlabel("Predicted Execution Probability")
    plt.ylabel("Count")
    plt.legend()
    dist_b64 = _fig_to_b64(fig4)

    threshold_rows = _threshold_rows(meta_y, probs)

    payload = {
        "sample_count": int(len(meta_y)),
        "positive_rate": float(np.mean(meta_y)),
        "threshold": float(threshold),
        "roc_auc": auc,
        "average_precision": ap,
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "brier_score": brier,
        "log_loss": ll,
        "confusion_matrix": cm,
        "context_metrics": context_metrics or {},
    }

    threshold_html = "".join(
        [
            (
                f"<tr><td>{r['threshold']:.2f}</td><td>{r['keep_rate']:.2%}</td>"
                f"<td>{r['precision']:.4f}</td><td>{r['recall']:.4f}</td>"
                f"<td>{r['f1']:.4f}</td><td>{r['accuracy']:.4f}</td></tr>"
            )
            for r in threshold_rows
        ]
    )

    summary_cards = "".join(
        [
            f"<div class='metric'><div class='v'>{payload['roc_auc']:.4f}</div><div class='k'>AUC（ROC曲线下面积）</div></div>",
            f"<div class='metric'><div class='v'>{payload['average_precision']:.4f}</div><div class='k'>AP（平均精确率）</div></div>",
            f"<div class='metric'><div class='v'>{payload['precision']:.4f}</div><div class='k'>精确率（Precision）</div></div>",
            f"<div class='metric'><div class='v'>{payload['recall']:.4f}</div><div class='k'>召回率（Recall）</div></div>",
            f"<div class='metric'><div class='v'>{payload['f1']:.4f}</div><div class='k'>F1分数</div></div>",
            f"<div class='metric'><div class='v'>{payload['accuracy']:.4f}</div><div class='k'>准确率（Accuracy）</div></div>",
            f"<div class='metric'><div class='v'>{payload['brier_score']:.4f}</div><div class='k'>Brier分数（越低越好）</div></div>",
            f"<div class='metric'><div class='v'>{payload['log_loss']:.4f}</div><div class='k'>对数损失（LogLoss，越低越好）</div></div>",
        ]
    )

    metric_explain_rows = "".join(
        [
            (
                f"<tr><td>AUC</td><td>{payload['roc_auc']:.4f}</td>"
                f"<td>衡量模型区分正负样本能力，越接近1越好。</td>"
                f"<td>{_describe_metric_level('roc_auc', payload['roc_auc'], payload['positive_rate'])}</td></tr>"
            ),
            (
                f"<tr><td>AP</td><td>{payload['average_precision']:.4f}</td>"
                f"<td>衡量排序质量，关注高分样本是否更可能为正例。</td>"
                f"<td>{_describe_metric_level('average_precision', payload['average_precision'], payload['positive_rate'])}</td></tr>"
            ),
            (
                f"<tr><td>精确率</td><td>{payload['precision']:.4f}</td>"
                f"<td>被模型判为可执行的信号中，真实可执行的比例。</td>"
                f"<td>{_describe_metric_level('precision', payload['precision'])}</td></tr>"
            ),
            (
                f"<tr><td>召回率</td><td>{payload['recall']:.4f}</td>"
                f"<td>所有真实可执行信号中，被模型成功保留的比例。</td>"
                f"<td>{_describe_metric_level('recall', payload['recall'])}</td></tr>"
            ),
            (
                f"<tr><td>F1分数</td><td>{payload['f1']:.4f}</td>"
                f"<td>精确率与召回率的综合平衡指标。</td>"
                f"<td>{_describe_metric_level('f1', payload['f1'])}</td></tr>"
            ),
            (
                f"<tr><td>准确率</td><td>{payload['accuracy']:.4f}</td>"
                f"<td>整体分类正确比例（类别不平衡时需结合其他指标）。</td>"
                f"<td>{_describe_metric_level('accuracy', payload['accuracy'])}</td></tr>"
            ),
            (
                f"<tr><td>Brier分数</td><td>{payload['brier_score']:.4f}</td>"
                f"<td>概率预测与真实结果的均方误差，越低越好。</td>"
                f"<td>{_describe_metric_level('brier_score', payload['brier_score'])}</td></tr>"
            ),
            (
                f"<tr><td>LogLoss</td><td>{payload['log_loss']:.4f}</td>"
                f"<td>概率预测损失，错且自信的预测惩罚更重。</td>"
                f"<td>{_describe_metric_level('log_loss', payload['log_loss'])}</td></tr>"
            ),
        ]
    )

    analysis_text = (
        "Meta模型作为二层执行过滤器，目标是从Primary产生的候选信号中保留高质量交易。"
        "当阈值提高时，通常精确率上升、召回率下降。"
        "建议结合阈值敏感性表与回测收益曲线联动选择最终阈值。"
    )

    html = f"""<!DOCTYPE html>
<html lang=\"zh-CN\"><head><meta charset=\"UTF-8\"><title>Meta模型可视化报告</title>
<style>
body{{font-family:'Segoe UI',Arial,sans-serif;background:#f6f8fb;color:#1f2937;margin:0;padding:24px;}}
.card{{background:#fff;border-radius:12px;padding:20px;margin-bottom:18px;box-shadow:0 2px 8px rgba(0,0,0,0.06);}}
.h1{{background:linear-gradient(135deg,#0f2c59,#195ba6);color:#fff;}}
.row{{display:grid;grid-template-columns:1fr 1fr;gap:16px;}}
.metrics{{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:12px;}}
.metric{{background:#eef4ff;border:1px solid #cddcf7;border-radius:10px;padding:12px;text-align:center;}}
.metric .v{{font-size:1.6em;font-weight:700;color:#124188;}}
.metric .k{{font-size:0.9em;color:#4b5563;}}
img{{max-width:100%;border-radius:8px;}}
pre{{background:#0f172a;color:#e2e8f0;padding:12px;border-radius:8px;overflow:auto;}}
table{{width:100%;border-collapse:collapse;}}
th,td{{padding:10px;border-bottom:1px solid #e5e7eb;text-align:left;}}
th{{background:#f3f6fb;}}
@media (max-width: 960px) {{ .row{{grid-template-columns:1fr;}} }}
</style></head><body>
<div class=\"card h1\"><h2>Meta 模型可视化分析</h2><p>二层执行过滤模型：控制信号通过率，平衡精度与覆盖率</p></div>

<div class=\"card\"><h3>模型定位与框架作用</h3>
<p>在 L3+L4 框架中，Primary 模型负责方向预测（SL/TIMEOUT/PT），Meta 模型负责执行过滤（是否放行）。Meta 直接影响实盘交易频次、胜率与回撤风险，是策略稳健性闸门。</p>
</div>

<div class=\"card\"><h3>核心指标</h3><div class=\"metrics\">{summary_cards}</div></div>

<div class=\"row\">
  <div class=\"card\"><h3>ROC 曲线</h3><img src=\"data:image/png;base64,{roc_b64}\"></div>
  <div class=\"card\"><h3>Precision-Recall 曲线</h3><img src=\"data:image/png;base64,{pr_b64}\"></div>
</div>

<div class=\"row\">
  <div class=\"card\"><h3>概率分布（正确 vs 错误）</h3><img src=\"data:image/png;base64,{dist_b64}\"></div>
  <div class=\"card\"><h3>系数重要性（Top 30）</h3><img src=\"data:image/png;base64,{coef_b64}\"></div>
</div>

<div class="card"><h3>阈值敏感性分析</h3>
<table>
    <tr><th>阈值</th><th>信号保留率</th><th>精确率</th><th>召回率</th><th>F1分数</th><th>准确率</th></tr>
  {threshold_html}
</table>
</div>

<div class="card"><h3>指标释义与当前效果解读</h3>
<table>
    <tr><th>指标</th><th>当前值</th><th>作用说明</th><th>结合结果解读</th></tr>
    {metric_explain_rows}
</table>
</div>

<div class="card"><h3>训练结果总结</h3><p>{analysis_text}</p>
<p>建议观察要点：
1) AUC / AP 是否显著高于随机基线；
2) 阈值上调后精确率提升是否值得以“保留率下降”为代价；
3) Brier 与 LogLoss 是否同步改善，避免“高分但不可靠”的概率输出。</p>
</div>

<div class=\"card\"><h3>原始指标JSON</h3><pre>{json.dumps(payload, indent=2, ensure_ascii=False)}</pre></div>
</body></html>"""

    out.write_text(html, encoding="utf-8")
    return str(out)
