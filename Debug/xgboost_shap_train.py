"""
XGBoost + SHAP V3 训练流程 — 双模型 (买点/卖点质量评估)
============================================================
核心思路:
  缠论中有不同级别的三类买卖点(1类/2类/3类), 本脚本学习每个买卖点
  作为"合格"买点或卖点的特征模式, 分别训练两个独立模型:
    - 买点模型: 学习合格买点的特征 (买点→下一个卖点涨幅 ≥ 阈值)
    - 卖点模型: 学习合格卖点的特征 (卖点→下一个买点跌幅 ≥ 阈值)

  合格标准:
    - 合格买点: 从该买点到当前级别下一个卖点(1/2/3类任一)出现时,
                涨幅 = (卖点价 - 买点价) / 买点价 ≥ QUALIFIED_THRESHOLD
    - 合格卖点: 从该卖点到当前级别下一个买点(1/2/3类任一)出现时,
                跌幅 = (卖点价 - 买点价) / 卖点价 ≥ QUALIFIED_THRESHOLD

  交易决策:
    - 买点合格 → 买入(开多) 或 平空
    - 卖点合格 → 卖出(开空) 或 平多

改进项 (相对 V2):
  1. 双模型架构: 买点模型 + 卖点模型, 各自独立训练
  2. BSP 类型分类: 1/2/3类独热编码, 每个模型区分三类买卖点
  3. 方向解耦: 移除 is_buy_signal 特征, 由模型本身隐含方向
  4. 抗过拟合: scale_pos_weight, 增强正则化, 特征精简
  5. 时间序列切分: TimeSeriesSplit 交叉验证
  6. 特征筛选由用户在 feature_center.py 中手动管理

输出 (每个方向各一套):
  - Debug/model_buy.json / model_sell.json         XGBoost 模型
  - Debug/meta_buy.json / meta_sell.json           特征元数据
  - Debug/shap_report_buy.html / ..._sell.html     SHAP 报告
  - Debug/metrics_buy.json / ..._sell.json         训练指标
============================================================
"""

import json
import os
import time
from collections import Counter
from typing import Dict, List

import numpy as np
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit

# ── 项目模块 ──────────────────────────────────────────────
from Chan import CChan
from ChanConfig import CChanConfig
from ChanModel.Features import CFeatures
from ChanModel.feature_center import build_features
from ChanModel.shap_analyzer import SHAPAnalyzer
from Common.CEnum import AUTYPE, DATA_SRC, KL_TYPE
from Common.CTime import CTime
from Plot.mpl.PlotDriver import CPlotDriver


# ============================================================
# 配置区
# ============================================================

CODE = "BTCUSDT"
BEGIN_TIME = "2021-02-04"
END_TIME = "2023-01-01"
DATA_SRC_TYPE = DATA_SRC.CSV
LV_LIST = [KL_TYPE.K_15M]

# 时间序列拆分比例
TRAIN_RATIO = 0.8

# XGBoost 参数 (V2: 加强正则化)
XGB_PARAMS = {
    "max_depth": 3,             # V1=4, 降低防过拟合
    "eta": 0.05,                # V1=0.1, 降低学习率
    "subsample": 0.8,
    "colsample_bytree": 0.7,    # V1=0.8, 减少每棵树使用的特征比例
    "objective": "binary:logistic",
    "eval_metric": "aucpr",
    "min_child_weight": 10,     # V1=5, 要求更多样本才分裂
    "gamma": 0.3,               # V1=0.1, 分裂最小增益
    "reg_alpha": 0.5,           # V1=0.1, L1 正则化
    "reg_lambda": 2.0,          # V1=1.0, L2 正则化
    "seed": 42,
    # scale_pos_weight: 动态计算
}
NUM_BOOST_ROUND = 300           # V1=200
EARLY_STOPPING_ROUNDS = 30      # V1=20

# 买卖点质量阈值: 到下一个相反买卖点时, 涨跌幅 >= 此值才标为"合格"
QUALIFIED_THRESHOLD = 0.015     # 1.5%

# 缠论配置
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

# 输出路径 (双模型: buy / sell)
OUTPUT_DIR = "Debug"

# 买点模型输出
MODEL_BUY_PATH = os.path.join(OUTPUT_DIR, "model_buy.json")
META_BUY_PATH = os.path.join(OUTPUT_DIR, "meta_buy.json")
REPORT_BUY_PATH = os.path.join(OUTPUT_DIR, "shap_report_buy.html")
METRICS_BUY_PATH = os.path.join(OUTPUT_DIR, "metrics_buy.json")
LIBSVM_BUY_PATH = os.path.join(OUTPUT_DIR, "feature_buy.libsvm")

# 卖点模型输出
MODEL_SELL_PATH = os.path.join(OUTPUT_DIR, "model_sell.json")
META_SELL_PATH = os.path.join(OUTPUT_DIR, "meta_sell.json")
REPORT_SELL_PATH = os.path.join(OUTPUT_DIR, "shap_report_sell.html")
METRICS_SELL_PATH = os.path.join(OUTPUT_DIR, "metrics_sell.json")
LIBSVM_SELL_PATH = os.path.join(OUTPUT_DIR, "feature_sell.libsvm")


# ============================================================
# 可视化
# ============================================================

def plot_labels(chan, plot_marker, output_path="Debug/label_shap.png"):
    plot_config = {
        "plot_kline": True, "plot_bi": True, "plot_seg": True,
        "plot_zs": True, "plot_bsp": True, "plot_marker": True,
    }
    plot_para = {
        "figure": {"x_range": 400},
        "marker": {"markers": plot_marker},
    }
    plot_driver = CPlotDriver(chan, plot_config=plot_config, plot_para=plot_para)
    plot_driver.save2img(output_path)
    print(f"  [Plot] 标签可视化: {output_path}")


# ============================================================
# 阶段一: BSP 特征收集
# ============================================================

def collect_features():
    """
    step_load 收集全部 BSP，记录：
      - 特征 (feature_center + 内置特征)
      - 方向 (is_buy)
      - 可交易价格 (trade_price = 检测时K线收盘价)
      - BSP 主类型 ('1' / '2' / '3')
    """
    print("=" * 60)
    print("[阶段1] BSP 特征收集")
    print(f"  标的: {CODE}  周期: {LV_LIST}  时间: {BEGIN_TIME} ~ {END_TIME}")
    print("=" * 60)

    config = CChanConfig(CHAN_CONFIG)
    chan = CChan(
        code=CODE, begin_time=BEGIN_TIME, end_time=END_TIME,
        data_src=DATA_SRC_TYPE, lv_list=LV_LIST, config=config, autype=AUTYPE.QFQ,
    )

    bsp_samples = []
    seen_idx = set()
    step_count = 0

    for chan_snapshot in chan.step_load():
        step_count += 1
        if step_count % 10000 == 0:
            print(f"  已处理 {step_count} 步 ...")

        last_klu = chan_snapshot[0][-1][-1]
        bsp_list = chan_snapshot.get_latest_bsp()
        if not bsp_list:
            continue

        last_bsp = bsp_list[0]
        cur_lv_chan = chan_snapshot[0]

        if last_bsp.klu.idx in seen_idx:
            continue
        if cur_lv_chan[-2].idx != last_bsp.klu.klc.idx:
            continue

        seen_idx.add(last_bsp.klu.idx)

        # BSP 主类型: '1'/'2'/'3'
        bsp_main_type = last_bsp.type[0].value[0]

        # 补充 feature_center 全量特征
        extra_feat = build_features(
            klu=last_klu,
            history=cur_lv_chan.lst,
            chan=cur_lv_chan,
            bsp=last_bsp,
        )
        last_bsp.features.add_feat(extra_feat)

        bsp_samples.append({
            "klu_idx": last_bsp.klu.idx,
            "feature": last_bsp.features,
            "is_buy": last_bsp.is_buy,
            "open_time": last_klu.time,
            "trade_price": float(last_klu.close),
            "bsp_main_type": bsp_main_type,
            "bsp_types_str": last_bsp.type2str(),
        })

    # 统计
    type_counts = Counter(s["bsp_main_type"] for s in bsp_samples)
    dir_counts = Counter("买" if s["is_buy"] else "卖" for s in bsp_samples)
    print(f"  ✓ 共收集 {len(bsp_samples)} 个 BSP")
    print(f"  ✓ 按类型: {dict(type_counts)}")
    print(f"  ✓ 按方向: {dict(dir_counts)}")

    return bsp_samples, chan


# ============================================================
# 阶段二: 买卖点质量标注 (按方向分组)
# ============================================================

def label_bsp_quality(bsp_samples):
    """
    买卖点质量标注 (按方向拆分):
      买点: 找下一个卖点(任意类型), 计算涨幅=(sell_price-buy_price)/buy_price
             涨幅 >= QUALIFIED_THRESHOLD → 1 (合格), 否则 → 0
      卖点: 找下一个买点(任意类型), 计算跌幅=(sell_price-buy_price)/sell_price
             跌幅 >= QUALIFIED_THRESHOLD → 1 (合格), 否则 → 0

    返回:
      buy_samples  - 买点样本 (学习买点质量)
      sell_samples - 卖点样本 (学习卖点质量)
    """
    print("\n" + "=" * 60)
    print("[阶段2] 买卖点质量标注 (按方向分组)")
    print(f"  合格阈值: {QUALIFIED_THRESHOLD:.2%}")
    print("=" * 60)

    n = len(bsp_samples)
    buy_samples = []
    sell_samples = []
    unpaired = 0

    for i, bsp in enumerate(bsp_samples):
        # 找下一个方向相反的 BSP
        partner = None
        for j in range(i + 1, n):
            if bsp_samples[j]["is_buy"] != bsp["is_buy"]:
                partner = bsp_samples[j]
                break

        if partner is None:
            unpaired += 1
            continue

        # 计算涨跌幅
        if bsp["is_buy"]:
            # 买点 → 下一个卖点, 涨幅
            change_pct = (
                (partner["trade_price"] - bsp["trade_price"])
                / (bsp["trade_price"] + 1e-9)
            )
        else:
            # 卖点 → 下一个买点, 跌幅
            change_pct = (
                (bsp["trade_price"] - partner["trade_price"])
                / (bsp["trade_price"] + 1e-9)
            )

        label = 1 if change_pct >= QUALIFIED_THRESHOLD else 0

        sample = {
            "klu_idx": bsp["klu_idx"],
            "feature": bsp["feature"],
            "is_buy": bsp["is_buy"],
            "open_time": bsp["open_time"],
            "trade_price": bsp["trade_price"],
            "bsp_main_type": bsp["bsp_main_type"],
            "bsp_types_str": bsp["bsp_types_str"],
            "label": label,
            "change_pct": change_pct,
            "partner_time": str(partner["open_time"]),
            "holding_bars": partner["klu_idx"] - bsp["klu_idx"],
        }

        if bsp["is_buy"]:
            buy_samples.append(sample)
        else:
            sell_samples.append(sample)

    # 统计
    def _print_stats(name, samples):
        if not samples:
            print(f"  {name}: 0 样本")
            return
        n_qual = sum(1 for s in samples if s["label"] == 1)
        n_unqual = len(samples) - n_qual
        print(f"  {name}: {len(samples)} 样本  "
              f"合格={n_qual}({n_qual / len(samples):.1%})  "
              f"不合格={n_unqual}({n_unqual / len(samples):.1%})")
        for t in ['1', '2', '3']:
            sub = [s for s in samples if s["bsp_main_type"] == t]
            if sub:
                q = sum(1 for s in sub if s["label"] == 1)
                print(f"    类型{t}: {len(sub)}样本, "
                      f"合格率={q / len(sub):.1%}")

    print(f"  ✓ 未配对(丢弃): {unpaired}")
    _print_stats("买点(买点→卖点)", buy_samples)
    _print_stats("卖点(卖点→买点)", sell_samples)

    return buy_samples, sell_samples


# ============================================================
# 阶段三: 构建特征矩阵 (单方向)
# ============================================================

def build_dataset(labeled_samples, direction_name=""):
    """
    构建 X, y 矩阵 (单方向模型):
      - bsp_type_1/2/3 独热编码 (区分三类买卖点)
      - feature_center 特征 + 内置特征
      - 不含 is_buy_signal (方向已由模型本身隐含)
    """
    print("\n" + "=" * 60)
    print(f"[阶段3] 构建数据集 [{direction_name}]")
    print("=" * 60)

    # 特征 schema - BSP 类型独热
    feature_meta = {}
    cur_idx = 0
    for name in ["bsp_type_1", "bsp_type_2", "bsp_type_3"]:
        feature_meta[name] = cur_idx
        cur_idx += 1

    # 再放所有 feature_center + 内置特征
    for sample in labeled_samples:
        for feat_name, _ in sample["feature"].items():
            if feat_name not in feature_meta:
                feature_meta[feat_name] = cur_idx
                cur_idx += 1

    n_features = len(feature_meta)
    n_samples = len(labeled_samples)
    feature_names = [""] * n_features
    for name, idx in feature_meta.items():
        feature_names[idx] = name

    X = np.full((n_samples, n_features), np.nan)
    y = np.zeros(n_samples, dtype=np.int32)
    sample_info = []

    for i, sample in enumerate(labeled_samples):
        y[i] = sample["label"]

        # BSP 类型独热
        X[i, feature_meta["bsp_type_1"]] = 1.0 if sample["bsp_main_type"] == '1' else 0.0
        X[i, feature_meta["bsp_type_2"]] = 1.0 if sample["bsp_main_type"] == '2' else 0.0
        X[i, feature_meta["bsp_type_3"]] = 1.0 if sample["bsp_main_type"] == '3' else 0.0

        # 特征值
        for feat_name, feat_value in sample["feature"].items():
            if feat_name in feature_meta:
                X[i, feature_meta[feat_name]] = feat_value

        sample_info.append({
            "klu_idx": sample["klu_idx"],
            "is_buy": sample["is_buy"],
            "open_time": str(sample["open_time"]),
            "label": sample["label"],
            "change_pct": sample["change_pct"],
            "bsp_main_type": sample["bsp_main_type"],
        })

    print(f"  ✓ 特征数: {n_features}  样本数: {n_samples}")
    print(f"  ✓ 正样本: {int(y.sum())} ({y.mean():.1%})  "
          f"负样本: {n_samples - int(y.sum())} ({1 - y.mean():.1%})")

    # 按 BSP 类型细分统计
    for t in ['1', '2', '3']:
        mask = X[:, feature_meta["bsp_type_" + t]] == 1.0
        if mask.sum() > 0:
            pos = y[mask].sum()
            print(f"    类型{t}: {int(mask.sum())}样本, 正例={int(pos)}({pos / mask.sum():.1%})")

    return X, y, feature_names, feature_meta, sample_info


# ============================================================
# 阶段四: 训练 + 评估
# ============================================================

def train_and_evaluate(X, y, feature_names, model_path, metrics_path, direction_name=""):
    """
    时间序列拆分 → XGBoost (scale_pos_weight + 增强正则化) → 评估
    """
    print("\n" + "=" * 60)
    print(f"[阶段4] 模型训练与评估 [{direction_name}]")
    print("=" * 60)

    split_idx = int(len(y) * TRAIN_RATIO)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # 动态计算 scale_pos_weight (处理类不平衡)
    n_pos_train = int(y_train.sum())
    n_neg_train = len(y_train) - n_pos_train
    scale_pw = n_neg_train / max(n_pos_train, 1)

    print(f"  训练集: {len(y_train)}  测试集: {len(y_test)}")
    print(f"  训练正例率: {y_train.mean():.1%}  测试正例率: {y_test.mean():.1%}")
    print(f"  scale_pos_weight: {scale_pw:.2f}")

    params = {**XGB_PARAMS, "scale_pos_weight": scale_pw}

    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names, missing=np.nan)
    dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feature_names, missing=np.nan)

    evals_result = {}
    print(f"\n  训练中... (max_rounds={NUM_BOOST_ROUND}, early_stop={EARLY_STOPPING_ROUNDS})")

    bst = xgb.train(
        params,
        dtrain=dtrain,
        num_boost_round=NUM_BOOST_ROUND,
        evals=[(dtrain, "train"), (dtest, "test")],
        evals_result=evals_result,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        verbose_eval=20,
    )

    bst.save_model(model_path)
    print(f"\n  ✓ 模型已保存: {model_path}")

    # 评估
    y_pred_prob = bst.predict(dtest)
    y_pred = (y_pred_prob >= 0.5).astype(int)

    metrics = {
        "direction": direction_name,
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "AUC": float(roc_auc_score(y_test, y_pred_prob)) if len(np.unique(y_test)) > 1 else 0.0,
        "Precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "F1": float(f1_score(y_test, y_pred, zero_division=0)),
        "Train_Samples": int(len(y_train)),
        "Test_Samples": int(len(y_test)),
        "Best_Iteration": int(bst.best_iteration) if hasattr(bst, "best_iteration") else NUM_BOOST_ROUND,
        "Best_AUC_Train": float(max(evals_result["train"]["aucpr"])),
        "Best_AUC_Test": float(max(evals_result["test"]["aucpr"])),
        "scale_pos_weight": round(scale_pw, 4),
        "n_features": len(feature_names),
    }

    print(f"\n  ═══════ [{direction_name}] 测试集评估 ═══════")
    print(f"  Accuracy  : {metrics['Accuracy']:.4f}")
    print(f"  AUC       : {metrics['AUC']:.4f}")
    print(f"  Precision : {metrics['Precision']:.4f}")
    print(f"  Recall    : {metrics['Recall']:.4f}")
    print(f"  F1 Score  : {metrics['F1']:.4f}")
    print(f"  Best Iter : {metrics['Best_Iteration']}")
    print()
    print("  分类报告:")
    print(classification_report(y_test, y_pred, target_names=["不合格", "合格"]))

    cm = confusion_matrix(y_test, y_pred)
    print(f"  混淆矩阵:\n  {cm}")

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    return bst, metrics, evals_result, (X_train, y_train, X_test, y_test)


# ============================================================
# 阶段五: SHAP 分析
# ============================================================

def shap_analysis(bst, X, y, feature_names, metrics, report_path, direction_name=""):
    """SHAP TreeExplainer + 报告"""
    print("\n" + "=" * 60)
    print(f"[阶段5] SHAP 可解释性分析 [{direction_name}]")
    print("=" * 60)

    analyzer = SHAPAnalyzer(model=bst, feature_names=feature_names)

    print("  计算 SHAP 值 ...")
    t0 = time.time()
    result = analyzer.analyze(X, y)
    elapsed = time.time() - t0
    print(f"  ✓ 计算完成 ({elapsed:.1f}s)")

    print(f"\n  ═══════ [{direction_name}] Top-10 特征 (Mean |SHAP|) ═══════")
    for _, row in result.importance_df.head(10).iterrows():
        print(f"  {int(row['rank']):>3d}. {row['feature']:<30s}  {row['mean_abs_shap']:.6f}")

    print(f"\n  生成报告 ...")
    analyzer.generate_report(
        result,
        output_path=report_path,
        model_metrics=metrics,
        top_dependence=6,
    )

    return result


# ============================================================
# 阶段六: 时间序列交叉验证
# ============================================================

def time_series_cv(X, y, feature_names, direction_name=""):
    """
    TimeSeriesSplit 5-Fold 交叉验证
    (expanding window, 每折动态调整 scale_pos_weight)
    """
    print("\n" + "=" * 60)
    print(f"[阶段6] TimeSeriesSplit 交叉验证 [{direction_name}]")
    print("=" * 60)

    tscv = TimeSeriesSplit(n_splits=5)
    fold_aucs = []

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]

        n_pos = int(y_tr.sum())
        n_neg = len(y_tr) - n_pos
        scale_pw = n_neg / max(n_pos, 1)

        params = {**XGB_PARAMS, "scale_pos_weight": scale_pw}

        dtrain = xgb.DMatrix(X_tr, label=y_tr, feature_names=feature_names, missing=np.nan)
        dtest = xgb.DMatrix(X_te, label=y_te, feature_names=feature_names, missing=np.nan)

        bst = xgb.train(
            params, dtrain,
            num_boost_round=NUM_BOOST_ROUND,
            evals=[(dtest, "eval")],
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            verbose_eval=False,
        )

        y_pred = bst.predict(dtest)
        if len(np.unique(y_te)) > 1:
            auc = roc_auc_score(y_te, y_pred)
            fold_aucs.append(auc)
            print(f"  Fold {fold + 1}: AUC={auc:.4f}  "
                  f"(训练={len(y_tr)}, 测试={len(y_te)}, 正例率={y_te.mean():.1%})")
        else:
            print(f"  Fold {fold + 1}: 跳过 (测试集只有单类)")

    if fold_aucs:
        print(f"\n  TimeSeriesSplit AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")

    return fold_aucs


# ============================================================
# 单方向训练流水线
# ============================================================

def train_direction_pipeline(labeled_samples, direction_name, model_path,
                             meta_path, report_path, metrics_path,
                             libsvm_path):
    """
    对一个方向 (买点/卖点) 执行完整训练流水线:
      构建数据集 → 保存 Meta/LibSVM → 训练 → SHAP → CV
    """
    print("\n")
    print("▓" * 60)
    print(f"  >>>  {direction_name}模型训练流水线  <<<")
    print("▓" * 60)

    if len(labeled_samples) < 20:
        print(f"  [跳过] {direction_name}样本不足 ({len(labeled_samples)}), 无法训练")
        return None

    # 3. 构建数据集
    X, y, feature_names, feature_meta, sample_info = build_dataset(
        labeled_samples, direction_name=direction_name
    )

    # 保存 meta
    new_meta = {name: i for i, name in enumerate(feature_names)}
    with open(meta_path, "w") as f:
        json.dump(new_meta, f, indent=2)
    print(f"  ✓ Meta 已保存: {meta_path} ({len(new_meta)} 特征)")

    # 保存 libsvm
    with open(libsvm_path, "w") as fid:
        for i in range(len(y)):
            pairs = [
                (j, float(X[i, j]))
                for j in range(X.shape[1])
                if not np.isnan(X[i, j])
            ]
            fid.write(f"{y[i]} " + " ".join(f"{idx}:{val}" for idx, val in pairs) + "\n")
    print(f"  ✓ LibSVM: {libsvm_path}")

    # 4. 训练 + 评估
    bst, metrics, evals_result, splits = train_and_evaluate(
        X, y, feature_names,
        model_path=model_path,
        metrics_path=metrics_path,
        direction_name=direction_name,
    )

    # 5. SHAP 分析
    shap_result = shap_analysis(
        bst, X, y, feature_names, metrics,
        report_path=report_path,
        direction_name=direction_name,
    )

    # 6. 时间序列交叉验证
    try:
        cv_aucs = time_series_cv(X, y, feature_names, direction_name=direction_name)
    except Exception as e:
        print(f"  [Warning] {direction_name}交叉验证跳过: {e}")

    return {
        "model": bst,
        "metrics": metrics,
        "feature_names": feature_names,
        "shap_result": shap_result,
    }


# ============================================================
# 主流程
# ============================================================

if __name__ == "__main__":
    start_time = time.time()

    print("\n" + "★" * 60)
    print("   XGBoost + SHAP V3 训练流程 — 双模型 (买点/卖点质量评估)")
    print("   [买卖点质量 + BSP类型分类 + 抗过拟合]")
    print("★" * 60)

    # 1. 收集 BSP 特征
    bsp_samples, chan = collect_features()

    # 2. 买卖点质量标注 + 按方向分组
    buy_samples, sell_samples = label_bsp_quality(bsp_samples)

    # 标签可视化
    try:
        plot_marker = {}
        all_labeled = buy_samples + sell_samples
        for s in all_labeled:
            plot_marker[s["open_time"].to_str()] = (
                "√" if s["label"] else "×",
                "down" if s["is_buy"] else "up",
            )
        plot_labels(chan, plot_marker)
    except Exception as e:
        print(f"  [Warning] 可视化跳过: {e}")

    # ── 买点模型训练 ──
    buy_result = train_direction_pipeline(
        labeled_samples=buy_samples,
        direction_name="买点",
        model_path=MODEL_BUY_PATH,
        meta_path=META_BUY_PATH,
        report_path=REPORT_BUY_PATH,
        metrics_path=METRICS_BUY_PATH,
        libsvm_path=LIBSVM_BUY_PATH,
    )

    # ── 卖点模型训练 ──
    sell_result = train_direction_pipeline(
        labeled_samples=sell_samples,
        direction_name="卖点",
        model_path=MODEL_SELL_PATH,
        meta_path=META_SELL_PATH,
        report_path=REPORT_SELL_PATH,
        metrics_path=METRICS_SELL_PATH,
        libsvm_path=LIBSVM_SELL_PATH,
    )

    # 完成
    total_time = time.time() - start_time
    print(f"\n{'★' * 60}")
    print(f"   全部完成! 耗时: {total_time:.1f}s")
    print(f"   ─── 买点模型 ───")
    print(f"   模型:       {MODEL_BUY_PATH}")
    print(f"   Meta:       {META_BUY_PATH}")
    print(f"   报告:       {REPORT_BUY_PATH}")
    print(f"   指标:       {METRICS_BUY_PATH}")
    print(f"   ─── 卖点模型 ───")
    print(f"   模型:       {MODEL_SELL_PATH}")
    print(f"   Meta:       {META_SELL_PATH}")
    print(f"   报告:       {REPORT_SELL_PATH}")
    print(f"   指标:       {METRICS_SELL_PATH}")
    print(f"{'★' * 60}")
