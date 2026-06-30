# -*- coding: utf-8 -*-
"""
08_prediction_model.py - 预测建模模块
======================================
功能：以"是否爆品"为目标变量，构建逻辑回归、随机森林、XGBoost三个模型
评估模型性能，输出特征重要性 Top5
输入：data/processed/burst_data.csv
输出：figures/模型对比图、figures/特征重要性图
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (roc_auc_score, accuracy_score, recall_score,
                              precision_score, f1_score, roc_curve,
                              classification_report, confusion_matrix)
import xgboost as xgb

warnings.filterwarnings('ignore')
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(PROJECT_DIR, 'data', 'processed')
FIGURES_DIR = os.path.join(PROJECT_DIR, 'figures')

COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1']



def restore_categorical(df):
    """CSV 读取后恢复 Categorical 列的正确排序"""
    if '价格区间' in df.columns:
        order = ['0-50', '50-100', '100-200', '200-500', '500-1000', '1000+']
        df['价格区间'] = pd.Categorical(df['价格区间'], categories=order, ordered=True)
    if '讲解时长区间' in df.columns:
        order = ['<2分钟', '2-5分钟', '5-10分钟', '10分钟+']
        df['讲解时长区间'] = pd.Categorical(df['讲解时长区间'], categories=order, ordered=True)
    if '直播间规模' in df.columns:
        order = ['小直播间(<500)', '中型(500-2K)', '大型(2K-5K)', '头部(5K+)']
        df['直播间规模'] = pd.Categorical(df['直播间规模'], categories=order, ordered=True)
    return df

def load_data():
    """加载数据"""
    path = os.path.join(PROCESSED_DIR, 'burst_data.csv')
    df = pd.read_csv(path, encoding='utf-8')
    df = restore_categorical(df)
    print(f"加载数据: {df.shape[0]} 行  x  {df.shape[1]} 列")
    return df


def prepare_features(df):
    """
    准备建模特征

    参数:
        df: 数据框

    返回:
        X: 特征矩阵
        y: 目标变量
        feature_names: 特征名列表
    """
    print(f"\n{'='*60}")
    print("【特征工程】")
    print(f"{'='*60}")

    # 选择特征列
    feature_cols = ['price', 'userbefore', 'popduration', 'avg_dis', 'no_pop']
    target_col = '是否爆品'

    # 添加时段编码
    if '时段' in df.columns:
        le = LabelEncoder()
        df['时段编码'] = le.fit_transform(df['时段'].astype(str))
        feature_cols.append('时段编码')

    X = df[feature_cols].copy()
    y = df[target_col].copy()

    print(f"特征数量: {len(feature_cols)}")
    print(f"特征列: {feature_cols}")
    print(f"目标变量分布: 爆品={y.sum()}, 普通品={len(y)-y.sum()}")
    print(f"正样本比例: {y.mean()*100:.1f}%")

    return X, y, feature_cols


def train_and_evaluate(X, y, feature_names):
    """
    训练三个模型并评估

    参数:
        X: 特征矩阵
        y: 目标变量
        feature_names: 特征名列表

    返回:
        dict: 各模型的评估结果
    """
    # 划分训练集和测试集（分层抽样，保持正负样本比例）
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\n训练集: {len(X_train)} 样本, 测试集: {len(X_test)} 样本")

    # 标准化：仅在训练集上 fit，避免数据泄漏
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train),
                            columns=feature_names, index=X_train.index)
    X_test = pd.DataFrame(scaler.transform(X_test),
                           columns=feature_names, index=X_test.index)

    results = {}

    # ---- 模型1：逻辑回归 ----
    print(f"\n--- 逻辑回归 ---")
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    y_prob_lr = lr.predict_proba(X_test)[:, 1]

    results['逻辑回归'] = {
        'model': lr,
        'y_pred': y_pred_lr,
        'y_prob': y_prob_lr,
        'auc': roc_auc_score(y_test, y_prob_lr),
        'accuracy': accuracy_score(y_test, y_pred_lr),
        'recall': recall_score(y_test, y_pred_lr),
        'precision': precision_score(y_test, y_pred_lr),
        'f1': f1_score(y_test, y_pred_lr),
        'feature_importance': np.abs(lr.coef_[0]),
    }
    print(f"  AUC: {results['逻辑回归']['auc']:.4f}")
    print(f"  准确率: {results['逻辑回归']['accuracy']:.4f}")
    print(f"  召回率: {results['逻辑回归']['recall']:.4f}")

    # ---- 模型2：随机森林 ----
    print(f"\n--- 随机森林 ---")
    rf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42,
                                 class_weight='balanced')
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    y_prob_rf = rf.predict_proba(X_test)[:, 1]

    results['随机森林'] = {
        'model': rf,
        'y_pred': y_pred_rf,
        'y_prob': y_prob_rf,
        'auc': roc_auc_score(y_test, y_prob_rf),
        'accuracy': accuracy_score(y_test, y_pred_rf),
        'recall': recall_score(y_test, y_pred_rf),
        'precision': precision_score(y_test, y_pred_rf),
        'f1': f1_score(y_test, y_pred_rf),
        'feature_importance': rf.feature_importances_,
    }
    print(f"  AUC: {results['随机森林']['auc']:.4f}")
    print(f"  准确率: {results['随机森林']['accuracy']:.4f}")
    print(f"  召回率: {results['随机森林']['recall']:.4f}")

    # ---- 模型3：XGBoost ----
    print(f"\n--- XGBoost ---")
    # 计算正负样本比例
    scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    xgb_model = xgb.XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        scale_pos_weight=scale_pos, random_state=42,
        eval_metric='auc', use_label_encoder=False
    )
    xgb_model.fit(X_train, y_train)
    y_pred_xgb = xgb_model.predict(X_test)
    y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]

    results['XGBoost'] = {
        'model': xgb_model,
        'y_pred': y_pred_xgb,
        'y_prob': y_prob_xgb,
        'auc': roc_auc_score(y_test, y_prob_xgb),
        'accuracy': accuracy_score(y_test, y_pred_xgb),
        'recall': recall_score(y_test, y_pred_xgb),
        'precision': precision_score(y_test, y_pred_xgb),
        'f1': f1_score(y_test, y_pred_xgb),
        'feature_importance': xgb_model.feature_importances_,
    }
    print(f"  AUC: {results['XGBoost']['auc']:.4f}")
    print(f"  准确率: {results['XGBoost']['accuracy']:.4f}")
    print(f"  召回率: {results['XGBoost']['recall']:.4f}")

    return results, X_test, y_test, feature_names


def plot_model_comparison(results, X_test, y_test, feature_names):
    """
    模型对比可视化

    参数:
        results: 模型评估结果字典
        X_test: 测试集特征
        y_test: 测试集标签
        feature_names: 特征名列表
    """
    print(f"\n{'='*60}")
    print("【模型对比可视化】")
    print(f"{'='*60}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 图1：ROC 曲线
    for i, (name, res) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(y_test, res['y_prob'])
        axes[0].plot(fpr, tpr, color=COLORS[i], linewidth=2,
                     label=f"{name} (AUC={res['auc']:.3f})")
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)
    axes[0].set_title('ROC 曲线对比', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].legend(fontsize=10)

    # 图2：模型性能指标对比
    metrics = ['AUC', '准确率', '召回率', '精确率', 'F1']
    metric_keys = ['auc', 'accuracy', 'recall', 'precision', 'f1']
    x = np.arange(len(metrics))
    width = 0.25

    for i, (name, res) in enumerate(results.items()):
        values = [res[k] for k in metric_keys]
        axes[1].bar(x + i * width, values, width, color=COLORS[i], label=name)

    axes[1].set_xticks(x + width)
    axes[1].set_xticklabels(metrics)
    axes[1].set_title('模型性能指标对比', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('得分')
    axes[1].set_ylim(0, 1.1)
    axes[1].legend(fontsize=9)

    # 图3：输出评估对比表
    axes[2].axis('off')
    table_data = []
    for name, res in results.items():
        table_data.append([
            name,
            f"{res['auc']:.4f}",
            f"{res['accuracy']:.4f}",
            f"{res['recall']:.4f}",
            f"{res['precision']:.4f}",
            f"{res['f1']:.4f}"
        ])
    table = axes[2].table(cellText=table_data,
                           colLabels=['模型', 'AUC', '准确率', '召回率', '精确率', 'F1'],
                           cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    # 高亮最优 AUC
    best_auc_idx = max(range(len(results)),
                        key=lambda i: list(results.values())[i]['auc'])
    for j in range(6):
        table[best_auc_idx + 1, j].set_facecolor('#FFEAA7')
    axes[2].set_title('模型性能对比表（黄色为最优AUC）', fontsize=12, fontweight='bold')

    plt.suptitle('预测模型对比', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '14_model_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] 14_model_comparison.png")


def plot_feature_importance(results, feature_names):
    """
    特征重要性可视化（取各模型平均重要性排名 Top5）

    参数:
        results: 模型评估结果字典
        feature_names: 特征名列表
    """
    print(f"\n{'='*60}")
    print("【特征重要性分析】")
    print(f"{'='*60}")

    # 汇总各模型特征重要性（排名法：各模型内排名后取平均排名）
    importance_df = pd.DataFrame({'特征': feature_names})
    for name, res in results.items():
        imp = res['feature_importance']
        # 归一化到 0-1 用于展示
        imp_norm = imp / imp.max() if imp.max() > 0 else imp
        importance_df[name] = imp_norm
        # 各模型内排名（1=最重要）
        importance_df[f'{name}_rank'] = pd.Series(imp_norm).rank(ascending=False).astype(int).values

    # 平均排名（跨模型，方法论上比直接平均归一化值更可靠）
    rank_cols = [c for c in importance_df.columns if c.endswith('_rank')]
    importance_df['平均排名'] = importance_df[rank_cols].mean(axis=1)
    importance_df = importance_df.sort_values('平均排名')
    # 平均归一化值用于可视化
    model_names = list(results.keys())
    importance_df['平均归一化'] = importance_df[model_names].mean(axis=1)

    print("\n特征重要性排名（各模型内排名后取平均排名）:")
    display_cols = ['特征'] + model_names + ['平均排名']
    print(importance_df[display_cols].to_string(index=False))

    # Top5
    top5 = importance_df.head(5)
    print(f"\nTop5 影响爆品的关键因素:")
    for i, row in top5.iterrows():
        print(f"  {row['特征']:<20s} 平均排名: {row['平均排名']:.1f}")

    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：各模型特征重要性对比
    x = np.arange(len(feature_names))
    width = 0.25
    for i, (name, res) in enumerate(results.items()):
        imp = res['feature_importance']
        imp_norm = imp / imp.max() if imp.max() > 0 else imp
        axes[0].barh(x + i * width, imp_norm, width, color=COLORS[i], label=name)
    axes[0].set_yticks(x + width)
    axes[0].set_yticklabels(feature_names)
    axes[0].set_title('各模型特征重要性对比', fontsize=12, fontweight='bold')
    axes[0].set_xlabel('归一化重要性')
    axes[0].legend(fontsize=9)
    axes[0].invert_yaxis()

    # 右图：Top5 平均特征重要性
    top5_sorted = top5.sort_values('平均归一化', ascending=True)
    colors_top = ['#FF6B6B' if i == len(top5_sorted) - 1 else '#4ECDC4'
                  for i in range(len(top5_sorted))]
    axes[1].barh(range(len(top5_sorted)), top5_sorted['平均归一化'],
                  color=colors_top, edgecolor='white')
    axes[1].set_yticks(range(len(top5_sorted)))
    axes[1].set_yticklabels(top5_sorted['特征'])
    axes[1].set_title('Top5 关键因素（平均归一化重要性）', fontsize=12, fontweight='bold')
    axes[1].set_xlabel('平均归一化重要性')
    for i, (_, row) in enumerate(top5_sorted.iterrows()):
        axes[1].text(row['平均归一化'] + 0.01, i, f"{row['平均归一化']:.3f}",
                     va='center', fontsize=10)

    plt.suptitle('特征重要性分析', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '15_feature_importance.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] 15_feature_importance.png")


def main():
    """主函数"""
    df = load_data()
    X, y, feature_names = prepare_features(df)
    results, X_test, y_test, feature_names = train_and_evaluate(X, y, feature_names)

    plot_model_comparison(results, X_test, y_test, feature_names)
    plot_feature_importance(results, feature_names)

    # 输出最优模型
    best_model = max(results.items(), key=lambda x: x[1]['auc'])
    print(f"\n最优模型: {best_model[0]} (AUC={best_model[1]['auc']:.4f})")
    print(f"\n[OK] 预测建模完成")


if __name__ == '__main__':
    main()
