# -*- coding: utf-8 -*-
"""
04_burst_definition.py - 爆品定义模块
======================================
功能：用两种方法定义"爆品"，输出爆品标记后的数据
方法A：相对排名法 — 销量排名前5%
方法B：绝对阈值法 — 销量 > 均值 + 2倍标准差
输入：data/processed/cleaned_data.csv
输出：data/processed/burst_data.csv（含爆品标记列）
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(PROJECT_DIR, 'data', 'processed')
FIGURES_DIR = os.path.join(PROJECT_DIR, 'figures')


def load_data():
    """加载清洗后的数据"""
    path = os.path.join(PROCESSED_DIR, 'cleaned_data.csv')
    df = pd.read_csv(path, encoding='utf-8')
    print(f"加载数据: {df.shape[0]} 行  x  {df.shape[1]} 列")
    return df


def define_burst_products(df):
    """
    用两种方法定义爆品

    参数:
        df: 数据框

    返回:
        pd.DataFrame: 增加爆品标记列的数据框
    """
    print(f"\n{'='*60}")
    print("【爆品定义】")
    print(f"{'='*60}")

    sales = df['sales']

    # ---- 方法A：相对排名法（前5%）----
    threshold_rank = sales.quantile(0.95)
    df['爆品_排名法'] = (sales >= threshold_rank).astype(int)
    count_a = df['爆品_排名法'].sum()
    print(f"\n方法A（相对排名法 - 前5%）:")
    print(f"  阈值: 销量 >= {threshold_rank:.0f}")
    print(f"  爆品数量: {count_a} ({count_a/len(df)*100:.1f}%)")
    print(f"  普通品数量: {len(df) - count_a} ({(len(df)-count_a)/len(df)*100:.1f}%)")

    # ---- 方法B：绝对阈值法（均值 + 2σ）----
    mean_sales = sales.mean()
    std_sales = sales.std()
    threshold_abs = mean_sales + 2 * std_sales
    df['爆品_阈值法'] = (sales >= threshold_abs).astype(int)
    count_b = df['爆品_阈值法'].sum()
    print(f"\n方法B（绝对阈值法 - 均值+2σ）:")
    print(f"  均值: {mean_sales:.1f}, 标准差: {std_sales:.1f}")
    print(f"  阈值: 销量 >= {threshold_abs:.0f}")
    print(f"  爆品数量: {count_b} ({count_b/len(df)*100:.1f}%)")
    print(f"  普通品数量: {len(df) - count_b} ({(len(df)-count_b)/len(df)*100:.1f}%)")

    # ---- 交集与差集 ----
    both = ((df['爆品_排名法'] == 1) & (df['爆品_阈值法'] == 1)).sum()
    only_rank = ((df['爆品_排名法'] == 1) & (df['爆品_阈值法'] == 0)).sum()
    only_thresh = ((df['爆品_排名法'] == 0) & (df['爆品_阈值法'] == 1)).sum()

    print(f"\n两种定义对比:")
    print(f"  交集（两种方法都认定为爆品）: {both}")
    print(f"  仅排名法认定: {only_rank}")
    print(f"  仅阈值法认定: {only_thresh}")

    # ---- 选择排名法作为主定义（更符合业务直觉）----
    df['是否爆品'] = df['爆品_排名法']
    df['爆品标签'] = df['是否爆品'].map({1: '爆品', 0: '普通品'})
    print(f"\n[结论] 主分析采用方法A（相对排名法），爆品占比约5%")

    # ---- 可视化 ----
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 图1：两种定义的 Venn 图（用柱状图替代）
    categories = ['仅排名法', '两者交集', '仅阈值法']
    values = [only_rank, both, only_thresh]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    bars = axes[0].bar(categories, values, color=colors, edgecolor='white')
    axes[0].set_title('两种爆品定义的交集与差集', fontsize=12)
    axes[0].set_ylabel('商品数量')
    for bar, val in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 10,
                     str(val), ha='center', fontsize=11, fontweight='bold')

    # 图2：销量分布 + 两条阈值线
    axes[1].hist(sales, bins=100, color='lightgray', alpha=0.7, edgecolor='white',
                 label='全部商品')
    axes[1].axvline(threshold_rank, color='red', linestyle='--', linewidth=2,
                    label=f'排名法阈值={threshold_rank:.0f}')
    axes[1].axvline(threshold_abs, color='blue', linestyle='--', linewidth=2,
                    label=f'阈值法阈值={threshold_abs:.0f}')
    axes[1].set_xlim(0, sales.quantile(0.99) * 1.2)
    axes[1].set_title('销量分布与爆品阈值', fontsize=12)
    axes[1].set_xlabel('销量')
    axes[1].set_ylabel('频次')
    axes[1].legend(fontsize=9)

    # 图3：爆品 vs 普通品的销量箱线图（对数刻度）
    df_plot = df.copy()
    df_plot['log_sales'] = np.log1p(df_plot['sales'])
    sns.boxplot(data=df_plot, x='爆品标签', y='log_sales', ax=axes[2],
                palette=['#FF6B6B', '#4ECDC4'])
    axes[2].set_title('爆品 vs 普通品 销量分布', fontsize=12)
    axes[2].set_xlabel('')
    axes[2].set_ylabel('log(sales + 1)')

    plt.suptitle('爆品定义与分布', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '05_burst_definition.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] 05_burst_definition.png")

    return df


def main():
    """主函数"""
    df = load_data()
    df = define_burst_products(df)

    # 保存带爆品标记的数据
    output_path = os.path.join(PROCESSED_DIR, 'burst_data.csv')
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n[OK] 爆品标记数据已保存至: {output_path}")

    return df


if __name__ == '__main__':
    main()
