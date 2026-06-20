# -*- coding: utf-8 -*-
"""
03_eda_analysis.py - 探索性数据分析 (EDA)
==========================================
功能：加载清洗后数据，输出描述性统计、分布图、相关性分析
输入：data/processed/cleaned_data.csv
输出：figures/ 下的多张图表
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

# ========== 中文字体配置 ==========
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

# ========== 路径配置 ==========
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(PROJECT_DIR, 'data', 'processed')
FIGURES_DIR = os.path.join(PROJECT_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

# ========== 图表风格 ==========
sns.set_style('whitegrid')
COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']


def load_data():
    """加载清洗后的数据"""
    path = os.path.join(PROCESSED_DIR, 'cleaned_data.csv')
    if not os.path.exists(path):
        raise FileNotFoundError("请先运行 02_data_cleaner.py")
    df = pd.read_csv(path, encoding='utf-8')
    print(f"加载数据: {df.shape[0]} 行  x  {df.shape[1]} 列")
    return df


def eda_overview(df):
    """
    数据概况：描述性统计 + 基本分布

    参数:
        df: 数据框
    """
    print(f"\n{'='*60}")
    print("【数据概况】")
    print(f"{'='*60}")

    # 核心数值字段的描述性统计
    numeric_cols = ['price', 'userbefore', 'sales', 'popduration', 'avg_dis']
    existing_cols = [c for c in numeric_cols if c in df.columns]

    stats = df[existing_cols].describe().T
    stats['median'] = df[existing_cols].median()
    stats['skew'] = df[existing_cols].skew()
    stats['kurtosis'] = df[existing_cols].kurtosis()

    print("\n核心字段描述性统计:")
    print(stats[['count', 'mean', 'median', 'std', 'min', 'max', 'skew']].to_string())

    # 各字段分布图
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for i, col in enumerate(existing_cols):
        ax = axes[i]
        df[col].hist(bins=50, ax=ax, color=COLORS[i], alpha=0.7, edgecolor='white')
        ax.set_title(f'{col} 分布', fontsize=12)
        ax.set_xlabel(col)
        ax.set_ylabel('频次')
        # 添加均值线
        mean_val = df[col].mean()
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'均值={mean_val:.1f}')
        ax.legend(fontsize=9)

    # 销量对数分布（因为 sales 右偏严重）
    if 'sales' in df.columns:
        ax = axes[len(existing_cols)]
        sales_positive = df[df['sales'] > 0]['sales']
        np.log1p(sales_positive).hist(bins=50, ax=ax, color=COLORS[len(existing_cols)],
                                       alpha=0.7, edgecolor='white')
        ax.set_title('log(sales+1) 分布', fontsize=12)
        ax.set_xlabel('log(sales+1)')
        ax.set_ylabel('频次')

    # 隐藏多余的子图
    for j in range(len(existing_cols) + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('数据分布概览', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '01_distributions.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] 01_distributions.png")


def eda_correlation(df):
    """
    相关性分析：数值字段的相关系数热力图

    参数:
        df: 数据框
    """
    print(f"\n{'='*60}")
    print("【相关性分析】")
    print(f"{'='*60}")

    numeric_cols = ['price', 'userbefore', 'sales', 'popduration', 'avg_dis', 'no_pop']
    existing_cols = [c for c in numeric_cols if c in df.columns]

    corr = df[existing_cols].corr()
    print("\n皮尔逊相关系数矩阵:")
    print(corr.to_string())

    # 热力图
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.3f', cmap='RdBu_r',
                center=0, square=True, ax=ax,
                cbar_kws={'shrink': 0.8, 'label': '相关系数'})
    ax.set_title('数值变量相关系数矩阵', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '02_correlation.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] 02_correlation.png")

    # 输出与 sales 最相关的变量
    if 'sales' in corr.columns:
        sales_corr = corr['sales'].drop('sales').sort_values(key=abs, ascending=False)
        print("\n与销量相关性排序:")
        for var, val in sales_corr.items():
            direction = "正相关" if val > 0 else "负相关"
            print(f"  {var:<20s} r = {val:+.4f} ({direction})")


def eda_categorical(df):
    """
    分类字段分析：时段分布、价格区间分布

    参数:
        df: 数据框
    """
    print(f"\n{'='*60}")
    print("【分类字段分析】")
    print(f"{'='*60}")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 时段分布
    if '时段' in df.columns:
        time_stats = df.groupby('时段')['sales'].agg(['count', 'mean', 'median']).round(1)
        print("\n各时段销量统计:")
        print(time_stats.to_string())

        time_counts = df['时段'].value_counts()
        axes[0].bar(time_counts.index, time_counts.values, color=COLORS[:len(time_counts)])
        axes[0].set_title('各时段商品数量', fontsize=12)
        axes[0].set_ylabel('商品数量')
        for i, (idx, val) in enumerate(time_counts.items()):
            axes[0].text(i, val + 50, str(val), ha='center', fontsize=10)

    # 价格区间分布
    if '价格区间' in df.columns:
        price_stats = df.groupby('价格区间', observed=True)['sales'].agg(['count', 'mean', 'median']).round(1)
        print("\n各价格区间销量统计:")
        print(price_stats.to_string())

        price_counts = df['价格区间'].value_counts().sort_index()
        axes[1].bar(range(len(price_counts)), price_counts.values,
                    color=COLORS[:len(price_counts)])
        axes[1].set_xticks(range(len(price_counts)))
        axes[1].set_xticklabels(price_counts.index, rotation=30)
        axes[1].set_title('各价格区间商品数量', fontsize=12)
        axes[1].set_ylabel('商品数量')

    # 直播间规模分布
    if '直播间规模' in df.columns:
        size_stats = df.groupby('直播间规模', observed=True)['sales'].agg(['count', 'mean', 'median']).round(1)
        print("\n各直播间规模销量统计:")
        print(size_stats.to_string())

        size_counts = df['直播间规模'].value_counts().sort_index()
        axes[2].bar(range(len(size_counts)), size_counts.values,
                    color=COLORS[:len(size_counts)])
        axes[2].set_xticks(range(len(size_counts)))
        axes[2].set_xticklabels(size_counts.index, rotation=30)
        axes[2].set_title('各直播间规模商品数量', fontsize=12)
        axes[2].set_ylabel('商品数量')

    plt.suptitle('分类字段分布', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '03_categorical.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] 03_categorical.png")


def eda_sales_distribution(df):
    """
    销量分布深度分析

    参数:
        df: 数据框
    """
    print(f"\n{'='*60}")
    print("【销量分布深度分析】")
    print(f"{'='*60}")

    sales = df['sales']
    print(f"  总商品数: {len(sales)}")
    print(f"  零销量商品: {(sales == 0).sum()} ({(sales == 0).mean()*100:.1f}%)")
    print(f"  非零销量商品: {(sales > 0).sum()} ({(sales > 0).mean()*100:.1f}%)")
    print(f"  均值: {sales.mean():.1f}")
    print(f"  中位数: {sales.median():.1f}")
    print(f"  标准差: {sales.std():.1f}")
    print(f"  P90: {sales.quantile(0.9):.1f}")
    print(f"  P95: {sales.quantile(0.95):.1f}")
    print(f"  P99: {sales.quantile(0.99):.1f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：销量分布（含零值）
    axes[0].hist(sales, bins=100, color=COLORS[0], alpha=0.7, edgecolor='white')
    axes[0].set_title('销量分布（含零值）', fontsize=12)
    axes[0].set_xlabel('销量')
    axes[0].set_ylabel('频次')
    axes[0].axvline(sales.mean(), color='red', linestyle='--', label=f'均值={sales.mean():.0f}')
    axes[0].axvline(sales.quantile(0.95), color='blue', linestyle='--', label=f'P95={sales.quantile(0.95):.0f}')
    axes[0].legend()

    # 右图：非零销量的 log 分布
    nonzero_sales = sales[sales > 0]
    log_sales = np.log10(nonzero_sales)
    axes[1].hist(log_sales, bins=50, color=COLORS[1], alpha=0.7, edgecolor='white')
    axes[1].set_title('非零销量 log10 分布', fontsize=12)
    axes[1].set_xlabel('log10(sales)')
    axes[1].set_ylabel('频次')

    plt.suptitle('销量分布分析', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '04_sales_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] 04_sales_distribution.png")


def main():
    """主函数"""
    df = load_data()
    eda_overview(df)
    eda_correlation(df)
    eda_categorical(df)
    eda_sales_distribution(df)
    print(f"\n[OK] EDA 完成，图表保存在 {FIGURES_DIR}")


if __name__ == '__main__':
    main()
