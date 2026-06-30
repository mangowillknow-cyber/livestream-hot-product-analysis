# -*- coding: utf-8 -*-
"""
06_radar_profile.py - 爆品多维画像 + 雷达图
=============================================
功能：综合所有维度生成爆品画像，用雷达图展示爆品与普通品的差异
输入：data/processed/burst_data.csv
输出：figures/radar_chart.png, figures/profile_card.png
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

BURST_COLOR = '#FF6B6B'
NORMAL_COLOR = '#4ECDC4'



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


def build_radar_chart(df):
    """
    构建雷达图：爆品 vs 普通品在各维度的标准化得分

    将各维度归一化到 0-1，比较两组的差异
    """
    print(f"\n{'='*60}")
    print("【爆品多维画像 - 雷达图】")
    print(f"{'='*60}")

    burst = df[df['是否爆品'] == 1]
    normal = df[df['是否爆品'] == 0]

    # 选取雷达图维度
    dimensions = {
        '平均价格': ('price', 'mean'),
        '观众规模': ('userbefore', 'mean'),
        '讲解时长': ('popduration', 'mean'),
        '主播动作强度': ('avg_dis', 'mean'),
        '非弹出比例': ('no_pop', 'mean'),
    }

    burst_scores = {}
    normal_scores = {}

    for label, (col, agg) in dimensions.items():
        b_val = getattr(burst[col], agg)()
        n_val = getattr(normal[col], agg)()
        burst_scores[label] = b_val
        normal_scores[label] = n_val
        print(f"  {label}: 爆品={b_val:.2f}, 普通品={n_val:.2f}")

    # 标准化到 0-1（使用全局 min-max）
    all_values = list(burst_scores.values()) + list(normal_scores.values())
    min_val = min(all_values)
    max_val = max(all_values)
    range_val = max_val - min_val if max_val != min_val else 1

    burst_norm = {k: (v - min_val) / range_val for k, v in burst_scores.items()}
    normal_norm = {k: (v - min_val) / range_val for k, v in normal_scores.items()}

    # 绘制雷达图
    labels = list(burst_norm.keys())
    burst_values = list(burst_norm.values())
    normal_values = list(normal_norm.values())

    # 闭合多边形
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    burst_values += burst_values[:1]
    normal_values += normal_values[:1]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    ax.fill(angles, burst_values, alpha=0.25, color=BURST_COLOR)
    ax.plot(angles, burst_values, 'o-', color=BURST_COLOR, linewidth=2, label='爆品')
    ax.fill(angles, normal_values, alpha=0.25, color=NORMAL_COLOR)
    ax.plot(angles, normal_values, 'o-', color=NORMAL_COLOR, linewidth=2, label='普通品')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_title('爆品 vs 普通品 多维画像对比', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1.1), fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '11_radar_chart.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] 11_radar_chart.png")

    return burst_scores, normal_scores


def build_profile_card(df, burst_scores, normal_scores):
    """
    生成爆品特征画像卡（一页纸格式）

    参数:
        df: 数据框
        burst_scores: 爆品各维度得分
        normal_scores: 普通品各维度得分
    """
    print(f"\n{'='*60}")
    print("【爆品特征画像卡】")
    print(f"{'='*60}")

    burst = df[df['是否爆品'] == 1]
    total = len(df)

    # 画像卡关键数据
    card_data = {
        '爆品数量': f"{len(burst)} 个（占比 {len(burst)/total*100:.1f}%）",
        '爆品平均销量': f"{burst['sales'].mean():.0f} 件",
        '爆品平均价格': f"{burst['price'].mean():.0f} 元",
        '爆品中位价格': f"{burst['price'].median():.0f} 元",
        '爆品平均观众数': f"{burst['userbefore'].mean():.0f} 人",
        '爆品平均讲解时长': f"{burst['popduration'].mean()/60:.1f} 分钟",
        '爆品平均动作位移': f"{burst['avg_dis'].mean():.2f}",
    }

    print("\n爆品特征画像卡:")
    print("-" * 50)
    for k, v in card_data.items():
        print(f"  {k:<15s}: {v}")
    print("-" * 50)

    # 与普通品对比的关键差异
    normal = df[df['是否爆品'] == 0]
    print("\n与普通品的关键差异:")
    comparisons = [
        ('价格', burst['price'].mean(), normal['price'].mean(), '元'),
        ('观众数', burst['userbefore'].mean(), normal['userbefore'].mean(), '人'),
        ('讲解时长', burst['popduration'].mean(), normal['popduration'].mean(), '秒'),
        ('动作位移', burst['avg_dis'].mean(), normal['avg_dis'].mean(), ''),
    ]
    for name, b_val, n_val, unit in comparisons:
        diff_pct = (b_val / n_val - 1) * 100 if n_val != 0 else 0
        direction = "高" if diff_pct > 0 else "低"
        print(f"  {name}: 爆品 {b_val:.1f}{unit} vs 普通品 {n_val:.1f}{unit} "
              f"({direction} {abs(diff_pct):.1f}%)")

    # 最优价格带分析
    if '价格区间' in df.columns:
        price_burst = df.groupby('价格区间', observed=True).agg(
            total=('是否爆品', 'count'),
            burst_count=('是否爆品', 'sum')
        )
        price_burst['爆品率'] = (price_burst['burst_count'] / price_burst['total'] * 100).round(2)
        best_price = price_burst['爆品率'].idxmax()
        best_rate = price_burst['爆品率'].max()
        print(f"\n最优价格带: {best_price}元（爆品率 {best_rate}%）")
        print("\n各价格区间爆品率:")
        print(price_burst.to_string())

    # 生成可视化画像卡
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 左上：价格区间爆品率
    if '价格区间' in df.columns:
        price_data = df.groupby('价格区间', observed=True).agg(
            total=('是否爆品', 'count'),
            burst_count=('是否爆品', 'sum')
        ).reset_index()
        price_data['爆品率'] = price_data['burst_count'] / price_data['total'] * 100
        colors = ['#FF6B6B' if r == price_data['爆品率'].max() else '#4ECDC4'
                  for r in price_data['爆品率']]
        bars = axes[0, 0].bar(range(len(price_data)), price_data['爆品率'],
                               color=colors, edgecolor='white')
        axes[0, 0].set_xticks(range(len(price_data)))
        axes[0, 0].set_xticklabels(price_data['价格区间'], rotation=20)
        axes[0, 0].set_title('各价格区间爆品率', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('爆品率（%）')
        avg = df['是否爆品'].mean() * 100
        axes[0, 0].axhline(avg, color='gray', linestyle='--', alpha=0.5)
        for bar, val in zip(bars, price_data['爆品率']):
            axes[0, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                             f'{val:.1f}%', ha='center', fontsize=9)

    # 右上：直播间规模爆品率
    if '直播间规模' in df.columns:
        size_data = df.groupby('直播间规模', observed=True).agg(
            total=('是否爆品', 'count'),
            burst_count=('是否爆品', 'sum')
        ).reset_index()
        size_data['爆品率'] = size_data['burst_count'] / size_data['total'] * 100
        colors = ['#FF6B6B' if r == size_data['爆品率'].max() else '#45B7D1'
                  for r in size_data['爆品率']]
        bars = axes[0, 1].bar(range(len(size_data)), size_data['爆品率'],
                               color=colors, edgecolor='white')
        axes[0, 1].set_xticks(range(len(size_data)))
        axes[0, 1].set_xticklabels(size_data['直播间规模'], rotation=20)
        axes[0, 1].set_title('各直播间规模爆品率', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('爆品率（%）')
        axes[0, 1].axhline(avg, color='gray', linestyle='--', alpha=0.5)
        for bar, val in zip(bars, size_data['爆品率']):
            axes[0, 1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                             f'{val:.1f}%', ha='center', fontsize=9)

    # 左下：讲解时长区间爆品率
    if '讲解时长区间' in df.columns:
        dur_data = df.groupby('讲解时长区间', observed=True).agg(
            total=('是否爆品', 'count'),
            burst_count=('是否爆品', 'sum')
        ).reset_index()
        dur_data['爆品率'] = dur_data['burst_count'] / dur_data['total'] * 100
        colors = ['#FF6B6B' if r == dur_data['爆品率'].max() else '#96CEB4'
                  for r in dur_data['爆品率']]
        bars = axes[1, 0].bar(range(len(dur_data)), dur_data['爆品率'],
                               color=colors, edgecolor='white')
        axes[1, 0].set_xticks(range(len(dur_data)))
        axes[1, 0].set_xticklabels(dur_data['讲解时长区间'], rotation=15)
        axes[1, 0].set_title('各讲解时长区间爆品率', fontsize=12, fontweight='bold')
        axes[1, 0].set_ylabel('爆品率（%）')
        axes[1, 0].axhline(avg, color='gray', linestyle='--', alpha=0.5)
        for bar, val in zip(bars, dur_data['爆品率']):
            axes[1, 0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                             f'{val:.1f}%', ha='center', fontsize=9)

    # 右下：关键指标对比（标准化柱状图）
    metrics = ['price', 'userbefore', 'popduration', 'avg_dis']
    metric_names = ['价格', '观众数', '讲解时长', '动作位移']
    burst_means = [burst[m].mean() for m in metrics]
    normal_means = [normal[m].mean() for m in metrics]
    # 标准化到 0-1
    max_vals = [max(b, n) for b, n in zip(burst_means, normal_means)]
    burst_pct = [b / m * 100 if m > 0 else 0 for b, m in zip(burst_means, max_vals)]
    normal_pct = [n / m * 100 if m > 0 else 0 for n, m in zip(normal_means, max_vals)]

    x = np.arange(len(metric_names))
    width = 0.35
    axes[1, 1].bar(x - width / 2, burst_pct, width, color=BURST_COLOR, label='爆品')
    axes[1, 1].bar(x + width / 2, normal_pct, width, color=NORMAL_COLOR, label='普通品')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(metric_names)
    axes[1, 1].set_title('爆品 vs 普通品 关键指标对比（标准化）', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('相对得分（%）')
    axes[1, 1].legend()

    plt.suptitle('爆品特征画像卡', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '12_profile_card.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] 12_profile_card.png")


def main():
    """主函数"""
    df = load_data()
    burst_scores, normal_scores = build_radar_chart(df)
    build_profile_card(df, burst_scores, normal_scores)
    print(f"\n[OK] 爆品画像生成完成")


if __name__ == '__main__':
    main()
