# -*- coding: utf-8 -*-
"""
05_dimension_comparison.py - 爆品 vs 普通品单维度对比分析
=========================================================
功能：从价格、直播间规模、主播特征、讲解时长、时段五个维度，
      对比爆品与普通品的分布差异
输入：data/processed/burst_data.csv
输出：figures/ 下的对比分析图表
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

warnings.filterwarnings('ignore')
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
PROCESSED_DIR = os.path.join(PROJECT_DIR, 'data', 'processed')
FIGURES_DIR = os.path.join(PROJECT_DIR, 'figures')

# 配色
BURST_COLOR = '#FF6B6B'
NORMAL_COLOR = '#4ECDC4'
PALETTE = {'爆品': BURST_COLOR, '普通品': NORMAL_COLOR}
COLORS = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']



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
    """加载带爆品标记的数据"""
    path = os.path.join(PROCESSED_DIR, 'burst_data.csv')
    df = pd.read_csv(path, encoding='utf-8')
    df = restore_categorical(df)
    print(f"加载数据: {df.shape[0]} 行  x  {df.shape[1]} 列")
    return df


def compare_price(df):
    """
    维度1：价格对比
    - 价格均值/中位数/分布
    - 箱线图 + 小提琴图
    """
    print(f"\n{'='*60}")
    print("【维度1：价格对比】")
    print(f"{'='*60}")

    burst = df[df['是否爆品'] == 1]['price']
    normal = df[df['是否爆品'] == 0]['price']

    mean_b, mean_n = burst.mean(), normal.mean()
    med_b, med_n = burst.median(), normal.median()

    print(f"  爆品  均价:{mean_b:.1f}元  中位数:{med_b:.1f}元  样本:{len(burst)}")
    print(f"  普通品 均价:{mean_n:.1f}元  中位数:{med_n:.1f}元  样本:{len(normal)}")
    print(f"  均价差异: {mean_b - mean_n:+.1f}元 ({(mean_b/mean_n - 1)*100:+.1f}%)")

    # 结论
    if mean_b < mean_n:
        print(f"  → 结论: 爆品的平均价格比普通品低 {(mean_n - mean_b)/mean_n*100:.1f}%，"
              f"说明直播间爆品更偏向中低价位带。")
    else:
        print(f"  → 结论: 爆品的平均价格比普通品高 {(mean_b/mean_n - 1)*100:.1f}%，"
              f"说明直播间爆品并不依赖低价策略。")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 箱线图（限制价格范围，避免极端值影响可视化）
    price_cap = df['price'].quantile(0.95)
    df_plot = df[df['price'] <= price_cap].copy()
    sns.boxplot(data=df_plot, x='爆品标签', y='price', ax=axes[0],
                palette=PALETTE, showfliers=False)
    axes[0].set_title('价格分布对比（P95以内）', fontsize=12)
    axes[0].set_xlabel('')
    axes[0].set_ylabel('价格（元）')
    # 添加均值标注
    for i, (label, color) in enumerate(PALETTE.items()):
        subset = df_plot[df_plot['爆品标签'] == label]['price']
        axes[0].text(i, subset.mean(), f'均值={subset.mean():.0f}',
                     ha='center', va='bottom', fontsize=10, fontweight='bold', color=color)

    # 小提琴图
    sns.violinplot(data=df_plot, x='爆品标签', y='price', ax=axes[1],
                   palette=PALETTE, inner='quartile')
    axes[1].set_title('价格分布密度对比', fontsize=12)
    axes[1].set_xlabel('')
    axes[1].set_ylabel('价格（元）')

    plt.suptitle('维度1：爆品 vs 普通品 — 价格对比', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '06_price_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] 06_price_comparison.png")


def compare_duration(df):
    """
    维度2：讲解时长对比
    - 主播讲解商品的时间长短与爆品的关系
    """
    print(f"\n{'='*60}")
    print("【维度2：讲解时长对比】")
    print(f"{'='*60}")

    burst = df[df['是否爆品'] == 1]['popduration']
    normal = df[df['是否爆品'] == 0]['popduration']

    mean_b, mean_n = burst.mean(), normal.mean()
    print(f"  爆品  平均讲解时长:{mean_b:.0f}秒({mean_b/60:.1f}分钟)  样本:{len(burst)}")
    print(f"  普通品 平均讲解时长:{mean_n:.0f}秒({mean_n/60:.1f}分钟)  样本:{len(normal)}")
    print(f"  差异: {mean_b - mean_n:+.0f}秒 ({(mean_b/mean_n - 1)*100:+.1f}%)")

    if mean_b > mean_n:
        print(f"  → 结论: 爆品的平均讲解时长比普通品多 {(mean_b - mean_n):.0f}秒，"
              f"说明更充分的讲解有助于提升销量。")

    # 讲解时长区间 vs 爆品率
    if '讲解时长区间' in df.columns:
        duration_burst = df.groupby('讲解时长区间', observed=True).agg(
            total=('是否爆品', 'count'),
            burst_count=('是否爆品', 'sum')
        )
        duration_burst['爆品率'] = (duration_burst['burst_count'] / duration_burst['total'] * 100).round(2)
        print(f"\n各讲解时长区间的爆品率:")
        print(duration_burst.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 箱线图
    sns.boxplot(data=df, x='爆品标签', y='popduration', ax=axes[0],
                palette=PALETTE, showfliers=False)
    axes[0].set_title('讲解时长分布对比', fontsize=12)
    axes[0].set_xlabel('')
    axes[0].set_ylabel('讲解时长（秒）')

    # 讲解时长区间 vs 爆品率柱状图
    if '讲解时长区间' in df.columns:
        dur_data = df.groupby('讲解时长区间', observed=True).agg(
            total=('是否爆品', 'count'),
            burst_count=('是否爆品', 'sum')
        ).reset_index()
        dur_data['爆品率'] = dur_data['burst_count'] / dur_data['total'] * 100

        bars = axes[1].bar(range(len(dur_data)), dur_data['爆品率'],
                           color=COLORS[:len(dur_data)], edgecolor='white')
        axes[1].set_xticks(range(len(dur_data)))
        axes[1].set_xticklabels(dur_data['讲解时长区间'], rotation=15)
        axes[1].set_title('各讲解时长区间的爆品率', fontsize=12)
        axes[1].set_ylabel('爆品率（%）')
        # 添加平均线
        avg_rate = df['是否爆品'].mean() * 100
        axes[1].axhline(avg_rate, color='red', linestyle='--', label=f'平均爆品率={avg_rate:.1f}%')
        axes[1].legend()
        for bar, val in zip(bars, dur_data['爆品率']):
            axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                         f'{val:.1f}%', ha='center', fontsize=10)

    plt.suptitle('维度2：爆品 vs 普通品 — 讲解时长对比', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '07_duration_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] 07_duration_comparison.png")


def compare_anchor(df):
    """
    维度3：主播特征对比
    - 不同主播的爆品率
    - 观众数（userbefore）与销量的关系
    """
    print(f"\n{'='*60}")
    print("【维度3：主播特征对比】")
    print(f"{'='*60}")

    # 各主播爆品率
    anchor_stats = df.groupby('uid').agg(
        total_products=('是否爆品', 'count'),
        burst_count=('是否爆品', 'sum'),
        avg_sales=('sales', 'mean'),
        avg_viewers=('userbefore', 'mean'),
        avg_price=('price', 'mean')
    )
    anchor_stats['爆品率'] = (anchor_stats['burst_count'] / anchor_stats['total_products'] * 100).round(1)
    anchor_stats = anchor_stats.sort_values('爆品率', ascending=False)

    print(f"\n各主播爆品率排名（共 {len(anchor_stats)} 位主播）:")
    print(anchor_stats.to_string())

    # 使用直播间规模字段对比（与报告和README保持一致）
    if '直播间规模' in df.columns:
        size_burst = df.groupby('直播间规模', observed=True).agg(
            total=('是否爆品', 'count'),
            burst_count=('是否爆品', 'sum')
        )
        size_burst['爆品率'] = (size_burst['burst_count'] / size_burst['total'] * 100).round(2)
        print(f"\n各直播间规模爆品率:")
        print(size_burst['爆品率'].to_string())

        best_size = size_burst['爆品率'].idxmax()
        worst_size = size_burst['爆品率'].idxmin()
        best_rate = size_burst.loc[best_size, '爆品率']
        worst_rate = max(size_burst.loc[worst_size, '爆品率'], 0.01)
        print(f"  → {best_size} 爆品率是 {worst_size} 的 {best_rate / worst_rate:.0f} 倍")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 散点图：观众数 vs 销量
    scatter = axes[0].scatter(df['userbefore'], df['sales'],
                               c=df['是否爆品'].map({1: BURST_COLOR, 0: '#cccccc'}),
                               alpha=0.3, s=10, edgecolors='none')
    axes[0].set_title('观众数 vs 销量', fontsize=12)
    axes[0].set_xlabel('直播间观众数')
    axes[0].set_ylabel('销量')
    # 添加图例
    from matplotlib.lines import Line2D
    legend_elements = [Line2D([0], [0], marker='o', color='w', markerfacecolor=BURST_COLOR,
                              markersize=8, label='爆品'),
                       Line2D([0], [0], marker='o', color='w', markerfacecolor='#cccccc',
                              markersize=8, label='普通品')]
    axes[0].legend(handles=legend_elements)

    # 主播爆品率柱状图（取前10）
    top_anchors = anchor_stats.head(10)
    x_labels = [str(int(uid))[-4:] for uid in top_anchors.index]  # 显示UID后4位
    bars = axes[1].bar(range(len(top_anchors)), top_anchors['爆品率'],
                       color=COLORS[:len(top_anchors)], edgecolor='white')
    axes[1].set_xticks(range(len(top_anchors)))
    axes[1].set_xticklabels(x_labels, rotation=45)
    axes[1].set_title('主播爆品率 Top10', fontsize=12)
    axes[1].set_ylabel('爆品率（%）')
    axes[1].set_xlabel('主播ID（后4位）')
    for bar, val in zip(bars, top_anchors['爆品率']):
        axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                     f'{val:.1f}%', ha='center', fontsize=9)

    plt.suptitle('维度3：爆品 vs 普通品 — 主播特征对比', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '08_anchor_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] 08_anchor_comparison.png")


def compare_body_movement(df):
    """
    维度4：主播身体动作对比
    - 论文核心变量：avg_dis（主播平均位移）与销量的关系
    - 验证论文的倒 U 型假设
    """
    print(f"\n{'='*60}")
    print("【维度4：主播身体动作(avg_dis)对比】")
    print(f"{'='*60}")

    burst = df[df['是否爆品'] == 1]['avg_dis']
    normal = df[df['是否爆品'] == 0]['avg_dis']

    mean_b, mean_n = burst.mean(), normal.mean()
    print(f"  爆品  平均位移:{mean_b:.2f}  样本:{len(burst)}")
    print(f"  普通品 平均位移:{mean_n:.2f}  样本:{len(normal)}")

    # 位移与销量的相关性
    r, p = stats.pearsonr(df['avg_dis'], df['sales'])
    print(f"  avg_dis 与 sales 的皮尔逊相关系数: r={r:.4f}, p={p:.2e}")

    # 检验倒U型关系：二次项回归
    x = df['avg_dis'].values
    y = df['sales'].values
    # 多项式拟合（2次）
    coeffs = np.polyfit(x, y, 2)
    print(f"  二次拟合: sales = {coeffs[0]:.4f} * avg_dis² + {coeffs[1]:.4f} * avg_dis + {coeffs[2]:.4f}")
    if coeffs[0] < 0:
        print(f"  → 二次项系数为负 ({coeffs[0]:.4f})，支持倒U型关系：适度的主播动作最有利于销量。")
    else:
        print(f"  → 二次项系数为正，不支持倒U型关系。")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 散点图 + 拟合曲线
    axes[0].scatter(df['avg_dis'], df['sales'],
                    c=df['是否爆品'].map({1: BURST_COLOR, 0: '#cccccc'}),
                    alpha=0.3, s=10, edgecolors='none')
    # 拟合二次曲线
    x_line = np.linspace(x.min(), x.max(), 200)
    y_line = np.polyval(coeffs, x_line)
    axes[0].plot(x_line, y_line, 'r-', linewidth=2, label=f'二次拟合')
    axes[0].set_title('主播身体动作位移 vs 销量', fontsize=12)
    axes[0].set_xlabel('avg_dis（主播平均位移）')
    axes[0].set_ylabel('销量')
    axes[0].legend()

    # 箱线图
    sns.boxplot(data=df, x='爆品标签', y='avg_dis', ax=axes[1],
                palette=PALETTE, showfliers=False)
    axes[1].set_title('身体动作位移分布对比', fontsize=12)
    axes[1].set_xlabel('')
    axes[1].set_ylabel('avg_dis')

    plt.suptitle('维度4：爆品 vs 普通品 — 主播身体动作对比', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '09_movement_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] 09_movement_comparison.png")


def compare_time(df):
    """
    维度5：时段对比
    - 不同时段的爆品率
    - 直播日期/星期与销量关系
    """
    print(f"\n{'='*60}")
    print("【维度5：时段/时间对比】")
    print(f"{'='*60}")

    # 时段爆品率
    if '时段' in df.columns:
        time_stats = df.groupby('时段').agg(
            total=('是否爆品', 'count'),
            burst_count=('是否爆品', 'sum'),
            avg_sales=('sales', 'mean')
        )
        time_stats['爆品率'] = (time_stats['burst_count'] / time_stats['total'] * 100).round(2)
        print(f"\n各时段爆品率:")
        print(time_stats.to_string())

    # 星期几爆品率
    if 'weekday' in df.columns:
        weekday_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                         'Friday', 'Saturday', 'Sunday']
        weekday_cn = {'Monday': '周一', 'Tuesday': '周二', 'Wednesday': '周三',
                      'Thursday': '周四', 'Friday': '周五', 'Saturday': '周六', 'Sunday': '周日'}
        wd_stats = df.groupby('weekday').agg(
            total=('是否爆品', 'count'),
            burst_count=('是否爆品', 'sum'),
            avg_sales=('sales', 'mean')
        )
        wd_stats['爆品率'] = (wd_stats['burst_count'] / wd_stats['total'] * 100).round(2)
        wd_stats = wd_stats.reindex([d for d in weekday_order if d in wd_stats.index])
        print(f"\n各星期爆品率:")
        print(wd_stats.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 时段爆品率柱状图
    if '时段' in df.columns:
        time_data = df.groupby('时段').agg(
            total=('是否爆品', 'count'),
            burst_count=('是否爆品', 'sum')
        ).reset_index()
        time_data['爆品率'] = time_data['burst_count'] / time_data['total'] * 100

        bars = axes[0].bar(time_data['时段'], time_data['爆品率'],
                           color=COLORS[:len(time_data)], edgecolor='white')
        avg_rate = df['是否爆品'].mean() * 100
        axes[0].axhline(avg_rate, color='red', linestyle='--', label=f'平均={avg_rate:.1f}%')
        axes[0].set_title('各时段爆品率', fontsize=12)
        axes[0].set_ylabel('爆品率（%）')
        axes[0].legend()
        for bar, val in zip(bars, time_data['爆品率']):
            axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                         f'{val:.1f}%', ha='center', fontsize=10)

    # 每日销量趋势（如有日期数据）
    if 'date' in df.columns:
        daily = df.groupby('date')['sales'].agg(['mean', 'sum', 'count']).reset_index()
        daily['date'] = pd.to_datetime(daily['date'])
        axes[1].plot(daily['date'], daily['mean'], color=BURST_COLOR,
                     marker='o', markersize=4, linewidth=1.5, label='日均销量')
        axes[1].set_title('每日平均销量趋势（2022年3月）', fontsize=12)
        axes[1].set_xlabel('日期')
        axes[1].set_ylabel('平均销量')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].legend()

    plt.suptitle('维度5：爆品 vs 普通品 — 时段/时间对比', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '10_time_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] 10_time_comparison.png")


def main():
    """主函数"""
    df = load_data()
    compare_price(df)
    compare_duration(df)
    compare_anchor(df)
    compare_body_movement(df)
    compare_time(df)
    print(f"\n[OK] 单维度对比分析完成，图表保存在 {FIGURES_DIR}")


if __name__ == '__main__':
    main()
