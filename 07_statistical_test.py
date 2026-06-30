# -*- coding: utf-8 -*-
"""
07_statistical_test.py - 统计检验模块
======================================
功能：对爆品与普通品的差异进行统计显著性检验
假设1：爆品的平均价格显著低于普通品（T检验/Mann-Whitney U）
假设2：爆品的讲解时长显著高于普通品
假设3：不同直播间规模的爆品率存在显著差异（卡方检验）
输入：data/processed/burst_data.csv
输出：统计检验结果打印 + figures/统计检验图表
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


def check_normality(data, name):
    """
    Shapiro-Wilk 正态性检验

    参数:
        data: 一维数据
        name: 变量名

    返回:
        bool: True=近似正态
    """
    # 样本大于5000时用 K-S 检验
    if len(data) > 5000:
        stat, p = stats.kstest(data, 'norm', args=(data.mean(), data.std()))
        test_name = 'K-S检验'
    else:
        stat, p = stats.shapiro(data[:5000])  # shapiro 限制样本量
        test_name = 'Shapiro-Wilk'

    is_normal = p > 0.05
    print(f"  {name} {test_name}: 统计量={stat:.4f}, p={p:.4e}, "
          f"{'近似正态' if is_normal else '非正态'}")
    return is_normal


def hypothesis_test_1(df):
    """
    假设1：爆品的平均价格显著低于普通品

    H0: 爆品价格 >= 普通品价格
    H1: 爆品价格 < 普通品价格（单侧检验）
    """
    print(f"\n{'='*60}")
    print("【假设1：爆品价格是否显著低于普通品？】")
    print(f"{'='*60}")

    burst_price = df[df['是否爆品'] == 1]['price']
    normal_price = df[df['是否爆品'] == 0]['price']

    # 描述性统计
    print(f"\n描述性统计:")
    print(f"  爆品 (n={len(burst_price)}): 均值={burst_price.mean():.2f}, "
          f"中位数={burst_price.median():.2f}, 标准差={burst_price.std():.2f}")
    print(f"  普通品 (n={len(normal_price)}): 均值={normal_price.mean():.2f}, "
          f"中位数={normal_price.median():.2f}, 标准差={normal_price.std():.2f}")

    # 正态性检验
    print(f"\n正态性检验:")
    is_normal_b = check_normality(burst_price, '爆品价格')
    is_normal_n = check_normality(normal_price, '普通品价格')

    # 选择检验方法
    if is_normal_b and is_normal_n:
        # 正态 -> 独立样本T检验（单侧）
        t_stat, p_two = stats.ttest_ind(burst_price, normal_price, equal_var=False)
        # 单侧 p 值：仅当 t < 0（爆品价格更低）时 p_one = p_two/2
        p_one = p_two / 2 if t_stat < 0 else 1 - p_two / 2
        test_name = "Welch T检验（单侧）"
        print(f"\n{test_name}:")
        print(f"  t统计量 = {t_stat:.4f}")
        print(f"  双侧p值 = {p_two:.4e}")
        print(f"  单侧p值 (H1: 爆品价格 < 普通品) = {p_one:.4e}")
    else:
        # 非正态 -> Mann-Whitney U 检验
        u_stat, p_two = stats.mannwhitneyu(burst_price, normal_price, alternative='two-sided')
        _, p_one = stats.mannwhitneyu(burst_price, normal_price, alternative='less')
        test_name = "Mann-Whitney U检验"
        print(f"\n{test_name}:")
        print(f"  U统计量 = {u_stat:.4f}")
        print(f"  双侧p值 = {p_two:.4e}")
        print(f"  单侧p值 (H1: 爆品价格 < 普通品) = {p_one:.4e}")

    # 均值差与置信区间（使用 Welch df 近似，不依赖正态假设）
    diff = burst_price.mean() - normal_price.mean()
    se = np.sqrt(burst_price.var() / len(burst_price) + normal_price.var() / len(normal_price))
    # Welch-Satterthwaite 自由度
    v_b = burst_price.var() / len(burst_price)
    v_n = normal_price.var() / len(normal_price)
    df_welch = (v_b + v_n)**2 / (v_b**2 / (len(burst_price) - 1) + v_n**2 / (len(normal_price) - 1))
    t_crit = stats.t.ppf(0.975, df_welch)
    ci_lower = diff - t_crit * se
    ci_upper = diff + t_crit * se
    print(f"\n均值差: {diff:.2f} 元")
    print(f"95%置信区间 (Welch): [{ci_lower:.2f}, {ci_upper:.2f}]")

    # 效应量 (Cohen's d)
    pooled_std = np.sqrt(((len(burst_price) - 1) * burst_price.var() +
                          (len(normal_price) - 1) * normal_price.var()) /
                         (len(burst_price) + len(normal_price) - 2))
    cohens_d = diff / pooled_std
    print(f"Cohen's d = {cohens_d:.4f} ({'小' if abs(cohens_d) < 0.2 else '中' if abs(cohens_d) < 0.5 else '大'}效应)")

    # 结论
    alpha = 0.05
    if p_one < alpha:
        if diff < 0:
            print(f"\n[OK] 结论: 在 α={alpha} 水平下，爆品价格显著低于普通品（p={p_one:.4e}）。")
        else:
            print(f"\n[X] 结论: 虽然差异显著（p={p_one:.4e}），但爆品价格实际高于普通品，假设1不成立。")
    else:
        print(f"\n[X] 结论: 在 α={alpha} 水平下，爆品与普通品价格无显著差异（p={p_one:.4e}）。")


def hypothesis_test_2(df):
    """
    假设2：爆品的讲解时长显著高于普通品
    """
    print(f"\n{'='*60}")
    print("【假设2：爆品讲解时长是否显著高于普通品？】")
    print(f"{'='*60}")

    burst_dur = df[df['是否爆品'] == 1]['popduration']
    normal_dur = df[df['是否爆品'] == 0]['popduration']

    print(f"\n描述性统计:")
    print(f"  爆品 (n={len(burst_dur)}): 均值={burst_dur.mean():.1f}秒, "
          f"中位数={burst_dur.median():.1f}秒")
    print(f"  普通品 (n={len(normal_dur)}): 均值={normal_dur.mean():.1f}秒, "
          f"中位数={normal_dur.median():.1f}秒")

    # 非正态数据用 Mann-Whitney U
    u_stat, p_two = stats.mannwhitneyu(burst_dur, normal_dur, alternative='two-sided')
    _, p_greater = stats.mannwhitneyu(burst_dur, normal_dur, alternative='greater')

    print(f"\nMann-Whitney U检验:")
    print(f"  U统计量 = {u_stat:.4f}")
    print(f"  双侧p值 = {p_two:.4e}")
    print(f"  单侧p值 (H1: 爆品讲解时长 > 普通品) = {p_greater:.4e}")

    # 效应量
    diff = burst_dur.mean() - normal_dur.mean()
    pooled_std = np.sqrt(((len(burst_dur) - 1) * burst_dur.var() +
                          (len(normal_dur) - 1) * normal_dur.var()) /
                         (len(burst_dur) + len(normal_dur) - 2))
    cohens_d = diff / pooled_std
    print(f"Cohen's d = {cohens_d:.4f}")

    alpha = 0.05
    if p_greater < alpha and diff > 0:
        print(f"\n[OK] 结论: 爆品的讲解时长显著高于普通品（p={p_greater:.4e}，差异={diff:.1f}秒）。")
    else:
        print(f"\n[X] 结论: 讲解时长差异不显著（p={p_greater:.4e}）。")


def hypothesis_test_3(df):
    """
    假设3：不同直播间规模的爆品率存在显著差异（卡方检验）
    """
    print(f"\n{'='*60}")
    print("【假设3：不同规模直播间的爆品率是否有显著差异？】")
    print(f"{'='*60}")

    if '直播间规模' not in df.columns:
        print("  [!] 缺少直播间规模字段，跳过此检验")
        return

    # 构建列联表
    contingency = pd.crosstab(df['直播间规模'], df['是否爆品'])
    contingency.columns = ['普通品', '爆品']
    print(f"\n列联表:")
    print(contingency.to_string())

    # 卡方检验
    chi2, p, dof, expected = stats.chi2_contingency(contingency)
    print(f"\n卡方检验:")
    print(f"  卡方统计量 = {chi2:.4f}")
    print(f"  自由度 = {dof}")
    print(f"  p值 = {p:.4e}")

    # Cramer's V 效应量
    n = contingency.sum().sum()
    min_dim = min(contingency.shape) - 1
    cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0
    print(f"  Cramer's V = {cramers_v:.4f}")

    # 各组爆品率
    contingency['爆品率'] = (contingency['爆品'] / (contingency['普通品'] + contingency['爆品']) * 100).round(2)
    print(f"\n各直播间规模爆品率:")
    print(contingency['爆品率'].to_string())

    alpha = 0.05
    if p < alpha:
        best = contingency['爆品率'].idxmax()
        print(f"\n[OK] 结论: 不同直播间规模的爆品率存在显著差异（p={p:.4e}）。")
        print(f"  爆品率最高的规模: {best}（{contingency['爆品率'][best]}%）")
    else:
        print(f"\n[X] 结论: 不同直播间规模的爆品率无显著差异（p={p:.4e}）。")


def hypothesis_test_4(df):
    """
    假设4（附加）：主播身体动作(avg_dis)与销量存在显著相关
    """
    print(f"\n{'='*60}")
    print("【假设4：主播身体动作与销量是否显著相关？】")
    print(f"{'='*60}")

    x = df['avg_dis']
    y = df['sales']

    # Pearson 相关
    r, p = stats.pearsonr(x, y)
    print(f"\n皮尔逊相关系数: r = {r:.4f}, p = {p:.4e}")

    # Spearman 秩相关（非参数）
    rho, p_rho = stats.spearmanr(x, y)
    print(f"Spearman 秩相关系数: rho = {rho:.4f}, p = {p_rho:.4e}")

    # 二次项检验（倒U型）
    x_centered = x - x.mean()
    poly_coeffs = np.polyfit(x_centered, y, 2)
    print(f"\n二次拟合（中心化）: y = {poly_coeffs[0]:.4f}x² + {poly_coeffs[1]:.4f}x + {poly_coeffs[2]:.4f}")

    if poly_coeffs[0] < 0:
        # 计算最优点
        optimal_x = -poly_coeffs[1] / (2 * poly_coeffs[0]) + x.mean()
        print(f"  倒U型曲线顶点: avg_dis = {optimal_x:.2f}")
        print(f"  → 结论: 主播身体动作与销量呈倒U型关系，最佳动作幅度约为 {optimal_x:.2f}")
    else:
        print(f"  → 未发现显著的倒U型关系")


def plot_test_results(df):
    """统计检验结果可视化"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 图1：价格分布对比（带核密度估计）
    burst = df[df['是否爆品'] == 1]
    normal = df[df['是否爆品'] == 0]
    price_cap = df['price'].quantile(0.95)
    df_plot = df[df['price'] <= price_cap]

    sns.histplot(data=df_plot, x='price', hue='爆品标签', ax=axes[0],
                 palette={'爆品': '#FF6B6B', '普通品': '#4ECDC4'},
                 kde=True, stat='density', common_norm=False, alpha=0.5)
    axes[0].set_title('价格分布（带KDE）', fontsize=12)
    axes[0].set_xlabel('价格（元）')

    # 图2：讲解时长分布对比（使用完整数据，不受价格过滤影响）
    sns.histplot(data=df, x='popduration', hue='爆品标签', ax=axes[1],
                 palette={'爆品': '#FF6B6B', '普通品': '#4ECDC4'},
                 kde=True, stat='density', common_norm=False, alpha=0.5)
    axes[1].set_title('讲解时长分布（带KDE）', fontsize=12)
    axes[1].set_xlabel('讲解时长（秒）')

    # 图3：卡方检验结果（标准化残差）
    if '直播间规模' in df.columns:
        contingency = pd.crosstab(df['直播间规模'], df['是否爆品'])
        chi2, p, dof, expected = stats.chi2_contingency(contingency)
        # 计算标准化残差
        residuals = (contingency.values - expected) / np.sqrt(expected)
        residual_df = pd.DataFrame(residuals, index=contingency.index,
                                    columns=['普通品', '爆品'])
        sns.heatmap(residual_df, annot=True, fmt='.2f', cmap='RdBu_r',
                    center=0, ax=axes[2], cbar_kws={'label': '标准化残差'})
        axes[2].set_title(f'卡方检验标准化残差\n(χ²={chi2:.1f}, p={p:.2e})', fontsize=12)

    plt.suptitle('统计检验结果可视化', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, '13_statistical_tests.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("[OK] 13_statistical_tests.png")


def main():
    """主函数"""
    df = load_data()

    # 执行所有假设检验
    hypothesis_test_1(df)  # 价格
    hypothesis_test_2(df)  # 讲解时长
    hypothesis_test_3(df)  # 直播间规模
    hypothesis_test_4(df)  # 主播动作（论文核心假设）

    # 可视化
    plot_test_results(df)

    print(f"\n{'='*60}")
    print("【统计检验总结】")
    print(f"{'='*60}")
    print("以上检验结果将汇总到分析报告中。")


if __name__ == '__main__':
    main()
