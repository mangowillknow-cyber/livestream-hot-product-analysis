# -*- coding: utf-8 -*-
"""
02_data_cleaner.py - 数据清洗模块
=================================
功能：加载原始数据，处理缺失值、异常值，输出清洗后数据
输入：data/raw/ 下的 main_data.xlsx
输出：data/processed/cleaned_data.csv
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# ========== 路径配置 ==========
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(PROJECT_DIR, 'data', 'raw')
PROCESSED_DIR = os.path.join(PROJECT_DIR, 'data', 'processed')
os.makedirs(PROCESSED_DIR, exist_ok=True)


def load_raw_data():
    """
    加载原始数据

    返回:
        pd.DataFrame: 原始数据框
    """
    # 在 raw 目录搜索 xlsx 文件
    xlsx_files = []
    for root, dirs, files in os.walk(RAW_DIR):
        for f in files:
            if f.endswith('.xlsx') and 'main_data' in f.lower():
                xlsx_files.append(os.path.join(root, f))

    if not xlsx_files:
        # 如果没有找到 main_data，搜索所有 xlsx
        for root, dirs, files in os.walk(RAW_DIR):
            for f in files:
                if f.endswith(('.xlsx', '.csv')):
                    xlsx_files.append(os.path.join(root, f))

    if not xlsx_files:
        raise FileNotFoundError("未找到数据文件，请先运行 01_data_fetcher.py")

    filepath = xlsx_files[0]
    print(f"加载数据: {filepath}")

    if filepath.endswith('.csv'):
        df = pd.read_csv(filepath, encoding='utf-8')
    else:
        df = pd.read_excel(filepath)

    print(f"原始数据: {df.shape[0]} 行  x  {df.shape[1]} 列")
    return df


def generate_data_quality_report(df, stage='清洗前'):
    """
    生成数据质量报告

    参数:
        df: 数据框
        stage: 阶段标签
    """
    print(f"\n{'='*60}")
    print(f"【数据质量报告 - {stage}】")
    print(f"{'='*60}")
    print(f"总行数: {len(df)}")
    print(f"总列数: {len(df.columns)}")
    print(f"\n各字段统计:")

    for col in df.columns:
        dtype = df[col].dtype
        null_count = df[col].isnull().sum()
        null_pct = null_count / len(df) * 100
        nunique = df[col].nunique()

        # 数值型字段额外统计
        if pd.api.types.is_numeric_dtype(df[col]):
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            outlier_count = ((df[col] < q1 - 3 * iqr) | (df[col] > q3 + 3 * iqr)).sum()
            print(f"  {col:<20s} 类型:{str(dtype):<10s} 缺失:{null_count}({null_pct:.1f}%) "
                  f"唯一值:{nunique} 异常值(3*IQR):{outlier_count}")
        else:
            print(f"  {col:<20s} 类型:{str(dtype):<10s} 缺失:{null_count}({null_pct:.1f}%) "
                  f"唯一值:{nunique}")


def clean_data(df):
    """
    数据清洗主流程

    步骤:
        1. 处理缺失值
        2. 处理异常值
        3. 数据类型转换
        4. 字段标准化命名

    参数:
        df: 原始数据框

    返回:
        pd.DataFrame: 清洗后的数据框
    """
    print(f"\n{'='*60}")
    print("【数据清洗流程】")
    print(f"{'='*60}")

    df_clean = df.copy()
    initial_count = len(df_clean)

    # ---- 步骤1: 处理缺失值 ----
    print(f"\n步骤1: 处理缺失值")
    missing_cols = df_clean.columns[df_clean.isnull().any()].tolist()
    if missing_cols:
        for col in missing_cols:
            n_missing = df_clean[col].isnull().sum()
            # 数值型用中位数填充，分类用众数
            if pd.api.types.is_numeric_dtype(df_clean[col]):
                fill_val = df_clean[col].median()
                df_clean[col].fillna(fill_val, inplace=True)
                print(f"  {col}: {n_missing} 个缺失值，用中位数 {fill_val:.2f} 填充")
            else:
                fill_val = df_clean[col].mode()[0]
                df_clean[col].fillna(fill_val, inplace=True)
                print(f"  {col}: {n_missing} 个缺失值，用众数 '{fill_val}' 填充")
    else:
        print("  无缺失值")

    # ---- 步骤2: 处理异常值 ----
    print(f"\n步骤2: 处理异常值")

    # 销量不能为负
    if 'sales' in df_clean.columns:
        neg_sales = (df_clean['sales'] < 0).sum()
        if neg_sales > 0:
            df_clean = df_clean[df_clean['sales'] >= 0]
            print(f"  删除 {neg_sales} 条销量为负的记录")
        else:
            print(f"  销量字段: 无负值异常")

    # 价格异常检测（使用 IQR 方法）
    if 'price' in df_clean.columns:
        q1 = df_clean['price'].quantile(0.25)
        q3 = df_clean['price'].quantile(0.75)
        iqr = q3 - q1
        lower = max(0, q1 - 3 * iqr)
        upper = q3 + 3 * iqr
        price_outliers = ((df_clean['price'] < lower) | (df_clean['price'] > upper)).sum()
        # 不删除价格异常值，只标记（因为高价商品可能是真实存在的奢侈品）
        print(f"  价格字段: IQR范围 [{lower:.0f}, {upper:.0f}]，"
              f"超出范围 {price_outliers} 条（保留，不删除）")

    # 观众数不能为负
    if 'userbefore' in df_clean.columns:
        neg_users = (df_clean['userbefore'] < 0).sum()
        if neg_users > 0:
            df_clean = df_clean[df_clean['userbefore'] >= 0]
            print(f"  删除 {neg_users} 条观众数为负的记录")
        else:
            print(f"  观众数字段: 无负值异常")

    # ---- 步骤3: 数据类型转换 ----
    print(f"\n步骤3: 数据类型转换")

    # 时间字段转换
    if 'startlivetime' in df_clean.columns:
        df_clean['startlivetime'] = pd.to_datetime(df_clean['startlivetime'], errors='coerce')
        # 提取日期、星期、小时等辅助字段
        df_clean['date'] = df_clean['startlivetime'].dt.date
        df_clean['weekday'] = df_clean['startlivetime'].dt.day_name()
        df_clean['hour'] = df_clean['startlivetime'].dt.hour
        print(f"  startlivetime -> datetime，提取 date/weekday/hour 字段")

    # 时段字段统一编码
    if 'starthours' in df_clean.columns:
        time_map = {'morning': '上午', 'noon': '中午', 'afternoon': '下午'}
        df_clean['时段'] = df_clean['starthours'].map(time_map).fillna(df_clean['starthours'])
        print(f"  starthours -> 时段（中文标签）")

    # ---- 步骤4: 生成衍生特征 ----
    print(f"\n步骤4: 生成衍生特征")

    # 价格分段
    if 'price' in df_clean.columns:
        bins = [0, 50, 100, 200, 500, 1000, float('inf')]
        labels = ['0-50', '50-100', '100-200', '200-500', '500-1000', '1000+']
        df_clean['价格区间'] = pd.cut(df_clean['price'], bins=bins, labels=labels)
        print(f"  生成价格区间字段（6档）")

    # 讲解时长分段（秒 -> 分钟更直观）
    if 'popduration' in df_clean.columns:
        df_clean['讲解时长_分钟'] = df_clean['popduration'] / 60
        bins_dur = [0, 120, 300, 600, float('inf')]
        labels_dur = ['<2分钟', '2-5分钟', '5-10分钟', '10分钟+']
        df_clean['讲解时长区间'] = pd.cut(df_clean['popduration'], bins=bins_dur, labels=labels_dur)
        print(f"  生成讲解时长_分钟和讲解时长区间字段")

    # 观众规模分段
    if 'userbefore' in df_clean.columns:
        bins_user = [0, 500, 2000, 5000, float('inf')]
        labels_user = ['小直播间(<500)', '中型(500-2K)', '大型(2K-5K)', '头部(5K+)']
        df_clean['直播间规模'] = pd.cut(df_clean['userbefore'], bins=bins_user, labels=labels_user)
        print(f"  生成直播间规模字段")

    final_count = len(df_clean)
    print(f"\n清洗完成: {initial_count} -> {final_count} 行（删除 {initial_count - final_count} 行）")

    return df_clean


def main():
    """主函数"""
    # 加载原始数据
    df = load_raw_data()

    # 清洗前质量报告
    generate_data_quality_report(df, '清洗前')

    # 执行清洗
    df_clean = clean_data(df)

    # 清洗后质量报告
    generate_data_quality_report(df_clean, '清洗后')

    # 保存清洗后数据
    output_path = os.path.join(PROCESSED_DIR, 'cleaned_data.csv')
    df_clean.to_csv(output_path, index=False, encoding='utf-8')
    print(f"\n[OK] 清洗后数据已保存至: {output_path}")

    return df_clean


if __name__ == '__main__':
    df = main()
