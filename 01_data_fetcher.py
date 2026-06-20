# -*- coding: utf-8 -*-
"""
01_data_fetcher.py - 数据获取模块
=================================
功能：从 Mendeley Data 下载抖音直播带货商品数据集
数据集：More than enough is too much - Douyin Livestream Commerce Data
来源：https://data.mendeley.com/datasets/7jvnfjg6y7/1
样本量：9,641 条抖音直播商品观测记录（2022年3月）
"""

import os
import sys
import time
import zipfile
import warnings
import requests
import pandas as pd

warnings.filterwarnings('ignore')

# ========== Windows 控制台 UTF-8 编码 ==========
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

# ========== 路径配置 ==========
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(PROJECT_DIR, 'data', 'raw')
os.makedirs(RAW_DIR, exist_ok=True)

# ========== Mendeley 数据集配置 ==========
MENDELEY_DATASET_ID = '7jvnfjg6y7'
MENDELEY_VERSION = '1'

# Mendeley API v2 下载地址（多种格式尝试）
MENDELEY_URLS = [
    # 直接下载链接（Mendeley 常用格式）
    f'https://data.mendeley.com/public-files/datasets/{MENDELEY_DATASET_ID}/files/direct',
    # API 端点
    f'https://data.mendeley.com/api/datasets-v2/datasets/{MENDELEY_DATASET_ID}/files',
    # 旧版 API
    f'https://data.mendeley.com/datasets/{MENDELEY_DATASET_ID}/{MENDELEY_VERSION}/files',
]

# 请求头（模拟浏览器访问）
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                  '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'application/octet-stream, */*',
}


def download_from_mendeley(max_retries=3):
    """
    从 Mendeley Data 下载数据集

    参数:
        max_retries: 最大重试次数

    返回:
        str: 下载文件的本地路径，失败返回 None
    """
    print("=" * 60)
    print("【数据获取】从 Mendeley Data 下载抖音直播带货数据集")
    print("=" * 60)
    print(f"\n数据集页面: https://data.mendeley.com/datasets/{MENDELEY_DATASET_ID}/{MENDELEY_VERSION}")
    print(f"本地缓存目录: {RAW_DIR}\n")

    # 检查是否已下载过（优先找数据文件，其次找 zip）
    existing_data = []
    existing_zip = []
    for root, dirs, files in os.walk(RAW_DIR):
        for f in files:
            full = os.path.join(root, f)
            if f.endswith(('.csv', '.xlsx', '.xls')):
                existing_data.append(full)
            elif f.endswith('.zip'):
                existing_zip.append(full)

    if existing_data or existing_zip:
        print(f"[OK] 发现已缓存的数据文件")
        # 优先返回已有的数据文件（优先 main_data）
        main_candidates = [f for f in existing_data if 'main_data' in f.lower()]
        if main_candidates:
            return main_candidates[0]
        if existing_data:
            return existing_data[0]
        # 只有 zip，尝试解压
        csv_path = extract_zip(existing_zip[0])
        return csv_path if csv_path else existing_zip[0]

    # 尝试多种下载方式
    for attempt, url in enumerate(MENDELEY_URLS, 1):
        print(f"尝试 {attempt}/{len(MENDELEY_URLS)}: {url}")
        for retry in range(max_retries):
            try:
                response = requests.get(
                    url,
                    headers=HEADERS,
                    timeout=60,
                    allow_redirects=True,
                    stream=True
                )
                response.raise_for_status()

                # 根据 Content-Type 判断文件类型
                content_type = response.headers.get('Content-Type', '')
                content_disp = response.headers.get('Content-Disposition', '')

                # 从 Content-Disposition 提取文件名
                if 'filename' in content_disp:
                    filename = content_disp.split('filename=')[-1].strip('"\'')
                elif 'zip' in content_type:
                    filename = 'dataset.zip'
                elif 'csv' in content_type or 'text' in content_type:
                    filename = 'dataset.csv'
                elif 'excel' in content_type or 'spreadsheet' in content_type:
                    filename = 'dataset.xlsx'
                else:
                    filename = 'dataset.zip'  # 默认假设为 zip

                filepath = os.path.join(RAW_DIR, filename)

                # 流式下载，显示进度
                total_size = int(response.headers.get('Content-Length', 0))
                downloaded = 0
                with open(filepath, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                pct = downloaded / total_size * 100
                                print(f"\r  下载进度: {pct:.1f}% ({downloaded}/{total_size} bytes)",
                                      end='', flush=True)
                print(f"\n[OK] 下载完成: {filepath}")

                # 如果是 zip 文件，解压到 raw 目录
                if filepath.endswith('.zip'):
                    csv_path = extract_zip(filepath)
                    return csv_path
                return filepath

            except requests.exceptions.RequestException as e:
                print(f"  [ x ] 重试 {retry + 1}/{max_retries}: {e}")
                time.sleep(2 * (retry + 1))  # 退避重试
                continue
        print(f"  该地址下载失败\n")

    return None


def extract_zip(zip_path):
    """
    解压 zip 文件到 raw 目录

    参数:
        zip_path: zip 文件路径

    返回:
        str: 解压后主要数据文件的路径
    """
    print(f"\n正在解压: {zip_path}")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(RAW_DIR)
        file_list = z.namelist()
        print(f"  解压文件: {file_list}")

    # 找到主要数据文件（优先 main_data，其次 CSV，最后其他 Excel）
    data_files = [f for f in file_list if f.endswith(('.csv', '.xlsx', '.xls'))]
    if data_files:
        # 优先选择 main_data 文件
        main_candidates = [f for f in data_files if 'main_data' in f.lower()]
        if main_candidates:
            main_file = os.path.join(RAW_DIR, main_candidates[0])
        else:
            main_file = os.path.join(RAW_DIR, data_files[0])
        print(f"[OK] 主数据文件: {main_file}")
        return main_file

    # 如果没有直接的数据文件，递归搜索子目录
    for root, dirs, files in os.walk(RAW_DIR):
        for f in files:
            if f.endswith(('.csv', '.xlsx', '.xls')):
                return os.path.join(root, f)

    print("[!] 解压后未找到数据文件，请手动检查")
    return None


def load_and_preview_data(filepath=None):
    """
    加载数据并输出预览信息

    参数:
        filepath: 数据文件路径，None 则自动查找

    返回:
        pd.DataFrame: 加载的数据框
    """
    # 自动查找数据文件（搜索子目录中的 xlsx/csv，优先 main_data）
    if filepath is None:
        # 先在所有目录搜索 main_data 文件
        for root, dirs, files in os.walk(RAW_DIR):
            for f in files:
                if f.endswith(('.xlsx', '.xls', '.csv')) and 'main_data' in f.lower():
                    filepath = os.path.join(root, f)
                    break
            if filepath:
                break
    if filepath is None:
        # 搜索任何数据文件（排除 zip）
        for root, dirs, files in os.walk(RAW_DIR):
            for f in sorted(files):
                if f.endswith(('.csv', '.xlsx', '.xls')):
                    filepath = os.path.join(root, f)
                    break
            if filepath:
                break

    if filepath is None:
        print("[!] 未找到任何数据文件")
        return None

    print(f"\n{'=' * 60}")
    print(f"【数据预览】加载文件: {filepath}")
    print(f"{'=' * 60}")

    # 根据文件类型选择读取方式
    if filepath.endswith('.csv'):
        # 尝试多种编码
        for encoding in ['utf-8', 'gbk', 'gb2312', 'latin-1']:
            try:
                df = pd.read_csv(filepath, encoding=encoding)
                print(f"[OK] CSV 编码: {encoding}")
                break
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
        else:
            df = pd.read_csv(filepath, encoding='utf-8', errors='ignore')
    else:
        df = pd.read_excel(filepath)

    # 输出数据概况
    print(f"\n数据形状: {df.shape[0]} 行  x  {df.shape[1]} 列")
    print(f"\n字段列表:")
    for i, col in enumerate(df.columns, 1):
        dtype = df[col].dtype
        nunique = df[col].nunique()
        null_count = df[col].isnull().sum()
        print(f"  {i:2d}. {col:<30s} 类型: {str(dtype):<10s} "
              f"唯一值: {nunique:<8d} 缺失: {null_count}")

    print(f"\n前 5 行数据:")
    print(df.head().to_string())

    print(f"\n描述性统计:")
    print(df.describe().to_string())

    return df


def main():
    """主函数：下载并预览数据"""
    # 第一步：下载数据
    filepath = download_from_mendeley()

    if filepath is None:
        print("\n" + "=" * 60)
        print("[!] 自动下载失败")
        print("请手动下载数据集:")
        print("  1. 访问 https://data.mendeley.com/datasets/7jvnfjg6y7/1")
        print("  2. 点击 'Download' 按钮下载 zip 文件")
        print("  3. 将 zip 文件放到 data/raw/ 目录下")
        print("  4. 重新运行本脚本")
        print("=" * 60)
        return None

    # 第二步：加载并预览
    df = load_and_preview_data(filepath)

    if df is not None:
        # 保存原始数据的副本到 processed 目录（供后续脚本使用）
        processed_dir = os.path.join(PROJECT_DIR, 'data', 'processed')
        os.makedirs(processed_dir, exist_ok=True)
        output_path = os.path.join(processed_dir, 'raw_data.csv')
        df.to_csv(output_path, index=False, encoding='utf-8')
        print(f"\n[OK] 原始数据已保存至: {output_path}")

    return df


if __name__ == '__main__':
    df = main()
