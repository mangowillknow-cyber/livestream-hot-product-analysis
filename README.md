# 直播带货爆品特征挖掘

## 项目背景
直播带货已成为电商核心渠道，但什么样的商品能在直播间成为"爆品"？是价格低？是讲解时间长？还是主播动作更丰富？本项目基于真实抖音直播电商数据，通过对比爆品与普通品的多维特征，挖掘爆品背后的可量化规律，为选品和运营提供数据支撑。

## 数据来源
- **数据集**：Mendeley Data - "More than enough is too much: Curvilinear relationship between anchor body movements and sales in live streaming e-commerce"
- **论文链接**：https://data.mendeley.com/datasets/7jvnfjg6y7/1
- **数据量**：9,641 条抖音直播商品观测记录（2022年3月）
- **主播数**：23 位主播

## 核心发现
1. 爆品平均价格145元，比普通品低62.7%，0-50元区间爆品率最高（12.8%）
2. 2-5分钟讲解区间爆品率最高（8.6%），过长讲解反而降低转化率
3. 头部直播间（5K+观众）爆品率高达12.5%，是小型直播间的200倍
4. 观众规模是影响爆品的最重要因素（特征重要性0.92），预测模型AUC达0.92

## 技术栈
- Python 3.8+
- 数据处理：pandas, numpy
- 可视化：matplotlib, seaborn, plotly
- 统计检验：scipy.stats
- 机器学习：scikit-learn, xgboost, shap

## 项目结构

```
 直播带货爆品特征挖掘/
├──  README.md               # 项目说明
├──  requirements.txt        # 依赖包清单
├──  01_data_fetcher.py      # 数据获取
├──  02_data_cleaner.py      # 数据清洗
├──  03_eda_analysis.py      # 探索性分析
├──  04_burst_definition.py  # 爆品定义
├──  05_dimension_comparison.py # 多维度对比分析
├──  06_radar_profile.py     # 多维画像 + 雷达图
├──  07_statistical_test.py  # 统计检验
├──  08_prediction_model.py  # 预测建模
├──  09_report_generator.py  # 报告生成
├──  analysis_report.md      # 完整分析报告
├──  data/
│   ├──  raw/                # 原始数据（.gitignore）
│   └──  processed/          # 清洗后数据
└──  figures/                # 所有图表输出
```

## 如何运行

### 1. 安装依赖
```bash
pip install -r requirements.txt
```

### 2. 准备数据
从 [Mendeley Data](https://data.mendeley.com/datasets/7jvnfjg6y7/1) 下载数据集 zip 文件，放到 `data/raw/` 目录。

### 3. 运行分析流程
```bash
# 数据获取与预览
python 01_data_fetcher.py

# 数据清洗
python 02_data_cleaner.py

# 探索性分析（EDA）
python 03_eda_analysis.py

# 爆品定义
python 04_burst_definition.py

# 多维度对比分析
python 05_dimension_comparison.py

# 爆品画像生成
python 06_radar_profile.py

# 统计检验
python 07_statistical_test.py

# 预测建模
python 08_prediction_model.py

# 生成报告
python 09_report_generator.py
```

## 分析维度

| 维度 | 对比内容 | 核心问题 |
|------|----------|----------|
| 价格 | 价格区间 vs 爆品率 | 直播间越低价越好吗？ |
| 讲解时长 | 讲解时长区间 vs 爆品率 | 讲多久最有效？ |
| 主播特征 | 不同主播的爆品率 | 哪类主播最能带货？ |
| 身体动作 | 主播动作位移 vs 销量 | 动作越多越好吗？ |
| 时段 | 不同时段的爆品率 | 什么时间直播最好？ |

## 图表清单

| 图表 | 文件名 | 说明 |
|------|--------|------|
| 数据分布 | 01_distributions.png | 各字段分布概览 |
| 相关系数 | 02_correlation.png | 变量相关系数热力图 |
| 分类字段 | 03_categorical.png | 时段/价格/规模分布 |
| 销量分布 | 04_sales_distribution.png | 销量深度分析 |
| 爆品定义 | 05_burst_definition.png | 两种定义对比 |
| 价格对比 | 06_price_comparison.png | 爆品vs普通品价格 |
| 讲解对比 | 07_duration_comparison.png | 讲解时长对比 |
| 主播对比 | 08_anchor_comparison.png | 主播特征对比 |
| 动作对比 | 09_movement_comparison.png | 身体动作对比 |
| 时段对比 | 10_time_comparison.png | 时段/时间对比 |
| 雷达图 | 11_radar_chart.png | 多维画像雷达图 |
| 画像卡 | 12_profile_card.png | 爆品特征画像卡 |
| 统计检验 | 13_statistical_tests.png | 检验结果可视化 |
| 模型对比 | 14_model_comparison.png | 三个模型ROC对比 |
| 特征重要性 | 15_feature_importance.png | Top5关键因素 |
