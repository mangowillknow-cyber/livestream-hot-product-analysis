# 代码审查错误整理清单

本文档记录项目开发过程中通过代码审查发现并修复的所有问题。

---

## 问题总览

| 编号 | 严重度 | 类别 | 问题 | 状态 |
|------|--------|------|------|------|
| 1 | P0 | ML流水线 | StandardScaler 在全集上 fit 导致数据泄漏 | 已修复 |
| 2 | P1 | 统计检验 | 单侧 p 值方向判断缺失 | 已修复 |
| 3 | P1 | 统计检验 | 非正态数据使用 z-based 置信区间 | 已修复 |
| 4 | P1 | ML流水线 | 跨模型特征重要性不可比 | 已修复 |
| 5 | P2 | 数据流 | CSV round-trip 丢失 Categorical 排序 | 已修复 |
| 6 | P2 | 计算逻辑 | 直播间对比阈值不一致（median split vs 5K bins） | 已修复 |
| 7 | P2 | 报告 | 多处硬编码数字，未随数据动态更新 | 已修复 |
| 8 | P3 | 可视化 | 讲解时长 KDE 图使用了价格过滤后的数据 | 已修复 |
| 9 | P3 | 数据清洗 | "evening" 时段未翻译为中文 | 已修复 |

---

## 详细说明

### 1. StandardScaler 数据泄漏（P0）

- **文件**: `08_prediction_model.py`
- **问题**: `StandardScaler().fit_transform(X)` 在全量数据（含测试集）上执行，导致模型评估指标虚高。scaler 的均值和方差受测试集"污染"，属于典型的数据泄漏。
- **影响**: 模型 AUC 等指标不可信，上线后性能会低于实验结果。
- **修复**: 将 scaler 移到 `train_and_evaluate()` 内部，train/test 分割后仅在 X_train 上 `fit_transform`，对 X_test 仅 `transform`。

```python
# 修复前（错误）
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feature_cols)  # 全集 fit
X_train, X_test, ... = train_test_split(X_scaled, y, ...)

# 修复后（正确）
X_train, X_test, ... = train_test_split(X, y, ...)
scaler = StandardScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), ...)  # 仅训练集 fit
X_test = pd.DataFrame(scaler.transform(X_test), ...)         # 测试集只 transform
```

---

### 2. 单侧 p 值方向判断缺失（P1）

- **文件**: `07_statistical_test.py`
- **问题**: `p_one = p_two / 2` 无条件执行。当检验统计量方向与假设相反时（如 t > 0 但 H1 是"爆品价格更低"），单侧 p 值应为 `1 - p_two / 2`。
- **影响**: 可能输出错误的显著性结论。
- **修复**: 加入方向判断条件。

```python
# 修复前
p_one = p_two / 2

# 修复后
p_one = p_two / 2 if t_stat < 0 else 1 - p_two / 2
```

---

### 3. 非正态数据使用 z-based 置信区间（P1）

- **文件**: `07_statistical_test.py`
- **问题**: 95% 置信区间用 `1.96 * se` 计算，这假设数据服从正态分布。但代码已判定数据非正态并选用 Mann-Whitney U 检验，此时 z-based CI 不成立。
- **影响**: 置信区间宽度不准确。
- **修复**: 改用 Welch-Satterthwaite 自由度的 t-based CI。

```python
# 修复前
ci_lower = diff - 1.96 * se

# 修复后
df_welch = (v_b + v_n)**2 / (v_b**2 / (n_b - 1) + v_n**2 / (n_n - 1))
t_crit = stats.t.ppf(0.975, df_welch)
ci_lower = diff - t_crit * se
```

---

### 4. 跨模型特征重要性不可比（P1）

- **文件**: `08_prediction_model.py`
- **问题**: 逻辑回归系数（绝对值）、随机森林 Gini 重要性、XGBoost gain 重要性被各自归一化到 [0,1] 后直接取平均。这三种度量的量纲和含义完全不同，简单平均没有统计意义。归一化使每个模型的最重要特征=1.0，会人为放大在单一模型中排首位的特征。
- **影响**: Top5 关键因素排名可能误导。
- **修复**: 改为 rank-based 方法——各模型内按重要性排名后取平均排名。

```python
# 修复前（错误）
imp_norm = imp / imp.max()
importance_df['平均重要性'] = importance_df[model_names].mean(axis=1)

# 修复后（正确）
importance_df[f'{name}_rank'] = pd.Series(imp_norm).rank(ascending=False).astype(int).values
importance_df['平均排名'] = importance_df[rank_cols].mean(axis=1)
```

---

### 5. CSV round-trip 丢失 Categorical 排序（P2）

- **文件**: `03_eda_analysis.py` ~ `09_report_generator.py`
- **问题**: `02_data_cleaner.py` 用 `pd.cut()` 创建有序 Categorical 列（如 `0-50 < 50-100 < 100-200`），写入 CSV 后再读取时变成普通字符串，排序变为字典序（`100-200` 排在 `50-100` 前面）。
- **影响**: 所有下游图表和报告中的价格区间、讲解时长区间、直播间规模的排序错误。
- **修复**: 在所有脚本的 `load_data()` 中加入 `restore_categorical()` 函数，读取后恢复正确排序。

```python
def restore_categorical(df):
    if '价格区间' in df.columns:
        order = ['0-50', '50-100', '100-200', '200-500', '500-1000', '1000+']
        df['价格区间'] = pd.Categorical(df['价格区间'], categories=order, ordered=True)
    # ... 讲解时长区间、直播间规模同理
    return df
```

---

### 6. 直播间对比阈值不一致（P2）

- **文件**: `05_dimension_comparison.py` vs `README.md` / `09_report_generator.py`
- **问题**: `compare_anchor()` 用中位数观众数做二分（大 vs 小），但报告和 README 按 `直播间规模` 的 4 档 bins 声称"头部(5K+) 爆品率是小型的 200 倍"。两处使用的比较基准不同，结论不可互相验证。
- **影响**: README 中的 "200倍" 结论无法从代码复现。
- **修复**: `compare_anchor()` 改为使用 `直播间规模` 字段进行分组对比。

---

### 7. 报告硬编码数字（P2）

- **文件**: `09_report_generator.py`
- **问题**: 多处使用硬编码数字而非动态变量：
  - "重要性0.92" — 实际值取决于模型运行结果
  - "293秒 vs 363秒" — 应使用 `dur_burst_mean` / `dur_normal_mean`
  - "2-5分钟" — 应使用 `best_dur_band`
  - 使用 `dir()` 检查变量是否存在 — 不可靠
- **影响**: 数据或模型参数变化后报告数字失真。
- **修复**: 所有数字改为 f-string 动态变量，`dir()` 改为哨兵值检查。

---

### 8. KDE 图使用错误数据子集（P3）

- **文件**: `07_statistical_test.py`
- **问题**: `plot_test_results()` 中讲解时长的 KDE 分布图使用了 `df_plot`（按 P95 价格过滤后的子集），排除了高价商品，不代表全量数据。
- **影响**: 讲解时长分布图不完整。
- **修复**: 讲解时长图改用完整 `df`。

---

### 9. 时段未翻译（P3）

- **文件**: `02_data_cleaner.py`
- **问题**: 时段映射 `time_map` 只包含 morning/noon/afternoon，遗漏了数据中的 evening，导致报告中出现英文 "evening时段"。
- **影响**: 报告中英混杂。
- **修复**: 补充 `'evening': '晚上'` 映射。

---

## 修复统计

| 严重度 | 数量 | 说明 |
|--------|------|------|
| P0 | 1 | 数据泄漏，影响模型可靠性 |
| P1 | 3 | 统计方法错误，影响结论正确性 |
| P2 | 3 | 数据流/逻辑不一致，影响可复现性 |
| P3 | 2 | 小问题，影响展示质量 |
| **合计** | **9** | |
