# 实验管理框架 / Experiment Management Framework

本文档是变压器油温预测实验的完整指南，包含88个系统化实验的配置、执行和分析方法。

---

## 目录 / Contents

1. [框架概览](#框架概览)
2. [快速开始](#快速开始)
3. [实验阶段说明](#实验阶段说明)
4. [查看和分析结果](#查看和分析结果)
5. [继续后续阶段](#继续后续阶段)
6. [自定义实验](#自定义实验)
7. [常见问题 FAQ](#常见问题-faq)
8. [results.csv 字段说明](#resultscsv-字段说明)
9. [注意事项](#注意事项)

---

## 框架概览

### 文件结构

```
experiments/
├── README.md                   # 本文件
├── experiment_configs.py       # 所有实验配置（88个模型）
├── results.csv                 # 实验结果汇总（自动生成）
├── failed_experiments.log      # 失败实验日志（自动生成）
├── stage_summaries/            # 各阶段总结报告（自动生成）
│   ├── stage1_summary.md
│   ├── stage2_summary.md
│   └── ...
└── visualizations/             # 对比图表（自动生成）
    ├── stage1_comparison.png
    └── ...

notebooks/
├── linear_regression.ipynb     # Linear模型 + 批量实验功能
├── rnn.ipynb                   # RNN模型 + 批量实验功能
└── utils.py                    # 共享工具函数
```

### 实验阶段规划

| 阶段 | 目标 | 模型数量 | 主要变量 | 预计时间 |
|------|------|----------|----------|----------|
| 1 | 建立基准 | 6 | 算法类型、预测场景 | 20-30分钟 |
| 2 | 数据预处理 | 10 | 异常值处理、划分方式 | 30-40分钟 |
| 3 | 特征工程 | 12 | 时间特征、特征选择 | 40-50分钟 |
| 4 | 时间窗口 | 12 | seq_length | 40-50分钟 |
| 5 | 算法对比 | 12 | Linear/RNN/informer | 40-60分钟 |
| 6 | 超参数精调 | 27 | 学习率、batch size等 | 60-90分钟 |
| 7 | 最终验证 | 9 | train1验证、消融实验 | 30-40分钟 |

**总计：88个模型，预计5-6小时**（支持断点续传，可分多次运行）

---

## 快速开始

### 步骤1：打开 Jupyter Notebook

```bash
cd notebooks
jupyter notebook
```

### 步骤2：运行 Linear 模型实验

1. **打开 `linear_regression.ipynb`**

2. **滚动到最后的"批量实验执行功能"部分**

3. **依次运行以下 cells 以加载必要的函数**：
   - "批量实验执行功能" - Markdown说明
   - "导入实验配置和工具" - 导入配置
   - "结果保存函数" - 定义保存函数
   - "批量实验执行函数" - 定义执行函数
   - "阶段分析和可视化函数" - 定义分析函数

4. **在新 cell 中运行以下代码**：

```python
# 获取阶段1的Linear模型配置（3个实验）
stage1_configs = get_stage_configs(1)
stage1_configs_linear = [c for c in stage1_configs if c['model_type'] == 'linear']

# 批量运行实验
results_stage1_linear = run_experiments_batch(
    stage1_configs_linear,
    skip_completed=True,  # 支持断点续传
    save_models=True      # 保存模型文件到 models/
)

# 这将运行3个实验：
# - stage1_linear_h (1小时预测, offset=4)
# - stage1_linear_d (1天预测, offset=96)
# - stage1_linear_w (1周预测, offset=672)
```

**预计时间**：每个模型约2-5分钟，总计10-15分钟。

### 步骤3：运行 RNN 模型实验

1. **打开 `rnn.ipynb`**

2. **滚动到最后的"批量实验执行功能"部分**

3. **运行加载函数的 cells**（同步骤2）

4. **在新 cell 中运行以下代码**：

```python
# 获取阶段1的RNN模型配置（3个实验）
stage1_configs = get_stage_configs(1)
stage1_configs_rnn = [c for c in stage1_configs if c['model_type'] == 'rnn']

# 批量运行RNN实验
results_stage1_rnn = run_rnn_experiments_batch(
    stage1_configs_rnn,
    skip_completed=True,
    save_models=True
)

# 这将运行3个实验：
# - stage1_rnn_h (1小时预测)
# - stage1_rnn_d (1天预测)
# - stage1_rnn_w (1周预测)
```

**预计时间**：每个模型约3-6分钟，总计15-20分钟。

---

## 实验阶段说明

### 阶段1：建立基准 (Baseline)

**目标**：确定各预测场景的基准性能

**实验配置**：
- **数据集**：train2.csv
- **算法**：Linear, RNN
- **预测场景**：1小时(offset=4), 1天(offset=96), 1周(offset=672)
- **模型数量**：2算法 × 3场景 = **6个模型**
- **固定参数**：
  - 异常值处理：无
  - 时间特征：无（仅6个负载特征）
  - 划分方式：sequential (80/20)
  - seq_length：16
  - 学习率：0.001
  - batch_size：32

**实验ID格式**：`stage1_{model}_{scenario}`
- 示例：`stage1_linear_h`, `stage1_rnn_d`

---

### 阶段2：数据预处理影响分析

**目标**：评估异常值处理和数据划分方式的影响

**基准配置**：使用阶段1中R²最高的算法

**实验分组**：

#### 2.1 异常值剔除比例 - 4个模型
- 无异常值处理
- 0.5% 阈值 (Z-score=3.0)
- 1.0% 阈值
- 5.0% 阈值

仅在**1小时预测**场景测试（最快迭代）

#### 2.2 数据划分方式 - 3个模型
- sequential：时序分割，无数据泄露
- random：分20组，随机分配80%/20%
- label_random：完全随机，可能存在窗口重叠泄露

#### 2.3 最优组合验证 - 3个模型
将2.1和2.2的最优配置组合，在3个预测场景下验证

**模型数量**：4 + 3 + 3 = **10个模型**

**实验ID格式**：`stage2_{dimension}_{value}_{scenario}`
- 示例：`stage2_outlier_1pct_h`, `stage2_split_random_h`

---

### 阶段3：特征工程影响分析

**目标**：评估时间特征和特征选择的影响

**基准配置**：使用阶段2的最优数据预处理配置

**实验分组**：

#### 3.1 时间特征影响 - 6个模型
- **无时间特征**：仅6个负载特征
- **有时间特征**：负载特征 + (hour, dayofweek, month, day, is_weekend)

在3个预测场景测试：2配置 × 3场景 = 6个模型

#### 3.2 特征选择影响 - 6个模型
- **全部负载特征**：6个 (HUFL, HULL, MUFL, MULL, LUFL, LULL)
- **筛选特征**：基于相关性分析选择Top3-4

在3个预测场景测试：2配置 × 3场景 = 6个模型

**模型数量**：6 + 6 = **12个模型**

---

### 阶段4：时间窗口长度影响

**目标**：确定最优 seq_length

**基准配置**：使用阶段3的最优特征配置

**测试参数**：
- seq_length：8, 16, 32, 64

**模型数量**：4窗口 × 3场景 = **12个模型**

**实验ID格式**：`stage4_seq{length}_{scenario}`
- 示例：`stage4_seq32_h`, `stage4_seq64_d`

---

### 阶段5：算法对比

**目标**：比较不同深度学习算法性能

**基准配置**：使用阶段4的最优时间窗口配置

**测试算法**：
- Linear：多层全连接网络
- RNN：循环神经网络
- LSTM：长短期记忆网络
- GRU：门控循环单元

**模型数量**：4算法 × 3场景 = **12个模型**

**实验ID格式**：`stage5_{algorithm}_{scenario}`
- 示例：`stage5_lstm_h`, `stage5_gru_w`

---

### 阶段6：超参数精调

**目标**：找到最优超参数组合

**基准配置**：使用阶段5的最优算法

**测试场景**：仅在**1小时预测**测试（快速迭代），最后验证其他场景

**实验分组**：

#### 6.1 学习率 - 3个模型
测试：0.0001, 0.001, 0.01

#### 6.2 Batch size - 3个模型
测试：16, 32, 64

#### 6.3 隐藏层大小 - 3个模型
测试：32, 64, 128

#### 6.4 Dropout率 - 3个模型
测试：0.0, 0.2, 0.4

#### 6.5 最优超参数验证 - 3个模型
将6.1-6.4的最优超参数组合应用到3个预测场景

**模型数量**：3 + 3 + 3 + 3 + 3 = **15个模型**
（保守估算；如果做网格搜索可能需要27个）

---

### 阶段7：最终验证与消融实验

**目标**：在train1上验证泛化能力，进行消融实验

**基准配置**：使用阶段6的所有最优配置

**实验分组**：

#### 7.1 train1数据集验证 - 3个模型
将最优配置应用到 **train1.csv**（不同数据分布）
- 1小时预测
- 1天预测
- 1周预测

#### 7.2 消融实验 (Ablation Study) - 6个模型
在train2上，逐个移除关键组件验证其贡献：
- 移除时间特征（如果阶段3证明有效）
- 移除异常值处理（如果阶段2证明有效）

每个消融在3个预测场景测试：2消融 × 3场景 = 6个模型

**模型数量**：3 + 6 = **9个模型**

**实验ID格式**：`stage7_{type}_{scenario}`
- 示例：`stage7_train1_h`, `stage7_ablation_no_time_d`

---

## 查看和分析结果

### 方法1：直接查看 CSV 文件

在任意 notebook 中运行：

```python
import pandas as pd

# 读取所有实验结果
results_df = pd.read_csv('../experiments/results.csv')

# 查看阶段1结果
stage1_df = results_df[results_df['stage'] == 1]
print(stage1_df[['experiment_id', 'model_type', 'prediction_scenario', 'test_r2', 'test_mae']])

# 按R²排序
print("\n按R²排序 / Sorted by R²:")
print(stage1_df.sort_values('test_r2', ascending=False)[
    ['experiment_id', 'test_r2', 'test_mae', 'training_time_sec']
])
```

### 方法2：使用分析函数

在 `linear_regression.ipynb` 中运行：

```python
# 分析阶段1结果（自动生成统计和图表）
summary = analyze_stage_results(1)

# 生成阶段1总结报告
generate_stage_report(1)

# 报告会自动保存到：
# - experiments/stage_summaries/stage1_summary.md
# - experiments/visualizations/stage1_comparison.png
```

**输出内容**：
- 实验数量统计
- R²、MAE的最优/最差/平均/中位数
- 最优模型信息
- 对比条形图（R² 和 MAE）

### 方法3：对比不同算法/配置

```python
# Linear vs RNN 对比
linear_results = stage1_df[stage1_df['model_type'] == 'linear']
rnn_results = stage1_df[stage1_df['model_type'] == 'rnn']

print("Linear 平均R²:", linear_results['test_r2'].mean())
print("RNN 平均R²:", rnn_results['test_r2'].mean())

# 不同预测场景的性能
for scenario in ['hour', 'day', 'week']:
    scenario_df = stage1_df[stage1_df['prediction_scenario'] == scenario]
    best_idx = scenario_df['test_r2'].idxmax()
    print(f"\n{scenario} 预测最优:")
    print(f"  模型: {scenario_df.loc[best_idx, 'experiment_id']}")
    print(f"  R²: {scenario_df.loc[best_idx, 'test_r2']:.6f}")
    print(f"  MAE: {scenario_df.loc[best_idx, 'test_mae']:.6f}")
```

---

## 继续后续阶段

### 运行阶段2：数据预处理影响分析

基于阶段1的结果，选择最优算法（假设是 Linear）：

```python
# 在 linear_regression.ipynb 中运行

# 获取阶段2配置（10个实验）
stage2_configs = get_stage_configs(2)

# 策略1：一次性运行所有实验
results_stage2 = run_experiments_batch(stage2_configs, skip_completed=True)

# 策略2：分批运行，逐步分析（推荐）
# 2.1 先运行异常值相关的4个实验
outlier_configs = [c for c in stage2_configs if 'outlier' in c['experiment_id']]
results_outlier = run_experiments_batch(outlier_configs, skip_completed=True)

# 查看初步结果，确定最优异常值处理方法
analyze_stage_results(2)

# 2.2 然后运行数据划分相关的3个实验
split_configs = [c for c in stage2_configs if 'split' in c['experiment_id']]
results_split = run_experiments_batch(split_configs, skip_completed=True)

# 2.3 最后运行最优组合验证的3个实验
# 注意：需要先根据上面的结果更新 combo_configs 中的参数
# 编辑 experiments/experiment_configs.py 中的 STAGE2_CONFIGS
combo_configs = [c for c in stage2_configs if 'combo' in c['experiment_id']]
results_combo = run_experiments_batch(combo_configs, skip_completed=True)

# 生成阶段2完整报告
analyze_stage_results(2)
generate_stage_report(2)
```

### 运行阶段3-7

依次类似地运行后续阶段：

```python
# 阶段3：特征工程影响分析
stage3_configs = get_stage_configs(3)
results_stage3 = run_experiments_batch(stage3_configs, skip_completed=True)
analyze_stage_results(3)
generate_stage_report(3)

# 阶段4：时间窗口长度影响
stage4_configs = get_stage_configs(4)
results_stage4 = run_experiments_batch(stage4_configs, skip_completed=True)
analyze_stage_results(4)
generate_stage_report(4)

# 阶段5：算法对比
stage5_configs = get_stage_configs(5)
results_stage5 = run_experiments_batch(stage5_configs, skip_completed=True)
analyze_stage_results(5)
generate_stage_report(5)

# 阶段6：超参数精调（可能需要较长时间）
stage6_configs = get_stage_configs(6)
results_stage6 = run_experiments_batch(stage6_configs, skip_completed=True)
analyze_stage_results(6)
generate_stage_report(6)

# 阶段7：最终验证与消融实验
stage7_configs = get_stage_configs(7)
results_stage7 = run_experiments_batch(stage7_configs, skip_completed=True)
analyze_stage_results(7)
generate_stage_report(7)
```

---

## 自定义实验

### 运行单个自定义实验

```python
# 在 linear_regression.ipynb 中

custom_config = {
    'dataset_path': '../dataset/train2.csv',
    'prediction_horizon': 'hour',
    'model_type': 'linear',
    'split_method': 'sequential',
    'time_features': ['hour', 'dayofweek', 'is_weekend'],  # 添加时间特征
    'remove_outliers': True,                               # 启用异常值处理
    'outlier_method': 'zscore',
    'outlier_threshold': 3.0,
    'num_epochs': 100,
    'learning_rate': 0.001,
    'batch_size': 64,                                      # 更大的 batch size
    'hidden_sizes': [128, 64, 32],                         # 更深的网络
    'dropout': 0.3,
    'experiment_id': 'custom_linear_test_1',
    'stage': 0,
    'notes': 'Custom experiment: time features + outlier removal + deeper network',
}

# 训练模型
result = train_single_model(config=custom_config)

# 保存结果到 CSV
save_experiment_result(result)

# 查看结果
print(f"R²: {result['metrics']['r2']:.6f}")
print(f"MAE: {result['metrics']['mae']:.6f}")
print(f"Training time: {result.get('training_time', 0):.2f}s")
```

### 修改预定义配置

```python
# 修改阶段1的某个配置进行快速测试
stage1_configs = get_stage_configs(1)
modified_config = stage1_configs[0].copy()  # 复制第一个配置

# 修改参数
modified_config['num_epochs'] = 50          # 减少训练轮次用于快速测试
modified_config['learning_rate'] = 0.01     # 更大的学习率
modified_config['experiment_id'] = 'stage1_linear_h_modified'
modified_config['notes'] = 'Modified: faster training for testing'

# 运行修改后的实验
result = train_single_model(config=modified_config)
save_experiment_result(result)
```

### 批量运行自定义配置

```python
# 创建一组自定义配置
custom_configs = []

for lr in [0.0001, 0.001, 0.01]:
    for bs in [16, 32, 64]:
        config = {
            'dataset_path': '../dataset/train2.csv',
            'prediction_horizon': 'hour',
            'model_type': 'linear',
            'split_method': 'sequential',
            'time_features': [],
            'num_epochs': 50,
            'learning_rate': lr,
            'batch_size': bs,
            'experiment_id': f'custom_lr{lr}_bs{bs}',
            'stage': 0,
            'notes': f'Custom: LR={lr}, BS={bs}',
        }
        custom_configs.append(config)

# 批量运行（9个实验：3学习率 × 3 batch size）
results = run_experiments_batch(custom_configs, skip_completed=True)
```

---

## 常见问题 / FAQ

### Q1: 实验中断后如何继续？

**A:** 使用 `skip_completed=True` 参数，系统会自动跳过已完成的实验：

```python
# 重新运行，会自动跳过已在 results.csv 中的实验
results = run_experiments_batch(configs, skip_completed=True)
```

所有结果保存在 `experiments/results.csv`，系统通过 `experiment_id` 判断是否已完成。

---

### Q2: 如何查看失败的实验？

**A:** 查看失败日志文件：

```python
import os

log_path = '../experiments/failed_experiments.log'

if os.path.exists(log_path):
    with open(log_path, 'r', encoding='utf-8') as f:
        print(f.read())
else:
    print("暂无失败实验")
```

失败日志格式：`时间戳: experiment_id - 错误信息`

---

### Q3: 如何删除某个实验结果重新运行？

**A:** 两种方法：

**方法1：编辑 CSV（推荐）**
1. 打开 `experiments/results.csv`
2. 找到并删除对应的行
3. 重新运行实验（不会被跳过）

**方法2：使用 pandas**
```python
import pandas as pd

# 读取结果
df = pd.read_csv('../experiments/results.csv')

# 删除特定实验
df = df[df['experiment_id'] != 'stage1_linear_h']

# 保存回文件
df.to_csv('../experiments/results.csv', index=False)
```

---

### Q4: 如何修改预定义的实验配置？

**A:** 编辑 `experiments/experiment_configs.py`：

1. 打开文件找到对应阶段（例如 `STAGE2_CONFIGS`）
2. 修改配置参数
3. 保存文件
4. 重新运行 notebook 中导入配置的 cell

示例：修改阶段2的异常值阈值
```python
# 在 experiment_configs.py 中找到
config['outlier_threshold'] = 3.0  # 改为 2.5
```

---

### Q5: 训练太慢怎么办？

**A:** 优化策略：

1. **减少训练轮次**
   ```python
   config['num_epochs'] = 50  # 从100减到50
   ```

2. **增加 batch size**（加速但可能影响性能）
   ```python
   config['batch_size'] = 64  # 从32增到64
   ```

3. **减少模型复杂度**
   ```python
   # 对于 RNN
   config['hidden_size'] = 32        # 从64减到32
   config['num_layers'] = 1          # 从2减到1

   # 对于 Linear
   config['hidden_sizes'] = [32]     # 从[64, 32]减到[32]
   ```

4. **使用 GPU**（如果可用）
   ```python
   # 在 notebooks/utils.py 中修改
   DEFAULT_CONFIG['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
   ```

---

### Q6: 如何只运行部分实验？

**A:** 使用列表切片或过滤：

```python
# 方法1：只运行前3个实验
configs = get_stage_configs(2)[:3]
results = run_experiments_batch(configs)

# 方法2：过滤特定实验
configs = get_stage_configs(2)
outlier_configs = [c for c in configs if 'outlier' in c['experiment_id']]
results = run_experiments_batch(outlier_configs)

# 方法3：按索引选择
configs = get_stage_configs(3)
selected_configs = [configs[0], configs[3], configs[6]]  # 选择特定索引
results = run_experiments_batch(selected_configs)

# 方法4：按条件过滤
configs = get_stage_configs(5)
lstm_gru_configs = [c for c in configs if c['model_type'] in ['lstm', 'gru']]
results = run_experiments_batch(lstm_gru_configs)
```

---

### Q7: 如何对比两个阶段的结果？

**A:** 使用 pandas 进行对比分析：

```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取结果
df = pd.read_csv('../experiments/results.csv')

# 提取两个阶段
stage1_df = df[df['stage'] == 1]
stage2_df = df[df['stage'] == 2]

# 对比平均性能
print("阶段1平均R²:", stage1_df['test_r2'].mean())
print("阶段2平均R²:", stage2_df['test_r2'].mean())

# 绘制对比图
plt.figure(figsize=(10, 5))
plt.bar(['Stage 1', 'Stage 2'],
        [stage1_df['test_r2'].mean(), stage2_df['test_r2'].mean()])
plt.ylabel('Average Test R²')
plt.title('Performance Comparison')
plt.show()
```

---

### Q8: 实验结果的 R² 很低怎么办？

**A:** 排查清单：

1. **检查数据归一化**
   ```python
   # 确认 StandardScaler 正确应用
   print("X_train 范围:", X_train_scaled.min(), X_train_scaled.max())
   print("y_train 范围:", y_train_scaled.min(), y_train_scaled.max())
   ```

2. **检查数据泄露**
   ```python
   # 使用 sequential 划分避免泄露
   config['split_method'] = 'sequential'
   ```

3. **检查序列创建**
   ```python
   # 确认 offset 正确设置
   print("预测场景:", config['prediction_horizon'])
   print("Offset:", HORIZON_CONFIGS[config['prediction_horizon']]['offset'])
   ```

4. **尝试不同超参数**
   - 增加 `num_epochs`
   - 调整 `learning_rate`
   - 改变模型架构（增加层数/hidden size）

5. **添加特征**
   ```python
   config['time_features'] = ['hour', 'dayofweek', 'month']
   ```

---

## results.csv 字段说明

所有实验结果统一保存在 `experiments/results.csv`，包含以下字段：

### 基本信息

| 字段 | 类型 | 说明 | 示例值 |
|------|------|------|--------|
| experiment_id | str | 实验唯一标识符 | stage1_linear_h |
| stage | int | 实验阶段编号 | 1, 2, 3, ..., 7 |
| dataset | str | 使用的数据集文件名 | train1.csv, train2.csv |
| notes | str | 实验备注说明 | "Baseline - LINEAR - hour" |

### 模型配置

| 字段 | 类型 | 说明 | 示例值 |
|------|------|------|--------|
| model_type | str | 模型类型 | linear, rnn, lstm, gru |
| prediction_scenario | str | 预测场景 | hour, day, week |
| offset | int | 预测时间偏移量（15分钟为单位） | 4, 96, 672 |
| seq_length | int | 输入序列长度 | 8, 16, 32, 64 |

### 数据预处理

| 字段 | 类型 | 说明 | 示例值 |
|------|------|------|--------|
| outlier_removal | str | 异常值处理方法 | none, zscore, iqr |
| split_method | str | 数据划分方式 | sequential, random, label_random |
| use_time_features | bool | 是否使用时间特征 | True, False |
| num_features | int | 总特征数量 | 6, 11 (6负载+5时间) |
| feature_list | str | 使用的负载特征列表（逗号分隔） | "HUFL,HULL,MUFL,MULL,LUFL,LULL" |

### 超参数

| 字段 | 类型 | 说明 | 示例值 |
|------|------|------|--------|
| learning_rate | float | 学习率 | 0.0001, 0.001, 0.01 |
| batch_size | int | 批次大小 | 16, 32, 64, 128 |
| hidden_size | int | 隐藏层大小（RNN系列） | 32, 64, 128, 256 |
| num_layers | int | RNN层数 | 1, 2, 3 |
| dropout | float | Dropout率 | 0.0, 0.2, 0.4 |
| bidirectional | bool | 是否双向RNN | True, False |
| epochs | int | 训练轮次（配置值） | 100 |
| early_stopping_patience | int | 早停耐心值 | 10 |

### 性能指标

| 字段 | 类型 | 说明 | 意义 |
|------|------|------|------|
| train_r2 | float | 训练集R²分数 | 拟合优度，越接近1越好 |
| test_r2 | float | **测试集R²分数（主指标）** | 泛化能力，越接近1越好 |
| train_mse | float | 训练集均方误差 | 越小越好 |
| test_mse | float | 测试集均方误差 | 越小越好 |
| train_mae | float | 训练集平均绝对误差 | 可解释误差（℃），越小越好 |
| test_mae | float | 测试集平均绝对误差 | 可解释误差（℃），越小越好 |
| training_time_sec | float | 训练时间（秒） | 效率指标 |

### 字段使用示例

```python
import pandas as pd

# 读取结果
df = pd.read_csv('../experiments/results.csv')

# 查找最优模型（按test_r2排序）
best_model = df.loc[df['test_r2'].idxmax()]
print("最优模型:", best_model['experiment_id'])
print("Test R²:", best_model['test_r2'])

# 对比不同超参数
lr_comparison = df[df['stage'] == 6].groupby('learning_rate')['test_r2'].mean()
print("不同学习率的平均R²:")
print(lr_comparison)

# 筛选特定配置
high_performance = df[(df['test_r2'] > 0.8) & (df['training_time_sec'] < 300)]
print(f"高性能且快速的模型: {len(high_performance)} 个")
```

---

## 注意事项

### 1. 数据集选择策略

- **阶段1-6**：主要在 `train2.csv` 上进行
  - 原因：train2 数据范围更大，更容易训练
- **阶段7**：在 `train1.csv` 上验证
  - 目的：测试模型泛化能力

### 2. 模型保存规则

- **自动保存**：每个实验的模型自动保存到 `models/{experiment_id}.pth`
- **最优模型**：每个阶段结束后，手动备份最优模型为 `models/stage{N}_best.pth`
- **磁盘空间**：88个模型约占500MB-1GB空间

### 3. 结果追加机制

- 所有实验结果**追加**到 `results.csv`，不覆盖
- 通过 `experiment_id` 去重，避免重复运行
- 如需重新运行，需手动删除 CSV 中的对应行

### 4. 可重现性保证

- 所有实验使用固定随机种子：`seed=42`
- 在 `notebooks/utils.py` 中设置：
  ```python
  np.random.seed(42)
  torch.manual_seed(42)
  ```
- 相同配置应产生相同结果（CPU模式下）

### 5. 早停机制

- 所有模型使用 Early Stopping（patience=10）
- 当测试集loss连续10个epoch不下降时停止训练
- 自动加载最佳模型权重（test loss最低的epoch）

### 6. 学习率调度

- 使用 `ReduceLROnPlateau`：
  - 当测试集loss连续5个epoch不下降
  - 学习率衰减：`lr = lr * 0.5`
- 帮助模型跳出局部最优

### 7. 时间预算建议

- **阶段1**：必须完成，是后续阶段的基础
- **阶段2-4**：强烈建议完成，影响最大
- **阶段5**：可选，如果阶段1已确定最优算法可跳过
- **阶段6**：可选，如果时间不足可使用默认超参数
- **阶段7**：建议完成，验证泛化能力

---

## 技术支持 / Support

### 相关文档

- **项目总体说明**：`../CLAUDE.md`
- **实验配置代码**：`experiments/experiment_configs.py`
- **Linear 模型实现**：`notebooks/linear_regression.ipynb`
- **RNN 模型实现**：`notebooks/rnn.ipynb`
- **共享工具函数**：`notebooks/utils.py`

### 常用命令速查

```python
# 获取阶段配置
configs = get_stage_configs(stage_number)

# 批量运行实验
results = run_experiments_batch(configs, skip_completed=True)

# 分析结果
summary = analyze_stage_results(stage_number)

# 生成报告
generate_stage_report(stage_number)

# 查看已完成实验
completed = load_completed_experiments()
print(f"已完成 {len(completed)} 个实验")
```

### 下一步行动计划

1. ✅ 运行阶段1实验（6个模型），确定最优算法
2. ✅ 分析阶段1结果，选择后续实验的基准算法
3. ✅ 运行阶段2实验（10个模型），优化数据预处理
4. ✅ 运行阶段3-4实验（24个模型），优化特征和窗口
5. ✅ 运行阶段5实验（12个模型），对比算法性能
6. ✅ 运行阶段6实验（15-27个模型），精调超参数
7. ✅ 运行阶段7实验（9个模型），最终验证
8. ✅ 生成最终实验报告，总结各因素影响
9. ✅ 撰写论文/报告

---

**祝实验顺利！ / Happy experimenting!** 🎉

如有问题，请参考[常见问题 FAQ](#常见问题-faq)章节。
