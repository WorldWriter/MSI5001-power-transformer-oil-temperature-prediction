# 实验参数化系统使用指南

## 📋 概述

本实验参数化系统提供了灵活的接口，支持系统化地运行多组对比实验，涵盖：
- **异常值剔除策略**：无剔除、IQR、百分比剔除（0.5%/1%/5%）
- **数据划分方式**：时序分割、滑动窗口随机、分组随机
- **特征配置**：全特征、仅时间特征、无时间特征
- **时间窗口配置**：不同的 lookback 倍数（1x/4x/8x）和预测时长（1h/1d/1w）

---

## 🚀 快速开始

### 方式1：批量运行所有实验

```bash
# 1. 运行所有实验（自动读取 experiment/experiment_group.csv）
python -m scripts.run_experiments \\
    --config experiment/experiment_group.csv \\
    --run-preprocessing

# 2. 仅运行指定实验
python -m scripts.run_experiments \\
    --config experiment/experiment_group.csv \\
    --exp-ids 1,2,3

# 3. 预览命令（不实际执行）
python -m scripts.run_experiments \\
    --config experiment/experiment_group.csv \\
    --dry-run
```

### 方式2：单独运行单个实验

```bash
# 步骤1: 预处理数据（如需要特定的异常值剔除策略）
python -m scripts.preprocessing_configurable \\
    --outlier-method percentile \\
    --outlier-percentile 1.0 \\
    --save-suffix "_1pct"

# 步骤2: 训练模型
python -m scripts.train_configurable \\
    --tx-id 1 \\
    --model RandomForest \\
    --split-method random_window \\
    --data-suffix "_1pct" \\
    --feature-mode full \\
    --lookback-multiplier 4 \\
    --horizon 1
```

---

## 📁 新增文件说明

### 1. `scripts/experiment_utils.py`
核心工具模块，提供：
- `select_features_by_mode()` - 特征选择
- `remove_outliers_configurable()` - 可配置的异常值检测
- `chronological_split()` - 时序划分
- `group_random_split()` - 分组随机划分
- `WindowConfig` - 窗口配置管理

### 2. `scripts/preprocessing_configurable.py`
可配置的数据预处理脚本，支持：
- 多种异常值检测方法
- 自定义剔除比例
- 输出不同版本的清洗数据

### 3. `scripts/train_configurable.py`
统一训练接口，支持：
- 所有模型（RandomForest, MLP, LinearRegression, Ridge）
- 所有数据划分方式
- 灵活的特征和窗口配置

### 4. `scripts/run_experiments.py`
批量实验运行器，支持：
- 从 CSV 读取实验配置
- 自动运行多组实验
- 结果汇总

---

## 🔧 详细参数说明

### 预处理参数（preprocessing_configurable.py）

| 参数 | 选项 | 说明 |
|------|------|------|
| `--outlier-method` | none, iqr, percentile | 异常值检测方法 |
| `--outlier-percentile` | 0.5, 1.0, 5.0 | 百分比剔除阈值 |
| `--iqr-multiplier` | 默认 1.5 | IQR 方法的倍数 |
| `--save-suffix` | 如 "_1pct" | 输出文件后缀 |

**示例**：
```bash
# 不剔除异常值
python -m scripts.preprocessing_configurable \\
    --outlier-method none \\
    --save-suffix "_no_outlier"

# 剔除最极端的 1%
python -m scripts.preprocessing_configurable \\
    --outlier-method percentile \\
    --outlier-percentile 1.0 \\
    --save-suffix "_1pct"

# 剔除最极端的 5%
python -m scripts.preprocessing_configurable \\
    --outlier-method percentile \\
    --outlier-percentile 5.0 \\
    --save-suffix "_5pct"

# 使用 IQR 方法（默认）
python -m scripts.preprocessing_configurable \\
    --outlier-method iqr
```

---

### 训练参数（train_configurable.py）

#### 数据配置

| 参数 | 选项 | 说明 |
|------|------|------|
| `--tx-id` | 1, 2 | 变压器 ID（必需） |
| `--data-suffix` | "", "_1pct", etc. | 数据文件后缀 |

#### 模型配置

| 参数 | 选项 | 说明 |
|------|------|------|
| `--model` | RandomForest, MLP, LinearRegression, Ridge | 模型类型（必需） |

#### 数据划分配置

| 参数 | 选项 | 说明 |
|------|------|------|
| `--split-method` | chronological, random_window, group_random | 划分方式（必需） |
| `--test-ratio` | 0.2（默认） | 测试集比例 |
| `--n-groups` | 20（默认） | 分组数量（group_random） |
| `--random-state` | 42（默认） | 随机种子 |

#### 特征配置

| 参数 | 选项 | 说明 |
|------|------|------|
| `--feature-mode` | full, time_only, no_time | 特征选择模式 |

**特征模式说明**：
- `full`：负载特征 + 时间特征（默认）
- `time_only`：仅时间特征（hour, dayofweek, season等）
- `no_time`：仅负载特征（HULL, MULL）

#### 窗口配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--lookback-multiplier` | 4.0 | Lookback = horizon × multiplier |
| `--horizon` | 1 | 预测步数 |
| `--gap` | 0 | 窗口和目标之间的间隔 |
| `--max-windows` | 40000 | 最大窗口数量 |

**示例**：
```bash
# 时序分割 + RandomForest
python -m scripts.train_configurable \\
    --tx-id 1 \\
    --model RandomForest \\
    --split-method chronological \\
    --feature-mode full

# 滑动窗口随机 + MLP + 仅时间特征
python -m scripts.train_configurable \\
    --tx-id 1 \\
    --model MLP \\
    --split-method random_window \\
    --feature-mode time_only \\
    --lookback-multiplier 8 \\
    --horizon 1

# 分组随机 + Ridge + 无时间特征
python -m scripts.train_configurable \\
    --tx-id 2 \\
    --model Ridge \\
    --split-method group_random \\
    --feature-mode no_time \\
    --n-groups 100

# 使用预处理的 1% 剔除数据
python -m scripts.train_configurable \\
    --tx-id 1 \\
    --model RandomForest \\
    --split-method random_window \\
    --data-suffix "_1pct"
```

---

### 批量运行参数（run_experiments.py）

| 参数 | 说明 |
|------|------|
| `--config` | 实验配置 CSV 文件路径（必需） |
| `--exp-ids` | 运行指定实验 ID（如 "1,2,3"） |
| `--run-preprocessing` | 自动运行预处理 |
| `--dry-run` | 仅显示命令，不执行 |
| `--continue-on-error` | 出错后继续运行 |

**示例**：
```bash
# 运行所有实验
python -m scripts.run_experiments \\
    --config experiment/experiment_group.csv \\
    --run-preprocessing

# 仅运行实验 1-10
python -m scripts.run_experiments \\
    --config experiment/experiment_group.csv \\
    --exp-ids 1,2,3,4,5,6,7,8,9,10

# 预览所有命令
python -m scripts.run_experiments \\
    --config experiment/experiment_group.csv \\
    --dry-run

# 出错后继续（适合长时间批量运行）
python -m scripts.run_experiments \\
    --config experiment/experiment_group.csv \\
    --continue-on-error
```

---

## 📊 实验结果输出

### 输出文件结构

```
models/experiments/
├── exp_001_model.joblib           # 训练好的模型
├── exp_001_metrics.json            # 评估指标（JSON）
└── experiment_summary.csv          # 所有实验汇总

tables/
├── exp_001_predictions.csv         # 预测结果
└── outlier_detection_summary.csv   # 异常值检测统计

figures/
├── exp_001_predictions.png         # 预测曲线图
└── exp_001_scatter.png             # 散点图

processed/
├── tx1_cleaned.csv                 # 默认清洗数据
├── tx1_cleaned_1pct.csv            # 1% 剔除数据
└── tx1_cleaned_no_outlier.csv      # 无剔除数据
```

### 结果汇总文件（experiment_summary.csv）

包含所有实验的关键信息：
- experiment_id, transformer_id, model
- split_method, feature_mode, data_suffix
- RMSE, MAE, R²
- train_time, n_train, n_test

**查看结果**：
```bash
# 查看所有结果
cat models/experiments/experiment_summary.csv

# 按 R² 排序查看最佳模型
sort -t',' -k10 -rn models/experiments/experiment_summary.csv | head
```

---

## 🎯 典型使用场景

### 场景1：对比异常值剔除策略

```bash
# 步骤1: 预处理不同版本的数据
python -m scripts.preprocessing_configurable --outlier-method none --save-suffix "_no_outlier"
python -m scripts.preprocessing_configurable --outlier-method percentile --outlier-percentile 1.0 --save-suffix "_1pct"
python -m scripts.preprocessing_configurable --outlier-method percentile --outlier-percentile 5.0 --save-suffix "_5pct"
python -m scripts.preprocessing_configurable --outlier-method iqr  # 默认版本

# 步骤2: 用相同配置训练
python -m scripts.train_configurable --tx-id 1 --model RandomForest --split-method random_window --data-suffix "_no_outlier"
python -m scripts.train_configurable --tx-id 1 --model RandomForest --split-method random_window --data-suffix "_1pct"
python -m scripts.train_configurable --tx-id 1 --model RandomForest --split-method random_window --data-suffix "_5pct"
python -m scripts.train_configurable --tx-id 1 --model RandomForest --split-method random_window  # 默认
```

### 场景2：对比数据划分方式

```bash
# 时序分割
python -m scripts.train_configurable \\
    --tx-id 1 --model RandomForest --split-method chronological

# 滑动窗口随机
python -m scripts.train_configurable \\
    --tx-id 1 --model RandomForest --split-method random_window

# 分组随机
python -m scripts.train_configurable \\
    --tx-id 1 --model RandomForest --split-method group_random
```

### 场景3：对比特征组合

```bash
# 全特征
python -m scripts.train_configurable \\
    --tx-id 1 --model RandomForest --split-method random_window --feature-mode full

# 仅时间特征
python -m scripts.train_configurable \\
    --tx-id 1 --model RandomForest --split-method random_window --feature-mode time_only

# 无时间特征
python -m scripts.train_configurable \\
    --tx-id 1 --model RandomForest --split-method random_window --feature-mode no_time
```

### 场景4：对比窗口大小

```bash
# 1倍窗口（lookback = horizon）
python -m scripts.train_configurable \\
    --tx-id 1 --model RandomForest --split-method random_window --lookback-multiplier 1 --horizon 1

# 4倍窗口（lookback = 4 * horizon）
python -m scripts.train_configurable \\
    --tx-id 1 --model RandomForest --split-method random_window --lookback-multiplier 4 --horizon 1

# 8倍窗口（lookback = 8 * horizon）
python -m scripts.train_configurable \\
    --tx-id 1 --model RandomForest --split-method random_window --lookback-multiplier 8 --horizon 1
```

---

## 🔍 实验CSV配置说明

`experiment/experiment_group.csv` 格式：

| 列名 | 示例值 | 映射 |
|------|--------|------|
| 验证序号 | 1 | 实验 ID |
| 验证目标 | 目标1: 训练/测试集划分方式的影响 | 实验描述 |
| 验证数据集 | TX1 | transformer_id=1 |
| 验证模型 | RandomForest | model=RandomForest |
| 数据划分方式 | 滑动窗口随机... | split_method=random_window |
| 异常值剔除 | 最多1% | outlier_method=percentile, percentile=1.0 |
| 预测时长 | 1 hour | horizon=1 |
| 有无时间特征... | 加入年月日等特征 | feature_mode=full |
| 时间窗口长度 | 固定时间窗口-4倍 | lookback_multiplier=4.0 |

---

## ⚠️ 注意事项

### 1. 数据文件匹配

如果使用 `--data-suffix`，确保对应的预处理数据文件存在：
```
processed/tx1_cleaned{suffix}.csv
```

否则会报错：
```
FileNotFoundError: Data file not found: processed/tx1_cleaned_1pct.csv
Please run preprocessing first with the same suffix.
```

### 2. 预处理顺序

如果实验需要特殊的异常值剔除策略，**必须先运行预处理**：
```bash
# 错误：直接训练会找不到文件
python -m scripts.train_configurable --tx-id 1 --model RF --data-suffix "_1pct"

# 正确：先预处理
python -m scripts.preprocessing_configurable --outlier-percentile 1.0 --save-suffix "_1pct"
python -m scripts.train_configurable --tx-id 1 --model RF --data-suffix "_1pct"

# 或使用批量运行器自动处理
python -m scripts.run_experiments --config experiment/experiment_group.csv --run-preprocessing
```

### 3. 窗口配置仅适用于部分划分方式

- `chronological`：不使用滑动窗口，忽略 lookback/horizon 参数
- `random_window` 和 `group_random`：使用滑动窗口，需要配置 lookback/horizon

### 4. 内存占用

大规模滑动窗口可能占用大量内存：
- 使用 `--max-windows` 限制窗口数量
- 或分批运行实验

---

## 📈 性能优化建议

### 1. 并行运行实验

修改 `run_experiments.py` 使用多进程：
```python
from multiprocessing import Pool

with Pool(4) as pool:  # 4个并行进程
    pool.map(run_experiment, experiment_configs)
```

### 2. GPU 加速

MLP 模型已支持 GPU（PyTorch），会自动检测：
- CUDA (NVIDIA GPU)
- MPS (Apple Silicon)

### 3. 减少重复预处理

使用 `--run-preprocessing` 时，系统会自动去重相同的预处理配置。

---

## 🐛 故障排除

### 问题1：找不到模块

```bash
ModuleNotFoundError: No module named 'scripts.experiment_utils'
```

**解决**：确保在项目根目录运行命令：
```bash
cd /path/to/MSI5001-power-transformer-oil-temperature-prediction
python -m scripts.train_configurable ...
```

### 问题2：数据文件不存在

```bash
FileNotFoundError: Data file not found: processed/tx1_cleaned_1pct.csv
```

**解决**：先运行预处理：
```bash
python -m scripts.preprocessing_configurable --outlier-percentile 1.0 --save-suffix "_1pct"
```

### 问题3：GPU 不可用

```bash
Using CPU (no GPU available)
```

**MLP 训练慢？** 安装 PyTorch GPU 版本：
```bash
# CUDA (NVIDIA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# MPS (Apple Silicon)
# macOS 13+ 自动支持，确保 torch >= 2.0
```

---

## 📚 附录

### A. 完整的参数映射表

| CSV 列 | 参数名 | 可选值 |
|--------|--------|--------|
| 验证数据集 | --tx-id | 1, 2 |
| 验证模型 | --model | RandomForest, MLP, LinearRegression, Ridge |
| 数据划分方式 | --split-method | chronological, random_window, group_random |
| 异常值剔除 | --outlier-method<br>--outlier-percentile | none/iqr/percentile<br>0.5/1.0/5.0 |
| 有无时间特征 | --feature-mode | full/time_only/no_time |
| 时间窗口长度 | --lookback-multiplier | 1.0/4.0/8.0 |
| 预测时长 | --horizon | 1 (1h) / 24 (1d) / 168 (1w) |

### B. 特征列表

**负载特征** (`LOAD_FEATURES`):
- HULL, MULL

**时间特征** (`TIME_FEATURES`):
- hour, dayofweek, month, day_of_year
- hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos, doy_sin, doy_cos
- is_weekend, is_worktime, season

**TX1 动态特征** (`TX1_DYNAMIC_FEATURES`):
- HULL_diff1, MULL_diff1 (一阶差分)
- HULL_roll12, MULL_roll12 (12步滚动均值)

---

## 📧 联系与支持

如有问题或建议，请：
1. 查看本文档的故障排除部分
2. 检查生成的日志文件
3. 使用 `--dry-run` 查看生成的命令是否正确

**祝实验顺利！** 🎉
