# 实验系统快速入门 ⚡

## 5 分钟快速上手

### 步骤1：验证系统功能

```bash
# 测试工具模块
python -c "from scripts.experiment_utils import select_features_by_mode; print('✓ 工具模块正常')"

# 查看预处理帮助
python -m scripts.preprocessing_configurable --help

# 查看训练帮助
python -m scripts.train_configurable --help
```

### 步骤2：运行单个测试实验

```bash
# 使用默认数据训练一个 RandomForest 模型
python -m scripts.train_configurable \\
    --tx-id 1 \\
    --model RandomForest \\
    --split-method random_window \\
    --feature-mode full
```

预期输出：
```
======================================================================
Experiment: tx1_RandomForest_random_window_full
======================================================================
...
Training RandomForest...
  Training time: XX.XXs

Evaluating...
  RMSE: X.XXXX
  MAE:  X.XXXX
  R²:   X.XXXX
```

### 步骤3：预览批量实验

```bash
# Dry-run: 查看前 3 个实验的命令（不实际运行）
python -m scripts.run_experiments \\
    --config experiment/experiment_group.csv \\
    --exp-ids 1,2,3 \\
    --dry-run
```

### 步骤4：运行批量实验

```bash
# 运行默认实验 (10, 28, 46) - 推荐使用无缓冲模式查看实时输出
python -u -m scripts.run_experiments \\
    --config experiment/experiment_group.csv

# 运行指定实验（如 Informer SOTA 实验）
python -u -m scripts.run_experiments \\
    --config experiment/experiment_group.csv \\
    --exp-ids 64,65,66

# 或运行所有实验（约需 2-4 小时）
python -u -m scripts.run_experiments \\
    --config experiment/experiment_group.csv \\
    --exp-ids 1,2,3,4,5,6,7,8,9,10 \\
    --run-preprocessing
```

**Windows用户**：在PowerShell中使用
```powershell
$env:PYTHONUNBUFFERED=1
python -m scripts.run_experiments --config experiment/experiment_group.csv --exp-ids 64,65,66
```

### 步骤5：查看实验结果

```bash
# 查看实验指标汇总（包含 R², RMSE, MAE, MSE）
cat experiment/metrics_summary.csv

# 查看特定实验的训练日志
cat experiment/logs/exp_010.log
cat experiment/logs/exp_028.log
cat experiment/logs/exp_046.log
```

---

## 常用命令速查

### 预处理数据

```bash
# 不剔除异常值
python -m scripts.preprocessing_configurable --outlier-method none --save-suffix "_no_outlier"

# 剔除 1% 极端值
python -m scripts.preprocessing_configurable --outlier-method percentile --outlier-percentile 1.0 --save-suffix "_1pct"

# 默认 IQR 方法
python -m scripts.preprocessing_configurable --outlier-method iqr
```

### 训练单个模型

```bash
# 时序分割 + RandomForest
python -m scripts.train_configurable --tx-id 1 --model RandomForest --split-method chronological

# 滑动窗口 + MLP
python -m scripts.train_configurable --tx-id 1 --model MLP --split-method random_window

# 滑动窗口 + RNN（新增！适合时序建模）
python -m scripts.train_configurable --tx-id 1 --model RNN --split-method random_window

# 滑动窗口 + Informer（新增！SOTA时序预测模型）
python -m scripts.train_configurable --tx-id 1 --model Informer --split-method random_window

# 使用特定预处理数据
python -m scripts.train_configurable --tx-id 1 --model RandomForest --split-method random_window --data-suffix "_1pct"
```

### 批量运行实验

```bash
# 运行默认实验 (10, 28, 46) - 推荐用于快速测试（使用 -u 查看实时输出）
python -u -m scripts.run_experiments --config experiment/experiment_group.csv

# 运行指定实验（如 Informer SOTA 实验 64-69）
python -u -m scripts.run_experiments --config experiment/experiment_group.csv --exp-ids 64,65,66,67,68,69

# 预览命令（不执行）
python -m scripts.run_experiments --config experiment/experiment_group.csv --dry-run

# 运行所有实验
python -u -m scripts.run_experiments --config experiment/experiment_group.csv --exp-ids 1,2,3,...,69
```

---

## 结果文件位置

### 实验结果汇总（新增！）
- **实验指标汇总**: `experiment/metrics_summary.csv` - 包含所有实验的 R², RMSE, MAE, MSE
- **训练日志**: `experiment/logs/exp_XXX.log` - 每个实验的完整训练输出

### 模型和预测结果
- **模型文件**: `models/experiments/exp_XXX_model.joblib`
- **评估指标**: `models/experiments/exp_XXX_metrics.json`
- **预测结果**: `tables/exp_XXX_predictions.csv`
- **可视化图**: `figures/exp_XXX_predictions.png`
- **结果汇总**: `models/experiments/experiment_summary.csv`

### 快速查看结果示例

```bash
# 查看实验 10, 28, 46 的关键指标
head -n 4 experiment/metrics_summary.csv | column -t -s,

# 检查实验 10 的训练过程
tail -n 50 experiment/logs/exp_010.log
```

---

## 实验参数对照表

| 实验目标 | 参数配置 |
|---------|---------|
| **异常值剔除** | `--outlier-method [none\|iqr\|percentile]` |
| **数据划分** | `--split-method [chronological\|random_window\|group_random]` |
| **特征选择** | `--feature-mode [full\|time_only\|no_time]` |
| **窗口大小** | `--lookback-multiplier [1.0\|4.0\|8.0]` |
| **预测步数** | `--horizon [1\|24\|168]` |

---

## 故障排除

### 问题：找不到数据文件
```bash
FileNotFoundError: Data file not found: processed/tx1_cleaned_1pct.csv
```
**解决**：先运行预处理
```bash
python -m scripts.preprocessing_configurable --outlier-percentile 1.0 --save-suffix "_1pct"
```

### 问题：模块导入错误
```bash
ModuleNotFoundError: No module named 'scripts'
```
**解决**：确保在项目根目录运行
```bash
cd /path/to/MSI5001-power-transformer-oil-temperature-prediction
```

### 问题：GPU 不可用
```bash
Using CPU (no GPU available)
```
**影响**：MLP/RNN/Informer 训练会较慢（但仍可正常运行）
**解决**：安装 PyTorch GPU 版本（可选）
- **NVIDIA GPU (Windows/Linux)**: 支持 CUDA (如 RTX 4070ti Super)
- **Apple Silicon (macOS)**: 支持 MPS (如 M1/M2/M3)

### 问题：实验看起来卡住，没有输出
```bash
# 运行命令后长时间没有输出
python -m scripts.run_experiments --config experiment/experiment_group.csv --exp-ids 64
```
**原因**：Python 输出缓冲（所有平台通用问题：Windows/Linux/macOS）
**解决方案 1 - 使用无缓冲模式（推荐）**：
```bash
# macOS/Linux
python -u -m scripts.run_experiments --config experiment/experiment_group.csv --exp-ids 64

# Windows PowerShell
$env:PYTHONUNBUFFERED=1
python -m scripts.run_experiments --config experiment/experiment_group.csv --exp-ids 64
```

**解决方案 2 - 实时查看日志**：
```bash
# 在另一个终端查看日志（macOS/Linux）
tail -f experiment/logs/exp_064.log

# Windows PowerShell
Get-Content experiment/logs/exp_064.log -Wait
```

---

## 下一步

- 📖 阅读完整文档：[EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md)
- 📊 查看实验计划：[experiment/experiment_group.csv](experiment/experiment_group.csv)
- 🔧 自定义实验参数：修改 CSV 或使用命令行参数

**Happy Experimenting!** 🚀
