# 实验结果存储位置说明 📁

## 📂 目录结构总览

```
项目根目录/
├── models/                              # 模型文件
│   ├── baseline/                        # 基线模型（时序分割）
│   ├── random_split/                    # 随机划分模型
│   ├── horizon_experiments/             # 多时距实验模型
│   └── experiments/                     # 新实验系统模型 🆕
│
├── results/                             # 实验结果
│   ├── figures/                         # 可视化图表
│   └── tables/                          # 数据表格
│
└── processed/                           # 预处理数据
    ├── tx1_cleaned.csv                  # 清洗后的数据
    ├── tx1_cleaned_1pct.csv             # 1%剔除版本
    └── ...
```

---

## 🗂️ 详细说明

### 1. 模型文件位置

#### A. 旧训练脚本的模型（保持不变）

| 脚本 | 输出目录 | 文件格式 |
|------|---------|---------|
| `model_training.py` | `models/baseline/` | `tx{id}_{model}.joblib` |
| `model_random_split.py` | `models/random_split/` | `tx{id}_{model}.joblib` |
| `model_horizon_experiments.py` | `models/horizon_experiments/` | `tx{id}_{config}_{model}.joblib` |

**示例**：
```
models/baseline/
├── tx1_RandomForest.joblib
├── tx1_MLP.joblib
├── tx2_RandomForest.joblib
└── tx2_MLP.joblib

models/random_split/
├── tx1_RandomForest.joblib
├── tx1_MLP.joblib
├── tx1_LinearRegression.joblib
└── tx1_Ridge.joblib

models/horizon_experiments/
├── tx1_1h_RandomForest.joblib
├── tx1_1d_MLP.joblib
└── ...
```

---

#### B. 新实验系统的模型 🆕

| 脚本 | 输出目录 | 文件格式 |
|------|---------|---------|
| `train_configurable.py` | `models/experiments/` | `{exp_id}_model.joblib` |
| `run_experiments.py` | `models/experiments/` | `exp_{序号}_model.joblib` |

**输出文件**：
```
models/experiments/
├── exp_001_model.joblib              # 实验 1 的模型
├── exp_001_metrics.json              # 实验 1 的指标
├── exp_002_model.joblib              # 实验 2 的模型
├── exp_002_metrics.json              # 实验 2 的指标
├── ...
└── experiment_summary.csv            # 所有实验汇总 ⭐
```

**查看方法**：
```bash
# 查看所有实验模型
ls models/experiments/

# 查看某个实验的指标
cat models/experiments/exp_001_metrics.json

# 查看所有实验汇总
cat models/experiments/experiment_summary.csv
```

---

### 2. 预测结果（CSV 表格）

**位置**：`results/tables/`

#### 旧脚本输出

```
results/tables/
├── tx1_RandomForest_predictions.csv         # TX1 RF 预测
├── tx1_MLP_predictions.csv                  # TX1 MLP 预测
├── random_tx1_RandomForest_predictions.csv  # 随机划分预测
└── horizon_tx1_1h_MLP.csv                   # 多时距预测
```

#### 新脚本输出 🆕

```
results/tables/
├── exp_001_predictions.csv                  # 实验 1 预测结果
├── exp_002_predictions.csv                  # 实验 2 预测结果
└── ...
```

**文件内容**（CSV 格式）：
```csv
timestamp,actual,predicted
2016-01-01 00:00:00,55.23,54.89
2016-01-01 00:15:00,55.45,55.12
...
```

---

### 3. 可视化图表（PNG 图片）

**位置**：`results/figures/`

#### 旧脚本输出

```
results/figures/
├── tx1_RandomForest_prediction.png          # 预测曲线图
├── tx1_MLP_scatter.png                      # 散点图
├── random_tx1_RandomForest_scatter.png      # 随机划分散点图
└── ...
```

#### 新脚本输出 🆕

```
results/figures/
├── exp_001_predictions.png                  # 实验 1 预测曲线
├── exp_001_scatter.png                      # 实验 1 散点图
├── exp_002_predictions.png                  # 实验 2 预测曲线
├── exp_002_scatter.png                      # 实验 2 散点图
└── ...
```

**图表类型**：
1. **预测曲线图**：时间序列曲线，实际值 vs 预测值
2. **散点图**：实际值 vs 预测值的散点分布

---

### 4. 评估指标（JSON 文件）🆕

**位置**：`models/experiments/{exp_id}_metrics.json`

**文件内容**（JSON 格式）：
```json
{
  "experiment_id": "exp_001",
  "transformer_id": 1,
  "model": "RandomForest",
  "split_method": "random_window",
  "feature_mode": "full",
  "data_suffix": "",
  "test_ratio": 0.2,
  "n_features": 17,
  "n_train": 32000,
  "n_test": 8000,
  "train_time": 12.5,
  "RMSE": 2.34,
  "MAE": 1.89,
  "R2": 0.95,
  "lookback": 4,
  "horizon": 1,
  "gap": 0,
  "lookback_multiplier": 4.0
}
```

---

### 5. 实验汇总（CSV 文件）⭐

**位置**：`models/experiments/experiment_summary.csv`

这是**最重要**的文件，包含所有实验的结果！

**文件内容**（示例）：
```csv
experiment_id,transformer_id,model,split_method,feature_mode,RMSE,MAE,R2,train_time
exp_001,1,RandomForest,chronological,full,2.34,1.89,0.95,12.5
exp_002,1,RandomForest,random_window,full,2.12,1.67,0.96,15.3
exp_003,1,RandomForest,group_random,full,2.45,1.98,0.94,14.1
exp_004,1,MLP,random_window,full,2.08,1.65,0.97,23.4
exp_005,1,RNN,random_window,full,2.05,1.62,0.97,35.2
...
```

**查看方法**：
```bash
# 直接查看（可能很长）
cat models/experiments/experiment_summary.csv

# 格式化查看（使用 column）
column -t -s',' models/experiments/experiment_summary.csv | less

# 查看前 10 行
head -10 models/experiments/experiment_summary.csv

# 按 R² 排序查看最佳模型
sort -t',' -k9 -rn models/experiments/experiment_summary.csv | head -10
```

---

## 🔍 如何查找特定实验的结果

### 方法1：通过实验 ID

如果您知道实验 ID（如 `exp_005`）：

```bash
# 查看指标
cat models/experiments/exp_005_metrics.json

# 查看预测结果
cat results/tables/exp_005_predictions.csv

# 查看预测曲线图
open results/figures/exp_005_predictions.png  # macOS
# 或
xdg-open results/figures/exp_005_predictions.png  # Linux
```

### 方法2：在汇总文件中搜索

```bash
# 查找所有 RNN 实验
grep "RNN" models/experiments/experiment_summary.csv

# 查找 TX1 的实验
grep "^exp_.*,1," models/experiments/experiment_summary.csv

# 查找 R² > 0.96 的实验
awk -F',' '$9 > 0.96' models/experiments/experiment_summary.csv
```

### 方法3：Python 分析

```python
import pandas as pd

# 读取汇总文件
df = pd.read_csv('models/experiments/experiment_summary.csv')

# 查看前 5 个实验
print(df.head())

# 按 R² 排序
best = df.nlargest(10, 'R2')
print(best)

# 筛选特定条件
rnn_experiments = df[df['model'] == 'RNN']
print(rnn_experiments)

# 对比不同模型
print(df.groupby('model')[['RMSE', 'MAE', 'R2']].mean())
```

---

## 📊 实验命名规则

### 自动生成的实验 ID

**格式**：`exp_{序号:03d}`

- `exp_001`：实验 1
- `exp_002`：实验 2
- `exp_045`：实验 45

### 自定义实验名称

使用 `--experiment-name` 参数：

```bash
python -m scripts.train_configurable \
    --tx-id 1 --model RNN \
    --experiment-name "my_rnn_test"
```

输出：
```
models/experiments/my_rnn_test_model.joblib
models/experiments/my_rnn_test_metrics.json
results/tables/my_rnn_test_predictions.csv
results/figures/my_rnn_test_predictions.png
```

---

## 🗄️ 数据文件位置

### 原始数据

```
data/
├── ETTh1.csv
└── ETTh2.csv
```

### 预处理数据

```
processed/
├── tx1_cleaned.csv                    # 默认（IQR 剔除）
├── tx1_standardized.csv               # 标准化版本
├── tx1_cleaned_no_outlier.csv         # 无剔除
├── tx1_cleaned_1pct.csv               # 1% 剔除
├── tx1_cleaned_5pct.csv               # 5% 剔除
├── tx2_cleaned.csv
├── tx2_standardized.csv
└── ...
```

---

## 💡 最佳实践

### 1. 批量运行实验

```bash
# 运行所有实验
python -m scripts.run_experiments \
    --config experiment/experiment_group.csv

# 结果会自动保存到 models/experiments/
# 汇总到 experiment_summary.csv
```

### 2. 组织实验结果

建议创建子目录：

```bash
# 创建实验批次目录
mkdir -p models/experiments/batch_1
mkdir -p models/experiments/batch_2

# 移动实验结果
mv models/experiments/exp_0* models/experiments/batch_1/
```

### 3. 备份重要结果

```bash
# 备份汇总文件
cp models/experiments/experiment_summary.csv \
   models/experiments/experiment_summary_$(date +%Y%m%d).csv

# 打包实验结果
tar -czf experiments_backup.tar.gz models/experiments/ results/
```

---

## 🔄 清理旧结果

如果想重新运行实验：

```bash
# 清理实验结果（谨慎！）
rm -rf models/experiments/*
rm -rf results/figures/exp_*
rm -rf results/tables/exp_*

# 或仅删除特定实验
rm models/experiments/exp_001*
rm results/tables/exp_001*
rm results/figures/exp_001*
```

---

## 📈 快速查看结果

### 命令行快速查看

```bash
# 查看最新的 5 个实验
ls -lt models/experiments/*.json | head -5

# 统计实验数量
ls models/experiments/exp_*_model.joblib | wc -l

# 查看模型类型分布
grep -o '"model": "[^"]*"' models/experiments/*.json | sort | uniq -c
```

### 可视化工具（可选）

使用 Jupyter Notebook 分析：

```python
import pandas as pd
import matplotlib.pyplot as plt

# 读取汇总
df = pd.read_csv('models/experiments/experiment_summary.csv')

# 绘制模型对比图
df.groupby('model')[['RMSE', 'MAE', 'R2']].mean().plot(kind='bar')
plt.title('Model Performance Comparison')
plt.show()

# 绘制不同划分方式的对比
df.groupby('split_method')['R2'].mean().plot(kind='bar')
plt.title('Split Method Comparison')
plt.show()
```

---

## 📚 相关文档

- **实验指南**：[EXPERIMENT_GUIDE.md](EXPERIMENT_GUIDE.md)
- **快速入门**：[EXPERIMENT_QUICKSTART.md](EXPERIMENT_QUICKSTART.md)
- **RNN 使用**：[RNN_MODEL_GUIDE.md](RNN_MODEL_GUIDE.md)

---

## ⭐ 重点总结

| 文件类型 | 位置 | 最重要 |
|---------|------|--------|
| **实验汇总** | `models/experiments/experiment_summary.csv` | ⭐⭐⭐ |
| 模型文件 | `models/experiments/{exp_id}_model.joblib` | ⭐⭐ |
| 评估指标 | `models/experiments/{exp_id}_metrics.json` | ⭐⭐ |
| 预测结果 | `results/tables/{exp_id}_predictions.csv` | ⭐ |
| 可视化图 | `results/figures/{exp_id}_predictions.png` | ⭐ |

---

**最重要的文件就是：`models/experiments/experiment_summary.csv`** 📊

这个文件包含了所有实验的完整结果，是分析和对比的核心！
