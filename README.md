# 电力变压器油温预测项目

该仓库提供一个可配置的实验框架，用于基于历史负载数据预测电力变压器的油温。项目涵盖数据预处理、传统机器学习模型以及简单的神经网络基线，重点演示如何在时间序列场景中避免数据泄露并记录实验配置。

## 仓库结构

```
.
├── README.md
├── PROJECT_STRUCTURE.md
├── docs/
│   ├── project_report.md
│   ├── review_report.md
│   └── README.md
├── scripts/
│   ├── __init__.py
│   ├── config.py
│   ├── preprocessing/
│   │   └── optimized_preprocessing.py
│   └── models/
│       ├── simple_deep_models.py
│       └── simple_ml_models.py
└── models/                # 保留历史模型文件的占位目录
```

所有新的数据制品会保存到 `artifacts/` 目录，便于与源码区分。

## 数据准备

仓库不包含原始数据文件。请自行获取 `trans_1.csv` 和 `trans_2.csv`（或格式相同的 CSV 文件），并确保包含以下字段：

| 列名 | 含义 |
|------|------|
| `date` | 时间戳，能够被 `pandas.to_datetime` 解析 |
| `HUFL`, `HULL`, `MUFL`, `MULL`, `LUFL`, `LULL` | 高频、中频、低频侧的有功/无功负载 |
| `OT` | 油温（预测目标） |

推荐将原始文件放在 `data/raw/` 目录下，例如：

```
project/
├── data/
│   └── raw/
│       ├── trans_1.csv
│       └── trans_2.csv
```

如需使用其他文件名，可通过命令行参数指定。

## 运行环境

- Python 3.9+
- 依赖库：`numpy`、`pandas`、`scikit-learn`、`joblib`

可使用 `pip install -r requirements.txt`（若自建）或手动安装上述依赖。

## 实验配置

`scripts/config.py` 集中维护三种预测配置：

| 名称 | 回溯窗口 (`lookback`) | 预测跨度 (`forecast_horizon`) | 步长 (`step`) | 说明 |
|------|-----------------------|------------------------------|---------------|------|
| `1h` | 32 个时间步（约 8 小时） | 4 个时间步（1 小时后） | 1 | 细粒度短期预测 |
| `1d` | 96 个时间步（1 天） | 96 个时间步（1 天后） | 2 | 中期预测，步长 2 以降低样本数量 |
| `1w` | 672 个时间步（1 周） | 672 个时间步（1 周后） | 6 | 长期预测，跨周窗口 |

若需自定义窗口，可复制 `ExperimentConfig` 并在命令行通过 `--configs` 指定。

## 快速开始

以下命令均假设在仓库根目录执行。

### 1. 数据预处理

```bash
python scripts/preprocessing/optimized_preprocessing.py \
  --data-dir data/raw \
  --output-dir artifacts
```

- 自动加载 `trans_*.csv`，按时间排序并拼接。
- 根据配置生成三维序列（样本 × 时间步 × 特征）。
- 采用时间顺序拆分训练/验证/测试集，避免泄露。
- 标准化器仅在训练集上拟合，并保存至 `artifacts/scaler_<config>.pkl`。
- 每个配置还会生成对应的 `metadata_<config>.json`。

### 2. 训练传统机器学习模型

```bash
python scripts/models/simple_ml_models.py \
  --preprocessed-dir artifacts \
  --results-dir artifacts/models \
  --metrics-path artifacts/simple_ml_results.csv
```

- 默认执行基准线性回归、Ridge 与随机森林。
- 启用时间序列交叉验证的网格搜索以选择超参数（可通过 `--disable-search` 关闭）。
- 输出：
  - 训练好的模型（`linear_*.pkl`、`ridge_*.pkl`、`random_forest_*.pkl`）。
  - 评估指标（验证集与测试集）写入 `simple_ml_results.csv`。
  - 最佳超参数写入 `simple_ml_best_params.json`。

### 3. 训练神经网络基线

```bash
python scripts/models/simple_deep_models.py \
  --preprocessed-dir artifacts \
  --model-dir artifacts/models \
  --metrics-path artifacts/simple_deep_results.csv
```

- 使用 `MLPRegressor` 作为全连接神经网络基线。
- 默认启用时间序列网格搜索，可用 `--disable-search` 固定默认结构。
- 结果写入 `simple_deep_results.csv` 和 `simple_deep_best_params.json`。
- 如需同时输出传统模型与神经网络的综合对比，可添加：

```bash
  --ml-metrics artifacts/simple_ml_results.csv \
  --combined-output artifacts/final_model_comparison.csv
```

### 4. 后续分析

`artifacts/metadata_<config>.json` 中记录了窗口长度、样本数量等信息，可辅助复现实验。若需可视化或进一步建模，可在此基础上扩展脚本。

## 评估指标

所有模型统一使用以下指标衡量性能：

- **MSE / RMSE**：均方误差及其平方根，衡量预测偏差。
- **MAE**：平均绝对误差，体现平均偏离程度。
- **R²**：决定系数，评估模型解释能力。

验证集用于调参与早停，最终测试集分数反映泛化性能。

## 注意事项

- 数据拆分完全按照时间顺序进行，确保未来信息不会泄露给训练阶段。
- 长期预测配置需要较长的历史窗口，若原始数据不足请调整 `lookback` 或 `forecast_horizon`。
- `artifacts/` 目录不会提交到版本库，可在 `.gitignore` 中保持忽略状态。
- 若使用新的特征列，请通过 `--feature-cols` 参数显式指定，并确保在三个脚本中保持一致。

## 更多信息

- 详细背景与方法论请参考 `docs/project_report.md`。
- 代码审查与改进建议记录在 `docs/review_report.md`。
- `PROJECT_STRUCTURE.md` 提供更细的目录结构说明。
