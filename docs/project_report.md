# 电力变压器油温预测项目报告

## 项目概述

项目目标是在缺乏实时温度传感器的场景下，通过历史负载数据预测电力变压器未来的油温变化。仓库中提供的脚本形成一个端到端流程：数据预处理 → 传统机器学习模型 → 神经网络基线模型。重点在于构建可复现实验、避免时间序列信息泄露，并记录超参数与评估结果。

## 数据集描述

- **数据来源**：两台变压器的运行记录，字段包括 6 个负载特征与油温。
- **时间粒度**：原始样本间隔约 15 分钟，可通过 `date` 列确认。
- **核心字段**：`HUFL`, `HULL`, `MUFL`, `MULL`, `LUFL`, `LULL`, `OT`。
- **数据准备**：用户需在 `data/` 目录下提供 `trans_1.csv`、`trans_2.csv` 或同结构文件。脚本会自动按时间戳排序并合并。

## 预测配置

`scripts/config.py` 定义了三个预测任务：

1. **1h**：
   - 回溯窗口：32 个时间步（约 8 小时）。
   - 预测跨度：4 个时间步（1 小时后）。
   - 滑动步长：1。
2. **1d**：
   - 回溯窗口：96 个时间步（1 天）。
   - 预测跨度：96 个时间步（1 天后）。
   - 滑动步长：2，减少高度重叠的样本。
3. **1w**：
   - 回溯窗口：672 个时间步（1 周）。
   - 预测跨度：672 个时间步（1 周后）。
   - 滑动步长：6，应对样本数量与内存压力。

每个配置都可通过命令行参数调整或扩展。

## 数据预处理流程

1. **加载与排序**：
   - 自动读取 `data/` 目录下符合模式的 CSV 文件。
   - 使用 `pandas.to_datetime` 解析时间并排序，避免不同文件的时间交叉。
2. **序列构建**：
   - 基于滑动窗口生成三维张量（样本 × 时间步 × 特征）。
   - 支持步长与最大样本控制，既能覆盖连续时间段又能限制内存占用。
3. **时间顺序划分**：
   - 采用 70%/15%/15% 的训练、验证、测试比例，并保持时间先后顺序。
   - 若样本数量不足，脚本会提示调整窗口或收集更多数据。
4. **标准化**：
   - `StandardScaler` 仅在训练集上拟合，再对验证与测试集变换。
   - 训练好的标准化器与元数据保存到 `artifacts/scaler_<config>.pkl` 与 `metadata_<config>.json`。

## 模型设计

### 传统机器学习

脚本 `scripts/models/simple_ml_models.py` 提供以下模型：

- **LinearRegression**：作为基线。
- **Ridge**：带 L2 正则项，通过网格搜索选择 `alpha`。
- **RandomForestRegressor**：调节树数量、深度与叶子节点大小。

训练流程：

1. 读取预处理生成的三维数据并展平。
2. 对训练集执行时间序列交叉验证（默认 3 折）进行超参数搜索。
3. 使用验证集评估不同模型；最终指标在测试集上报告。
4. 将训练好的模型参数与性能写入 `artifacts/` 目录。

### 神经网络基线

`scripts/models/simple_deep_models.py` 使用 `MLPRegressor` 作为全连接神经网络：

- 默认结构为三层（128-64-32），启用 `early_stopping=True`。
- 网格搜索可调节隐藏层结构、学习率与 L2 正则系数。
- 同样使用时间序列交叉验证，避免随机打乱导致的泄露。
- 可通过 `--ml-metrics` 与 `--combined-output` 合并传统模型与神经网络的结果，形成综合对比。

## 评估指标

- **均方误差 (MSE)** 与 **均方根误差 (RMSE)**：衡量预测偏差。
- **平均绝对误差 (MAE)**：反映平均偏离程度。
- **决定系数 (R²)**：衡量模型解释能力。

所有指标均在验证与测试集上分别记录，便于判断是否过拟合。

## 实验复现指南

1. 准备数据：
   ```bash
   mkdir -p data/raw
   # 将 trans_1.csv 与 trans_2.csv 放入 data/raw/
   ```
2. 生成序列数据：
   ```bash
   python scripts/preprocessing/optimized_preprocessing.py --data-dir data/raw --output-dir artifacts
   ```
3. 训练传统模型：
   ```bash
   python scripts/models/simple_ml_models.py --preprocessed-dir artifacts --results-dir artifacts/models
   ```
4. 训练神经网络：
   ```bash
   python scripts/models/simple_deep_models.py --preprocessed-dir artifacts --model-dir artifacts/models --ml-metrics artifacts/simple_ml_results.csv --combined-output artifacts/final_model_comparison.csv
   ```

## 结果与分析

由于仓库未附带原始数据，默认不会生成即用的性能指标。上述脚本会输出以下文件，供研究者自行分析：

- `artifacts/simple_ml_results.csv`：传统模型在验证/测试集的表现。
- `artifacts/simple_deep_results.csv`：神经网络基线的表现。
- `artifacts/final_model_comparison.csv`：如提供 `--combined-output`，则包含全部模型的对比。
- `artifacts/simple_*_best_params.json`：每个配置的最佳超参数。

研究者可使用这些结果绘制性能曲线、比较不同窗口或模型的效果，并根据需要扩展更多模型。

## 改进方向

- **特征增强**：加入滚动统计量、日内周期、环境温度等外生变量。
- **模型扩展**：引入 LSTM/GRU、梯度提升树等更适合时间序列的模型。
- **评估策略**：实现滚动预测或多窗口回测，以检验模型在真实调度中的表现。
- **自动化实验**：结合 `hydra` 或 `mlflow` 等工具管理配置与结果。

## 总结

本项目将原始的示例代码整理为可复现的时间序列实验框架，强调以下原则：

1. **时间一致性**：所有拆分与交叉验证都保持时间顺序，防止未来信息泄露。
2. **配置集中管理**：窗口长度、预测跨度等关键参数集中在 `scripts/config.py` 中维护。
3. **结果可追踪**：预处理元数据、模型参数与评估指标全部持久化，便于比较与复现实验。

借助该框架，研究者可以快速替换模型、调整预测配置，并在相同的数据准备流程下公平比较不同方案的表现。
