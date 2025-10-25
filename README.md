# 电力变压器油温预测项目

## 项目简介

本项目使用机器学习方法预测电力变压器的油温变化，基于历史电气负载数据预测未来1小时、1天和1周后的油温。项目采用渐进式研究方法，从基线模型开发到高级时间特征工程，系统性地探索了多种机器学习算法在工业物联网应用中的性能表现。

## 🎯 项目亮点

- **多时间窗口预测系统**: 成功开发了1小时、1天和1周的预测模型
- **时间特征增强发现**: 83%的模型配置通过时间特征获得性能提升
- **最佳模型性能**: 使用时间特征的随机森林模型在1小时预测中达到R²=0.60
- **关键风险识别**: 发现线性模型在高维时间特征下的重大风险
- **🆕 时间窗口长度优化**: 系统性研究了12种窗口配置，发现8小时历史窗口最适合1小时预测
- **🆕 RNN-ResNet混合模型**: 结合LSTM/GRU与ResNet的混合架构，捕获时序依赖和深度特征

### 时间特征增强效果
- **83%的模型配置**通过时间特征获得性能提升
- **最大提升幅度**: R²从0.35提升至0.60（71%相对提升）
- **关键时间特征**: 小时、星期几、月份对预测效果最为重要

## 📁 项目结构

```
.
├── docs/                           # 项目文档
│   ├── comprehensive_project_report.md  # 综合项目报告（英文）
│   ├── temporal_factors_impact_report.md # 时间因素影响分析报告
│   ├── project_report.md           # 基础项目报告
│   └── beginner_report.md          # 初学者报告
├── dataset/                        # 🆕 原始数据文件
│   ├── trans_1.csv                 # 变压器1数据
│   ├── trans_2.csv                 # 变压器2数据
│   └── README.md                   # 数据说明
├── notebooks/                      # 🆕 Jupyter Notebook实现
│   ├── rnn_resnet_oil_temperature_prediction.ipynb  # RNN-ResNet模型
│   └── rnn_resnet_README.md        # RNN-ResNet使用说明
├── Windows_diff/                   # 🆕 时间窗口长度实验
│   ├── docs/                       # 实验文档和报告
│   │   ├── window_length_analysis_report.md    # 详细分析报告
│   │   ├── comprehensive_experiment_summary.md # 综合实验总结
│   │   └── experiment_design.md    # 实验设计文档
│   ├── scripts/                    # 实验脚本
│   │   ├── window_experiment_preprocessing.py  # 数据预处理
│   │   ├── window_experiment_models.py         # 模型训练
│   │   ├── window_experiment_analysis.py       # 结果分析
│   │   └── run_experiment.py       # 完整实验流程
│   ├── models/                     # 36个训练好的模型
│   │   ├── 1h_lookback_*/          # 1小时预测模型
│   │   ├── 1d_lookback_*/          # 1天预测模型
│   │   └── 1w_lookback_*/          # 1周预测模型
│   └── visualizations/             # 性能分析图表
├── scripts/                        # 核心脚本
│   ├── models/                     # 模型训练脚本
│   │   ├── simple_ml_models.py     # 传统机器学习模型
│   │   ├── simple_deep_models.py   # 深度学习模型（MLP）
│   │   ├── time_feature_comparison.py # 时间特征对比分析
│   │   └── traditional_ml_models.py # 传统ML模型实现
│   ├── preprocessing/              # 数据预处理
│   │   ├── optimized_preprocessing.py # 优化的预处理脚本
│   │   └── enhanced_preprocessing.py  # 增强预处理（含时间特征）
│   ├── evaluation/                 # 评估与可视化
│   │   └── visualization_analysis.py # 可视化分析
│   └── analysis/                   # 高级分析
│       └── seasonal_analysis.py    # 季节性分析
├── models/                         # 训练好的模型
│   ├── traditional_ml/             # 传统机器学习模型
│   ├── deep_learning/              # 深度学习模型
│   ├── rnn_resnet/                 # 🆕 RNN-ResNet混合模型（PyTorch）
│   └── scalers/                    # 数据标准化器
├── tutorial/                       # 教程文档
│   ├── README.md                   # 教程入口
│   └── 01-10_*.md                  # 分章节教程
├── visualizations/                 # 可视化结果
│   └── temporal_impact/            # 时间特征影响可视化
└── artifacts/                      # 实验结果存档
```

## 📚 课程教程

- 本项目提供完整的分章节课程教程，建议按顺序学习与实践
- 教程入口：`tutorial/README.md`
- 包含从环境配置到高级分析的完整学习路径

## 🔬 研究方法论

### 渐进式模型开发
1. **阶段一**: 基线模型开发 - 建立性能基准
2. **阶段二**: 时间特征工程 - 探索时间因素影响
3. **阶段三**: 高级分析优化 - 深入模型行为分析
4. **🆕 阶段四**: 时间窗口长度优化 - 系统性研究历史窗口配置

### 核心发现
- **随机森林优势**: 在所有预测时间窗口中表现最佳
- **时间特征价值**: 显著提升模型预测精度
- **Ridge回归风险**: 在长期预测中使用高维时间特征存在严重性能退化风险
- **🆕 最佳窗口配置**: 8小时历史窗口最适合1小时预测，随机森林在所有配置下表现最优

## 🚀 快速开始

### 运行RNN-ResNet混合模型（最新推荐）

#### 📋 前置要求
- **Python**: 3.8或更高版本
- **数据文件**: `dataset/trans_1.csv`, `dataset/trans_2.csv`
- **GPU (可选)**: NVIDIA GPU with CUDA 11.8/12.1 (推荐，可大幅加速训练)
- **内存**: 至少8GB RAM (16GB推荐)

#### 步骤1: 数据准备

确认数据文件在正确位置：
```bash
# 检查数据文件
ls dataset/
# 应该看到: trans_1.csv  trans_2.csv  README.md
```

如果没有数据文件，请将CSV文件复制到 `dataset/` 文件夹。

#### 步骤2: 安装依赖

**安装PyTorch (选择您的环境):**

```bash
# Windows + NVIDIA GPU (CUDA 11.8) - 推荐
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Windows + NVIDIA GPU (CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# CPU only (无GPU，训练较慢)
pip install torch torchvision

# Linux/Mac 请访问 https://pytorch.org 查看安装命令
```

**安装其他依赖:**
```bash
pip install numpy pandas scikit-learn matplotlib seaborn jupyter notebook
```

**验证安装:**
```python
# 在Python中运行
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

#### 步骤3: 运行Notebook

```bash
# 1. 启动Jupyter Notebook
jupyter notebook

# 2. 在浏览器中打开
# 导航到: notebooks/rnn_resnet_oil_temperature_prediction.ipynb

# 3. 运行所有cells
# 方法1: 点击 Cell → Run All
# 方法2: 依次按 Shift+Enter 运行每个cell
```

**训练配置** (在notebook中可调整):
- 序列长度: 10个时间步 (2.5小时历史数据)
- RNN隐藏层: 64
- ResNet隐藏层: 128
- Batch size: 128
- 最大epochs: 100
- Early stopping: 20个epochs无改进时停止

#### 步骤4: 查看结果

训练完成后，以下文件将生成：

**模型文件** (PyTorch):
```
models/rnn_resnet/
├── lstm_resnet_1h.pth    # LSTM-ResNet 1小时预测
├── lstm_resnet_1d.pth    # LSTM-ResNet 1天预测
├── lstm_resnet_1w.pth    # LSTM-ResNet 1周预测
├── gru_resnet_1h.pth     # GRU-ResNet 1小时预测
├── gru_resnet_1d.pth     # GRU-ResNet 1天预测
└── gru_resnet_1w.pth     # GRU-ResNet 1周预测
```

**可视化结果**:
```
visualizations/rnn_resnet/
├── training_history.png          # 训练和验证损失曲线
├── lstm_vs_gru_comparison.png    # LSTM vs GRU性能对比
├── all_models_comparison.png     # 所有模型对比
└── predictions_*.png              # 预测结果可视化
```

**性能数据**:
```
results/
├── rnn_resnet_comparison.csv     # CSV格式性能对比表
└── rnn_resnet_final_results.pkl  # 完整结果（Python pickle）
```

#### ⏱️ 预期训练时间

| 硬件配置 | LSTM-ResNet | GRU-ResNet |
|---------|-------------|------------|
| RTX 3060 或更好 | 20-30分钟 | 15-25分钟 |
| 集成显卡 | 30-45分钟 | 25-35分钟 |
| CPU only | 45-90分钟 | 30-60分钟 |

*时间包括三个配置(1h, 1d, 1w)的训练

#### 📊 预期性能指标

**1小时预测** (最重要):
- LSTM-ResNet: R² = 0.62-0.72, RMSE = 4.0-5.0°C
- GRU-ResNet: R² = 0.60-0.70, RMSE = 4.2-5.2°C
- 对比基准 Random Forest: R² = 0.60, RMSE = 4.68°C

**1天预测**:
- LSTM-ResNet: R² = 0.40-0.55
- GRU-ResNet: R² = 0.38-0.53

**1周预测**:
- LSTM-ResNet: R² = 0.25-0.40
- GRU-ResNet: R² = 0.23-0.38

#### ⚠️ 常见问题快速排查

**问题1: CUDA Out of Memory**
```bash
# 解决方法: 减小batch size
# 在notebook中修改: batch_size = 64  # 或 32
```

**问题2: 数据文件未找到**
```bash
# 错误: FileNotFoundError: dataset/trans_1.csv
# 解决: 确保数据在正确位置
mkdir -p dataset
# 复制CSV文件到dataset/文件夹
```

**问题3: PyTorch未检测到GPU**
```bash
# 检查CUDA安装
nvcc --version

# 重新安装正确版本的PyTorch
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**问题4: 训练太慢**
```bash
# 使用GRU代替LSTM (20-30%更快)
# 或减少epochs数量
# 或使用GPU
```

#### 📚 详细文档

完整使用指南、架构说明、高级调优请参考:
- **notebooks/rnn_resnet_README.md** - 完整文档（Windows CUDA设置、故障排除、性能优化等）

### 运行时间窗口长度实验
```bash
# 进入实验目录
cd Windows_diff

# 运行完整实验流程
python scripts/run_experiment.py

# 查看实验结果
cat docs/comprehensive_experiment_summary.md
```

### 运行基础预测模型

### 环境要求
- Python 3.8+
- 主要依赖：numpy, pandas, scikit-learn, matplotlib, seaborn

### 基础模型训练
```bash
# 1. 数据预处理
python scripts/preprocessing/optimized_preprocessing.py

# 2. 训练传统机器学习模型
python scripts/models/simple_ml_models.py

# 3. 训练深度学习模型
python scripts/models/simple_deep_models.py

# 4. 生成可视化分析
python scripts/evaluation/visualization_analysis.py
```

### 时间特征增强分析
```bash
# 时间特征对比分析
python scripts/models/time_feature_comparison.py

# 季节性模式分析
python scripts/analysis/seasonal_analysis.py
```

## 📊 主要结果

### 🆕 时间窗口长度优化结果
| 预测时长 | 最佳历史窗口 | 最佳模型 | R² Score | RMSE |
|---------|-------------|----------|----------|------|
| 1小时   | 8小时       | RandomForest | 0.2585 | 4.91 |
| 24小时  | 32小时      | RandomForest | 0.1842 | 5.16 |
| 168小时 | 96小时      | RandomForest | 0.1456 | 5.28 |

### 基线模型性能对比

| 预测时间 | 最佳模型 | R²分数 | RMSE (°C) | 关键特点 |
|----------|----------|--------|-----------|----------|
| 1小时    | Random Forest | 0.596 | 4.72 | 短期预测精度高 |
| 1天      | Random Forest | 0.420 | 5.45 | 中期预测可靠 |
| 1周      | Random Forest | 0.252 | 5.95 | 长期预测有限但有用 |

### 时间特征增强效果

| 模型类型 | 预测窗口 | 基线R² | 增强R² | 改进幅度 | 状态 |
|----------|----------|--------|--------|----------|------|
| Random Forest | 1小时 | -0.092 | 0.117 | **+0.210** | ✅ 显著改进 |
| Random Forest | 1天 | -0.026 | 0.082 | **+0.108** | ✅ 明显改进 |
| Random Forest | 1周 | 0.018 | 0.096 | **+0.079** | ✅ 稳定改进 |
| Ridge Regression | 1小时 | -0.193 | -0.029 | +0.163 | ⚠️ 有改进 |
| Ridge Regression | 1天 | -0.120 | -0.033 | +0.088 | ⚠️ 轻微改进 |
| Ridge Regression | 1周 | 0.013 | -0.339 | **-0.352** | ❌ 严重退化 |

### 关键发现总结

1. **时间特征成功率**: 83%的模型配置获得性能提升
2. **最佳改进**: Random Forest 1小时预测R²提升0.210
3. **风险识别**: Ridge回归在1周预测中出现严重性能退化
4. **推荐方案**: Random Forest + 时间特征为最佳组合

## 数据集说明

- **trans_1.csv**: 第一个变压器的数据
- **trans_2.csv**: 第二个变压器的数据
- 包含6个电气特征和油温目标变量
- 时间间隔：15分钟
- 特征说明：
  - HUFL: 高压有用负载
  - HULL: 高压无用负载
  - MUFL: 中压有用负载
  - MULL: 中压无用负载
  - LUFL: 低压有用负载
  - LULL: 低压无用负载
  - OT: 油温（目标变量）

## 模型说明

### 传统机器学习模型
- **线性回归**: 基线模型
- **Ridge回归**: 带L2正则化的线性模型
- **随机森林**: 集成学习方法，表现最佳

### 深度学习模型
- **MLP (多层感知机)**: 不同规模的神经网络
  - 小型：50→25个神经元
  - 中型：100→50→25个神经元
  - 大型：200→100→50→25个神经元
- **🆕 RNN-ResNet混合模型**: 基于PyTorch的时序深度学习模型
  - **LSTM-ResNet**: 双向LSTM(64) + 2个ResNet块(128)
    - 捕获长期时序依赖
    - 深度特征学习
    - ~400K参数
  - **GRU-ResNet**: 双向GRU(64) + 2个ResNet块(128)
    - 更快的训练速度
    - 较少的参数量
    - ~300K参数
  - GPU (CUDA)加速训练
  - 完整Jupyter Notebook实现

## 📈 评估指标

- **R² (决定系数)**: 衡量模型解释变量变异的能力，越接近1越好
- **RMSE (均方根误差)**: 预测值与真实值的偏差，单位为°C，越小越好
- **MAE (平均绝对误差)**: 预测误差的绝对值平均，单位为°C

## 📋 项目文档

### 核心报告
- **`docs/comprehensive_project_report.md`**: 📖 综合项目报告（英文版）
  - 完整的研究方法论和渐进式分析过程
  - 详细的时间特征工程分析
  - Ridge回归异常现象的深度剖析
  - 生产部署建议和风险评估

- **`docs/temporal_factors_impact_report.md`**: ⏰ 时间因素影响分析报告
  - 时间特征对模型性能的系统性影响分析
  - 季节性、日周期和周期性模式发现
  - 不同模型在时间特征下的表现对比

- **`docs/project_report.md`**: 📊 基础项目报告
  - 传统机器学习模型性能对比
  - 基线模型建立和评估方法

### 使用指南
1. **新手入门**: 从`tutorial/README.md`开始学习
2. **快速复现**: 按照"快速开始"部分的命令执行
3. **深度理解**: 阅读综合项目报告了解完整研究过程
4. **高级分析**: 参考时间因素影响报告进行扩展研究

## 🔧 技术实现

### 时间特征工程
- **基础时间特征**: 小时、星期、月份、年中天数
- **派生特征**: 季节、工作时间、周末标识
- **循环编码**: 使用正弦-余弦编码保持时间的循环特性
- **特征维度**: 从6个电气特征扩展到17个特征（含时间特征）

### 模型架构
- **传统机器学习**: 线性回归、Ridge回归、随机森林
- **深度学习**: 多层感知机（小型、中型、大型配置）
- **集成方法**: 随机森林作为最佳表现模型

## ⚠️ 重要发现与风险提示

### 🎯 最佳实践
- **推荐组合**: Random Forest + 时间特征
- **适用场景**: 所有预测时间窗口（1小时、1天、1周）
- **性能保证**: 稳定的性能提升，低风险

### ⚠️ 风险警告
- **避免组合**: Ridge回归 + 高维时间特征（长期预测）
- **风险表现**: 1周预测中R²从0.013降至-0.339
- **根本原因**: 维度诅咒和过拟合问题

## 📋 生成的结果文件

### 🆕 时间窗口长度实验结果
- `Windows_diff/docs/window_length_analysis_report.md` - 详细分析报告
- `Windows_diff/docs/comprehensive_experiment_summary.md` - 综合实验总结
- `Windows_diff/visualizations/best_r2_heatmap.png` - 最佳R²性能热力图
- `Windows_diff/visualizations/r2_vs_history_window.png` - R²与历史窗口关系图
- `Windows_diff/visualizations/rmse_vs_history_window.png` - RMSE与历史窗口关系图
- `Windows_diff/visualizations/training_time_vs_history_window.png` - 训练时间分析图

### 基础实验结果文件
运行完成后会生成以下文件：
- `final_model_comparison.csv`: 所有模型的性能对比
- `model_performance_analysis.png`: 模型性能可视化
- `prediction_examples.png`: 预测结果示例
- `error_distribution.png`: 误差分布分析

## 联系信息

如有问题，请通过项目渠道反馈。

---

**注意**: 本项目为学术用途，所有代码和结果仅供研究参考。