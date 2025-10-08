# 电力变压器油温预测项目

## 项目简介

本项目使用机器学习方法预测电力变压器的油温变化，基于历史电气负载数据预测未来1小时、1天和1周后的油温。

## 项目结构

```
.
├── optimized_preprocessing.py    # 数据预处理脚本
├── simple_ml_models.py          # 传统机器学习模型
├── simple_deep_models.py        # 深度学习模型（MLP）
├── visualization_analysis.py    # 可视化分析
├── project_report.md           # 项目详细报告
├── final_model_comparison.csv   # 模型性能对比结果
└── README.md                   # 项目说明
```

## 环境要求

- Python 3.8+
- 主要依赖：
  - numpy
  - pandas
  - scikit-learn
  - matplotlib
  - seaborn

## 快速开始

### 1. 数据预处理
```bash
python optimized_preprocessing.py
```

### 2. 训练传统机器学习模型
```bash
python simple_ml_models.py
```

### 3. 训练深度学习模型
```bash
python simple_deep_models.py
```

### 4. 生成可视化分析
```bash
python visualization_analysis.py
```

### 5. 运行静态类型检查
```bash
mypy --ignore-missing-imports scripts/preprocessing/optimized_preprocessing.py \
    scripts/models/simple_ml_models.py scripts/models/simple_deep_models.py
```
> 使用 ``--ignore-missing-imports`` 可以在未安装第三方库类型存根的环境下完成基本一致性检查。

## 主要结果

### 最佳模型性能

| 预测时间 | 最佳模型 | R²分数 | RMSE |
|----------|----------|--------|------|
| 1小时    | Random Forest | 0.6033 | 4.6821 |
| 1天      | Random Forest | 0.4236 | 5.4321 |
| 1周      | Random Forest | 0.2740 | 5.8604 |

### 关键发现

1. **随机森林**在所有配置中表现最佳
2. **预测精度**随时间跨度增加而降低
3. **传统机器学习**模型优于简单的深度学习模型
4. **特征工程**对提升模型性能至关重要

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

## 评估指标

- **R² (决定系数)**: 衡量模型解释变量变异的能力
- **RMSE (均方根误差)**: 预测值与真实值的偏差
- **MAE (平均绝对误差)**: 预测误差的绝对值平均

## 使用说明

1. 确保已安装所有依赖包
2. 按顺序运行脚本（预处理→训练→分析）
3. 查看生成的CSV文件和PNG图表
4. 详细报告见 `project_report.md`

## 注意事项

- 由于数据量较大，部分模型使用了数据采样
- 深度学习模型使用了简化的架构
- 所有模型都经过了交叉验证

## 结果文件

运行完成后会生成以下文件：
- `final_model_comparison.csv`: 所有模型的性能对比
- `model_performance_analysis.png`: 模型性能可视化
- `prediction_examples.png`: 预测结果示例
- `error_distribution.png`: 误差分布分析

## 联系信息

如有问题，请通过项目渠道反馈。

---

**注意**: 本项目为学术用途，所有代码和结果仅供研究参考。