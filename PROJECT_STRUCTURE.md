# 电力变压器油温预测项目 - 文件结构树

```
msi5001-project/
├── data/                           # 数据文件夹
│   ├── raw/                       # 原始数据
│   │   ├── trans_1.csv           # 变压器1原始数据
│   │   └── trans_2.csv           # 变压器2原始数据
│   └── processed/                 # 处理后的数据
│       ├── X_train_1h.npy        # 1小时预测训练特征
│       ├── X_train_1d.npy        # 1天预测训练特征
│       ├── X_train_1w.npy        # 1周预测训练特征
│       ├── X_test_1h.npy         # 1小时预测测试特征
│       ├── X_test_1d.npy         # 1天预测测试特征
│       ├── X_test_1w.npy         # 1周预测测试特征
│       ├── y_train_1h.npy        # 1小时预测训练目标
│       ├── y_train_1d.npy        # 1天预测训练目标
│       ├── y_train_1w.npy        # 1周预测训练目标
│       ├── y_test_1h.npy         # 1小时预测测试目标
│       ├── y_test_1d.npy         # 1天预测测试目标
│       ├── y_test_1w.npy         # 1周预测测试目标
│       └── scaler_*.pkl          # 数据标准化器（各配置）
│
├── scripts/                       # 脚本文件夹
│   ├── preprocessing/             # 预处理脚本
│   │   ├── data_preprocessing.py # 原始预处理脚本
│   │   └── optimized_preprocessing.py # 优化版预处理
│   ├── models/                    # 模型训练脚本
│   │   ├── traditional_ml_models.py    # 传统ML模型（完整版）
│   │   ├── simple_ml_models.py        # 简化ML模型
│   │   ├── deep_learning_models.py    # 深度学习模型（TensorFlow版）
│   │   └── simple_deep_models.py      # 简化深度学习模型（MLP版）
│   ├── evaluation/                # 评估分析脚本
│   │   └── visualization_analysis.py   # 可视化分析
│   └── exploratory/               # 探索性分析
│       └── transformer_analysis.py     # 初始数据分析
│
├── models/                        # 训练好的模型
│   ├── linear_regression_1h.pkl  # 线性回归模型（1小时）
│   ├── ridge_regression_1h.pkl   # Ridge回归模型（1小时）
│   ├── rf_1h.pkl                 # 随机森林模型（1小时）
│   ├── rf_1d.pkl                 # 随机森林模型（1天）
│   ├── rf_1w.pkl                 # 随机森林模型（1周）
│   ├── mlp_small_*.pkl           # 小型MLP模型（各配置）
│   ├── mlp_medium_*.pkl          # 中型MLP模型（各配置）
│   ├── mlp_large_*.pkl           # 大型MLP模型（各配置）
│   └── scaler_*.pkl              # 数据标准化器（各配置）
│
├── results/                       # 结果文件夹
│   └── csv/                      # CSV结果文件
│       ├── final_model_comparison.csv   # 最终模型对比结果
│       ├── simple_ml_results.csv        # 传统ML模型结果
│       └── mlp_results.csv              # MLP模型结果
│
├── visualization/                 # 可视化文件夹
│   └── plots/                    # 图表文件
│       ├── model_performance_analysis.png  # 模型性能对比图
│       ├── prediction_examples.png         # 预测示例图
│       ├── error_distribution.png          # 误差分布图
│       └── Transformer *_features_timeseries.png # 原始时间序列图
│
└── docs/                          # 文档文件夹
    ├── problem.md                # 原始问题描述
    ├── grading.md                # 评分标准
    ├── project_report.md         # 项目技术报告
    └── README.md                 # 项目说明文档

```

## 文件说明

### 核心执行流程
1. **数据预处理**: `scripts/preprocessing/optimized_preprocessing.py`
2. **模型训练**:
   - 传统ML: `scripts/models/simple_ml_models.py`
   - 深度学习: `scripts/models/simple_deep_models.py`
3. **可视化分析**: `scripts/evaluation/visualization_analysis.py`

### 主要结果文件
- **最终对比**: `results/csv/final_model_comparison.csv`
- **性能图表**: `visualization/plots/model_performance_analysis.png`
- **技术报告**: `docs/project_report.md`

### 最佳模型
- **1小时预测**: `models/rf_1h.pkl` (R² = 0.6033)
- **1天预测**: `models/rf_1d.pkl` (R² = 0.4236)
- **1周预测**: `models/rf_1w.pkl` (R² = 0.2740)

## 使用说明

1. 按顺序执行脚本获取完整结果
2. 查看`docs/project_report.md`获取详细技术分析
3. 查看`results/csv/final_model_comparison.csv`获取模型性能对比
4. 查看`visualization/plots/`目录下的PNG文件获取可视化结果

---

*文件结构生成时间: $(date)*