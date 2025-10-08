# 电力变压器油温预测项目 - 目录概览

```
MSI5001-power-transformer-oil-temperature-prediction/
├── README.md                     # 使用说明与运行命令
├── PROJECT_STRUCTURE.md          # 本文件
├── docs/                         # 文档资料
│   ├── README.md
│   ├── project_report.md
│   └── review_report.md
├── scripts/
│   ├── __init__.py               # 允许作为包导入
│   ├── config.py                 # 实验窗口及默认路径配置
│   ├── preprocessing/
│   │   └── optimized_preprocessing.py  # 时间序列预处理入口
│   └── models/
│       ├── simple_ml_models.py           # 传统机器学习模型训练
│       └── simple_deep_models.py         # 神经网络基线训练
├── models/                       # 历史模型产物占位目录
├── data/                         # 建议存放原始 CSV 的目录（需用户自建）
└── artifacts/                    # 预处理与训练产物的默认输出目录（运行时生成）
```

## 核心流程
1. **预处理**：`scripts/preprocessing/optimized_preprocessing.py`
   - 读取 `data/` 下的 `trans_*.csv`
   - 生成 `X_*_<config>.npy`、`y_*_<config>.npy`
   - 保存 `scaler_<config>.pkl` 与 `metadata_<config>.json`
2. **传统模型**：`scripts/models/simple_ml_models.py`
   - 线性回归、Ridge、随机森林
   - 可选时间序列交叉验证网格搜索
   - 输出模型权重与 `simple_ml_results.csv`
3. **神经网络基线**：`scripts/models/simple_deep_models.py`
   - 基于 `MLPRegressor` 的多层感知机
   - 可选网格搜索和综合结果输出

## 运行产物
- `artifacts/X_train_<config>.npy` 等数组文件：三维时间序列数据。
- `artifacts/scaler_<config>.pkl`：仅以训练集拟合的标准化器。
- `artifacts/simple_ml_results.csv`、`artifacts/simple_deep_results.csv`：评估指标。
- `artifacts/simple_ml_best_params.json`、`artifacts/simple_deep_best_params.json`：最佳超参数记录。
- 若指定 `--combined-output`，会额外生成 `artifacts/final_model_comparison.csv`。

## 注意事项
- `data/` 与 `artifacts/` 目录默认被忽略，不会随仓库分发。
- 若调整特征列或新增实验配置，请同步更新 `scripts/config.py` 或在命令行传入新参数。
- 训练脚本依赖预处理输出，运行顺序应为：预处理 → 传统模型 → 神经网络模型。
