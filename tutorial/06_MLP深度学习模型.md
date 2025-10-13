# 06｜MLP 深度学习模型

本章使用 `scripts/models/simple_deep_models.py` 训练三档 MLP（多层感知机）模型，并合并到最终结果表中。另附 `deep_learning_models.py`（含 LSTM 与 Dense NN）供进阶探索。

## 模型档位
- 小型：`(50, 25)`
- 中型：`(100, 50, 25)`
- 大型：`(200, 100, 50, 25)`

所有模型使用 `relu` 激活与 `adam` 优化器，最大迭代 `200`，训练前会对子采样加速。

## 运行训练与合并结果
```bash
python scripts/models/simple_deep_models.py
```

该脚本会：
- 按 `1h/1d/1w` 训练 MLP，并保存 `mlp_small/medium/large_*.pkl`。
- 读取 `simple_ml_results.csv`，与 MLP 结果合并写入 `final_model_comparison.csv`。
- 控制台打印综合 Top-N 结果与类型平均 `R²`，便于课堂讨论。

## 进阶：LSTM 与 Dense NN（可选）
`scripts/models/deep_learning_models.py` 展示了 LSTM 序列模型与更深的全连接网络：
- 需将 `X` 重塑为 `[n_samples, n_timesteps, n_features]`（已在脚本中提供辅助函数）。
- 配合早停与自适应学习率调度，便于稳定训练。

## 练习
- 对比 MLP 三档结构在不同预测跨度下的表现差异，记录参数与结果。
- 运行 LSTM 与 Dense NN，观察它们与随机森林的相对强弱，并思考数据量与时序结构对结果的影响。

完成本章后，你将掌握在本任务上的 MLP 训练流程，并具备与传统 ML 的系统性对比能力。