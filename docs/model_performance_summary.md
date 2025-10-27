# 80/20 Chronological Split – Baseline Modeling Summary

## 数据设置
- 原始数据：`processed/tx{1,2}_cleaned.csv`（已清除异常但保持原量纲）与 `processed/tx{1,2}_standardized.csv`（逐步扩展 z-score）。  
- 特征：`HULL`, `MULL` + 时间特征（小时/星期/月、年内日、正弦余弦编码、工作日/工作时、季节）。  
- 切分方式：按时间顺序 80% 训练 / 20% 测试（独立 per transformer）。

## 模型
- **Random Forest**（120棵树，max_depth=12，min_samples_leaf=5）  
- **MLP**（128-64 神经元，ReLU, Adam, lr=1e-3, 早停）

训练脚本：`python -m scripts.model_training [--standardized]`

## 测试集指标

| Transformer | 单位  | RandomForest | MLP |
|-------------|-------|--------------|-----|
| **TX1 (raw)** | RMSE | 8.03 °C | 8.13 °C |
|             | MAE   | 7.03 °C | 7.08 °C |
|             | R²    | -4.61  | -4.75  |
| **TX1 (std)** | RMSE | 0.69 σ | 0.62 σ |
|             | MAE   | 0.57 σ | 0.51 σ |
|             | R²    | -1.31  | -0.87  |
| **TX2 (raw)** | RMSE | 6.45 °C | 6.77 °C |
|             | MAE   | 4.97 °C | 5.41 °C |
|             | R²    | **0.63** | 0.59 |
| **TX2 (std)** | RMSE | 0.51 σ | 0.53 σ |
|             | MAE   | 0.40 σ | 0.43 σ |
|             | R²    | **0.66** | 0.63 |

- 完整指标表：`results/tables/model_performance_all.csv`  
- 预测详情：`results/tables/tx{1,2}_{Model}_predictions[_std].csv`

## 观察
1. TX2 在仅用 HULL/MULL + 时间特征时，RandomForest R²≈0.63（raw）/0.66（std），可成为基线。  
2. TX1 的 R² 炫目为负，表明这些特征难以解释其测试区间；标准化后仍然无法扭转 → 需加入滞后/动态特征或重新划分窗口。  
3. MLP 表现略弱于 RandomForest，但趋势一致。  
4. 测试预测曲线见 `results/figures/tx{1,2}_{Model}_prediction[_std].png`，可视化对比模型偏差。

## 下一步建议
- 为 TX1 引入滞后窗口 ≥48 步、差分/滚动均值等动态特征；  
- 尝试季节性或局部模型（按季节/负载区间训练）；  
- 可基于 standardized 数据继续做超参调优或替换模型（如 Gradient Boosting、LSTM）。
