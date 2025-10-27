# Transformer Oil Temperature Data Assessment (MSI5001-final)

## 1. 数据理解与探索性分析

### 1.1 数据质量检查
- **缺失值**: `results/tables/missing_values_summary.csv` 显示所有字段缺失率为 0%。无需填补。
- **数据类型**: 见 `results/tables/data_types.csv`，六个负荷特征与油温已强制转换为数值型，后续可直接做数值运算。
- **变压器差异**:  
  - 两台设备样本量 69,680 条，时间跨度 2018-07-01 至 2020-06-26（`results/tables/transformer_summary.csv`）。  
  - 油温均值/方差差异显著：TX1 平均 13.3 °C、标准差 8.6 °C；TX2 平均 26.6 °C、标准差 11.9 °C。  
  - `results/figures/ot_trend_by_transformer.png` 明确展示了两台设备的温度运行区间不同 → **必须分开建模和标准化**。

### 1.2 统计相关性
- **同瞬时相关性（皮尔森）**  
  - TX1: HULL 与 MULL 对油温的相关性最高，仅 ~0.22，其余特征相关性较弱（`results/tables/tx1_correlation_matrix.csv`、`results/figures/tx1_correlation_heatmap.png`）。  
  - TX2: MULL (~0.50)、HULL (~0.34) 对油温影响显著，而 LUFL/LULL 与油温呈负相关（`results/tables/tx2_correlation_matrix.csv`）。
- **滞后相关性（0–24h）**  
  - TX1 在 10–12 小时滞后时 HULL/MULL 与油温相关性提升至 ~0.26（`results/tables/tx1_lag_correlation.csv`、`results/figures/tx1_lag_correlation_heatmap.png`），提示需要更长历史窗口。  
  - TX2 的 MULL/HULL 在零滞后即达到最大相关，表明短期负载即可解释油温变化（`results/tables/tx2_lag_correlation.csv`）。

**结论**: 模型应重点关注 HULL、MULL，并针对 TX1 引入较长的历史窗口与动态特征（差分、滚动均值）。

## 2. 数据预处理

### 2.1 异常值检测与处理
- 检测策略：IQR（全局）、滑动窗口 Z-score（24×15min=6 小时）以及物理约束 (OT ∈ [-20, 120] °C；仅针对高压/中压负载做非负约束)。  
- 结果摘要见 `results/tables/outlier_detection_summary.csv`：  
  - 只要在 HULL、MULL 或 OT 上被 IQR **或** 滑动 Z-score 标记，即视为异常；低压负载允许为负值。  
  - 共有 4,118（TX1）与 2,761（TX2）条样本被剔除，清洗后仍保有 65,562 与 66,919 条有效记录。  
- 清洗数据输出 `processed/tx1_cleaned.csv`、`processed/tx2_cleaned.csv`，对应标准化结果为 `processed/tx1_standardized.csv`、`processed/tx2_standardized.csv`。  
- **建议**: 如需进一步减少损失，可针对业务已知的维护窗口单独处理或插值。

### 2.2 标准化策略
- 采用逐步扩展均值/方差（expanding z-score）：保证每个时间点只依赖其之前的统计量，避免未来信息泄露。  
- 结果写入 `processed/tx1_standardized.csv`、`processed/tx2_standardized.csv`，参数记录在 `processed/standardization_params.json`。

## 3. 关键发现与推荐自变量

1. **核心负荷特征**  
   - HULL（高压无功）、MULL（中压无功）：两个变压器中与油温最相关，需必选。  
   - HUFL、MUFL：在滞后相关中对 TX1 有一定贡献，建议保留。  
   - LUFL/LULL：对 TX2 体现负相关，保留以捕捉反向关系。
2. **时间特征**  
   - 小时、星期、月份及其正弦/余弦编码：体现季节与日常模式。  
   - 工作日/工作时间标记：区分运行模式。
3. **动态特征（特别是 TX1）**  
   - 一阶差分 (`ΔHUFL` 等) 与 4/12/24 步滚动均值，可增强对速度/惯性的刻画。  
   - 组合特征（如 HULL/MULL 比值）可在下一阶段探索。

## 4. 下一步数据处理策略

1. **异常值再评估**：结合运维日志区分真实停机 vs. 传感器故障；对必要的异常可使用插值或局部回归恢复，以防样本大幅减少。  
2. **窗口配置实验**：基于滞后分析，TX1 建议测试 ≥ 48 步历史窗口；TX2 可保持 24 步。  
3. **标准化流程固化**：后续建模需在训练集上拟合 expanding 统计量，并将参数版本化保存，以便线上部署。  
4. **变量清单**：  
   - 基础：`HUFL, HULL, MUFL, MULL, LUFL, LULL, OT` (目标)  
   - 时间：`hour, dayofweek, month, day_of_year, hour_sin, hour_cos, dow_sin, dow_cos, month_sin, month_cos, season, is_weekend, is_worktime`  
   - 动态（建议新增）：`ΔHUFL…ΔLULL`, `rolling_mean_{4,12,24}` 等。

如需进一步验证，可在此基础上构建 per-transformer 预测模型，并对比是否改善 TX1 的表现。
