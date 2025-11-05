# 电力变压器油温预测整体流程说明（MSI5001-final）

本文完整记录了本项目在 `MSI5001-final/` 目录下的处理流程、参数选择逻辑、模型训练与评估结果，供后续对照与复现。

---

## 1. 数据理解与基础分析
### 1.1 数据概况
- 原始数据：`data/trans_1.csv`、`data/trans_2.csv`，采样间隔 15 分钟，时间范围 2018-07-01 至 2020-06-26。
- 质量检查（`results/tables/missing_values_summary.csv`、`data_types.csv`）显示无缺失值、字段类型正确。

### 1.2 变压器差异
- `results/tables/transformer_summary.csv`、`results/figures/ot_trend_by_transformer.png` 显示 TX1 与 TX2 的油温均值、波动范围差异显著，故后续步骤分开建模。

### 1.3 时间与动态特征
在 `scripts/common.py` 的 `add_time_features` 中，为每条记录添加以下特征：
- `hour`, `dayofweek`, `month`, `day_of_year`；
- 对上述变量做 `sin/cos` 编码，例如 `hour_sin/hour_cos`；
- `is_weekend`, `is_worktime`, `season`。

这些时间因素是整个流程的亮点，能够描述日内与季节节律。  
此外，针对 TX1 又构造了差分与 12 步滚动均值特征 (`HULL_diff1`, `MULL_diff1`, `HULL_roll12`, `MULL_roll12`)，以捕捉其滞后惯性。

---

## 2. 数据预处理
### 2.1 清洗策略
- 对 HULL、MULL、油温 OT 使用 IQR 与 6 小时滚动 Z-score；只要任一指标越界（或 OT 超出 [−20, 120] °C），即剔除该样本。
- 低压负载 (LUFL/LULL) 的负值可能代表反向潮流/无功补偿，保留处理。
- 清洗后样本数：`processed/tx1_cleaned.csv` → 65,562 条，`processed/tx2_cleaned.csv` → 66,919 条；统计记录在 `results/tables/outlier_detection_summary.csv`。

### 2.2 标准化
- 为避免信息泄露，对清洗后的数据进行扩展窗口 z-score 标准化（`processed/tx{1,2}_standardized.csv`），参数写入 `processed/standardization_params.json`。

---

## 3. 基础建模流程
### 3.1 时间顺序 80/20 划分
- 脚本：`scripts/model_training.py`。
- 步骤：训练区间为前 80% 时间，测试为后 20%；模型包含随机森林（120 棵树、max_depth=12、min_samples_leaf=5）和 MLP（128-64 单元）。
- 输出：模型存放 `models/baseline/`，指标表 `results/tables/model_performance_all.csv`，预测曲线 `results/figures/tx{1,2}_{Model}[_std]_prediction.png`。
- 结果摘要：
  - TX1：随机森林/MLP R² 为负（约 −4.7），说明简单特征难以解释测试段漂移；
  - TX2：随机森林 R²≈0.63，表现良好。

### 3.2 随机滑窗 80/20 划分
- 脚本：`scripts/model_random_split.py`。
- 步骤：根据 15 分钟数据构建滑动窗口（TX1 lookback 48、TX2 lookback 24），最多抽样 30k 个窗口后随机 80/20 划分；模型涵盖线性、Ridge、随机森林、MLP（TX1 加动态特征）。
- 输出：`models/random_split/`，`results/tables/random_split_performance.csv`，散点对比 `results/figures/random_tx{1,2}_{Model}_scatter.png`。
- 结果摘要：TX1/RandomForest R²≈0.94，TX2/RandomForest R²≈0.97，说明在随机划分下特征与模型容量充足；线性模型表现差距大。

---

## 4. 三种预测任务（1h / 1d / 1w）
依据 `docs/problem.md` 的要求，在 `scripts/model_horizon_experiments.py` 中统一采用**小时级**数据（`resample('1H')`），并按照业务约束设置 lookback/gap/horizon：

| 配置 | lookback (小时) | gap (小时)* | horizon (小时) | 说明 |
|------|-----------------|-------------|----------------|------|
| 1h   | 24              | 0           | 1              | 使用最近 24h 预测下一小时 |
| 1d   | 48              | 6           | 24             | 离目标最近的 6h 数据不使用 |
| 1w   | 168             | 24          | 168            | 离目标最近 24h 数据不使用 |

\* gap 的设置源自题目要求：“起始于预测点前第 6 个/第 24 个时间点”，可理解为预测时与目标时刻之间应留出缓冲区，避免模型“偷看”过近的负载信息。

### 4.1 实验流程
1. 加载并小时聚合 → 添加时间特征 → 若为 TX1，补充差分/滚动特征。  
2. 按上述配置构造滑动窗口 `(lookback, gap, horizon)`；生成后随机 80/20 划分。  
3. 训练集上拟合 `StandardScaler`，四种模型（LinearRegression、Ridge、RandomForest、MLP）共享该 scaler；测试集
   使用相同 scaler。  
4. 模型 artefact 保存至 `models/horizon_experiments/`（内含模型、scaler、特征列表）；指标 `results/tables/horizon_experiment_metrics.csv`，预测曲线 `results/figures/horizon_tx{1,2}_{config}_{Model}.png`。

### 4.2 指标结果（随机滑窗，小时粒度）
| 变压器 | 配置 | 最佳模型 | R² | RMSE (°C) | MAE (°C) |
|--------|------|----------|----|-----------|----------|
| TX1 | 1h | RandomForest | 0.94 | 1.80 | 1.23 |
| TX1 | 1d | MLP          | 0.95 | 1.73 | 1.28 |
| TX1 | 1w | RandomForest | 0.97 | 1.32 | 0.93 |
| TX2 | 1h | MLP          | 0.97 | 2.00 | 1.43 |
| TX2 | 1d | MLP          | 0.98 | 1.54 | 1.11 |
| TX2 | 1w | MLP          | 0.98 | 1.69 | 1.24 |

线性/Ridge 在所有配置下均显著落后，验证了非线性模型在此任务上的必要性。上述指标属于“随机划分下的容量评估”，要验证部署可靠性尚需对应的时间顺序或 rolling window 流程。

---

## 5. 核心逻辑说明
1. **时间特征**：通过 hour/day/month/day-of-year 及其正余弦编码，模型能捕捉日内、季节等周期性，是本项目的亮点。  
2. **lookback 选择**：结合滞后相关分析和业务经验确定窗口长度（TX1 12h、TX2 6h），以适应不同变压器的热惯性。  
3. **gap 设定**：来源于题目要求，在预测 1d/1w 时需留出缓冲区，以模拟“提前预警”场景，防止使用目标时间附近的负载数据而产生信息泄露。  
4. **随机 vs. 时间顺序**：随机划分用于衡量模型容量与特征有效性；时间顺序划分更贴近部署情境。滚动验证可作为未来工作，兼顾数据利用率与评估真实度。  
5. **模型选择**：实验表明线性模型难以胜任，需要 RandomForest/MLP 等非线性方法；TX1 若添加动态特征，可显著提升准确度。

---

## 6. 主要目录与文件
```
MSI5001-final/
├── data/                          # 原始 CSV
├── processed/                     # 清洗、标准化后的数据与参数
├── results/
│   ├── tables/
│   │   ├── model_performance_all.csv
│   │   ├── random_split_performance.csv
│   │   ├── horizon_experiment_metrics.csv
│   │   └── ...（相关性、缺失率、预测结果等）
│   └── figures/
│       ├── ot_trend_by_transformer.png
│       ├── random_tx{1,2}_{Model}_scatter.png
│       ├── horizon_tx{1,2}_{config}_{Model}.png
│       └── ...（特征相关图等）
├── models/
│   ├── baseline/                  # 时间顺序模型
│   ├── random_split/              # 随机滑窗模型
│   └── horizon_experiments/       # 多时间跨度模型+scaler
└── scripts/
    ├── model_training.py
    ├── model_random_split.py
    ├── model_horizon_experiments.py
    └── 共用工具在 common.py
```

---

## 7. 结论与未来工作
1. 经过清洗与特征扩展后，HULL/MULL + 时间特征（含周期编码）是性能最稳定的组合；TX1 需额外的差分与滚动均值以刻画惯性。  
2. 时间顺序划分——尤其对 TX2——已具有一定部署可能；随机滑窗和多时距实验说明模型容量与特征设计可满足不同预测任务。  
3. 后续可考虑引入环境温度、运维日志，采用 Gradient Boosting、LSTM/TCN 等序列模型，并使用 rolling window 做综合评估。

本流程已覆盖从数据理解、特征、清洗、模型对比到三种预测任务的完整链路，可作为后续开发与部署的参照。
