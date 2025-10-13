# 项目实验分析报告（初学者版）

本报告面向初学者，图文并茂地解释这次油温预测实验的思考过程、模型对比结果与原因分析，帮助快速理解结论与后续改进方向。

---

## 一、任务与数据
- 目标：预测变压器油温 `OT` 在三种时间跨度的未来值：`1小时 (1h)`、`1天 (1d)`、`1周 (1w)`。
- 输入特征：六个负载相关特征 `HUFL, HULL, MUFL, MULL, LUFL, LULL`；`OT` 仅用于可视化与评估，不作为输入特征。
- 直观观察：
  - 特征分布存在偏态与量纲差异（见直方图）。
  - 油温呈现明显的日内周期与慢变趋势（见小时平均曲线）。
  - 相关性热图显示与 `OT` 的线性相关有限，提示可能存在非线性关系与多变量交互。

---

## 二、方法与评估
- 对比模型：
  - 传统 ML：`Linear Regression`、`Ridge Regression`、`Random Forest`。
  - 简化深度学习：`MLP`（小/中/大规模）。
- 数据切分与管线：统一的数据预处理、序列构建、训练/评估流程，保证公平对比。
- 评估指标：
  - `R2`（拟合度，越高越好）
  - `RMSE`（均方根误差，越低越好）
  - `MAE`（平均绝对误差，辅助衡量稳健性）

---

## 三、关键可视化（直观理解）
> 以下为本次最新运行生成的图表，可直接点击或在文件管理器中打开。

- 模型整体表现对比图：
  - 路径：`../artifacts/run_20251008_200030/model_performance_analysis.png`
  - 包含 R² 热力图、RMSE 热力图、分组柱状对比、复杂度-性能散点，一图看清各模型在三种跨度的整体表现。

- 预测示例散点图（随机森林 1h/1d）：
  - 路径：`../artifacts/run_20251008_200030/prediction_examples.png`
  - 横轴为实际油温、纵轴为预测油温，点越贴近对角线预测越好；图中附有 `R²` 与 `RMSE` 文本标注。

- 误差分布与正态性检验（随机森林 1h）：
  - 路径：`../artifacts/run_20251008_200030/error_distribution.png`
  - 左图为误差直方图（均值接近 0，分布较集中），右图为 Q-Q 图（整体接近直线，轻尾现象可见）。

- 数据探索（选择性查看）：
  - 特征直方图：`../artifacts/run_20251008_200030/Transformer 1_feature_histograms.png`、`../artifacts/run_20251008_200030/Transformer 2_feature_histograms.png`
  - 油温小时平均曲线：`../artifacts/run_20251008_200030/Transformer 1_hourly_profile_ot.png`、`../artifacts/run_20251008_200030/Transformer 2_hourly_profile_ot.png`
  - 相关性热图：`../artifacts/run_20251008_200030/Transformer 1_correlation_matrix.png`、`../artifacts/run_20251008_200030/Transformer 2_correlation_matrix.png`

---

## 四、结果对比（核心结论）
来自最新结果 `final_model_comparison.csv`（同目录 `../artifacts/run_20251008_200030/`）：

- 1小时预测（1h）：
  - `Random Forest`：`R2 = 0.596`，`RMSE = 4.723`
  - `Ridge`：`R2 = 0.475`，`RMSE = 5.384`
  - `Linear`：`R2 = 0.462`，`RMSE = 5.453`

- 1天预测（1d）：
  - `Random Forest`：`R2 = 0.420`，`RMSE = 5.448`
  - `Ridge`：`R2 = 0.176`，`RMSE = 6.495`
  - `Linear`：`R2 = 0.117`，`RMSE = 6.722`

- 1周预测（1w）：
  - `Random Forest`：`R2 = 0.252`，`RMSE = 5.947`
  - `Ridge`：`R2 = -0.153`，`RMSE = 7.386`
  - `Linear`：`R2 = -0.558`，`RMSE = 8.586`

结论：在三种时间跨度下，`Random Forest` 的 `R2` 更高、`RMSE` 更低，综合表现最优。

---

## 五、为什么随机森林更适合本任务（教学解释）
- 非线性与交互：负载与油温的关系并非线性，且存在多变量交互；随机森林通过多棵树的集成能灵活拟合复杂关系。
- 鲁棒与易用：对特征缩放不敏感（直方图显示量纲差异）、对噪声更稳健；在中小数据与有限调参下通常更稳定。
- 贴近数据规律：油温具有周期性与慢变趋势，线性模型难以全面捕捉；随机森林在图表中表现为更高的 `R2` 与更低的误差。

---

## 六、误差分析（从图表读结论）
- 误差集中度：`error_distribution.png` 的直方图显示误差以 0 为中心、分布较集中，说明整体偏差小。
- 正态性近似：Q-Q 图整体接近直线，轻尾偏离提示个别场景存在较大误差（例如极端负载或温度变化）。
- 教学提示：可检查误差随时间或负载水平的变化是否有系统性偏移，针对性增加滞后/滚动统计或分段建模。

---

## 七、局限与改进建议
- 局限：
  - 长跨度预测更难（`1w` 的 `R2≈0.25`），现有特征与样本规模有限。
  - MLP 未进行系统化调参，可能欠优化；未引入外生影响（如环境温度、季节）。
- 改进：
  - 特征工程：加入滞后、滚动统计、周期/节假日标识；基于 CCF 选择不同跨度的有效滞后。
  - 模型优化：尝试 `XGBoost/LightGBM/CatBoost`；扩大 `n_estimators`、优化 `max_depth/min_samples_split`；对 MLP 做学习率/正则/早停等系统调参。
  - 评估策略：采用时间序列交叉验证（`blocked/expanding window`），更贴近真实部署而非随机划分。

---

## 八、复现与查看
- 一键生成最新结果与图表：
  - `python scripts/run_pipeline.py`
- 单独生成评估图表：
  - `python -c "from scripts.evaluation import visualization_analysis as viz; viz.main()"`
- 查看文件：
  - 指标表：`../artifacts/run_YYYYMMDD_HHMMSS/final_model_comparison.csv`
  - 图表：`../artifacts/run_YYYYMMDD_HHMMSS/model_performance_analysis.png`、`prediction_examples.png`、`error_distribution.png`

---

## 九、总结
- 当前设定下，`Random Forest` 在 `1h/1d/1w` 三个预测任务上综合表现最佳，原因在于其对非线性与交互的良好拟合能力与稳健性。
- 通过进一步的特征工程、时间序列化的交叉验证与更强的树模型或序列模型（如 LSTM/GRU），整体性能预计可继续提升，尤其是长跨度预测。