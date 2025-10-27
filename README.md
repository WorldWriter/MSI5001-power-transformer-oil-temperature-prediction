# Transformer Oil Temperature Forecasting (MSI5001-final)

## 📁 项目结构
```
MSI5001-final/
├── data/                     # 原始 CSV 数据（trans_1/trans_2）
├── processed/                # 清洗 & 标准化后的数据及参数
├── results/
│   ├── tables/               # 指标、预测结果、统计表
│   └── figures/              # 相关图、预测曲线、散点图
├── models/
│   ├── baseline/             # 时间顺序 80/20 模型
│   ├── random_split/         # 随机滑窗 80/20 模型
│   └── horizon_experiments/  # 1h/1d/1w 多时距模型（含 scaler）
├── scripts/                  # 核心脚本（详见下方）
└── docs/                     # 报告、流程说明与问题描述
```

## 🛠 核心脚本
- `scripts/model_training.py` ：时间顺序 80/20 划分（RandomForest/MLP）。  
- `scripts/model_random_split.py` ：滑动窗口 + 随机 80/20（Linear/Ridge/RandomForest/MLP，TX1 含动态特征）。  
- `scripts/model_horizon_experiments.py` ：按题目要求生成 1h / 1d / 1w 滑窗，设定 gap 约束，支持四类模型。  
- `scripts/common.py` ：特征工具函数，包含时间特征与 TX1 动态特征构造。

可参考以下命令复现：
```bash
# 时间顺序划分
python -m scripts.model_training

# 随机滑窗划分
python -m scripts.model_random_split

# 1h / 1d / 1w 多时距实验
python -m scripts.model_horizon_experiments
```

## 📊 模型与指标
所有指标输出在 `results/tables/` 中：
- `model_performance_all.csv`：时间顺序结果  
- `random_split_performance.csv`：随机滑窗结果  
- `horizon_experiment_metrics.csv`：1h/1d/1w 综合数据  
对应预测曲线与散点图在 `results/figures/`。

Summary Highlights：
| 模型/配置 | TX1 R² | TX2 R² |
|-----------|--------|--------|
| Chronological RF (15 分钟) | −4.61 | **0.63** |
| Random RF (15 分钟) | **0.94** | **0.97** |
| Horizon 1h (TX1 RF / TX2 MLP) | 0.94 | **0.97** |
| Horizon 1d (TX1 MLP / TX2 MLP) | **0.95** | **0.98** |
| Horizon 1w (TX1 RF / TX2 MLP) | **0.97** | **0.98** |

> 注：随机划分结果主要体现模型容量，实际部署需采用时间顺序或 rolling window 验证。

## 📝 文档
- `docs/project_pipeline_cn.md` ：中文完整流程与思路、参数逻辑、指标总结。  
- `docs/project_report_en.md` / `project_report_en.md` ：英文项目报告。  
- `docs/project_report.md` ：中文简版报告。  
- `docs/project_report_overleaf.tex` ：LaTeX 学术版。  
- `docs/project_report_en.md` ：英文版本。  
- `docs/problem.md` ：原始任务说明。

## 🚀 下一步建议
1. 在时间顺序评估下继续提升 TX1 表现（引入环境、运维信息；滚动验证）。  
2. 尝试 Gradient Boosting、LSTM/TCN 等时序模型。  
3. 扩展至多时间跨度的 rolling-window 交叉验证，评估稳定性。

如需更多细节，请参阅 `docs/project_pipeline_cn.md` 或相应脚本。안마
