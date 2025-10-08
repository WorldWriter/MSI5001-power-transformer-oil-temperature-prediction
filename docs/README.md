# 电力变压器油温预测项目

本目录下的文档补充了仓库中各模块的设计说明。若需快速运行项目，请参阅仓库根目录的 `README.md`。以下仅概述关键文件：

- `project_report.md`：详细的项目背景、方法和实验总结。
- `review_report.md`：对原始实现的代码与文档差异进行审查。

运行流程概览：

1. **数据预处理** – `python scripts/preprocessing/optimized_preprocessing.py`
2. **传统模型** – `python scripts/models/simple_ml_models.py`
3. **神经网络模型** – `python scripts/models/simple_deep_models.py`

上述脚本均支持通过命令行参数调整输入目录、输出目录和参与的实验配置。更多细节与参数说明见根目录 `README.md`。
