# 问题概述 / Problem Overview

这是一个电力变压器逐小时数据集。我们为你提供了2个变压器的数据。

This is an electric transformer hour-by-hour dataset. We provide you with the data of 2 transformers.

# 数据集 / Dataset

数据集包含每个时间点的6个特征，如下所示：

The dataset contains 6 features per time point which is as follows:

- **HUFL** = 高压有用负载（高压侧有功功率）/ High Voltage Useful Load (active power on high voltage side)
- **HULL** = 高压无用负载（高压侧无功功率）/ High Voltage Useless Load (reactive power on high voltage side)
- **MUFL** = 中压有用负载（中压侧有功功率）/ Medium Voltage Useful Load (active power on medium voltage side)
- **MULL** = 中压无用负载（中压侧无功功率）/ Medium Voltage Useless Load (reactive power on medium voltage side)
- **LUFL** = 低压有用负载（低压侧有功功率）/ Low Voltage Useful Load (active power on low voltage side)
- **LULL** = 低压无用负载（低压侧无功功率）/ Low Voltage Useless Load (reactive power on low voltage side)
- **OT** = 油温（需要预测的目标变量）/ Oil Temperature (the target variable to predict)

# 任务描述 / Task Description

你需要基于之前时间点的数据来预测某个时间点的油温，且不能使用任何时间点的油温信息。

You need to predict the oil temperature of a time point based on data from previous time points without ever using oil temperature information from any time point.

我们希望尝试三种基于历史数据使用的配置：

We want you to try three configurations based on the usage of data from previous time points:

1. **1小时预测 / 1-hour prediction**: 使用过去N个时间点（可自行调整）的数据，从待预测时间点的前4个时间点开始 / use past N time point data starting from the 4 time points prior to the one to be forecasted

2. **1天预测 / 1-day prediction**: 使用过去N个时间点（可自行调整）的数据，从待预测时间点的前96个时间点开始 / use past N time point data starting from the 96 time points prior to the one to be forecasted for

3. **1周预测 / 1-week prediction**: 使用过去N个时间点（可自行调整）的数据，从待预测时间点的前672个时间点开始 / use past N time point data starting from the 672 time points prior to the one to be forecasted for

## 重要限制 / Important Restrictions

- 在上述任何配置中，都不能使用待预测油温时间点的任何负载信息 / You cannot use any load information from the time point for which you are predicting the oil temperature for in any of the above configurations
- 不能使用任何时间点的油温数据 / You cannot use oil temperature from any time point

# 训练-测试数据划分 / Train-Test Data

训练数据应该与测试数据完全分离。如果你在训练样本中使用了时间点1-50,000，那么测试样本不能包含这些时间点。

Your training data should be completely disjoint from your test data. If you are using time point 1-50,000 in any of your training samples, then the test samples cannot belong to any of those time points.

我的建议是将数据分成时间点组，随机将80%的组分配给训练，20%分配给测试。考虑进行一些均值归一化。

My recommendation is to divide the data into time point groups and randomly assign 80% of the groups to training and 20% to testing. Consider some mean normalization.

在训练数据集上执行交叉验证来开发和选择模型；测试集应该仅用于测试最终模型。

Perform cross validation on the training dataset to develop and choose your model; the test set should be kept only for testing the final model.

# 期望成果 / Expectations

- 适当的时间序列特征可视化 / Proper time-series feature visualization
- 通过相关性分析进行特征选择尝试 / Feature selection attempts through correlation analysis
- 你需要探索基于深度学习的时间序列建模和传统机器学习技术 / You are expected to explore both deep learning based time series modeling and traditional ML techniques