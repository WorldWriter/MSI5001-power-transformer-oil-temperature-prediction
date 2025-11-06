我们要进行一个参数对比实验：

1. 对比异常数据异常值剔除， 对于结果的影响。 无 / 0.5% / 1% / 5%； 
2. 测试集和训练集划分方式的影响
    - **滑动窗口随机**：当前的滑动窗口完全随机
    - 随机打乱所有样本，按80/20分割
    - 优点：充分利用数据
    - 缺点：存在数据泄露（滑动窗口重叠）
    - 注意：R²可能虚高！

    - **分组随机**：分组随机
    - 将数据分为20组（每组约3,484个样本）
    - 随机选择16组（80%）训练，4组（20%）测试
    - 优点：测试集覆盖整个时间范围
    - 缺点：可能存在轻微数据泄露（组内连续）

    - **时序分割**：时序分割（前80%训练，后20%测试）
    - 优点：无数据泄露，最接近真实预测场景
    - 缺点：测试集仅覆盖时间末段

---

## 目标12：优化Informer在长期预测（horizon=168）的性能

### 问题分析
从实验91-96的结果看：
- 短期预测（horizon=1）：R² ≈ 0.97 ✅
- 中期预测（horizon=24）：R² ≈ 0.61 ⚠️
- **长期预测（horizon=168）：R² < 0 ❌** （比均值预测还差）

原因可能包括：
1. 历史窗口太短（当前lookback=672步，仅42小时）
2. 模型容量不足以捕捉长期依赖
3. 训练不充分（max_iter=10太少）
4. 注意力机制参数不适合长序列

### 实验设计

#### **实验组A：增加历史窗口长度（112-117号）**
测试更长的lookback是否能提升长期预测

| 实验号 | 数据集 | 模型 | 预测时长 | horizon | lookback_multiplier | lookback | 说明 |
|--------|--------|------|----------|---------|---------------------|----------|------|
| 112 | TX1 | Informer | 1 week | 168 | 8x | 1344步 (14天) | 基准增加1倍 |
| 113 | TX1 | Informer | 1 week | 168 | 12x | 2016步 (21天) | 基准增加2倍 |
| 114 | TX1 | Informer | 1 week | 168 | 16x | 2688步 (28天) | 基准增加3倍 |
| 115 | TX2 | Informer | 1 week | 168 | 8x | 1344步 (14天) | 基准增加1倍 |
| 116 | TX2 | Informer | 1 week | 168 | 12x | 2016步 (21天) | 基准增加2倍 |
| 117 | TX2 | Informer | 1 week | 168 | 16x | 2688步 (28天) | 基准增加3倍 |

**固定参数**：split_method=chronological, feature_mode=full

#### **实验组B：调整Encoder输入长度seq_len（118-123号）**
测试不同seq_len对模型容量的影响（当前336）

| 实验号 | 数据集 | seq_len | label_len | pred_len | 说明 |
|--------|--------|---------|-----------|----------|------|
| 118 | TX1 | 672 | 336 | 168 | 2倍增长 |
| 119 | TX1 | 1008 | 504 | 168 | 3倍增长 |
| 120 | TX1 | 1344 | 672 | 168 | 4倍增长 |
| 121 | TX2 | 672 | 336 | 168 | 2倍增长 |
| 122 | TX2 | 1008 | 504 | 168 | 3倍增长 |
| 123 | TX2 | 1344 | 672 | 168 | 4倍增长 |

**固定参数**：lookback_multiplier=4x

#### **实验组C：增加模型深度（124-129号）**
测试更深的encoder/decoder层数（当前e_layers=2, d_layers=1）

| 实验号 | 数据集 | e_layers | d_layers | 说明 |
|--------|--------|----------|----------|------|
| 124 | TX1 | 3 | 2 | encoder+1, decoder+1 |
| 125 | TX1 | 4 | 2 | encoder+2, decoder+1 |
| 126 | TX1 | 3 | 3 | encoder+1, decoder+2 |
| 127 | TX2 | 3 | 2 | encoder+1, decoder+1 |
| 128 | TX2 | 4 | 2 | encoder+2, decoder+1 |
| 129 | TX2 | 3 | 3 | encoder+1, decoder+2 |

**固定参数**：seq_len=336, label_len=168, lookback_multiplier=4x

#### **实验组D：优化训练策略（130-135号）**
测试更长训练时间和更小学习率（当前max_iter=10, lr=1e-4）

| 实验号 | 数据集 | max_iter | learning_rate | early_stopping_patience | 说明 |
|--------|--------|----------|---------------|------------------------|------|
| 130 | TX1 | 50 | 5e-5 | 10 | 5倍epoch + 更小lr |
| 131 | TX1 | 100 | 5e-5 | 15 | 10倍epoch + 更小lr |
| 132 | TX1 | 50 | 1e-4 | 10 | 仅5倍epoch |
| 133 | TX2 | 50 | 5e-5 | 10 | 5倍epoch + 更小lr |
| 134 | TX2 | 100 | 5e-5 | 15 | 10倍epoch + 更小lr |
| 135 | TX2 | 50 | 1e-4 | 10 | 仅5倍epoch |

**固定参数**：默认架构参数

#### **实验组E：最优组合测试（136-137号）**
基于A-D组最佳参数的组合实验

| 实验号 | 数据集 | 说明 |
|--------|--------|------|
| 136 | TX1 | 从A-D组选择最优参数组合 |
| 137 | TX2 | 从A-D组选择最优参数组合 |

### 实验执行命令模板

```bash
# 实验组A示例（增加lookback）
python -m scripts.train_configurable \
    --tx-id 1 \
    --model Informer \
    --split-method chronological \
    --feature-mode full \
    --horizon 168 \
    --lookback-multiplier 8.0 \
    --exp-id 112

# 实验组B示例（调整seq_len）
python -m scripts.train_configurable \
    --tx-id 1 \
    --model Informer-Long \
    --split-method chronological \
    --feature-mode full \
    --horizon 168 \
    --seq-len 672 \
    --label-len 336 \
    --exp-id 118

# 实验组C示例（增加深度）
python -m scripts.train_configurable \
    --tx-id 1 \
    --model Informer-Long \
    --split-method chronological \
    --feature-mode full \
    --horizon 168 \
    --e-layers 3 \
    --d-layers 2 \
    --exp-id 124

# 实验组D示例（优化训练）
python -m scripts.train_configurable \
    --tx-id 1 \
    --model Informer-Long \
    --split-method chronological \
    --feature-mode full \
    --horizon 168 \
    --max-iter 50 \
    --learning-rate 5e-5 \
    --exp-id 130
```

### 预期成果
- 系统化测试26个新实验配置
- 找到Informer在长期预测（horizon=168）的最优参数
- **目标：将R²从负值提升至0.5以上**
- 分析不同参数对长期预测的影响权重
