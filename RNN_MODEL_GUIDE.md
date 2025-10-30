# RNN 模型使用指南 🆕

## 📖 概述

RNN (Recurrent Neural Network) 模型已成功集成到实验系统中，提供原生的时间序列建模能力。

### 🎯 主要特性

- ✅ **时间序列建模**：原生支持序列数据，保持时间步结构
- ✅ **GPU 加速**：自动检测 CUDA/MPS/CPU
- ✅ **scikit-learn 兼容**：与其他模型相同的使用接口
- ✅ **灵活配置**：支持单向/双向、多层堆叠
- ✅ **Early Stopping**：防止过拟合

---

## 🚀 快速开始

### 基础使用

```bash
# 使用 RNN 训练模型
python -m scripts.train_configurable \
    --tx-id 1 \
    --model RNN \
    --split-method random_window \
    --feature-mode full \
    --lookback-multiplier 4 \
    --horizon 1
```

### 与 MLP 对比

```bash
# MLP（输入：展平向量）
python -m scripts.train_configurable \
    --tx-id 1 --model MLP \
    --split-method random_window

# RNN（输入：序列数据）
python -m scripts.train_configurable \
    --tx-id 1 --model RNN \
    --split-method random_window
```

---

## 🔧 模型参数

### 当前配置（在 train_configurable.py 中）

```python
"RNN": lambda: PyTorchRNNRegressor(
    hidden_size=64,           # RNN 隐藏层大小
    num_layers=2,             # RNN 层数
    dropout=0.2,              # Dropout 比例
    bidirectional=False,      # 是否双向
    learning_rate_init=1e-3,  # 学习率
    max_iter=120,             # 最大 epoch
    batch_size=32,            # Batch size
    random_state=42,          # 随机种子
    early_stopping=True,      # Early stopping
    verbose=False,            # 显示训练进度
    device="auto",            # 自动检测 GPU
)
```

### 参数调整建议

| 参数 | 推荐值 | 说明 |
|------|--------|------|
| **hidden_size** | 32-128 | 更大 = 更强表达能力，更慢 |
| **num_layers** | 1-3 | 更多 = 更深网络，可能过拟合 |
| **dropout** | 0.1-0.3 | 防止过拟合 |
| **bidirectional** | False/True | 双向可看到未来信息（慎用） |
| **batch_size** | 32-128 | GPU 可用时可增大 |

---

## 💡 数据格式

### RNN vs 其他模型的数据差异

```python
# 其他模型（MLP, RandomForest, etc.）
X.shape = (n_samples, lookback * n_features)  # 展平
# 示例：(1000, 68) = 1000个样本，每个样本 17特征 × 4时间步 展平

# RNN 模型
X.shape = (n_samples, lookback, n_features)   # 序列
# 示例：(1000, 4, 17) = 1000个样本，每个样本 4时间步 × 17特征
```

### 自动处理

系统会**自动**根据模型类型选择数据格式：
- `--model RNN` → 使用 `create_sliding_windows_for_rnn()` → 3D 序列
- 其他模型 → 使用 `create_sliding_windows()` → 2D 展平

**您无需手动处理！**

---

## 📊 适用场景

### ✅ RNN 更适合

- 需要捕捉**时间依赖关系**
- 序列长度较长（lookback > 10）
- 数据有明显的**时序模式**
- 希望利用 GPU 加速

### ⚠️ RNN 可能不适合

- 数据无时序依赖（考虑 MLP 或 RandomForest）
- 训练数据很少（< 1000 样本）
- 需要极快的训练速度（RandomForest 更快）
- 使用 `chronological` 划分（RNN 需要滑动窗口）

---

## 🎯 使用示例

### 示例 1：基本 RNN

```bash
python -m scripts.train_configurable \
    --tx-id 1 \
    --model RNN \
    --split-method random_window \
    --lookback-multiplier 4 \
    --horizon 1
```

**说明**：
- TX1 变压器
- 随机窗口划分
- Lookback = 4 × horizon = 4 时间步
- 预测 1 步ahead

### 示例 2：长序列 RNN

```bash
python -m scripts.train_configurable \
    --tx-id 1 \
    --model RNN \
    --split-method random_window \
    --lookback-multiplier 8 \
    --horizon 24
```

**说明**：
- Lookback = 8 × 24 = 192 时间步（约 8 天）
- 预测 24 步ahead（1 天后）
- 适合长期预测

### 示例 3：仅时间特征 RNN

```bash
python -m scripts.train_configurable \
    --tx-id 1 \
    --model RNN \
    --split-method random_window \
    --feature-mode time_only \
    --lookback-multiplier 4 \
    --horizon 1
```

**说明**：
- 仅使用时间特征（hour, dayofweek, season, etc.）
- 排除负载特征（HULL, MULL）
- 测试时间模式的预测能力

### 示例 4：分组随机 + RNN

```bash
python -m scripts.train_configurable \
    --tx-id 1 \
    --model RNN \
    --split-method group_random \
    --n-groups 20 \
    --lookback-multiplier 4 \
    --horizon 1
```

**说明**：
- 分组随机划分（减少数据泄露）
- 20 个组，80/20 划分
- 比纯随机更严格的评估

---

## 🔄 模型对比实验

### RNN vs MLP

```bash
# 训练 MLP
python -m scripts.train_configurable \
    --tx-id 1 --model MLP \
    --split-method random_window \
    --experiment-name "exp_mlp"

# 训练 RNN
python -m scripts.train_configurable \
    --tx-id 1 --model RNN \
    --split-method random_window \
    --experiment-name "exp_rnn"

# 比较结果
cat models/experiments/exp_mlp_metrics.json
cat models/experiments/exp_rnn_metrics.json
```

### 不同 Lookback 倍数

```bash
# 1x: lookback = 1 × horizon
python -m scripts.train_configurable \
    --tx-id 1 --model RNN --lookback-multiplier 1 --horizon 1

# 4x: lookback = 4 × horizon
python -m scripts.train_configurable \
    --tx-id 1 --model RNN --lookback-multiplier 4 --horizon 1

# 8x: lookback = 8 × horizon
python -m scripts.train_configurable \
    --tx-id 1 --model RNN --lookback-multiplier 8 --horizon 1
```

---

## ⚙️ 高级配置

### 修改 RNN 架构

如需调整 RNN 参数，编辑 `scripts/train_configurable.py`：

```python
# 文件: scripts/train_configurable.py
# 行数: ~83-95

"RNN": lambda: PyTorchRNNRegressor(
    hidden_size=128,          # ← 增大隐藏层
    num_layers=3,             # ← 增加层数
    dropout=0.3,              # ← 增加 dropout
    bidirectional=True,       # ← 启用双向 RNN
    learning_rate_init=5e-4,  # ← 降低学习率
    max_iter=200,             # ← 更多 epoch
    batch_size=64,            # ← 更大 batch
    early_stopping=True,
    verbose=True,             # ← 显示训练进度
    device="auto",
),
```

### 双向 RNN

```python
"RNN": lambda: PyTorchRNNRegressor(
    bidirectional=True,  # 启用双向
    hidden_size=32,      # 双向时 hidden_size 可以减半
    ...
)
```

**注意**：双向 RNN 可以看到"未来"的信息，仅在某些场景下合适（如后处理、批量分析）。

---

## 🐛 故障排除

### 问题 1：形状错误

```
ValueError: RNN expects 3D input (n_samples, seq_length, n_features), got shape (1000, 68)
```

**原因**：数据被错误地展平了
**解决**：确保使用 `--model RNN`，系统会自动使用正确的数据格式

### 问题 2：GPU 不可用

```
Using CPU (no GPU available)
```

**影响**：RNN 训练较慢
**解决**：
- 安装 PyTorch GPU 版本
- CUDA (NVIDIA): `pip install torch --index-url https://download.pytorch.org/whl/cu118`
- macOS 自动支持 MPS（需 macOS 13+, torch >= 2.0）

### 问题 3：训练很慢

**可能原因**：
- CPU 训练（无 GPU）
- Batch size 太小
- 序列太长
- 层数太多

**解决方案**：
- 启用 GPU
- 增大 `batch_size`（如 64-128）
- 减少 `lookback` 或 `num_layers`
- 使用 `--max-windows` 限制样本数

### 问题 4：过拟合

**症状**：训练集 R² 很高，测试集 R² 很低

**解决方案**：
- 增大 `dropout`（0.3-0.5）
- 启用 `early_stopping=True`
- 减少 `num_layers` 或 `hidden_size`
- 增加训练数据

---

## 📈 性能对比

### 训练时间（参考）

| 模型 | 设备 | 时间（相对） | 说明 |
|------|------|-------------|------|
| RandomForest | CPU | 1.0x | 最快 |
| LinearRegression | CPU | 0.5x | 非常快 |
| MLP | CPU | 3.0x | 较慢 |
| MLP | GPU | 1.2x | GPU 加速 |
| RNN | CPU | 5.0x | 很慢 |
| RNN | GPU | 1.8x | GPU 加速（仍比 MLP 慢） |

### 预测性能（取决于数据）

RNN 在以下情况下可能优于 MLP/RandomForest：
- 数据有强时序依赖
- Lookback 窗口较长
- 需要建模长期依赖

---

## 📚 技术细节

### RNN 网络架构

```python
class RNNNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, ...):
        # RNN 层
        self.rnn = nn.RNN(
            input_size=input_size,      # 特征数
            hidden_size=hidden_size,     # 隐藏层大小
            num_layers=num_layers,       # 层数
            batch_first=True,            # (batch, seq, feature) 格式
            dropout=dropout,             # Dropout
            bidirectional=bidirectional  # 单向/双向
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # 输出层
        fc_input_size = hidden_size * 2 if bidirectional else hidden_size
        self.fc = nn.Linear(fc_input_size, 1)

    def forward(self, x):
        # x: (batch, seq_length, input_size)
        rnn_out, hidden = self.rnn(x)

        # 取最后一个时间步
        last_output = rnn_out[:, -1, :]

        # Dropout + FC
        output = self.fc(self.dropout(last_output))
        return output
```

### 与 LSTM/GRU 的区别

| 模型 | 复杂度 | 训练速度 | 长期依赖 | 说明 |
|------|--------|---------|---------|------|
| **RNN** | 低 | 快 | 弱 | 简单，适合短序列 |
| **LSTM** | 高 | 慢 | 强 | 复杂，适合长序列 |
| **GRU** | 中 | 中 | 中 | LSTM 的简化版 |

**当前实现**：使用标准 RNN
**未来扩展**：可以类似地添加 LSTM/GRU 模型

---

## 🔮 后续扩展建议

### 短期（可选）

1. **LSTM 模型**：类似 RNN，但用 `nn.LSTM` 替代 `nn.RNN`
2. **GRU 模型**：使用 `nn.GRU`
3. **注意力机制**：添加 Attention 层

### 中期（可选）

1. **Seq2Seq**：多步预测
2. **Transformer**：更强的序列建模能力
3. **超参数搜索**：自动寻找最佳配置

---

## 📧 总结

### ✅ 已实现

- RNN 模型完整集成
- scikit-learn 兼容接口
- GPU 自动加速
- 序列数据自动处理
- 与现有实验系统无缝集成

### 📖 使用建议

- **首选场景**：时序依赖明显的数据
- **数据划分**：使用 `random_window` 或 `group_random`
- **窗口配置**：lookback = 4-8 × horizon
- **GPU 加速**：强烈推荐

### 🎯 快速命令

```bash
# 最简单的使用
python -m scripts.train_configurable \
    --tx-id 1 --model RNN --split-method random_window

# 完整配置
python -m scripts.train_configurable \
    --tx-id 1 --model RNN \
    --split-method random_window \
    --feature-mode full \
    --lookback-multiplier 4 \
    --horizon 1 \
    --batch-size 32 \
    --max-windows 40000
```

---

**RNN 模型已就绪，开始时序建模吧！** 🚀
