# 实验112-137号执行命令集

基于 `experiment_plan.md` 中设计的Informer优化实验，以下是完整的执行命令。

## 修改说明

`train_configurable.py` 已更新，支持以下动态参数：

### 架构参数
- `--seq-len`: Encoder输入序列长度
- `--label-len`: Decoder起始token长度
- `--pred-len`: 预测序列长度
- `--d-model`: 模型维度
- `--n-heads`: 注意力头数量
- `--e-layers`: Encoder层数
- `--d-layers`: Decoder层数
- `--d-ff`: 前馈网络维度
- `--factor`: ProbSparse注意力因子

### 训练参数
- `--train-epochs`: 训练epoch数
- `--batch-size`: 批大小
- `--learning-rate`: 学习率
- `--patience`: 早停耐心值

## 实验组A：增加历史窗口长度（112-117号）

### TX1实验

```bash
# 实验112: lookback_multiplier=8x (1344步，约14天)
python -m scripts.train_configurable ^
    --tx-id 1 ^
    --model Informer-Long ^
    --split-method chronological ^
    --feature-mode full ^
    --horizon 168 ^
    --lookback-multiplier 8.0 ^
    --experiment-name exp_112

# 实验113: lookback_multiplier=12x (2016步，约21天)
python -m scripts.train_configurable ^
    --tx-id 1 ^
    --model Informer-Long ^
    --split-method chronological ^
    --feature-mode full ^
    --horizon 168 ^
    --lookback-multiplier 12.0 ^
    --experiment-name exp_113

# 实验114: lookback_multiplier=16x (2688步，约28天)
python -m scripts.train_configurable ^
    --tx-id 1 ^
    --model Informer-Long ^
    --split-method chronological ^
    --feature-mode full ^
    --horizon 168 ^
    --lookback-multiplier 16.0 ^
    --experiment-name exp_114
```

### TX2实验

```bash
# 实验115: lookback_multiplier=8x
python -m scripts.train_configurable ^
    --tx-id 2 ^
    --model Informer-Long ^
    --split-method chronological ^
    --feature-mode full ^
    --horizon 168 ^
    --lookback-multiplier 8.0 ^
    --experiment-name exp_115

# 实验116: lookback_multiplier=12x
python -m scripts.train_configurable ^
    --tx-id 2 ^
    --model Informer-Long ^
    --split-method chronological ^
    --feature-mode full ^
    --horizon 168 ^
    --lookback-multiplier 12.0 ^
    --experiment-name exp_116

# 实验117: lookback_multiplier=16x
python -m scripts.train_configurable ^
    --tx-id 2 ^
    --model Informer-Long ^
    --split-method chronological ^
    --feature-mode full ^
    --horizon 168 ^
    --lookback-multiplier 16.0 ^
    --experiment-name exp_117
```

## 实验组B：调整Encoder输入长度seq_len（118-123号）

### TX1实验

```bash
# 实验118: seq_len=672, label_len=336
python -m scripts.train_configurable ^
    --tx-id 1 ^
    --model Informer-Long ^
    --split-method chronological ^
    --feature-mode full ^
    --horizon 168 ^
    --lookback-multiplier 4.0 ^
    --seq-len 672 ^
    --label-len 336 ^
    --experiment-name exp_118

# 实验119: seq_len=1008, label_len=504
python -m scripts.train_configurable ^
    --tx-id 1 ^
    --model Informer-Long ^
    --split-method chronological ^
    --feature-mode full ^
    --horizon 168 ^
    --lookback-multiplier 4.0 ^
    --seq-len 1008 ^
    --label-len 504 ^
    --experiment-name exp_119

# 实验120: seq_len=1344, label_len=672
python -m scripts.train_configurable ^
    --tx-id 1 ^
    --model Informer-Long ^
    --split-method chronological ^
    --feature-mode full ^
    --horizon 168 ^
    --lookback-multiplier 4.0 ^
    --seq-len 1344 ^
    --label-len 672 ^
    --experiment-name exp_120
```

### TX2实验

```bash
# 实验121: seq_len=672, label_len=336
python -m scripts.train_configurable ^
    --tx-id 2 ^
    --model Informer-Long ^
    --split-method chronological ^
    --feature-mode full ^
    --horizon 168 ^
    --lookback-multiplier 4.0 ^
    --seq-len 672 ^
    --label-len 336 ^
    --experiment-name exp_121

# 实验122: seq_len=1008, label_len=504
python -m scripts.train_configurable ^
    --tx-id 2 ^
    --model Informer-Long ^
    --split-method chronological ^
    --feature-mode full ^
    --horizon 168 ^
    --lookback-multiplier 4.0 ^
    --seq-len 1008 ^
    --label-len 504 ^
    --experiment-name exp_122

# 实验123: seq_len=1344, label_len=672
python -m scripts.train_configurable ^
    --tx-id 2 ^
    --model Informer-Long ^
    --split-method chronological ^
    --feature-mode full ^
    --horizon 168 ^
    --lookback-multiplier 4.0 ^
    --seq-len 1344 ^
    --label-len 672 ^
    --experiment-name exp_123
```

## 实验组C：增加模型深度（124-129号）

### TX1实验

```bash
# 实验124: e_layers=3, d_layers=2
python -m scripts.train_configurable ^
    --tx-id 1 ^
    --model Informer-Long ^
    --split-method chronological ^
    --feature-mode full ^
    --horizon 168 ^
    --lookback-multiplier 4.0 ^
    --e-layers 3 ^
    --d-layers 2 ^
    --experiment-name exp_124

# 实验125: e_layers=4, d_layers=2
python -m scripts.train_configurable ^
    --tx-id 1 ^
    --model Informer-Long ^
    --split-method chronological ^
    --feature-mode full ^
    --horizon 168 ^
    --lookback-multiplier 4.0 ^
    --e-layers 4 ^
    --d-layers 2 ^
    --experiment-name exp_125

# 实验126: e_layers=3, d_layers=3
python -m scripts.train_configurable ^
    --tx-id 1 ^
    --model Informer-Long ^
    --split-method chronological ^
    --feature-mode full ^
    --horizon 168 ^
    --lookback-multiplier 4.0 ^
    --e-layers 3 ^
    --d-layers 3 ^
    --experiment-name exp_126
```

### TX2实验

```bash
# 实验127: e_layers=3, d_layers=2
python -m scripts.train_configurable ^
    --tx-id 2 ^
    --model Informer-Long ^
    --split-method chronological ^
    --feature-mode full ^
    --horizon 168 ^
    --lookback-multiplier 4.0 ^
    --e-layers 3 ^
    --d-layers 2 ^
    --experiment-name exp_127

# 实验128: e_layers=4, d_layers=2
python -m scripts.train_configurable ^
    --tx-id 2 ^
    --model Informer-Long ^
    --split-method chronological ^
    --feature-mode full ^
    --horizon 168 ^
    --lookback-multiplier 4.0 ^
    --e-layers 4 ^
    --d-layers 2 ^
    --experiment-name exp_128

# 实验129: e_layers=3, d_layers=3
python -m scripts.train_configurable ^
    --tx-id 2 ^
    --model Informer-Long ^
    --split-method chronological ^
    --feature-mode full ^
    --horizon 168 ^
    --lookback-multiplier 4.0 ^
    --e-layers 3 ^
    --d-layers 3 ^
    --experiment-name exp_129
```

## 实验组D：优化训练策略（130-135号）

### TX1实验

```bash
# 实验130: train_epochs=50, learning_rate=5e-5
python -m scripts.train_configurable ^
    --tx-id 1 ^
    --model Informer-Long ^
    --split-method chronological ^
    --feature-mode full ^
    --horizon 168 ^
    --lookback-multiplier 4.0 ^
    --train-epochs 50 ^
    --learning-rate 5e-5 ^
    --patience 10 ^
    --experiment-name exp_130

# 实验131: train_epochs=100, learning_rate=5e-5
python -m scripts.train_configurable ^
    --tx-id 1 ^
    --model Informer-Long ^
    --split-method chronological ^
    --feature-mode full ^
    --horizon 168 ^
    --lookback-multiplier 4.0 ^
    --train-epochs 100 ^
    --learning-rate 5e-5 ^
    --patience 15 ^
    --experiment-name exp_131

# 实验132: train_epochs=50, learning_rate=1e-4 (仅增加epoch)
python -m scripts.train_configurable ^
    --tx-id 1 ^
    --model Informer-Long ^
    --split-method chronological ^
    --feature-mode full ^
    --horizon 168 ^
    --lookback-multiplier 4.0 ^
    --train-epochs 50 ^
    --learning-rate 1e-4 ^
    --patience 10 ^
    --experiment-name exp_132
```

### TX2实验

```bash
# 实验133: train_epochs=50, learning_rate=5e-5
python -m scripts.train_configurable ^
    --tx-id 2 ^
    --model Informer-Long ^
    --split-method chronological ^
    --feature-mode full ^
    --horizon 168 ^
    --lookback-multiplier 4.0 ^
    --train-epochs 50 ^
    --learning-rate 5e-5 ^
    --patience 10 ^
    --experiment-name exp_133

# 实验134: train_epochs=100, learning_rate=5e-5
python -m scripts.train_configurable ^
    --tx-id 2 ^
    --model Informer-Long ^
    --split-method chronological ^
    --feature-mode full ^
    --horizon 168 ^
    --lookback-multiplier 4.0 ^
    --train-epochs 100 ^
    --learning-rate 5e-5 ^
    --patience 15 ^
    --experiment-name exp_134

# 实验135: train_epochs=50, learning_rate=1e-4 (仅增加epoch)
python -m scripts.train_configurable ^
    --tx-id 2 ^
    --model Informer-Long ^
    --split-method chronological ^
    --feature-mode full ^
    --horizon 168 ^
    --lookback-multiplier 4.0 ^
    --train-epochs 50 ^
    --learning-rate 1e-4 ^
    --patience 10 ^
    --experiment-name exp_135
```

## 实验组E：最优组合测试（136-137号）

**说明**：这组实验需要等A-D组完成后，根据结果选择最优参数组合。

示例命令模板（根据最优结果调整参数）：

```bash
# 实验136: TX1最优组合
python -m scripts.train_configurable ^
    --tx-id 1 ^
    --model Informer-Long ^
    --split-method chronological ^
    --feature-mode full ^
    --horizon 168 ^
    --lookback-multiplier 12.0 ^
    --seq-len 1008 ^
    --label-len 504 ^
    --e-layers 4 ^
    --d-layers 2 ^
    --train-epochs 100 ^
    --learning-rate 5e-5 ^
    --patience 15 ^
    --experiment-name exp_136

# 实验137: TX2最优组合
python -m scripts.train_configurable ^
    --tx-id 2 ^
    --model Informer-Long ^
    --split-method chronological ^
    --feature-mode full ^
    --horizon 168 ^
    --lookback-multiplier 12.0 ^
    --seq-len 1008 ^
    --label-len 504 ^
    --e-layers 4 ^
    --d-layers 2 ^
    --train-epochs 100 ^
    --learning-rate 5e-5 ^
    --patience 15 ^
    --experiment-name exp_137
```

## 快速测试（使用最少epoch验证配置）

在正式运行前，建议用少量epoch测试配置是否正确：

```bash
python -m scripts.train_configurable ^
    --tx-id 2 ^
    --model Informer-Long ^
    --split-method chronological ^
    --feature-mode full ^
    --horizon 168 ^
    --lookback-multiplier 8.0 ^
    --train-epochs 2 ^
    --experiment-name quick_test
```

## 注意事项

1. **Windows命令行**：使用 `^` 作为续行符
2. **Linux/Mac命令行**：将 `^` 替换为 `\`
3. **GPU内存**：如遇显存不足，降低 `--batch-size` 或 `--d-model`
4. **训练时间**：
   - 实验组A（增加lookback）：最耗时，建议先小规模测试
   - 实验组D（增加epoch）：次耗时
   - 实验组B、C：相对较快
5. **结果保存**：
   - 日志：`results/logs/exp_XXX.log`
   - 指标：通过日志文件解析
   - 模型：Informer使用自己的checkpoint系统

## 批量执行建议

创建批处理脚本（Windows `.bat` 或 Linux `.sh`）来批量运行：

```bash
@echo off
REM 实验组A批量执行
for %%i in (8.0 12.0 16.0) do (
    python -m scripts.train_configurable --tx-id 1 --model Informer-Long --split-method chronological --feature-mode full --horizon 168 --lookback-multiplier %%i --experiment-name exp_112_%%i
)
```
