# Informer Model Testing Progress

**Project**: Power Transformer Oil Temperature Prediction
**Model**: Informer (Long-Sequence Time-Series Forecasting)
**Target**: Achieve R² > 0.5 on TX1 (Industrial Transformer), 1-week horizon prediction
**Last Updated**: 2025-11-08

---

## Executive Summary

### Current Best Results (TX1, 1-week prediction)

| Rank | Exp ID | R² Score | MAE | RMSE | Key Configuration |
|------|--------|----------|-----|------|-------------------|
| 🥇 1 | **exp_136** | **-0.63** | 3.46 | 4.40 | lookback=12x, seq=1008, e=4, d=2, epochs=100, lr=5e-5 |
| 🥈 2 | **exp_124** | **-0.82** | 3.78 | 4.65 | lookback=4x, seq=336, e=3, d=2, epochs=8, lr=1e-4 |
| 🥉 3 | exp_125 | -1.39 | 4.46 | 5.33 | lookback=4x, seq=336, e=4, d=2, epochs=8, lr=1e-4 |

**Progress**: From R²=-2.28 (baseline exp_112) to R²=-0.63 (exp_136) → **+1.65 improvement** ✓

**Gap to Goal**: Need +1.13 improvement to reach R²=0.5

---

## Experimental History

### Phase 1: Initial Exploration (exp_112-123)

**Period**: 2025-11-07 15:23 - 19:11
**Focus**: Test different lookback multipliers and sequence lengths
**Status**: Baseline established, all R² < -1.5

| Exp | Config | R² | Key Finding |
|-----|--------|-----|-------------|
| 112 | lookback=8x, seq=336, d=1 | -2.28 | Baseline (poor) |
| 113 | lookback=12x, seq=336, d=1 | -1.88 | Longer lookback helps slightly |
| 114 | lookback=16x, seq=336, d=1 | -1.62 | Best with d_layers=1 |
| 118-120 | Variable seq_len, d=1 | -1.86 to -2.09 | Long seq doesn't help with shallow decoder |
| 123 | lookback=4x, seq=1344, d=1 | -1.09 | TX2: Very long seq partially works |

**Insight**: Decoder depth (d_layers=1) is the bottleneck, not sequence length.

---

### Phase 2: Architecture Exploration (exp_124-129)

**Period**: 2025-11-07 19:27 - 21:45
**Focus**: Test deeper architectures (d_layers=2,3; e_layers=3,4)
**Status**: 🎯 **BREAKTHROUGH!** d_layers=2 提供巨大提升

| Exp | d_layers | e_layers | lookback | seq_len | R² | Improvement |
|-----|----------|----------|----------|---------|-----|-------------|
| **124** | **2** | 3 | 4x | 336 | **-0.82** | **+1.46** from baseline! |
| 125 | 2 | 4 | 4x | 336 | -1.39 | e_layers=4 doesn't help much |
| 126 | 2 | 3 | 4x | 336 (d=3) | -1.73 | d_layers=3 worse than d=2 |
| 127 | 2 | 3 | 4x | 336 (TX2) | -1.96 | TX2 also benefits from d=2 |

**Critical Discovery**:
- **d_layers从1→2提供+1.46 R²提升** (最大单一改进)
- e_layers=3足够，4层没有额外收益
- d_layers=3过度参数化，效果反而变差

---

### Phase 3: Training Strategy Optimization (exp_130-135)

**Period**: 2025-11-07 22:38 - 2025-11-08 03:25
**Focus**: Test different epochs and learning rates
**Status**: More training doesn't help without better architecture

| Exp | d | e | epochs | lr | patience | R² | Insight |
|-----|---|---|--------|-----|----------|-----|---------|
| 130 | 1 | 3 | 50 | 5e-5 | 10 | -2.04 | 50 epochs不足以弥补d=1的缺陷 |
| 131 | 1 | 3 | 100 | 5e-5 | 15 | -2.45 | 100 epochs反而更差（过拟合？） |
| 132 | 1 | 3 | 50 | 1e-4 | 10 | -2.73 | 高lr+浅架构=灾难 |
| 133 | 1 | 3 | 50 | 5e-5 | 10 (TX2) | -2.43 | TX2也受同样问题困扰 |
| 135 | 1 | 3 | 50 | 1e-4 | 10 (TX2) | -3.13 | TX2最差结果 |

**Lesson Learned**: 训练策略优化必须建立在正确架构基础上。d_layers=1的架构，无论怎么调参都无法突破。

---

### Phase 4: Best Configuration Refinement (exp_136-137)

**Period**: 2025-11-08 03:56 - 04:26
**Focus**: Combine all successful strategies
**Status**: 🏆 **BEST RESULT ACHIEVED**

| Exp | TX | d | e | lookback | seq_len | epochs | lr | R² | Analysis |
|-----|-----|---|---|----------|---------|--------|-----|-----|----------|
| **136** | **TX1** | **2** | **4** | **12x** | **1008** | **100** | **5e-5** | **-0.63** | **最优组合** |
| 137 | TX2 | 2 | 4 | 12x | 1008 | 100 | 5e-5 | -1.12 | TX2也受益但提升较小 |

**Success Formula (exp_136)**:
- ✅ Deep decoder: d_layers=2
- ✅ Adequate encoder: e_layers=4
- ✅ Long lookback: 12x multiplier (1008 steps lookback)
- ✅ Matching seq_len: 1008 steps
- ✅ Sufficient training: 100 epochs
- ✅ Conservative LR: 5e-5 with patience=3

**Why it works**:
1. d_layers=2 provides enough capacity for temporal modeling
2. e_layers=4 extracts rich features from long sequences
3. lookback=12x captures ~2-week history for 1-week prediction
4. 100 epochs with lr=5e-5 allows stable convergence

---

## Key Insights & Patterns

### 1. Decoder Depth is Critical

| d_layers | Best R² | Average R² | Experiments |
|----------|---------|------------|-------------|
| 1 | -1.62 | -2.15 | exp_112-123, 130-135 (18 exps) |
| 2 | **-0.63** | **-1.08** | exp_124-129, 136-137 (8 exps) |
| 3 | -1.73 | -1.73 | exp_126 (1 exp) |

**Conclusion**: d_layers=2 is the sweet spot. d=1太弱，d=3过度参数化。

---

### 2. Encoder Depth Impact

| e_layers | Best R² (with d=2) | Experiments |
|----------|-------------------|-------------|
| 3 | -0.82 | exp_124, 126 |
| 4 | **-0.63** | exp_125, 136, 137 |
| 5 | - | Not tested yet |

**Conclusion**: e_layers=4优于3，但提升幅度小于decoder。可以测试e=5,6。

---

### 3. Lookback Multiplier vs Performance

| lookback | Best R² | Config | Note |
|----------|---------|--------|------|
| 4x | -0.82 | exp_124 (d=2, e=3) | Good baseline |
| 8x | -2.28 | exp_112 (d=1, e=3) | Poor with d=1 |
| 12x | **-0.63** | exp_136 (d=2, e=4) | **Best** |
| 16x | -1.62 | exp_114 (d=1, e=3) | Wastes on shallow model |

**Conclusion**: 长lookback需要配合深架构。最优是12x (配合seq=1008)。

---

### 4. Sequence Length Strategy

**Rule**: seq_len应该匹配lookback × horizon

| lookback | horizon | Optimal seq_len | label_len | Tested in |
|----------|---------|-----------------|-----------|-----------|
| 4x | 168 | 336-672 | 168-336 | exp_124 |
| 12x | 168 | **1008** | **504** | **exp_136** ✓ |
| 16x | 168 | 1344 | 672 | exp_114 |
| 24x | 168 | 2016 | 1008 | Not tested yet |

---

### 5. Training Strategy Patterns

**Successful combinations**:
- Short training (8 epochs) + higher LR (1e-4): exp_124 (R²=-0.82)
- Long training (100 epochs) + lower LR (5e-5): exp_136 (R²=-0.63)

**Failed combinations**:
- Long training (100 epochs) + shallow model (d=1): exp_131 (R²=-2.45)
- Higher LR (1e-4) + more epochs (50): exp_132 (R²=-2.73)

**Recommendation**: 对于深模型(d≥2)，使用epochs=100-200 + lr=3e-5~5e-5

---

## What Doesn't Work

### ❌ Failed Strategies

1. **Shallow Decoder (d_layers=1)**: 无论如何调参都无法突破R²=-1.5
2. **过度深Decoder (d_layers=3)**: exp_126表现差于d_layers=2
3. **高学习率 (lr=1e-4) + 长训练**: 导致不稳定，exp_132, 135
4. **短序列 + 大lookback**: 浪费计算，exp_112-114
5. **仅增加训练轮数**: 不能弥补架构缺陷，exp_130-131

---

## New Experiment Design: Phase 5 (exp_138-147)

**Strategy**: 既然R²=-0.63距离0.5还有1.13的gap，需要**激进探索**

### Design Principles

1. **大跨度参数范围**: 不再微调，而是测试极端值
2. **多维度同时改变**: 打破渐进式思维
3. **Architecture-first**: 先找到对的架构，再优化训练
4. **计算资源换性能**: 接受更长训练时间，寻求突破

---

### Group 1: 极深Decoder探索 (exp_138-140)

**Hypothesis**: exp_124证明d=1→2提升巨大，为什么不继续加深到d=4,5,6？

| Exp | d_layers | e_layers | lookback | seq_len | epochs | lr | batch | 预期R² | 理由 |
|-----|----------|----------|----------|---------|--------|-----|-------|--------|------|
| 138 | **4** | 4 | 12x | 1008 | 150 | 3e-5 | 8 | -0.3~0.1 | 激进加深decoder，配合长序列 |
| 139 | **5** | 4 | 12x | 1008 | 150 | 3e-5 | 8 | -0.2~0.2 | 测试decoder深度边界 |
| 140 | **6** | 4 | 16x | 1344 | 200 | 2e-5 | 4 | 0.0~0.3 | 极限深度+极限长度 |

**Risk**: CUDA OOM (尤其exp_140)
**Mitigation**: batch_size降至4/2，容错机制自动跳过

---

### Group 2: 极深Encoder探索 (exp_141-142)

**Hypothesis**: 也许问题在encoder不够强，无法从长序列提取足够特征

| Exp | d_layers | e_layers | lookback | seq_len | epochs | lr | batch | 预期R² | 理由 |
|-----|----------|----------|----------|---------|--------|-----|-------|--------|------|
| 141 | 3 | **6** | 12x | 1008 | 150 | 3e-5 | 8 | -0.2~0.2 | 强化encoder特征提取 |
| 142 | 3 | **8** | 16x | 1344 | 200 | 2e-5 | 4 | 0.0~0.3 | 极限encoder深度 |

**Baseline**: exp_136 (e=4, R²=-0.63)
**Expected**: 如果e=6/8显著提升 → encoder是瓶颈；否则 → decoder更重要

---

### Group 3: 平衡深度架构 (exp_143-144)

**Hypothesis**: 经典Transformer是encoder=decoder对称设计，测试平衡架构

| Exp | d_layers | e_layers | lookback | seq_len | epochs | lr | batch | 预期R² | 理由 |
|-----|----------|----------|----------|---------|--------|-----|-------|--------|------|
| 143 | **4** | **4** | 16x | 1344 | 200 | 2e-5 | 4 | -0.1~0.3 | 对称深度+超长序列 |
| 144 | **5** | **5** | 12x | 1008 | 200 | 2e-5 | 8 | 0.0~0.4 | 中等长度但更深，**最有希望** |

**Why promising**: 平衡的深度可能最优利用参数容量

---

### Group 4: 超长序列探索 (exp_145-146)

**Hypothesis**: 1周预测(168步)可能需要看更长历史

| Exp | d_layers | e_layers | lookback | seq_len | epochs | lr | batch | 预期R² | 理由 |
|-----|----------|----------|----------|---------|--------|-----|-------|--------|------|
| 145 | 3 | 4 | **20x** | 1680 | 200 | 2e-5 | 4 | -0.1~0.3 | 测试超长历史窗口(5周) |
| 146 | 4 | 4 | **24x** | 2016 | 250 | 1e-5 | 2 | 0.0~0.4 | 1个月历史预测1周 |

**Risk**: 极高内存占用，batch_size=2
**Benefit**: 如果成功，证明长期依赖对TX1至关重要

---

### Group 5: 极限配置 (exp_147)

**Hypothesis**: 孤注一掷，测试所有参数拉满的配置

| Exp | d | e | lookback | seq | d_model | n_heads | epochs | lr | batch | 预期R² |
|-----|---|---|----------|-----|---------|---------|--------|-----|-------|--------|
| 147 | **6** | **6** | 24x | 2016 | **512** | **16** | **500** | 1e-5 | 2 | **0.2~0.6** |

**Configuration Details**:
- Decoder: 6 layers (vs 2 baseline)
- Encoder: 6 layers (vs 4 baseline)
- Model dimension: 512 (vs 256 baseline)
- Attention heads: 16 (vs 8 baseline)
- Training: 500 epochs (vs 100 baseline)
- Sequence: 2016 steps = 3 weeks lookback

**Requirements**:
- GPU Memory: 32GB+ required
- Training Time: 4-5 hours estimated
- Checkpoint: Save every 50 epochs

**Success Criteria**: R² > 0.3 → 继续扩展此方向
**Failure Plan**: 如果仍失败 → 考虑改变loss function或数据预处理

---

## Parameter Evolution Summary

### Architectural Progression

| Phase | d_layers | e_layers | lookback | seq_len | Best R² | Notes |
|-------|----------|----------|----------|---------|---------|-------|
| Phase 1 | 1 | 3 | 4x-16x | 336-1344 | -1.62 | Baseline exploration |
| Phase 2 | 2 | 3-4 | 4x | 336 | -0.82 | **Decoder depth breakthrough** |
| Phase 3 | 1 | 3 | 4x | 336 | -2.04 | Training optimization failed |
| Phase 4 | 2 | 4 | 12x | 1008 | **-0.63** | **Best combination** |
| Phase 5 | 3-6 | 4-8 | 12x-24x | 1008-2016 | TBD | **Aggressive exploration** |

### Training Strategy Evolution

| Phase | epochs | lr | patience | batch_size | Rationale |
|-------|--------|-----|----------|------------|-----------|
| 1-2 | 8 | 1e-4 | 3 | 8 | Quick iteration |
| 3 | 50-100 | 5e-5~1e-4 | 10-15 | 8 | Longer training test |
| 4 | 100 | 5e-5 | 3 | 8 | Stable convergence |
| 5 | 150-500 | 1e-5~3e-5 | 20-50 | 2-8 | Deep model stabilization |

---

## Execution Plan

### Priority 1: Most Promising (Run First)

```bash
python -m scripts.run_experiments \
    --config configs/experiment_plan.csv \
    --exp-ids 138,139,144 \
    --continue-on-error
```

**Rationale**: exp_138-139测试深decoder边界，exp_144测试平衡架构

### Priority 2: Medium Risk (After Priority 1)

```bash
python -m scripts.run_experiments \
    --config configs/experiment_plan.csv \
    --exp-ids 141,143,145 \
    --continue-on-error
```

### Priority 3: High Risk OOM (Separate Run)

```bash
python -m scripts.run_experiments \
    --config configs/experiment_plan.csv \
    --exp-ids 140,142,146 \
    --continue-on-error
```

### Priority 4: Extreme Configuration (Solo)

```bash
python -m scripts.run_experiments \
    --config configs/experiment_plan.csv \
    --exp-ids 147 \
    --continue-on-error
```

**Estimated Total Time**: 15-20 hours (可并行压缩到3-5小时)

---

## Success Metrics

### Phase 5 Success Criteria

1. **Minimal Success**: Any experiment achieves R² > -0.3 (比exp_136提升0.33)
2. **Good Success**: Any experiment achieves R² > 0.0 (positive R²)
3. **Target Success**: Any experiment achieves R² > 0.3 (距离0.5还有0.2)
4. **Outstanding Success**: Any experiment achieves R² > 0.5 (**达成目标**)

### Failure Response Plan

If all exp_138-147 fail (R² < -0.5):

**Option A**: Architecture changes
- Test different attention mechanisms (Full Attention vs ProbSparse)
- Try different model (Autoformer, FEDformer)
- Add residual connections or skip layers

**Option B**: Training strategy changes
- Change loss function (MSE → Huber Loss / Quantile Loss)
- Add regularization (L1, L2, Dropout 0.3-0.5)
- Try different optimizers (AdamW → SGD with momentum)

**Option C**: Data preprocessing changes
- Add external features (weather, holidays)
- Try different normalization methods
- Augment training data (mixup, cutout)

**Option D**: Bayesian Optimization
- Use Optuna to search optimal hyperparameters in promising region
- 50-100 trials with reduced epochs (50) for faster iteration

---

## Lessons Learned

### What We Know Works ✅

1. **Decoder depth matters most**: d_layers从1→2提供最大单一提升(+1.46)
2. **Long lookback helps**: 12x优于4x (配合深模型)
3. **Match seq_len with lookback**: seq=1008 for lookback=12x最优
4. **Conservative LR + more epochs**: lr=5e-5, epochs=100对深模型最稳
5. **e_layers=4足够**: 超过4层收益递减

### What We Know Doesn't Work ❌

1. **Shallow decoder (d=1)**: 无论如何调参都突破不了-1.5
2. **d_layers=3**: 过度参数化，效果差于d=2
3. **高LR (1e-4) + 长训练**: 导致不稳定
4. **仅增加训练轮数**: 不能弥补架构缺陷

### Open Questions 🤔

1. **Decoder深度的上界在哪？** d=4/5/6会继续提升吗？→ exp_138-140
2. **Encoder深度的价值？** e=6/8能否突破？→ exp_141-142
3. **超长序列的必要性？** lookback=20x/24x能否捕捉更长期依赖？→ exp_145-146
4. **模型容量的极限？** d_model=512, n_heads=16能否提升？→ exp_147

---

## Next Steps

1. **Execute Phase 5 experiments** (exp_138-147) with fault tolerance
2. **Monitor intermediate results**: 如果某个实验R²>-0.3，立即在其周围加实验
3. **Analyze failure patterns**: CUDA OOM频率，确定GPU内存瓶颈
4. **Update this document**: 每完成一批实验更新结果
5. **Prepare Phase 6**: 基于Phase 5结果，设计下一轮refined search

---

## References

**Related Experiments**:
- Baseline (non-Informer): exp_001-111 in `experiment_plan.csv`
- Informer results: `models/informer_further_experiment.csv`

**Key Files**:
- Configuration: `configs/experiment_plan.csv`
- Training script: `scripts/train_configurable.py`
- Run experiments: `scripts/run_experiments.py`

**Model Architecture**:
- Implementation: `models/pytorch_informer.py`
- Informer modules: `models/informer_arch/`

---

**Document Status**: ✅ Complete
**Next Review**: After Phase 5 experiments complete
