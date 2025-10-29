"""
实验配置定义 / Experiment Configuration Definitions

定义所有7个阶段的实验配置
"""

import os
import sys

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(project_root, 'notebooks'))

from utils import DEFAULT_CONFIG

# ============================================================================
# 阶段1：建立基准 (Baseline) - 6个模型
# ============================================================================

STAGE1_CONFIGS = []

# 基准配置
baseline_config = {
    'dataset_path': '../dataset/train2.csv',
    'remove_outliers': False,
    'split_method': 'sequential',
    'time_features': [],  # 不使用时间特征
    'num_epochs': 100,
    'learning_rate': 0.001,
    'batch_size': 32,
}

# 2个算法 × 3个预测场景
for model_type in ['linear', 'rnn']:
    for horizon in ['hour', 'day', 'week']:
        config = baseline_config.copy()
        config['model_type'] = model_type
        config['prediction_horizon'] = horizon
        config['experiment_id'] = f'stage1_{model_type}_{horizon[:1]}h' if horizon == 'hour' else f'stage1_{model_type}_{horizon[:1]}{"d" if horizon == "day" else "w"}'
        config['stage'] = 1
        config['notes'] = f'Baseline - {model_type.upper()} - {horizon}'
        STAGE1_CONFIGS.append(config)

print(f"阶段1配置数量 / Stage 1 configs: {len(STAGE1_CONFIGS)}")

# ============================================================================
# 阶段2：数据预处理影响分析 - 10个模型
# ============================================================================

STAGE2_CONFIGS = []

# 基准配置（使用阶段1的最优模型类型，这里先假设是'linear'，实际运行时会动态调整）
stage2_base = {
    'dataset_path': '../dataset/train2.csv',
    'prediction_horizon': 'hour',  # 只在1小时场景测试
    'model_type': 'linear',  # 待阶段1确定
    'split_method': 'sequential',
    'time_features': [],
    'num_epochs': 100,
    'learning_rate': 0.001,
    'batch_size': 32,
}

# 2.1 异常值剔除比例测试 - 4个模型
for outlier_pct in [None, 0.5, 1.0, 5.0]:
    config = stage2_base.copy()
    if outlier_pct is None:
        config['remove_outliers'] = False
        config['outlier_threshold'] = 3.0
        outlier_label = 'none'
    else:
        config['remove_outliers'] = True
        config['outlier_threshold'] = 3.0  # Z-score threshold
        config['outlier_method'] = 'zscore'
        outlier_label = f'{outlier_pct}pct'

    config['experiment_id'] = f'stage2_outlier_{outlier_label}_1h'
    config['stage'] = 2
    config['notes'] = f'Outlier removal: {outlier_label}'
    STAGE2_CONFIGS.append(config)

# 2.2 训练/测试划分方式测试 - 3个模型
for split_method in ['sequential', 'random', 'label_random']:
    config = stage2_base.copy()
    config['split_method'] = split_method
    config['remove_outliers'] = False  # 使用2.1的最优配置，这里先默认False
    config['experiment_id'] = f'stage2_split_{split_method}_1h'
    config['stage'] = 2
    config['notes'] = f'Split method: {split_method}'
    STAGE2_CONFIGS.append(config)

# 2.3 最优组合验证 - 3个模型（在3个预测场景下）
# 注意：这些配置需要在2.1和2.2完成后，根据最优结果动态生成
# 这里先预留占位符
for horizon in ['hour', 'day', 'week']:
    config = stage2_base.copy()
    config['prediction_horizon'] = horizon
    # 下面两个参数需要根据2.1和2.2的结果动态设置
    config['remove_outliers'] = False  # TODO: 从2.1结果获取
    config['split_method'] = 'sequential'  # TODO: 从2.2结果获取
    suffix = 'h' if horizon == 'hour' else ('d' if horizon == 'day' else 'w')
    config['experiment_id'] = f'stage2_best_combo_{suffix}'
    config['stage'] = 2
    config['notes'] = f'Best preprocessing combo - {horizon}'
    STAGE2_CONFIGS.append(config)

print(f"阶段2配置数量 / Stage 2 configs: {len(STAGE2_CONFIGS)}")

# ============================================================================
# 阶段3：特征工程影响分析 - 12个模型
# ============================================================================

STAGE3_CONFIGS = []

# 基准配置（使用阶段2的最优数据预处理配置）
stage3_base = {
    'dataset_path': '../dataset/train2.csv',
    'model_type': 'linear',  # 待阶段1确定
    'num_epochs': 100,
    'learning_rate': 0.001,
    'batch_size': 32,
    # 下面参数从阶段2获取
    'remove_outliers': False,
    'split_method': 'sequential',
}

# 3.1 时间特征影响 - 6个模型
for use_time in [False, True]:
    for horizon in ['hour', 'day', 'week']:
        config = stage3_base.copy()
        config['prediction_horizon'] = horizon
        if use_time:
            config['time_features'] = ['hour', 'dayofweek', 'month', 'day', 'is_weekend']
            time_label = 'with_time'
        else:
            config['time_features'] = []
            time_label = 'no_time'

        suffix = 'h' if horizon == 'hour' else ('d' if horizon == 'day' else 'w')
        config['experiment_id'] = f'stage3_time_{time_label}_{suffix}'
        config['stage'] = 3
        config['notes'] = f'Time features: {use_time} - {horizon}'
        STAGE3_CONFIGS.append(config)

# 3.2 自变量特征筛选 - 6个模型
# 注意：这需要先做相关性分析，这里假设从6个特征中选择Top4
for use_selected in [False, True]:
    for horizon in ['hour', 'day', 'week']:
        config = stage3_base.copy()
        config['prediction_horizon'] = horizon
        if use_selected:
            # TODO: 基于相关性分析确定特征子集
            config['load_features'] = ['HUFL', 'HULL', 'MUFL', 'MULL']  # 示例
            feat_label = 'selected'
        else:
            config['load_features'] = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']
            feat_label = 'all'

        suffix = 'h' if horizon == 'hour' else ('d' if horizon == 'day' else 'w')
        config['experiment_id'] = f'stage3_feat_{feat_label}_{suffix}'
        config['stage'] = 3
        config['notes'] = f'Feature selection: {feat_label} - {horizon}'
        STAGE3_CONFIGS.append(config)

print(f"阶段3配置数量 / Stage 3 configs: {len(STAGE3_CONFIGS)}")

# ============================================================================
# 阶段4：时间窗口长度影响 - 12个模型
# ============================================================================

STAGE4_CONFIGS = []

# 基准配置（使用阶段3的最优特征配置）
stage4_base = {
    'dataset_path': '../dataset/train2.csv',
    'model_type': 'linear',
    'num_epochs': 100,
    'learning_rate': 0.001,
    'batch_size': 32,
    # 从前面阶段继承
    'remove_outliers': False,
    'split_method': 'sequential',
    'time_features': [],  # TODO: 从阶段3获取
    'load_features': ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL'],
}

# 4个seq_length × 3个预测场景
for seq_len in [8, 16, 32, 64]:
    for horizon in ['hour', 'day', 'week']:
        config = stage4_base.copy()
        config['prediction_horizon'] = horizon
        config['seq_length'] = seq_len

        suffix = 'h' if horizon == 'hour' else ('d' if horizon == 'day' else 'w')
        config['experiment_id'] = f'stage4_seq{seq_len}_{suffix}'
        config['stage'] = 4
        config['notes'] = f'Sequence length: {seq_len} - {horizon}'
        STAGE4_CONFIGS.append(config)

print(f"阶段4配置数量 / Stage 4 configs: {len(STAGE4_CONFIGS)}")

# ============================================================================
# 阶段5：算法对比 - 12个模型
# ============================================================================

STAGE5_CONFIGS = []

# 基准配置（使用阶段4的最优时间窗口配置）
stage5_base = {
    'dataset_path': '../dataset/train2.csv',
    'num_epochs': 100,
    'learning_rate': 0.001,
    'batch_size': 32,
    # 从前面阶段继承
    'remove_outliers': False,
    'split_method': 'sequential',
    'time_features': [],
    'load_features': ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL'],
    'seq_length': 16,  # TODO: 从阶段4获取最优值
}

# 4个算法 × 3个预测场景
for model_type in ['linear', 'rnn', 'lstm', 'gru']:
    for horizon in ['hour', 'day', 'week']:
        config = stage5_base.copy()
        config['model_type'] = model_type
        config['prediction_horizon'] = horizon

        # RNN系列模型的特定参数
        if model_type in ['rnn', 'lstm', 'gru']:
            config['hidden_size'] = 64
            config['num_layers'] = 2
            config['dropout'] = 0.2
            config['bidirectional'] = False

        suffix = 'h' if horizon == 'hour' else ('d' if horizon == 'day' else 'w')
        config['experiment_id'] = f'stage5_{model_type}_{suffix}'
        config['stage'] = 5
        config['notes'] = f'Algorithm: {model_type.upper()} - {horizon}'
        STAGE5_CONFIGS.append(config)

print(f"阶段5配置数量 / Stage 5 configs: {len(STAGE5_CONFIGS)}")

# ============================================================================
# 阶段6：超参数精调 - 27个模型
# ============================================================================

STAGE6_CONFIGS = []

# 基准配置（使用阶段5的最优算法）
stage6_base = {
    'dataset_path': '../dataset/train2.csv',
    'prediction_horizon': 'hour',  # 只在1小时场景调优
    'model_type': 'linear',  # TODO: 从阶段5获取最优算法
    'num_epochs': 100,
    'batch_size': 32,
    # 从前面阶段继承
    'remove_outliers': False,
    'split_method': 'sequential',
    'time_features': [],
    'load_features': ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL'],
    'seq_length': 16,
}

# 6.1 学习率调优 - 3个模型
for lr in [0.0001, 0.001, 0.01]:
    config = stage6_base.copy()
    config['learning_rate'] = lr
    config['experiment_id'] = f'stage6_lr{lr}_1h'
    config['stage'] = 6
    config['notes'] = f'Learning rate: {lr}'
    STAGE6_CONFIGS.append(config)

# 6.2 Batch size调优 - 3个模型
for bs in [16, 32, 64]:
    config = stage6_base.copy()
    config['batch_size'] = bs
    config['learning_rate'] = 0.001  # TODO: 使用6.1的最优值
    config['experiment_id'] = f'stage6_bs{bs}_1h'
    config['stage'] = 6
    config['notes'] = f'Batch size: {bs}'
    STAGE6_CONFIGS.append(config)

# 6.3 隐藏层大小调优 - 3个模型（仅RNN系列）
for hs in [32, 64, 128]:
    config = stage6_base.copy()
    config['hidden_size'] = hs  # 对于linear模型，这会影响hidden_sizes参数
    config['learning_rate'] = 0.001
    config['batch_size'] = 32
    config['experiment_id'] = f'stage6_hs{hs}_1h'
    config['stage'] = 6
    config['notes'] = f'Hidden size: {hs}'
    STAGE6_CONFIGS.append(config)

# 6.4 Dropout率调优 - 3个模型
for dp in [0.0, 0.2, 0.4]:
    config = stage6_base.copy()
    config['dropout'] = dp
    config['learning_rate'] = 0.001
    config['batch_size'] = 32
    config['experiment_id'] = f'stage6_dp{dp}_1h'
    config['stage'] = 6
    config['notes'] = f'Dropout: {dp}'
    STAGE6_CONFIGS.append(config)

# 6.5 最优超参数组合验证 - 3个模型（在3个预测场景）
for horizon in ['hour', 'day', 'week']:
    config = stage6_base.copy()
    config['prediction_horizon'] = horizon
    # TODO: 使用6.1-6.4的最优超参数
    config['learning_rate'] = 0.001
    config['batch_size'] = 32
    config['hidden_size'] = 64
    config['dropout'] = 0.2

    suffix = 'h' if horizon == 'hour' else ('d' if horizon == 'day' else 'w')
    config['experiment_id'] = f'stage6_best_combo_{suffix}'
    config['stage'] = 6
    config['notes'] = f'Best hyperparams - {horizon}'
    STAGE6_CONFIGS.append(config)

print(f"阶段6配置数量 / Stage 6 configs: {len(STAGE6_CONFIGS)}")

# ============================================================================
# 阶段7：最终验证与消融实验 - 9个模型
# ============================================================================

STAGE7_CONFIGS = []

# 7.1 train1数据集验证 - 3个模型
stage7_best = {
    'dataset_path': '../dataset/train1.csv',  # 切换到train1
    'model_type': 'linear',  # TODO: 最优算法
    'num_epochs': 100,
    # TODO: 使用阶段6的所有最优配置
    'learning_rate': 0.001,
    'batch_size': 32,
    'remove_outliers': False,
    'split_method': 'sequential',
    'time_features': [],
    'load_features': ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL'],
    'seq_length': 16,
}

for horizon in ['hour', 'day', 'week']:
    config = stage7_best.copy()
    config['prediction_horizon'] = horizon
    suffix = 'h' if horizon == 'hour' else ('d' if horizon == 'day' else 'w')
    config['experiment_id'] = f'stage7_train1_{suffix}'
    config['stage'] = 7
    config['notes'] = f'Train1 validation - {horizon}'
    STAGE7_CONFIGS.append(config)

# 7.2 消融实验 - 6个模型
# 在最优配置下，逐个移除关键组件
ablation_scenarios = [
    {'time_features': [], 'notes': 'Ablation: Remove time features'},
    {'remove_outliers': False, 'notes': 'Ablation: No outlier removal'},
]

for scenario in ablation_scenarios:
    for horizon in ['hour', 'day', 'week']:
        config = stage7_best.copy()
        config['dataset_path'] = '../dataset/train2.csv'  # 消融实验在train2
        config['prediction_horizon'] = horizon
        config.update(scenario)

        suffix = 'h' if horizon == 'hour' else ('d' if horizon == 'day' else 'w')
        ablation_label = scenario['notes'].split(':')[1].strip().replace(' ', '_').lower()
        config['experiment_id'] = f'stage7_ablation_{ablation_label}_{suffix}'
        config['stage'] = 7
        STAGE7_CONFIGS.append(config)

print(f"阶段7配置数量 / Stage 7 configs: {len(STAGE7_CONFIGS)}")

# ============================================================================
# 汇总所有配置
# ============================================================================

ALL_CONFIGS = {
    1: STAGE1_CONFIGS,
    2: STAGE2_CONFIGS,
    3: STAGE3_CONFIGS,
    4: STAGE4_CONFIGS,
    5: STAGE5_CONFIGS,
    6: STAGE6_CONFIGS,
    7: STAGE7_CONFIGS,
}

# 统计
total_models = sum(len(configs) for configs in ALL_CONFIGS.values())
print(f"\n{'='*60}")
print(f"实验配置总结 / Experiment Configuration Summary")
print(f"{'='*60}")
for stage, configs in ALL_CONFIGS.items():
    print(f"阶段{stage} / Stage {stage}: {len(configs)} 个模型")
print(f"总计 / Total: {total_models} 个模型")
print(f"{'='*60}\n")


def get_stage_configs(stage):
    """获取指定阶段的配置 / Get configs for a specific stage"""
    return ALL_CONFIGS.get(stage, [])


def get_all_configs():
    """获取所有配置 / Get all configs"""
    all_configs_list = []
    for configs in ALL_CONFIGS.values():
        all_configs_list.extend(configs)
    return all_configs_list
