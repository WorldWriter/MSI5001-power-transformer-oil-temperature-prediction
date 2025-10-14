#!/usr/bin/env python3
"""
时间窗口长度实验 - 数据预处理脚本

本脚本为不同时间窗口长度配置生成训练和测试数据集。
支持系统性地测试历史窗口长度对预测性能的影响。

作者: MSI5001项目组
日期: 2024年
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

def load_raw_data():
    """加载原始数据文件"""
    print("=== 加载原始数据文件 ===")
    
    # 原始数据文件路径
    trans1_path = "../../data/trans_1.csv"
    trans2_path = "../../data/trans_2.csv"
    
    print(f"从以下路径加载数据: {trans1_path} 和 {trans2_path}")
    
    # 检查文件是否存在
    if not os.path.exists(trans1_path):
        raise FileNotFoundError(f"数据文件未找到: {trans1_path}")
    if not os.path.exists(trans2_path):
        raise FileNotFoundError(f"数据文件未找到: {trans2_path}")
    
    # 加载数据
    df1 = pd.read_csv(trans1_path)
    df2 = pd.read_csv(trans2_path)
    
    print(f"变压器1数据形状: {df1.shape}")
    print(f"变压器2数据形状: {df2.shape}")
    
    # 合并数据
    df1['transformer'] = 1
    df2['transformer'] = 2
    df_combined = pd.concat([df1, df2], ignore_index=True)
    
    print(f"合并后数据形状: {df_combined.shape}")
    print(f"列名: {df_combined.columns.tolist()}")
    
    return df_combined

def preprocess_data(df):
    """预处理数据"""
    print("=== 数据预处理 ===")
    
    # 转换日期列
    df['date'] = pd.to_datetime(df['date'])
    
    # 按时间排序
    df = df.sort_values('date').reset_index(drop=True)
    
    # 选择特征列（除了date, OT, transformer）
    feature_columns = [col for col in df.columns if col not in ['date', 'OT', 'transformer']]
    
    # 添加变压器标识作为特征
    df['transformer_1'] = (df['transformer'] == 1).astype(int)
    df['transformer_2'] = (df['transformer'] == 2).astype(int)
    feature_columns.extend(['transformer_1', 'transformer_2'])
    
    # 添加时间特征
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    
    # 添加周期性特征
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    feature_columns.extend(['hour', 'day_of_week', 'month', 'hour_sin', 'hour_cos', 'day_sin', 'day_cos'])
    
    print(f"特征列 ({len(feature_columns)}): {feature_columns}")
    
    # 处理缺失值
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # 创建最终的特征DataFrame
    features_df = df[feature_columns + ['date', 'OT']].copy()
    
    print(f"预处理后数据形状: {features_df.shape}")
    print(f"时间范围: {features_df['date'].min()} 到 {features_df['date'].max()}")
    
    return features_df, feature_columns, 'OT'

def create_sequences(data, feature_cols, target_col, lookback, forecast_horizon, max_samples=8000):
    """创建时间序列序列"""
    print(f"创建序列: lookback={lookback}, forecast_horizon={forecast_horizon}")
    
    # 确保数据按时间排序
    data = data.sort_values('date').reset_index(drop=True)
    
    # 提取特征和目标
    X_data = data[feature_cols].values
    y_data = data[target_col].values
    
    # 创建序列
    X_sequences = []
    y_sequences = []
    
    for i in range(len(data) - lookback - forecast_horizon + 1):
        # 输入序列 (lookback个时间步)
        X_seq = X_data[i:i+lookback]
        # 目标值 (forecast_horizon步后的值)
        y_seq = y_data[i+lookback+forecast_horizon-1]
        
        X_sequences.append(X_seq)
        y_sequences.append(y_seq)
        
        # 限制样本数量以避免内存问题
        if len(X_sequences) >= max_samples:
            break
    
    X_sequences = np.array(X_sequences)
    y_sequences = np.array(y_sequences)
    
    print(f"生成序列形状: X={X_sequences.shape}, y={y_sequences.shape}")
    
    # 展平序列用于传统ML模型
    X_flat = X_sequences.reshape(X_sequences.shape[0], -1)
    
    return X_flat, y_sequences

def prepare_dataset(data, feature_cols, target_col, lookback, forecast_horizon, config_name):
    """准备特定配置的数据集"""
    print(f"\n=== 准备数据集: {config_name} ===")
    
    # 创建序列
    X, y = create_sequences(data, feature_cols, target_col, lookback, forecast_horizon)
    
    # 分割训练和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )
    
    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"训练集形状: X={X_train_scaled.shape}, y={y_train.shape}")
    print(f"测试集形状: X={X_test_scaled.shape}, y={y_test.shape}")
    
    # 保存数据
    results_dir = f"../results/{config_name}"
    os.makedirs(results_dir, exist_ok=True)
    
    np.save(f"{results_dir}/X_train.npy", X_train_scaled)
    np.save(f"{results_dir}/X_test.npy", X_test_scaled)
    np.save(f"{results_dir}/y_train.npy", y_train)
    np.save(f"{results_dir}/y_test.npy", y_test)
    
    # 保存标准化器
    joblib.dump(scaler, f"{results_dir}/scaler.pkl")
    
    print(f"✓ 数据已保存到: {results_dir}")
    
    return True

def main():
    """主函数"""
    print("=== 时间窗口实验 - 数据预处理 ===")
    
    # 加载原始数据
    raw_data = load_raw_data()
    data, feature_cols, target_col = preprocess_data(raw_data)
    
    # 实验配置定义
    experiment_configs = [
        # 1小时预测 (4个15分钟间隔)
        {"name": "1h_lookback_16", "lookback": 16, "forecast_horizon": 4, "description": "1小时预测，4小时历史"},
        {"name": "1h_lookback_32", "lookback": 32, "forecast_horizon": 4, "description": "1小时预测，8小时历史"},
        {"name": "1h_lookback_48", "lookback": 48, "forecast_horizon": 4, "description": "1小时预测，12小时历史"},
        {"name": "1h_lookback_96", "lookback": 96, "forecast_horizon": 4, "description": "1小时预测，24小时历史"},
        
        # 1天预测 (96个15分钟间隔)
        {"name": "1d_lookback_96", "lookback": 96, "forecast_horizon": 96, "description": "1天预测，1天历史"},
        {"name": "1d_lookback_192", "lookback": 192, "forecast_horizon": 96, "description": "1天预测，2天历史"},
        {"name": "1d_lookback_288", "lookback": 288, "forecast_horizon": 96, "description": "1天预测，3天历史"},
        {"name": "1d_lookback_672", "lookback": 672, "forecast_horizon": 96, "description": "1天预测，7天历史"},
        
        # 1周预测 (672个15分钟间隔)
        {"name": "1w_lookback_672", "lookback": 672, "forecast_horizon": 672, "description": "1周预测，1周历史"},
        {"name": "1w_lookback_1344", "lookback": 1344, "forecast_horizon": 672, "description": "1周预测，2周历史"},
        {"name": "1w_lookback_2016", "lookback": 2016, "forecast_horizon": 672, "description": "1周预测，3周历史"},
        {"name": "1w_lookback_2688", "lookback": 2688, "forecast_horizon": 672, "description": "1周预测，4周历史"},
    ]
    
    # 保存实验配置摘要
    config_summary = []
    successful_configs = []
    
    print(f"\n总共 {len(experiment_configs)} 个配置需要处理")
    
    for i, config in enumerate(experiment_configs, 1):
        print(f"\n[{i}/{len(experiment_configs)}] 处理配置: {config['name']}")
        
        try:
            # 检查数据长度是否足够
            min_required_length = config['lookback'] + config['forecast_horizon'] + 1000  # 额外缓冲
            if len(data) < min_required_length:
                print(f"⚠️  数据长度不足: 需要{min_required_length}, 实际{len(data)}")
                config_summary.append({
                    'config_name': config['name'],
                    'status': 'failed',
                    'reason': f'数据长度不足: 需要{min_required_length}, 实际{len(data)}',
                    **config
                })
                continue
            
            # 准备数据集
            success = prepare_dataset(
                data, feature_cols, target_col,
                config['lookback'], config['forecast_horizon'], 
                config['name']
            )
            
            if success:
                successful_configs.append(config['name'])
                config_summary.append({
                    'config_name': config['name'],
                    'status': 'success',
                    'reason': 'Successfully processed',
                    **config
                })
                print(f"✓ 配置 {config['name']} 处理成功")
            
        except Exception as e:
            print(f"✗ 配置 {config['name']} 处理失败: {e}")
            config_summary.append({
                'config_name': config['name'],
                'status': 'failed',
                'reason': str(e),
                **config
            })
    
    # 保存配置摘要
    config_df = pd.DataFrame(config_summary)
    config_df.to_csv("../results/experiment_config_summary.csv", index=False)
    
    print(f"\n=== 预处理完成 ===")
    print(f"成功处理的配置: {len(successful_configs)}/{len(experiment_configs)}")
    print(f"成功的配置: {successful_configs}")
    print(f"配置摘要已保存到: ../results/experiment_config_summary.csv")

if __name__ == "__main__":
    main()