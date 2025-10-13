import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import warnings
warnings.filterwarnings('ignore')

def add_time_features(df):
    """添加时间相关特征"""
    # 确保date列是datetime类型
    if not pd.api.types.is_datetime64_any_dtype(df['date']):
        df['date'] = pd.to_datetime(df['date'])
    
    # 基础时间特征
    df['hour'] = df['date'].dt.hour
    df['dayofweek'] = df['date'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['month'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear
    
    # 周期性编码（避免边界问题）
    # 小时周期（24小时）
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    
    # 星期周期（7天）
    df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
    
    # 月份周期（12个月，季节性）
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # 年内天数周期（365天，长期季节性）
    df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    
    # 分类特征
    df['is_weekend'] = (df['dayofweek'] >= 5).astype(int)  # 周末标记
    df['is_worktime'] = ((df['hour'] >= 8) & (df['hour'] <= 18) & (df['dayofweek'] < 5)).astype(int)  # 工作时间
    
    # 季节分类（1=春，2=夏，3=秋，4=冬）
    df['season'] = ((df['month'] % 12) // 3 + 1)
    
    return df

def load_and_preprocess_data(sample_size=50000, include_time_features=True):
    """加载和预处理数据（可选择是否包含时间特征）"""
    # 加载数据
    trans1 = pd.read_csv('data/trans_1.csv')
    trans2 = pd.read_csv('data/trans_2.csv')

    # 转换日期列
    trans1['date'] = pd.to_datetime(trans1['date'])
    trans2['date'] = pd.to_datetime(trans2['date'])

    # 合并数据
    combined_data = pd.concat([trans1, trans2], ignore_index=True)
    combined_data = combined_data.sort_values('date').reset_index(drop=True)

    # 添加时间特征
    if include_time_features:
        combined_data = add_time_features(combined_data)
        print("✓ 已添加时间特征")

    # 如果数据太大，进行采样
    if len(combined_data) > sample_size:
        combined_data = combined_data.iloc[:sample_size]

    print(f"数据形状: {combined_data.shape}")
    print(f"时间范围: {combined_data['date'].min()} 到 {combined_data['date'].max()}")

    # 特征列定义
    basic_features = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']
    
    if include_time_features:
        time_features = [
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos', 
            'month_sin', 'month_cos', 'doy_sin', 'doy_cos',
            'is_weekend', 'is_worktime', 'season'
        ]
        feature_cols = basic_features + time_features
        print(f"✓ 使用特征: {len(basic_features)}个电气特征 + {len(time_features)}个时间特征")
    else:
        feature_cols = basic_features
        print(f"✓ 仅使用 {len(basic_features)}个电气特征")
    
    target_col = 'OT'

    return combined_data, feature_cols, target_col

def create_sequences(data, feature_cols, target_col, lookback, forecast_horizon, max_samples=10000):
    """创建时间序列序列（限制样本数量）"""
    X, y = [], []

    # 限制处理的数据量
    max_start_idx = min(len(data) - lookback - forecast_horizon, max_samples)

    for i in range(lookback, lookback + max_start_idx):
        # 使用过去lookback个时间点的特征
        X.append(data[feature_cols].iloc[i-lookback:i].values.flatten())  # 展平为2D数组
        # 预测forecast_horizon个时间点后的油温
        y.append(data[target_col].iloc[i + forecast_horizon])

    return np.array(X), np.array(y)

def prepare_datasets(config_type='1h', include_time_features=True):
    """准备不同配置的数据集"""
    # 配置参数
    configs = {
        '1h': {'lookback': 16, 'forecast_horizon': 4},   # 4小时历史预测1小时后
        '1d': {'lookback': 32, 'forecast_horizon': 96},  # 8小时历史预测1天后  
        '1w': {'lookback': 64, 'forecast_horizon': 672}  # 16小时历史预测1周后
    }
    
    config = configs[config_type]
    print(f"\n=== 准备 {config_type} 预测数据集 ===")
    print(f"回溯窗口: {config['lookback']} 个时间点")
    print(f"预测步长: {config['forecast_horizon']} 个时间点")
    
    # 加载数据
    data, feature_cols, target_col = load_and_preprocess_data(include_time_features=include_time_features)
    
    # 创建序列
    X, y = create_sequences(data, feature_cols, target_col, 
                           config['lookback'], config['forecast_horizon'])
    
    print(f"序列形状: X={X.shape}, y={y.shape}")
    
    # 分割数据集（时间序列分割）
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"训练集: {X_train_scaled.shape}, 测试集: {X_test_scaled.shape}")
    
    # 保存数据
    suffix = f"_{config_type}_time" if include_time_features else f"_{config_type}"
    
    np.save(f'X_train{suffix}.npy', X_train_scaled)
    np.save(f'X_test{suffix}.npy', X_test_scaled)
    np.save(f'y_train{suffix}.npy', y_train)
    np.save(f'y_test{suffix}.npy', y_test)
    joblib.dump(scaler, f'scaler{suffix}.pkl')
    
    print(f"✓ 数据已保存: X_train{suffix}.npy, X_test{suffix}.npy, y_train{suffix}.npy, y_test{suffix}.npy")
    print(f"✓ 标准化器已保存: scaler{suffix}.pkl")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

def main():
    """主函数 - 生成对比实验数据"""
    print("=== 变压器油温预测 - 增强版数据预处理 ===")
    print("生成两组数据进行对比：")
    print("1. 仅电气特征")
    print("2. 电气特征 + 时间特征")
    
    for config in ['1h', '1d', '1w']:
        print(f"\n{'='*50}")
        print(f"处理配置: {config}")
        
        # 生成仅电气特征的数据
        print(f"\n--- {config} 配置：仅电气特征 ---")
        prepare_datasets(config, include_time_features=False)
        
        # 生成包含时间特征的数据
        print(f"\n--- {config} 配置：电气特征 + 时间特征 ---")
        prepare_datasets(config, include_time_features=True)
    
    print(f"\n{'='*50}")
    print("✓ 所有数据集准备完成！")
    print("\n对比实验建议：")
    print("1. 使用相同模型分别训练两组数据")
    print("2. 比较 R²、RMSE、MAE 指标的改善")
    print("3. 观察时间特征对不同预测跨度的影响")

if __name__ == "__main__":
    main()