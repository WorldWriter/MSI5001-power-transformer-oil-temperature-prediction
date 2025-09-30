import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data():
    """加载和预处理数据"""
    # 加载数据
    trans1 = pd.read_csv('trans_1.csv')
    trans2 = pd.read_csv('trans_2.csv')

    # 转换日期列
    trans1['date'] = pd.to_datetime(trans1['date'])
    trans2['date'] = pd.to_datetime(trans2['date'])

    # 合并数据（按时间排序）
    combined_data = pd.concat([trans1, trans2], ignore_index=True)
    combined_data = combined_data.sort_values('date').reset_index(drop=True)

    print(f"合并后数据形状: {combined_data.shape}")
    print(f"时间范围: {combined_data['date'].min()} 到 {combined_data['date'].max()}")

    # 检查缺失值
    print(f"\n缺失值统计:")
    print(combined_data.isnull().sum())

    # 特征列
    feature_cols = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']
    target_col = 'OT'

    return combined_data, feature_cols, target_col

def create_sequences(data, feature_cols, target_col, lookback, forecast_horizon):
    """
    创建时间序列序列

    Args:
        data: 数据框
        feature_cols: 特征列名
        target_col: 目标列名
        lookback: 回溯时间点数
        forecast_horizon: 预测时间点数（相对于当前点）

    Returns:
        X, y: 特征和目标数组
    """
    X, y = [], []

    for i in range(lookback, len(data) - forecast_horizon):
        # 使用过去lookback个时间点的特征
        X.append(data[feature_cols].iloc[i-lookback:i].values)
        # 预测forecast_horizon个时间点后的油温
        y.append(data[target_col].iloc[i + forecast_horizon])

    return np.array(X), np.array(y)

def prepare_datasets(config_type='1h'):
    """
    为三种配置准备数据集

    Args:
        config_type: '1h' (1小时), '1d' (1天), '1w' (1周)

    Returns:
        X_train, X_test, y_train, y_test, scaler
    """
    # 加载数据
    data, feature_cols, target_col = load_and_preprocess_data()

    # 根据配置类型设置参数
    if config_type == '1h':
        lookback = 24  # 使用过去24个时间点（6小时）
        forecast_horizon = 4  # 预测4个时间点（1小时）后
    elif config_type == '1d':
        lookback = 48  # 使用过去48个时间点（12小时）
        forecast_horizon = 96  # 预测96个时间点（1天）后
    elif config_type == '1w':
        lookback = 96  # 使用过去96个时间点（1天）
        forecast_horizon = 672  # 预测672个时间点（1周）后
    else:
        raise ValueError("config_type 必须是 '1h', '1d' 或 '1w'")

    print(f"\n配置: {config_type}")
    print(f"回溯时间点数: {lookback}")
    print(f"预测时间点数: {forecast_horizon}")

    # 标准化特征
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[feature_cols])

    # 创建标准化后的数据框
    scaled_data = pd.DataFrame(scaled_features, columns=feature_cols)
    scaled_data[target_col] = data[target_col].values
    scaled_data['date'] = data['date'].values

    # 创建序列
    X, y = create_sequences(scaled_data, feature_cols, target_col, lookback, forecast_horizon)

    print(f"\n序列形状:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    # 时间分组分割（确保训练集和测试集不重叠）
    n_samples = len(X)
    n_groups = n_samples // 1000  # 每1000个样本为一组

    # 创建组索引
    group_indices = np.arange(n_groups)

    # 随机分割组（80%训练，20%测试）
    train_groups, test_groups = train_test_split(group_indices, test_size=0.2, random_state=42)

    # 根据组索引获取样本索引
    train_indices = []
    test_indices = []

    for group in train_groups:
        start_idx = group * 1000
        end_idx = min((group + 1) * 1000, n_samples)
        train_indices.extend(range(start_idx, end_idx))

    for group in test_groups:
        start_idx = group * 1000
        end_idx = min((group + 1) * 1000, n_samples)
        test_indices.extend(range(start_idx, end_idx))

    # 分割数据
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    print(f"\n数据分割:")
    print(f"训练集大小: {X_train.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test, scaler

# 主函数
def main():
    # 测试三种配置
    configs = ['1h', '1d', '1w']

    for config in configs:
        print(f"\n{'='*50}")
        print(f"准备 {config} 配置的数据")
        print('='*50)

        X_train, X_test, y_train, y_test, scaler = prepare_datasets(config)

        # 保存数据
        np.save(f'X_train_{config}.npy', X_train)
        np.save(f'X_test_{config}.npy', X_test)
        np.save(f'y_train_{config}.npy', y_train)
        np.save(f'y_test_{config}.npy', y_test)

        # 保存标准化器
        import joblib
        joblib.dump(scaler, f'scaler_{config}.pkl')

        print(f"数据已保存为: X_train_{config}.npy, X_test_{config}.npy, y_train_{config}.npy, y_test_{config}.npy")
        print(f"标准化器已保存为: scaler_{config}.pkl")

if __name__ == "__main__":
    main()