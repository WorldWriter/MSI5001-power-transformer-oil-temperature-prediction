"""
共享工具模块 / Shared Utility Module

用于电力变压器油温预测项目的通用函数
Common functions for power transformer oil temperature prediction project
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

# ============================================================================
# 配置 / Configuration
# ============================================================================

DEFAULT_CONFIG = {
    # 数据配置 / Data Configuration
    'dataset_path': '../dataset/train1.csv',
    'prediction_horizon': 'hour',  # 'hour' (4 offsets) / 'day' (96 offsets) / 'week' (672 offsets)

    # 分割方式 / Split Method
    'split_method': 'sequential',  # 'sequential' / 'random' / 'label_random'
    'train_ratio': 0.8,
    'n_groups': 20,  # 仅用于 random 方式 / Only for 'random' method

    # 特征选择 / Feature Selection
    'load_features': ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL'],  # 可选任意组合
    'time_features': [],  # 可选: ['hour', 'day', 'month', 'dayofweek', 'is_weekend', 'minute']

    # 异常值处理 / Outlier Handling
    'remove_outliers': False,  # 默认不剔除
    'outlier_method': 'iqr',   # 'iqr' or 'zscore'
    'outlier_threshold': 3.0,  # IQR倍数或Z-score阈值

    # 模型超参数 / Model Hyperparameters
    'seq_length': 16,
    'hidden_sizes': [64, 32],  # For Linear Regression
    'dropout': 0.2,
    'batch_size': 32,
    'num_epochs': 100,
    'learning_rate': 0.001,
    'patience': 10,  # Early stopping patience

    # 其他 / Others
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# 预测时间范围配置 / Prediction Horizon Configurations
HORIZON_CONFIGS = {
    'hour': {'offset': 4, 'seq_length': 16, 'description': '1小时预测 (4个时间点)'},
    'day': {'offset': 96, 'seq_length': 16, 'description': '1天预测 (96个时间点)'},
    'week': {'offset': 672, 'seq_length': 16, 'description': '1周预测 (672个时间点)'}
}


# ============================================================================
# 数据处理函数 / Data Processing Functions
# ============================================================================

def load_data(filepath):
    """
    加载数据集 / Load dataset

    Parameters:
    -----------
    filepath : str
        数据文件路径

    Returns:
    --------
    df : pd.DataFrame
        加载的数据，时间列作为索引
    """
    df = pd.read_csv(filepath)

    # 检测时间列 / Detect datetime column
    date_col = None
    if 'date' in df.columns:
        date_col = 'date'
    elif df.columns[0].lower() in ['date', 'datetime', 'time', 'timestamp']:
        date_col = df.columns[0]

    if date_col:
        df[date_col] = pd.to_datetime(df[date_col])
        df = df.set_index(date_col)

    return df


def detect_outliers(df, method='iqr', threshold=3.0, columns=None):
    """
    检测异常值 / Detect outliers

    Parameters:
    -----------
    df : pd.DataFrame
    method : str ('iqr' or 'zscore')
    threshold : float (IQR倍数或Z-score阈值)
    columns : list (要检测的列，None表示所有数值列)

    Returns:
    --------
    outlier_mask : np.ndarray (True表示异常值)
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()

    outlier_mask = np.zeros(len(df), dtype=bool)

    if method == 'iqr':
        for col in columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outlier_mask |= (df[col] < lower_bound) | (df[col] > upper_bound)

    elif method == 'zscore':
        for col in columns:
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            outlier_mask |= z_scores > threshold

    return outlier_mask


def remove_outliers(df, method='iqr', threshold=3.0, columns=None):
    """
    移除异常值行 / Remove outlier rows

    Returns:
    --------
    df_cleaned : pd.DataFrame
    n_removed : int (移除的行数)
    """
    outlier_mask = detect_outliers(df, method, threshold, columns)
    df_cleaned = df[~outlier_mask].reset_index(drop=True)
    n_removed = outlier_mask.sum()
    return df_cleaned, n_removed


def extract_time_features(df, features_list):
    """
    从索引（datetime）提取时间特征 / Extract time features from datetime index

    Parameters:
    -----------
    df : pd.DataFrame (索引必须是datetime类型)
    features_list : list (['hour', 'day', 'month', 'dayofweek', 'is_weekend', 'minute'])

    Returns:
    --------
    time_features_df : pd.DataFrame (时间特征)
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame索引必须是DatetimeIndex类型")

    time_features = {}

    if 'hour' in features_list:
        time_features['hour'] = df.index.hour
    if 'day' in features_list:
        time_features['day'] = df.index.day
    if 'month' in features_list:
        time_features['month'] = df.index.month
    if 'dayofweek' in features_list:
        time_features['dayofweek'] = df.index.dayofweek  # 0=Monday, 6=Sunday
    if 'is_weekend' in features_list:
        time_features['is_weekend'] = (df.index.dayofweek >= 5).astype(int)
    if 'minute' in features_list:
        time_features['minute'] = df.index.minute

    return pd.DataFrame(time_features, index=df.index)


def create_sequences_with_offset(X, y, seq_length=16, offset=4):
    """
    创建带偏移的时间序列预测序列 / Create sequences with offset for prediction

    Parameters:
    -----------
    X : np.ndarray (特征矩阵)
    y : np.ndarray (目标向量)
    seq_length : int (序列长度)
    offset : int (预测偏移量，从最后输入点到预测点的距离)

    Returns:
    --------
    X_seq : np.ndarray, shape (n_samples, seq_length, n_features)
    y_seq : np.ndarray, shape (n_samples,)

    说明：每个样本使用 [t-offset-seq_length : t-offset] 的特征预测 t 时刻的目标
    """
    X_seq, y_seq = [], []
    lookback = offset + seq_length

    for i in range(lookback, len(X)):
        # 输入窗口：从 i-lookback 到 i-offset
        start_idx = i - lookback
        end_idx = i - offset
        X_seq.append(X[start_idx:end_idx])
        y_seq.append(y[i])

    return np.array(X_seq), np.array(y_seq)


# ============================================================================
# 数据分割函数 / Data Splitting Functions
# ============================================================================

def split_sequential(X_seq, y_seq, train_ratio=0.8):
    """
    方式a: 前80%连续时间用于训练，后20%用于测试
    Method a: First 80% continuous time for training, last 20% for testing
    """
    split_idx = int(len(X_seq) * train_ratio)
    X_train = X_seq[:split_idx]
    X_test = X_seq[split_idx:]
    y_train = y_seq[:split_idx]
    y_test = y_seq[split_idx:]
    return X_train, X_test, y_train, y_test


def split_random_groups(X_seq, y_seq, n_groups=20, train_ratio=0.8, random_seed=42):
    """
    方式b: 分组后随机分配，组内时间连续
    Method b: Divide into groups, randomly assign groups to train/test
    """
    n_samples = len(X_seq)
    group_size = n_samples // n_groups

    # 创建组索引
    groups = []
    for i in range(n_groups):
        start = i * group_size
        end = start + group_size if i < n_groups - 1 else n_samples
        groups.append(np.arange(start, end))

    # 随机分配组
    np.random.seed(random_seed)
    np.random.shuffle(groups)
    n_train_groups = int(n_groups * train_ratio)

    train_indices = np.concatenate(groups[:n_train_groups])
    test_indices = np.concatenate(groups[n_train_groups:])

    X_train = X_seq[train_indices]
    X_test = X_seq[test_indices]
    y_train = y_seq[train_indices]
    y_test = y_seq[test_indices]

    return X_train, X_test, y_train, y_test


def split_label_random(X_seq, y_seq, train_ratio=0.8, random_seed=42):
    """
    方式c: 完全随机抽样（可能存在数据泄露：训练/测试窗口重叠）
    Method c: Completely random sampling (may have data leakage: overlapping windows)
    """
    n_samples = len(y_seq)
    indices = np.arange(n_samples)
    np.random.seed(random_seed)
    np.random.shuffle(indices)

    split_idx = int(n_samples * train_ratio)
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    X_train = X_seq[train_indices]
    X_test = X_seq[test_indices]
    y_train = y_seq[train_indices]
    y_test = y_seq[test_indices]

    return X_train, X_test, y_train, y_test


# ============================================================================
# PyTorch工具函数 / PyTorch Utility Functions
# ============================================================================

def create_dataloaders(X_train, y_train, X_test, y_test, batch_size=32):
    """
    创建PyTorch DataLoader / Create PyTorch DataLoaders

    Parameters:
    -----------
    X_train, y_train : np.ndarray
        训练数据
    X_test, y_test : np.ndarray
        测试数据
    batch_size : int

    Returns:
    --------
    train_loader, test_loader : DataLoader
    """
    # 转换为Tensor
    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
    X_test_t = torch.FloatTensor(X_test)
    y_test_t = torch.FloatTensor(y_test).unsqueeze(1)

    # 创建Dataset
    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


# ============================================================================
# 可视化函数 / Visualization Functions
# ============================================================================

def plot_training_history(history, title='Training History'):
    """
    绘制训练历史 / Plot training history

    Parameters:
    -----------
    history : dict
        包含 'train_loss', 'test_loss', 'train_mae', 'test_mae' 的字典
    title : str
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss', alpha=0.8)
    axes[0].plot(history['test_loss'], label='Test Loss', alpha=0.8)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss (MSE)')
    axes[0].set_title(f'{title} - Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # MAE
    axes[1].plot(history['train_mae'], label='Train MAE', alpha=0.8)
    axes[1].plot(history['test_mae'], label='Test MAE', alpha=0.8)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].set_title(f'{title} - MAE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_predictions(y_true, y_pred, title='Predictions vs Actual', n_samples=500):
    """
    绘制预测结果 / Plot predictions vs actual

    Parameters:
    -----------
    y_true : np.ndarray
        真实值
    y_pred : np.ndarray
        预测值
    title : str
    n_samples : int
        时间序列图显示的样本数
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 时间序列对比（只显示前n_samples个样本）
    n = min(n_samples, len(y_true))
    axes[0].plot(y_true[:n], label='Actual', alpha=0.7, linewidth=1)
    axes[0].plot(y_pred[:n], label='Predicted', alpha=0.7, linewidth=1)
    axes[0].set_xlabel('Sample')
    axes[0].set_ylabel('Oil Temperature (OT)')
    axes[0].set_title(f'{title} - Time Series')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 散点图
    axes[1].scatter(y_true, y_pred, alpha=0.5, s=10)
    axes[1].plot([y_true.min(), y_true.max()],
                 [y_true.min(), y_true.max()],
                 'r--', linewidth=2, label='Perfect Prediction')
    axes[1].set_xlabel('Actual OT')
    axes[1].set_ylabel('Predicted OT')
    axes[1].set_title(f'{title} - Scatter Plot')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_comparison_summary(comparison_results):
    """
    绘制多个模型的对比结果 / Plot comparison summary for multiple models

    Parameters:
    -----------
    comparison_results : dict
        键为模型名称，值为包含'metrics'的结果字典
    """
    # 提取指标
    model_names = list(comparison_results.keys())
    r2_scores = [comparison_results[name]['metrics']['r2'] for name in model_names]
    mse_scores = [comparison_results[name]['metrics']['mse'] for name in model_names]
    mae_scores = [comparison_results[name]['metrics']['mae'] for name in model_names]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # R² Score
    axes[0].bar(model_names, r2_scores, alpha=0.7, color='skyblue')
    axes[0].set_ylabel('R² Score')
    axes[0].set_title('R² Score Comparison')
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].tick_params(axis='x', rotation=45)

    # MSE
    axes[1].bar(model_names, mse_scores, alpha=0.7, color='lightcoral')
    axes[1].set_ylabel('MSE')
    axes[1].set_title('MSE Comparison')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].tick_params(axis='x', rotation=45)

    # MAE
    axes[2].bar(model_names, mae_scores, alpha=0.7, color='lightgreen')
    axes[2].set_ylabel('MAE')
    axes[2].set_title('MAE Comparison')
    axes[2].grid(True, alpha=0.3, axis='y')
    axes[2].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.show()

    # 打印数值表格
    print(f"\n{'='*80}")
    print("对比结果汇总 / Comparison Summary")
    print(f"{'='*80}")
    print(f"{'Model':<20} {'R²':<15} {'MSE':<15} {'MAE':<15}")
    print(f"{'-'*80}")
    for name in model_names:
        metrics = comparison_results[name]['metrics']
        print(f"{name:<20} {metrics['r2']:<15.6f} {metrics['mse']:<15.6f} {metrics['mae']:<15.6f}")
    print(f"{'='*80}\n")
