from typing import List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import warnings

warnings.filterwarnings('ignore')


def load_and_preprocess_data(sample_size: int = 50000) -> Tuple[pd.DataFrame, List[str], str]:
    """加载并清洗原始变压器数据集。

    参数
    ----
    sample_size:
        限制合并后数据集的行数，以避免在本地实验中占用过多内存。

    返回
    ----
    data:
        经过时间排序的数据框，包含原始特征和目标列。
    feature_cols:
        用于建模的特征列名称列表。
    target_col:
        目标列名称 ``"OT"``。

    副作用
    ------
    将数据规模和时间范围打印到标准输出，方便人工检查数据加载是否成功。
    """
    # 加载数据
    trans1 = pd.read_csv('trans_1.csv')
    trans2 = pd.read_csv('trans_2.csv')

    # 转换日期列
    trans1['date'] = pd.to_datetime(trans1['date'])
    trans2['date'] = pd.to_datetime(trans2['date'])

    # 合并数据
    combined_data = pd.concat([trans1, trans2], ignore_index=True)
    combined_data = combined_data.sort_values('date').reset_index(drop=True)

    # 如果数据太大，进行采样
    if len(combined_data) > sample_size:
        combined_data = combined_data.iloc[:sample_size]

    print(f"数据形状: {combined_data.shape}")
    print(f"时间范围: {combined_data['date'].min()} 到 {combined_data['date'].max()}")

    # 特征列
    feature_cols = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']
    target_col = 'OT'

    return combined_data, feature_cols, target_col


def create_sequences(
    data: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    lookback: int,
    forecast_horizon: int,
    max_samples: int = 10000,
) -> Tuple[NDArray[np.float_], NDArray[np.float_]]:
    """将连续时间点转换为监督学习样本。

    参数
    ----
    data:
        已完成标准化的时间序列数据，必须包含 ``feature_cols`` 和 ``target_col``。
    feature_cols:
        需要回溯的特征列集合。
    target_col:
        预测目标列名称。
    lookback:
        输入序列包含的历史时间步数。
    forecast_horizon:
        目标需要提前预测的时间步数。
    max_samples:
        限制生成的序列数量，以便在资源受限的环境中运行。

    返回
    ----
    X:
        形状为 ``(n_samples, lookback * len(feature_cols))`` 的二维特征数组。
    y:
        形状为 ``(n_samples,)`` 的目标数组。

    副作用
    ------
    无副作用，仅在内存中构造数组。
    """
    X, y = [], []

    # 限制处理的数据量
    max_start_idx = min(len(data) - lookback - forecast_horizon, max_samples)

    for i in range(lookback, lookback + max_start_idx):
        # 使用过去lookback个时间点的特征
        X.append(data[feature_cols].iloc[i-lookback:i].values.flatten())  # 展平为2D数组
        # 预测forecast_horizon个时间点后的油温
        y.append(data[target_col].iloc[i + forecast_horizon])

    return np.array(X), np.array(y)


def prepare_datasets(config_type: str = '1h') -> Tuple[
    NDArray[np.float_],
    NDArray[np.float_],
    NDArray[np.float_],
    NDArray[np.float_],
    StandardScaler,
]:
    """根据预测窗口配置生成模型训练/验证数据。

    参数
    ----
    config_type:
        预测粒度，可选 ``"1h"``、``"1d"`` 或 ``"1w"``，分别表示 1 小时、1 天、1 周的预测任务。

    返回
    ----
    X_train, X_test, y_train, y_test:
        分割后的特征与标签数组。
    scaler:
        已拟合的 ``StandardScaler``，用于在推理阶段还原或复用标准化参数。

    副作用
    ------
    将当前配置的参数、序列形状及数据划分信息打印到标准输出，辅助调试。
    """
    # 加载数据
    data, feature_cols, target_col = load_and_preprocess_data()

    # 根据配置类型设置参数
    if config_type == '1h':
        lookback = 16  # 使用过去16个时间点（4小时）
        forecast_horizon = 4  # 预测4个时间点（1小时）后
    elif config_type == '1d':
        lookback = 32  # 使用过去32个时间点（8小时）
        forecast_horizon = 96  # 预测96个时间点（1天）后
    elif config_type == '1w':
        lookback = 64  # 使用过去64个时间点（16小时）
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

    # 创建序列（限制样本数量）
    X, y = create_sequences(scaled_data, feature_cols, target_col, lookback, forecast_horizon)

    print(f"\n序列形状:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")

    # 简单的训练测试分割（80%-20%）
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    print(f"\n数据分割:")
    print(f"训练集大小: {X_train.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")

    return X_train, X_test, y_train, y_test, scaler


# 主函数
def main() -> None:
    """脚本入口：依次处理不同预测配置并持久化结果文件。

    副作用
    ------
    读取磁盘上的原始 ``trans_*.csv`` 文件，写入 ``.npy`` 和 ``.pkl`` 中间件，以供后续模型训练使用。
    """
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
        joblib.dump(scaler, f'scaler_{config}.pkl')

        print(f"数据已保存")

if __name__ == "__main__":
    main()