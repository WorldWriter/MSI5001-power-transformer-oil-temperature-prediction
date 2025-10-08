from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import warnings

warnings.filterwarnings('ignore')


def load_data(config_type: str = '1h') -> Tuple[
    NDArray[np.float_],
    NDArray[np.float_],
    NDArray[np.float_],
    NDArray[np.float_],
]:
    """从磁盘加载指定配置的序列化数据集。

    参数
    ----
    config_type:
        预处理阶段保存的预测窗口标签（例如 ``"1h"``）。

    返回
    ----
    X_train, X_test, y_train, y_test:
        通过 ``numpy.load`` 读取的特征和目标数组。

    副作用
    ------
    依赖已存在的 ``.npy`` 文件；若文件缺失会触发 ``IOError``。
    """
    X_train = np.load(f'X_train_{config_type}.npy')
    X_test = np.load(f'X_test_{config_type}.npy')
    y_train = np.load(f'y_train_{config_type}.npy')
    y_test = np.load(f'y_test_{config_type}.npy')
    return X_train, X_test, y_train, y_test


def evaluate_model(
    y_true: NDArray[np.float_],
    y_pred: NDArray[np.float_],
    model_name: str,
    config_type: str,
) -> Dict[str, float | str]:
    """计算回归指标并返回结果字典。

    参数
    ----
    y_true:
        测试集真实值。
    y_pred:
        模型推理得到的预测值。
    model_name:
        当前评估的模型名称。
    config_type:
        数据集的时间窗口标识。

    返回
    ----
    results:
        包含 MSE、RMSE、MAE、R² 等指标的字典，便于进一步汇总分析。

    副作用
    ------
    将模型名称及关键指标打印到标准输出，便于在命令行中观察训练表现。
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    results = {
        'Model': model_name,
        'Config': config_type,
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'R2': r2
    }

    print(f"{model_name} ({config_type}): RMSE={rmse:.4f}, R²={r2:.4f}")
    return results


def train_models(config_type: str = '1h') -> List[Dict[str, float | str]]:
    """训练传统机器学习模型并返回评估结果。

    参数
    ----
    config_type:
        指定要加载的数据配置标签。

    返回
    ----
    results:
        每个模型评估指标的列表，用于写入 CSV 或后续分析。

    副作用
    ------
    训练期间会在本地保存 ``joblib`` 模型文件，并输出训练进度。
    """
    print(f"\n训练 {config_type} 配置的模型")

    # 加载数据
    X_train, X_test, y_train, y_test = load_data(config_type)

    # 限制训练数据大小以加快速度
    max_samples = 2000
    if len(X_train) > max_samples:
        indices = np.random.choice(len(X_train), max_samples, replace=False)
        X_train_small = X_train[indices]
        y_train_small = y_train[indices]
    else:
        X_train_small = X_train
        y_train_small = y_train

    results = []

    # 1. 线性回归（基线）
    print("训练线性回归...")
    lr = LinearRegression()
    lr.fit(X_train_small, y_train_small)
    y_pred = lr.predict(X_test)
    results.append(evaluate_model(y_test, y_pred, 'Linear Regression', config_type))
    joblib.dump(lr, f'lr_{config_type}.pkl')

    # 2. Ridge回归
    print("训练Ridge回归...")
    ridge = Ridge(alpha=1.0)
    ridge.fit(X_train_small, y_train_small)
    y_pred = ridge.predict(X_test)
    results.append(evaluate_model(y_test, y_pred, 'Ridge Regression', config_type))
    joblib.dump(ridge, f'ridge_{config_type}.pkl')

    # 3. 随机森林
    print("训练随机森林...")
    rf = RandomForestRegressor(n_estimators=50, max_depth=10, random_state=42)
    rf.fit(X_train_small, y_train_small)
    y_pred = rf.predict(X_test)
    results.append(evaluate_model(y_test, y_pred, 'Random Forest', config_type))
    joblib.dump(rf, f'rf_{config_type}.pkl')

    return results


def main() -> None:
    """脚本入口：遍历所有配置并将结果写入 ``simple_ml_results.csv``。"""
    configs = ['1h', '1d', '1w']
    all_results = []

    for config in configs:
        results = train_models(config)
        all_results.extend(results)

    # 保存结果
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('simple_ml_results.csv', index=False)

    print(f"\n{'='*50}")
    print("所有模型训练完成！")
    print("结果已保存到 simple_ml_results.csv")

    # 显示最佳结果
    print(f"\n最佳模型 (按R²分数排序):")
    best_models = results_df.nlargest(5, 'R2')[['Model', 'Config', 'R2', 'RMSE']]
    print(best_models.to_string(index=False))

if __name__ == "__main__":
    main()