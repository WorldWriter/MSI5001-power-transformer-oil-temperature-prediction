from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor

import warnings

warnings.filterwarnings('ignore')


def load_data(config_type: str = '1h') -> Tuple[
    NDArray[np.float_],
    NDArray[np.float_],
    NDArray[np.float_],
    NDArray[np.float_],
]:
    """加载指定预测窗口的预处理数据。

    参数
    ----
    config_type:
        与预处理输出一致的配置名称。

    返回
    ----
    X_train, X_test, y_train, y_test:
        用于训练和评估的 ``numpy`` 数组。

    副作用
    ------
    读取 ``.npy`` 文件，若文件不存在将抛出 ``FileNotFoundError``。
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
    """计算并记录神经网络在测试集上的指标。

    参数
    ----
    y_true:
        真实的目标序列。
    y_pred:
        模型推理后的预测值。
    model_name:
        评估的模型名称，用于日志与结果表。
    config_type:
        使用的数据配置标签。

    返回
    ----
    results:
        记录 MSE、RMSE、MAE、R² 等指标的字典。

    副作用
    ------
    将指标打印到标准输出，便于实时监控训练质量。
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


def train_mlp_models(config_type: str = '1h') -> List[Dict[str, float | str]]:
    """针对给定配置训练不同规模的 MLP 模型。

    参数
    ----
    config_type:
        选择要加载的预处理数据配置。

    返回
    ----
    results:
        不同网络结构的评估结果集合。

    副作用
    ------
    在本地保存训练好的 MLP ``joblib`` 模型文件，并输出训练进度信息。
    """
    print(f"\n训练 {config_type} 配置的MLP模型")

    # 加载数据
    X_train, X_test, y_train, y_test = load_data(config_type)

    # 限制数据大小
    max_samples = 2000
    if len(X_train) > max_samples:
        indices = np.random.choice(len(X_train), max_samples, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]

    results = []

    # 1. 小型MLP
    print("训练小型MLP...")
    mlp_small = MLPRegressor(
        hidden_layer_sizes=(50, 25),
        activation='relu',
        solver='adam',
        max_iter=200,
        random_state=42
    )
    mlp_small.fit(X_train, y_train)
    y_pred = mlp_small.predict(X_test)
    results.append(evaluate_model(y_test, y_pred, 'MLP Small', config_type))
    joblib.dump(mlp_small, f'mlp_small_{config_type}.pkl')

    # 2. 中型MLP
    print("训练中型MLP...")
    mlp_medium = MLPRegressor(
        hidden_layer_sizes=(100, 50, 25),
        activation='relu',
        solver='adam',
        max_iter=200,
        random_state=42
    )
    mlp_medium.fit(X_train, y_train)
    y_pred = mlp_medium.predict(X_test)
    results.append(evaluate_model(y_test, y_pred, 'MLP Medium', config_type))
    joblib.dump(mlp_medium, f'mlp_medium_{config_type}.pkl')

    # 3. 大型MLP
    print("训练大型MLP...")
    mlp_large = MLPRegressor(
        hidden_layer_sizes=(200, 100, 50, 25),
        activation='relu',
        solver='adam',
        max_iter=200,
        random_state=42
    )
    mlp_large.fit(X_train, y_train)
    y_pred = mlp_large.predict(X_test)
    results.append(evaluate_model(y_test, y_pred, 'MLP Large', config_type))
    joblib.dump(mlp_large, f'mlp_large_{config_type}.pkl')

    return results


def final_model_comparison() -> pd.DataFrame:
    """训练深度模型并与传统模型结果合并。

    返回
    ----
    all_results:
        包含传统模型与 MLP 模型性能的综合数据框。

    副作用
    ------
    读取和写入多份 CSV 文件，便于后续分析或可视化。
    """
    # 加载所有结果
    ml_results = pd.read_csv('simple_ml_results.csv')

    # 训练MLP模型
    dl_results = []
    configs = ['1h', '1d', '1w']

    for config in configs:
        results = train_mlp_models(config)
        dl_results.extend(results)

    # 合并结果
    dl_df = pd.DataFrame(dl_results)

    # 保存深度学习结果
    dl_df.to_csv('mlp_results.csv', index=False)

    # 合并所有结果
    all_results = pd.concat([ml_results, dl_df], ignore_index=True)
    all_results.to_csv('final_model_comparison.csv', index=False)

    # 显示最佳模型
    print(f"\n{'='*60}")
    print("最终模型性能比较（按R²分数排序）:")
    print('='*60)
    best_models = all_results.nlargest(10, 'R2')[['Model', 'Config', 'R2', 'RMSE', 'MAE']]
    print(best_models.to_string(index=False))

    # 分析结果
    analyze_results(all_results)

    return all_results


def analyze_results(results_df: pd.DataFrame) -> None:
    """打印各配置及整体的性能分析摘要。

    参数
    ----
    results_df:
        模型评估指标的合并数据框。

    副作用
    ------
    将分析结果打印到标准输出。
    """
    print(f"\n{'='*60}")
    print("结果分析:")
    print('='*60)

    # 按配置分组分析
    for config in ['1h', '1d', '1w']:
        config_results = results_df[results_df['Config'] == config]
        best_model = config_results.loc[config_results['R2'].idxmax()]

        print(f"\n{config} 配置:")
        print(f"  最佳模型: {best_model['Model']}")
        print(f"  R²分数: {best_model['R2']:.4f}")
        print(f"  RMSE: {best_model['RMSE']:.4f}")
        print(f"  MAE: {best_model['MAE']:.4f}")

    # 整体分析
    print(f"\n整体最佳模型:")
    overall_best = results_df.loc[results_df['R2'].idxmax()]
    print(f"  模型: {overall_best['Model']} ({overall_best['Config']})")
    print(f"  R²分数: {overall_best['R2']:.4f}")
    print(f"  RMSE: {overall_best['RMSE']:.4f}")

    # 模型类型比较
    print(f"\n按模型类型分析:")
    model_types = ['Linear Regression', 'Ridge Regression', 'Random Forest', 'MLP Small', 'MLP Medium', 'MLP Large']

    for model_type in model_types:
        type_results = results_df[results_df['Model'] == model_type]
        if len(type_results) > 0:
            avg_r2 = type_results['R2'].mean()
            print(f"  {model_type}: 平均R² = {avg_r2:.4f}")

def main() -> None:
    """脚本入口：执行最终模型比较并输出分析信息。"""
    # 运行最终比较
    final_results = final_model_comparison()

    print(f"\n{'='*60}")
    print("分析完成！所有结果已保存到 final_model_comparison.csv")
    print('='*60)

if __name__ == "__main__":
    main()