import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_data(config_type='1h'):
    """加载预处理的数据"""
    X_train = np.load(f'X_train_{config_type}.npy')
    X_test = np.load(f'X_test_{config_type}.npy')
    y_train = np.load(f'y_train_{config_type}.npy')
    y_test = np.load(f'y_test_{config_type}.npy')
    return X_train, X_test, y_train, y_test

def evaluate_model(y_true, y_pred, model_name, config_type):
    """评估模型性能"""
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

def train_models(config_type='1h'):
    """训练简化版模型"""
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

def main():
    """主函数"""
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