import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, GridSearchCV
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

    print(f"\n{model_name} ({config_type}) 性能指标:")
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")

    return results

def train_baseline_model(X_train, y_train, X_test, y_test, config_type):
    """训练基线模型（简单线性回归）"""
    print(f"\n训练基线模型 (Linear Regression) - {config_type}")

    model = LinearRegression()
    model.fit(X_train, y_train)

    # 预测
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # 评估
    train_results = evaluate_model(y_train, y_pred_train, 'Linear Regression (Train)', config_type)
    test_results = evaluate_model(y_test, y_pred_test, 'Linear Regression (Test)', config_type)

    # 保存模型
    joblib.dump(model, f'linear_regression_{config_type}.pkl')

    return train_results, test_results

def train_ridge_model(X_train, y_train, X_test, y_test, config_type):
    """训练Ridge回归模型"""
    print(f"\n训练 Ridge 回归模型 - {config_type}")

    # 超参数调优
    param_grid = {'alpha': [0.1, 1.0, 10.0, 100.0]}
    ridge = Ridge()
    grid_search = GridSearchCV(ridge, param_grid, cv=3, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    # 预测
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    # 评估
    train_results = evaluate_model(y_train, y_pred_train, 'Ridge Regression (Train)', config_type)
    test_results = evaluate_model(y_test, y_pred_test, 'Ridge Regression (Test)', config_type)

    # 保存模型
    joblib.dump(best_model, f'ridge_regression_{config_type}.pkl')

    return train_results, test_results

def train_random_forest_model(X_train, y_train, X_test, y_test, config_type):
    """训练随机森林模型"""
    print(f"\n训练随机森林模型 - {config_type}")

    # 超参数调优
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5]
    }
    rf = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='neg_mean_squared_error')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    # 预测
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)

    # 评估
    train_results = evaluate_model(y_train, y_pred_train, 'Random Forest (Train)', config_type)
    test_results = evaluate_model(y_test, y_pred_test, 'Random Forest (Test)', config_type)

    # 保存模型
    joblib.dump(best_model, f'random_forest_{config_type}.pkl')

    return train_results, test_results

def train_gradient_boosting_model(X_train, y_train, X_test, y_test, config_type):
    """训练梯度提升模型"""
    print(f"\n训练梯度提升模型 - {config_type}")

    # 使用较小的模型以加快训练速度
    gb = GradientBoostingRegressor(
        n_estimators=50,
        max_depth=5,
        learning_rate=0.1,
        random_state=42
    )
    gb.fit(X_train, y_train)

    # 预测
    y_pred_train = gb.predict(X_train)
    y_pred_test = gb.predict(X_test)

    # 评估
    train_results = evaluate_model(y_train, y_pred_train, 'Gradient Boosting (Train)', config_type)
    test_results = evaluate_model(y_test, y_pred_test, 'Gradient Boosting (Test)', config_type)

    # 保存模型
    joblib.dump(gb, f'gradient_boosting_{config_type}.pkl')

    return train_results, test_results

def train_svr_model(X_train, y_train, X_test, y_test, config_type):
    """训练支持向量回归模型"""
    print(f"\n训练 SVR 模型 - {config_type}")

    # 使用较小的子集进行训练（SVR训练较慢）
    subset_size = min(2000, len(X_train))
    indices = np.random.choice(len(X_train), subset_size, replace=False)
    X_train_subset = X_train[indices]
    y_train_subset = y_train[indices]

    # 超参数调优
    param_grid = {
        'C': [1.0, 10.0],
        'gamma': ['scale', 'auto']
    }
    svr = SVR(kernel='rbf')
    grid_search = GridSearchCV(svr, param_grid, cv=3, scoring='neg_mean_squared_error')
    grid_search.fit(X_train_subset, y_train_subset)

    best_model = grid_search.best_estimator_

    # 预测
    y_pred_train = best_model.predict(X_train_subset)
    y_pred_test = best_model.predict(X_test)

    # 评估
    train_results = evaluate_model(y_train_subset, y_pred_train, 'SVR (Train)', config_type)
    test_results = evaluate_model(y_test, y_pred_test, 'SVR (Test)', config_type)

    # 保存模型
    joblib.dump(best_model, f'svr_{config_type}.pkl')

    return train_results, test_results

def main():
    """主函数：训练所有模型"""
    configs = ['1h', '1d', '1w']
    all_results = []

    for config in configs:
        print(f"\n{'='*60}")
        print(f"训练 {config} 配置的模型")
        print('='*60)

        # 加载数据
        X_train, X_test, y_train, y_test = load_data(config)

        # 训练各种模型
        models = [
            train_baseline_model,
            train_ridge_model,
            train_random_forest_model,
            train_gradient_boosting_model,
            train_svr_model
        ]

        for model_func in models:
            try:
                train_results, test_results = model_func(X_train, y_train, X_test, y_test, config)
                all_results.append(train_results)
                all_results.append(test_results)
            except Exception as e:
                print(f"模型训练失败: {e}")

    # 保存结果
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('traditional_ml_results.csv', index=False)
    print(f"\n所有结果已保存到 traditional_ml_results.csv")

    # 显示最佳模型
    print(f"\n最佳模型 (按测试集R²分数):")
    test_results = results_df[results_df['Model'].str.contains('Test')]
    best_models = test_results.nlargest(5, 'R2')[['Model', 'Config', 'R2', 'RMSE']]
    print(best_models.to_string(index=False))

if __name__ == "__main__":
    main()