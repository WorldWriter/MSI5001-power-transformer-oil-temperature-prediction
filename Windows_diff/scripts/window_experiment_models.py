#!/usr/bin/env python3
"""
时间窗口长度实验 - 模型训练脚本

本脚本在不同时间窗口配置下训练多种机器学习模型，
用于分析历史窗口长度对预测性能的影响。

作者: MSI5001项目组
日期: 2024年
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import os
import time
import warnings
warnings.filterwarnings('ignore')

def load_experiment_data(config_name):
    """加载特定配置的实验数据"""
    data_dir = f"../results/{config_name}"
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"配置 {config_name} 的数据目录不存在: {data_dir}")
    
    X_train = np.load(f"{data_dir}/X_train.npy")
    X_test = np.load(f"{data_dir}/X_test.npy")
    y_train = np.load(f"{data_dir}/y_train.npy")
    y_test = np.load(f"{data_dir}/y_test.npy")
    
    return X_train, X_test, y_train, y_test

def train_random_forest(X_train, y_train, X_test, y_test):
    """训练随机森林模型"""
    print("    训练 Random Forest...")
    start_time = time.time()
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    training_time = time.time() - start_time
    
    # 计算评估指标
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    return {
        'model_type': 'RandomForest',
        'r2_score': r2,
        'rmse': rmse,
        'mae': mae,
        'training_time': training_time,
        'model': model
    }

def train_ridge_regression(X_train, y_train, X_test, y_test):
    """训练Ridge回归模型"""
    print("    训练 Ridge Regression...")
    start_time = time.time()
    
    model = Ridge(alpha=1.0, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    training_time = time.time() - start_time
    
    # 计算评估指标
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    return {
        'model_type': 'Ridge',
        'r2_score': r2,
        'rmse': rmse,
        'mae': mae,
        'training_time': training_time,
        'model': model
    }

def train_mlp(X_train, y_train, X_test, y_test):
    """训练MLP神经网络模型"""
    print("    训练 MLP...")
    start_time = time.time()
    
    model = MLPRegressor(
        hidden_layer_sizes=(100, 50),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    training_time = time.time() - start_time
    
    # 计算评估指标
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    return {
        'model_type': 'MLP',
        'r2_score': r2,
        'rmse': rmse,
        'mae': mae,
        'training_time': training_time,
        'model': model
    }

def train_models_for_config(config_name, config_info):
    """为特定配置训练所有模型"""
    print(f"\n=== 训练配置: {config_name} ===")
    print(f"历史窗口: {config_info['history_hours']:.1f} 小时")
    print(f"预测跨度: {config_info['forecast_hours']:.1f} 小时")
    
    try:
        # 加载数据
        X_train, X_test, y_train, y_test = load_experiment_data(config_name)
        print(f"数据加载成功: 训练集 {X_train.shape}, 测试集 {X_test.shape}")
        
        # 训练所有模型
        results = []
        
        # Random Forest
        rf_result = train_random_forest(X_train, y_train, X_test, y_test)
        rf_result['config'] = config_name
        rf_result['lookback'] = config_info['lookback']
        rf_result['forecast_horizon'] = config_info['forecast_horizon']
        rf_result['history_hours'] = config_info['history_hours']
        rf_result['forecast_hours'] = config_info['forecast_hours']
        results.append(rf_result)
        
        # Ridge Regression
        ridge_result = train_ridge_regression(X_train, y_train, X_test, y_test)
        ridge_result['config'] = config_name
        ridge_result['lookback'] = config_info['lookback']
        ridge_result['forecast_horizon'] = config_info['forecast_horizon']
        ridge_result['history_hours'] = config_info['history_hours']
        ridge_result['forecast_hours'] = config_info['forecast_hours']
        results.append(ridge_result)
        
        # MLP
        mlp_result = train_mlp(X_train, y_train, X_test, y_test)
        mlp_result['config'] = config_name
        mlp_result['lookback'] = config_info['lookback']
        mlp_result['forecast_horizon'] = config_info['forecast_horizon']
        mlp_result['history_hours'] = config_info['history_hours']
        mlp_result['forecast_hours'] = config_info['forecast_hours']
        results.append(mlp_result)
        
        # 保存模型
        model_dir = f"../models/{config_name}"
        os.makedirs(model_dir, exist_ok=True)
        
        joblib.dump(rf_result['model'], f"{model_dir}/random_forest.pkl")
        joblib.dump(ridge_result['model'], f"{model_dir}/ridge.pkl")
        joblib.dump(mlp_result['model'], f"{model_dir}/mlp.pkl")
        
        print(f"✓ 模型已保存到: {model_dir}/")
        
        # 显示结果摘要
        print(f"\n结果摘要:")
        for result in results:
            print(f"  {result['model_type']:12} - R²: {result['r2_score']:.4f}, RMSE: {result['rmse']:.4f}, 训练时间: {result['training_time']:.2f}s")
        
        return results
        
    except Exception as e:
        print(f"❌ 配置 {config_name} 训练失败: {str(e)}")
        return []

def main():
    """主函数 - 训练所有配置的模型"""
    print("=== 时间窗口长度实验 - 模型训练 ===")
    print("在不同历史窗口长度配置下训练机器学习模型\n")
    
    # 读取配置摘要
    config_summary_path = "../results/experiment_config_summary.csv"
    if not os.path.exists(config_summary_path):
        print("❌ 未找到配置摘要文件，请先运行 window_experiment_preprocessing.py")
        return
    
    config_df = pd.read_csv(config_summary_path)
    successful_configs = config_df[config_df['status'] == 'success']
    
    print(f"发现 {len(successful_configs)} 个成功的配置，开始训练...")
    
    # 创建模型目录
    os.makedirs("../models", exist_ok=True)
    
    # 收集所有结果
    all_results = []
    
    # 为每个配置训练模型
    for _, config_row in successful_configs.iterrows():
        config_name = config_row['config_name']
        config_info = {
            'lookback': config_row['lookback'],
            'forecast_horizon': config_row['forecast_horizon'],
            'history_hours': config_row['lookback'] / 4,
            'forecast_hours': config_row['forecast_horizon'] / 4
        }
        
        results = train_models_for_config(config_name, config_info)
        all_results.extend(results)
    
    # 保存所有结果
    if all_results:
        results_df = pd.DataFrame([
            {
                'config': r['config'],
                'model_type': r['model_type'],
                'lookback': r['lookback'],
                'forecast_horizon': r['forecast_horizon'],
                'history_hours': r['history_hours'],
                'forecast_hours': r['forecast_hours'],
                'r2_score': r['r2_score'],
                'rmse': r['rmse'],
                'mae': r['mae'],
                'training_time': r['training_time']
            }
            for r in all_results
        ])
        
        results_df.to_csv("../results/model_training_results.csv", index=False)
        
        print(f"\n{'='*60}")
        print("✓ 模型训练完成！")
        print(f"✓ 成功训练配置数: {len(successful_configs)}")
        print(f"✓ 总模型数: {len(all_results)}")
        print(f"✓ 结果已保存: ../results/model_training_results.csv")
        
        # 显示最佳结果
        print(f"\n各模型类型的最佳R²分数:")
        for model_type in results_df['model_type'].unique():
            model_results = results_df[results_df['model_type'] == model_type]
            best_result = model_results.loc[model_results['r2_score'].idxmax()]
            print(f"  {model_type:12} - R²: {best_result['r2_score']:.4f} (配置: {best_result['config']})")
        
        print(f"\n下一步:")
        print("运行 window_experiment_analysis.py 进行详细分析")
    
    else:
        print("❌ 没有成功训练任何模型")

if __name__ == "__main__":
    main()