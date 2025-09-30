import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
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

def reshape_for_lstm(X, n_timesteps):
    """将数据重塑为LSTM所需的形状"""
    n_samples = X.shape[0]
    n_features = X.shape[1] // n_timesteps
    return X.reshape((n_samples, n_timesteps, n_features))

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

def create_lstm_model(input_shape):
    """创建LSTM模型"""
    model = Sequential([
        LSTM(64, return_sequences=True, input_shape=input_shape),
        Dropout(0.2),
        LSTM(32, return_sequences=True),
        Dropout(0.2),
        LSTM(16),
        Dropout(0.2),
        Dense(8, activation='relu'),
        Dense(1)
    ])
    return model

def create_simple_lstm_model(input_shape):
    """创建简化的LSTM模型"""
    model = Sequential([
        LSTM(32, input_shape=input_shape),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    return model

def create_dense_model(input_shape):
    """创建全连接神经网络"""
    model = Sequential([
        Dense(64, activation='relu', input_shape=(input_shape[0] * input_shape[1],)),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)
    ])
    return model

def train_lstm_models(config_type='1h'):
    """训练LSTM模型"""
    print(f"\n训练 {config_type} 配置的深度学习模型")

    # 加载数据
    X_train, X_test, y_train, y_test = load_data(config_type)

    # 限制数据大小
    max_samples = 2000
    if len(X_train) > max_samples:
        indices = np.random.choice(len(X_train), max_samples, replace=False)
        X_train = X_train[indices]
        y_train = y_train[indices]

    results = []

    # 根据配置类型确定时间步数
    if config_type == '1h':
        n_timesteps = 16
    elif config_type == '1d':
        n_timesteps = 32
    elif config_type == '1w':
        n_timesteps = 64

    # 重塑数据为LSTM格式
    X_train_lstm = reshape_for_lstm(X_train, n_timesteps)
    X_test_lstm = reshape_for_lstm(X_test, n_timesteps)

    # 1. 简单LSTM模型
    print("训练简单LSTM模型...")
    model = create_simple_lstm_model((n_timesteps, 6))
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

    # 早停和学习率调度
    callbacks = [
        EarlyStopping(patience=10, restore_best_weights=True),
        ReduceLROnPlateau(factor=0.5, patience=5, min_lr=0.0001)
    ]

    # 训练模型
    history = model.fit(
        X_train_lstm, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=0
    )

    # 预测和评估
    y_pred = model.predict(X_test_lstm, verbose=0)
    results.append(evaluate_model(y_test, y_pred, 'Simple LSTM', config_type))
    model.save(f'simple_lstm_{config_type}.h5')

    # 2. 全连接神经网络
    print("训练全连接神经网络...")
    dense_model = create_dense_model((n_timesteps, 6))
    dense_model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

    # 重塑数据为2D格式
    X_train_dense = X_train.reshape(X_train.shape[0], -1)
    X_test_dense = X_test.reshape(X_test.shape[0], -1)

    # 训练模型
    history = dense_model.fit(
        X_train_dense, y_train,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        callbacks=callbacks,
        verbose=0
    )

    # 预测和评估
    y_pred = dense_model.predict(X_test_dense, verbose=0)
    results.append(evaluate_model(y_test, y_pred, 'Dense Neural Network', config_type))
    dense_model.save(f'dense_nn_{config_type}.h5')

    return results

def compare_models():
    """比较传统ML和深度学习模型"""
    # 加载传统ML结果
    ml_results = pd.read_csv('simple_ml_results.csv')

    # 训练深度学习模型
    dl_results = []
    configs = ['1h', '1d', '1w']

    for config in configs:
        results = train_lstm_models(config)
        dl_results.extend(results)

    # 合并结果
    dl_df = pd.DataFrame(dl_results)
    comparison = pd.concat([ml_results, dl_df], ignore_index=True)

    # 保存比较结果
    comparison.to_csv('model_comparison.csv', index=False)

    # 显示最佳模型
    print(f"\n{'='*60}")
    print("模型性能比较（按R²分数排序）:")
    print('='*60)
    best_models = comparison.nlargest(10, 'R2')[['Model', 'Config', 'R2', 'RMSE']]
    print(best_models.to_string(index=False))

    return comparison

def main():
    """主函数"""
    # 训练深度学习模型
    dl_results = []
    configs = ['1h', '1d', '1w']

    for config in configs:
        results = train_lstm_models(config)
        dl_results.extend(results)

    # 保存结果
    dl_df = pd.DataFrame(dl_results)
    dl_df.to_csv('deep_learning_results.csv', index=False)

    print(f"\n{'='*50}")
    print("深度学习模型训练完成！")
    print("结果已保存到 deep_learning_results.csv")

    # 显示最佳深度学习模型
    print(f"\n最佳深度学习模型:")
    best_dl = dl_df.nlargest(5, 'R2')[['Model', 'Config', 'R2', 'RMSE']]
    print(best_dl.to_string(index=False))

if __name__ == "__main__":
    main()