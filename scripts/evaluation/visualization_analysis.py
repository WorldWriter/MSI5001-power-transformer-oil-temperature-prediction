from __future__ import annotations

import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score

warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

ARTIFACTS = Path('artifacts')
MODELS = ARTIFACTS / 'models'


def load_final_results(path: Path | None = None) -> pd.DataFrame:
    """加载最终结果"""
    path = path or (ARTIFACTS / 'final_model_comparison.csv')
    return pd.read_csv(path)


def load_array(name: str) -> np.ndarray:
    array = np.load(ARTIFACTS / name)
    if array.ndim == 3:
        array = array.reshape(array.shape[0], -1)
    return array


def load_model(name: str):
    return joblib.load(MODELS / name)

def plot_model_performance(results_df):
    """绘制模型性能对比图"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('模型性能对比分析', fontsize=16)

    # 1. R²分数对比
    ax1 = axes[0, 0]
    pivot_r2 = results_df.pivot_table(values='R2', index='Model', columns='Config')
    sns.heatmap(pivot_r2, annot=True, cmap='RdYlGn', center=0, ax=ax1, fmt='.3f')
    ax1.set_title('R²分数热力图')
    ax1.set_xlabel('配置类型')
    ax1.set_ylabel('模型类型')

    # 2. RMSE对比
    ax2 = axes[0, 1]
    pivot_rmse = results_df.pivot_table(values='RMSE', index='Model', columns='Config')
    sns.heatmap(pivot_rmse, annot=True, cmap='RdYlBu_r', ax=ax2, fmt='.2f')
    ax2.set_title('RMSE热力图')
    ax2.set_xlabel('配置类型')
    ax2.set_ylabel('模型类型')

    # 3. 按配置分组的R²分数
    ax3 = axes[1, 0]
    configs = ['1h', '1d', '1w']
    x = np.arange(len(configs))
    width = 0.15

    models = [
        'RandomForestRegressor',
        'MLPRegressor',
        'Ridge',
        'LinearRegression',
    ]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']

    for i, model in enumerate(models):
        model_data = results_df[results_df['Model'] == model]
        r2_scores = [
            model_data[model_data['Config'] == config]['R2'].values[0]
            if len(model_data[model_data['Config'] == config]) > 0
            else 0
            for config in configs
        ]
        ax3.bar(x + i * width, r2_scores, width, label=model, color=colors[i])

    ax3.set_xlabel('配置类型')
    ax3.set_ylabel('R²分数')
    ax3.set_title('不同配置的R²分数对比')
    ax3.set_xticks(x + width * 1.5)
    ax3.set_xticklabels(configs)
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. 模型复杂度vs性能
    ax4 = axes[1, 1]

    # 定义模型复杂度（主观评分）
    complexity_map = {
        'LinearRegression': 1,
        'Ridge': 2,
        'RandomForestRegressor': 4,
        'MLPRegressor': 5,
    }

    for config in configs:
        config_data = results_df[results_df['Config'] == config]
        complexities = [complexity_map.get(model, 3) for model in config_data['Model']]
        ax4.scatter(complexities, config_data['R2'], label=f'{config}配置', s=60, alpha=0.7)

    ax4.set_xlabel('模型复杂度')
    ax4.set_ylabel('R²分数')
    ax4.set_title('模型复杂度vs性能')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('model_performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_prediction_examples():
    """绘制预测示例"""
    # 加载最佳模型（Random Forest 1h）
    from sklearn.ensemble import RandomForestRegressor

    # 加载数据
    X_test_1h = load_array('X_test_1h.npy')
    y_test_1h = np.load(ARTIFACTS / 'y_test_1h.npy')
    X_test_1d = load_array('X_test_1d.npy')
    y_test_1d = np.load(ARTIFACTS / 'y_test_1d.npy')

    # 加载模型
    rf_1h = load_model('random_forest_1h.pkl')
    rf_1d = load_model('random_forest_1d.pkl')

    # 预测
    y_pred_1h = rf_1h.predict(X_test_1h)
    y_pred_1d = rf_1d.predict(X_test_1d)

    # 绘制预测vs实际
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('预测结果 vs 实际值', fontsize=16)

    # 1小时预测
    ax1 = axes[0]
    ax1.scatter(y_test_1h, y_pred_1h, alpha=0.6, s=20)
    ax1.plot([y_test_1h.min(), y_test_1h.max()], [y_test_1h.min(), y_test_1h.max()], 'r--', lw=2)
    ax1.set_xlabel('实际油温 (°C)')
    ax1.set_ylabel('预测油温 (°C)')
    ax1.set_title('1小时预测 (Random Forest)')
    ax1.grid(True, alpha=0.3)

    # 计算R²和RMSE
    r2_1h = r2_score(y_test_1h, y_pred_1h)
    rmse_1h = np.sqrt(mean_squared_error(y_test_1h, y_pred_1h))
    ax1.text(0.05, 0.95, f'R² = {r2_1h:.3f}\nRMSE = {rmse_1h:.3f}',
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # 1天预测
    ax2 = axes[1]
    ax2.scatter(y_test_1d, y_pred_1d, alpha=0.6, s=20)
    ax2.plot([y_test_1d.min(), y_test_1d.max()], [y_test_1d.min(), y_test_1d.max()], 'r--', lw=2)
    ax2.set_xlabel('实际油温 (°C)')
    ax2.set_ylabel('预测油温 (°C)')
    ax2.set_title('1天预测 (Random Forest)')
    ax2.grid(True, alpha=0.3)

    # 计算R²和RMSE
    r2_1d = r2_score(y_test_1d, y_pred_1d)
    rmse_1d = np.sqrt(mean_squared_error(y_test_1d, y_pred_1d))
    ax2.text(0.05, 0.95, f'R² = {r2_1d:.3f}\nRMSE = {rmse_1d:.3f}',
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig('prediction_examples.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_error_distribution():
    """绘制误差分布"""
    from sklearn.ensemble import RandomForestRegressor

    # 加载数据和模型
    X_test_1h = load_array('X_test_1h.npy')
    y_test_1h = np.load(ARTIFACTS / 'y_test_1h.npy')
    rf_1h = load_model('random_forest_1h.pkl')
    y_pred_1h = rf_1h.predict(X_test_1h)

    # 计算误差
    errors = y_test_1h - y_pred_1h

    # 绘制误差分布
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('预测误差分析 (1小时 Random Forest)', fontsize=16)

    # 误差直方图
    ax1 = axes[0]
    ax1.hist(errors, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax1.set_xlabel('预测误差 (°C)')
    ax1.set_ylabel('频次')
    ax1.set_title('误差分布直方图')
    ax1.grid(True, alpha=0.3)

    # 添加统计信息
    mean_error = np.mean(errors)
    std_error = np.std(errors)
    ax1.axvline(mean_error, color='red', linestyle='--', label=f'均值: {mean_error:.3f}')
    ax1.axvline(mean_error + std_error, color='orange', linestyle='--', label=f'+1σ: {mean_error+std_error:.3f}')
    ax1.axvline(mean_error - std_error, color='orange', linestyle='--', label=f'-1σ: {mean_error-std_error:.3f}')
    ax1.legend()

    # Q-Q图
    ax2 = axes[1]
    from scipy import stats
    stats.probplot(errors, dist="norm", plot=ax2)
    ax2.set_title('Q-Q图 (正态性检验)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('error_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_summary_table():
    """生成总结表格"""
    results_df = load_final_results()

    # 创建总结表
    summary = []

    for config in ['1h', '1d', '1w']:
        config_data = results_df[results_df['Config'] == config]

        # 最佳模型
        best_model = config_data.loc[config_data['R2'].idxmax()]

        # 传统ML最佳
        ml_models = ['LinearRegression', 'Ridge', 'RandomForestRegressor']
        ml_data = config_data[config_data['Model'].isin(ml_models)]
        best_ml = ml_data.loc[ml_data['R2'].idxmax()]

        # 深度学习最佳
        dl_models = ['MLPRegressor']
        dl_data = config_data[config_data['Model'].isin(dl_models)]
        if len(dl_data) > 0:
            best_dl = dl_data.loc[dl_data['R2'].idxmax()]
        else:
            best_dl = None

        summary.append({
            '配置': config,
            '最佳模型': best_model['Model'],
            'R²': f"{best_model['R2']:.4f}",
            'RMSE': f"{best_model['RMSE']:.4f}",
            '传统ML最佳': best_ml['Model'],
            'ML R²': f"{best_ml['R2']:.4f}",
            '深度学习最佳': best_dl['Model'] if best_dl is not None else 'N/A',
            'DL R²': f"{best_dl['R2']:.4f}" if best_dl is not None else 'N/A'
        })

    summary_df = pd.DataFrame(summary)
    print("模型性能总结表:")
    print(summary_df.to_string(index=False))

    return summary_df

def main():
    """主函数：运行所有可视化分析"""
    print("开始生成可视化分析...")

    # 加载数据
    results_df = load_final_results()

    # 1. 模型性能对比
    print("\n1. 生成模型性能对比图...")
    plot_model_performance(results_df)

    # 2. 预测示例
    print("2. 生成预测示例图...")
    plot_prediction_examples()

    # 3. 误差分布
    print("3. 生成误差分布图...")
    plot_error_distribution()

    # 4. 总结表格
    print("4. 生成总结表格...")
    summary_df = generate_summary_table()

    print("\n可视化分析完成！")
    print("生成的图表文件:")
    print("- model_performance_analysis.png")
    print("- prediction_examples.png")
    print("- error_distribution.png")

    return summary_df

if __name__ == "__main__":
    main()