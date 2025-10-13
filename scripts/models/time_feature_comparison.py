import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

def load_dataset(config, with_time=False):
    """加载数据集"""
    suffix = f"_{config}_time" if with_time else f"_{config}"
    
    X_train = np.load(f'X_train{suffix}.npy')
    X_test = np.load(f'X_test{suffix}.npy')
    y_train = np.load(f'y_train{suffix}.npy')
    y_test = np.load(f'y_test{suffix}.npy')
    
    return X_train, X_test, y_train, y_test

def evaluate_model(model, X_test, y_test):
    """评估模型性能"""
    y_pred = model.predict(X_test)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    return r2, rmse, mae, y_pred

def train_and_evaluate(config, model_name='RandomForest'):
    """训练并评估有无时间特征的模型"""
    print(f"\n=== {config.upper()} 预测 - {model_name} 对比实验 ===")
    
    results = {}
    
    for with_time in [False, True]:
        feature_type = "时间特征" if with_time else "仅电气特征"
        print(f"\n--- {feature_type} ---")
        
        # 加载数据
        X_train, X_test, y_train, y_test = load_dataset(config, with_time)
        print(f"特征维度: {X_train.shape[1]}")
        
        # 选择模型
        if model_name == 'RandomForest':
            model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        elif model_name == 'Ridge':
            model = Ridge(alpha=1.0, random_state=42)
        
        # 训练模型
        model.fit(X_train, y_train)
        
        # 评估
        r2, rmse, mae, y_pred = evaluate_model(model, X_test, y_test)
        
        print(f"R²: {r2:.4f}")
        print(f"RMSE: {rmse:.4f}°C")
        print(f"MAE: {mae:.4f}°C")
        
        # 保存结果
        results[feature_type] = {
            'r2': r2, 'rmse': rmse, 'mae': mae,
            'y_test': y_test, 'y_pred': y_pred,
            'feature_dim': X_train.shape[1]
        }
    
    # 计算改善程度
    baseline = results["仅电气特征"]
    enhanced = results["时间特征"]
    
    r2_improvement = enhanced['r2'] - baseline['r2']
    rmse_improvement = baseline['rmse'] - enhanced['rmse']  # RMSE越小越好
    mae_improvement = baseline['mae'] - enhanced['mae']    # MAE越小越好
    
    print(f"\n=== 改善效果 ===")
    print(f"R² 改善: {r2_improvement:+.4f} ({r2_improvement/baseline['r2']*100:+.1f}%)")
    print(f"RMSE 改善: {rmse_improvement:+.4f}°C ({rmse_improvement/baseline['rmse']*100:+.1f}%)")
    print(f"MAE 改善: {mae_improvement:+.4f}°C ({mae_improvement/baseline['mae']*100:+.1f}%)")
    
    return results, {
        'config': config,
        'model': model_name,
        'baseline_r2': baseline['r2'],
        'enhanced_r2': enhanced['r2'],
        'r2_improvement': r2_improvement,
        'baseline_rmse': baseline['rmse'],
        'enhanced_rmse': enhanced['rmse'],
        'rmse_improvement': rmse_improvement,
        'baseline_mae': baseline['mae'],
        'enhanced_mae': enhanced['mae'],
        'mae_improvement': mae_improvement,
        'baseline_features': baseline['feature_dim'],
        'enhanced_features': enhanced['feature_dim']
    }

def create_comparison_plots(all_results):
    """创建对比可视化"""
    # 准备数据
    plot_data = []
    for result in all_results:
        for feature_type in ['baseline', 'enhanced']:
            plot_data.append({
                'Config': result['config'].upper(),
                'Model': result['model'],
                'Feature_Type': '仅电气特征' if feature_type == 'baseline' else '电气+时间特征',
                'R²': result[f'{feature_type}_r2'],
                'RMSE': result[f'{feature_type}_rmse'],
                'MAE': result[f'{feature_type}_mae']
            })
    
    df = pd.DataFrame(plot_data)
    
    # 创建子图
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('时间特征对模型性能的影响', fontsize=16, fontweight='bold')
    
    # R² 对比
    sns.barplot(data=df, x='Config', y='R²', hue='Feature_Type', ax=axes[0,0])
    axes[0,0].set_title('R² 对比')
    axes[0,0].set_ylim(0, max(df['R²']) * 1.1)
    
    # RMSE 对比
    sns.barplot(data=df, x='Config', y='RMSE', hue='Feature_Type', ax=axes[0,1])
    axes[0,1].set_title('RMSE 对比 (°C)')
    
    # MAE 对比
    sns.barplot(data=df, x='Config', y='MAE', hue='Feature_Type', ax=axes[0,2])
    axes[0,2].set_title('MAE 对比 (°C)')
    
    # 改善程度条形图
    improvement_data = []
    for result in all_results:
        improvement_data.append({
            'Config': result['config'].upper(),
            'R²改善': result['r2_improvement'],
            'RMSE改善': result['rmse_improvement'],
            'MAE改善': result['mae_improvement']
        })
    
    imp_df = pd.DataFrame(improvement_data)
    
    # R² 改善
    axes[1,0].bar(imp_df['Config'], imp_df['R²改善'], color='green', alpha=0.7)
    axes[1,0].set_title('R² 改善程度')
    axes[1,0].set_ylabel('R² 改善值')
    axes[1,0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # RMSE 改善
    axes[1,1].bar(imp_df['Config'], imp_df['RMSE改善'], color='blue', alpha=0.7)
    axes[1,1].set_title('RMSE 改善程度 (°C)')
    axes[1,1].set_ylabel('RMSE 改善值')
    axes[1,1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # MAE 改善
    axes[1,2].bar(imp_df['Config'], imp_df['MAE改善'], color='orange', alpha=0.7)
    axes[1,2].set_title('MAE 改善程度 (°C)')
    axes[1,2].set_ylabel('MAE 改善值')
    axes[1,2].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('time_feature_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

def create_summary_table(all_results):
    """创建汇总表格"""
    summary_data = []
    for result in all_results:
        summary_data.append({
            '预测跨度': result['config'].upper(),
            '模型': result['model'],
            '基线R²': f"{result['baseline_r2']:.4f}",
            '增强R²': f"{result['enhanced_r2']:.4f}",
            'R²改善': f"{result['r2_improvement']:+.4f}",
            '基线RMSE': f"{result['baseline_rmse']:.2f}",
            '增强RMSE': f"{result['enhanced_rmse']:.2f}",
            'RMSE改善': f"{result['rmse_improvement']:+.2f}",
            '基线特征数': result['baseline_features'],
            '增强特征数': result['enhanced_features']
        })
    
    df = pd.DataFrame(summary_data)
    
    # 保存为CSV
    df.to_csv('time_feature_comparison_results.csv', index=False, encoding='utf-8-sig')
    
    print("\n" + "="*80)
    print("时间特征影响汇总表")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    return df

def main():
    """主函数"""
    print("=== 时间特征对变压器油温预测的影响分析 ===")
    
    configs = ['1h', '1d', '1w']
    models = ['RandomForest', 'Ridge']
    
    all_results = []
    
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"模型: {model_name}")
        print('='*60)
        
        for config in configs:
            try:
                results, summary = train_and_evaluate(config, model_name)
                all_results.append(summary)
            except FileNotFoundError as e:
                print(f"警告: {config} 配置的数据文件未找到，跳过...")
                continue
    
    if all_results:
        # 创建可视化
        create_comparison_plots(all_results)
        
        # 创建汇总表
        summary_df = create_summary_table(all_results)
        
        print(f"\n✓ 分析完成！")
        print(f"✓ 可视化图表已保存: time_feature_comparison.png")
        print(f"✓ 详细结果已保存: time_feature_comparison_results.csv")
        
        # 关键发现
        print(f"\n=== 关键发现 ===")
        
        # 找出最大改善
        best_r2_improvement = max(all_results, key=lambda x: x['r2_improvement'])
        best_rmse_improvement = max(all_results, key=lambda x: x['rmse_improvement'])
        
        print(f"• 最大R²改善: {best_r2_improvement['config'].upper()} + {best_r2_improvement['model']} "
              f"({best_r2_improvement['r2_improvement']:+.4f})")
        print(f"• 最大RMSE改善: {best_rmse_improvement['config'].upper()} + {best_rmse_improvement['model']} "
              f"({best_rmse_improvement['rmse_improvement']:+.2f}°C)")
        
        # 统计正面改善的比例
        positive_r2 = sum(1 for r in all_results if r['r2_improvement'] > 0)
        positive_rmse = sum(1 for r in all_results if r['rmse_improvement'] > 0)
        
        print(f"• R²改善的实验比例: {positive_r2}/{len(all_results)} ({positive_r2/len(all_results)*100:.0f}%)")
        print(f"• RMSE改善的实验比例: {positive_rmse}/{len(all_results)} ({positive_rmse/len(all_results)*100:.0f}%)")
        
    else:
        print("错误: 没有找到可用的数据文件，请先运行 enhanced_preprocessing.py")

if __name__ == "__main__":
    main()