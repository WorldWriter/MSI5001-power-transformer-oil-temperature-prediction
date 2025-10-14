#!/usr/bin/env python3
"""
时间窗口长度实验 - 结果分析脚本

本脚本分析不同时间窗口配置下的模型性能，
生成详细的对比分析报告。

作者: MSI5001项目组
日期: 2024年
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_results():
    """加载模型训练结果"""
    results_path = "../results/model_training_results.csv"
    if not os.path.exists(results_path):
        raise FileNotFoundError("未找到模型训练结果文件，请先运行 window_experiment_models.py")
    
    df = pd.read_csv(results_path)
    print(f"加载了 {len(df)} 个模型训练结果")
    return df

def analyze_window_length_impact(df):
    """分析时间窗口长度对性能的影响"""
    print("\n=== 分析时间窗口长度对性能的影响 ===")
    
    analysis_results = {}
    
    # 按预测跨度分组分析
    for forecast_hours in df['forecast_hours'].unique():
        forecast_data = df[df['forecast_hours'] == forecast_hours]
        
        print(f"\n预测跨度: {forecast_hours:.1f} 小时")
        
        # 按模型类型分析
        model_analysis = {}
        for model_type in forecast_data['model_type'].unique():
            model_data = forecast_data[forecast_data['model_type'] == model_type]
            
            # 计算相关性
            correlation_r2 = model_data['history_hours'].corr(model_data['r2_score'])
            correlation_rmse = model_data['history_hours'].corr(model_data['rmse'])
            
            # 找到最佳配置
            best_idx = model_data['r2_score'].idxmax()
            best_config = model_data.loc[best_idx]
            
            model_analysis[model_type] = {
                'correlation_r2_history': correlation_r2,
                'correlation_rmse_history': correlation_rmse,
                'best_r2': best_config['r2_score'],
                'best_history_hours': best_config['history_hours'],
                'best_config': best_config['config'],
                'performance_trend': 'improving' if correlation_r2 > 0.1 else 'declining' if correlation_r2 < -0.1 else 'stable'
            }
            
            print(f"  {model_type:12} - 最佳R²: {best_config['r2_score']:.4f} (历史窗口: {best_config['history_hours']:.1f}h)")
        
        analysis_results[f"{forecast_hours:.1f}h"] = model_analysis
    
    return analysis_results

def create_performance_comparison_plots(df):
    """创建性能对比图表"""
    print("\n=== 生成性能对比图表 ===")
    
    # 创建可视化目录
    viz_dir = "../visualizations"
    os.makedirs(viz_dir, exist_ok=True)
    
    # 1. R²分数 vs 历史窗口长度
    plt.figure(figsize=(15, 10))
    
    forecast_hours_list = sorted(df['forecast_hours'].unique())
    
    for i, forecast_hours in enumerate(forecast_hours_list):
        plt.subplot(2, 2, i+1)
        forecast_data = df[df['forecast_hours'] == forecast_hours]
        
        for model_type in forecast_data['model_type'].unique():
            model_data = forecast_data[forecast_data['model_type'] == model_type]
            plt.plot(model_data['history_hours'], model_data['r2_score'], 
                    marker='o', label=model_type, linewidth=2, markersize=6)
        
        plt.xlabel('历史窗口长度 (小时)')
        plt.ylabel('R² 分数')
        plt.title(f'预测跨度: {forecast_hours:.1f} 小时')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{viz_dir}/r2_vs_history_window.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. RMSE vs 历史窗口长度
    plt.figure(figsize=(15, 10))
    
    for i, forecast_hours in enumerate(forecast_hours_list):
        plt.subplot(2, 2, i+1)
        forecast_data = df[df['forecast_hours'] == forecast_hours]
        
        for model_type in forecast_data['model_type'].unique():
            model_data = forecast_data[forecast_data['model_type'] == model_type]
            plt.plot(model_data['history_hours'], model_data['rmse'], 
                    marker='s', label=model_type, linewidth=2, markersize=6)
        
        plt.xlabel('历史窗口长度 (小时)')
        plt.ylabel('RMSE')
        plt.title(f'预测跨度: {forecast_hours:.1f} 小时')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{viz_dir}/rmse_vs_history_window.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. 训练时间对比
    plt.figure(figsize=(12, 8))
    
    # 按模型类型和历史窗口长度分组
    pivot_time = df.pivot_table(values='training_time', 
                               index='history_hours', 
                               columns='model_type', 
                               aggfunc='mean')
    
    for model_type in pivot_time.columns:
        plt.plot(pivot_time.index, pivot_time[model_type], 
                marker='d', label=model_type, linewidth=2, markersize=6)
    
    plt.xlabel('历史窗口长度 (小时)')
    plt.ylabel('平均训练时间 (秒)')
    plt.title('训练时间 vs 历史窗口长度')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # 使用对数刻度
    
    plt.tight_layout()
    plt.savefig(f"{viz_dir}/training_time_vs_history_window.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. 热力图 - 最佳配置总览
    plt.figure(figsize=(12, 8))
    
    # 创建最佳R²分数的热力图
    heatmap_data = df.pivot_table(values='r2_score', 
                                 index='model_type', 
                                 columns='history_hours', 
                                 aggfunc='max')
    
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                cbar_kws={'label': 'R² 分数'})
    plt.title('各模型在不同历史窗口长度下的最佳R²分数')
    plt.xlabel('历史窗口长度 (小时)')
    plt.ylabel('模型类型')
    
    plt.tight_layout()
    plt.savefig(f"{viz_dir}/best_r2_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ 图表已保存到: {viz_dir}/")

def generate_detailed_report(df, analysis_results):
    """生成详细的分析报告"""
    print("\n=== 生成详细分析报告 ===")
    
    report_path = "../docs/window_length_analysis_report.md"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 时间窗口长度对预测性能影响分析报告\n\n")
        f.write(f"**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # 执行摘要
        f.write("## 执行摘要\n\n")
        f.write("本报告分析了不同历史时间窗口长度对变压器油温预测性能的影响。")
        f.write("通过系统性实验，我们测试了多种窗口长度配置在1小时、1天和1周预测任务上的表现。\n\n")
        
        # 实验配置
        f.write("## 实验配置\n\n")
        f.write("### 测试的时间窗口配置\n\n")
        
        config_summary = df.groupby(['forecast_hours', 'history_hours']).size().reset_index(name='model_count')
        
        for forecast_hours in sorted(config_summary['forecast_hours'].unique()):
            f.write(f"**{forecast_hours:.1f}小时预测**:\n")
            forecast_configs = config_summary[config_summary['forecast_hours'] == forecast_hours]
            for _, row in forecast_configs.iterrows():
                f.write(f"- 历史窗口: {row['history_hours']:.1f}小时 ({int(row['history_hours']/0.25)}个时间点)\n")
            f.write("\n")
        
        f.write("### 评估模型\n")
        f.write("- **Random Forest**: 集成学习方法，适合处理非线性关系\n")
        f.write("- **Ridge Regression**: 线性回归，具有L2正则化\n")
        f.write("- **MLP**: 多层感知机神经网络\n\n")
        
        f.write("### 评估指标\n")
        f.write("- **R² Score**: 决定系数，衡量模型解释方差的能力\n")
        f.write("- **RMSE**: 均方根误差，衡量预测精度\n")
        f.write("- **MAE**: 平均绝对误差\n")
        f.write("- **Training Time**: 模型训练时间\n\n")
        
        # 主要发现
        f.write("## 主要发现\n\n")
        
        # 最佳配置
        f.write("### 最佳配置总结\n\n")
        best_configs = df.loc[df.groupby(['forecast_hours', 'model_type'])['r2_score'].idxmax()]
        
        for forecast_hours in sorted(best_configs['forecast_hours'].unique()):
            f.write(f"**{forecast_hours:.1f}小时预测的最佳配置**:\n\n")
            forecast_best = best_configs[best_configs['forecast_hours'] == forecast_hours]
            
            for _, row in forecast_best.iterrows():
                f.write(f"- **{row['model_type']}**: ")
                f.write(f"历史窗口 {row['history_hours']:.1f}小时, ")
                f.write(f"R² = {row['r2_score']:.4f}, ")
                f.write(f"RMSE = {row['rmse']:.4f}\n")
            f.write("\n")
        
        # 性能趋势分析
        f.write("### 性能趋势分析\n\n")
        
        for forecast_key, models in analysis_results.items():
            f.write(f"**{forecast_key}预测**:\n\n")
            
            for model_type, analysis in models.items():
                trend_desc = {
                    'improving': '随历史窗口增长而改善',
                    'declining': '随历史窗口增长而下降',
                    'stable': '相对稳定'
                }
                
                f.write(f"- **{model_type}**: {trend_desc[analysis['performance_trend']]} ")
                f.write(f"(R²与历史窗口相关性: {analysis['correlation_r2_history']:.3f})\n")
            f.write("\n")
        
        # 详细结果表格
        f.write("## 详细实验结果\n\n")
        
        for forecast_hours in sorted(df['forecast_hours'].unique()):
            f.write(f"### {forecast_hours:.1f}小时预测结果\n\n")
            
            forecast_data = df[df['forecast_hours'] == forecast_hours]
            
            # 创建结果表格
            f.write("| 历史窗口(h) | 模型类型 | R² Score | RMSE | MAE | 训练时间(s) |\n")
            f.write("|-------------|----------|----------|------|-----|-------------|\n")
            
            for _, row in forecast_data.sort_values(['history_hours', 'model_type']).iterrows():
                f.write(f"| {row['history_hours']:.1f} | {row['model_type']} | ")
                f.write(f"{row['r2_score']:.4f} | {row['rmse']:.4f} | ")
                f.write(f"{row['mae']:.4f} | {row['training_time']:.2f} |\n")
            
            f.write("\n")
        
        # 关键洞察
        f.write("## 关键洞察与建议\n\n")
        
        # 找出整体最佳模型
        overall_best = df.loc[df['r2_score'].idxmax()]
        f.write(f"### 整体最佳配置\n\n")
        f.write(f"- **模型**: {overall_best['model_type']}\n")
        f.write(f"- **预测跨度**: {overall_best['forecast_hours']:.1f}小时\n")
        f.write(f"- **历史窗口**: {overall_best['history_hours']:.1f}小时\n")
        f.write(f"- **性能**: R² = {overall_best['r2_score']:.4f}, RMSE = {overall_best['rmse']:.4f}\n\n")
        
        # 计算改进效果
        f.write("### 窗口长度优化效果\n\n")
        
        for forecast_hours in sorted(df['forecast_hours'].unique()):
            forecast_data = df[df['forecast_hours'] == forecast_hours]
            
            for model_type in forecast_data['model_type'].unique():
                model_data = forecast_data[forecast_data['model_type'] == model_type]
                
                if len(model_data) > 1:
                    best_r2 = model_data['r2_score'].max()
                    worst_r2 = model_data['r2_score'].min()
                    improvement = ((best_r2 - worst_r2) / worst_r2) * 100
                    
                    f.write(f"- **{model_type}** ({forecast_hours:.1f}h预测): ")
                    f.write(f"最优窗口相比最差窗口提升 {improvement:.1f}%\n")
        
        f.write("\n### 实用建议\n\n")
        f.write("1. **短期预测(1小时)**: 建议使用较长的历史窗口以捕获更多模式\n")
        f.write("2. **中期预测(1天)**: 平衡历史信息量与计算效率\n")
        f.write("3. **长期预测(1周)**: 关注长期趋势，避免过度拟合短期波动\n")
        f.write("4. **模型选择**: Random Forest通常表现最佳，MLP适合复杂模式识别\n")
        f.write("5. **计算资源**: 考虑训练时间与性能的权衡\n\n")
        
        # 局限性和未来工作
        f.write("## 局限性与未来工作\n\n")
        f.write("### 当前局限性\n")
        f.write("- 实验基于单一数据集，泛化性有待验证\n")
        f.write("- 未考虑季节性和周期性因素的影响\n")
        f.write("- 计算资源限制了更大窗口长度的测试\n\n")
        
        f.write("### 未来改进方向\n")
        f.write("- 测试更多样化的数据集\n")
        f.write("- 引入自适应窗口长度选择机制\n")
        f.write("- 结合领域知识优化特征工程\n")
        f.write("- 探索深度学习模型的窗口长度敏感性\n\n")
        
        f.write("---\n")
        f.write("*本报告由MSI5001项目组自动生成*\n")
    
    print(f"✓ 详细报告已保存: {report_path}")

def main():
    """主函数"""
    print("=== 时间窗口长度实验 - 结果分析 ===")
    print("分析不同历史窗口长度对预测性能的影响\n")
    
    try:
        # 加载结果
        df = load_results()
        
        # 分析窗口长度影响
        analysis_results = analyze_window_length_impact(df)
        
        # 创建可视化图表
        create_performance_comparison_plots(df)
        
        # 生成详细报告
        generate_detailed_report(df, analysis_results)
        
        print(f"\n{'='*60}")
        print("✓ 分析完成！")
        print("✓ 生成的文件:")
        print("  - ../docs/window_length_analysis_report.md (详细报告)")
        print("  - ../visualizations/*.png (性能对比图表)")
        print(f"\n总结:")
        print(f"- 测试了 {len(df)} 个模型配置")
        print(f"- 涵盖 {len(df['forecast_hours'].unique())} 种预测跨度")
        print(f"- 使用 {len(df['model_type'].unique())} 种模型类型")
        
        # 显示最佳配置
        best_overall = df.loc[df['r2_score'].idxmax()]
        print(f"\n🏆 整体最佳配置:")
        print(f"   {best_overall['model_type']} - {best_overall['forecast_hours']:.1f}h预测")
        print(f"   历史窗口: {best_overall['history_hours']:.1f}h, R²: {best_overall['r2_score']:.4f}")
        
    except Exception as e:
        print(f"❌ 分析过程中出现错误: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()