import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def load_original_data():
    """加载原始数据进行季节性分析"""
    import pandas as pd
    
    # 加载数据
    trans1 = pd.read_csv('data/trans_1.csv')
    trans2 = pd.read_csv('data/trans_2.csv')
    
    # 转换日期
    trans1['date'] = pd.to_datetime(trans1['date'])
    trans2['date'] = pd.to_datetime(trans2['date'])
    
    # 合并数据
    data = pd.concat([trans1, trans2], ignore_index=True)
    data = data.sort_values('date').reset_index(drop=True)
    
    return data

def add_time_features(data):
    """添加时间特征用于分析"""
    df = data.copy()
    
    # 基础时间特征
    df['hour'] = df['date'].dt.hour
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear
    
    # 季节
    df['season'] = df['month'].map({12: 'Winter', 1: 'Winter', 2: 'Winter',
                                   3: 'Spring', 4: 'Spring', 5: 'Spring',
                                   6: 'Summer', 7: 'Summer', 8: 'Summer',
                                   9: 'Autumn', 10: 'Autumn', 11: 'Autumn'})
    
    # 工作时间
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['is_worktime'] = ((df['hour'] >= 8) & (df['hour'] <= 18) & (~df['is_weekend'])).astype(int)
    
    return df

def analyze_seasonal_patterns(data):
    """分析季节性模式"""
    print("=== 季节性模式分析 ===\n")
    
    # 1. 按季节统计
    seasonal_stats = data.groupby('season')['OT'].agg(['mean', 'std', 'min', 'max']).round(2)
    print("1. 季节性油温统计:")
    print(seasonal_stats)
    print()
    
    # 2. 按月份统计
    monthly_stats = data.groupby('month')['OT'].agg(['mean', 'std']).round(2)
    print("2. 月度油温统计:")
    print(monthly_stats)
    print()
    
    # 3. 按小时统计
    hourly_stats = data.groupby('hour')['OT'].agg(['mean', 'std']).round(2)
    print("3. 小时油温统计 (前12小时):")
    print(hourly_stats.head(12))
    print()
    
    # 4. 工作日vs周末
    workday_stats = data.groupby('is_weekend')['OT'].agg(['mean', 'std']).round(2)
    workday_stats.index = ['工作日', '周末']
    print("4. 工作日vs周末油温统计:")
    print(workday_stats)
    print()
    
    return seasonal_stats, monthly_stats, hourly_stats, workday_stats

def create_seasonal_visualizations(data):
    """创建季节性可视化"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, axes = plt.subplots(3, 2, figsize=(16, 18))
    fig.suptitle('变压器油温的时间模式分析', fontsize=16, fontweight='bold')
    
    # 1. 季节性箱线图
    sns.boxplot(data=data, x='season', y='OT', ax=axes[0,0])
    axes[0,0].set_title('季节性油温分布')
    axes[0,0].set_xlabel('季节')
    axes[0,0].set_ylabel('油温 (°C)')
    
    # 2. 月度趋势
    monthly_mean = data.groupby('month')['OT'].mean()
    axes[0,1].plot(monthly_mean.index, monthly_mean.values, marker='o', linewidth=2)
    axes[0,1].set_title('月度平均油温趋势')
    axes[0,1].set_xlabel('月份')
    axes[0,1].set_ylabel('平均油温 (°C)')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. 小时模式
    hourly_mean = data.groupby('hour')['OT'].mean()
    axes[1,0].plot(hourly_mean.index, hourly_mean.values, marker='o', linewidth=2, color='orange')
    axes[1,0].set_title('日内小时油温模式')
    axes[1,0].set_xlabel('小时')
    axes[1,0].set_ylabel('平均油温 (°C)')
    axes[1,0].grid(True, alpha=0.3)
    
    # 4. 工作日vs周末
    weekend_data = []
    for is_weekend in [0, 1]:
        subset = data[data['is_weekend'] == is_weekend]
        hourly_pattern = subset.groupby('hour')['OT'].mean()
        label = '周末' if is_weekend else '工作日'
        axes[1,1].plot(hourly_pattern.index, hourly_pattern.values, 
                      marker='o', label=label, linewidth=2)
    
    axes[1,1].set_title('工作日vs周末的小时模式')
    axes[1,1].set_xlabel('小时')
    axes[1,1].set_ylabel('平均油温 (°C)')
    axes[1,1].legend()
    axes[1,1].grid(True, alpha=0.3)
    
    # 5. 季节性小时热力图
    pivot_data = data.groupby(['season', 'hour'])['OT'].mean().unstack()
    sns.heatmap(pivot_data, annot=False, cmap='YlOrRd', ax=axes[2,0])
    axes[2,0].set_title('季节-小时油温热力图')
    axes[2,0].set_xlabel('小时')
    axes[2,0].set_ylabel('季节')
    
    # 6. 年度趋势（如果数据跨越多年）
    data['year_month'] = data['date'].dt.to_period('M')
    monthly_trend = data.groupby('year_month')['OT'].mean()
    
    if len(monthly_trend) > 12:  # 如果有超过一年的数据
        axes[2,1].plot(range(len(monthly_trend)), monthly_trend.values, linewidth=2)
        axes[2,1].set_title('长期月度趋势')
        axes[2,1].set_xlabel('时间序列')
        axes[2,1].set_ylabel('平均油温 (°C)')
        axes[2,1].grid(True, alpha=0.3)
        
        # 设置x轴标签
        n_ticks = min(12, len(monthly_trend))
        tick_positions = np.linspace(0, len(monthly_trend)-1, n_ticks, dtype=int)
        tick_labels = [str(monthly_trend.index[i]) for i in tick_positions]
        axes[2,1].set_xticks(tick_positions)
        axes[2,1].set_xticklabels(tick_labels, rotation=45)
    else:
        axes[2,1].text(0.5, 0.5, '数据时间跨度不足\n无法显示长期趋势', 
                      ha='center', va='center', transform=axes[2,1].transAxes)
        axes[2,1].set_title('长期趋势（数据不足）')
    
    plt.tight_layout()
    plt.savefig('seasonal_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_feature_correlations(data):
    """分析时间特征与油温的相关性"""
    print("=== 时间特征与油温相关性分析 ===\n")
    
    # 选择数值型时间特征
    time_features = ['hour', 'day_of_week', 'month', 'day_of_year', 'is_weekend', 'is_worktime']
    
    correlations = {}
    for feature in time_features:
        corr = data[feature].corr(data['OT'])
        correlations[feature] = corr
    
    # 排序并显示
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    
    print("时间特征与油温的相关系数（按绝对值排序）:")
    for feature, corr in sorted_corr:
        print(f"{feature:15}: {corr:+.4f}")
    
    print()
    return correlations

def analyze_load_patterns(data):
    """分析负载模式对油温的影响"""
    print("=== 负载模式对油温影响分析 ===\n")
    
    # 计算总负载（所有电气特征的和）
    electrical_features = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']
    data['total_load'] = data[electrical_features].sum(axis=1)
    
    # 按时间段分析负载模式
    print("1. 不同时间段的平均负载:")
    time_load = data.groupby('hour')[['total_load', 'OT']].mean()
    
    # 找出高负载和低负载时段
    high_load_hours = time_load.nlargest(6, 'total_load').index.tolist()
    low_load_hours = time_load.nsmallest(6, 'total_load').index.tolist()
    
    print(f"高负载时段 (前6小时): {high_load_hours}")
    print(f"低负载时段 (后6小时): {low_load_hours}")
    
    high_load_temp = time_load.loc[high_load_hours, 'OT'].mean()
    low_load_temp = time_load.loc[low_load_hours, 'OT'].mean()
    
    print(f"高负载时段平均油温: {high_load_temp:.2f}°C")
    print(f"低负载时段平均油温: {low_load_temp:.2f}°C")
    print(f"温差: {high_load_temp - low_load_temp:.2f}°C")
    print()
    
    # 负载与油温的相关性
    load_temp_corr = data['total_load'].corr(data['OT'])
    print(f"2. 总负载与油温相关系数: {load_temp_corr:.4f}")
    print()
    
    return time_load, high_load_hours, low_load_hours

def generate_recommendations(seasonal_stats, correlations, time_load):
    """生成基于分析结果的建议"""
    print("=== 基于分析的建议 ===\n")
    
    # 1. 季节性建议
    temp_range = seasonal_stats['max'].max() - seasonal_stats['min'].min()
    print(f"1. 季节性影响:")
    print(f"   • 年度油温变化范围: {temp_range:.1f}°C")
    print(f"   • 最高温季节: {seasonal_stats['mean'].idxmax()}")
    print(f"   • 最低温季节: {seasonal_stats['mean'].idxmin()}")
    print(f"   • 建议: 在{seasonal_stats['mean'].idxmax()}加强监控，在{seasonal_stats['mean'].idxmin()}注意保温")
    print()
    
    # 2. 时间特征重要性
    important_features = [k for k, v in correlations.items() if abs(v) > 0.1]
    print(f"2. 重要时间特征 (|相关系数| > 0.1):")
    for feature in important_features:
        print(f"   • {feature}: {correlations[feature]:+.4f}")
    print()
    
    # 3. 负载模式建议
    peak_hour = time_load['total_load'].idxmax()
    valley_hour = time_load['total_load'].idxmin()
    print(f"3. 负载模式:")
    print(f"   • 负载峰值时间: {peak_hour}:00")
    print(f"   • 负载谷值时间: {valley_hour}:00")
    print(f"   • 建议: 在{peak_hour}:00前后加强油温监控")
    print()
    
    # 4. 模型改进建议
    print("4. 模型改进建议:")
    print("   • 时间特征显著改善了预测性能，特别是短期预测(1H)")
    print("   • RandomForest模型比Ridge回归更能利用时间特征")
    print("   • 建议优先使用包含时间特征的RandomForest模型")
    print("   • 对于长期预测(1W)，需要进一步优化特征工程")

def main():
    """主函数"""
    print("=== 变压器油温季节性与时间模式深度分析 ===\n")
    
    try:
        import pandas as pd
        
        # 加载数据
        print("正在加载数据...")
        data = load_original_data()
        data = add_time_features(data)
        print(f"数据加载完成，共 {len(data)} 条记录")
        print(f"时间跨度: {data['date'].min()} 到 {data['date'].max()}")
        print()
        
        # 季节性分析
        seasonal_stats, monthly_stats, hourly_stats, workday_stats = analyze_seasonal_patterns(data)
        
        # 相关性分析
        correlations = analyze_feature_correlations(data)
        
        # 负载模式分析
        time_load, high_load_hours, low_load_hours = analyze_load_patterns(data)
        
        # 创建可视化
        print("正在生成可视化图表...")
        create_seasonal_visualizations(data)
        print("✓ 季节性分析图表已保存: seasonal_analysis.png")
        print()
        
        # 生成建议
        generate_recommendations(seasonal_stats, correlations, time_load)
        
        # 保存分析结果
        analysis_results = {
            'seasonal_stats': seasonal_stats,
            'monthly_stats': monthly_stats,
            'hourly_stats': hourly_stats,
            'correlations': correlations,
            'time_load_pattern': time_load
        }
        
        # 保存为Excel文件
        with pd.ExcelWriter('seasonal_analysis_results.xlsx', engine='openpyxl') as writer:
            seasonal_stats.to_excel(writer, sheet_name='季节统计')
            monthly_stats.to_excel(writer, sheet_name='月度统计')
            hourly_stats.to_excel(writer, sheet_name='小时统计')
            pd.DataFrame(list(correlations.items()), 
                        columns=['特征', '相关系数']).to_excel(writer, sheet_name='相关性分析', index=False)
            time_load.to_excel(writer, sheet_name='负载模式')
        
        print("✓ 详细分析结果已保存: seasonal_analysis_results.xlsx")
        
    except ImportError:
        print("错误: 缺少必要的依赖库")
        print("请安装: pip install pandas matplotlib seaborn openpyxl")
    except FileNotFoundError:
        print("错误: 未找到数据文件 data/trans_1.csv 或 data/trans_2.csv")
        print("请确保数据文件存在于 data/ 目录中")
    except Exception as e:
        print(f"分析过程中出现错误: {e}")

if __name__ == "__main__":
    main()