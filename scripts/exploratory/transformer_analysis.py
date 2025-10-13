import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 加载数据
def load_data():
    """加载两个变压器数据文件"""
    trans1 = pd.read_csv('trans_1.csv')
    trans2 = pd.read_csv('trans_2.csv')

    # 转换日期列
    trans1['date'] = pd.to_datetime(trans1['date'])
    trans2['date'] = pd.to_datetime(trans2['date'])

    return trans1, trans2

# 数据探索
def explore_data(df, name):
    """探索数据的基本统计信息"""
    print(f"\n=== {name} 数据概览 ===")
    print(f"数据形状: {df.shape}")
    print(f"时间范围: {df['date'].min()} 到 {df['date'].max()}")
    print(f"数据点数量: {len(df)}")
    print("\n基本统计信息:")
    print(df.describe())
    print(f"\n缺失值检查:")
    print(df.isnull().sum())
    print(f"\n重复值检查: {df.duplicated().sum()}")

    return df

# 时间序列可视化
def visualize_time_series(df, name):
    """可视化时间序列数据"""
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle(f'{name} 时间序列可视化', fontsize=16)

    # 定义特征
    features = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']

    for i, feature in enumerate(features):
        row = i // 2
        col = i % 2

        # 绘制前1000个点以便观察模式
        axes[row, col].plot(df['date'][:1000], df[feature][:1000], alpha=0.7)
        axes[row, col].set_title(f'{feature} - 高压有用负载' if 'HUFL' in feature else
                                f'{feature} - 高压无用负载' if 'HULL' in feature else
                                f'{feature} - 中压有用负载' if 'MUFL' in feature else
                                f'{feature} - 中压无用负载' if 'MULL' in feature else
                                f'{feature} - 低压有用负载' if 'LUFL' in feature else
                                f'{feature} - 低压无用负载')
        axes[row, col].set_xlabel('时间')
        axes[row, col].set_ylabel('数值')
        axes[row, col].tick_params(axis='x', rotation=45)

    plt.tight_layout()
    plt.savefig(f'{name}_features_timeseries.png', dpi=300, bbox_inches='tight')
    plt.show()

# 油温可视化
def visualize_oil_temperature(df, name):
    """可视化油温数据"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # 时间序列图
    ax1.plot(df['date'], df['OT'], alpha=0.7, color='red')
    ax1.set_title(f'{name} 油温时间序列')
    ax1.set_xlabel('时间')
    ax1.set_ylabel('油温 (°C)')
    ax1.tick_params(axis='x', rotation=45)

    # 分布图
    ax2.hist(df['OT'], bins=50, alpha=0.7, color='red', edgecolor='black')
    ax2.set_title(f'{name} 油温分布')
    ax2.set_xlabel('油温 (°C)')
    ax2.set_ylabel('频次')

    plt.tight_layout()
    plt.savefig(f'{name}_oil_temperature.png', dpi=300, bbox_inches='tight')
    plt.show()

def feature_histograms(df, name):
    """绘制六个负载特征的分布直方图，帮助直观理解取值范围与偏态"""
    features = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(f'{name} 特征分布直方图', fontsize=16)

    for i, feature in enumerate(features):
        r, c = divmod(i, 2)
        axes[r, c].hist(df[feature].dropna(), bins=50, alpha=0.8, color='#4c78a8', edgecolor='black')
        axes[r, c].set_title(feature)
        axes[r, c].set_xlabel('数值')
        axes[r, c].set_ylabel('频次')
        axes[r, c].grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(f'{name}_feature_histograms.png', dpi=300, bbox_inches='tight')
    plt.show()

def hourly_profile_ot(df, name):
    """绘制按小时聚合的油温平均值曲线，直观看到日内周期模式"""
    if not np.issubdtype(df['date'].dtype, np.datetime64):
        df['date'] = pd.to_datetime(df['date'])
    df['hour'] = df['date'].dt.hour
    hourly_mean = df.groupby('hour')['OT'].mean()

    plt.figure(figsize=(10, 5))
    plt.plot(hourly_mean.index, hourly_mean.values, marker='o', color='#e45756')
    plt.title(f'{name} 按小时的油温平均值')
    plt.xlabel('小时 (0-23)')
    plt.ylabel('油温平均值 (°C)')
    plt.grid(True, alpha=0.3)
    plt.xticks(range(0, 24))
    plt.tight_layout()
    plt.savefig(f'{name}_hourly_profile_ot.png', dpi=300, bbox_inches='tight')
    plt.show()

# 相关性分析
def correlation_analysis(df, name):
    """分析特征之间的相关性"""
    features = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
    correlation_matrix = df[features].corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, linewidths=0.5)
    plt.title(f'{name} 特征相关性矩阵')
    plt.tight_layout()
    plt.savefig(f'{name}_correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

    return correlation_matrix

# 主函数
def main():
    # 加载数据
    trans1, trans2 = load_data()

    # 数据探索
    trans1 = explore_data(trans1, 'Transformer 1')
    trans2 = explore_data(trans2, 'Transformer 2')

    # 时间序列可视化
    visualize_time_series(trans1, 'Transformer 1')
    visualize_time_series(trans2, 'Transformer 2')

    # 油温可视化
    visualize_oil_temperature(trans1, 'Transformer 1')
    visualize_oil_temperature(trans2, 'Transformer 2')

    # 特征分布直方图
    feature_histograms(trans1, 'Transformer 1')
    feature_histograms(trans2, 'Transformer 2')

    # 小白友好的日内模式图（小时聚合）
    hourly_profile_ot(trans1, 'Transformer 1')
    hourly_profile_ot(trans2, 'Transformer 2')

    # 相关性分析
    corr1 = correlation_analysis(trans1, 'Transformer 1')
    corr2 = correlation_analysis(trans2, 'Transformer 2')

    # 保存相关性结果
    corr1.to_csv('trans1_correlation.csv')
    corr2.to_csv('trans2_correlation.csv')

    print("\n=== 分析完成 ===")
    print("相关性矩阵已保存为 trans1_correlation.csv 和 trans2_correlation.csv")
    print("可视化图表已生成并保存")

if __name__ == "__main__":
    main()