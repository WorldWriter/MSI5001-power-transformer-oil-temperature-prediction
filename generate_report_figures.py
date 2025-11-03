"""
生成项目报告中的所有图表
Generate all figures for the project report
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# 设置英文字体
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10

# 设置路径
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / 'MSI5001-power-transformer-oil-temperature-prediction' / 'external' / 'Informer2020' / 'data'
RESULTS_DIR = BASE_DIR / 'MSI5001-power-transformer-oil-temperature-prediction' / 'results' / 'tables'
FIG_DIR = BASE_DIR / 'figures'
FIG_DIR.mkdir(exist_ok=True)

print("=" * 60)
print("开始生成报告图表...")
print("=" * 60)

# ============================================================================
# 图1a: TX1 vs TX2负载与油温周度趋势对比 (一周数据，双Y轴)
# ============================================================================
print("\n[1a/8] 绘制图1a: TX1 vs TX2负载与油温周度趋势对比 (一周)...")

try:
    tx1 = pd.read_csv(DATA_DIR / 'TX1.csv', parse_dates=['date'])
    tx2 = pd.read_csv(DATA_DIR / 'TX2.csv', parse_dates=['date'])

    # 选取一周数据 (2018年10月15-21日)
    start, end = '2018-10-15', '2018-10-22'
    tx1_week = tx1[(tx1['date'] >= start) & (tx1['date'] < end)]
    tx2_week = tx2[(tx2['date'] >= start) & (tx2['date'] < end)]

    fig, ax1 = plt.subplots(figsize=(14, 6))

    # 左Y轴：油温OT（实线）
    ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Oil Temperature (°C)', fontsize=12, fontweight='bold', color='black')

    line1 = ax1.plot(tx1_week['date'], tx1_week['OT'], label='TX1 OT (Industrial)',
                     linewidth=2.5, color='steelblue', linestyle='-', marker='o', markersize=2)
    line2 = ax1.plot(tx2_week['date'], tx2_week['OT'], label='TX2 OT (Residential)',
                     linewidth=2.5, color='darkorange', linestyle='-', marker='s', markersize=2)

    ax1.tick_params(axis='y', labelcolor='black')
    ax1.set_ylim([tx1_week['OT'].min()-2, tx2_week['OT'].max()+2])

    # 右Y轴：负载HUFL（虚线）
    ax2 = ax1.twinx()
    ax2.set_ylabel('High Voltage Active Load - HUFL (kW)', fontsize=12, fontweight='bold', color='gray')

    line3 = ax2.plot(tx1_week['date'], tx1_week['HUFL'], label='TX1 HUFL',
                     linewidth=2, color='steelblue', linestyle='--', alpha=0.7)
    line4 = ax2.plot(tx2_week['date'], tx2_week['HUFL'], label='TX2 HUFL',
                     linewidth=2, color='darkorange', linestyle='--', alpha=0.7)

    ax2.tick_params(axis='y', labelcolor='gray')

    # 标注周末
    weekend_start = pd.Timestamp('2018-10-20')
    weekend_end = pd.Timestamp('2018-10-22')
    ax1.axvspan(weekend_start, weekend_end, alpha=0.15, color='gray', zorder=0)

    # 合并图例
    lines = line1 + line2 # + line3 + line4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, fontsize=10, loc='upper left', framealpha=0.9)

    ax1.set_title('Weekly Load and Oil Temperature Trend: TX1 vs TX2\n(Solid: OT, Dashed: HUFL)',
                  fontsize=14, fontweight='bold', pad=15)
    ax1.grid(alpha=0.3, linestyle='--')

    # 格式化日期
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax1.xaxis.set_major_locator(mdates.DayLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    fig.tight_layout()
    plt.savefig(FIG_DIR / 'fig1a_load_comparison_weekly.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 图1a已保存: figures/fig1a_load_comparison_weekly.png")
except Exception as e:
    print(f"✗ 图1a生成失败: {e}")

# ============================================================================
# 图1b: TX1 vs TX2负载长期趋势对比 (全部数据)
# ============================================================================
print("\n[1b/8] 绘制图1b: TX1 vs TX2负载长期趋势对比 (全部数据)...")

try:
    # 使用每日平均值减少数据点
    tx1['date_only'] = tx1['date'].dt.date
    tx2['date_only'] = tx2['date'].dt.date

    # 计算OT和HUFL的每日平均值
    tx1_daily = tx1.groupby('date_only').agg({'OT': 'mean', 'HUFL': 'mean'}).reset_index()
    tx2_daily = tx2.groupby('date_only').agg({'OT': 'mean', 'HUFL': 'mean'}).reset_index()

    tx1_daily['date_only'] = pd.to_datetime(tx1_daily['date_only'])
    tx2_daily['date_only'] = pd.to_datetime(tx2_daily['date_only'])

    fig, ax1 = plt.subplots(figsize=(14, 6))

    # 左Y轴：油温OT（实线）
    ax1.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Daily Average Oil Temperature (°C)', fontsize=12, fontweight='bold', color='black')

    line1 = ax1.plot(tx1_daily['date_only'], tx1_daily['OT'], label='TX1 OT (Industrial)',
                     linewidth=2, color='steelblue', linestyle='-', alpha=0.9)
    line2 = ax1.plot(tx2_daily['date_only'], tx2_daily['OT'], label='TX2 OT (Residential)',
                     linewidth=2, color='darkorange', linestyle='-', alpha=0.9)

    ax1.tick_params(axis='y', labelcolor='black')

    # 右Y轴：负载HUFL（虚线）
    # ax2 = ax1.twinx()
    # ax2.set_ylabel('Daily Average HUFL (kW)', fontsize=12, fontweight='bold', color='gray')

    # line3 = ax2.plot(tx1_daily['date_only'], tx1_daily['HUFL'], label='TX1 HUFL',
    #                  linewidth=1.5, color='steelblue', linestyle='--', alpha=0.7)
    # line4 = ax2.plot(tx2_daily['date_only'], tx2_daily['HUFL'], label='TX2 HUFL',
    #                  linewidth=1.5, color='darkorange', linestyle='--', alpha=0.7)

    # ax2.tick_params(axis='y', labelcolor='gray')

    # 合并图例
    lines = line1 + line2  # + line3 + line4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, fontsize=10, loc='upper left', framealpha=0.9)

    ax1.set_title('Long-term Load and Oil Temperature Trend: TX1 vs TX2\n(Solid: OT, Dashed: HUFL)',
                  fontsize=14, fontweight='bold', pad=15)
    ax1.grid(alpha=0.3, linestyle='--')

    # 格式化日期
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')

    fig.tight_layout()
    plt.savefig(FIG_DIR / 'fig1b_load_comparison_longterm.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 图1b已保存: figures/fig1b_load_comparison_longterm.png")
except Exception as e:
    print(f"✗ 图1b生成失败: {e}")

# ============================================================================
# 图2: HULL与OT的滞后相关性曲线
# ============================================================================
print("\n[2/8] 绘制图2: HULL和MULL与OT的滞后相关性曲线...")

try:
    # 计算HULL和MULL的滞后相关性
    lags = range(0, 97)  # 0-24小时 (每15分钟一个点)
    hull_correlations = []
    mull_correlations = []

    for lag in lags:
        hull_corr = tx1['HULL'].corr(tx1['OT'].shift(-lag))
        mull_corr = tx1['MULL'].corr(tx1['OT'].shift(-lag))
        hull_correlations.append(hull_corr)
        mull_correlations.append(mull_corr)

    lag_hours = np.array(list(lags)) * 0.25  # 转换为小时

    fig, ax = plt.subplots(figsize=(11, 5))

    # 绘制HULL和MULL曲线
    ax.plot(lag_hours, hull_correlations, linewidth=2.5, color='steelblue',
            label='HULL (High Voltage Reactive Load)', marker='o', markersize=3, markevery=8)
    ax.plot(lag_hours, mull_correlations, linewidth=2.5, color='darkorange',
            label='MULL (Medium Voltage Reactive Load)', marker='s', markersize=3, markevery=8)

    ax.axhline(0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)

    ax.set_xlabel('Lag Time (hours)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Pearson Correlation Coefficient', fontsize=12, fontweight='bold')
    ax.set_title('Lag Correlation Analysis: Reactive Loads vs Oil Temperature',
                 fontsize=14, fontweight='bold', pad=15)
    ax.grid(alpha=0.3, linestyle='--')
    ax.legend(fontsize=10, loc='upper right')

    # 限定纵轴范围到0-0.7，使峰值更明显
    # ax.autoscale(enable=True, axis='y', tight=False)
    ax.set_ylim(0.2, 0.3)
    
    # 标注HULL最大相关性点
    hull_max_idx = np.argmax(hull_correlations)
    hull_max_corr = hull_correlations[hull_max_idx]
    hull_max_lag = lag_hours[hull_max_idx]
    ax.scatter(hull_max_lag, hull_max_corr, s=150, color='red', zorder=5,
               edgecolors='darkred', linewidths=2)
    ax.annotate(f'HULL Peak: r={hull_max_corr:.3f}\nLag={hull_max_lag:.1f}h',
                xy=(hull_max_lag, hull_max_corr),
                xytext=(hull_max_lag -3 , hull_max_corr -0.1),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.8))

    # 标注MULL最大相关性点
    mull_max_idx = np.argmax(mull_correlations)
    mull_max_corr = mull_correlations[mull_max_idx]
    mull_max_lag = lag_hours[mull_max_idx]
    ax.scatter(mull_max_lag, mull_max_corr, s=150, color='darkgreen', zorder=5,
               edgecolors='darkgreen', linewidths=2)
    ax.annotate(f'MULL Peak: r={mull_max_corr:.3f}\nLag={mull_max_lag:.1f}h',
                xy=(mull_max_lag, mull_max_corr),
                xytext=(mull_max_lag + 3, mull_max_corr - 0.12),
                arrowprops=dict(arrowstyle='->', color='darkgreen', lw=2),
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig2_lag_correlation.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 图2已保存: figures/fig2_lag_correlation.png")
except Exception as e:
    print(f"✗ 图2生成失败: {e}")

# ============================================================================
# 图2b: 特征与OT的相关性热力图
# ============================================================================
print("\n[2b/8] 绘制图2b: 特征与OT的相关性热力图...")

try:
    # 选择所有特征和OT
    features = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
    corr_matrix = tx1[features].corr()

    # 只保留与OT的相关性(最后一行)
    ot_corr = corr_matrix['OT'].drop('OT').sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(10, 6))

    # 创建热力图
    sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='RdYlBu_r',
                center=0, vmin=-0.5, vmax=1.0,
                square=True, linewidths=1, cbar_kws={"shrink": 0.8},
                ax=ax, annot_kws={'fontsize': 10, 'fontweight': 'bold'})

    ax.set_title('Feature Correlation Matrix with Oil Temperature (OT)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Features', fontsize=12, fontweight='bold')
    ax.set_ylabel('Features', fontsize=12, fontweight='bold')

    # 旋转标签
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=11)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=11)

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig2b_correlation_heatmap.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 图2b已保存: figures/fig2b_correlation_heatmap.png")
except Exception as e:
    print(f"✗ 图2b生成失败: {e}")

# ============================================================================
# 图3: 数据划分方式示意图 (改进版:测试集更分散)
# ============================================================================
print("\n[3/8] 绘制图3: 数据划分方式示意图...")

try:
    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True)

    # 时序分割
    ax = axes[0]
    ax.barh(0, 80, left=0, height=0.5, color='steelblue', label='Train Set (First 80%)',
            edgecolor='black', linewidth=1.5)
    ax.barh(0, 20, left=80, height=0.5, color='orange', label='Test Set (Last 20%)',
            edgecolor='black', linewidth=1.5)
    ax.set_ylabel('Time-Series\nSplit\n(Correct)', fontsize=10, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([])
    ax.legend(loc='upper right', fontsize=10)
    ax.set_title('Data Split Strategy Comparison: Time-Series vs Sliding Window Random',
                 fontsize=14, fontweight='bold', pad=15)
    ax.text(40, 0, 'Jul 2018 - Jan 2019', ha='center', va='center', fontsize=9, fontweight='bold')
    ax.text(90, 0, 'Feb-Mar\n2019', ha='center', va='center', fontsize=8, fontweight='bold')

    # 滑动窗口随机 - 测试集更分散
    ax = axes[1]
    # 更分散的测试集分布(随机20%)
    np.random.seed(42)
    n_segments = 50  # 增加分段数使测试集更分散
    segment_width = 100 / n_segments
    test_indices = np.random.choice(n_segments, size=int(n_segments * 0.2), replace=False)

    for i in range(n_segments):
        start = i * segment_width
        width = segment_width
        if i in test_indices:
            ax.barh(0, width, left=start, height=0.5, color='orange',
                   edgecolor='black', linewidth=0.3)
        else:
            ax.barh(0, width, left=start, height=0.5, color='steelblue',
                   edgecolor='black', linewidth=0.3)

    ax.set_ylabel('Sliding Window\nRandom Split\n(Data Leakage)', fontsize=10, fontweight='bold', color='red')
    ax.set_xlabel('Time Progress (%)', fontsize=12, fontweight='bold')
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.6, 0.6)
    ax.set_yticks([])

    # 标注泄露区域
    ax.annotate('⚠ Data Leakage Risk\n(Test samples scattered\nacross timeline)',
                xy=(85, 0), xytext=(50, -2),
                arrowprops=dict(arrowstyle='->', color='red', lw=3),
                fontsize=9, fontweight='bold', color='red',
                bbox=dict(boxstyle='round,pad=0.6', facecolor='yellow',
                         alpha=0.8, edgecolor='red', linewidth=2))

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig3_data_split.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 图3已保存: figures/fig3_data_split.png")
except Exception as e:
    print(f"✗ 图3生成失败: {e}")

# ============================================================================
# 图4: 五模型R²对比柱状图(TX1/TX2) - 包含LinearRegression baseline
# ============================================================================
print("\n[4/8] 绘制图4: 五模型R²对比柱状图...")

try:
    models = ['LinearRegression', 'RandomForest', 'MLP', 'RNN', 'Informer']
    tx1_r2 = [-6.95, -4.36, -3.81, -4.30, 0.9735]
    tx2_r2 = [0.72, 0.69, 0.42, 0.42, 0.9728]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(13, 6))
    bars1 = ax.bar(x - width/2, tx1_r2, width, label='TX1 (Industrial)',
                   color='steelblue', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, tx2_r2, width, label='TX2 (Residential)',
                   color='coral', edgecolor='black', linewidth=1.5)

    # 标注负值为红色
    for i, bar in enumerate(bars1):
        if tx1_r2[i] < 0:
            bar.set_color('darkred')
            bar.set_hatch('//')

    ax.set_ylabel('R² (Coefficient of Determination)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax.set_title('Five Model Performance Comparison on TX1/TX2\n(Time-Series Split, 1-hour Prediction, LinearRegression as Baseline)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11, rotation=15, ha='right')
    ax.legend(fontsize=11, loc='upper left')
    ax.axhline(0, color='black', linewidth=1.2, linestyle='--', alpha=0.6)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # 添加数值标签
    for i, (v1, v2) in enumerate(zip(tx1_r2, tx2_r2)):
        ax.text(i - width/2, v1 + (0.2 if v1 > 0 else -0.3), f'{v1:.2f}',
               ha='center', va='bottom' if v1 > 0 else 'top', fontsize=9, fontweight='bold')
        ax.text(i + width/2, v2 + 0.05, f'{v2:.2f}',
               ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig4_model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 图4已保存: figures/fig4_model_comparison.png")
except Exception as e:
    print(f"✗ 图4生成失败: {e}")

# ============================================================================
# 图5: 预测时长vs R²趋势曲线 - 包含LinearRegression baseline
# ============================================================================
print("\n[5/8] 绘制图5: 预测时长vs R²趋势曲线...")

try:
    horizons = ['1 hour', '1 day', '1 week']
    x_pos = [0, 1, 2]

    lr_r2 = [0.72, 0.73, 0.69]  # LinearRegression (baseline)
    rf_r2 = [0.69, 0.64, 0.64]
    mlp_r2 = [0.42, 0.25, 0.39]
    rnn_r2 = [0.42, 0.63, 0.68]
    informer_r2 = [0.9728, 0.6075, -1.9836]

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(x_pos, lr_r2, marker='*', linewidth=3, markersize=15,
            label='LinearRegression (Baseline)', linestyle='-', color='darkblue', alpha=0.8)
    ax.plot(x_pos, rf_r2, marker='o', linewidth=2.5, markersize=10,
            label='RandomForest', linestyle='--', color='green')
    ax.plot(x_pos, mlp_r2, marker='s', linewidth=2.5, markersize=10,
            label='MLP', linestyle='--', color='purple')
    ax.plot(x_pos, rnn_r2, marker='^', linewidth=2.5, markersize=10,
            label='RNN', linestyle='-', color='orange')
    ax.plot(x_pos, informer_r2, marker='D', linewidth=3.5, markersize=12,
            label='Informer', linestyle='-', color='red')

    ax.set_xticks(x_pos)
    ax.set_xticklabels(horizons, fontsize=12)
    ax.set_ylabel('R² (Coefficient of Determination)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Prediction Horizon', fontsize=13, fontweight='bold')
    ax.set_title('Multi-Horizon Performance Trends of Different Models on TX2',
                 fontsize=14, fontweight='bold', pad=15)
    ax.axhline(0, color='black', linewidth=1.2, linestyle='--', alpha=0.6)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(alpha=0.3, linestyle='--')

    # 标注关键点
    ax.annotate('Informer Collapse', xy=(2, -1.98), xytext=(1.5, -1.2),
                arrowprops=dict(arrowstyle='->', color='red', lw=2),
                fontsize=10, fontweight='bold', color='red')
    ax.annotate('LinearRegression\nStable & Best\nfor Mid/Long-term', xy=(1, 0.73), xytext=(0.2, 0.85),
                arrowprops=dict(arrowstyle='->', color='darkblue', lw=2),
                fontsize=9, fontweight='bold', color='darkblue',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='lightcyan', alpha=0.8))

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig5_horizon_trend.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 图5已保存: figures/fig5_horizon_trend.png")
except Exception as e:
    print(f"✗ 图5生成失败: {e}")

# ============================================================================
# 图6: 时间特征影响对比柱状图
# ============================================================================
print("\n[6/8] 绘制图6: 时间特征影响对比柱状图...")

try:
    models = ['RandomForest', 'MLP', 'RNN']
    with_time = [0.6907, 0.4192, 0.4234]
    without_time = [-0.0435, 0.1174, -0.0659]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    bars1 = ax.bar(x - width/2, with_time, width, label='With Time Features (Objective 9)',
                   color='forestgreen', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, without_time, width, label='Without Time Features (Objective 10)',
                   color='lightcoral', edgecolor='black', linewidth=1.5)

    # 标注负值
    for i, val in enumerate(without_time):
        if val < 0:
            bars2[i].set_hatch('//')

    ax.set_ylabel('R² (Coefficient of Determination)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Model', fontsize=13, fontweight='bold')
    ax.set_title('Impact of Time Features on Model Performance\n(TX2, Time-Series Split, 1-hour Prediction)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.legend(fontsize=10, loc='upper left')
    ax.axhline(0, color='black', linewidth=1.2, linestyle='--', alpha=0.6)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # 添加提升幅度标注
    for i, (w, wo) in enumerate(zip(with_time, without_time)):
        improvement = w - wo
        y_pos = max(w, wo) + 0.08
        ax.text(i, y_pos, f'↑ +{improvement:.2f}',
               ha='center', va='bottom', fontsize=10, fontweight='bold', color='darkgreen',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig6_time_feature_impact.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 图6已保存: figures/fig6_time_feature_impact.png")
except Exception as e:
    print(f"✗ 图6生成失败: {e}")

# ============================================================================
# 图7: TX1 vs TX2负载波动性对比
# ============================================================================
print("\n[7/8] 绘制图7: TX1 vs TX2负载波动性对比...")

try:
    features = ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL']

    # 计算变异系数 (CV = std/mean)
    tx1_cv = [tx1[f].std() / tx1[f].mean() for f in features]
    tx2_cv = [tx2[f].std() / tx2[f].mean() for f in features]

    x = np.arange(len(features))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 6))
    bars1 = ax.bar(x - width/2, tx1_cv, width, label='TX1 (Industrial)',
                   color='steelblue', edgecolor='black', linewidth=1.5)
    bars2 = ax.bar(x + width/2, tx2_cv, width, label='TX2 (Residential)',
                   color='coral', edgecolor='black', linewidth=1.5)

    ax.set_ylabel('Coefficient of Variation (CV = σ/μ)', fontsize=13, fontweight='bold')
    ax.set_xlabel('Load Features', fontsize=13, fontweight='bold')
    ax.set_title('Load Volatility Comparison: TX1 vs TX2', fontsize=14, fontweight='bold', pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(features, fontsize=11)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # 添加数值标签
    for i, (v1, v2) in enumerate(zip(tx1_cv, tx2_cv)):
        ax.text(i - width/2, v1 + 0.01, f'{v1:.2f}',
               ha='center', va='bottom', fontsize=9, fontweight='bold')
        ax.text(i + width/2, v2 + 0.01, f'{v2:.2f}',
               ha='center', va='bottom', fontsize=9, fontweight='bold')

    # 添加说明文本
    avg_tx1 = np.mean(tx1_cv)
    avg_tx2 = np.mean(tx2_cv)
    ax.text(0.02, 0.98, f'TX1 Avg CV: {avg_tx1:.3f}\nTX2 Avg CV: {avg_tx2:.3f}\nDifference: {(avg_tx1/avg_tx2-1)*100:.1f}%',
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='lightyellow', alpha=0.8, edgecolor='black', linewidth=1.5))

    plt.tight_layout()
    plt.savefig(FIG_DIR / 'fig7_volatility_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ 图7已保存: figures/fig7_volatility_comparison.png")
except Exception as e:
    print(f"✗ 图7生成失败: {e}")

print("\n" + "=" * 60)
print("✓ All figures generated successfully!")
# ==========================
# 图8: 五模型完整R²对比矩阵 - 包含LinearRegression baseline
# ==========================
print("\n[8a/10] 生成图8a: TX1模型R²对比...")
print("[8b/10] 生成图8b: TX2模型R²对比...")

try:
    # 定义R²数据矩阵 (来自实验结果summary)
    # 格式: {模型: {TX1: [1h, 1d, 1w], TX2: [1h, 1d, 1w]}}
    r2_data = {
        'LinearRegression': {
            'TX1': [-6.9521, -7.1568, -6.9768],  # exp_106, 107, 108 (baseline)
            'TX2': [0.7180, 0.7265, 0.6901]       # exp_109, 110, 111 (baseline)
        },
        'RandomForest': {
            'TX1': [-4.3593, -4.1881, -4.2344],  # exp_073, 074, 075
            'TX2': [0.6907, 0.6433, 0.6362]       # exp_076, 077, 078
        },
        'MLP': {
            'TX1': [-3.8132, -4.0192, -3.3901],  # exp_079, 080, 081
            'TX2': [0.4192, 0.2471, 0.3925]       # exp_082, 083, 084
        },
        'RNN': {
            'TX1': [-4.3009, -5.1718, -6.6095],  # exp_085, 086, 087
            'TX2': [0.4234, 0.6263, 0.6085]       # exp_088, 089, 090
        },
        'Informer': {
            'TX1': [0.9735, 0.6150, -1.5864],    # exp_091, 092, 093
            'TX2': [0.9728, 0.6075, -1.9836]      # exp_094, 095, 096
        }
    }

    horizons = ['1 hour', '1 day', '1 week']
    models = list(r2_data.keys())

    x = np.arange(len(horizons))
    width = 0.16  # 缩小宽度以容纳5个模型
    colors = ['#1f77b4', '#2E86AB', '#A23B72', '#F18F01', '#C73E1D']  # 添加一个颜色给LinearRegression

    figure_settings = [
        {
            'label': '8a',
            'tx': 'TX1',
            'filename': 'fig8a_tx1_model_comparison.png',
            'ylims': (-7.5, 1.2),
            'title': 'TX1 Performance Comparison\n(Objective 9 & 11: Time-Series Split)'
        },
        {
            'label': '8b',
            'tx': 'TX2',
            'filename': 'fig8b_tx2_model_comparison.png',
            'ylims': (-1.0, 1.0),
            'title': 'TX2 Performance Comparison\n(Objective 9 & 11: Time-Series Split)'
        }
    ]

    for setting in figure_settings:
        plt.figure(figsize=(10, 6))
        ax = plt.gca()
        tx = setting['tx']

        for i, model in enumerate(models):
            values = r2_data[model][tx]
            offset = (i - 2) * width  # 调整偏移以居中5个柱子
            bars = ax.bar(
                x + offset,
                values,
                width,
                label=model,
                color=colors[i],
                alpha=0.85,
                edgecolor='black',
                linewidth=0.5
            )

            # 在柱子上标注数值
            for bar, val in zip(bars, values):
                height = bar.get_height()
                text_color = 'black'

                if setting['tx'] == 'TX2' and model == 'Informer' and val < 0:
                    # 将负值标签靠近柱子底部
                    va = 'top'
                    y_pos = -0.1
                    text_color = 'white'
                elif val >= 0:
                    va = 'bottom'
                    y_pos = height + 0.03
                else:
                    va = 'top'
                    y_pos = height - 0.15
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    y_pos,
                    f'{val:.2f}',
                    ha='center',
                    va=va,
                    fontsize=8,
                    fontweight='bold',
                    color=text_color
                )

        ax.set_xlabel('Prediction Horizon', fontsize=13, fontweight='bold')
        ax.set_ylabel('R² Score', fontsize=13, fontweight='bold')
        ax.set_title(setting['title'], fontsize=14, fontweight='bold', pad=15)
        ax.set_xticks(x)
        ax.set_xticklabels(horizons, fontsize=11)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.5)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        ymin, ymax = setting['ylims']
        ax.set_ylim(ymin, ymax)
        ax.axhspan(0, ymax, facecolor='lightgreen', alpha=0.1, zorder=0)
        ax.axhspan(ymin, 0, facecolor='lightcoral', alpha=0.1, zorder=0)

        # 统一保持绘图区长宽比例
        ax.set_box_aspect(0.6)

        # 调整布局，预留右侧空间给图例
        plt.gcf().subplots_adjust(left=0.12, right=0.78, top=0.88, bottom=0.12)

        # 将图例放在图像右侧，避免遮挡主体内容
        legend = ax.legend(
            fontsize=8,
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
            ncol=1,
            frameon=False,
            columnspacing=1.0,
            handlelength=1.4,
            borderaxespad=0.8
        )

        plt.savefig(FIG_DIR / setting['filename'], dpi=300, bbox_inches='tight')
        plt.close()

        print(f"✓ 图{setting['label']}已保存: figures/{setting['filename']}")
except Exception as e:
    print(f"✗ 图8a/8b生成失败: {e}")

print(f"\nFigures saved to: {FIG_DIR}")
print("=" * 60)
print("\nGenerated figures:")
print("  - fig1a_load_comparison_weekly.png")
print("  - fig1b_load_comparison_longterm.png")
print("  - fig2_lag_correlation.png")
print("  - fig2b_correlation_heatmap.png")
print("  - fig3_data_split.png")
print("  - fig4_model_comparison.png")
print("  - fig5_horizon_trend.png")
print("  - fig6_time_feature_impact.png")
print("  - fig7_volatility_comparison.png")
print("  - fig8a_tx1_model_comparison.png")
print("  - fig8b_tx2_model_comparison.png")
print("=" * 60)
