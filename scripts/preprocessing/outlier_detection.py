try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy import stats
    import warnings
    warnings.filterwarnings('ignore')
except ImportError as e:
    print(f"导入错误: {e}")
    print("请确保已安装所需的库:")
    print("pip install pandas numpy matplotlib seaborn scipy")
    exit(1)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

class OutlierDetector:
    """异常值检测器"""
    
    def __init__(self, data, feature_cols=None):
        """
        初始化异常值检测器
        
        Args:
            data: DataFrame，输入数据
            feature_cols: list，要检测的特征列名
        """
        self.data = data.copy()
        self.feature_cols = feature_cols or ['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'OT']
        self.outlier_indices = {}
        
    def detect_iqr_outliers(self, column, multiplier=1.5):
        """
        使用IQR方法检测异常值
        
        Args:
            column: str，列名
            multiplier: float，IQR倍数（默认1.5）
            
        Returns:
            异常值的索引
        """
        Q1 = self.data[column].quantile(0.25)
        Q3 = self.data[column].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        outliers = self.data[(self.data[column] < lower_bound) | 
                           (self.data[column] > upper_bound)].index
        
        return outliers, lower_bound, upper_bound
    
    def detect_zscore_outliers(self, column, threshold=3):
        """
        使用Z-Score方法检测异常值
        
        Args:
            column: str，列名
            threshold: float，Z-Score阈值（默认3）
            
        Returns:
            异常值的索引
        """
        z_scores = np.abs(stats.zscore(self.data[column].dropna()))
        outliers = self.data[z_scores > threshold].index
        
        return outliers
    
    def detect_modified_zscore_outliers(self, column, threshold=3.5):
        """
        使用修正Z-Score方法检测异常值（基于中位数，更稳健）
        
        Args:
            column: str，列名
            threshold: float，修正Z-Score阈值（默认3.5）
            
        Returns:
            异常值的索引
        """
        median = self.data[column].median()
        mad = np.median(np.abs(self.data[column] - median))
        
        # 避免除零
        if mad == 0:
            mad = np.std(self.data[column])
            
        modified_z_scores = 0.6745 * (self.data[column] - median) / mad
        outliers = self.data[np.abs(modified_z_scores) > threshold].index
        
        return outliers
    
    def detect_all_outliers(self):
        """检测所有特征的异常值"""
        results = {}
        
        for col in self.feature_cols:
            if col in self.data.columns:
                print(f"\n=== {col} 异常值检测 ===")
                
                # IQR方法
                iqr_outliers, lower, upper = self.detect_iqr_outliers(col)
                print(f"IQR方法: 发现 {len(iqr_outliers)} 个异常值")
                print(f"  正常范围: [{lower:.2f}, {upper:.2f}]")
                
                # Z-Score方法
                zscore_outliers = self.detect_zscore_outliers(col)
                print(f"Z-Score方法: 发现 {len(zscore_outliers)} 个异常值")
                
                # 修正Z-Score方法
                modified_zscore_outliers = self.detect_modified_zscore_outliers(col)
                print(f"修正Z-Score方法: 发现 {len(modified_zscore_outliers)} 个异常值")
                
                results[col] = {
                    'iqr_outliers': iqr_outliers,
                    'zscore_outliers': zscore_outliers,
                    'modified_zscore_outliers': modified_zscore_outliers,
                    'bounds': (lower, upper)
                }
        
        self.outlier_indices = results
        return results
    
    def visualize_outliers(self, column, save_fig=True):
        """可视化异常值"""
        if column not in self.outlier_indices:
            print(f"请先运行 detect_all_outliers() 方法")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{column} 异常值可视化', fontsize=16)
        
        # 1. 箱线图
        axes[0, 0].boxplot(self.data[column].dropna())
        axes[0, 0].set_title('箱线图')
        axes[0, 0].set_ylabel(column)
        
        # 2. 直方图
        axes[0, 1].hist(self.data[column].dropna(), bins=50, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('分布直方图')
        axes[0, 1].set_xlabel(column)
        axes[0, 1].set_ylabel('频次')
        
        # 3. 散点图（时间序列）
        if 'date' in self.data.columns:
            axes[1, 0].scatter(range(len(self.data)), self.data[column], alpha=0.6, s=1)
            
            # 标记异常值
            iqr_outliers = self.outlier_indices[column]['iqr_outliers']
            if len(iqr_outliers) > 0:
                axes[1, 0].scatter(iqr_outliers, self.data.loc[iqr_outliers, column], 
                                 color='red', s=10, label=f'IQR异常值 ({len(iqr_outliers)}个)')
            
            axes[1, 0].set_title('时间序列散点图')
            axes[1, 0].set_xlabel('时间索引')
            axes[1, 0].set_ylabel(column)
            axes[1, 0].legend()
        
        # 4. Q-Q图
        stats.probplot(self.data[column].dropna(), dist="norm", plot=axes[1, 1])
        axes[1, 1].set_title('Q-Q图（正态性检验）')
        
        plt.tight_layout()
        
        if save_fig:
            plt.savefig(f'{column}_outlier_analysis.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_outlier_summary(self):
        """获取异常值检测摘要"""
        if not self.outlier_indices:
            print("请先运行 detect_all_outliers() 方法")
            return
            
        print(f"\n{'='*60}")
        print("异常值检测摘要")
        print('='*60)
        
        summary_data = []
        for col in self.feature_cols:
            if col in self.outlier_indices:
                iqr_count = len(self.outlier_indices[col]['iqr_outliers'])
                zscore_count = len(self.outlier_indices[col]['zscore_outliers'])
                modified_zscore_count = len(self.outlier_indices[col]['modified_zscore_outliers'])
                
                summary_data.append({
                    '特征': col,
                    'IQR异常值': iqr_count,
                    'Z-Score异常值': zscore_count,
                    '修正Z-Score异常值': modified_zscore_count,
                    '异常值比例(IQR)': f"{iqr_count/len(self.data)*100:.2f}%"
                })
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        return summary_df
    
    def remove_outliers(self, method='iqr', columns=None):
        """
        移除异常值
        
        Args:
            method: str，移除方法 ('iqr', 'zscore', 'modified_zscore')
            columns: list，要处理的列名，None表示所有列
            
        Returns:
            清理后的数据
        """
        if not self.outlier_indices:
            print("请先运行 detect_all_outliers() 方法")
            return self.data
            
        columns = columns or self.feature_cols
        all_outliers = set()
        
        for col in columns:
            if col in self.outlier_indices:
                if method == 'iqr':
                    outliers = self.outlier_indices[col]['iqr_outliers']
                elif method == 'zscore':
                    outliers = self.outlier_indices[col]['zscore_outliers']
                elif method == 'modified_zscore':
                    outliers = self.outlier_indices[col]['modified_zscore_outliers']
                else:
                    print(f"未知方法: {method}")
                    continue
                    
                all_outliers.update(outliers)
        
        cleaned_data = self.data.drop(index=all_outliers)
        
        print(f"原始数据: {len(self.data)} 行")
        print(f"移除异常值: {len(all_outliers)} 行")
        print(f"清理后数据: {len(cleaned_data)} 行")
        print(f"数据保留率: {len(cleaned_data)/len(self.data)*100:.2f}%")
        
        return cleaned_data

def main():
    """主函数 - 演示异常值检测"""
    try:
        # 读取数据
        trans1 = pd.read_csv('data/trans_1.csv')
        trans2 = pd.read_csv('data/trans_2.csv')
        
        # 合并数据
        combined_data = pd.concat([trans1, trans2], ignore_index=True)
        print(f"加载数据成功，总计 {len(combined_data)} 行")
        
    except FileNotFoundError:
        print("数据文件未找到，请确保 trans_1.csv 和 trans_2.csv 在 data 文件夹中")
        return
    
    # 创建异常值检测器
    detector = OutlierDetector(combined_data)
    
    # 检测异常值
    print("开始异常值检测...")
    outlier_results = detector.detect_all_outliers()
    
    # 显示摘要
    summary = detector.get_outlier_summary()
    
    # 可视化几个关键特征的异常值
    key_features = ['OT', 'HUFL', 'MUFL']
    for feature in key_features:
        if feature in combined_data.columns:
            print(f"\n生成 {feature} 的异常值可视化...")
            detector.visualize_outliers(feature)
    
    # 演示异常值处理
    print(f"\n{'='*60}")
    print("异常值处理示例")
    print('='*60)
    
    # 使用IQR方法清理数据
    cleaned_data_iqr = detector.remove_outliers(method='iqr', columns=['OT'])
    
    # 保存清理后的数据
    cleaned_data_iqr.to_csv('cleaned_data_no_outliers.csv', index=False)
    print("清理后的数据已保存为 cleaned_data_no_outliers.csv")

if __name__ == "__main__":
    main()