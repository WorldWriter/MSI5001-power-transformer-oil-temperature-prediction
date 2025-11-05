#!/usr/bin/env python3
"""
Extract training time, dataset, and evaluation metrics from experiment logs
and create a summary table matching the experiment_group.csv format.
"""

import os
import re
import json
import pandas as pd
from pathlib import Path

def parse_log_file(log_path):
    """Parse a single log file to extract key information."""
    
    # 初始化提取的字段
    extracted_data = {
        'experiment_id': '',
        'transformer_id': '',
        'model': '',
        'split_method': '',
        'feature_mode': '',
        'test_ratio': '',
        'n_features': '',
        'n_train': '',
        'n_test': '',
        'train_time': '',
        'R2': '',      # 按重要性排序：R2最重要，排在前面
        'MAE': '',
        'MSE': '',
        'RMSE': '',
        'lookback': '',
        'horizon': '',
        'gap': '',
        'lookback_multiplier': '',
        'status': 'unknown',
        'error_message': ''
    }
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取实验ID
        exp_id_match = re.search(r'exp_(\d+)', log_path.name)
        if exp_id_match:
            extracted_data['experiment_id'] = f"exp_{exp_id_match.group(1)}"
        
        # 提取变压器ID
        tx_match = re.search(r'TX(\d+)', content)
        if tx_match:
            extracted_data['transformer_id'] = tx_match.group(1)
        
        # 提取模型类型
        model_match = re.search(r'Model:\s*(\w+)', content)
        if model_match:
            extracted_data['model'] = model_match.group(1)
        
        # 提取数据划分方式
        split_match = re.search(r'Split method:\s*(\w+)', content)
        if split_match:
            extracted_data['split_method'] = split_match.group(1)
        
        # 提取特征模式
        feature_match = re.search(r'Feature mode:\s*(\w+)', content)
        if feature_match:
            extracted_data['feature_mode'] = feature_match.group(1)
        
        # 提取特征数量
        n_features_match = re.search(r'(\d+)\s+features', content)
        if n_features_match:
            extracted_data['n_features'] = n_features_match.group(1)
        
        # 提取训练集和测试集大小
        train_match = re.search(r'Train:\s*(\d+)', content)
        test_match = re.search(r'Test:\s*(\d+)', content)
        if train_match:
            extracted_data['n_train'] = train_match.group(1)
        if test_match:
            extracted_data['n_test'] = test_match.group(1)
        
        # 提取训练时间
        train_time_match = re.search(r'Training time:\s*([\d.]+)', content)
        if train_time_match:
            extracted_data['train_time'] = train_time_match.group(1)
        
        # 检查是否为Informer模型
        is_informer = 'informer' in extracted_data['model'].lower() if extracted_data['model'] else False
        
        if is_informer and 'original scale' in content.lower():
            # 对于Informer模型，优先提取original scale的指标
            print(f"检测到Informer模型，优先提取original scale指标: {log_path.name}")
            
            # 查找original scale部分的指标
            original_scale_section = re.search(r'Metrics after inverse transformation \(original scale\):(.*?)(?=\n\n|$)', content, re.DOTALL)
            
            if original_scale_section:
                original_content = original_scale_section.group(1)
                
                # 从original scale部分提取指标
                r2_match = re.search(r'R2:\s*(-?\d+\.?\d*)', original_content)
                mae_match = re.search(r'MAE:\s*(-?\d+\.?\d*)', original_content)
                rmse_match = re.search(r'RMSE:\s*(-?\d+\.?\d*)', original_content)
                
                if r2_match:
                    extracted_data['R2'] = r2_match.group(1)
                if mae_match:
                    extracted_data['MAE'] = mae_match.group(1)
                if rmse_match:
                    extracted_data['RMSE'] = rmse_match.group(1)
                    
                print(f"  提取到original scale指标 - R2: {extracted_data['R2']}, MAE: {extracted_data['MAE']}, RMSE: {extracted_data['RMSE']}")
        else:
            # 对于非Informer模型或没有original scale的情况，使用原来的提取逻辑
            # R2指标：提取冒号后到行尾的数字（支持负号）
            r2_match = re.search(r'R2:\s*(-?\d+\.?\d*)', content)
            # MAE指标：提取冒号后到行尾的数字（支持负号）
            mae_match = re.search(r'MAE:\s*(-?\d+\.?\d*)', content)
            # RMSE指标：提取冒号后到行尾的数字（支持负号）
            rmse_match = re.search(r'RMSE:\s*(-?\d+\.?\d*)', content)
            
            if r2_match:
                extracted_data['R2'] = r2_match.group(1)
            if mae_match:
                extracted_data['MAE'] = mae_match.group(1)
            if rmse_match:
                extracted_data['RMSE'] = rmse_match.group(1)
        
        # 提取时间窗口参数
        lookback_match = re.search(r'lookback_multiplier=([\d.]+)', content)
        if lookback_match:
            extracted_data['lookback_multiplier'] = lookback_match.group(1)
        
        # 检查实验状态
        if 'Experiment complete' in content:
            extracted_data['status'] = 'success'
        elif 'Error' in content or 'Exception' in content:
            extracted_data['status'] = 'failed'
            # 提取错误信息
            error_match = re.search(r'(Error|Exception):\s*(.+)', content)
            if error_match:
                extracted_data['error_message'] = error_match.group(2).strip()
        
        return extracted_data
        
    except Exception as e:
        extracted_data['status'] = 'error'
        extracted_data['error_message'] = str(e)
        return extracted_data

def load_experiment_config(config_path):
    """Load experiment configuration from CSV."""
    return pd.read_csv(config_path)

def create_summary_table(logs_dir, config_df):
    """Create summary table from log files and config."""
    
    # 获取所有日志文件
    log_files = list(Path(logs_dir).glob('exp_*.log'))
    log_files.sort()
    
    # 解析所有日志文件
    all_data = []
    for log_file in log_files:
        print(f"Processing {log_file.name}...")
        data = parse_log_file(log_file)
        all_data.append(data)
    
    # 创建DataFrame
    results_df = pd.DataFrame(all_data)
    
    # 合并配置信息
    if config_df is not None:
        # 从experiment_id提取验证序号，处理可能的NaN值
        results_df['验证序号'] = results_df['experiment_id'].str.extract(r'(\d+)')
        # 将验证序号转换为数值类型，处理NaN值
        results_df['验证序号'] = pd.to_numeric(results_df['验证序号'], errors='coerce')
        
        # 合并配置信息
        merged_df = pd.merge(
            config_df, 
            results_df, 
            left_on='验证序号', 
            right_on='验证序号', 
            how='right'
        )
        
        # 重命名列以匹配原始格式
        column_mapping = {
            '验证序号': '验证序号',
            '验证目标': '验证目标',
            '验证数据集': '验证数据集',
            '验证模型': '验证模型',
            '数据划分方式': '数据划分方式',
            '异常值剔除': '异常值剔除',
            '特征模式': '特征模式',
            '时间窗口': '时间窗口',
            '预测时长': '预测时长',
            'train_time': '训练时间(秒)',
            'status': '状态',
            'experiment_id': '实验ID',
            'n_train': '训练样本数',
            'n_test': '测试样本数',
            'n_features': '特征数'
        }
        
        # 选择并重命名列
        available_columns = {k: v for k, v in column_mapping.items() if k in merged_df.columns}
        summary_df = merged_df[list(available_columns.keys())].rename(columns=available_columns)
        
        # 确保评估指标列存在 - 按重要性排序：R2, MAE, RMSE（注意：日志中没有MSE）
        for col in ['R2', 'MAE', 'RMSE']:
            if col in results_df.columns and col not in summary_df.columns:
                # 从results_df中复制这些列
                summary_df[col] = results_df[col]
        
        # 添加缺失的列
        for col in ['日志文件']:
            if col not in summary_df.columns:
                summary_df[col] = ''
        
        # 添加日志文件路径
        if '实验ID' in summary_df.columns:
            summary_df['日志文件'] = summary_df['实验ID'].apply(lambda x: f"logs/{x}.log")
        
        # 确保数值列的数据类型正确
        numeric_columns = ['R2', 'MAE', 'RMSE', '训练时间(秒)', '训练样本数', '测试样本数', '特征数']
        for col in numeric_columns:
            if col in summary_df.columns:
                summary_df[col] = pd.to_numeric(summary_df[col], errors='coerce')
        
        return summary_df
    
    else:
        return results_df

def main():
    """Main function."""
    
    # 使用相对路径，从当前脚本位置开始
    current_dir = Path(__file__).parent
    experiment_dir = current_dir
    logs_dir = experiment_dir / 'logs'
    config_path = experiment_dir / 'experiment_group.csv'
    output_path = experiment_dir / 'experiment_results_summary.csv'
    
    print("开始提取实验日志信息...")
    
    # 加载配置文件
    try:
        config_df = load_experiment_config(config_path)
        print(f"成功加载配置文件，共{len(config_df)}条实验配置")
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        config_df = None
    
    # 创建汇总表
    summary_df = create_summary_table(logs_dir, config_df)
    
    # 保存结果
    summary_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"汇总表已保存至: {output_path}")
    
    # 显示统计信息
    print("\n=== 实验结果统计 ===")
    if '状态' in summary_df.columns:
        status_counts = summary_df['状态'].value_counts()
        print("实验状态分布:")
        for status, count in status_counts.items():
            print(f"  {status}: {count}")
    
    if '验证模型' in summary_df.columns and 'R2' in summary_df.columns:
        # 按模型类型统计
        model_stats = summary_df.groupby('验证模型')['R2'].agg(['count', 'mean', 'std'])
        print("\n按模型类型的R²统计:")
        print(model_stats)
    
    if '验证数据集' in summary_df.columns and 'R2' in summary_df.columns:
        # 按变压器统计
        transformer_stats = summary_df.groupby('验证数据集')['R2'].agg(['count', 'mean', 'std'])
        print("\n按变压器数据集的R²统计:")
        print(transformer_stats)
    
    return summary_df

if __name__ == "__main__":
    summary_df = main()