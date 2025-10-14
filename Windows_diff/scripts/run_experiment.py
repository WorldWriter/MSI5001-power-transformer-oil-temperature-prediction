#!/usr/bin/env python3
"""
时间窗口长度实验 - 主运行脚本

本脚本按顺序执行完整的时间窗口长度实验流程:
1. 数据预处理和序列生成
2. 模型训练
3. 结果分析和报告生成

作者: MSI5001项目组
日期: 2024年
"""

import os
import sys
import subprocess
import time
from datetime import datetime

def run_script(script_name, description):
    """运行指定的Python脚本"""
    print(f"\n{'='*60}")
    print(f"开始执行: {description}")
    print(f"脚本: {script_name}")
    print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print('='*60)
    
    start_time = time.time()
    
    try:
        # 运行脚本
        result = subprocess.run([sys.executable, script_name], 
                              capture_output=True, 
                              text=True, 
                              cwd=os.path.dirname(os.path.abspath(__file__)))
        
        execution_time = time.time() - start_time
        
        if result.returncode == 0:
            print("✓ 执行成功!")
            print(f"✓ 执行时间: {execution_time:.2f} 秒")
            
            # 显示输出
            if result.stdout:
                print("\n--- 输出信息 ---")
                print(result.stdout)
            
            return True
        else:
            print("❌ 执行失败!")
            print(f"❌ 错误代码: {result.returncode}")
            
            if result.stderr:
                print("\n--- 错误信息 ---")
                print(result.stderr)
            
            if result.stdout:
                print("\n--- 输出信息 ---")
                print(result.stdout)
            
            return False
            
    except Exception as e:
        execution_time = time.time() - start_time
        print(f"❌ 执行异常: {str(e)}")
        print(f"❌ 执行时间: {execution_time:.2f} 秒")
        return False

def check_prerequisites():
    """检查运行前提条件"""
    print("=== 检查运行前提条件 ===")
    
    # 检查artifacts目录中的预处理数据
    artifacts_dirs = [
        "../../artifacts/run_20251008_200030",
        "../../artifacts/run_20251008_195045", 
        "../../artifacts/run_20251008_194953",
        "../../artifacts/run_20251008_194826"
    ]
    
    data_found = False
    for artifacts_dir in artifacts_dirs:
        required_files = [
            f"{artifacts_dir}/X_train_1h.npy",
            f"{artifacts_dir}/y_train_1h.npy",
            f"{artifacts_dir}/scaler_1h.pkl"
        ]
        
        if all(os.path.exists(f) for f in required_files):
            print(f"✓ 找到预处理数据: {artifacts_dir}")
            data_found = True
            break
    
    if not data_found:
        print("❌ 未找到预处理数据文件")
        print("请先运行主项目的数据预处理脚本或run_pipeline.py")
        return False
    
    # 检查脚本文件
    required_scripts = [
        "window_experiment_preprocessing.py",
        "window_experiment_models.py", 
        "window_experiment_analysis.py"
    ]
    
    for script in required_scripts:
        if not os.path.exists(script):
            print(f"❌ 未找到脚本: {script}")
            return False
        else:
            print(f"✓ 脚本存在: {script}")
    
    # 检查目录结构
    required_dirs = ["../results", "../models", "../docs", "../visualizations"]
    for dir_path in required_dirs:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
            print(f"✓ 创建目录: {dir_path}")
        else:
            print(f"✓ 目录存在: {dir_path}")
    
    return True

def main():
    """主函数 - 执行完整的实验流程"""
    print("=== 时间窗口长度实验 - 完整流程执行 ===")
    print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 记录总开始时间
    total_start_time = time.time()
    
    # 检查前提条件
    if not check_prerequisites():
        print("\n❌ 前提条件检查失败，实验终止")
        return
    
    print("\n✓ 前提条件检查通过，开始执行实验...")
    
    # 实验步骤
    steps = [
        {
            'script': 'window_experiment_preprocessing.py',
            'description': '数据预处理和序列生成',
            'required': True
        },
        {
            'script': 'window_experiment_models.py', 
            'description': '模型训练',
            'required': True
        },
        {
            'script': 'window_experiment_analysis.py',
            'description': '结果分析和报告生成', 
            'required': True
        }
    ]
    
    # 执行各个步骤
    success_count = 0
    
    for i, step in enumerate(steps, 1):
        print(f"\n🚀 步骤 {i}/{len(steps)}: {step['description']}")
        
        success = run_script(step['script'], step['description'])
        
        if success:
            success_count += 1
            print(f"✅ 步骤 {i} 完成")
        else:
            print(f"❌ 步骤 {i} 失败")
            
            if step['required']:
                print(f"❌ 关键步骤失败，实验终止")
                break
            else:
                print(f"⚠️  非关键步骤失败，继续执行")
    
    # 计算总执行时间
    total_execution_time = time.time() - total_start_time
    
    # 输出最终结果
    print(f"\n{'='*60}")
    print("=== 实验执行完成 ===")
    print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总执行时间: {total_execution_time:.2f} 秒 ({total_execution_time/60:.1f} 分钟)")
    print(f"成功步骤: {success_count}/{len(steps)}")
    
    if success_count == len(steps):
        print("\n🎉 实验全部成功完成!")
        print("\n📊 生成的文件:")
        print("  - ../results/experiment_config_summary.csv (实验配置摘要)")
        print("  - ../results/model_training_results.csv (模型训练结果)")
        print("  - ../docs/window_length_analysis_report.md (详细分析报告)")
        print("  - ../visualizations/*.png (性能对比图表)")
        print("  - ../models/*/ (训练好的模型文件)")
        
        print("\n📈 主要发现:")
        print("  请查看生成的分析报告了解详细结果")
        
        print("\n🔍 下一步:")
        print("  1. 查看 ../docs/window_length_analysis_report.md 了解详细分析")
        print("  2. 查看 ../visualizations/ 中的图表")
        print("  3. 根据结果优化生产环境的时间窗口配置")
        
    else:
        print(f"\n⚠️  实验部分完成 ({success_count}/{len(steps)} 步骤成功)")
        print("请检查失败步骤的错误信息并重新运行")
    
    print(f"\n{'='*60}")

if __name__ == "__main__":
    main()