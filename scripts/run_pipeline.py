import os
import shutil
from datetime import datetime
import sys


def ensure_output_dir(base_dir: str) -> str:
    """Create a timestamped output directory under base_dir/artifacts and return its path."""
    artifacts_root = os.path.join(base_dir, 'artifacts')
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir = os.path.join(artifacts_root, f'run_{ts}')
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def find_and_copy_data(data_dir: str, dest_dir: str) -> None:
    """Copy required CSVs (trans_1.csv, trans_2.csv) from data_dir (or data_dir/raw) into dest_dir."""
    candidates = [
        os.path.join(data_dir, 'trans_1.csv'),
        os.path.join(data_dir, 'trans_2.csv'),
        os.path.join(data_dir, 'raw', 'trans_1.csv'),
        os.path.join(data_dir, 'raw', 'trans_2.csv'),
    ]

    found = {}
    for path in candidates:
        name = os.path.basename(path)
        if os.path.exists(path) and os.path.isfile(path):
            found[name] = path

    required = ['trans_1.csv', 'trans_2.csv']
    missing = [f for f in required if f not in found]
    if missing:
        raise FileNotFoundError(
            f"缺少原始数据文件: {', '.join(missing)}。请将 CSV 放在 '{data_dir}' 或 '{data_dir}/raw'。"
        )

    for name, src in found.items():
        dst = os.path.join(dest_dir, name)
        shutil.copy2(src, dst)


def run_preprocessing():
    """Run optimized preprocessing to generate npy splits and scalers."""
    from scripts.preprocessing import optimized_preprocessing as opt
    opt.main()


def run_simple_ml_models():
    """Train simple ML models and write results and model files."""
    from scripts.models import simple_ml_models as sml
    sml.main()


def create_final_comparison_from_simple_results():
    """Create final_model_comparison.csv from simple_ml_results.csv for downstream visualization."""
    import pandas as pd
    src = 'simple_ml_results.csv'
    dst = 'final_model_comparison.csv'
    if os.path.exists(src):
        df = pd.read_csv(src)
        df.to_csv(dst, index=False)
    else:
        raise FileNotFoundError(
            f"未找到 {src}。请确认简化模型训练已完成并生成该文件。"
        )


def run_visualizations():
    """Generate visualization figures: performance, predictions, error distribution."""
    from scripts.evaluation import visualization_analysis as viz
    viz.main()


def run_exploratory():
    """Generate exploratory feature and distribution plots for raw transformers."""
    from scripts.exploratory import transformer_analysis as exp
    exp.main()


def main():
    # Resolve base paths
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    data_root = os.path.join(project_root, 'data')

    # Ensure project root is on sys.path so imports like 'scripts.*' work even after chdir
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    # Create output directory and switch working directory so existing scripts write into it
    output_dir = ensure_output_dir(project_root)
    print(f"输出目录: {output_dir}")

    # Copy data files into output dir to satisfy scripts that use relative paths
    try:
        find_and_copy_data(data_root, output_dir)
    except FileNotFoundError as e:
        print(str(e))
        sys.exit(1)

    # Change working directory to output_dir so all artifacts are written there
    os.chdir(output_dir)
    print(f"当前工作目录切换为: {os.getcwd()}")

    # 1) Preprocessing
    print("开始数据预处理...")
    run_preprocessing()

    # 2) Train and evaluate simple ML models (fast baseline)
    print("开始训练简化版传统ML模型...")
    run_simple_ml_models()

    # 3) Create final comparison CSV for visualization
    print("生成最终模型对比 CSV...")
    create_final_comparison_from_simple_results()

    # 4) Visualizations based on models and test splits
    print("生成可视化图表...")
    run_visualizations()

    # 5) Exploratory analysis on raw data (trends & distributions)
    print("生成原始数据的探索性可视化...")
    run_exploratory()

    print("\n流程完成。所有新生成文件均已保存在:")
    print(output_dir)


if __name__ == '__main__':
    main()