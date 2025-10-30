"""
Batch experiment runner for systematic parameter studies.

This script reads experiment configurations from CSV and runs them automatically,
collecting all results into a summary file.

Usage:
    # Run all experiments
    python -m scripts.run_experiments --config experiment/experiment_group.csv

    # Run specific experiments
    python -m scripts.run_experiments --config experiment/experiment_group.csv --exp-ids 1,2,3

    # Dry run (show commands without executing)
    python -m scripts.run_experiments --config experiment/experiment_group.csv --dry-run

    # Run with preprocessing
    python -m scripts.run_experiments --config experiment/experiment_group.csv --run-preprocessing
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


# Mapping from Chinese descriptions to parameter values
CONFIG_MAPPINGS = {
    "split_method": {
        "时序分割": "chronological",
        "时序分割: 前80%训练, 后20%测试": "chronological",
        "滑动窗口随机": "random_window",
        "滑动窗口随机: 形成滑动窗口数据对, 随机80%训练, 20%测试": "random_window",
        "分组随机": "group_random",
        "分组随机: 先将数据随机分成100组, 随机80%组训练, 20%组测试": "group_random",
    },
    "outlier_method": {
        "无": "none",
        "默认-IQR": "iqr",
        "最多0.5%": "percentile",
        "最多1%": "percentile",
        "最多5%": "percentile",
    },
    "outlier_percentile": {
        "无": 0.0,
        "默认-IQR": None,
        "最多0.5%": 0.5,
        "最多1%": 1.0,
        "最多5%": 5.0,
    },
    "data_suffix": {
        "无": "_no_outlier",
        "默认-IQR": "",
        "最多0.5%": "_0.5pct",
        "最多1%": "_1pct",
        "最多5%": "_5pct",
    },
    "feature_mode": {
        "加入年月日等特征": "full",
        "不加入年月日等特征": "no_time",
        "仅使用年月日等特征": "time_only",
    },
    "window_multiplier": {
        "固定时间窗口-1倍": 1.0,
        "固定时间窗口-2倍": 2.0,
        "固定时间窗口-4倍": 4.0,
        "固定时间窗口-6倍": 6.0,
        "固定时间窗口-8倍": 8.0,
    },
    "horizon": {
        "1 hour": 1,
        "1 day": 24,
        "1 week": 168,
    }
}


def parse_experiment_row(row: pd.Series) -> Dict:
    """
    Parse a row from experiment CSV into training parameters.

    Parameters
    ----------
    row : pd.Series
        Row from experiment configuration CSV

    Returns
    -------
    Dict
        Parsed experiment parameters
    """
    exp_id = int(row["验证序号"])
    exp_name = str(row["验证目标"])

    # Extract transformer ID (TX1 or TX2)
    tx_str = str(row["验证数据集"])
    if "TX1" in tx_str.upper():
        tx_id = 1
    elif "TX2" in tx_str.upper():
        tx_id = 2
    else:
        tx_id = 1  # default

    # Extract model name
    model_str = str(row["验证模型"])
    model = model_str  # RandomForest, MLP, etc.

    # Parse split method
    split_str = str(row["数据划分方式"])
    split_method = CONFIG_MAPPINGS["split_method"].get(split_str, "random_window")

    # Parse outlier method
    outlier_str = str(row["异常值剔除"])
    outlier_method = CONFIG_MAPPINGS["outlier_method"].get(outlier_str, "iqr")
    outlier_percentile = CONFIG_MAPPINGS["outlier_percentile"].get(outlier_str, None)
    data_suffix = CONFIG_MAPPINGS["data_suffix"].get(outlier_str, "")

    # Parse feature mode
    feature_str = str(row["有无时间特征-日/周/月/年的整体变化趋势图"])
    feature_mode = CONFIG_MAPPINGS["feature_mode"].get(feature_str, "full")

    # Parse window configuration
    window_str = str(row["时间窗口长度"])
    lookback_multiplier = CONFIG_MAPPINGS["window_multiplier"].get(window_str, 4.0)

    horizon_str = str(row["预测时长"])
    horizon = CONFIG_MAPPINGS["horizon"].get(horizon_str, 1)

    return {
        "exp_id": exp_id,
        "exp_name": exp_name,
        "tx_id": tx_id,
        "model": model,
        "split_method": split_method,
        "outlier_method": outlier_method,
        "outlier_percentile": outlier_percentile,
        "data_suffix": data_suffix,
        "feature_mode": feature_mode,
        "lookback_multiplier": lookback_multiplier,
        "horizon": horizon,
    }


def build_preprocessing_command(params: Dict) -> List[str]:
    """Build preprocessing command from parameters."""
    cmd = [
        sys.executable, "-m", "scripts.preprocessing_configurable",
        "--outlier-method", params["outlier_method"],
    ]

    if params["outlier_percentile"] is not None:
        cmd.extend(["--outlier-percentile", str(params["outlier_percentile"])])

    if params["data_suffix"]:
        cmd.extend(["--save-suffix", params["data_suffix"]])

    return cmd


def build_training_command(params: Dict) -> List[str]:
    """Build training command from parameters."""
    exp_name = f"exp_{params['exp_id']:03d}"

    cmd = [
        sys.executable, "-m", "scripts.train_configurable",
        "--tx-id", str(params["tx_id"]),
        "--model", params["model"],
        "--split-method", params["split_method"],
        "--feature-mode", params["feature_mode"],
        "--lookback-multiplier", str(params["lookback_multiplier"]),
        "--horizon", str(params["horizon"]),
        "--experiment-name", exp_name,
    ]

    if params["data_suffix"]:
        cmd.extend(["--data-suffix", params["data_suffix"]])

    return cmd


def run_command(cmd: List[str], dry_run: bool = False, log_file: Optional[Path] = None) -> bool:
    """
    Run a command and return success status.

    Parameters
    ----------
    cmd : List[str]
        Command to run
    dry_run : bool
        If True, only print command without running
    log_file : Optional[Path]
        If provided, capture output to this log file

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    cmd_str = " ".join(cmd)

    if dry_run:
        print(f"[DRY RUN] {cmd_str}")
        return True

    print(f"\nRunning: {cmd_str}")
    if log_file:
        print(f"  Logging to: {log_file}")

    try:
        if log_file:
            # Ensure log directory exists
            log_file.parent.mkdir(parents=True, exist_ok=True)

            # Capture output to both console and log file
            with open(log_file, "w") as f:
                # Write header to log file
                f.write(f"{'='*70}\n")
                f.write(f"Experiment Log\n")
                f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Command: {cmd_str}\n")
                f.write(f"{'='*70}\n\n")
                f.flush()

                # Run command and capture output
                result = subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True
                )

                # Write output to log file
                f.write(result.stdout)
                f.write(f"\n{'='*70}\n")
                f.write(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Return code: {result.returncode}\n")
                f.write(f"{'='*70}\n")

                # Also print to console
                print(result.stdout)

                if result.returncode != 0:
                    raise subprocess.CalledProcessError(result.returncode, cmd)

                return True
        else:
            result = subprocess.run(cmd, check=True, capture_output=False)
            return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Command failed with return code {e.returncode}")
        return False
    except Exception as e:
        print(f"ERROR: {e}")
        return False


def collect_results(output_dir: Path) -> pd.DataFrame:
    """
    Collect all experiment results into a summary dataframe.

    Parameters
    ----------
    output_dir : Path
        Directory containing experiment results

    Returns
    -------
    pd.DataFrame
        Summary of all experiments
    """
    import json

    results = []

    for metrics_file in output_dir.glob("exp_*_metrics.json"):
        try:
            with open(metrics_file) as f:
                metrics = json.load(f)
                results.append(metrics)
        except Exception as e:
            print(f"Warning: Could not read {metrics_file}: {e}")

    if not results:
        return pd.DataFrame()

    return pd.DataFrame(results).sort_values("experiment_id")


def create_metrics_summary(
    experiments: List[Dict],
    exp_df: pd.DataFrame,
    output_dir: Path,
    summary_path: Path
) -> None:
    """
    Create a comprehensive metrics summary CSV combining experiment config and results.

    Parameters
    ----------
    experiments : List[Dict]
        List of parsed experiment parameters
    exp_df : pd.DataFrame
        Original experiment configuration dataframe
    output_dir : Path
        Directory containing experiment results (models/experiments/)
    summary_path : Path
        Path to save the metrics summary CSV (experiment/metrics_summary.csv)
    """
    summary_data = []

    for params in experiments:
        exp_id = params["exp_id"]
        exp_name = f"exp_{exp_id:03d}"

        # Get original row from CSV
        row = exp_df[exp_df["验证序号"] == exp_id].iloc[0]

        # Read metrics from results
        metrics_file = output_dir / f"{exp_name}_metrics.json"

        metrics_dict = {
            "验证序号": exp_id,
            "验证目标": row["验证目标"],
            "验证数据集": row["验证数据集"],
            "验证模型": params["model"],
            "数据划分方式": row["数据划分方式"],
            "异常值剔除": row["异常值剔除"],
            "特征模式": row["有无时间特征-日/周/月/年的整体变化趋势图"],
            "时间窗口": row["时间窗口长度"],
            "预测时长": row["预测时长"],
        }

        # Try to read metrics
        if metrics_file.exists():
            try:
                with open(metrics_file) as f:
                    metrics = json.load(f)

                rmse = metrics.get("RMSE", None)
                mse = metrics.get("MSE", None)
                # Calculate MSE from RMSE if not present
                if mse is None and rmse is not None:
                    mse = rmse ** 2

                metrics_dict.update({
                    "RMSE": rmse,
                    "MAE": metrics.get("MAE", None),
                    "MSE": mse,
                    "R2": metrics.get("R2", None),
                    "训练时间(秒)": metrics.get("train_time", None),
                    "状态": "success",
                    "日志文件": f"logs/{exp_name}.log"
                })
            except Exception as e:
                metrics_dict.update({
                    "RMSE": None,
                    "MAE": None,
                    "MSE": None,
                    "R2": None,
                    "训练时间(秒)": None,
                    "状态": f"error: {e}",
                    "日志文件": f"logs/{exp_name}.log"
                })
        else:
            metrics_dict.update({
                "RMSE": None,
                "MAE": None,
                "MSE": None,
                "R2": None,
                "训练时间(秒)": None,
                "状态": "not_run",
                "日志文件": f"logs/{exp_name}.log"
            })

        summary_data.append(metrics_dict)

    # Create DataFrame and save
    summary_df = pd.DataFrame(summary_data)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(summary_path, index=False, encoding="utf-8-sig")

    print(f"\nMetrics summary saved to: {summary_path}")
    print(f"\nKey metrics for completed experiments:")

    # Show only successful experiments
    success_df = summary_df[summary_df["状态"] == "success"]
    if not success_df.empty:
        print("\n" + success_df[["验证序号", "验证模型", "RMSE", "MAE", "R2"]].to_string(index=False))
    else:
        print("  No successful experiments found.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch experiment runner"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment configuration CSV"
    )
    parser.add_argument(
        "--exp-ids",
        type=str,
        default="10,28,46",
        help="Comma-separated list of experiment IDs to run (default: '10,28,46')"
    )
    parser.add_argument(
        "--run-preprocessing",
        action="store_true",
        help="Run preprocessing for each unique outlier configuration"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show commands without executing them"
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running experiments even if one fails"
    )

    args = parser.parse_args()

    # Load experiment configuration
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Configuration file not found: {config_path}")
        sys.exit(1)

    print(f"Loading experiment configuration from: {config_path}")
    exp_df = pd.read_csv(config_path)

    # Filter experiments (default to 10, 28, 46)
    exp_ids = [int(x.strip()) for x in args.exp_ids.split(",")]
    exp_df = exp_df[exp_df["验证序号"].isin(exp_ids)]
    print(f"Running {len(exp_df)} experiments: {exp_ids}")

    # Parse all experiments
    experiments = []
    for idx, row in exp_df.iterrows():
        try:
            params = parse_experiment_row(row)
            experiments.append(params)
        except Exception as e:
            print(f"Warning: Could not parse experiment {row.get('验证序号', idx)}: {e}")

    print(f"\nParsed {len(experiments)} experiments successfully")

    # Run preprocessing if requested
    if args.run_preprocessing:
        print("\n" + "="*70)
        print("PREPROCESSING")
        print("="*70)

        # Get unique preprocessing configurations
        unique_configs = {}
        for params in experiments:
            key = (params["outlier_method"], params["outlier_percentile"], params["data_suffix"])
            if key not in unique_configs:
                unique_configs[key] = params

        print(f"\nFound {len(unique_configs)} unique preprocessing configurations")

        for i, params in enumerate(unique_configs.values(), 1):
            print(f"\n[{i}/{len(unique_configs)}] Preprocessing: "
                  f"{params['outlier_method']}, suffix={params['data_suffix']}")

            cmd = build_preprocessing_command(params)
            success = run_command(cmd, args.dry_run)

            if not success and not args.continue_on_error:
                print("\nPreprocessing failed. Stopping.")
                sys.exit(1)

    # Run training experiments
    print("\n" + "="*70)
    print("TRAINING EXPERIMENTS")
    print("="*70)

    success_count = 0
    failed_experiments = []

    # Get project root for log file paths
    project_root = Path(__file__).resolve().parents[1]

    for i, params in enumerate(experiments, 1):
        print(f"\n{'='*70}")
        print(f"Experiment {i}/{len(experiments)}: ID={params['exp_id']}")
        print(f"  {params['exp_name']}")
        print(f"  TX{params['tx_id']}, {params['model']}, {params['split_method']}")
        print(f"{'='*70}")

        cmd = build_training_command(params)

        # Create log file path for this experiment
        exp_name = f"exp_{params['exp_id']:03d}"
        log_file = project_root / "experiment" / "logs" / f"{exp_name}.log"

        success = run_command(cmd, args.dry_run, log_file=log_file if not args.dry_run else None)

        if success:
            success_count += 1
        else:
            failed_experiments.append(params['exp_id'])
            if not args.continue_on_error:
                print(f"\nExperiment {params['exp_id']} failed. Stopping.")
                sys.exit(1)

    # Collect and summarize results
    if not args.dry_run:
        print("\n" + "="*70)
        print("COLLECTING RESULTS")
        print("="*70)

        output_dir = Path(__file__).resolve().parents[1] / "models" / "experiments"
        summary_df = collect_results(output_dir)

        if not summary_df.empty:
            summary_path = output_dir / "experiment_summary.csv"
            summary_df.to_csv(summary_path, index=False)
            print(f"\nSaved summary to: {summary_path}")

            print("\nSummary statistics:")
            print(f"  Total experiments: {len(summary_df)}")
            print(f"  Average RMSE: {summary_df['RMSE'].mean():.4f}")
            print(f"  Average MAE: {summary_df['MAE'].mean():.4f}")
            print(f"  Average R²: {summary_df['R2'].mean():.4f}")

            print("\nTop 5 experiments by R²:")
            top5 = summary_df.nlargest(5, 'R2')[['experiment_id', 'model', 'split_method', 'R2', 'RMSE']]
            print(top5.to_string(index=False))
        else:
            print("\nNo results found to summarize.")

        # Create comprehensive metrics summary in experiment folder
        print("\n" + "="*70)
        print("CREATING METRICS SUMMARY")
        print("="*70)

        metrics_summary_path = project_root / "experiment" / "metrics_summary.csv"
        create_metrics_summary(experiments, exp_df, output_dir, metrics_summary_path)

    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"Total experiments: {len(experiments)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {len(failed_experiments)}")

    if failed_experiments:
        print(f"Failed experiment IDs: {failed_experiments}")

    if not args.dry_run and success_count == len(experiments):
        print("\n✓ All experiments completed successfully!")
    elif args.dry_run:
        print("\n✓ Dry run completed (no experiments were actually run)")


if __name__ == "__main__":
    main()
