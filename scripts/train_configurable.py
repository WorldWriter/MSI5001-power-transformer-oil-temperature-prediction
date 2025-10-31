"""
Configurable training script for systematic experiments.

This script provides a unified interface for training models with various
configurations for data splitting, feature selection, and window parameters.

Usage Examples:
    # Time-series split with RandomForest
    python -m scripts.train_configurable --tx-id 1 --model RandomForest \\
        --split-method chronological --feature-mode full

    # Random window split with MLP
    python -m scripts.train_configurable --tx-id 1 --model MLP \\
        --split-method random_window --lookback-multiplier 4 --horizon 1

    # Group random split with custom features
    python -m scripts.train_configurable --tx-id 1 --model Ridge \\
        --split-method group_random --feature-mode time_only

    # Use preprocessed data with specific suffix
    python -m scripts.train_configurable --tx-id 1 --model RandomForest \\
        --data-suffix "_1pct" --split-method random_window
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from models.pytorch_mlp import PyTorchMLPRegressor
from models.pytorch_rnn import PyTorchRNNRegressor
try:
    from models.pytorch_informer import PyTorchInformerRegressor
    INFORMER_AVAILABLE = True
except ImportError:
    INFORMER_AVAILABLE = False

from .common import (
    CLEAN_DIR,
    FIG_DIR,
    TABLE_DIR,
    TARGET_COL,
    add_time_features,
)
from .experiment_utils import (
    group_random_split,
    select_features_by_mode,
    WindowConfig,
    create_window_config,
)


# Model builders with default hyperparameters
MODEL_BUILDERS = {
    "RandomForest": lambda: RandomForestRegressor(
        n_estimators=120,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    ),
    "MLP": lambda: PyTorchMLPRegressor(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        learning_rate_init=1e-3,
        alpha=1e-4,
        max_iter=120,
        batch_size=64,
        random_state=42,
        early_stopping=True,
        verbose=False,
        device="auto",
    ),
    "RNN": lambda: PyTorchRNNRegressor(
        hidden_size=64,
        num_layers=2,
        dropout=0.2,
        bidirectional=False,
        learning_rate_init=1e-3,
        max_iter=120,
        batch_size=32,
        random_state=42,
        early_stopping=True,
        verbose=False,
        device="auto",
    ),
    # Informer variants for different horizons (GPU memory optimized)
    "Informer-Short": lambda: PyTorchInformerRegressor(
        # Short-term prediction (1 hour horizon)
        seq_len=32,        # 32 hours lookback
        label_len=16,      # 16 hours decoder start
        pred_len=1,        # Single-step prediction
        # Model architecture (compact for short sequences)
        d_model=256,       # Reduced from 512
        n_heads=8,
        e_layers=2,
        d_layers=1,
        d_ff=1024,         # Reduced from 2048
        # Informer-specific
        factor=5,
        dropout=0.05,
        attn='prob',
        activation='gelu',
        # Training config
        learning_rate_init=1e-4,
        max_iter=30,
        batch_size=32,     # Can afford larger batch for short sequences
        random_state=42,
        early_stopping=True,
        n_iter_no_change=5,
        verbose=False,
        device="auto",
    ) if INFORMER_AVAILABLE else None,
    "Informer": lambda: PyTorchInformerRegressor(
        # Medium-term prediction (1 day horizon)
        seq_len=96,        # 96 hours (4 days) lookback
        label_len=48,      # 48 hours (2 days) decoder start
        pred_len=24,       # 1 day prediction
        # Model architecture (balanced configuration)
        d_model=256,       # GPU memory optimized
        n_heads=8,
        e_layers=2,
        d_layers=1,
        d_ff=1024,         # GPU memory optimized
        # Informer-specific
        factor=5,
        dropout=0.05,
        attn='prob',
        activation='gelu',
        # Training config
        learning_rate_init=1e-4,
        max_iter=30,
        batch_size=16,     # Reduced for GPU memory
        random_state=42,
        early_stopping=True,
        n_iter_no_change=5,
        verbose=False,
        device="auto",
    ) if INFORMER_AVAILABLE else None,
    "Informer-Long": lambda: PyTorchInformerRegressor(
        # Long-term prediction (1 week horizon)
        seq_len=336,       # 336 hours (2 weeks) lookback
        label_len=168,     # 168 hours (1 week) decoder start
        pred_len=168,      # 1 week prediction
        # Model architecture (optimized for long sequences)
        d_model=256,       # Keep compact for GPU memory
        n_heads=8,
        e_layers=3,        # More layers for long-term dependencies
        d_layers=1,
        d_ff=1024,         # Keep compact for GPU memory
        # Informer-specific
        factor=5,          # ProbSparse attention crucial for long sequences
        dropout=0.05,
        attn='prob',
        activation='gelu',
        # Training config
        learning_rate_init=1e-4,
        max_iter=20,       # Fewer epochs for long sequences
        batch_size=8,      # Small batch for GPU memory
        random_state=42,
        early_stopping=True,
        n_iter_no_change=5,
        verbose=False,
        device="auto",
    ) if INFORMER_AVAILABLE else None,
    "LinearRegression": lambda: LinearRegression(),
    "Ridge": lambda: Ridge(alpha=5.0, random_state=42),
}


def load_dataset(tx_id: int, data_suffix: str = "") -> pd.DataFrame:
    """
    Load cleaned dataset for a transformer.

    Parameters
    ----------
    tx_id : int
        Transformer ID (1 or 2)
    data_suffix : str
        Data file suffix (e.g., '_1pct', '_no_outlier')

    Returns
    -------
    pd.DataFrame
        Loaded dataframe
    """
    filename = f"tx{tx_id}_cleaned{data_suffix}.csv"
    filepath = CLEAN_DIR / filename

    if not filepath.exists():
        raise FileNotFoundError(
            f"Data file not found: {filepath}\n"
            f"Please run preprocessing first with the same suffix."
        )

    df = pd.read_csv(filepath, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def enrich_tx1_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add TX1-specific dynamic features."""
    enriched = df.copy()
    enriched["HULL_diff1"] = enriched["HULL"].diff().fillna(0)
    enriched["MULL_diff1"] = enriched["MULL"].diff().fillna(0)
    enriched["HULL_roll12"] = enriched["HULL"].rolling(window=12, min_periods=1).mean()
    enriched["MULL_roll12"] = enriched["MULL"].rolling(window=12, min_periods=1).mean()
    return enriched


def create_sliding_windows(
    df: pd.DataFrame,
    feature_cols: List[str],
    window_config: WindowConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create sliding windows from time series data.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    feature_cols : List[str]
        Feature column names
    window_config : WindowConfig
        Window configuration

    Returns
    -------
    X : np.ndarray
        Windowed features (shape: [n_windows, lookback * n_features])
    y : np.ndarray
        Target values (shape: [n_windows])
    timestamps : np.ndarray
        Timestamps for each window
    """
    values = df[feature_cols].to_numpy(dtype=np.float32)
    targets = df[TARGET_COL].to_numpy(dtype=np.float32)
    timestamps = df["date"].to_numpy()

    X_list: List[np.ndarray] = []
    y_list: List[float] = []
    ts_list: List[np.datetime64] = []

    lookback = window_config.lookback
    horizon = window_config.horizon
    gap = window_config.gap

    for target_idx in range(lookback + gap + horizon, len(df)):
        window_end = target_idx - horizon - gap
        window_start = window_end - lookback

        if window_start < 0:
            continue

        window = values[window_start:window_end]
        X_list.append(window.reshape(-1))
        y_list.append(targets[target_idx])
        ts_list.append(timestamps[target_idx])

    return np.array(X_list), np.array(y_list), np.array(ts_list)


def create_sliding_windows_for_rnn(
    df: pd.DataFrame,
    feature_cols: List[str],
    window_config: WindowConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create sliding windows for RNN models (keeps sequence structure).

    This function is similar to create_sliding_windows() but does NOT flatten
    the window, keeping the sequence structure required by RNN models.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    feature_cols : List[str]
        Feature column names
    window_config : WindowConfig
        Window configuration

    Returns
    -------
    X : np.ndarray
        Windowed features (shape: [n_windows, lookback, n_features])
        Note: NOT flattened, keeps sequence structure for RNN
    y : np.ndarray
        Target values (shape: [n_windows])
    timestamps : np.ndarray
        Timestamps for each window
    """
    values = df[feature_cols].to_numpy(dtype=np.float32)
    targets = df[TARGET_COL].to_numpy(dtype=np.float32)
    timestamps = df["date"].to_numpy()

    X_list: List[np.ndarray] = []
    y_list: List[float] = []
    ts_list: List[np.datetime64] = []

    lookback = window_config.lookback
    horizon = window_config.horizon
    gap = window_config.gap

    for target_idx in range(lookback + gap + horizon, len(df)):
        window_end = target_idx - horizon - gap
        window_start = window_end - lookback

        if window_start < 0:
            continue

        # Keep sequence structure (DO NOT flatten for RNN)
        window = values[window_start:window_end]  # shape: (lookback, n_features)
        X_list.append(window)
        y_list.append(targets[target_idx])
        ts_list.append(timestamps[target_idx])

    # Return 3D array for RNN: (n_windows, lookback, n_features)
    return np.array(X_list), np.array(y_list), np.array(ts_list)


def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate evaluation metrics."""
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def train_informer_native(
    tx_id: int,
    model_name: str,
    horizon: int,
    exp_id: str,
    output_dir: Path,
    log_dir: Path,
) -> Dict[str, float]:
    """
    Train Informer model using the native Informer2020 implementation.

    Parameters
    ----------
    tx_id : int
        Transformer ID (1 or 2)
    model_name : str
        Model variant name (Informer-Short, Informer, or Informer-Long)
    horizon : int
        Prediction horizon in hours
    exp_id : str
        Experiment identifier
    output_dir : Path
        Directory to save model outputs
    log_dir : Path
        Directory to save training logs

    Returns
    -------
    Dict[str, float]
        Dictionary containing metrics and training time
    """
    # Get project root
    project_root = Path(__file__).resolve().parents[1]
    informer_dir = project_root / "external" / "Informer2020"

    # Configure model parameters based on variant
    if model_name == "Informer-Short":
        # Short-term prediction (1 hour)
        seq_len = 96
        label_len = 48
        pred_len = 1
        d_model = 256
        d_ff = 1024
        e_layers = 2
        train_epochs = 10
        batch_size = 32
        model_type = "informer"
    elif model_name == "Informer":
        # Medium-term prediction (1 day)
        seq_len = 96
        label_len = 48
        pred_len = 24
        d_model = 256
        d_ff = 1024
        e_layers = 2
        train_epochs = 10
        batch_size = 16
        model_type = "informer"
    elif model_name == "Informer-Long":
        # Long-term prediction (1 week)
        seq_len = 336
        label_len = 168
        pred_len = 168
        d_model = 256
        d_ff = 1024
        e_layers = 3
        train_epochs = 8
        batch_size = 8
        model_type = "informerstack"
    else:
        raise ValueError(f"Unknown Informer variant: {model_name}")

    # Build command
    data_name = f"TX{tx_id}"
    cmd = [
        sys.executable,
        str(informer_dir / "main_informer.py"),
        "--model", model_type,
        "--data", "custom",
        "--root_path", "./data/",
        "--data_path", f"{data_name}.csv",
        "--features", "MS",  # Multivariate to Single
        "--target", "OT",
        "--freq", "h",
        "--seq_len", str(seq_len),
        "--label_len", str(label_len),
        "--pred_len", str(pred_len),
        "--enc_in", "6",  # 6 input features (HUFL, HULL, MUFL, MULL, LUFL, LULL) - OT excluded
        "--dec_in", "6",
        "--c_out", "1",  # Single output (OT)
        "--d_model", str(d_model),
        "--n_heads", "8",
        "--e_layers", str(e_layers),
        "--d_layers", "1",
        "--d_ff", str(d_ff),
        "--factor", "5",
        "--dropout", "0.05",
        "--attn", "prob",
        "--embed", "timeF",
        "--activation", "gelu",
        "--batch_size", str(batch_size),
        "--train_epochs", str(train_epochs),
        "--patience", "3",
        "--learning_rate", "0.0001",
        "--des", exp_id,
        "--itr", "1",  # Run once
        "--inverse",   # Enable inverse transformation to original scale
    ]

    print(f"\nCalling native Informer from: {informer_dir}")
    print(f"Command: {' '.join(cmd)}")

    # Prepare log file
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{exp_id}.log"

    # Run training and capture output
    start_time = time.time()

    try:
        with open(log_file, "w") as f:
            f.write(f"{'='*70}\n")
            f.write(f"Informer Native Training Log\n")
            f.write(f"Experiment: {exp_id}\n")
            f.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"{'='*70}\n\n")
            f.write(f"Command:\n{' '.join(cmd)}\n\n")
            f.write(f"{'='*70}\n\n")
            f.flush()

            # Run in Informer directory
            result = subprocess.run(
                cmd,
                cwd=str(informer_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=3600  # 1 hour timeout
            )

            # Write output to log
            f.write(result.stdout)
            f.write(f"\n{'='*70}\n")
            f.write(f"Completed: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Return code: {result.returncode}\n")
            f.write(f"{'='*70}\n")

            # Also print to console
            print(result.stdout)

            if result.returncode != 0:
                raise RuntimeError(f"Informer training failed with return code {result.returncode}")

        train_time = time.time() - start_time

        # Parse metrics from output
        output_text = result.stdout

        # Look for test metrics in output (Informer prints: "mse:X.XX, mae:Y.YY, r2:Z.ZZ")
        mse_match = re.search(r'mse:\s*([\d.]+)', output_text, re.IGNORECASE)
        mae_match = re.search(r'mae:\s*([\d.]+)', output_text, re.IGNORECASE)
        r2_match = re.search(r'r2:\s*([-\d.]+)', output_text, re.IGNORECASE)

        if mse_match and mae_match:
            mse = float(mse_match.group(1))
            mae = float(mae_match.group(1))
            rmse = np.sqrt(mse)
            r2 = float(r2_match.group(1)) if r2_match else None

            print(f"\nParsed metrics from Informer output:")
            print(f"  MSE:  {mse:.4f}")
            print(f"  MAE:  {mae:.4f}")
            print(f"  RMSE: {rmse:.4f}")
            if r2 is not None:
                print(f"  R2:   {r2:.4f}")

            return {
                "RMSE": float(rmse),
                "MAE": float(mae),
                "MSE": float(mse),
                "R2": r2,
                "train_time": train_time,
            }
        else:
            print("\nWarning: Could not parse metrics from Informer output")
            print("Please check the log file for details")
            return {
                "RMSE": None,
                "MAE": None,
                "MSE": None,
                "R2": None,
                "train_time": train_time,
            }

    except subprocess.TimeoutExpired:
        print(f"\nError: Training timeout after 1 hour")
        return {
            "RMSE": None,
            "MAE": None,
            "MSE": None,
            "R2": None,
            "train_time": 3600.0,
            "error": "timeout"
        }
    except Exception as e:
        print(f"\nError during Informer training: {e}")
        return {
            "RMSE": None,
            "MAE": None,
            "MSE": None,
            "R2": None,
            "train_time": time.time() - start_time,
            "error": str(e)
        }


def plot_predictions(
    timestamps: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    """Plot prediction results."""
    plt.figure(figsize=(12, 4))

    # Sort by timestamp for better visualization
    sort_idx = np.argsort(timestamps)
    ts_sorted = timestamps[sort_idx]
    y_true_sorted = y_true[sort_idx]
    y_pred_sorted = y_pred[sort_idx]

    # Limit to first 1000 points if too many
    if len(ts_sorted) > 1000:
        step = len(ts_sorted) // 1000
        ts_sorted = ts_sorted[::step]
        y_true_sorted = y_true_sorted[::step]
        y_pred_sorted = y_pred_sorted[::step]

    plt.plot(ts_sorted, y_true_sorted, label="Actual", linewidth=1.5, alpha=0.8)
    plt.plot(ts_sorted, y_pred_sorted, label="Predicted", linewidth=1.5, alpha=0.8)
    plt.title(title)
    plt.xlabel("Timestamp")
    plt.ylabel("Oil Temperature (°C)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_scatter(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    """Plot scatter plot of predictions vs actual."""
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.4, s=20)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()],
             "r--", linewidth=1.2)
    plt.title(title)
    plt.xlabel("Actual OT (°C)")
    plt.ylabel("Predicted OT (°C)")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Configurable model training for experiments"
    )

    # Data configuration
    parser.add_argument("--tx-id", type=int, required=True, choices=[1, 2],
                       help="Transformer ID")
    parser.add_argument("--data-suffix", type=str, default="",
                       help="Data file suffix (e.g., '_1pct', '_no_outlier')")

    # Model configuration
    available_models = [k for k, v in MODEL_BUILDERS.items() if v is not None]
    parser.add_argument("--model", type=str, required=True,
                       choices=available_models,
                       help="Model to train")

    # Data splitting configuration
    parser.add_argument("--split-method", type=str, required=True,
                       choices=["chronological", "random_window", "group_random"],
                       help="Data splitting method")
    parser.add_argument("--test-ratio", type=float, default=0.2,
                       help="Test set ratio (default: 0.2)")
    parser.add_argument("--n-groups", type=int, default=20,
                       help="Number of groups for group_random split (default: 20)")
    parser.add_argument("--random-state", type=int, default=42,
                       help="Random seed (default: 42)")

    # Feature configuration
    parser.add_argument("--feature-mode", type=str, default="full",
                       choices=["full", "time_only", "no_time", "full_6loads"],
                       help="Feature selection mode (default: full)")

    # Window configuration (for random_window and group_random)
    parser.add_argument("--lookback-multiplier", type=float, default=4.0,
                       help="Lookback window size as multiple of horizon (default: 4.0)")
    parser.add_argument("--horizon", type=int, default=1,
                       help="Prediction horizon in steps (default: 1)")
    parser.add_argument("--gap", type=int, default=0,
                       help="Gap between lookback and target (default: 0)")
    parser.add_argument("--max-windows", type=int, default=40000,
                       help="Max sliding windows to use (default: 40000)")

    # Output configuration
    parser.add_argument("--output-dir", type=str, default="",
                       help="Custom output directory (default: auto-generated)")
    parser.add_argument("--experiment-name", type=str, default="",
                       help="Experiment name for output files")

    args = parser.parse_args()

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(__file__).resolve().parents[1] / "models" / "experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate experiment identifier
    if args.experiment_name:
        exp_id = args.experiment_name
    else:
        exp_id = (f"tx{args.tx_id}_{args.model}_{args.split_method}_"
                 f"{args.feature_mode}{args.data_suffix}")

    print("="*70)
    print(f"Experiment: {exp_id}")
    print("="*70)
    print(f"Transformer:     TX{args.tx_id}")
    print(f"Model:           {args.model}")
    print(f"Split method:    {args.split_method}")
    print(f"Feature mode:    {args.feature_mode}")
    print(f"Data suffix:     {args.data_suffix if args.data_suffix else '(default)'}")

    # Load data
    print(f"\nLoading data...")
    df = load_dataset(args.tx_id, args.data_suffix)
    print(f"  Total samples: {len(df)}")

    # Add TX1-specific features if needed
    if args.tx_id == 1 and args.feature_mode != "time_only":
        df = enrich_tx1_features(df)
        print(f"  Added TX1 dynamic features")

    # Select features
    include_tx1_dynamic = (args.tx_id == 1 and args.feature_mode != "time_only")
    use_full_loads = (args.feature_mode == "full_6loads")

    # Map full_6loads to full for the function call
    feature_mode_mapped = "full" if args.feature_mode == "full_6loads" else args.feature_mode
    feature_cols = select_features_by_mode(feature_mode_mapped, include_tx1_dynamic, use_full_loads)
    print(f"  Features: {len(feature_cols)} columns")

    # Prepare data based on split method
    start_time = time.time()

    if args.split_method == "chronological":
        # Check if model requires sequence data
        is_sequence_model = args.model in ["RNN", "Informer", "Informer-Short", "Informer-Long"]

        if is_sequence_model:
            # For sequence models: create windows first, then split chronologically
            window_config = create_window_config(
                horizon=args.horizon,
                lookback_multiplier=args.lookback_multiplier,
                gap=args.gap
            )
            print(f"\nWindow config: {window_config}")
            print(f"Creating sliding windows for {args.model} (keeping sequence structure)...")
            X, y, timestamps = create_sliding_windows_for_rnn(df, feature_cols, window_config)
            print(f"  Created {len(X)} windows (shape: {X.shape})")

            # Split windows chronologically
            print(f"Splitting windows (chronological, {int(args.test_ratio*100)}% test)...")
            cutoff_idx = int(len(X) * (1 - args.test_ratio))
            X_train = X[:cutoff_idx]
            y_train = y[:cutoff_idx]
            X_test = X[cutoff_idx:]
            y_test = y[cutoff_idx:]
            ts_test = timestamps[cutoff_idx:]

            print(f"  Train: {len(X_train)} windows")
            print(f"  Test:  {len(X_test)} windows")
        else:
            # For non-sequence models: create windows first, then split chronologically
            window_config = create_window_config(
                horizon=args.horizon,
                lookback_multiplier=args.lookback_multiplier,
                gap=args.gap
            )
            print(f"\nWindow config: {window_config}")
            print(f"Creating sliding windows...")
            X, y, timestamps = create_sliding_windows(df, feature_cols, window_config)
            print(f"  Created {len(X)} windows")

            # Split windows chronologically
            print(f"Splitting windows (chronological, {int(args.test_ratio*100)}% test)...")
            cutoff_idx = int(len(X) * (1 - args.test_ratio))
            X_train = X[:cutoff_idx]
            y_train = y[:cutoff_idx]
            X_test = X[cutoff_idx:]
            y_test = y[cutoff_idx:]
            ts_test = timestamps[cutoff_idx:]

            print(f"  Train: {len(X_train)} windows")
            print(f"  Test:  {len(X_test)} windows")

    elif args.split_method in ["random_window", "group_random"]:
        # Create sliding windows first
        window_config = create_window_config(
            horizon=args.horizon,
            lookback_multiplier=args.lookback_multiplier,
            gap=args.gap
        )
        print(f"\nWindow config: {window_config}")

        # Check if model is RNN or Informer (requires 3D sequential data)
        is_sequence_model = args.model in ["RNN", "Informer", "Informer-Short", "Informer-Long"]

        if is_sequence_model:
            print(f"Creating sliding windows for {args.model} (keeping sequence structure)...")
            X, y, timestamps = create_sliding_windows_for_rnn(df, feature_cols, window_config)
            print(f"  Created {len(X)} windows (shape: {X.shape})")
        else:
            print(f"Creating sliding windows...")
            X, y, timestamps = create_sliding_windows(df, feature_cols, window_config)
            print(f"  Created {len(X)} windows")

        # Limit windows if specified
        if args.max_windows and len(X) > args.max_windows:
            print(f"  Sampling {args.max_windows} windows...")
            rng = np.random.default_rng(args.random_state)
            indices = rng.choice(len(X), size=args.max_windows, replace=False)
            X = X[indices]
            y = y[indices]
            timestamps = timestamps[indices]

        # Split windows
        if args.split_method == "random_window":
            print(f"Splitting windows (random, {int(args.test_ratio*100)}% test)...")
            X_train, X_test, y_train, y_test, _, ts_test = train_test_split(
                X, y, timestamps,
                test_size=args.test_ratio,
                random_state=args.random_state,
                shuffle=True
            )
        else:  # group_random
            print(f"Splitting windows (group random, {args.n_groups} groups, {int(args.test_ratio*100)}% test)...")
            # Create a temporary dataframe for group split
            temp_df = pd.DataFrame({
                'date': timestamps,
                TARGET_COL: y
            })
            # Add X as columns
            for i in range(X.shape[1]):
                temp_df[f'X_{i}'] = X[:, i]

            train_df, test_df = group_random_split(
                temp_df,
                n_groups=args.n_groups,
                test_ratio=args.test_ratio,
                random_state=args.random_state
            )

            X_cols = [f'X_{i}' for i in range(X.shape[1])]
            X_train = train_df[X_cols].to_numpy(dtype=np.float32)
            X_test = test_df[X_cols].to_numpy(dtype=np.float32)
            y_train = train_df[TARGET_COL].to_numpy(dtype=np.float32)
            y_test = test_df[TARGET_COL].to_numpy(dtype=np.float32)
            ts_test = test_df["date"].to_numpy()

        print(f"  Train: {len(X_train)} windows")
        print(f"  Test:  {len(X_test)} windows")

    # Train model
    print(f"\nTraining {args.model}...")

    # Check if this is a native Informer model
    if args.model in ["Informer-Short", "Informer", "Informer-Long"]:
        # Use native Informer2020 implementation
        log_dir = Path(__file__).resolve().parents[1] / "experiment" / "logs"

        result = train_informer_native(
            tx_id=args.tx_id,
            model_name=args.model,
            horizon=args.horizon,
            exp_id=exp_id,
            output_dir=output_dir,
            log_dir=log_dir,
        )

        # Extract metrics from result
        train_time = result.get("train_time", 0.0)
        metrics = {
            "RMSE": result.get("RMSE"),
            "MAE": result.get("MAE"),
            "R2": result.get("R2"),
        }

        # Set dummy values for test data (Informer handles its own train/test split)
        y_pred = None
        ts_test = None

        print(f"  Training time: {train_time:.2f}s")

        if metrics["RMSE"] is not None:
            print(f"  RMSE: {metrics['RMSE']:.4f}")
            print(f"  MAE:  {metrics['MAE']:.4f}")
            if metrics['R2'] is not None:
                print(f"  R2:   {metrics['R2']:.4f}")
        else:
            print("  Warning: Metrics could not be parsed from Informer output")
    else:
        # Use standard sklearn/pytorch models
        model_builder = MODEL_BUILDERS[args.model]
        if model_builder is None:
            raise ValueError(f"Model '{args.model}' is not available. Informer may not be properly installed.")
        model = model_builder()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        print(f"  Training time: {train_time:.2f}s")

        # Evaluate
        print(f"\nEvaluating...")
        y_pred = model.predict(X_test)
        metrics = evaluate_model(y_test, y_pred)

        print(f"  RMSE: {metrics['RMSE']:.4f}")
        print(f"  MAE:  {metrics['MAE']:.4f}")
        print(f"  R2:   {metrics['R2']:.4f}")

    # Save results
    print(f"\nSaving results...")

    # Save model (skip for native Informer)
    if args.model not in ["Informer-Short", "Informer", "Informer-Long"]:
        model_path = output_dir / f"{exp_id}_model.joblib"
        joblib.dump(model, model_path)
        print(f"  Model: {model_path}")

        # Save predictions
        pred_df = pd.DataFrame({
            "timestamp": ts_test,
            "actual": y_test,
            "predicted": y_pred
        }).sort_values("timestamp")
        pred_path = TABLE_DIR / f"{exp_id}_predictions.csv"
        pred_df.to_csv(pred_path, index=False)
        print(f"  Predictions: {pred_path}")
    else:
        print(f"  Model: Native Informer (saved in external/Informer2020/checkpoints/)")
        print(f"  Predictions: Not saved (native Informer handles this internally)")

    # Save metrics
    metrics_data = {
        "experiment_id": exp_id,
        "transformer_id": args.tx_id,
        "model": args.model,
        "split_method": args.split_method,
        "feature_mode": args.feature_mode,
        "data_suffix": args.data_suffix,
        "test_ratio": args.test_ratio,
        "n_features": len(feature_cols),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "train_time": train_time,
        **metrics
    }

    if args.split_method in ["random_window", "group_random"]:
        metrics_data.update({
            "lookback": window_config.lookback,
            "horizon": window_config.horizon,
            "gap": window_config.gap,
            "lookback_multiplier": args.lookback_multiplier,
        })

    metrics_path = output_dir / f"{exp_id}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=2)
    print(f"  Metrics: {metrics_path}")

    # Plot predictions (skip for native Informer)
    if args.model not in ["Informer-Short", "Informer", "Informer-Long"]:
        plot_path = FIG_DIR / f"{exp_id}_predictions.png"
        plot_predictions(ts_test, y_test, y_pred, plot_path,
                        f"TX{args.tx_id} - {args.model} - {args.split_method}")
        print(f"  Plot: {plot_path}")

        # Plot scatter
        scatter_path = FIG_DIR / f"{exp_id}_scatter.png"
        plot_scatter(y_test, y_pred, scatter_path,
                    f"TX{args.tx_id} - {args.model}")
        print(f"  Scatter: {scatter_path}")
    else:
        print(f"  Plot: Skipped (native Informer handles visualization internally)")

    print("\n" + "="*70)
    print(f"Experiment complete: {exp_id}")
    print("="*70)


if __name__ == "__main__":
    main()
