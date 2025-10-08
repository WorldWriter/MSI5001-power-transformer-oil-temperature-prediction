"""Preprocessing pipeline with time-aware splits and configurable windows."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.config import (  # noqa: E402  pylint: disable=wrong-import-position
    DEFAULT_DATA_DIR,
    DEFAULT_OUTPUT_DIR,
    EXPERIMENT_CONFIGS,
    ExperimentConfig,
)

LOGGER = logging.getLogger(__name__)


def load_raw_data(data_dir: Path, file_names: Iterable[str] | None = None) -> pd.DataFrame:
    """Load and concatenate raw CSV files sorted by timestamp."""
    if file_names is None:
        file_names = sorted(str(path.name) for path in data_dir.glob("trans_*.csv"))
    frames: List[pd.DataFrame] = []
    for name in file_names:
        csv_path = data_dir / name
        if not csv_path.exists():
            LOGGER.warning("Skip missing file: %s", csv_path)
            continue
        LOGGER.info("Loading %s", csv_path)
        frame = pd.read_csv(csv_path)
        if "date" not in frame.columns:
            raise ValueError(f"Expected 'date' column in {csv_path}")
        frame["date"] = pd.to_datetime(frame["date"])
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(
            f"No CSV files found in {data_dir}. Provide files via --data-dir or --files."
        )
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("date").reset_index(drop=True)
    LOGGER.info(
        "Loaded %d rows covering %s to %s",
        len(combined),
        combined["date"].min(),
        combined["date"].max(),
    )
    return combined


def create_sequences(
    data: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str,
    config: ExperimentConfig,
) -> Tuple[np.ndarray, np.ndarray]:
    """Construct sliding-window sequences respecting the experiment configuration."""
    max_index = len(data) - config.lookback - config.forecast_horizon
    if max_index <= 0:
        raise ValueError("Not enough rows to create sequences with the given configuration.")

    step = max(config.step, 1)
    candidate_indices = list(range(0, max_index + 1, step))
    if config.max_sequences is not None and len(candidate_indices) > config.max_sequences:
        if config.sample_strategy == "random":
            rng = np.random.default_rng(42)
            selected = np.sort(rng.choice(candidate_indices, config.max_sequences, replace=False))
        elif config.sample_strategy == "stride":
            stride = len(candidate_indices) / config.max_sequences
            selected = [candidate_indices[int(i * stride)] for i in range(config.max_sequences)]
        else:
            selected = candidate_indices[: config.max_sequences]
        candidate_indices = list(selected)

    feature_array = data.loc[:, feature_cols].to_numpy(dtype=float)
    target_array = data.loc[:, target_col].to_numpy(dtype=float)

    sequences = np.stack(
        [feature_array[start : start + config.lookback] for start in candidate_indices]
    )
    targets = np.array(
        [target_array[start + config.lookback + config.forecast_horizon - 1] for start in candidate_indices]
    )
    LOGGER.info("Created %d sequences with shape %s", sequences.shape[0], sequences.shape[1:])
    return sequences, targets


def split_sequences(
    X: np.ndarray,
    y: np.ndarray,
    config: ExperimentConfig,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split sequences into chronological train/validation/test sets."""
    total_samples = len(X)
    train_count = int(total_samples * config.train_ratio)
    val_count = int(total_samples * config.val_ratio)
    test_count = total_samples - train_count - val_count
    if min(train_count, val_count, test_count) <= 0:
        raise ValueError(
            "Split produced an empty subset. Adjust ratios or increase available sequences."
        )
    train_end = train_count
    val_end = train_end + val_count
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    LOGGER.info(
        "Split into %d train / %d val / %d test samples",
        len(X_train),
        len(X_val),
        len(X_test),
    )
    if len(X_train) == 0 or len(X_val) == 0 or len(X_test) == 0:
        raise ValueError("Split produced empty dataset; adjust ratios or window parameters.")
    return X_train, X_val, X_test, y_train, y_val, y_test


def scale_datasets(
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """Fit a StandardScaler on training data and transform all splits."""
    scaler = StandardScaler()
    n_features = X_train.shape[-1]
    scaler.fit(X_train.reshape(-1, n_features))
    X_train_scaled = scaler.transform(X_train.reshape(-1, n_features)).reshape(X_train.shape)
    X_val_scaled = scaler.transform(X_val.reshape(-1, n_features)).reshape(X_val.shape)
    X_test_scaled = scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler


def save_numpy_arrays(
    output_dir: Path,
    config: ExperimentConfig,
    X_train: np.ndarray,
    X_val: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    scaler: StandardScaler,
    metadata: dict,
) -> None:
    """Persist datasets, scaler, and metadata."""
    output_dir.mkdir(parents=True, exist_ok=True)
    np.save(output_dir / f"X_train_{config.name}.npy", X_train)
    np.save(output_dir / f"X_val_{config.name}.npy", X_val)
    np.save(output_dir / f"X_test_{config.name}.npy", X_test)
    np.save(output_dir / f"y_train_{config.name}.npy", y_train)
    np.save(output_dir / f"y_val_{config.name}.npy", y_val)
    np.save(output_dir / f"y_test_{config.name}.npy", y_test)
    joblib.dump(scaler, output_dir / f"scaler_{config.name}.pkl")
    with (output_dir / f"metadata_{config.name}.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    LOGGER.info("Artifacts written to %s", output_dir)


def preprocess_configuration(
    config: ExperimentConfig,
    data: pd.DataFrame,
    output_dir: Path,
    feature_columns: Sequence[str],
    target_column: str,
) -> None:
    """Run preprocessing for a single experiment configuration."""
    LOGGER.info("Preparing configuration %s", config.name)
    sequences, targets = create_sequences(data, feature_columns, target_column, config)
    X_train, X_val, X_test, y_train, y_val, y_test = split_sequences(sequences, targets, config)
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_datasets(X_train, X_val, X_test)
    metadata = {
        "config": asdict(config),
        "n_train": len(X_train_scaled),
        "n_val": len(X_val_scaled),
        "n_test": len(X_test_scaled),
        "features": list(feature_columns),
        "target": target_column,
    }
    save_numpy_arrays(
        output_dir,
        config,
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        y_train,
        y_val,
        y_test,
        scaler,
        metadata,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs",
        nargs="*",
        default=list(EXPERIMENT_CONFIGS.keys()),
        help="Experiment configurations to generate (default: all).",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing the raw CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for the generated numpy arrays and metadata.",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        help="Optional explicit list of CSV file names to load from the data directory.",
    )
    parser.add_argument(
        "--feature-cols",
        nargs="*",
        default=["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"],
        help="Feature columns to use from the raw CSV files.",
    )
    parser.add_argument(
        "--target-col",
        default="OT",
        help="Target column representing the oil temperature.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ...).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s - %(levelname)s - %(message)s")
    LOGGER.info("Starting preprocessing with configs: %s", args.configs)
    raw_data = load_raw_data(args.data_dir, args.files)
    for name in args.configs:
        if name not in EXPERIMENT_CONFIGS:
            raise KeyError(f"Unknown configuration '{name}'. Available: {sorted(EXPERIMENT_CONFIGS)}")
        preprocess_configuration(
            EXPERIMENT_CONFIGS[name],
            raw_data,
            args.output_dir,
            args.feature_cols,
            args.target_col,
        )


if __name__ == "__main__":
    main()
