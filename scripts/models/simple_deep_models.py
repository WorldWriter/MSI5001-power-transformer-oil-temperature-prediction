"""Train neural models on preprocessed sequences with optional hyperparameter search."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neural_network import MLPRegressor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts.config import (  # noqa: E402  pylint: disable=wrong-import-position
    DEFAULT_OUTPUT_DIR,
    EXPERIMENT_CONFIGS,
    ExperimentConfig,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class ModelResult:
    model: str
    config: str
    split: str
    mse: float
    rmse: float
    mae: float
    r2: float


def load_dataset(output_dir: Path, config: ExperimentConfig) -> Tuple[np.ndarray, ...]:
    """Load scaled sequences and targets for a configuration."""
    X_train = np.load(output_dir / f"X_train_{config.name}.npy")
    X_val = np.load(output_dir / f"X_val_{config.name}.npy")
    X_test = np.load(output_dir / f"X_test_{config.name}.npy")
    y_train = np.load(output_dir / f"y_train_{config.name}.npy")
    y_val = np.load(output_dir / f"y_val_{config.name}.npy")
    y_test = np.load(output_dir / f"y_test_{config.name}.npy")
    return X_train, X_val, X_test, y_train, y_val, y_test


def flatten_sequences(*arrays: np.ndarray) -> List[np.ndarray]:
    """Flatten sequence arrays for feed-forward models."""
    flattened: List[np.ndarray] = []
    for array in arrays:
        if array.ndim == 3:
            samples = array.shape[0]
            flattened.append(array.reshape(samples, -1))
        else:
            flattened.append(array)
    return flattened


def evaluate(
    records: List[ModelResult],
    model_name: str,
    config_name: str,
    split: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    records.append(ModelResult(model_name, config_name, split, mse, rmse, mae, r2))


def build_cv(n_samples: int) -> TimeSeriesSplit:
    splits = 3
    if n_samples <= splits:
        splits = max(2, n_samples - 1)
    if splits < 2:
        raise ValueError("Not enough samples for cross-validation; provide more data or adjust windows.")
    return TimeSeriesSplit(n_splits=splits)


def run_search(
    base_model: MLPRegressor,
    param_grid: Dict[str, Iterable],
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> GridSearchCV:
    cv = build_cv(len(X_train))
    search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    LOGGER.info("Best params for %s: %s", base_model.__class__.__name__, search.best_params_)
    return search


def train_deep_models(
    config: ExperimentConfig,
    output_dir: Path,
    model_dir: Path,
    enable_search: bool,
) -> Tuple[List[ModelResult], Dict[str, Dict[str, float]]]:
    LOGGER.info("Training neural models for %s", config.name)
    model_dir.mkdir(parents=True, exist_ok=True)
    X_train, X_val, X_test, y_train, y_val, y_test = load_dataset(output_dir, config)
    X_train_f, X_val_f, X_test_f = flatten_sequences(X_train, X_val, X_test)
    records: List[ModelResult] = []
    param_summary: Dict[str, Dict[str, float]] = {}

    default_model = MLPRegressor(
        hidden_layer_sizes=(128, 64, 32),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=400,
        random_state=42,
        early_stopping=True,
    )

    if enable_search:
        search = run_search(
            default_model,
            {
                "hidden_layer_sizes": [(128, 64, 32), (256, 128, 64), (64, 32)],
                "learning_rate_init": [0.001, 0.0005],
                "alpha": [0.0001, 0.001, 0.01],
            },
            X_train_f,
            y_train,
        )
        mlp = search.best_estimator_
        param_summary["MLPRegressor"] = search.best_params_
    else:
        mlp = default_model
        mlp.fit(X_train_f, y_train)
        param_summary["MLPRegressor"] = {
            "hidden_layer_sizes": mlp.hidden_layer_sizes,
            "learning_rate_init": mlp.learning_rate_init,
            "alpha": mlp.alpha,
        }

    evaluate(records, "MLPRegressor", config.name, "val", y_val, mlp.predict(X_val_f))
    evaluate(records, "MLPRegressor", config.name, "test", y_test, mlp.predict(X_test_f))
    joblib.dump(mlp, model_dir / f"mlp_{config.name}.pkl")
    return records, param_summary


def save_results(
    metrics: List[ModelResult],
    params: Dict[str, Dict[str, float]],
    metrics_path: Path,
    params_path: Path,
) -> None:
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    params_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame([m.__dict__ for m in metrics])
    df.to_csv(metrics_path, index=False)
    with params_path.open("w", encoding="utf-8") as handle:
        json.dump(_to_serializable(params), handle, indent=2)


def maybe_merge_results(
    deep_metrics_path: Path,
    ml_metrics_path: Optional[Path],
    combined_path: Optional[Path],
) -> None:
    if ml_metrics_path is None or combined_path is None:
        return
    if not ml_metrics_path.exists():
        LOGGER.warning("ML metrics file %s not found; skipping merge.", ml_metrics_path)
        return
    ml_df = pd.read_csv(ml_metrics_path)
    dl_df = pd.read_csv(deep_metrics_path)
    combined = pd.concat([ml_df, dl_df], ignore_index=True)
    combined_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(combined_path, index=False)
    LOGGER.info("Merged metrics written to %s", combined_path)


def _to_serializable(value):
    if isinstance(value, dict):
        return {key: _to_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item()
        except Exception:  # pragma: no cover
            return str(value)
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--configs",
        nargs="*",
        default=list(EXPERIMENT_CONFIGS.keys()),
        help="Configurations to train (default: all).",
    )
    parser.add_argument(
        "--preprocessed-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory containing scaled numpy arrays.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("artifacts/models"),
        help="Directory to store trained neural models.",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("artifacts/simple_deep_results.csv"),
        help="Output CSV for neural model metrics.",
    )
    parser.add_argument(
        "--params-path",
        type=Path,
        default=Path("artifacts/simple_deep_best_params.json"),
        help="Output JSON for best neural hyperparameters.",
    )
    parser.add_argument(
        "--ml-metrics",
        type=Path,
        help="Optional path to previously generated ML metrics for merging.",
    )
    parser.add_argument(
        "--combined-output",
        type=Path,
        help="Optional path to write merged ML + neural metrics.",
    )
    parser.add_argument(
        "--disable-search",
        action="store_true",
        help="Skip hyperparameter search and train with default configuration.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s - %(levelname)s - %(message)s")
    enable_search = not args.disable_search
    all_metrics: List[ModelResult] = []
    config_params: Dict[str, Dict[str, float]] = {}

    for name in args.configs:
        if name not in EXPERIMENT_CONFIGS:
            raise KeyError(f"Unknown configuration '{name}'. Available: {sorted(EXPERIMENT_CONFIGS)}")
        config = EXPERIMENT_CONFIGS[name]
        metrics, params = train_deep_models(config, args.preprocessed_dir, args.model_dir, enable_search)
        all_metrics.extend(metrics)
        config_params[name] = params

    save_results(all_metrics, config_params, args.metrics_path, args.params_path)
    maybe_merge_results(args.metrics_path, args.ml_metrics, args.combined_output)
    LOGGER.info("Completed neural training for %d configurations", len(args.configs))


if __name__ == "__main__":
    main()
