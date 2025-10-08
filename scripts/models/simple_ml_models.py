"""Train classical ML models with time-series aware validation and tuning."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

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
    """Load preprocessed arrays for the requested configuration."""
    prefix = output_dir
    X_train = np.load(prefix / f"X_train_{config.name}.npy")
    X_val = np.load(prefix / f"X_val_{config.name}.npy")
    X_test = np.load(prefix / f"X_test_{config.name}.npy")
    y_train = np.load(prefix / f"y_train_{config.name}.npy")
    y_val = np.load(prefix / f"y_val_{config.name}.npy")
    y_test = np.load(prefix / f"y_test_{config.name}.npy")
    return X_train, X_val, X_test, y_train, y_val, y_test


def flatten_sequences(*arrays: np.ndarray) -> List[np.ndarray]:
    """Flatten 3D arrays (samples, time, features) into 2D for sklearn models."""
    flattened: List[np.ndarray] = []
    for array in arrays:
        if array.ndim == 3:
            samples, _, features = array.shape
            flattened.append(array.reshape(samples, -1))
        else:
            flattened.append(array)
    return flattened


def evaluate_predictions(
    records: List[ModelResult],
    model_name: str,
    config_name: str,
    split: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> None:
    """Append evaluation metrics for a particular split."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    records.append(ModelResult(model_name, config_name, split, mse, rmse, mae, r2))


def build_time_series_cv(n_samples: int) -> TimeSeriesSplit:
    """Create a TimeSeriesSplit with a safe number of splits for the data size."""
    splits = 3
    if n_samples <= splits:
        splits = max(2, n_samples - 1)
    if splits < 2:
        raise ValueError("Not enough samples for cross-validation; gather more data or adjust windows.")
    return TimeSeriesSplit(n_splits=splits)


def grid_search(
    estimator,
    param_grid: Dict[str, Iterable],
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> GridSearchCV:
    """Run grid search with time-series cross validation."""
    cv = build_time_series_cv(len(X_train))
    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=cv,
        n_jobs=-1,
        refit=True,
    )
    search.fit(X_train, y_train)
    LOGGER.info("Best params for %s: %s", estimator.__class__.__name__, search.best_params_)
    return search


def train_models(
    config: ExperimentConfig,
    output_dir: Path,
    enable_search: bool,
    results_dir: Path,
) -> Tuple[List[ModelResult], Dict[str, Dict[str, float]]]:
    """Train baseline models and optional hyperparameter searches."""
    LOGGER.info("Training classical models for %s", config.name)
    results_dir.mkdir(parents=True, exist_ok=True)
    X_train, X_val, X_test, y_train, y_val, y_test = load_dataset(output_dir, config)
    X_train_f, X_val_f, X_test_f = flatten_sequences(X_train, X_val, X_test)
    records: List[ModelResult] = []
    best_params: Dict[str, Dict[str, float]] = {}

    # Baseline linear regression
    linear = LinearRegression()
    linear.fit(X_train_f, y_train)
    evaluate_predictions(records, "LinearRegression", config.name, "val", y_val, linear.predict(X_val_f))
    evaluate_predictions(records, "LinearRegression", config.name, "test", y_test, linear.predict(X_test_f))
    joblib.dump(linear, results_dir / f"linear_{config.name}.pkl")

    if enable_search:
        ridge_search = grid_search(
            Ridge(),
            {"alpha": [0.1, 1.0, 10.0, 100.0]},
            X_train_f,
            y_train,
        )
        best_params["Ridge"] = ridge_search.best_params_
        ridge = ridge_search.best_estimator_
    else:
        ridge = Ridge(alpha=1.0)
        ridge.fit(X_train_f, y_train)
        best_params["Ridge"] = {"alpha": ridge.alpha}

    evaluate_predictions(records, "Ridge", config.name, "val", y_val, ridge.predict(X_val_f))
    evaluate_predictions(records, "Ridge", config.name, "test", y_test, ridge.predict(X_test_f))
    joblib.dump(ridge, results_dir / f"ridge_{config.name}.pkl")

    if enable_search:
        rf_search = grid_search(
            RandomForestRegressor(random_state=42),
            {
                "n_estimators": [200, 400],
                "max_depth": [None, 12, 20],
                "min_samples_leaf": [1, 2, 4],
            },
            X_train_f,
            y_train,
        )
        best_params["RandomForestRegressor"] = rf_search.best_params_
        rf = rf_search.best_estimator_
    else:
        rf = RandomForestRegressor(n_estimators=200, max_depth=None, random_state=42)
        rf.fit(X_train_f, y_train)
        best_params["RandomForestRegressor"] = {
            "n_estimators": rf.n_estimators,
            "max_depth": rf.max_depth,
            "min_samples_leaf": rf.min_samples_leaf,
        }

    evaluate_predictions(records, "RandomForestRegressor", config.name, "val", y_val, rf.predict(X_val_f))
    evaluate_predictions(records, "RandomForestRegressor", config.name, "test", y_test, rf.predict(X_test_f))
    joblib.dump(rf, results_dir / f"random_forest_{config.name}.pkl")

    return records, best_params


def save_results(
    metrics: List[ModelResult],
    best_params: Dict[str, Dict[str, float]],
    output_path: Path,
    params_path: Path,
) -> None:
    """Persist evaluation metrics and parameter summaries."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    params_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_df = pd.DataFrame([r.__dict__ for r in metrics])
    metrics_df.to_csv(output_path, index=False)
    with params_path.open("w", encoding="utf-8") as handle:
        json.dump(_to_serializable(best_params), handle, indent=2)


def _to_serializable(value):
    if isinstance(value, dict):
        return {key: _to_serializable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(v) for v in value]
    if hasattr(value, "item") and callable(getattr(value, "item")):
        try:
            return value.item()
        except Exception:  # pragma: no cover - fallback for unexpected types
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
        help="Directory containing the preprocessed numpy arrays.",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("artifacts/models"),
        help="Directory to store trained models.",
    )
    parser.add_argument(
        "--metrics-path",
        type=Path,
        default=Path("artifacts/simple_ml_results.csv"),
        help="File path to write evaluation metrics.",
    )
    parser.add_argument(
        "--params-path",
        type=Path,
        default=Path("artifacts/simple_ml_best_params.json"),
        help="File path to write best hyperparameters.",
    )
    parser.add_argument(
        "--disable-search",
        action="store_true",
        help="Skip hyperparameter grid searches and use default settings.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging verbosity level.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=args.log_level.upper(), format="%(asctime)s - %(levelname)s - %(message)s")
    enable_search = not args.disable_search
    all_metrics: List[ModelResult] = []
    aggregated_params: Dict[str, Dict[str, float]] = {}

    for name in args.configs:
        if name not in EXPERIMENT_CONFIGS:
            raise KeyError(f"Unknown configuration '{name}'. Available: {sorted(EXPERIMENT_CONFIGS)}")
        config = EXPERIMENT_CONFIGS[name]
        config_metrics, config_params = train_models(
            config,
            args.preprocessed_dir,
            enable_search,
            args.results_dir,
        )
        all_metrics.extend(config_metrics)
        aggregated_params[name] = config_params

    save_results(all_metrics, aggregated_params, args.metrics_path, args.params_path)
    LOGGER.info("Completed training for %d configurations", len(args.configs))


if __name__ == "__main__":
    main()
