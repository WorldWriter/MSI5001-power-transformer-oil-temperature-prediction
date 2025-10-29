from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from models.pytorch_mlp import PyTorchMLPRegressor

from .common import FIG_DIR, TABLE_DIR, TARGET_COL

CLEAN_DIR = Path(__file__).resolve().parents[1] / "processed"
MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "baseline"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_SET = [
    "HULL",
    "MULL",
    "hour",
    "dayofweek",
    "month",
    "day_of_year",
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "month_sin",
    "month_cos",
    "doy_sin",
    "doy_cos",
    "is_weekend",
    "is_worktime",
    "season",
]


def load_dataset(tx_id: int, standardized: bool = False) -> pd.DataFrame:
    fname = f"tx{tx_id}_{'standardized' if standardized else 'cleaned'}.csv"
    df = pd.read_csv(CLEAN_DIR / fname, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def chronological_split(
    df: pd.DataFrame, train_ratio: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = int(len(df) * train_ratio)
    train = df.iloc[:cutoff].reset_index(drop=True)
    test = df.iloc[cutoff:].reset_index(drop=True)
    return train, test


def train_models(
    X_train: np.ndarray, y_train: np.ndarray
) -> Dict[str, object]:
    models = {
        "RandomForest": RandomForestRegressor(
            n_estimators=120, max_depth=12, random_state=42, n_jobs=-1, min_samples_leaf=5
        ),
        "MLP": PyTorchMLPRegressor(
            hidden_layer_sizes=(128, 64),
            activation="relu",
            alpha=1e-4,
            learning_rate_init=1e-3,
            max_iter=120,
            batch_size=64,
            random_state=42,
            early_stopping=True,
            verbose=True,
            device="auto",
        ),
    }
    fitted = {}
    for name, model in models.items():
        print(f"Training {name}")
        model.fit(X_train, y_train)
        fitted[name] = model
    return fitted


def evaluate_model(
    model: object, X_test: np.ndarray, y_test: np.ndarray
) -> Dict[str, float]:
    preds = model.predict(X_test)
    metrics = {
        "RMSE": float(np.sqrt(mean_squared_error(y_test, preds))),
        "MAE": float(mean_absolute_error(y_test, preds)),
        "R2": float(r2_score(y_test, preds)),
    }
    return metrics, preds


def plot_predictions(
    timestamps: pd.Series,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    tx_id: int,
    model_name: str,
) -> None:
    plt.figure(figsize=(12, 4))
    plt.plot(timestamps, y_true, label="Actual", linewidth=1.5)
    plt.plot(timestamps, y_pred, label="Predicted", linewidth=1.5, alpha=0.8)
    plt.title(f"Transformer {tx_id} - {model_name} Predictions (Test set)")
    plt.xlabel("Timestamp")
    plt.ylabel("Oil Temperature (°C)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"tx{tx_id}_{model_name}_prediction.png", dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--standardized",
        action="store_true",
        help="Use standardized datasets instead of cleaned raw values.",
    )
    args = parser.parse_args()

    metrics_records = []

    for tx_id in [1, 2]:
        df = load_dataset(tx_id, standardized=args.standardized)
        train_df, test_df = chronological_split(df, train_ratio=0.8)

        X_train = train_df[FEATURE_SET].to_numpy(dtype=np.float32)
        y_train = train_df[TARGET_COL].to_numpy(dtype=np.float32)
        X_test = test_df[FEATURE_SET].to_numpy(dtype=np.float32)
        y_test = test_df[TARGET_COL].to_numpy(dtype=np.float32)

        fitted_models = train_models(X_train, y_train)

        for name, model in fitted_models.items():
            metrics, preds = evaluate_model(model, X_test, y_test)
            metrics_records.append(
                {
                    "Transformer": tx_id,
                    "Model": name,
                    "Standardized": args.standardized,
                    **metrics,
                }
            )

            model_path = MODEL_DIR / f"tx{tx_id}_{name}{'_std' if args.standardized else ''}.joblib"
            joblib.dump(model, model_path)

            result_df = pd.DataFrame(
                {
                    "timestamp": test_df["date"],
                    "actual": y_test,
                    "predicted": preds,
                }
            )
            result_csv = TABLE_DIR / f"tx{tx_id}_{name}_predictions{'_std' if args.standardized else ''}.csv"
            result_df.to_csv(result_csv, index=False)

            plot_predictions(test_df["date"], y_test, preds, tx_id, f"{name}{'_std' if args.standardized else ''}")

    metrics_df = pd.DataFrame(metrics_records)
    metrics_df.to_csv(
        TABLE_DIR / f"model_performance{'_std' if args.standardized else ''}.csv",
        index=False,
    )

    print("Model training and evaluation complete.")


if __name__ == "__main__":
    main()
