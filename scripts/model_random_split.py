from __future__ import annotations

import argparse
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

from .common import (
    CLEAN_DIR,
    FIG_DIR,
    TABLE_DIR,
    TARGET_COL,
    add_time_features,
)


MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "random_split"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


BASE_FEATURES: List[str] = [
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

TX1_DYNAMIC_FEATURES: List[str] = [
    "HULL_diff1",
    "MULL_diff1",
    "HULL_roll12",
    "MULL_roll12",
]

DEFAULT_LOOKBACK = 24
DEFAULT_HORIZON = 1

LOOKBACK_OVERRIDES = {
    1: {"lookback": 48, "horizon": 1},
    2: {"lookback": 24, "horizon": 1},
}

MODEL_BUILDERS = {
    "LinearRegression": lambda: LinearRegression(),
    "Ridge": lambda: Ridge(alpha=5.0, random_state=42),
    "RandomForest": lambda: RandomForestRegressor(
        n_estimators=120, max_depth=12, min_samples_leaf=5, random_state=42, n_jobs=-1
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
        verbose=True,
        device="auto",
    ),
}


def load_cleaned_tx(tx_id: int) -> pd.DataFrame:
    df = pd.read_csv(CLEAN_DIR / f"tx{tx_id}_cleaned.csv", parse_dates=["date"])
    df = add_time_features(df)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def enrich_tx1(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["HULL_diff1"] = enriched["HULL"].diff().fillna(0)
    enriched["MULL_diff1"] = enriched["MULL"].diff().fillna(0)
    enriched["HULL_roll12"] = enriched["HULL"].rolling(window=12, min_periods=1).mean()
    enriched["MULL_roll12"] = enriched["MULL"].rolling(window=12, min_periods=1).mean()
    return enriched


def create_windows(
    df: pd.DataFrame,
    feature_cols: List[str],
    lookback: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = df[feature_cols].to_numpy(dtype=np.float32)
    targets = df[TARGET_COL].to_numpy(dtype=np.float32)
    timestamps = df["date"].to_numpy()
    X_windows: List[np.ndarray] = []
    y_windows: List[float] = []
    ts_list: List[np.datetime64] = []
    for end_idx in range(lookback, len(df) - horizon):
        start_idx = end_idx - lookback
        target_idx = end_idx + horizon
        X_windows.append(values[start_idx:end_idx].reshape(-1))
        y_windows.append(targets[target_idx])
        ts_list.append(timestamps[target_idx])
    return np.array(X_windows), np.array(y_windows), np.array(ts_list)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def plot_scatter(y_true: np.ndarray, y_pred: np.ndarray, tx_id: int, model_name: str) -> None:
    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.4, s=20)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", linewidth=1.2)
    plt.title(f"TX{tx_id} - {model_name} (Random split)")
    plt.xlabel("Actual OT")
    plt.ylabel("Predicted OT")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"random_tx{tx_id}_{model_name}_scatter.png", dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Random split sliding-window modelling.")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test ratio (default 0.2).")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for shuffle split.")
    parser.add_argument(
        "--max-windows",
        type=int,
        default=40000,
        help="Maximum number of window samples per transformer (default 40k).",
    )
    args = parser.parse_args()

    metrics_records = []

    for tx_id in [1, 2]:
        print(f"\n=== Transformer {tx_id} ===")
        df = load_cleaned_tx(tx_id)
        feature_cols = BASE_FEATURES.copy()
        lookback = LOOKBACK_OVERRIDES.get(tx_id, {"lookback": DEFAULT_LOOKBACK}).get("lookback", DEFAULT_LOOKBACK)
        horizon = LOOKBACK_OVERRIDES.get(tx_id, {"horizon": DEFAULT_HORIZON}).get("horizon", DEFAULT_HORIZON)

        if tx_id == 1:
            df = enrich_tx1(df)
            feature_cols += TX1_DYNAMIC_FEATURES

        X, y, timestamps = create_windows(df, feature_cols, lookback, horizon)
        if args.max_windows and len(X) > args.max_windows:
            rng = np.random.default_rng(args.random_state)
            indices = rng.choice(len(X), size=args.max_windows, replace=False)
            X = X[indices]
            y = y[indices]
            timestamps = timestamps[indices]

        X_train, X_test, y_train, y_test, ts_train, ts_test = train_test_split(
            X,
            y,
            timestamps,
            test_size=args.test_size,
            random_state=args.random_state,
            shuffle=True,
        )

        for model_name, builder in MODEL_BUILDERS.items():
            print(f"Training {model_name}")
            model = builder()
            model.fit(X_train, y_train)

            preds_test = model.predict(X_test)
            metrics = evaluate_predictions(y_test, preds_test)
            metrics_records.append(
                {
                    "Transformer": tx_id,
                    "Model": model_name,
                    "Split": "random_80_20",
                    "Lookback": lookback,
                    "Horizon": horizon,
                    **metrics,
                }
            )

            model_path = MODEL_DIR / f"tx{tx_id}_{model_name}.joblib"
            joblib.dump(model, model_path)

            result_df = pd.DataFrame(
                {"timestamp": ts_test, "actual": y_test, "predicted": preds_test}
            )
            result_df = result_df.sort_values("timestamp").reset_index(drop=True)
            result_df.to_csv(
                TABLE_DIR / f"random_tx{tx_id}_{model_name}_predictions.csv", index=False
            )

            plot_scatter(y_test, preds_test, tx_id, model_name)

    metrics_df = pd.DataFrame(metrics_records)
    metrics_df.to_csv(TABLE_DIR / "random_split_performance.csv", index=False)
    print("\nRandom split modelling complete.")


if __name__ == "__main__":
    main()
