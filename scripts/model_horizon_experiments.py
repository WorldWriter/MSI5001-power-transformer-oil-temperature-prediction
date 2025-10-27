from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from .common import (
    CLEAN_DIR,
    FIG_DIR,
    TABLE_DIR,
    TARGET_COL,
    add_time_features,
)

MODEL_DIR = Path(__file__).resolve().parents[1] / "models" / "horizon_experiments"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BASE_TIME_FEATURES: List[str] = [
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

CONFIGS = {
    "1h": {"lookback": 24, "horizon": 1, "gap": 0},
    "1d": {"lookback": 48, "horizon": 24, "gap": 6},
    "1w": {"lookback": 168, "horizon": 168, "gap": 24},
}

MODEL_BUILDERS = {
    "LinearRegression": lambda: LinearRegression(),
    "Ridge": lambda: Ridge(alpha=10.0, random_state=42),
    "RandomForest": lambda: RandomForestRegressor(
        n_estimators=200, max_depth=14, min_samples_leaf=5, random_state=42, n_jobs=-1
    ),
    "MLP": lambda: MLPRegressor(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        learning_rate_init=1e-3,
        alpha=1e-4,
        max_iter=150,
        random_state=42,
        early_stopping=True,
    ),
}


def resample_hourly(df: pd.DataFrame) -> pd.DataFrame:
    hourly = df.set_index("date").resample("1H").mean().dropna().reset_index()
    return hourly


def enrich_tx1(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["HULL_diff1"] = enriched["HULL"].diff().fillna(0)
    enriched["MULL_diff1"] = enriched["MULL"].diff().fillna(0)
    enriched["HULL_roll12"] = (
        enriched["HULL"].rolling(window=12, min_periods=1).mean().fillna(method="bfill")
    )
    enriched["MULL_roll12"] = (
        enriched["MULL"].rolling(window=12, min_periods=1).mean().fillna(method="bfill")
    )
    return enriched


def create_windows(
    df: pd.DataFrame,
    feature_cols: List[str],
    lookback: int,
    horizon: int,
    gap: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = df[feature_cols].to_numpy(dtype=np.float32)
    target = df[TARGET_COL].to_numpy(dtype=np.float32)
    timestamps = df["date"].to_numpy()

    X_list: List[np.ndarray] = []
    y_list: List[float] = []
    ts_list: List[np.datetime64] = []

    for tgt_idx in range(lookback + gap + horizon, len(df)):
        end_idx = tgt_idx - horizon - gap
        start_idx = end_idx - lookback
        if start_idx < 0:
            continue
        window = values[start_idx:end_idx]
        if window.shape[0] != lookback:
            continue
        X_list.append(window.reshape(-1))
        y_list.append(target[tgt_idx])
        ts_list.append(timestamps[tgt_idx])

    return np.array(X_list), np.array(y_list), np.array(ts_list)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def plot_prediction(
    timestamps: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    transformer: int,
    config_name: str,
    model_name: str,
) -> None:
    plt.figure(figsize=(12, 4))
    order = np.argsort(timestamps)
    ts_sorted = timestamps[order]
    y_true_sorted = y_true[order]
    y_pred_sorted = y_pred[order]
    plt.plot(ts_sorted, y_true_sorted, label="Actual", linewidth=1.5)
    plt.plot(ts_sorted, y_pred_sorted, label="Predicted", linewidth=1.5, alpha=0.8)
    plt.title(f"TX{transformer} - {config_name} - {model_name}")
    plt.xlabel("Timestamp")
    plt.ylabel("Oil Temperature (°C)")
    plt.legend()
    plt.tight_layout()
    fname = FIG_DIR / f"horizon_tx{transformer}_{config_name}_{model_name}.png"
    plt.savefig(fname, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Horizon experiments (1h/1d/1w).")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    args = parser.parse_args()

    metrics_records: List[Dict[str, float]] = []

    for tx_id in [1, 2]:
        print(f"\n=== Transformer {tx_id} ===")
        df = pd.read_csv(CLEAN_DIR / f"tx{tx_id}_cleaned.csv", parse_dates=["date"])
        df_hourly = resample_hourly(df)
        df_hourly = add_time_features(df_hourly)
        feature_cols = ["HULL", "MULL"] + BASE_TIME_FEATURES

        if tx_id == 1:
            df_hourly = enrich_tx1(df_hourly)
            feature_cols += TX1_DYNAMIC_FEATURES

        for config_name, cfg in CONFIGS.items():
            lookback = cfg["lookback"]
            horizon = cfg["horizon"]
            gap = cfg["gap"]
            print(f"Config {config_name}: lookback={lookback}, horizon={horizon}, gap={gap}")

            X, y, timestamps = create_windows(
                df_hourly, feature_cols, lookback, horizon, gap
            )

            if len(X) == 0:
                print("  Skipped due to insufficient data.")
                continue

            X_train, X_test, y_train, y_test, ts_train, ts_test = train_test_split(
                X,
                y,
                timestamps,
                test_size=args.test_size,
                random_state=args.random_state,
                shuffle=True,
            )

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            for model_name, builder in MODEL_BUILDERS.items():
                model = builder()
                model.fit(X_train_scaled, y_train)
                preds = model.predict(X_test_scaled)
                metrics = evaluate(y_test, preds)
                metrics_records.append(
                    {
                        "Transformer": tx_id,
                        "Config": config_name,
                        "Lookback": lookback,
                        "Horizon_steps": horizon,
                        "Gap_steps": gap,
                        "Model": model_name,
                        **metrics,
                    }
                )

                model_path = MODEL_DIR / f"tx{tx_id}_{config_name}_{model_name}.joblib"
                joblib.dump({"model": model, "scaler": scaler, "feature_cols": feature_cols}, model_path)

                result_df = pd.DataFrame(
                    {"timestamp": ts_test, "actual": y_test, "predicted": preds}
                ).sort_values("timestamp")
                result_df.to_csv(
                    TABLE_DIR / f"horizon_tx{tx_id}_{config_name}_{model_name}.csv",
                    index=False,
                )

                plot_prediction(ts_test, y_test, preds, tx_id, config_name, model_name)

    metrics_df = pd.DataFrame(metrics_records)
    metrics_df.to_csv(TABLE_DIR / "horizon_experiment_metrics.csv", index=False)
    print("\nHorizon experiments complete.")


if __name__ == "__main__":
    main()
