from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from .common import (
    CLEAN_DIR,
    LOAD_FEATURES,
    TABLE_DIR,
    TARGET_COL,
    add_time_features,
    list_transformers,
    load_raw_data,
)

ROLLING_WINDOW = 24  # 6 hours (24 * 15min)
Z_THRESHOLD = 3.0
OT_LIMITS = (-20.0, 120.0)
# Focus anomaly screening on high/medium voltage reactive loads
DETECTION_FEATURES = ["HULL", "MULL", TARGET_COL]


def detect_iqr_outliers(series: pd.Series) -> pd.Series:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (series < lower) | (series > upper)


def detect_rolling_zscore(series: pd.Series) -> pd.Series:
    rolling_mean = series.rolling(window=ROLLING_WINDOW, min_periods=12).mean()
    rolling_std = series.rolling(window=ROLLING_WINDOW, min_periods=12).std()
    zscores = (series - rolling_mean) / rolling_std
    return zscores.abs() > Z_THRESHOLD


def detect_physical_outliers(df: pd.DataFrame) -> pd.Series:
    flags = pd.Series(False, index=df.index)
    flags = flags | (df[TARGET_COL].lt(OT_LIMITS[0]) | df[TARGET_COL].gt(OT_LIMITS[1]))
    return flags


def sequential_standardize(series: pd.Series) -> pd.Series:
    exp_mean = series.expanding(min_periods=2).mean().shift(1)
    exp_std = series.expanding(min_periods=2).std(ddof=0).shift(1)
    standardized = (series - exp_mean) / exp_std
    standardized = standardized.fillna(0.0).replace([np.inf, -np.inf], 0.0)
    return standardized


def main() -> None:
    raw = load_raw_data()
    enriched = add_time_features(raw)
    # Remove rows with missing values in key columns
    base_columns = LOAD_FEATURES + [TARGET_COL]
    enriched = enriched.dropna(subset=base_columns).reset_index(drop=True)

    outlier_records = []
    scaler_params: Dict[str, Dict[str, Dict[str, float]]] = {}

    for tx_id in list_transformers(enriched):
        subset = enriched[enriched["transformer_id"] == tx_id].copy()
        subset = subset.sort_values("date").reset_index(drop=True)

        flags_iqr = pd.DataFrame(False, index=subset.index, columns=DETECTION_FEATURES)
        flags_zscore = flags_iqr.copy()
        flags_joint = flags_iqr.copy()

        for feature in DETECTION_FEATURES:
            flags_iqr[feature] = detect_iqr_outliers(subset[feature])
            flags_zscore[feature] = detect_rolling_zscore(subset[feature])
            flags_joint[feature] = flags_iqr[feature] & flags_zscore[feature]

        flags_physical = detect_physical_outliers(subset)
        combined_flag = (
            flags_physical | flags_iqr.any(axis=1) | flags_zscore.any(axis=1)
        )

        outlier_records.append(
            {
                "transformer_id": tx_id,
                "total_rows": len(subset),
                "iqr_outliers": int(flags_iqr.any(axis=1).sum()),
                "rolling_z_outliers": int(flags_zscore.any(axis=1).sum()),
                "joint_outliers": int(flags_joint.any(axis=1).sum()),
                "physical_outliers": int(flags_physical.sum()),
                "combined_outliers": int(combined_flag.sum()),
            }
        )

        cleaned = subset.loc[~combined_flag].reset_index(drop=True)
        cleaned_path = CLEAN_DIR / f"tx{tx_id}_cleaned.csv"
        cleaned.to_csv(cleaned_path, index=False)

        standardized = cleaned.copy()
        scaler_params[f"tx{tx_id}"] = {}
        for feature in LOAD_FEATURES + [TARGET_COL]:
            standardized[feature] = sequential_standardize(cleaned[feature])
            scaler_params[f"tx{tx_id}"][feature] = {
                "method": "expanding_zscore",
                "window": None,
            }

        standardized_path = CLEAN_DIR / f"tx{tx_id}_standardized.csv"
        standardized.to_csv(standardized_path, index=False)

    pd.DataFrame(outlier_records).to_csv(TABLE_DIR / "outlier_detection_summary.csv", index=False)

    with open(CLEAN_DIR / "standardization_params.json", "w", encoding="utf-8") as fp:
        json.dump(scaler_params, fp, indent=2)

    print("Stage 2 preprocessing complete. Cleaned and standardized datasets saved.")


if __name__ == "__main__":
    main()
