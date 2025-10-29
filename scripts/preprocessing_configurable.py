"""
Configurable preprocessing script for experiments.

This script provides flexible outlier detection and data cleaning options
to support systematic experiments with different preprocessing strategies.

Usage:
    # No outlier removal
    python -m scripts.preprocessing_configurable --outlier-method none --save-suffix "_no_outlier"

    # Remove 1% outliers
    python -m scripts.preprocessing_configurable --outlier-method percentile --outlier-percentile 1.0 --save-suffix "_1pct"

    # Remove 5% outliers
    python -m scripts.preprocessing_configurable --outlier-method percentile --outlier-percentile 5.0 --save-suffix "_5pct"

    # Default IQR method
    python -m scripts.preprocessing_configurable --outlier-method iqr
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

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
from .experiment_utils import remove_outliers_configurable

# Physical limits for Oil Temperature
OT_LIMITS = (-20.0, 120.0)
DETECTION_FEATURES = ["HULL", "MULL", TARGET_COL]


def detect_physical_outliers(df: pd.DataFrame) -> pd.Series:
    """Detect physically impossible values."""
    flags = pd.Series(False, index=df.index)
    flags = flags | (df[TARGET_COL].lt(OT_LIMITS[0]) | df[TARGET_COL].gt(OT_LIMITS[1]))
    return flags


def sequential_standardize(series: pd.Series) -> pd.Series:
    """Sequential standardization using expanding window."""
    exp_mean = series.expanding(min_periods=2).mean().shift(1)
    exp_std = series.expanding(min_periods=2).std(ddof=0).shift(1)
    standardized = (series - exp_mean) / exp_std
    standardized = standardized.fillna(0.0).replace([np.inf, -np.inf], 0.0)
    return standardized


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Configurable preprocessing with outlier detection options"
    )
    parser.add_argument(
        "--outlier-method",
        type=str,
        choices=["none", "iqr", "percentile"],
        default="iqr",
        help="Outlier detection method (default: iqr)"
    )
    parser.add_argument(
        "--outlier-percentile",
        type=float,
        default=1.0,
        help="Percentile to remove for 'percentile' method (e.g., 1.0, 5.0)"
    )
    parser.add_argument(
        "--iqr-multiplier",
        type=float,
        default=1.5,
        help="IQR multiplier for 'iqr' method (default: 1.5)"
    )
    parser.add_argument(
        "--save-suffix",
        type=str,
        default="",
        help="Suffix for output files (e.g., '_no_outlier', '_1pct')"
    )
    parser.add_argument(
        "--keep-physical-check",
        action="store_true",
        help="Always apply physical limits check regardless of outlier method"
    )

    args = parser.parse_args()

    # Load and enrich data
    print("Loading raw data...")
    raw = load_raw_data()
    enriched = add_time_features(raw)

    # Remove missing values
    base_columns = LOAD_FEATURES + [TARGET_COL]
    enriched = enriched.dropna(subset=base_columns).reset_index(drop=True)

    print(f"\nOutlier detection method: {args.outlier_method}")
    if args.outlier_method == "percentile":
        print(f"  Percentile threshold: {args.outlier_percentile}%")
    elif args.outlier_method == "iqr":
        print(f"  IQR multiplier: {args.iqr_multiplier}")
    elif args.outlier_method == "none":
        print("  No outlier removal (keeping all data)")

    outlier_records = []
    scaler_params: Dict[str, Dict[str, Dict[str, float]]] = {}

    for tx_id in list_transformers(enriched):
        print(f"\nProcessing Transformer {tx_id}...")
        subset = enriched[enriched["transformer_id"] == tx_id].copy()
        subset = subset.sort_values("date").reset_index(drop=True)

        total_rows = len(subset)

        # Apply configurable outlier detection
        if args.outlier_method == "none":
            combined_flag = pd.Series(False, index=subset.index)
            outlier_count = 0
        else:
            cleaned_subset, outlier_flags = remove_outliers_configurable(
                subset,
                method=args.outlier_method,
                features=DETECTION_FEATURES,
                percentile=args.outlier_percentile,
                iqr_multiplier=args.iqr_multiplier
            )
            combined_flag = outlier_flags
            outlier_count = int(combined_flag.sum())

        # Always check physical limits if requested
        if args.keep_physical_check or args.outlier_method == "none":
            flags_physical = detect_physical_outliers(subset)
            combined_flag = combined_flag | flags_physical
            physical_count = int(flags_physical.sum())
        else:
            physical_count = 0

        outlier_records.append(
            {
                "transformer_id": tx_id,
                "total_rows": total_rows,
                "method": args.outlier_method,
                "outliers_detected": outlier_count,
                "physical_outliers": physical_count,
                "combined_outliers": int(combined_flag.sum()),
                "remaining_rows": total_rows - int(combined_flag.sum()),
                "removal_rate": f"{100 * combined_flag.sum() / total_rows:.2f}%"
            }
        )

        # Save cleaned data
        cleaned = subset.loc[~combined_flag].reset_index(drop=True)
        cleaned_filename = f"tx{tx_id}_cleaned{args.save_suffix}.csv"
        cleaned_path = CLEAN_DIR / cleaned_filename
        cleaned.to_csv(cleaned_path, index=False)
        print(f"  Saved: {cleaned_path}")
        print(f"  Rows: {total_rows} -> {len(cleaned)} "
              f"(removed {int(combined_flag.sum())} = {100 * combined_flag.sum() / total_rows:.2f}%)")

        # Standardize data
        standardized = cleaned.copy()
        scaler_params[f"tx{tx_id}"] = {}
        for feature in LOAD_FEATURES + [TARGET_COL]:
            standardized[feature] = sequential_standardize(cleaned[feature])
            scaler_params[f"tx{tx_id}"][feature] = {
                "method": "expanding_zscore",
                "window": None,
            }

        standardized_filename = f"tx{tx_id}_standardized{args.save_suffix}.csv"
        standardized_path = CLEAN_DIR / standardized_filename
        standardized.to_csv(standardized_path, index=False)
        print(f"  Saved: {standardized_path}")

    # Save outlier detection summary
    summary_df = pd.DataFrame(outlier_records)
    summary_filename = f"outlier_detection_summary{args.save_suffix}.csv"
    summary_path = TABLE_DIR / summary_filename
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved outlier summary: {summary_path}")

    # Save standardization parameters
    params_filename = f"standardization_params{args.save_suffix}.json"
    params_path = CLEAN_DIR / params_filename
    with open(params_path, "w", encoding="utf-8") as fp:
        json.dump(scaler_params, fp, indent=2)
    print(f"Saved standardization params: {params_path}")

    print("\n" + "="*60)
    print("Preprocessing complete!")
    print("="*60)
    print("\nSummary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
