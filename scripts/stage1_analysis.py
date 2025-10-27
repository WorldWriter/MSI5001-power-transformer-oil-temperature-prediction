from __future__ import annotations

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from typing import Dict

from .common import (
    FIG_DIR,
    TABLE_DIR,
    LOAD_FEATURES,
    TARGET_COL,
    add_time_features,
    list_transformers,
    load_raw_data,
    summarize_transformer,
)


def ensure_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    converted = df.copy()
    for col in columns:
        converted[col] = pd.to_numeric(converted[col], errors="coerce")
    return converted


def save_missing_summary(df: pd.DataFrame) -> None:
    total = len(df)
    summary = (
        df.isna()
        .sum()
        .to_frame("missing_count")
        .assign(missing_pct=lambda d: d["missing_count"] / total * 100)
    )
    summary.to_csv(TABLE_DIR / "missing_values_summary.csv")


def check_data_types(df: pd.DataFrame) -> None:
    dtype_df = (
        df.dtypes.astype(str)
        .to_frame("dtype")
        .assign(is_numeric=lambda d: d["dtype"].isin(["float64", "float32", "int64", "int32"]))
    )
    dtype_df.to_csv(TABLE_DIR / "data_types.csv")


def plot_transformer_trends(df: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 6))
    for tx_id, group in df.groupby("transformer_id"):
        plt.plot(group["date"], group[TARGET_COL], label=f"Transformer {tx_id}", alpha=0.75)
    plt.title("Oil Temperature Trends by Transformer")
    plt.xlabel("Date")
    plt.ylabel("Oil Temperature (°C)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG_DIR / "ot_trend_by_transformer.png", dpi=200)
    plt.close()


def compute_correlations(df: pd.DataFrame, transformer_id: int) -> pd.DataFrame:
    corr = df[LOAD_FEATURES + [TARGET_COL]].corr()
    corr.to_csv(TABLE_DIR / f"tx{transformer_id}_correlation_matrix.csv")

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="RdYlBu", center=0, fmt=".2f")
    plt.title(f"Transformer {transformer_id} Load vs OT Correlation")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"tx{transformer_id}_correlation_heatmap.png", dpi=200)
    plt.close()
    return corr


def compute_lag_correlations(df: pd.DataFrame, transformer_id: int, max_lag_hours: int = 24) -> None:
    df = df.sort_values("date").reset_index(drop=True)
    steps_per_hour = 4  # 15 min sampling
    lag_hours = range(0, max_lag_hours + 1)

    matrix = []
    for feature in LOAD_FEATURES:
        row = []
        series = df[feature]
        for h in lag_hours:
            lag_steps = h * steps_per_hour
            shifted = series.shift(lag_steps)
            valid = shifted.notna() & df[TARGET_COL].notna()
            if valid.sum() > 10:
                row.append(shifted[valid].corr(df.loc[valid, TARGET_COL]))
            else:
                row.append(np.nan)
        matrix.append(row)

    columns = [f"{h}h" for h in lag_hours]
    corr_df = pd.DataFrame(matrix, index=LOAD_FEATURES, columns=columns)
    corr_df.to_csv(TABLE_DIR / f"tx{transformer_id}_lag_correlation.csv")

    plt.figure(figsize=(12, 6))
    sns.heatmap(corr_df, annot=True, fmt=".2f", cmap="RdYlBu", center=0)
    plt.title(f"Transformer {transformer_id} - Lagged Load vs OT Correlation")
    plt.xlabel("Lag (hours) of load preceding OT")
    plt.ylabel("Load feature")
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"tx{transformer_id}_lag_correlation_heatmap.png", dpi=200)
    plt.close()


def summarize_transformers(df: pd.DataFrame) -> None:
    records: Dict[str, Dict[str, object]] = {}
    for tx_id, group in df.groupby("transformer_id"):
        summary = summarize_transformer(group)
        records[f"tx{tx_id}"] = {
            "samples": summary.samples,
            "start": summary.start,
            "end": summary.end,
            "ot_mean": summary.ot_mean,
            "ot_std": summary.ot_std,
        }
    pd.DataFrame.from_dict(records, orient="index").to_csv(
        TABLE_DIR / "transformer_summary.csv"
    )


def main() -> None:
    raw = load_raw_data()
    raw = ensure_numeric(raw, LOAD_FEATURES + [TARGET_COL])
    save_missing_summary(raw)
    check_data_types(raw)
    summarize_transformers(raw)

    enriched = add_time_features(raw)
    plot_transformer_trends(enriched)

    for tx_id in list_transformers(enriched):
        subset = enriched[enriched["transformer_id"] == tx_id]
        compute_correlations(subset, tx_id)
        compute_lag_correlations(subset, tx_id)

    print("Stage 1 analysis complete. Tables and figures stored in results/.")


if __name__ == "__main__":
    main()
