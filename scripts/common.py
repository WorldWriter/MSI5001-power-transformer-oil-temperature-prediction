from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
FIG_DIR = RESULTS_DIR / "figures"
TABLE_DIR = RESULTS_DIR / "tables"
CLEAN_DIR = PROJECT_ROOT / "processed"

FIG_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)
CLEAN_DIR.mkdir(parents=True, exist_ok=True)

LOAD_FEATURES: List[str] = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]
TARGET_COL = "OT"


@dataclass(frozen=True)
class DatasetSummary:
    samples: int
    start: pd.Timestamp
    end: pd.Timestamp
    ot_mean: float
    ot_std: float


def load_raw_data() -> pd.DataFrame:
    frames = []
    for idx, filename in enumerate(["trans_1.csv", "trans_2.csv"], start=1):
        path = DATA_DIR / filename
        df = pd.read_csv(path)
        if "date" not in df.columns:
            raise ValueError(f"{filename} missing 'date' column.")
        df["date"] = pd.to_datetime(df["date"])
        df["transformer_id"] = idx
        frames.append(df)
    data = pd.concat(frames, ignore_index=True).sort_values("date").reset_index(drop=True)
    return data


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    enriched["hour"] = enriched["date"].dt.hour
    enriched["dayofweek"] = enriched["date"].dt.dayofweek
    enriched["month"] = enriched["date"].dt.month
    enriched["day_of_year"] = enriched["date"].dt.dayofyear

    enriched["hour_sin"] = np.sin(2 * np.pi * enriched["hour"] / 24)
    enriched["hour_cos"] = np.cos(2 * np.pi * enriched["hour"] / 24)
    enriched["dow_sin"] = np.sin(2 * np.pi * enriched["dayofweek"] / 7)
    enriched["dow_cos"] = np.cos(2 * np.pi * enriched["dayofweek"] / 7)
    enriched["month_sin"] = np.sin(2 * np.pi * enriched["month"] / 12)
    enriched["month_cos"] = np.cos(2 * np.pi * enriched["month"] / 12)
    enriched["doy_sin"] = np.sin(2 * np.pi * enriched["day_of_year"] / 365)
    enriched["doy_cos"] = np.cos(2 * np.pi * enriched["day_of_year"] / 365)

    enriched["is_weekend"] = enriched["dayofweek"].isin([5, 6]).astype(int)
    enriched["is_worktime"] = (
        (enriched["hour"].between(8, 18)) & (enriched["is_weekend"] == 0)
    ).astype(int)
    enriched["season"] = ((enriched["month"] % 12) // 3 + 1).astype(int)
    return enriched


def list_transformers(df: pd.DataFrame) -> List[int]:
    return sorted(df["transformer_id"].unique().tolist())


def summarize_transformer(df: pd.DataFrame) -> DatasetSummary:
    return DatasetSummary(
        samples=len(df),
        start=df["date"].min(),
        end=df["date"].max(),
        ot_mean=df[TARGET_COL].mean(),
        ot_std=df[TARGET_COL].std(),
    )

