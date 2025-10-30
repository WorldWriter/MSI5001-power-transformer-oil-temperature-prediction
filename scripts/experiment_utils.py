"""
Experiment utilities for configurable parameter experiments.

This module provides parameterized versions of data processing, feature selection,
and data splitting functions to support systematic experiments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


# ============================================================================
# Feature Configuration
# ============================================================================

LOAD_FEATURES = ["HULL", "MULL"]
LOAD_FEATURES_FULL = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]  # All 6 load features

TIME_FEATURES = [
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

TX1_DYNAMIC_FEATURES = [
    "HULL_diff1",
    "MULL_diff1",
    "HULL_roll12",
    "MULL_roll12",
]


def select_features_by_mode(
    feature_mode: str = "full",
    include_tx1_dynamic: bool = False,
    use_full_loads: bool = False
) -> List[str]:
    """
    Select feature columns based on experiment mode.

    Parameters
    ----------
    feature_mode : str
        Feature selection mode:
        - 'full': Load features + Time features
        - 'time_only': Only time features
        - 'no_time': Only load features
    include_tx1_dynamic : bool
        Whether to include TX1-specific dynamic features (diff, rolling mean)
    use_full_loads : bool
        Whether to use all 6 load features (HUFL, HULL, MUFL, MULL, LUFL, LULL)
        instead of just 2 (HULL, MULL)

    Returns
    -------
    List[str]
        List of feature column names
    """
    if feature_mode == "time_only":
        features = TIME_FEATURES.copy()
    elif feature_mode == "no_time":
        features = LOAD_FEATURES_FULL.copy() if use_full_loads else LOAD_FEATURES.copy()
    elif feature_mode == "full":
        features = LOAD_FEATURES_FULL.copy() if use_full_loads else LOAD_FEATURES.copy()
        features.extend(TIME_FEATURES)
    else:
        raise ValueError(f"Unknown feature_mode: {feature_mode}. "
                        f"Must be one of: full, time_only, no_time")

    if include_tx1_dynamic and feature_mode != "time_only":
        features.extend(TX1_DYNAMIC_FEATURES)

    return features


# ============================================================================
# Outlier Detection Configuration
# ============================================================================

def remove_outliers_by_percentile(
    df: pd.DataFrame,
    features: List[str],
    percentile: float = 1.0
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Remove outliers by percentile threshold.

    Remove the most extreme percentile% of data points for each feature.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    features : List[str]
        Features to check for outliers
    percentile : float
        Percentage to remove (e.g., 1.0 means remove top/bottom 0.5% each)
        Set to 0 to keep all data.

    Returns
    -------
    cleaned_df : pd.DataFrame
        Dataframe with outliers removed
    outlier_flags : pd.Series
        Boolean series indicating which rows were outliers
    """
    if percentile == 0:
        return df, pd.Series(False, index=df.index)

    outlier_flags = pd.Series(False, index=df.index)

    for feature in features:
        if feature not in df.columns:
            continue

        lower_quantile = percentile / 200  # e.g., 1% -> 0.005
        upper_quantile = 1 - (percentile / 200)  # e.g., 1% -> 0.995

        lower_bound = df[feature].quantile(lower_quantile)
        upper_bound = df[feature].quantile(upper_quantile)

        feature_outliers = (df[feature] < lower_bound) | (df[feature] > upper_bound)
        outlier_flags |= feature_outliers

    cleaned_df = df[~outlier_flags].reset_index(drop=True)

    return cleaned_df, outlier_flags


def remove_outliers_iqr(
    df: pd.DataFrame,
    features: List[str],
    iqr_multiplier: float = 1.5
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Remove outliers using IQR (Interquartile Range) method.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    features : List[str]
        Features to check for outliers
    iqr_multiplier : float
        IQR multiplier for bounds (default 1.5, standard boxplot rule)

    Returns
    -------
    cleaned_df : pd.DataFrame
        Dataframe with outliers removed
    outlier_flags : pd.Series
        Boolean series indicating which rows were outliers
    """
    outlier_flags = pd.Series(False, index=df.index)

    for feature in features:
        if feature not in df.columns:
            continue

        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - iqr_multiplier * IQR
        upper_bound = Q3 + iqr_multiplier * IQR

        feature_outliers = (df[feature] < lower_bound) | (df[feature] > upper_bound)
        outlier_flags |= feature_outliers

    cleaned_df = df[~outlier_flags].reset_index(drop=True)

    return cleaned_df, outlier_flags


def remove_outliers_configurable(
    df: pd.DataFrame,
    method: str = "iqr",
    features: List[str] = None,
    **kwargs
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Remove outliers using configurable method.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe
    method : str
        Outlier detection method:
        - 'none': No outlier removal
        - 'iqr': IQR method (default)
        - 'percentile': Remove by percentile
    features : List[str], optional
        Features to check. If None, uses ["HULL", "MULL", "OT"]
    **kwargs
        Additional parameters for specific methods:
        - iqr_multiplier: float (for 'iqr')
        - percentile: float (for 'percentile')

    Returns
    -------
    cleaned_df : pd.DataFrame
        Dataframe with outliers removed
    outlier_flags : pd.Series
        Boolean series indicating which rows were outliers
    """
    if features is None:
        features = ["HULL", "MULL", "OT"]

    if method == "none":
        return df, pd.Series(False, index=df.index)

    elif method == "iqr":
        iqr_multiplier = kwargs.get("iqr_multiplier", 1.5)
        return remove_outliers_iqr(df, features, iqr_multiplier)

    elif method == "percentile":
        percentile = kwargs.get("percentile", 1.0)
        return remove_outliers_by_percentile(df, features, percentile)

    else:
        raise ValueError(f"Unknown outlier detection method: {method}. "
                        f"Must be one of: none, iqr, percentile")


# ============================================================================
# Data Splitting Configuration
# ============================================================================

def chronological_split(
    df: pd.DataFrame,
    train_ratio: float = 0.8
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data chronologically (time-series split).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (should be sorted by date)
    train_ratio : float
        Ratio of training data

    Returns
    -------
    train_df : pd.DataFrame
        Training data
    test_df : pd.DataFrame
        Testing data
    """
    cutoff = int(len(df) * train_ratio)
    train_df = df.iloc[:cutoff].reset_index(drop=True)
    test_df = df.iloc[cutoff:].reset_index(drop=True)
    return train_df, test_df


def group_random_split(
    df: pd.DataFrame,
    n_groups: int = 20,
    test_ratio: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data by random group selection.

    Divide data into groups, then randomly select groups for train/test.
    This reduces data leakage compared to pure random split.

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe (should be sorted by date)
    n_groups : int
        Number of groups to divide data into
    test_ratio : float
        Ratio of test groups
    random_state : int
        Random seed

    Returns
    -------
    train_df : pd.DataFrame
        Training data
    test_df : pd.DataFrame
        Testing data
    """
    df = df.sort_values("date").reset_index(drop=True)

    # Create groups
    group_size = len(df) // n_groups
    df["_group_id"] = df.index // group_size

    # Randomly select test groups
    rng = np.random.default_rng(random_state)
    unique_groups = df["_group_id"].unique()
    n_test_groups = int(len(unique_groups) * test_ratio)
    test_groups = rng.choice(unique_groups, size=n_test_groups, replace=False)

    # Split data
    train_mask = ~df["_group_id"].isin(test_groups)
    train_df = df[train_mask].drop("_group_id", axis=1).reset_index(drop=True)
    test_df = df[~train_mask].drop("_group_id", axis=1).reset_index(drop=True)

    return train_df, test_df


# ============================================================================
# Window Configuration
# ============================================================================

@dataclass
class WindowConfig:
    """
    Configuration for sliding window creation.

    Attributes
    ----------
    horizon : int
        Number of steps to predict ahead
    lookback_multiplier : float
        Multiplier for lookback window (lookback = horizon * multiplier)
    gap : int
        Gap between lookback window and prediction target
    """
    horizon: int
    lookback_multiplier: float
    gap: int = 0

    @property
    def lookback(self) -> int:
        """Calculate lookback window size."""
        return int(self.horizon * self.lookback_multiplier)

    def __repr__(self) -> str:
        return (f"WindowConfig(horizon={self.horizon}, "
                f"lookback={self.lookback}, gap={self.gap})")


def create_window_config(
    horizon: int,
    lookback_multiplier: float,
    gap: int = 0
) -> WindowConfig:
    """
    Create a window configuration.

    Parameters
    ----------
    horizon : int
        Prediction horizon (e.g., 1 for 1 hour, 24 for 1 day)
    lookback_multiplier : float
        Lookback window size as a multiple of horizon
        (e.g., 4 means lookback = 4 * horizon)
    gap : int
        Gap between lookback and target (default 0)

    Returns
    -------
    WindowConfig
        Window configuration object
    """
    return WindowConfig(
        horizon=horizon,
        lookback_multiplier=lookback_multiplier,
        gap=gap
    )


# Predefined window configurations for common experiments
PREDEFINED_WINDOW_CONFIGS = {
    "1h_1x": WindowConfig(horizon=1, lookback_multiplier=1, gap=0),
    "1h_4x": WindowConfig(horizon=1, lookback_multiplier=4, gap=0),
    "1h_8x": WindowConfig(horizon=1, lookback_multiplier=8, gap=0),
    "1d_1x": WindowConfig(horizon=24, lookback_multiplier=1, gap=6),
    "1d_4x": WindowConfig(horizon=24, lookback_multiplier=4, gap=6),
    "1d_8x": WindowConfig(horizon=24, lookback_multiplier=8, gap=6),
    "1w_1x": WindowConfig(horizon=168, lookback_multiplier=1, gap=24),
    "1w_4x": WindowConfig(horizon=168, lookback_multiplier=4, gap=24),
    "1w_8x": WindowConfig(horizon=168, lookback_multiplier=8, gap=24),
}


def get_window_config(config_name: str) -> WindowConfig:
    """
    Get predefined window configuration by name.

    Parameters
    ----------
    config_name : str
        Configuration name (e.g., '1h_4x', '1d_1x', '1w_8x')

    Returns
    -------
    WindowConfig
        Window configuration object

    Raises
    ------
    ValueError
        If configuration name is not found
    """
    if config_name not in PREDEFINED_WINDOW_CONFIGS:
        raise ValueError(
            f"Unknown window config: {config_name}. "
            f"Available: {list(PREDEFINED_WINDOW_CONFIGS.keys())}"
        )
    return PREDEFINED_WINDOW_CONFIGS[config_name]
