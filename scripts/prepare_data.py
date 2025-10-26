#!/usr/bin/env python3
"""
Data Preprocessing Script for MSI5001 Project
==============================================

This script processes the raw ETT dataset and creates train/test splits.

Author: MSI5001 Group Project Team
Date: October 2025

Usage:
    python scripts/prepare_data.py
"""

import pandas as pd
import os
from pathlib import Path

# Configuration
RAW_DATA_DIR = Path(__file__).parent.parent / 'dataset' / 'raw_data'
PROCESSED_DATA_DIR = Path(__file__).parent.parent / 'dataset' / 'processed_data'
TRAIN_SPLIT_RATIO = 0.8

def load_raw_data(filename='train1.csv'):
    """
    Load raw data from CSV file

    Parameters:
    -----------
    filename : str
        Name of the CSV file in raw_data directory

    Returns:
    --------
    df : pd.DataFrame
        Loaded dataframe
    """
    filepath = RAW_DATA_DIR / filename
    print(f"Loading data from: {filepath}")

    df = pd.read_csv(filepath)
    print(f"Loaded {len(df):,} rows")

    return df


def split_data(df, train_ratio=0.8):
    """
    Split data into train and test sets (temporal split)

    IMPORTANT: For time series data, we split by time order, NOT randomly

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    train_ratio : float
        Proportion of data for training (default: 0.8)

    Returns:
    --------
    train_df : pd.DataFrame
        Training set
    test_df : pd.DataFrame
        Test set
    """
    # Calculate split point
    split_idx = int(len(df) * train_ratio)

    # Split data (temporal order)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    print(f"\nData split:")
    print(f"  Training set: {len(train_df):,} rows ({train_ratio*100:.1f}%)")
    print(f"  Test set:     {len(test_df):,} rows ({(1-train_ratio)*100:.1f}%)")

    return train_df, test_df


def save_processed_data(train_df, test_df):
    """
    Save processed train and test sets

    Parameters:
    -----------
    train_df : pd.DataFrame
        Training set
    test_df : pd.DataFrame
        Test set
    """
    # Ensure output directory exists
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Save train set
    train_path = PROCESSED_DATA_DIR / 'train.csv'
    train_df.to_csv(train_path, index=False)
    print(f"\nSaved training set to: {train_path}")
    print(f"  Size: {os.path.getsize(train_path) / 1024 / 1024:.2f} MB")

    # Save test set
    test_path = PROCESSED_DATA_DIR / 'test.csv'
    test_df.to_csv(test_path, index=False)
    print(f"Saved test set to: {test_path}")
    print(f"  Size: {os.path.getsize(test_path) / 1024 / 1024:.2f} MB")


def print_data_summary(df, name='Dataset'):
    """
    Print summary statistics of the dataset

    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    name : str
        Name of the dataset for display
    """
    print(f"\n{'='*60}")
    print(f"{name} Summary")
    print(f"{'='*60}")
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nDate range:")
    if 'date' in df.columns:
        print(f"  Start: {df['date'].iloc[0]}")
        print(f"  End:   {df['date'].iloc[-1]}")
    print(f"\nBasic statistics:")
    print(df.describe())
    print(f"{'='*60}\n")


def main():
    """
    Main preprocessing pipeline
    """
    print("="*60)
    print("ETT Dataset Preprocessing")
    print("="*60)
    print(f"Raw data directory: {RAW_DATA_DIR}")
    print(f"Processed data directory: {PROCESSED_DATA_DIR}")
    print(f"Train/Test split ratio: {TRAIN_SPLIT_RATIO*100:.0f}% / {(1-TRAIN_SPLIT_RATIO)*100:.0f}%")
    print("="*60)

    # Step 1: Load raw data
    print("\n[Step 1] Loading raw data...")
    df = load_raw_data('train1.csv')

    # Step 2: Print summary
    print_data_summary(df, 'Raw Data')

    # Step 3: Split data
    print("\n[Step 2] Splitting data...")
    train_df, test_df = split_data(df, train_ratio=TRAIN_SPLIT_RATIO)

    # Step 4: Save processed data
    print("\n[Step 3] Saving processed data...")
    save_processed_data(train_df, test_df)

    # Step 5: Verify saved data
    print("\n[Step 4] Verifying saved data...")
    print_data_summary(train_df, 'Training Set')
    print_data_summary(test_df, 'Test Set')

    print("\n" + "="*60)
    print("✅ Data preprocessing completed successfully!")
    print("="*60)
    print("\nNext steps:")
    print("  1. Update notebooks to use new data paths")
    print("  2. Run training with the new datasets")
    print("  3. Compare model performance")


if __name__ == '__main__':
    main()
