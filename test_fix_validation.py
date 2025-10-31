#!/usr/bin/env python
"""
Quick validation script to test the fixed Informer data loading
"""
import sys
sys.path.append('./external/Informer2020')

from data.data_loader import Dataset_Custom

# Test MS mode with fixed code
dataset = Dataset_Custom(
    root_path='./external/Informer2020/data/',
    flag='train',
    size=[96, 48, 1],  # seq_len, label_len, pred_len
    features='MS',
    data_path='TX2.csv',
    target='OT',
    scale=True,
    inverse=False,
    timeenc=0,
    freq='h',
    cols=None
)

print("=" * 80)
print("Data Loading Validation - Fixed MS Mode")
print("=" * 80)
print(f"Dataset size: {len(dataset)}")
print(f"\ndata_x shape: {dataset.data_x.shape}")
print(f"data_y shape: {dataset.data_y.shape}")
print(f"\nExpected:")
print(f"  data_x: (N, 6) - 6 load features (HUFL-LULL)")
print(f"  data_y: (N, 1) - 1 target (OT)")

# Get a sample
seq_x, seq_y, seq_x_mark, seq_y_mark = dataset[0]
print(f"\nSample batch:")
print(f"  seq_x shape: {seq_x.shape} (should be (96, 6))")
print(f"  seq_y shape: {seq_y.shape} (should be (49, 1))")

# Check if target_scaler exists
if hasattr(dataset, 'target_scaler'):
    print(f"\n✓ target_scaler exists (for separate OT scaling)")
    print(f"  OT mean: {dataset.target_scaler.mean[0]:.2f}")
    print(f"  OT std: {dataset.target_scaler.std[0]:.2f}")
else:
    print(f"\n✗ target_scaler missing!")

# Check scaler for input features
print(f"\n✓ Input scaler (for 6 loads):")
print(f"  Shape: {dataset.scaler.mean.shape}")

print("\n" + "=" * 80)
if dataset.data_x.shape[1] == 6 and dataset.data_y.shape[1] == 1:
    print("✓ VALIDATION PASSED: data_x and data_y shapes are correct!")
else:
    print("✗ VALIDATION FAILED: Incorrect shapes!")
print("=" * 80)
