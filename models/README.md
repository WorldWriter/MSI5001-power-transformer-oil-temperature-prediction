# Models Directory

This directory contains all trained models and preprocessing components organized by type.

## Directory Structure

### `traditional_ml/`
Contains traditional machine learning models:
- **Linear Regression**: `lr_1h.pkl`, `lr_1d.pkl`, `lr_1w.pkl`, `linear_regression_1h.pkl`
- **Random Forest**: `rf_1h.pkl`, `rf_1d.pkl`, `rf_1w.pkl`
- **Ridge Regression**: `ridge_1h.pkl`, `ridge_1d.pkl`, `ridge_1w.pkl`, `ridge_regression_1h.pkl`

### `deep_learning/`
Contains Multi-Layer Perceptron (MLP) models with different architectures:
- **Small MLP**: `mlp_small_1h.pkl`, `mlp_small_1d.pkl`, `mlp_small_1w.pkl`
- **Medium MLP**: `mlp_medium_1h.pkl`, `mlp_medium_1d.pkl`, `mlp_medium_1w.pkl`
- **Large MLP**: `mlp_large_1h.pkl`, `mlp_large_1d.pkl`, `mlp_large_1w.pkl`

### `scalers/`
Contains data preprocessing scalers for feature normalization:
- **Original Features**: `scaler_1h.pkl`, `scaler_1d.pkl`, `scaler_1w.pkl`
- **Time-Enhanced Features**: `scaler_1h_time.pkl`, `scaler_1d_time.pkl`, `scaler_1w_time.pkl`

## Naming Convention

All models follow the pattern: `{model_type}_{prediction_span}.pkl`
- **model_type**: lr (Linear Regression), rf (Random Forest), ridge (Ridge Regression), mlp_{size} (MLP variants)
- **prediction_span**: 1h (1 hour), 1d (1 day), 1w (1 week)
- **time suffix**: Models trained with time features have `_time` suffix in their corresponding scalers

## Usage Notes

1. Each model requires its corresponding scaler for proper data preprocessing
2. Time-enhanced models should use scalers with `_time` suffix
3. Models are trained for specific prediction horizons (1h, 1d, 1w)
4. All models are saved using joblib/pickle format for easy loading