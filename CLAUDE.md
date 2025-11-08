# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an MSI5001 academic project for **multi-horizon time-series forecasting** of power transformer oil temperature. The project evaluates 5 models (LinearRegression, RandomForest, MLP, RNN, Informer) across 3 prediction horizons (1 hour, 1 day, 1 week) on 2 transformer datasets (TX1=industrial, TX2=residential).

**Key Focus**: Rigorous methodology with data leakage prevention via chronological splits.

## Common Commands

### Installation & Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Verify installation
python -m scripts.generate_report_figures
```

### Training Individual Models
```bash
# Main training interface
python -m scripts.train_configurable \
    --tx-id {1|2} \
    --model {LinearRegression|RandomForest|MLP|RNN|Informer} \
    --split-method {chronological|random_window|group_random} \
    --feature-mode {full|no_time|time_only|loads_6_only} \
    --horizon {1|96|672} \
    --lookback-multiplier {2.0|4.0|8.0}

# Example: Train Informer on TX1 for 1-hour prediction
python -m scripts.train_configurable \
    --tx-id 1 \
    --model Informer \
    --split-method chronological \
    --feature-mode full \
    --horizon 1
```

### Data Preprocessing
```bash
# Preprocess with custom outlier removal
python -m scripts.preprocessing_configurable \
    --outlier-method {none|iqr|percentile} \
    --outlier-percentile {0.5|1.0|5.0} \
    --save-suffix "_1pct"
```

### Running Batch Experiments
```bash
# Run all experiments from CSV config
python -m scripts.run_experiments \
    --config configs/experiment_plan.csv \
    --run-preprocessing

# Run specific experiments
python -m scripts.run_experiments \
    --config configs/experiment_plan.csv \
    --exp-ids 1,2,3

# Preview commands without executing
python -m scripts.run_experiments \
    --config configs/experiment_plan.csv \
    --dry-run
```

### Visualization
```bash
# Generate all 11 report figures
python -m scripts.generate_report_figures

# Output: results/figures/*.png
```

### Working with Jupyter Notebook
```bash
# Main deliverable (course evaluation)
jupyter notebook MSI5001_Model_Training_Workflow.ipynb

# Set FORCE_RETRAIN = True to retrain all models from scratch
# Set FORCE_RETRAIN = False to load cached results (default)
```

## Architecture & Code Structure

### Data Pipeline Flow
1. **Raw Data**: `data/raw/trans_{1,2}.csv` (69,680 samples each, 15-min interval)
2. **Feature Engineering**: `scripts/common.py::add_time_features()` adds cyclical time encoding (sin/cos for hour/day/month/year)
3. **Preprocessing**: `scripts/preprocessing_configurable.py` handles outlier removal
4. **Data Splitting**: `scripts/experiment_utils.py` provides 3 split strategies:
   - `chronological_split()`: Temporal split (80/20) - **PREFERRED for true generalization**
   - Sliding window random split - causes ~5× data leakage inflation
   - `group_random_split()` - grouped random split
5. **Training**: `scripts/train_configurable.py` orchestrates model training
6. **Results**: `results/experiments/exp_*.csv` and `results/figures/exp_*.png`

### Model Implementations
All models follow sklearn-compatible interface with `.fit(X, y)` and `.predict(X)`:

- **Traditional ML**: `sklearn.ensemble.RandomForestRegressor`, `sklearn.linear_model.LinearRegression`
- **MLP**: `models/pytorch_mlp.py::PyTorchMLPRegressor` - PyTorch MLP with GPU support
- **RNN**: `models/pytorch_rnn.py::PyTorchRNNRegressor` - LSTM-based time series model
- **Informer**: `models/pytorch_informer.py::PyTorchInformerRegressor` - wraps Informer architecture from `models/informer_arch/`

All PyTorch models:
- Auto-detect best device (CUDA > MPS > CPU) via `get_device()`
- Support early stopping and model checkpointing
- Use `random_state=42` for reproducibility

### Critical Feature Engineering Details

**Temporal Features** (`scripts/common.py:49-70`):
- Cyclical encoding (sin/cos) for hour (0-23), day of week (0-6), month (1-12), day of year (1-365)
- Binary features: `is_weekend`, `is_worktime` (8am-6pm weekdays)
- Season categorization (1-4)
- **Impact**: Removing these causes 0.5-0.7 R² degradation across all models

**Load Features** (`scripts/common.py:22`):
```python
LOAD_FEATURES = ["HUFL", "HULL", "MUFL", "MULL", "LUFL", "LULL"]
TARGET_COL = "OT"  # Oil Temperature
```

### Experiment Configuration System

**Configuration File**: `configs/experiment_plan.csv` (111 experiments)
- Columns: transformer ID, model type, split method, outlier removal, horizon, features, lookback multiplier
- Each row maps to one `train_configurable.py` execution
- Results stored with experiment IDs (exp_001 to exp_111)

**Experiment Tracking**:
- **Config**: `models/experiments/exp_*.json` (hyperparameters, metadata)
- **Metrics**: `results/tables/exp_*_predictions.csv` (predictions, R², RMSE, MAE)
- **Logs**: `results/logs/exp_*.log` (training progress)
- **Plots**: `results/figures/exp_*_predictions.png` (actual vs predicted)

### Key Parameters

**Horizon Mapping**:
- `--horizon 1` = 1 hour (1 × 15-min interval)
- `--horizon 96` = 1 day (96 × 15-min intervals)
- `--horizon 672` = 1 week (672 × 15-min intervals)

**Lookback Window**:
- `lookback = horizon × lookback_multiplier`
- Default: `lookback_multiplier=4.0` (e.g., 4 hours to predict 1 hour)

**Feature Modes** (`scripts/experiment_utils.py::select_features_by_mode()`):
- `full`: All 6 load features + 14 temporal features
- `no_time`: Only 6 load features (HUFL, HULL, MUFL, MULL, LUFL, LULL)
- `time_only`: Only 14 temporal features
- `loads_6_only`: Alias for `no_time`

## Important Findings & Best Practices

### Data Split Strategy is Critical
**ALWAYS use `--split-method chronological` for realistic evaluation.**
- Random sliding window splits cause ~5× performance overestimation due to data leakage
- Example: TX1 RandomForest shows R²=0.93 (random) vs R²=-4.36 (chronological)
- See `results/figures/fig3_data_split.png` for visualization

### Temporal Features are Essential
- Average R² improvement: +0.51 across all models when including time features
- RandomForest most affected: R²=0.69 (with time) vs R²=-0.04 (without time)
- Never use `--feature-mode no_time` in production

### TX1 (Industrial) vs TX2 (Residential)
- **TX1**: High load volatility, LinearRegression fails (R²=-6.95), requires deep learning
- **TX2**: Stable residential loads, LinearRegression works (R²=0.72)
- TX1 is 7.67 R² points harder due to 2-3× higher volatility

### Model Selection by Use Case
| Scenario | Model | Rationale |
|----------|-------|-----------|
| TX1, 1-hour | Informer | Only viable option (R²=0.97, RMSE=0.56°C) |
| TX1, long-term | N/A | All models fail, needs external weather features |
| TX2, 1-hour (accuracy) | Informer | Best performance (R²=0.97, RMSE=0.78°C) |
| TX2, 1-hour (practical) | RandomForest | Fast, simple deployment (R²=0.69) |
| TX2, 1-day/1-week | LinearRegression | Simplest, best performance (R²=0.73/0.69) |

## Directory Structure Highlights

```
├── data/
│   ├── raw/                    # Original trans_1.csv, trans_2.csv
│   └── processed/              # Cleaned data with various outlier removal strategies
├── scripts/
│   ├── common.py               # Core utilities: data loading, feature engineering
│   ├── train_configurable.py  # Main training script (CLI)
│   ├── experiment_utils.py     # Data splits, feature selection, window config
│   └── run_experiments.py      # Batch experiment runner
├── models/
│   ├── pytorch_mlp.py          # MLP with GPU support
│   ├── pytorch_rnn.py          # RNN/LSTM with GPU support
│   ├── pytorch_informer.py     # Informer wrapper
│   ├── informer_arch/          # Informer architecture (attn, encoder, decoder, etc.)
│   └── experiments/            # Saved models and configs (exp_*.json, exp_*.pth)
├── results/
│   ├── experiments/            # Experiment metrics (exp_*.csv)
│   ├── figures/                # Visualizations (11 report figures + experiment plots)
│   ├── logs/                   # Training logs
│   └── tables/                 # Prediction CSVs
├── configs/
│   └── experiment_plan.csv     # 111 experiment configurations
└── MSI5001_Model_Training_Workflow.ipynb  # Main deliverable notebook
```

## Reproducibility

All random processes use `random_state=42`:
```python
# In scripts and models
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
```

To reproduce key results:
```bash
# Baseline comparison (exp_106-111)
for tx in 1 2; do
    for horizon in 1 96 672; do
        python -m scripts.train_configurable \
            --tx-id $tx \
            --model LinearRegression \
            --split-method chronological \
            --feature-mode full \
            --horizon $horizon
    done
done
```

## Technical Notes

1. **Informer Availability**: Check `INFORMER_AVAILABLE` flag in `train_configurable.py:53`. If import fails, Informer will be unavailable.

2. **GPU Acceleration**: All PyTorch models auto-detect GPU (CUDA/MPS/CPU). Training time for Informer on 1-week horizon: ~30-40 minutes on CPU, ~5-10 minutes on GPU.

3. **Memory Constraints**: 1-week horizon with lookback=4× creates large sliding windows. If OOM errors occur, reduce `--lookback-multiplier` or use smaller batch sizes.

4. **Data Paths**: All paths use `PROJECT_ROOT = Path(__file__).resolve().parents[1]` for portability. Never hardcode absolute paths.

5. **Model Checkpointing**: PyTorch models save best checkpoint during training. Final model loaded from best validation loss epoch.

## Documentation

- **Main Report**: `docs/project_report_v3.1_submit_version.md` (Chinese, final submission)
- **Workflow Notebook**: `MSI5001_Model_Training_Workflow.ipynb` (English, comprehensive)
- **Guides**: `docs/guides/EXPERIMENT_GUIDE.md`, `RNN_MODEL_GUIDE.md`, `PYTORCH_MIGRATION.md`
