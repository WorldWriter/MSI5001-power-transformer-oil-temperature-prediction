# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an MSI5001 course project for predicting power transformer oil temperature (OT) using time-series machine learning. The project explores both traditional ML (linear regression) and deep learning approaches (RNN, Informer) on the Electric Transformer Temperature (ETT) dataset.

**Key Challenge**: Predict oil temperature at a future time point using only historical load data (HUFL, HULL, MUFL, MULL, LUFL, LULL) without using any oil temperature values from any time point.

## Dataset Structure

Located in `dataset/`:
- `train1.csv`, `train2.csv`: Two transformer datasets with hourly measurements
- Each row contains 6 load features + 1 target (OT - Oil Temperature)
- Features represent active/reactive power at high/medium/low voltage levels

**Critical Constraints**:
- Cannot use OT (oil temperature) from ANY time point as input
- Cannot use load information from the target prediction time point
- Three prediction horizons to test:
  - 1-hour: Start from 4 time points prior
  - 1-day: Start from 96 time points prior
  - 1-week: Start from 672 time points prior

## Environment Setup

```bash
# Create virtual environment
python -m venv .venv && source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install base dependencies
pip install -r requirements.txt

# For PyTorch/deep learning models (RTX 4070 Ti Super with CUDA 12.1)
pip install -r requirements-torch.txt
# Or manually: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Verify CUDA (optional)
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## Running Notebooks

The primary development workflow uses Jupyter notebooks in `notebooks/`:

```bash
# Start Jupyter
jupyter notebook

# Or use VS Code Jupyter extension
```

**Available Notebooks**:
- `linear_regression.ipynb`: PyTorch-based linear regression baseline
- `rnn.ipynb`: PyTorch RNN model for time-series prediction
- `informer.ipynb`: Informer transformer architecture experiments
- `MSI5001_A5_example.ipynb`: Course assignment reference

**Common Workflow in Notebooks**:
1. Load data from `dataset/train1.csv` or `dataset/train2.csv`
2. Visualize time-series features and correlations
3. Create sliding window sequences (configurable `seq_length`)
4. Split data temporally (not randomly) - 80/20 train/test recommended
5. Normalize features using sklearn StandardScaler
6. Train model (PyTorch DataLoader + training loop)
7. Evaluate using R² score, MSE, MAE
8. Generate prediction plots

## Data Preprocessing Pattern

All notebooks follow this sequence creation pattern:

```python
# Sliding window approach
def create_sequences(data, seq_length, pred_offset):
    """
    data: Feature matrix (no OT column)
    seq_length: Number of historical time steps to use
    pred_offset: Gap between last input and prediction target (4/96/672)
    """
    # Returns (X, y) where:
    # X: [num_samples, seq_length, num_features]
    # y: [num_samples] - OT at target time
```

**Critical**: Ensure train/test split is disjoint in time. If training uses time points 1-50,000, testing must use completely separate time ranges.

## Model Architecture Notes

### Linear Regression (`linear_regression.ipynb`)
- PyTorch implementation: `nn.Linear(seq_length * num_features, 1)`
- Flattens input sequences to fit standard regression
- Baseline model for comparison

### RNN (`rnn.ipynb`)
- Architecture: `nn.RNN` → `nn.Linear` head
- Handles variable-length sequences naturally
- Use `batch_first=True` for DataLoader compatibility

### Informer (`informer.ipynb`)
- Transformer-based architecture for long-sequence forecasting
- More complex hyperparameter tuning required

## Current Branch: feature/improve-rnn-performance

Recent commits focused on:
- Bug fixes in notebooks (4 key bugs fixed in commit 1736dd5)
- Added multi-variate linear regression
- Model training showing R² ≈ 0 (需要改进 / needs improvement)

## Evaluation Metrics

Primary metrics used across all notebooks:
- **R² (R-squared)**: Coefficient of determination - main comparison metric
- **MSE (Mean Squared Error)**: Loss function and evaluation
- **MAE (Mean Absolute Error)**: Interpretable error in temperature units

Target: Achieve R² > 0.7 on test set for acceptable performance.

## Git Workflow

- Main branch: `main`
- Feature branches follow pattern: `feature/improve-rnn-performance`
- Commit messages in Chinese or English with present tense
- Large data files in `dataset/` are not committed (see `.gitignore`)

## File Organization

```
.
├── dataset/              # Raw CSV data (train1.csv, train2.csv)
│   ├── dataset_readme.md # Task description and constraints
│   └── ETT_Dataset_Research_Report.md
├── notebooks/           # Jupyter experiments
│   ├── linear_regression.ipynb
│   ├── rnn.ipynb
│   ├── informer.ipynb
│   └── AGENTS.md       # AI agent guidelines (if using automation)
├── models/             # Saved model checkpoints (currently empty)
├── requirements.txt    # Base dependencies
└── requirements-torch.txt  # PyTorch with CUDA 12.1
```

## Known Issues & Debugging

1. **R² near 0**: Current models struggling - check:
   - Data normalization (both features and target)
   - Sequence length tuning
   - Temporal split correctness (no data leakage)
   - Learning rate and optimizer settings

2. **CUDA errors**: Ensure correct PyTorch build for CUDA 12.1:
   ```bash
   pip uninstall torch torchvision
   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
   ```

3. **Modified notebooks not running**: Check for:
   - Missing imports after editing
   - Cells executed out of order
   - Restart kernel and run all cells sequentially

## Cross-Validation Strategy

Per assignment requirements:
- Divide data into time point groups
- Randomly assign 80% groups to training, 20% to testing
- Use cross-validation on training set for model selection
- Test set reserved only for final model evaluation
- Apply mean normalization (StandardScaler recommended)

## Bilingual Notes

This is a Chinese-English bilingual project. Code comments and markdown cells in notebooks use both languages. Commit messages and documentation accept either language.
