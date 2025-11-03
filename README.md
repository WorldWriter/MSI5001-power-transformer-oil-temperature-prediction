# Power Transformer Oil Temperature Prediction

**Multi-Horizon Time-Series Forecasting with Deep Learning**

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Course**: MSI5001 - Machine Learning in Practice
> **Institution**: National University of Singapore
> **Focus**: Rigorous time-series forecasting methodology with data leakage prevention

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Findings](#key-findings)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Main Deliverable](#main-deliverable)
- [Model Comparison](#model-comparison)
- [Usage Guide](#usage-guide)
- [Reproducibility](#reproducibility)
- [Documentation](#documentation)
- [Citation](#citation)

---

## 🎯 Overview

This project develops and evaluates machine learning models for predicting power transformer oil temperature (OT) across **three time horizons**: 1 hour, 1 day, and 1 week. Accurate oil temperature forecasting enables:

- **Preventive Maintenance**: Early detection of thermal anomalies
- **Load Management**: Optimizing transformer capacity planning
- **Failure Prevention**: Reducing equipment degradation risk

### Dataset

| Transformer | Type | Period | Samples | Interval |
|------------|------|--------|---------|----------|
| **TX1** | Industrial | 2018-07-01 - 2020-06-26 | 69,680 | 15 min |
| **TX2** | Residential | 2018-07-01 - 2020-06-26 | 69,680 | 15 min |

**Features**: 6 electrical load measurements (HUFL, HULL, MUFL, MULL, LUFL, LULL)
**Target**: Oil Temperature (°C)

### Models Evaluated

1. **LinearRegression** - Baseline model
2. **RandomForest** - Ensemble learning
3. **MLP** - Multi-layer perceptron (PyTorch)
4. **RNN** - Recurrent neural network (LSTM)
5. **Informer** - State-of-the-art transformer for long-sequence forecasting

---

## 🔍 Key Findings

### 1. Informer Dominates Short-Term Prediction
- **1-hour forecasting**: R² > 0.97, RMSE < 0.8°C on both transformers
- **Best performance**: TX1 (R²=0.9735, RMSE=0.56°C)
- **10x better** than traditional ML models on industrial dataset

### 2. Temporal Features Are Critical
- **Average R² improvement**: +0.51 across all models
- **RandomForest most affected**: R² drops from 0.69 to -0.04 without time features
- **Impact**: Removing time features causes 0.5-0.7 R² degradation
- **Recommendation**: Always include cyclical time encoding (hour, day, month)

### 3. TX1 (Industrial) vs TX2 (Residential) Difficulty Gap
- **TX1**: LinearRegression completely fails (R² = -6.95)
- **TX2**: LinearRegression works well (R² = 0.72)
- **Difference**: 7.67 R² points difficulty gap
- **Cause**: TX1 exhibits 2-3× higher load volatility (industrial batch processes)

### 4. Data Split Strategy Critically Matters
- **Sliding window random split**: R² = 0.93 (misleading)
- **Chronological split**: R² = -4.36 (true generalization)
- **Performance inflation**: **5.29× overestimation** due to data leakage
- **Conclusion**: Only chronological split reveals real-world forecasting capability

---

## 📁 Project Structure

```
MSI5001-power-transformer-oil-temperature-prediction/
│
├── MSI5001_Model_Training_Workflow.ipynb  # Main deliverable (course evaluation)
├── README.md                               # This file
├── requirements.txt                        # Python dependencies
│
├── data/
│   ├── raw/                                # Original CSV files (trans_1.csv, trans_2.csv)
│   └── processed/                          # Cleaned & standardized data
│
├── results/
│   ├── experiments/                        # Experiment metrics (exp_*.csv)
│   ├── figures/                            # All visualizations (11 report figures)
│   ├── logs/                               # Training logs
│   └── tables/                             # Prediction CSVs & summary tables
│
├── configs/
│   └── experiment_plan.csv                 # 111 experiment configurations
│
├── scripts/
│   ├── train_configurable.py               # Main training script (CLI)
│   ├── common.py                           # Data loading & feature engineering
│   ├── experiment_utils.py                 # Experiment helpers
│   ├── generate_report_figures.py          # Generate all report figures
│   └── ...                                 # Additional utilities
│
├── models/
│   ├── pytorch_mlp.py                      # MLP implementation
│   ├── pytorch_rnn.py                      # RNN/LSTM implementation
│   ├── pytorch_informer.py                 # Informer wrapper
│   ├── informer_arch/                      # Informer architecture modules
│   └── experiments/                        # Saved trained models
│
├── docs/
│   ├── guides/                             # Markdown documentation
│   └── *.md                                # Project reports & analysis
│
├── notebooks/                              # Exploratory notebooks
├── tests/                                  # Test files
└── external/                               # Third-party implementations (Informer2020)
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd MSI5001-power-transformer-oil-temperature-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Verify Installation

```bash
# Generate all report figures (~30 seconds)
python -m scripts.generate_report_figures

# Run a quick test experiment
python -m scripts.train_configurable \
    --tx-id 2 \
    --model RandomForest \
    --split-method chronological \
    --feature-mode full \
    --horizon 1
```

---

## 📊 Main Deliverable

### MSI5001_Model_Training_Workflow.ipynb

**Comprehensive Jupyter notebook demonstrating the complete ML workflow**

**Contents**:
- **Part 1: Methodology** (5 sections)
  - Data pipeline & preprocessing
  - Feature engineering (temporal + domain-specific)
  - Model selection rationale
  - Data split strategy (preventing leakage)
  - Baseline comparison

- **Part 2: Results & Analysis** (3 sections)
  - Multi-model comparison across 3 horizons
  - Feature ablation studies
  - Final recommendations & reproducibility

**Grading Criteria Satisfied**:
1. ✅ **Clean Data Pipeline**: Systematic preprocessing with feature engineering
2. ✅ **Appropriate Model Choice**: 5 models from traditional ML to SOTA deep learning
3. ✅ **Baseline Comparison**: LinearRegression establishes clear baseline
4. ✅ **Reproducibility**: Documented parameters, experiment IDs, random seeds
5. ✅ **Creativity**: TX1-specific features, data split analysis, multi-horizon insights

**How to Run**:
```bash
jupyter notebook MSI5001_Model_Training_Workflow.ipynb
```

Set `FORCE_RETRAIN = True` to retrain all models from scratch (default: load cached results).

---

## 📈 Model Comparison

### Performance Summary (Chronological Split)

| Horizon | Transformer | LinearRegression | RandomForest | MLP | RNN | **Informer** |
|---------|------------|-----------------|-------------|-----|-----|--------------|
| **1 hour** | TX1 | R²=-6.95<br>RMSE=9.72 | R²=-4.36<br>RMSE=7.98 | R²=-3.81<br>RMSE=7.56 | R²=-4.30<br>RMSE=7.94 | **R²=0.97**<br>**RMSE=0.56** |
| **1 hour** | TX2 | R²=0.72<br>RMSE=5.68 | R²=0.69<br>RMSE=5.95 | R²=0.42<br>RMSE=8.15 | R²=0.42<br>RMSE=8.13 | **R²=0.97**<br>**RMSE=0.78** |
| **1 day** | TX1 | R²=-6.27<br>RMSE=9.30 | R²=-3.78<br>RMSE=7.54 | R²=-5.12<br>RMSE=8.53 | R²=-4.15<br>RMSE=7.82 | **R²=0.62**<br>**RMSE=2.14** |
| **1 day** | TX2 | **R²=0.73**<br>**RMSE=5.59** | R²=0.64<br>RMSE=6.42 | R²=0.38<br>RMSE=8.42 | R²=0.45<br>RMSE=7.93 | R²=0.61<br>RMSE=6.68 |
| **1 week** | TX1 | R²=-5.89<br>RMSE=9.05 | R²=-2.94<br>RMSE=6.84 | R²=-4.56<br>RMSE=8.13 | R²=-3.21<br>RMSE=7.08 | R²=-1.59<br>RMSE=5.54 |
| **1 week** | TX2 | **R²=0.69**<br>**RMSE=5.89** | R²=0.58<br>RMSE=6.89 | R²=0.32<br>RMSE=8.78 | R²=0.68<br>RMSE=6.02 | R²=0.65<br>RMSE=6.31 |

### Recommendations by Use Case

| Scenario | Recommended Model | R² | RMSE (°C) | Rationale |
|----------|-------------------|-----|-----------|-----------|
| **TX1, 1-hour monitoring** | Informer | 0.97 | 0.56 | Only viable option |
| **TX1, long-term** | None | N/A | N/A | All models fail, requires external features |
| **TX2, 1-hour (accuracy)** | Informer | 0.97 | 0.78 | Best performance |
| **TX2, 1-hour (practical)** | RandomForest | 0.69 | 5.95 | Fast, simple deployment |
| **TX2, 1-day** | LinearRegression | 0.73 | 5.59 | Simplest, best performance |
| **TX2, 1-week** | LinearRegression | 0.69 | 5.89 | Most stable long-term |
| **Production (general)** | RandomForest | 0.64-0.69 | ~6.0 | Consistent, reliable |

---

## 🛠 Usage Guide

### Training a Single Model

```bash
python -m scripts.train_configurable \
    --tx-id 2 \
    --model Informer \
    --split-method chronological \
    --feature-mode full \
    --horizon 1 \
    --lookback-multiplier 4.0
```

**Parameters**:
- `--tx-id`: Transformer ID (1=Industrial, 2=Residential)
- `--model`: Choose from {LinearRegression, RandomForest, MLP, RNN, Informer}
- `--split-method`: Choose from {chronological, random_window, group_random}
- `--feature-mode`: Choose from {full, no_time, time_only, loads_6_only}
- `--horizon`: Prediction steps (1=1h, 96=1d, 672=1w)
- `--lookback-multiplier`: Lookback = horizon × multiplier (default: 4.0)

### Generating Report Figures

```bash
# Generate all 11 figures for the report
python -m scripts.generate_report_figures

# Output: results/figures/*.png
```

### Feature Ablation Study

```bash
# With time features (full)
python -m scripts.train_configurable --feature-mode full ...

# Without time features
python -m scripts.train_configurable --feature-mode no_time ...

# Time features only
python -m scripts.train_configurable --feature-mode time_only ...
```

---

## 🔬 Reproducibility

### Experiment Tracking

All experiments are tracked with:
- **Unique IDs**: exp_001 through exp_111
- **Configuration files**: `models/experiments/exp_*.json`
- **Metrics**: `results/tables/exp_*_predictions.csv`
- **Visualizations**: `results/figures/exp_*_predictions.png`

### Random Seed Management

All random processes use **seed=42**:
```python
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)
```

### Reproducing Key Results

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

# Feature ablation study (exp_097-105)
python -m scripts.train_configurable \
    --tx-id 2 \
    --model RandomForest \
    --split-method chronological \
    --feature-mode no_time \
    --horizon 1
```

---

## 📚 Documentation

### Main Reports

- **`docs/project_report_v3.1_submit_version.md`** - Final submission report (Chinese)
- **`MSI5001_Model_Training_Workflow.ipynb`** - Complete workflow notebook (English)

### Technical Guides

Located in `docs/guides/`:
- `EXPERIMENT_GUIDE.md` - Comprehensive experiment documentation
- `EXPERIMENT_QUICKSTART.md` - Quick start guide
- `RNN_MODEL_GUIDE.md` - RNN implementation details
- `PYTORCH_MIGRATION.md` - PyTorch migration notes

### Key Findings Documents

- **Data Leakage Analysis**: `docs/guides/` (Data split comparison)
- **Feature Engineering**: `MSI5001_Model_Training_Workflow.ipynb` Section 7
- **Model Selection**: `MSI5001_Model_Training_Workflow.ipynb` Section 3

---

## 📖 Citation

If you use this project or methodology in your research, please cite:

```bibtex
@misc{transformer_oil_temp_2025,
  title={Power Transformer Oil Temperature Prediction: Multi-Horizon Time-Series Forecasting},
  author={MSI5001 Group Project},
  year={2025},
  institution={National University of Singapore},
  note={Course: MSI5001 - Machine Learning in Practice}
}
```

---

## 🔗 Related Work

This project builds upon:
- **Informer**: Zhou et al. (2021) - "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting" (AAAI 2021)
- **Dataset**: ETT (Electricity Transformer Temperature) benchmark adaptation

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **Course**: MSI5001 - Machine Learning in Practice, National University of Singapore
- **Dataset**: Power transformer monitoring data (anonymized)
- **Frameworks**: PyTorch, scikit-learn, pandas, matplotlib

---

## 📞 Contact

For questions or collaboration:
- **Course**: MSI5001, National University of Singapore
- **Project**: Power Transformer Oil Temperature Prediction

---

**Last Updated**: 2025-01-03
**Status**: ✅ Complete & Reproducible
