# Transformer Oil Temperature Forecasting – Project Report

## 1 Introduction
We develop a transparent pipeline for forecasting transformer oil temperature on the MSI5001 dataset. TX1 and TX2 share identical sensors, yet their thermal behaviours diverge. The study aims to:
(i) audit data quality and transformer-specific statistics;  
(ii) identify dominant predictors—particularly the reactive loads HULL/MULL and calendar features;  
(iii) design cleaning/standardisation without temporal leakage;  
(iv) benchmark Linear Regression, Ridge, RandomForest, and MLP under chronological vs. shuffled sliding-window splits;  
(v) extend the analysis to 1-hour, 1-day, and 1-week forecasting tasks.  
All artefacts reside in `MSI5001-final/`; figures and tables are referenced by filename.

## 2 Methodology
### 2.1 Data Understanding
- Raw files: `trans_1.csv`, `trans_2.csv`; 15-minute sampling (Jul 2018–Jun 2020).  
- QA (`missing_values_summary.csv`, `data_types.csv`) confirms no missing values.  
- Transformer summary (`transformer_summary.csv`, `ot_trend_by_transformer.png`) shows TX1 is cooler (mean 13.3 °C, σ 8.6 °C) whereas TX2 is warmer (mean 26.6 °C, σ 11.9 °C); modelling is performed per transformer.

### 2.2 Feature Engineering (Temporal Highlight)
`add_time_features` augments each sample with:
- calendar terms: hour, day-of-week, month, day-of-year;  
- periodic encodings: sine/cosine forms for hour/month/day-of-year to preserve cyclic structure;  
- categorical flags: `is_weekend`, `is_worktime`, `season`.  
These descriptors capture diurnal and seasonal rhythms and proved essential for stable performance.

Correlation heatmaps (`tx{1,2}_correlation_heatmap.png`) confirm HULL/MULL dominate predictive power. Lag matrices (`tx{1,2}_lag_correlation_heatmap.png`) indicate TX1 benefits from 8–12 h history, while TX2 reacts almost immediately. Accordingly we set:
- TX1: lookback 48 samples (≈12 h), 1-step (15 min) horizon; plus dynamic features `HULL_diff1`, `MULL_diff1`, `HULL_roll12`, `MULL_roll12`.  
- TX2: lookback 24 samples (≈6 h), 1-step horizon.

### 2.3 Cleaning and Standardisation
- Outlier detection (HULL/MULL/OT) uses IQR and a 6-hour rolling Z-score; a sample is removed if any criterion or the physical limit (OT outside [−20, 120] °C) triggers. Low-voltage loads may be negative (reverse flow) and are retained. Clean datasets: TX1 65,562 samples; TX2 66,919 (`processed/tx{1,2}_cleaned.csv`).  
- Expanding z-score standardisation avoids leakage; outputs in `tx{1,2}_standardized.csv`, parameters logged in `standardization_params.json`.

### 2.4 Modelling Pipelines
1. **Chronological split** (`scripts/model_training.py`): train on earliest 80 %, test on latest 20 %. Models: RandomForest (120 estimators, max depth 12, min samples leaf 5) and MLP (128–64 units, early stopping). Artefacts: `models/baseline/`, `model_performance_all.csv`, `tx{1,2}_{Model}[_std]_prediction.png`.  
2. **Random sliding-window split** (`scripts/model_random_split.py`): sample ≤30k windows, shuffle 80/20, train LinearRegression, Ridge (α=5), RandomForest, MLP (TX1 includes dynamic features). Artefacts: `models/random_split/`, `random_split_performance.csv`, `random_tx{1,2}_{Model}_scatter.png`.  
3. **Multi-horizon experiments** (`scripts/model_horizon_experiments.py`): resample hourly and create windows for 1h/1d/1w forecasts with gap constraints (24/48/168-hour lookbacks; gaps 0/6/24 hours; horizons 1/24/168 hours). StandardScaler is fitted on training windows only.

## 3 Results
### 3.1 Chronological 80/20 Split
| Transformer | Model        | R²    | RMSE (°C) | MAE (°C) |
|-------------|--------------|-------|-----------|----------|
| TX1         | RandomForest | −4.61 | 8.03      | 7.03     |
| TX1         | MLP          | −4.75 | 8.13      | 7.08     |
| TX2         | RandomForest | **0.63** | 6.45      | 4.97     |
| TX2         | MLP          | 0.59  | 6.77      | 5.41     |

TX2 attains reasonable accuracy; TX1 remains challenging because the hold-out interval exhibits low-variance drift, requiring richer dynamics.

### 3.2 Random Sliding-Window Split
| Transformer | Model            | R²    | RMSE (°C) | MAE (°C) |
|-------------|------------------|-------|-----------|----------|
| TX1         | LinearRegression | −0.81 | 10.19     | 4.36     |
| TX1         | Ridge            | −0.31 | 8.69      | 4.21     |
| TX1         | RandomForest     | **0.94** | 1.89      | 1.30     |
| TX1         | MLP              | 0.85  | 2.96      | 2.23     |
| TX2         | LinearRegression | −2.62 | 22.85     | 4.24     |
| TX2         | Ridge            | 0.43  | 9.04      | 3.95     |
| TX2         | RandomForest     | 0.97  | 2.23      | 1.63     |
| TX2         | MLP              | 0.91  | 3.57      | 2.75     |

Random shuffling showcases model capacity but inflates deployable performance since training/test windows share temporal regimes.

### 3.3 Multi-Horizon (Random Split, Hourly Windows)
| Transformer | Horizon | Best Model   | R²    | RMSE (°C) | MAE (°C) |
|-------------|---------|--------------|-------|-----------|----------|
| TX1         | 1h      | RandomForest | 0.94  | 1.80      | 1.23     |
| TX1         | 1d      | MLP          | 0.95  | 1.73      | 1.28     |
| TX1         | 1w      | RandomForest | 0.97  | 1.32      | 0.93     |
| TX2         | 1h      | MLP          | 0.97  | 2.00      | 1.43     |
| TX2         | 1d      | MLP          | 0.98  | 1.54      | 1.11     |
| TX2         | 1w      | MLP          | 0.98  | 1.69      | 1.24     |

These metrics (`horizon_experiment_metrics.csv`) represent optimistic upper bounds. Chronological or rolling evaluation should confirm robustness before deployment.

## 4 Discussion
\begin{itemize}
    \item \textbf{Temporal encodings are central}: calendar and sine/cosine features, together with weekend/worktime flags, capture diurnal and seasonal structure; they were indispensable in every experiment.
    \item \textbf{Window + dynamics}: TX1 needs a 12-hour lookback and dynamic terms, while TX2 performs well with 6 hours. The problem statement’s gaps (6 h for 1-day horizon, 24 h for 1-week) were honoured in the multi-horizon experiments.
    \item \textbf{Evaluation strategy}: chronological splits reflect real deployments; random/short-horizon results are useful for capacity checks but must be complemented with rolling-window evaluation in practice.
    \item \textbf{Linear baselines}: Linear/Ridge models underperform, underscoring the importance of non-linear relationships captured by RandomForest/MLP.
\end{itemize}

## 5 Conclusion
The pipeline provides transformer-specific cleaned datasets, validates HULL/MULL + temporal descriptors (with dynamic features for TX1), and clarifies the disparity between chronological and random evaluations. TX2 approaches deployment readiness; TX1 benefits from dynamic features but still requires richer context (e.g., environment, maintenance logs) and rolling validation to ensure stability.

## Appendix
- Data QA: `missing_values_summary.csv`, `data_types.csv`  
- Transformer statistics & trends: `transformer_summary.csv`, `ot_trend_by_transformer.png`  
- Correlation/lag plots: `tx{1,2}_correlation_heatmap.png`, `tx{1,2}_lag_correlation_heatmap.png`  
- Clean/standardised datasets: `processed/tx{1,2}_cleaned.csv`, `tx{1,2}_standardized.csv`  
- Chronological models: `models/baseline/`, `model_performance_all.csv`, `tx{1,2}_{Model}[_std]_prediction.png`  
- Random and horizon models: `models/random_split/`, `models/horizon_experiments/`, `random_split_performance.csv`, `horizon_experiment_metrics.csv`

## References
1. MSI5001 Course Dataset (internal).  
2. Pedregosa, F. et al. “Scikit-learn: Machine Learning in Python,” \textit{Journal of Machine Learning Research}, 12 (2011).  
3. Bishop, C. M., \textit{Pattern Recognition and Machine Learning}, Springer, 2006.
