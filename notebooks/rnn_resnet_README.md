# RNN-ResNet Hybrid Model for Transformer Oil Temperature Prediction

This directory contains the implementation of a **hybrid deep learning architecture** combining **Recurrent Neural Networks (RNN)** and **Residual Networks (ResNet)** for transformer oil temperature prediction.

## 🎯 Model Overview

### Architecture: RNN → ResNet Pipeline

```
Time Series Input
       ↓
┌──────────────────┐
│  RNN Layer       │  ← Captures temporal dependencies
│  (LSTM/GRU)      │     Processes sequences of electrical load data
└──────────────────┘
       ↓
┌──────────────────┐
│  Projection      │  ← Maps RNN output to ResNet dimension
│  Layer           │
└──────────────────┘
       ↓
┌──────────────────┐
│  ResNet Block 1  │  ← Deep feature learning with skip connections
│  (with skip)     │
└──────────────────┘
       ↓
┌──────────────────┐
│  ResNet Block 2  │
│  (with skip)     │
└──────────────────┘
       ↓
┌──────────────────┐
│  Output Layer    │  ← Oil temperature prediction
└──────────────────┘
```

### Two Model Variants

1. **LSTM-ResNet**:
   - Uses LSTM (Long Short-Term Memory) for RNN layer
   - Better at capturing long-term dependencies
   - More parameters (~400K)
   - Slightly slower training

2. **GRU-ResNet**:
   - Uses GRU (Gated Recurrent Unit) for RNN layer
   - Faster training, fewer parameters (~300K)
   - Good performance with less complexity
   - Recommended for initial experiments

## 📊 Key Features

- **Bidirectional RNN**: Processes sequences in both forward and backward directions
- **Residual Connections**: Enables deeper networks without vanishing gradients
- **Batch Normalization**: Stabilizes training
- **Dropout Regularization**: Prevents overfitting
- **Early Stopping**: Automatically stops when validation loss plateaus
- **CUDA Support**: GPU-accelerated training on Windows

## 📁 Data Requirements

The notebook expects data in the `dataset/` folder:

```
project_root/
├── dataset/
│   ├── trans_1.csv  # Transformer 1 data
│   ├── trans_2.csv  # Transformer 2 data
│   └── README.md
└── notebooks/
    ├── rnn_resnet_oil_temperature_prediction.ipynb
    └── rnn_resnet_README.md (this file)
```

**Data Format**:
- CSV files with columns: `HUFL`, `HULL`, `MUFL`, `MULL`, `LUFL`, `LULL`, `OT`
- `OT` (Oil Temperature) is the target variable
- Other columns are electrical load features
- 15-minute time intervals

## 🚀 Quick Start

### Prerequisites

1. **Python 3.8+** installed
2. **NVIDIA GPU with CUDA** (optional but recommended)
3. **PyTorch with CUDA support**

### Installation

```bash
# 1. Navigate to project directory
cd MSI5001-power-transformer-oil-temperature-prediction

# 2. Ensure you're on the correct branch
git checkout feature/rnn-resnet

# 3. Install PyTorch with CUDA 11.8 (Windows)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CPU only (slower):
# pip install torch torchvision

# 4. Install other dependencies
pip install numpy pandas scikit-learn matplotlib seaborn jupyter notebook
```

### Running the Notebook

```bash
# 1. Launch Jupyter Notebook
jupyter notebook

# 2. Navigate to notebooks/
# 3. Open: rnn_resnet_oil_temperature_prediction.ipynb
# 4. Run cells sequentially or use Cell → Run All
```

## 🔧 Model Configuration

Default hyperparameters (can be modified in notebook):

```python
{
    'seq_length': 10,           # Look back 10 time steps (2.5 hours)
    'rnn_hidden_dim': 64,       # RNN hidden layer size
    'rnn_num_layers': 2,        # Number of RNN layers
    'bidirectional': True,      # Use bidirectional RNN
    'resnet_hidden_dim': 128,   # ResNet hidden layer size
    'resnet_num_blocks': 2,     # Number of ResNet blocks
    'dropout': 0.3,             # Dropout rate
    'learning_rate': 0.001,     # Adam optimizer learning rate
    'batch_size': 128,          # Training batch size
    'epochs': 100,              # Maximum epochs
    'patience': 20              # Early stopping patience
}
```

## 📈 Expected Performance

Based on the hybrid architecture design:

### 1-Hour Prediction

| Model | Expected R² | Expected RMSE | Training Time (GPU) |
|-------|-------------|---------------|---------------------|
| Random Forest | 0.60 | 4.68°C | 1 min |
| Pure ResNet | 0.55-0.65 | 4.5-5.5°C | 10-15 min |
| **LSTM-ResNet** | **0.62-0.72** | **4.0-5.0°C** | 20-30 min |
| **GRU-ResNet** | **0.60-0.70** | **4.2-5.2°C** | 15-25 min |

### 1-Day Prediction

| Model | Expected R² | Expected RMSE |
|-------|-------------|---------------|
| **LSTM-ResNet** | 0.40-0.55 | 5.0-6.0°C |
| **GRU-ResNet** | 0.38-0.53 | 5.2-6.2°C |

### 1-Week Prediction

| Model | Expected R² | Expected RMSE |
|-------|-------------|---------------|
| **LSTM-ResNet** | 0.25-0.40 | 5.5-6.5°C |
| **GRU-ResNet** | 0.23-0.38 | 5.7-6.7°C |

## 💾 Output Files

After training, the following files will be generated:

### Models
```
models/rnn_resnet/
├── lstm_resnet_1h.pth      # LSTM-ResNet for 1-hour prediction
├── lstm_resnet_1d.pth      # LSTM-ResNet for 1-day prediction
├── lstm_resnet_1w.pth      # LSTM-ResNet for 1-week prediction
├── gru_resnet_1h.pth       # GRU-ResNet for 1-hour prediction
├── gru_resnet_1d.pth       # GRU-ResNet for 1-day prediction
└── gru_resnet_1w.pth       # GRU-ResNet for 1-week prediction
```

### Visualizations
```
visualizations/rnn_resnet/
├── training_history.png                # Training/validation loss curves
├── lstm_vs_gru_comparison.png         # LSTM vs GRU performance
├── all_models_comparison.png          # All models comparison
├── predictions_lstm_resnet.png        # LSTM-ResNet predictions
├── predictions_gru_resnet.png         # GRU-ResNet predictions
└── error_analysis.png                 # Error distribution analysis
```

### Results
```
results/
├── rnn_resnet_comparison.csv          # Performance metrics (CSV)
└── rnn_resnet_final_results.pkl       # Complete results (pickle)
```

## 🔍 Notebook Structure

The notebook contains 16 sections:

1. **Environment Setup**: Import libraries, check CUDA
2. **Device Configuration**: GPU/CPU setup
3. **Data Loading**: Load from dataset/ folder
4. **Data Preprocessing**: Create time series sequences
5. **LSTM-ResNet Architecture**: Define LSTM-based model
6. **GRU-ResNet Architecture**: Define GRU-based model
7. **Training Utilities**: Helper functions
8. **Data Loaders**: PyTorch DataLoader creation
9. **LSTM-ResNet Training**: Train for 1h, 1d, 1w
10. **GRU-ResNet Training**: Train for 1h, 1d, 1w
11. **Training History**: Visualize loss curves
12. **Performance Comparison**: Compare all models
13. **Predictions Visualization**: Actual vs predicted
14. **Error Analysis**: Analyze prediction errors
15. **Model Comparison Summary**: Final comparison table
16. **Save Results**: Save models and results

## ⚙️ Troubleshooting

### Issue: CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solutions**:
```python
# Reduce batch size
batch_size = 64  # or even 32

# Reduce sequence length
seq_length = 5  # instead of 10

# Reduce model size
rnn_hidden_dim = 32  # instead of 64
resnet_hidden_dim = 64  # instead of 128
```

### Issue: Training Too Slow

**Solutions**:
1. Use GRU instead of LSTM (20-30% faster)
2. Reduce number of epochs
3. Enable CUDA if not already
4. Reduce validation frequency

### Issue: Poor Performance

**Solutions**:
1. Increase sequence length (try 15 or 20)
2. Add more RNN layers (try 3)
3. Increase hidden dimensions
4. Reduce dropout rate
5. Train for more epochs

### Issue: Data Not Found

**Error**: `FileNotFoundError: Data file not found`

**Solution**:
```bash
# Ensure data is in correct location
ls dataset/  # Should show trans_1.csv and trans_2.csv

# If not, copy data files:
mkdir -p dataset
cp /path/to/trans_1.csv dataset/
cp /path/to/trans_2.csv dataset/
```

## 🎓 Why RNN-ResNet Hybrid?

### Advantages of Hybrid Architecture

1. **Temporal Feature Extraction (RNN)**:
   - LSTM/GRU captures time-series patterns
   - Learns dependencies across time steps
   - Handles variable-length sequences

2. **Deep Feature Learning (ResNet)**:
   - ResNet blocks enable deeper networks
   - Skip connections prevent vanishing gradients
   - Non-linear transformations capture complex patterns

3. **Best of Both Worlds**:
   - RNN for sequential patterns
   - ResNet for deep representations
   - Better than either architecture alone

### When to Use Each Variant

**Use LSTM-ResNet when**:
- Long-term dependencies are important
- You have sufficient GPU memory
- Training time is not a constraint
- You want maximum accuracy

**Use GRU-ResNet when**:
- Faster training is needed
- GPU memory is limited
- Good performance with less complexity is acceptable
- You're experimenting with hyperparameters

## 📚 References

### Academic Papers

1. **LSTM**: Hochreiter & Schmidhuber (1997) - Long Short-Term Memory
2. **GRU**: Cho et al. (2014) - Learning Phrase Representations using RNN Encoder-Decoder
3. **ResNet**: He et al. (2015) - Deep Residual Learning for Image Recognition

### PyTorch Documentation

- [LSTM Layer](https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html)
- [GRU Layer](https://pytorch.org/docs/stable/generated/torch.nn.GRU.html)
- [Time Series with PyTorch](https://pytorch.org/tutorials/beginner/timeseries_tutorial.html)

## 🤝 Tips for Best Results

1. **Start with GRU-ResNet**: Faster iteration for hyperparameter tuning
2. **Tune sequence length first**: Has largest impact on performance
3. **Monitor overfitting**: Watch train vs validation loss
4. **Use early stopping**: Saves time and prevents overfitting
5. **Try ensemble**: Average LSTM-ResNet and GRU-ResNet predictions

## 💡 Advanced Experiments

Want to improve performance further? Try:

1. **Attention Mechanisms**: Add attention layer after RNN
2. **Deeper Networks**: Increase ResNet blocks to 3-4
3. **Feature Engineering**: Add time-based features (hour, day, season)
4. **Ensemble Methods**: Combine with Random Forest predictions
5. **Multi-horizon Prediction**: Predict multiple time steps simultaneously

---

**Created**: October 2025
**Framework**: PyTorch 2.0+
**Compatible with**: Windows (CUDA), Linux, macOS (CPU)
