# ResNet Model for Transformer Oil Temperature Prediction

This directory contains a Jupyter Notebook implementing a **Residual Neural Network (ResNet)** for predicting transformer oil temperature using PyTorch with CUDA support.

## 📋 Overview

The ResNet model is implemented as an alternative deep learning approach to compare against traditional machine learning models (Random Forest, Ridge Regression) and simple MLPs.

### Model Architecture
- **Framework**: PyTorch with GPU (CUDA) acceleration
- **Architecture**: Lightweight ResNet with 2-3 residual blocks
- **Features**:
  - Residual connections (skip connections) to enable deeper networks
  - Batch Normalization for training stability
  - Dropout for regularization
  - Early stopping to prevent overfitting

### Prediction Tasks
- **1-hour ahead**: Short-term prediction
- **1-day ahead**: Medium-term prediction
- **1-week ahead**: Long-term prediction

## 🖥️ Windows CUDA Setup

### Prerequisites

1. **NVIDIA GPU with CUDA support**
   - Check if you have an NVIDIA GPU: Open Device Manager → Display adapters
   - Verify CUDA compatibility: [NVIDIA CUDA GPUs](https://developer.nvidia.com/cuda-gpus)

2. **CUDA Toolkit**
   - Download and install: [CUDA Toolkit Downloads](https://developer.nvidia.com/cuda-downloads)
   - Recommended version: CUDA 11.8 or 12.1
   - Verify installation: Open Command Prompt and run `nvcc --version`

3. **Python 3.8+**
   - Download from [python.org](https://www.python.org/downloads/)
   - Make sure to check "Add Python to PATH" during installation

### Installation Steps

#### Step 1: Clone the Repository (if not done)
```bash
git clone https://github.com/YOUR_USERNAME/MSI5001-power-transformer-oil-temperature-prediction.git
cd MSI5001-power-transformer-oil-temperature-prediction
git checkout feature/resnet-model
```

#### Step 2: Create Virtual Environment (Recommended)
```bash
# Open Command Prompt or PowerShell
python -m venv venv_resnet

# Activate virtual environment
# On Windows Command Prompt:
venv_resnet\Scripts\activate.bat

# On Windows PowerShell:
venv_resnet\Scripts\Activate.ps1
```

#### Step 3: Install PyTorch with CUDA Support
```bash
# For CUDA 11.8 (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CPU only (if no GPU available)
pip install torch torchvision torchaudio
```

#### Step 4: Install Other Dependencies
```bash
pip install -r requirements-resnet.txt
```

#### Step 5: Verify CUDA Installation
```python
# Run in Python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

Expected output:
```
PyTorch version: 2.x.x+cuXXX
CUDA available: True
CUDA version: 11.8 (or your version)
GPU: NVIDIA GeForce RTX XXXX (or your GPU model)
```

## 🚀 Running the Notebook

### Step 1: Prepare Data

Make sure you have run the preprocessing pipeline first:

```bash
# From project root directory
python scripts/run_pipeline.py
```

This will create preprocessed data files in `artifacts/run_YYYYMMDD_HHMMSS/`

### Step 2: Launch Jupyter Notebook

```bash
# From project root
jupyter notebook
```

Or use Jupyter Lab:
```bash
jupyter lab
```

### Step 3: Open the ResNet Notebook

Navigate to: `notebooks/resnet_oil_temperature_prediction.ipynb`

### Step 4: Run the Notebook

- Click "Cell" → "Run All" to execute all cells
- Or run cells sequentially with Shift+Enter

**Training Time Estimates**:
- With GPU (CUDA): 5-15 minutes for all 3 configurations
- With CPU only: 20-60 minutes for all 3 configurations

## 📊 Notebook Structure

1. **Setup & Imports** - Load required libraries
2. **Device Configuration** - CUDA setup
3. **Data Loading** - Load preprocessed data
4. **Model Architecture** - ResNet definition
5. **Training Utilities** - Helper functions
6. **Training** - Train models for 1h, 1d, 1w
7. **Evaluation** - Calculate metrics
8. **Visualization** - Plot results
9. **Comparison** - Compare with baseline models
10. **Analysis** - Error analysis and insights

## 📁 Output Files

After running the notebook, the following files will be generated:

### Models
```
models/resnet/
├── resnet_1h.pth    # 1-hour prediction model
├── resnet_1d.pth    # 1-day prediction model
└── resnet_1w.pth    # 1-week prediction model
```

### Visualizations
```
visualizations/resnet/
├── training_history.png    # Training & validation loss curves
├── model_comparison.png    # Performance comparison with baselines
├── predictions.png         # Actual vs predicted plots
└── error_analysis.png      # Error distribution and patterns
```

### Results
```
results/
├── resnet_comparison.csv       # Comparison table (CSV format)
└── resnet_final_results.pkl    # Complete results (pickle format)
```

## 🎯 Expected Performance

Based on the model architecture and training configuration:

| Configuration | Expected R² | Expected RMSE |
|--------------|-------------|---------------|
| 1-hour       | 0.55 - 0.65 | 4.5 - 5.5 °C  |
| 1-day        | 0.35 - 0.50 | 5.0 - 6.0 °C  |
| 1-week       | 0.20 - 0.35 | 5.5 - 6.5 °C  |

**Current Best (Random Forest)**:
- 1-hour: R² = 0.60, RMSE = 4.68 °C

## 🔧 Troubleshooting

### Issue: CUDA Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solution**:
- Reduce batch size in notebook (e.g., from 256 to 128 or 64)
- Close other GPU-intensive applications
- Reduce model size (fewer blocks or smaller hidden dimensions)

### Issue: CUDA Not Available

**Error**: `CUDA available: False`

**Solutions**:
1. Verify NVIDIA GPU drivers are installed
2. Reinstall PyTorch with correct CUDA version
3. Check CUDA toolkit installation
4. The code will automatically fall back to CPU (slower but works)

### Issue: Import Error

**Error**: `ModuleNotFoundError: No module named 'torch'`

**Solution**:
- Ensure virtual environment is activated
- Reinstall PyTorch: `pip install torch torchvision`

### Issue: Preprocessing Data Not Found

**Error**: `FileNotFoundError: X_train_1h.npy`

**Solution**:
```bash
# Run preprocessing first
python scripts/preprocessing/optimized_preprocessing.py
```

Or run the complete pipeline:
```bash
python scripts/run_pipeline.py
```

## 📈 Hyperparameter Tuning

To improve performance, you can modify training parameters in the notebook:

```python
training_config = {
    'hidden_dims': [64, 64],      # Try [128, 128] or [64, 128, 64]
    'num_blocks': 2,              # Try 3 for deeper network
    'dropout': 0.2,               # Try 0.1, 0.3, 0.4
    'learning_rate': 0.001,       # Try 0.0001 or 0.01
    'batch_size': 256,            # Try 128, 512
    'epochs': 100,                # Increase to 150 or 200
    'patience': 15                # Early stopping patience
}
```

## 🔬 Advanced Usage

### Load Trained Model

```python
import torch
from notebooks.resnet_oil_temperature_prediction import ResNetRegressor

# Load model
checkpoint = torch.load('models/resnet/resnet_1h.pth')
model = ResNetRegressor(input_dim=6, hidden_dims=[64, 64], num_blocks=2)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Make predictions
import numpy as np
X_new = np.array([[...]])  # Your input data
X_tensor = torch.FloatTensor(X_new)
prediction = model(X_tensor).item()
```

### Export for Production

```python
# Export to ONNX format (cross-platform)
dummy_input = torch.randn(1, 6)
torch.onnx.export(model, dummy_input, "resnet_1h.onnx")
```

## 📚 References

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [ResNet Paper](https://arxiv.org/abs/1512.03385) - Deep Residual Learning for Image Recognition
- [CUDA Installation Guide](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/)

## 🤝 Contributing

For improvements or issues:
1. Create a new branch: `git checkout -b feature/your-feature`
2. Make changes and test
3. Commit: `git commit -m "Description"`
4. Push: `git push origin feature/your-feature`
5. Create Pull Request

## 📝 Notes

- Training time varies significantly based on GPU model
- RTX 3060 or better recommended for smooth training
- CPU training is possible but much slower
- All code is compatible with both Windows and Linux
- macOS users: Install CPU-only version (no CUDA support on Mac)

---

**Last Updated**: October 2025
**Tested On**: Windows 11, CUDA 11.8, PyTorch 2.0+
