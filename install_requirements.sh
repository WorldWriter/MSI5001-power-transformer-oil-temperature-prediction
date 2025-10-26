#!/bin/bash
# MSI5001 Project - Dependency Installation Script
# MSI5001 项目 - 依赖安装脚本
# ================================================
#
# This script installs all required dependencies for the project
# 此脚本安装项目所需的所有依赖
#
# Usage / 使用方法:
#   chmod +x install_requirements.sh
#   ./install_requirements.sh
#
# Or / 或者:
#   bash install_requirements.sh

set -e  # Exit on error / 遇到错误时退出

echo "=========================================="
echo "MSI5001 Project - Installing Dependencies"
echo "MSI5001 项目 - 安装依赖"
echo "=========================================="
echo ""

# Check Python version / 检查 Python 版本
echo "Checking Python version / 检查 Python 版本..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"
echo ""

# Upgrade pip / 升级 pip
echo "Upgrading pip / 升级 pip..."
pip3 install --upgrade pip
echo ""

# Install basic dependencies / 安装基础依赖
echo "=========================================="
echo "Step 1: Installing basic dependencies"
echo "步骤 1: 安装基础依赖"
echo "=========================================="
pip3 install -r requirements.txt
echo ""

# Install PyTorch with CUDA support / 安装支持 CUDA 的 PyTorch
echo "=========================================="
echo "Step 2: Installing PyTorch with CUDA 12.1"
echo "步骤 2: 安装 PyTorch (CUDA 12.1)"
echo "=========================================="
echo ""
echo "This installation is optimized for RTX 4070 Ti Super"
echo "此安装针对 RTX 4070 Ti Super 显卡优化"
echo ""

# Ask user for installation preference / 询问用户安装偏好
echo "Please select PyTorch installation option:"
echo "请选择 PyTorch 安装选项:"
echo "  1) CUDA 12.1 (Recommended for RTX 40 series / 推荐用于 RTX 40 系列)"
echo "  2) CUDA 11.8 (Fallback / 备用选项)"
echo "  3) CPU only (No GPU / 仅 CPU)"
echo ""
read -p "Enter your choice (1-3) / 输入选择 (1-3) [default: 1]: " cuda_choice
cuda_choice=${cuda_choice:-1}

case $cuda_choice in
    1)
        echo "Installing PyTorch with CUDA 12.1..."
        echo "安装 PyTorch (CUDA 12.1)..."
        pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
        ;;
    2)
        echo "Installing PyTorch with CUDA 11.8..."
        echo "安装 PyTorch (CUDA 11.8)..."
        pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
        ;;
    3)
        echo "Installing PyTorch (CPU only)..."
        echo "安装 PyTorch (仅 CPU)..."
        pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
        ;;
    *)
        echo "Invalid choice. Installing CUDA 12.1 by default..."
        echo "无效选择。默认安装 CUDA 12.1..."
        pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
        ;;
esac
echo ""

# Verify installation / 验证安装
echo "=========================================="
echo "Step 3: Verifying installation"
echo "步骤 3: 验证安装"
echo "=========================================="
echo ""

python3 -c "
import sys
print('Python version:', sys.version)
print('')

try:
    import pandas as pd
    print(f'✓ pandas: {pd.__version__}')
except ImportError:
    print('✗ pandas: Not installed')

try:
    import numpy as np
    print(f'✓ numpy: {np.__version__}')
except ImportError:
    print('✗ numpy: Not installed')

try:
    import sklearn
    print(f'✓ scikit-learn: {sklearn.__version__}')
except ImportError:
    print('✗ scikit-learn: Not installed')

try:
    import matplotlib
    print(f'✓ matplotlib: {matplotlib.__version__}')
except ImportError:
    print('✗ matplotlib: Not installed')

try:
    import seaborn as sns
    print(f'✓ seaborn: {sns.__version__}')
except ImportError:
    print('✗ seaborn: Not installed')

print('')
print('PyTorch installation:')
try:
    import torch
    print(f'✓ PyTorch: {torch.__version__}')
    print(f'  CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'  CUDA version: {torch.version.cuda}')
        print(f'  GPU device: {torch.cuda.get_device_name(0)}')
        print(f'  Number of GPUs: {torch.cuda.device_count()}')
    else:
        print('  Running on CPU mode')
except ImportError:
    print('✗ PyTorch: Not installed')

try:
    import torchvision
    print(f'✓ torchvision: {torchvision.__version__}')
except ImportError:
    print('✗ torchvision: Not installed')
"

echo ""
echo "=========================================="
echo "Installation Complete! / 安装完成！"
echo "=========================================="
echo ""
echo "You can now run the notebooks:"
echo "您现在可以运行以下 notebook:"
echo "  - notebooks/rnn.ipynb"
echo "  - notebooks/informer.ipynb"
echo ""
echo "To start Jupyter Notebook / 启动 Jupyter Notebook:"
echo "  jupyter notebook"
echo ""
