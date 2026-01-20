#!/bin/bash
# ============================================================
# EEG Image Decode - Environment Setup Script
# ============================================================
# This script creates a conda environment with all dependencies
# for reproducing the experiments in the paper.
#
# Usage: bash setup.sh
# ============================================================

set -e

ENV_NAME="BCI"
PYTHON_VERSION="3.10"

echo "============================================================"
echo "Creating conda environment: $ENV_NAME with Python $PYTHON_VERSION"
echo "============================================================"

# Create conda environment
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# Get conda base path
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate $ENV_NAME

echo "Installing base packages via conda..."
conda install numpy matplotlib tqdm scikit-image jupyterlab -y
conda install -c conda-forge accelerate -y

echo "Installing PyTorch ecosystem (CUDA 12.4)..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo "Installing Hugging Face packages..."
pip install transformers==4.36.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install diffusers==0.30.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install huggingface-hub==0.30.2 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install accelerate==1.5.2 -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "Installing CLIP packages..."
# pip install git+https://github.com/openai/CLIP.git  # Skipped due to GitHub connection issues
pip install open_clip_torch -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install clip-retrieval -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "Installing EEG processing packages..."
pip install braindecode==0.8.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install mne==1.9.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "Installing image generation packages..."
pip install dalle2-pytorch==1.15.6 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pytorch-msssim==1.0.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install kornia==0.8.0 -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "Installing deep learning utilities..."
pip install einops==0.8.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install info-nce-pytorch==0.1.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install reformer_pytorch==1.4.4 -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "Installing logging and visualization..."
pip install wandb==0.19.10 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install seaborn==0.13.2 -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "Installing other utilities..."
pip install ftfy==6.3.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install regex==2024.11.6 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install h5py==3.13.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install pandas==2.3.3 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install imageio==2.37.0 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install scipy==1.15.3 -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install scikit-learn==1.6.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

echo "============================================================"
echo "Environment setup complete!"
echo "Activate with: conda activate $ENV_NAME"
echo "============================================================"
