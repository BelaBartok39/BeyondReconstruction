#!/bin/bash
# Setup Python environment on cluster
# Run this script once after syncing the project to the cluster

set -e

echo "Setting up RF Anomaly Detection environment..."

# Load required modules
module purge
module load cuda/12.1
module load anaconda3

# Create virtual environment
ENV_NAME="rf_anomaly_env"
ENV_PATH="$HOME/$ENV_NAME"

if [ -d "$ENV_PATH" ]; then
    echo "Environment already exists at $ENV_PATH"
    echo "To recreate, first run: rm -rf $ENV_PATH"
    exit 0
fi

echo "Creating virtual environment at $ENV_PATH..."
python -m venv "$ENV_PATH"

# Activate environment
source "$ENV_PATH/bin/activate"

# Upgrade pip
pip install --upgrade pip

# Install PyTorch with CUDA support
echo "Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
echo "Installing requirements..."
cd ~/CLP_Project
pip install -r requirements.txt

# Verify installation
echo ""
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA version: {torch.version.cuda}')"

echo ""
echo "=========================================="
echo "Environment setup complete!"
echo "To activate: source ~/rf_anomaly_env/bin/activate"
echo "=========================================="
