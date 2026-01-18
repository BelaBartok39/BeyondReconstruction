#!/bin/bash
#SBATCH --job-name=rf_autoencoder
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# RF Anomaly Detection Training Job
# Usage: sbatch train.sbatch [config_file]

set -e

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Load required modules (adjust for your cluster)
module purge
module load cuda/12.1
module load anaconda3

# Activate environment
source ~/rf_anomaly_env/bin/activate

# Navigate to project directory
cd ~/CLP_Project

# Create logs directory if not exists
mkdir -p logs checkpoints

# Get config file from argument or use default
CONFIG=${1:-configs/default.yaml}

echo "Using config: $CONFIG"
echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "=========================================="

# Run training
python experiments/train_baseline.py \
    --config "$CONFIG" \
    --device cuda \
    --output-dir "checkpoints/job_${SLURM_JOB_ID}"

echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="
