#!/bin/bash
#SBATCH --job-name=rf_detection
#SBATCH --partition=igpuq
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/detection_%j.out
#SBATCH --error=logs/detection_%j.err

# Detection Methods Experiment
# Tests improved hybrid detection with enhanced frequency features

set -e

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Experiment: Improved Detection Methods"
echo "=========================================="

# Load required modules
module purge
module load nvhpc/23.11
module load python/3.10.13/gcc.8.5.0
module load cuda/12.3
module load cudnn/8.9.7.29

# Activate environment
conda deactivate 2>/dev/null || true
source /project/ndrdmond/pythonGPU/bin/activate

# Navigate to project directory
cd ~/CLP_Project

# Create directories
mkdir -p logs results

echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "=========================================="

# Run improved detection test
python experiments/test_improved_detection.py

echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="
