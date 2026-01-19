#!/bin/bash
#SBATCH --job-name=rf_complex_enc
#SBATCH --partition=igpuq
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/complex_encoder_%j.out
#SBATCH --error=logs/complex_encoder_%j.err

# Complex Encoder Training Experiment
# Tests complex-valued convolutions for phase preservation

set -e

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Experiment: Complex-Valued Encoder"
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
mkdir -p logs checkpoints

echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "=========================================="

# Run complex encoder training experiment
python experiments/train_complex_encoder.py

echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="
