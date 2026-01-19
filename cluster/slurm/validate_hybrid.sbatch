#!/bin/bash
#SBATCH --job-name=validate_hybrid
#SBATCH --partition=igpuq
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/validate_hybrid_%j.out
#SBATCH --error=logs/validate_hybrid_%j.err

# Overfitting Validation: Compare Latent vs Hybrid Detection
# Tests seed stability, generalization, SNR robustness, severity sensitivity

set -e

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Experiment: Overfitting Validation (Latent vs Hybrid)"
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

# Run with LATENT detection
echo ""
echo "=========================================="
echo "LATENT-ONLY DETECTION"
echo "=========================================="
python experiments/validate_no_overfit.py --detection-method latent

# Run with HYBRID detection
echo ""
echo "=========================================="
echo "HYBRID DETECTION (freq_weight=0.5)"
echo "=========================================="
python experiments/validate_no_overfit.py --detection-method hybrid --freq-weight 0.5

echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="
