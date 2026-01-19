#!/bin/bash
#SBATCH --job-name=rf_arch_revision
#SBATCH --partition=igpuq
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/arch_revision_%j.out
#SBATCH --error=logs/arch_revision_%j.err

# RF Anomaly Detection - Architecture Revision Training
# Trains model with:
#   - Probabilistic decoder (NLL loss)
#   - Smoothness prior
#   - Optional Bayesian last layer
#
# Usage: sbatch train_arch_revision.sbatch

set -e

echo "=========================================="
echo "Architecture Revision Training Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
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
mkdir -p logs checkpoints results

echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
if python -c 'import torch; print(torch.cuda.is_available())' | grep -q "True"; then
    echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0))')"
fi
echo "=========================================="

# Run tests first to verify implementation
echo "Running unit tests..."
python -m pytest tests/test_architecture_revision.py -v --tb=short || {
    echo "Tests failed! Aborting training."
    exit 1
}
echo "Tests passed!"
echo "=========================================="

# Train with probabilistic decoder + smoothness prior (Phase 1+2)
echo "Training with probabilistic decoder + smoothness prior..."
python experiments/train_baseline.py \
    --config configs/default.yaml \
    --device cuda \
    --output-dir "checkpoints/arch_revision_${SLURM_JOB_ID}"

echo "=========================================="
echo "Training Complete!"
echo "End Time: $(date)"
echo "Results saved to: checkpoints/arch_revision_${SLURM_JOB_ID}"
echo "=========================================="
