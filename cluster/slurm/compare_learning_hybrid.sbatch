#!/bin/bash
#SBATCH --job-name=compare_hybrid
#SBATCH --partition=igpuq
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/compare_hybrid_%j.out
#SBATCH --error=logs/compare_hybrid_%j.err

# Continuous Learning Comparison: Latent vs Hybrid Detection
# Tests online learning, EWC, periodic retraining with concept drift

set -e

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Experiment: Continuous Learning (Latent vs Hybrid)"
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

CHECKPOINT="checkpoints/20260118_184144/best_model.pt"

# Run with LATENT detection
echo ""
echo "=========================================="
echo "LATENT-ONLY DETECTION"
echo "=========================================="
python experiments/compare_learning.py \
    --baseline-checkpoint "$CHECKPOINT" \
    --detection-method latent \
    --concept-drift \
    --num-streaming-samples 5000 \
    --eval-interval 500 \
    --output-dir results/compare_latent_$(date +%Y%m%d_%H%M%S)

# Run with HYBRID detection
echo ""
echo "=========================================="
echo "HYBRID DETECTION (freq_weight=0.5)"
echo "=========================================="
python experiments/compare_learning.py \
    --baseline-checkpoint "$CHECKPOINT" \
    --detection-method hybrid \
    --freq-weight 0.5 \
    --concept-drift \
    --num-streaming-samples 5000 \
    --eval-interval 500 \
    --output-dir results/compare_hybrid_$(date +%Y%m%d_%H%M%S)

echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="
