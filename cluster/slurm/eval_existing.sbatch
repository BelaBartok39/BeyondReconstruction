#!/bin/bash
#SBATCH --job-name=eval_existing
#SBATCH --partition=igpuq
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

# Evaluate existing checkpoint
set -e

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
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

cd ~/CLP_Project

echo "Testing checkpoint: checkpoints/20260118_184144/best_model.pt"

python experiments/reproduce_production.py \
    --checkpoint checkpoints/20260118_184144/best_model.pt \
    --device cuda \
    --freq-weight 0.5

echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="
