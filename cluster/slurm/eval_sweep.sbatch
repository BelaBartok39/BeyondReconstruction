#!/bin/bash
#SBATCH --job-name=eval_sweep
#SBATCH --partition=igpuq
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/eval_sweep_%j.out
#SBATCH --error=logs/eval_sweep_%j.err

# Sweep freq_weight to find optimal setting
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

conda deactivate 2>/dev/null || true
source /project/ndrdmond/pythonGPU/bin/activate

cd ~/CLP_Project

CHECKPOINT="checkpoints/20260118_184144/best_model.pt"

echo "Testing different freq_weights..."
echo ""

for FW in 0.3 0.4 0.5 0.6 0.7; do
    echo "=========================================="
    echo "Testing freq_weight = $FW"
    echo "=========================================="
    python experiments/reproduce_production.py \
        --checkpoint $CHECKPOINT \
        --device cuda \
        --freq-weight $FW 2>&1 | grep -E "(AUROC|FAIL|PASS|Hybrid)"
    echo ""
done

# Also try production_scaled if it exists
if [ -f "checkpoints/production_scaled/best_model.pt" ]; then
    echo "=========================================="
    echo "Testing production_scaled checkpoint"
    echo "=========================================="
    python experiments/reproduce_production.py \
        --checkpoint checkpoints/production_scaled/best_model.pt \
        --device cuda \
        --freq-weight 0.5 2>&1 | grep -E "(AUROC|FAIL|PASS|Hybrid)"
fi

echo ""
echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="
