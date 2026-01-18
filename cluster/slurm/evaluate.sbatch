#!/bin/bash
#SBATCH --job-name=rf_evaluate
#SBATCH --partition=igpuq
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

# RF Anomaly Detection Evaluation Job
# Usage: sbatch evaluate.sbatch <checkpoint_path>

set -e

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Check checkpoint argument
if [ -z "$1" ]; then
    echo "Error: Checkpoint path required"
    echo "Usage: sbatch evaluate.sbatch <checkpoint_path>"
    exit 1
fi

CHECKPOINT=$1

# Load modules
module purge
module load nvhpc/23.11
module load python/3.10.13/gcc.8.5.0
module load cuda/12.3
module load cudnn/8.9.7.29

# Activate environment
conda deactivate 2>/dev/null || true
source /project/ndrdmond/pythonGPU/bin/activate

cd ~/CLP_Project
mkdir -p logs

echo "Evaluating checkpoint: $CHECKPOINT"
echo "=========================================="

python experiments/evaluate.py \
    --checkpoint "$CHECKPOINT" \
    --num-test-samples 10000 \
    --anomaly-ratio 0.2 \
    --save-plots

echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="
