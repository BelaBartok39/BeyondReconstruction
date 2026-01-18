#!/bin/bash
#SBATCH --job-name=rf_evaluate
#SBATCH --partition=gpu
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
module load cuda/12.1
module load anaconda3

# Activate environment
source ~/rf_anomaly_env/bin/activate

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
