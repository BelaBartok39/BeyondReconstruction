#!/bin/bash
#SBATCH --job-name=rf_interactive
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/interactive_%j.out
#SBATCH --error=logs/interactive_%j.err

# Interactive GPU session for debugging
# Usage: sbatch interactive.sbatch

echo "=========================================="
echo "Interactive Session Started"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "=========================================="

# Load modules
module purge
module load cuda/12.1
module load anaconda3

# Activate environment
source ~/rf_anomaly_env/bin/activate

cd ~/CLP_Project

echo ""
echo "Environment ready. You can now run Python commands."
echo "Example: python -c 'import torch; print(torch.cuda.is_available())'"
echo ""

# Keep session alive - replace with actual commands or use srun for interactive
sleep infinity
