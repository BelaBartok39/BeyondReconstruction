#!/bin/bash
#SBATCH --job-name=reproduce_prod
#SBATCH --partition=igpuq
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/reproduce_%j.out
#SBATCH --error=logs/reproduce_%j.err

# Reproduce Production Model (0.95+ AUROC)
# This script trains the model and evaluates with both latent and hybrid detection
# Usage: sbatch reproduce_production.sbatch

set -e

echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "Goal: Reproduce 0.95+ AUROC production model"
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

# Create directories if not exists
mkdir -p logs checkpoints results

echo "Python: $(which python)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
echo "=========================================="

# Run reproduction script
python experiments/reproduce_production.py \
    --config configs/default.yaml \
    --device cuda \
    --output-dir "checkpoints/reproduce_${SLURM_JOB_ID}" \
    --freq-weight 0.5

echo "=========================================="
echo "End Time: $(date)"
echo "=========================================="

# Show results summary
echo ""
echo "Results saved to:"
echo "  - checkpoints/reproduce_${SLURM_JOB_ID}/"
echo "  - results/reproduction_test.json"
echo ""
if [ -f results/reproduction_test.json ]; then
    echo "Quick Results:"
    python -c "import json; r=json.load(open('results/reproduction_test.json')); print(f\"  Latent AUROC: {r['latent_detection']['auroc']:.4f}\"); print(f\"  Hybrid AUROC: {r['hybrid_detection']['auroc']:.4f}\")"
fi
