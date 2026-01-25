#!/bin/bash
#SBATCH --job-name=prod_scaled
#SBATCH --partition=igpuq
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/prod_scaled_%j.out
#SBATCH --error=logs/prod_scaled_%j.err

# Scaled Production Model Training
# Goal: Beat 0.95 AUROC with 5x more data and longer training
#
# Changes from baseline:
#   - 50,000 training samples (was 10,000)
#   - 300 epochs (was 100)
#   - Patience 30 (was 10)
#   - Batch size 128 (was 64)

set -e

echo "=========================================="
echo "Scaled Production Model Training"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
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

# Navigate to project
cd ~/CLP_Project
mkdir -p logs checkpoints/production_scaled

# Print environment
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\")')"
echo "=========================================="

CONFIG="configs/production_scaled.yaml"
OUTPUT_DIR="checkpoints/production_scaled/job_${SLURM_JOB_ID}"

echo "Configuration:"
echo "  Config: $CONFIG"
echo "  Output: $OUTPUT_DIR"
echo "  Training samples: 50,000"
echo "  Epochs: 300"
echo "  Early stopping patience: 30"
echo "=========================================="

# Run training
python experiments/train_baseline.py \
    --config "$CONFIG" \
    --device cuda \
    --output-dir "$OUTPUT_DIR"

echo "=========================================="
echo "Training Complete!"
echo "End Time: $(date)"
echo "Results: $OUTPUT_DIR"
echo "=========================================="

# Print final metrics if available
if [ -f "$OUTPUT_DIR/metrics.json" ]; then
    echo ""
    echo "Final Metrics:"
    python -c "
import json
with open('$OUTPUT_DIR/metrics.json') as f:
    m = json.load(f)
    print(f'  Best Val Loss: {m.get(\"best_val_loss\", \"N/A\")}')
    print(f'  Best Epoch: {m.get(\"best_epoch\", \"N/A\")}')
    if 'detection' in m:
        for method, scores in m['detection'].items():
            if isinstance(scores, dict):
                print(f'  {method} AUROC: {scores.get(\"auroc\", \"N/A\")}')
"
fi
