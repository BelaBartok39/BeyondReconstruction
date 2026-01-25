#!/bin/bash
#SBATCH --job-name=vq_vae_train
#SBATCH --partition=igpuq
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/vq_vae_%j.out
#SBATCH --error=logs/vq_vae_%j.err

# VQ-VAE Training for RF Anomaly Detection
# Usage: sbatch train_vq_vae.sbatch
#
# Target: Beat 0.95 AUROC with VQ-VAE architecture
# Training: 250 epochs with early stopping (patience=25)

set -e

echo "=========================================="
echo "VQ-VAE Training Job"
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

# Activate environment (case-sensitive path!)
conda deactivate 2>/dev/null || true
source /project/ndrdmond/pythonGPU/bin/activate

# Navigate to project directory
cd ~/CLP_Project

# Create directories
mkdir -p logs checkpoints/vq_vae

# Print environment info
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU: $(python -c 'import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A")')"
echo "=========================================="

# Configuration
CONFIG="configs/vq_vae.yaml"
EPOCHS=250
NUM_EMBEDDINGS=512
LATENT_DIM=32
CHECKPOINT_DIR="checkpoints/vq_vae/job_${SLURM_JOB_ID}"

echo "Configuration:"
echo "  Config file: $CONFIG"
echo "  Epochs: $EPOCHS"
echo "  Codebook size: $NUM_EMBEDDINGS"
echo "  Embedding dim: $LATENT_DIM"
echo "  Checkpoint dir: $CHECKPOINT_DIR"
echo "=========================================="

# Run VQ-VAE training
python experiments/train_vq_vae.py \
    --config "$CONFIG" \
    --epochs $EPOCHS \
    --num-embeddings $NUM_EMBEDDINGS \
    --latent-dim $LATENT_DIM \
    --checkpoint-dir "$CHECKPOINT_DIR"

# Save job info
echo "Saving job metadata..."
cat > "$CHECKPOINT_DIR/job_info.json" << EOF
{
    "job_id": "$SLURM_JOB_ID",
    "node": "$SLURM_NODELIST",
    "start_time": "$(date -Iseconds)",
    "config": "$CONFIG",
    "epochs": $EPOCHS,
    "num_embeddings": $NUM_EMBEDDINGS,
    "latent_dim": $LATENT_DIM
}
EOF

echo "=========================================="
echo "Training Complete!"
echo "End Time: $(date)"
echo "Results: $CHECKPOINT_DIR"
echo "=========================================="

# Print final results summary
if [ -f "$CHECKPOINT_DIR/results.json" ]; then
    echo ""
    echo "Final Results:"
    python -c "
import json
with open('$CHECKPOINT_DIR/results.json') as f:
    results = json.load(f)
    print(f'  Best Epoch: {results[\"best_epoch\"]}')
    print(f'  Best Val Loss: {results[\"best_val_loss\"]:.4f}')
    for method, metrics in results['detection_results'].items():
        print(f'  {method}: AUROC={metrics[\"auroc\"]:.4f}, AUPRC={metrics[\"auprc\"]:.4f}')
    usage = results['codebook_usage']
    print(f'  Codebook: {usage[\"active_codes\"]}/{usage[\"total_codes\"]} active ({usage[\"utilization\"]*100:.1f}%)')
"
fi
