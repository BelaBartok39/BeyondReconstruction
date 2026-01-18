# RF Anomaly Detection with Continuous Learning Autoencoder

A PyTorch-based research project implementing autoencoder architectures for RF signal anomaly detection with continuous learning capabilities.

## Features

- **SNR-Conditioned VAE**: Autoencoder that adapts to signal quality
- **Continuous Learning**: Online learning, EWC, and periodic retraining
- **Synthetic RF Data**: Realistic signal generation with various modulations and anomalies
- **Comprehensive Evaluation**: SNR-stratified metrics and visualization tools

## Project Structure

```
CLP_Project/
├── src/
│   ├── models/          # Autoencoder architectures (AE, VAE, SNR-VAE)
│   ├── data/            # Data generation and loading
│   ├── learning/        # Continuous learning modules
│   ├── detection/       # Anomaly detection logic
│   └── utils/           # Configuration and visualization
├── experiments/         # Training and evaluation scripts
├── notebooks/           # Interactive exploration
├── tests/               # Unit tests
├── configs/             # Configuration files
└── cluster/             # SLURM deployment scripts
```

## Installation

```bash
# Clone and enter directory
cd CLP_Project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Explore the Data and Models

```bash
jupyter notebook notebooks/exploration.ipynb
```

### 2. Train Baseline Model

```bash
python experiments/train_baseline.py --config configs/default.yaml
```

### 3. Evaluate Model

```bash
python experiments/evaluate.py \
    --checkpoint checkpoints/<timestamp>/best_model.pt \
    --save-plots
```

### 4. Compare Learning Methods

```bash
python experiments/compare_learning.py \
    --baseline-checkpoint checkpoints/<timestamp>/best_model.pt
```

## Model Architectures

### Base Autoencoder
1D convolutional autoencoder for IQ time series reconstruction.

### Variational Autoencoder (VAE)
Adds KL divergence regularization for a smoother latent space.

### SNR-Conditioned VAE
Conditions both encoder and decoder on estimated SNR, enabling:
- SNR-dependent reconstruction
- Adaptive anomaly thresholds
- Better performance across signal quality levels

## Continuous Learning

### Online Learning
Incremental model updates with each new batch using reduced learning rate.

### Elastic Weight Consolidation (EWC)
Prevents catastrophic forgetting by penalizing changes to important parameters.

### Periodic Retraining
Buffers new samples and retrains at intervals with optional experience replay.

## Anomaly Detection Methods

1. **Reconstruction-based**: Threshold on reconstruction error
2. **Latent-space**: Mahalanobis distance in latent space
3. **Hybrid**: Combination of both methods
4. **SNR-adaptive**: Per-SNR-bin thresholds

## Cluster Deployment

### Setup

```bash
# Sync code to cluster
./cluster/sync.sh push

# SSH to cluster and setup environment
ssh -i ~/.ssh/school_gpu_key ndrdmond@bigblue.memphis.edu
cd ~/CLP_Project
bash cluster/setup_env.sh
```

### Submit Jobs

```bash
# Training job
sbatch cluster/slurm/train.sbatch

# Evaluation job
sbatch cluster/slurm/evaluate.sbatch checkpoints/best_model.pt

# Pull results back
./cluster/sync.sh pull
```

## Configuration

Edit `configs/default.yaml` to customize:

```yaml
model:
  type: "snr_vae"
  latent_dim: 32
  hidden_channels: [32, 64, 128, 256]

training:
  batch_size: 64
  learning_rate: 1e-3
  num_epochs: 100

continuous_learning:
  online:
    learning_rate: 1e-4
  ewc:
    lambda: 1000.0
  periodic:
    interval: 1000
```

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## Key Technical Details

### Data Format
- Input: `[batch, 2, sequence_length]` (I and Q channels)
- SNR: Normalized to [0, 1] for model input

### Anomaly Score
```python
anomaly_score = reconstruction_error / expected_error(snr)
```

### Supported Modulations
- BPSK, QPSK, 16-QAM, 64-QAM

### Supported Anomaly Types
- Narrowband interference
- Frequency drift
- Amplitude spikes
- Phase noise
- Burst noise

## Dependencies

- PyTorch >= 2.0
- NumPy, SciPy
- scikit-learn
- Matplotlib, Seaborn
- PyYAML, tqdm

## License

Research use only.
