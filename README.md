# RF Anomaly Detection with Continuous Learning

[![Tests](https://github.com/YOUR_USERNAME/rf-anomaly-detection/actions/workflows/test.yml/badge.svg)](https://github.com/YOUR_USERNAME/rf-anomaly-detection/actions/workflows/test.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![arXiv](https://img.shields.io/badge/arXiv-XXXX.XXXXX-b31b1b.svg)](https://arxiv.org/abs/XXXX.XXXXX)

A PyTorch-based research project for **unsupervised anomaly detection in raw I/Q RF signals** using SNR-conditioned variational autoencoders with latent-space Mahalanobis distance scoring.

## Key Results

| Metric | Value |
|--------|-------|
| **Overall AUROC (hybrid)** | 0.9549 |
| **HackRF live WiFi** | 0.9735 |
| **POWDER LTE+DSSS** | 0.8882 |
| **Latent vs Reconstruction** | 2.2x improvement (0.93 vs 0.42) |

**Key Innovation:** Reconstruction-based anomaly detection fails for normalized RF signals. Our latent-space approach using Mahalanobis distance achieves 2.2x better AUROC.

## Features

- **SNR-Conditioned VAE**: Adapts to signal quality conditions
- **Latent-Space Detection**: Mahalanobis distance outperforms reconstruction error
- **Power Conditioning**: Preserves amplitude info lost in normalization
- **Hybrid Detection**: Combines learned + physics-based features
- **Continuous Learning**: Online learning, EWC, periodic retraining
- **Real-World Validated**: Tested on HackRF and POWDER datasets

## Quick Start

```bash
# Clone and install
git clone https://github.com/YOUR_USERNAME/rf-anomaly-detection.git
cd rf-anomaly-detection
pip install -r requirements.txt

# Run quickstart example
python examples/quickstart.py

# Or use the production model
python experiments/evaluate.py --checkpoint checkpoints/snr_vae_hybrid_v1_20260118/best_model.pt
```

## Project Structure

```
rf-anomaly-detection/
├── src/
│   ├── models/          # VAE architectures (SNR-conditioned)
│   ├── data/            # Synthetic data generation
│   ├── learning/        # Continuous learning (EWC, online)
│   ├── detection/       # Anomaly detection (Mahalanobis, hybrid)
│   └── utils/           # Config, visualization
├── experiments/         # Training and evaluation scripts
├── examples/            # Quickstart examples
├── tests/               # Unit tests
├── configs/             # YAML configurations
└── checkpoints/         # Trained models
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# For development
pip install -r requirements-dev.txt
```

## Usage

### Train a Model

```bash
python experiments/train_baseline.py --config configs/default.yaml
```

### Evaluate Model

```bash
python experiments/evaluate.py \
    --checkpoint checkpoints/snr_vae_hybrid_v1_20260118/best_model.pt \
    --save-plots
```

### Test on POWDER Dataset

```bash
python experiments/test_powder_data.py
```

## Model Architecture

```
I/Q Signal [batch, 2, 1024] → SNRConditionedVAE → Latent [batch, 32] → AnomalyDetector → Score
                              ↑                                         ↑
                         SNR + Power                              Mahalanobis Distance
                         Conditioning                             (not reconstruction error)
```

**Why latent-space detection?** Reconstruction-based methods fail because VAEs can reconstruct anomalies *better* than normal signals after normalization. [Bouman & Heskes (2025)](https://arxiv.org/abs/2501.13864) prove this theoretically.

## Anomaly Detection Methods

| Method | Best For | AUROC |
|--------|----------|-------|
| Amplitude Threshold | Power anomalies | 0.93 |
| VAE Latent (Mahalanobis) | General anomalies | 0.93 |
| Hybrid (latent + freq) | Balanced detection | 0.95 |
| ChirpDetector | Frequency drift | 0.92 |

## Supported Anomaly Types

- Narrowband interference
- Frequency drift
- Amplitude spikes
- Phase noise
- Burst noise
- DSSS interference (validated on POWDER)

## Validation Datasets

| Dataset | Type | AUROC | Notes |
|---------|------|-------|-------|
| Synthetic | Generated | 0.9549 | 5 anomaly types |
| HackRF WiFi | Live capture | 0.9735 | 200 samples at 2.437 GHz |
| POWDER LTE+DSSS | Real LTE | 0.8882 | Unseen anomaly type |

## Configuration

Edit `configs/default.yaml`:

```yaml
model:
  type: "snr_vae"
  latent_dim: 32
  use_power_conditioning: true

detection:
  method: "latent"  # NOT "reconstruction"
  snr_adaptive: true

training:
  batch_size: 64
  learning_rate: 1e-3
  num_epochs: 100
```

## Testing

```bash
# Run all tests
pytest tests/ -v

# With coverage
pytest tests/ --cov=src --cov-report=html
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- University of Memphis for compute resources. 

