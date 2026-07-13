# Beyond Reconstruction: A Latent-Space Anomaly Detection Study in Raw I/Q RF Signals Using SNR-Conditioned Variational Autoencoders

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

## Features

- **SNR-Conditioned VAE**: Adapts to signal quality conditions
- **Latent-Space Detection**: Mahalanobis distance outperforms reconstruction error
- **Power Conditioning**: Preserves amplitude info lost in normalization
- **Hybrid Detection**: Combines learned + physics-based features
- **Continuous Learning**: Online learning, EWC, periodic retraining
- **Real-World Validated**: Tested on HackRF and POWDER datasets

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


## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## Acknowledgments

- University of Memphis for compute resources. 

