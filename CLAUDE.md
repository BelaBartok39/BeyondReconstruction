# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RF Anomaly Detection with Continuous Learning - A PyTorch research project for detecting anomalies in raw I/Q RF signals using SNR-conditioned VAEs with latent-space anomaly detection.

**Key Innovation:** Latent-only detection (Mahalanobis distance) outperforms reconstruction-based detection by 2x (0.93 vs 0.42 AUROC).

## Common Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run all tests
python -m pytest tests/ -v

# Run single test file
python -m pytest tests/test_models.py -v

# Run tests with coverage
python -m pytest tests/ --cov=src --cov-report=html

# Train baseline model
python experiments/train_baseline.py --config configs/default.yaml

# Evaluate model
python experiments/evaluate.py --checkpoint checkpoints/<timestamp>/best_model.pt --save-plots

# Compare continuous learning methods
python experiments/compare_learning.py --baseline-checkpoint checkpoints/<timestamp>/best_model.pt

# Validate no overfitting (requires checkpoint)
python experiments/validate_no_overfit.py --checkpoint <path>

# Test improved detection methods
python experiments/test_improved_detection.py
```

## Architecture

### Data Flow
```
I/Q Signal [batch, 2, 1024] → SNRConditionedVAE → Latent [batch, 32] → AnomalyDetector → Score
                              ↑                                         ↑
                         SNR + Power                              Mahalanobis Distance
                         Conditioning                             (not reconstruction error)
```

### Core Components

**`src/models/`**
- `snr_encoder.py`: Main model - `SNRConditionedVAE` with `SNREncoder` and `SNRDecoder`. Use `create_model(config)` to instantiate.
- `blocks.py`: Convolutional building blocks (`ConvBlock`, `ConvTransposeBlock`)
- `bayesian.py`: Bayesian linear layers for uncertainty estimation (currently disabled in default config)

**`src/data/`**
- `synthetic.py`: `SyntheticRFGenerator` - generates I/Q signals with configurable modulations and anomaly types
- `datasets.py`: `RFDataset` wrapping the generator for PyTorch DataLoader
- `snr_estimation.py`: M2M4, wavelet, and spectral SNR estimation methods

**`src/detection/`**
- `detector.py`: `AnomalyDetector` class - supports reconstruction, latent (Mahalanobis), and hybrid methods
- `phase_detector.py`: `PhaseAnomalyDetector`, `EnhancedFrequencyDetector`, `ChirpDetector` for frequency drift detection
- `metrics.py`: SNR-stratified evaluation metrics (AUROC, AUPRC, F1 by SNR bin)

**`src/learning/`**
- `online.py`: Online learning with gradient updates
- `ewc.py`: Elastic Weight Consolidation (prevents catastrophic forgetting)
- `periodic.py`: Periodic retraining with replay buffer
- `replay_buffer.py`: Experience replay with reservoir/FIFO/uniform sampling

### Key Configuration (`configs/default.yaml`)

```yaml
model:
  type: "snr_vae"
  latent_dim: 32  # Critical: 16 was too small (gave 0.40 AUROC)
  use_power_conditioning: true

detection:
  method: "latent"  # NOT "reconstruction" - latent is 2x better
  snr_adaptive: true
  snr_bins: 7

data:
  anomaly_severity: 4.0  # 1.0 is too subtle for detection
```

### Model Detection in Code

When checking model type in learning modules:
```python
# Correct way to detect SNRConditionedVAE
has_snr_conditioning = hasattr(model, 'cond_embed')  # NOT 'snr_embed'
```

## Critical Technical Notes

1. **Use latent-only detection**: Reconstruction-based fails because VAE reconstructs anomalies BETTER than normal signals (due to normalization compressing high-amplitude anomalies to near-zero).

2. **Power conditioning is essential**: High-amplitude anomalies (spikes, bursts) get compressed 10-14x during normalization. Power conditioning preserves this information.

3. **YAML scientific notation**: Values like `1e6` may parse as strings. The `ConfigLoader` in `src/utils/config.py` handles this with `_convert_value()`.

4. **EWC lazy initialization**: Must do a dummy forward pass before creating Fisher accumulator (see `src/learning/ewc.py`).

5. **Detector fitting**: Always fit detector on training data, never on test data (which contains anomalies).

6. **Frequency drift detection**: Hardest anomaly type (0.80 AUROC with latent-only). Use `ChirpDetector` for 0.92+ AUROC, or `HybridPhaseLatentDetector` with `freq_weight=0.5`.

## GPU Cluster (University of Memphis)

```bash
# Sync code to cluster
./cluster/sync.sh push

# Interactive GPU session
srun -c 2 --mem=10G --gres=gpu:2 -t 1-00:00:00 -p igpuq --pty bash

# Submit batch job
sbatch cluster/slurm/train.sbatch

# Pull results
./cluster/sync.sh pull
```

Environment path: `/project/ndrdmond/pythonGPU` (case-sensitive)

## Test Structure

- `tests/test_models.py`: Model architecture tests
- `tests/test_detection.py`: Anomaly detection tests
- `tests/test_learning.py`: Continuous learning tests
- `tests/test_synthetic.py`: Data generation tests
- `tests/test_architecture_revision.py`: Extended architecture validation

## Current Best Results

| Metric | Value |
|--------|-------|
| Overall AUROC (hybrid) | 0.9549 |
| Frequency drift (ChirpDetector) | 0.9245 |
| Generalization to unseen anomalies | 0.9970 |

Production model: `snr_conditioned_vae_hybrid_v1.pt`
