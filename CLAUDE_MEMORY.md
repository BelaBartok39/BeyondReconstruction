# Claude Session Memory - RF Anomaly Detection Project

**Last Updated:** 2026-01-18 12:45
**Session Status:** All experiments completed - analysis ready

---

## Project Overview

This is a research pipeline for **raw I/Q anomaly detection in RF signals** with two key innovations:

1. **SNR as a learned feature** - SNR is embedded directly into the model architecture (SNRConditionedVAE), not just used as a threshold
2. **Continuous learning** - Three approaches implemented: Online Learning, EWC (Elastic Weight Consolidation), and Periodic Retraining with replay buffers

### Goals
- More granular analysis using SNR conditioning
- Higher accuracy with fewer false positives
- Adaptive detection across varying signal conditions

---

## Project Structure

```
/home/babynicky/Work/CLP_Project/
├── src/
│   ├── models/          # ConvAutoencoder, VAE, SNRConditionedVAE
│   ├── data/            # Synthetic I/Q generation, SNR estimation, datasets
│   ├── learning/        # Online, EWC, Periodic retraining, Replay buffers
│   ├── detection/       # SNR-adaptive anomaly detection, metrics
│   └── utils/           # Config, visualization
├── experiments/         # train_baseline.py, evaluate.py, compare_learning.py
├── configs/             # default.yaml
├── cluster/             # SLURM scripts for GPU cluster
│   ├── slurm/           # train.sbatch, evaluate.sbatch, interactive.sbatch
│   ├── setup_env.sh
│   └── sync.sh          # Push/pull code to cluster
├── tests/               # pytest test suite
└── notebooks/           # Jupyter exploration notebook
```

---

## GPU Cluster Information

**Cluster:** University of Memphis - bigblue.memphis.edu
**User:** ndrdmond
**SSH Key:** ~/.ssh/school_gpu_key
**Partition:** igpuq (interactive GPU queue)

### Environment on Cluster
- **Path:** `/project/ndrdmond/pythonGPU` (lowercase 'p'!)
- **Modules loaded:**
  - nvhpc/23.11
  - python/3.10.13/gcc.8.5.0
  - cuda/12.3
  - cudnn/8.9.7.29

### Useful Commands
```bash
# Interactive GPU session (allocates 2 GPUs for 1 day)
srun -c 2 --mem=10G --gres=gpu:2 -t 1-00:00:00 -p igpuq --pty bash

# Sync code to cluster
./cluster/sync.sh push

# Pull results from cluster
./cluster/sync.sh pull

# Check cluster status
./cluster/sync.sh status

# Submit batch job
sbatch cluster/slurm/train.sbatch
```

---

## Work Completed This Session

### 1. Reviewed and Fixed SLURM Scripts
- Changed partition from `gpu` to `igpuq`
- Updated modules to match working setup (nvhpc, cuda 12.3, cudnn, python 3.10.13)
- Fixed environment path: `/project/ndrdmond/pythonGPU` (was incorrectly capitalized)
- Added `conda deactivate` before activating venv

### 2. Code Simplification
Ran code-simplifier agent on all source files:

| Area | Before | After | Reduction |
|------|--------|-------|-----------|
| Models (src/models/) | 1,292 lines | 810 lines | 37% |
| Data (src/data/) | 1,149 lines | 989 lines | 14% |
| Learning (src/learning/) | ~1,127 lines | ~853 lines | 20% |
| Detection/Utils | 1,442 lines | 1,220 lines | 15% |
| Experiments | 1,153 lines | 1,039 lines | 10% |

Key improvements:
- Extracted helper functions to eliminate duplication
- Used list/dict comprehensions
- Vectorized operations where possible
- Consistent patterns across files

### 3. Bug Fixes

**Fix 1: EWC lazy initialization bug** (`src/learning/ewc.py`)
- Issue: Fisher accumulator created before lazy layers initialized
- Fix: Added dummy forward pass before creating accumulator

**Fix 2: YAML scientific notation parsing** (`src/utils/config.py`)
- Issue: Values like `1e6` and `1e-3` parsed as strings instead of floats
- Fix: Added `_convert_value()` method to convert numeric strings

**Fix 3: Type conversion in synthetic.py** (`src/data/synthetic.py`)
- Issue: `sample_rate` passed as string caused TypeError
- Fix: Added explicit `float()` conversion in `__init__`

### 4. Test Suite
- All 50 tests passing (1 skipped for CUDA on local machine)
- Tests located in `/tests/` directory
- Run with: `python -m pytest tests/ -v`

---

## Current Status

### All Training Jobs Complete

| Job ID | Mode | Status | Best Checkpoint |
|--------|------|--------|-----------------|
| 1988436 | Baseline (unsupervised) | Complete | `checkpoints/job_1988436/best_model.pt` |
| 1988439 | Semi-supervised (15% anomalies) | Complete | `checkpoints/semi_supervised_1988439/best_model.pt` |

### Output Locations (on cluster)
- Baseline: `~/CLP_Project/checkpoints/job_1988436/`
- Semi-supervised: `~/CLP_Project/checkpoints/semi_supervised_1988439/`

---

## Next Steps (Suggested)

1. **Pull results locally** - `./cluster/sync.sh pull`
2. **Compare continuous learning methods** - Run `experiments/compare_learning.py`
3. **Investigate why anomalies have lower error** - This is the core issue
4. **Try different architectures** - Current VAE may be too good at reconstructing everything

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `src/models/snr_encoder.py` | SNRConditionedVAE - main model |
| `src/learning/ewc.py` | Elastic Weight Consolidation |
| `src/learning/online.py` | Online learning updates |
| `src/learning/periodic.py` | Periodic retraining |
| `src/learning/replay_buffer.py` | 4 replay strategies |
| `src/detection/detector.py` | SNR-adaptive anomaly detection |
| `src/detection/metrics.py` | SNR-stratified evaluation metrics |
| `experiments/train_baseline.py` | Main training script |
| `experiments/evaluate.py` | Evaluation pipeline |
| `experiments/compare_learning.py` | Compare learning methods |
| `configs/default.yaml` | All configuration settings |

---

## Config Highlights (configs/default.yaml)

```yaml
data:
  sequence_length: 1024
  sample_rate: 1e6
  snr_range: [-5, 30]  # dB

model:
  type: "snr_vae"
  latent_dim: 32
  snr_embedding_dim: 16

training:
  batch_size: 64
  learning_rate: 1e-3
  num_epochs: 100
  early_stopping_patience: 10

detection:
  snr_adaptive: true
  snr_bins: 7
```

---

## Critical Discovery: Inverted Anomaly Detection

**Problem Identified:** The original model had anomalies with LOWER reconstruction error than normal signals (AUROC=0.40, Cohen's d=-0.53). This is backwards from expected behavior.

### Fixes Implemented

**Fix 1: Score Inversion Option** (`src/detection/detector.py`)
- Added `invert_scores` parameter to AnomalyDetector
- Added `hybrid_weights` for tunable reconstruction/latent balance
- Use `--invert-scores` flag in evaluate.py

**Fix 2: Stronger Anomaly Generation** (`src/data/synthetic.py`)
- Interference: SIR range changed from [-5,10] to [-10,5] dB
- Frequency drift: rate increased from ±10 to ±30 Hz/sample
- Amplitude spikes: amplitude increased from 2-5x to 3-10x
- Phase noise: std increased from 0.3-1.0 to 0.5-2.0 rad
- Burst noise: more bursts (2-8 vs 1-5), stronger noise

**Fix 3: Semi-Supervised Training** (`experiments/train_baseline.py`)
- Added `--semi-supervised` flag
- Added `--train-anomaly-ratio` (default 0.1)
- Added `--contrastive-weight` (default 1.0)
- Contrastive loss encourages higher reconstruction error for anomalies

### Experimental Results Summary

| Approach | AUROC | AUPRC | Precision | Recall | Cohen's d |
|----------|-------|-------|-----------|--------|-----------|
| Baseline (no fix) | 0.40 | 0.09 | 0.10 | 0.32 | -0.53 |
| Baseline + invert | **0.56** | 0.29 | **0.71** | 0.18 | +0.29 |
| Semi-supervised (no invert) | 0.49 | 0.14 | 0.10 | 0.68 | -0.29 |
| Semi-supervised + invert | **0.56** | 0.44 | **0.72** | 0.27 | +0.29 |

**Key Finding:** Score inversion is the key fix. Semi-supervised training with contrastive loss did not improve discrimination. Both approaches converge to ~0.56 AUROC when properly inverted.

### Root Cause Analysis

The model reconstructs anomalies *better* than normal signals. Possible reasons:
1. Anomalies may have simpler/more regular structure (especially burst noise, interference)
2. VAE may be learning a representation that captures signal energy rather than modulation patterns
3. Training data (all normal) may not provide enough variation for the model to learn "normalcy"

### Recommendations for Further Work

1. **Architecture changes:**
   - Try masked autoencoder (reconstruct only parts of signal)
   - Add frequency-domain loss component
   - Use adversarial training (discriminator for anomalies)

2. **Data augmentation:**
   - Add more diverse normal signals
   - Train on multiple modulation types with more variation

3. **Alternative detection methods:**
   - One-class SVM on latent space
   - Isolation forest on latent features
   - Ensemble of reconstruction + latent distance methods

---

## Root Cause Analysis: Normalization Issue (Session 2)

### Discovery
The normalization step `signal / max(|signal|)` was destroying anomaly signatures:

| Anomaly Type | Pre-Norm Power | Compression Factor | Post-Norm Amp Variance |
|--------------|----------------|-------------------|------------------------|
| **Amplitude Spike** | 6.69x normal | **14.36x** | -65.5% vs normal |
| **Burst Noise** | 4.44x normal | **11.59x** | -67.4% vs normal |
| Interference | 3.79x normal | 3.92x | -42.8% vs normal |

Signals with high-amplitude anomalies got compressed so much they became nearly flat/zero, making them **trivially easy to reconstruct**.

### Solution Implemented: Power Conditioning

Added signal power (pre-normalization) as an additional conditioning input to the VAE:

**Power Distribution Differences:**
```
Normal:             -8.90 dB +/- 1.73
Interference:       -4.18 dB (+4.72 dB vs normal)
Amplitude spike:    -2.11 dB (+6.79 dB vs normal)
Burst noise:        -3.75 dB (+5.15 dB vs normal)
Frequency drift:    -8.94 dB (~same as normal)
Phase noise:        -8.98 dB (~same as normal)
```

### Files Modified
- `src/data/synthetic.py` - Compute and return `signal_power_db` in metadata
- `src/data/datasets.py` - Add `power` and `power_db` to dataset output
- `src/models/snr_encoder.py` - Add `use_power_conditioning` parameter
- `src/detection/detector.py` - Pass power through detection pipeline
- `experiments/train_baseline.py` - Handle power in training loop
- `configs/default.yaml` - Add `use_power_conditioning: true` and `power_range`

### Usage
```yaml
# configs/default.yaml
model:
  use_power_conditioning: true

data:
  power_range: [-20, 10]  # dB range for normalization
```

Now the model learns: "If power is high AND the signal looks empty (compressed), it's likely an anomaly."

---

## Troubleshooting Notes

1. **YAML parses scientific notation as strings** - Fixed in config.py with `_convert_value()`
2. **Environment path is case-sensitive** - Use `pythonGPU` not `PythonGPU`
3. **Lazy layer initialization** - Do dummy forward pass before accessing all parameters
4. **DataLoader workers warning** - Cluster recommends max 2 workers, config has 4 (non-fatal)
5. **Inverted anomaly scores** - Use `--invert-scores` or `invert_scores: true` in config
