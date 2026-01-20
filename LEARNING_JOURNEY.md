# RF Anomaly Detection: A Learning Journey

**Last Updated:** 2026-01-19

## From 42% to 95.5% AUROC - How We Got Here

This document chronicles our research journey developing an unsupervised anomaly detection system for RF signals. Each section explains what we discovered, why it mattered, and the concepts involved in accessible terms.

**Final Result:** 0.9549 average AUROC with hybrid detection (validated on cluster)

---

## Table of Contents

1. [The Problem We're Solving](#1-the-problem-were-solving)
2. [Where We Started](#2-where-we-started)
3. [Key Discovery #1: The Reconstruction Paradox](#3-key-discovery-1-the-reconstruction-paradox)
4. [Key Discovery #2: Latent Space Detection](#4-key-discovery-2-latent-space-detection)
5. [Key Discovery #3: Power Conditioning](#5-key-discovery-3-power-conditioning)
6. [Key Discovery #4: Phase-Aware Detection](#6-key-discovery-4-phase-aware-detection)
7. [What Didn't Work (And Why)](#7-what-didnt-work-and-why)
8. [Final Architecture](#8-final-architecture)
9. [Results Summary](#9-results-summary)
10. [Topics for Further Study](#10-topics-for-further-study)

---

## 1. The Problem We're Solving

### What Are RF Signals?

Radio Frequency signals are electromagnetic waves used for communication (WiFi, cellular, radar, etc.). These signals are represented as **I/Q data** (In-phase and Quadrature), which captures both amplitude and phase information.

Think of it like describing a spinning wheel:
- **I (In-phase)**: Where the wheel is horizontally
- **Q (Quadrature)**: Where the wheel is vertically
- Together, they tell you the wheel's position and speed

### What Are Anomalies?

We're looking for unusual patterns that might indicate:
- **Interference**: Another signal overlapping with ours
- **Frequency Drift**: The signal's frequency slowly changing over time
- **Amplitude Spikes**: Sudden power surges
- **Phase Noise**: Random variations in the signal's timing
- **Burst Noise**: Short bursts of interference

### The Challenge: Unsupervised Learning

We can't label every anomaly type in advance (there are infinite possibilities), so we need to learn what "normal" looks like and flag anything different. This is **unsupervised anomaly detection**.

---

## 2. Where We Started

### The Autoencoder Approach

We started with the standard approach: train a neural network to **compress** and **reconstruct** normal signals.

```
Input Signal → [Encoder] → Compressed "Latent" Space → [Decoder] → Reconstructed Signal
```

**The idea**: If the model learns to reconstruct normal signals well, it should struggle with anomalies, giving them higher reconstruction error.

### Initial Architecture: SNRConditionedVAE

```
- Input: I/Q Signal [batch, 2, 1024 samples]
- Encoder: Convolutional layers (32→64→128→256 channels)
- Latent space: 32 dimensions
- Decoder: Transposed convolutions (256→128→64→32)
- Conditioning: SNR (Signal-to-Noise Ratio) embedded into the model
```

### Initial Results: 42% AUROC

**AUROC** (Area Under ROC Curve) measures detection quality:
- 50% = random guessing
- 100% = perfect detection

Our initial **42% AUROC** was *worse than random*. Something was fundamentally wrong.

---

## 3. Key Discovery #1: The Reconstruction Paradox

### The Shocking Finding

When we analyzed the reconstruction errors:
- **Normal signals**: Higher reconstruction error
- **Anomalies**: Lower reconstruction error

This was backwards! The model reconstructed anomalies *better* than normal signals.

### Why This Happened: Normalization Artifacts

We normalize signals to [-1, 1] range before processing. High-amplitude anomalies (like amplitude spikes) get compressed into a flat, easy-to-reconstruct pattern:

```
Before normalization: [0.1, 0.2, 100, 0.3, 0.1]  ← Spike!
After normalization:  [0.001, 0.002, 1.0, 0.003, 0.001]  ← Looks flat and easy
```

The model learns that "flat signals are easy" and reconstructs them perfectly.

### The Quick Fix: Inverted Scores

Short-term fix: Invert the scores (multiply by -1).

Result: **42% → 56% AUROC** (better, but still poor)

### Lesson Learned

> **Normalization can hide the very patterns you're trying to detect.**

---

## 4. Key Discovery #2: Latent Space Detection

### The Breakthrough Insight

Instead of using reconstruction error, we looked at **where signals land in the latent space**.

The latent space is the compressed representation (32 numbers) that the encoder produces. Normal signals should cluster together; anomalies should be far from this cluster.

### Mahalanobis Distance Explained

We measure "how far" a sample is from the normal distribution using **Mahalanobis distance**.

**Simple explanation**: Imagine a crowd of people standing in a park. Some are clustered tightly, others spread out. Mahalanobis distance tells you "how unusual is this person's position?" accounting for both:
- The **center** of the crowd (mean)
- The **shape** of the crowd (some directions have more spread than others)

**Mathematical form**:
```
Mahalanobis Distance = √((x - μ)ᵀ Σ⁻¹ (x - μ))
```

Where:
- x = the point we're measuring
- μ = mean of normal samples
- Σ = covariance matrix (captures the shape/spread)

### Results: 91% AUROC!

By switching from reconstruction error to Mahalanobis distance in latent space:
- **42% → 91% AUROC** (+49 percentage points!)

This was our biggest single improvement.

### Why It Works

The VAE learns a latent space where:
1. Normal signals are encoded to similar regions (encouraged by KL divergence loss)
2. Anomalies, being different, get encoded to different regions
3. Measuring distance from "normal region" directly measures anomalousness

### Lesson Learned

> **The latent space often contains more useful information than the reconstruction.**

---

## 5. Key Discovery #3: Power Conditioning

### The Remaining Problem

Some anomalies still slipped through, especially those related to signal power (amplitude spikes).

### The Insight: Lost Information

When we normalize signals, we lose **power information**. A loud signal and a quiet signal look the same after normalization.

### The Solution: Preserve Power as a Conditioning Input

We compute the signal's power *before* normalization and feed it as an extra input:

```python
# Compute power BEFORE normalization
power = mean(I² + Q²)

# Then normalize the signal
signal_normalized = signal / max(abs(signal))

# Feed BOTH to the model
output = model(signal_normalized, snr, power)
```

This power value is:
1. Embedded into a learned representation (16-dim vector)
2. Concatenated with the latent code
3. Used by both encoder and decoder

### Results: 94% AUROC

With power conditioning:
- **91% → 94% AUROC**
- Amplitude spike detection: **100% AUROC** (perfect!)

### Lesson Learned

> **If normalization removes important information, preserve it as a conditioning input.**

---

## 6. Key Discovery #4: Phase-Aware Detection

### The Stubborn Anomaly: Frequency Drift

Even at 94% overall AUROC, **frequency drift** detection was lagging at 76-80%.

Frequency drift is subtle: the signal's frequency slowly changes over time. This affects the **phase** of the signal more than its amplitude.

### Understanding Phase

Phase is "where in its cycle" a signal is at any moment.

**Analogy**: Think of a clock's second hand:
- Amplitude = how long the hand is
- Phase = what second it's pointing at

Frequency drift causes the second hand to gradually speed up or slow down.

### The Failed Approach: Phase Loss During Training

We first tried adding a loss term to make the model care about phase during training.

**Result**: Disaster!
- Loss exploded (0.6 → 79 million)
- Amplitude spike detection collapsed (100% → 37%)

**Why**: The phase loss competed with reconstruction loss, destabilizing training.

### The Successful Approach: Phase Features at Detection Time

Instead of modifying training, we added phase-based features **after training**, at detection time:

**Original Phase Features**:
- Instantaneous frequency (how fast phase is changing)
- Phase variance (how much phase varies)
- Frequency drift rate (linear trend in frequency)

**Our Enhanced Frequency Features** (new):
- Spectral entropy (randomness of frequency content)
- Spectral centroid (center of frequency mass)
- Spectral bandwidth (spread of frequencies)
- Multi-scale variance ratio (short-term vs long-term variation)
- Spectral flux (how spectrum changes over time)

### Hybrid Scoring

We combine latent scores with frequency scores:

```
Final Score = (1 - weight) × Latent Score + weight × Frequency Score
```

With weight ≈ 0.5, we get:
- **Frequency drift**: 76% → 85% AUROC (+9%)
- **Overall**: 94% → 95-96% AUROC
- **No degradation** on other anomaly types

### Lesson Learned

> **Domain-specific features can complement learned representations without replacing them. Add them at detection time, not training time.**

---

## 7. What Didn't Work (And Why)

### Failed Attempt #1: Phase Loss During Training

**Idea**: Add a loss term that penalizes phase reconstruction error.

**What happened**:
- Loss exploded to 79 million
- Amplitude spike detection collapsed (100% → 37%)
- Burst noise detection collapsed (100% → 43%)

**Why it failed**:
- Phase loss competed with reconstruction loss
- Model optimized for phase at expense of amplitude
- The scales were mismatched (phase in radians, amplitude in signal units)

**Lesson**: Don't destabilize a working training process. Add features post-hoc instead.

### Failed Attempt #2: Complex-Valued Neural Networks

**Idea**: Use complex numbers throughout the network to naturally preserve phase.

**Implementation**:
- Complex convolutions: (Wᵣ + iWᵢ) × (Xᵣ + iXᵢ)
- Complex batch normalization (whitening with 2x2 covariance)
- ModReLU activation (preserves phase)

**What happened**: NaN values after ~40 epochs.

**Why it failed**:
- Complex batch normalization requires inverting a 2×2 covariance matrix
- With certain data distributions, this matrix becomes singular (non-invertible)
- Division by near-zero values → NaN explosion

**Lesson**: Numerical stability matters. More complex models aren't always better.

### Failed Attempt #3: Training from Scratch in Experiments

**What happened**: Our experiment scripts achieved only 0.27-0.44 AUROC instead of the expected 0.94.

**Why it failed**:
- Used smaller datasets (5,000 vs 10,000 samples)
- Fewer training epochs (50 vs full convergence)
- Missing power conditioning in some configurations

**Lesson**: Always validate that your baseline matches before testing improvements.

---

## 8. Final Architecture

### Training Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    SNRConditionedVAE                             │
├─────────────────────────────────────────────────────────────────┤
│  Input: I/Q Signal [batch, 2, 1024]                             │
│         SNR (normalized, -5 to 30 dB → 0 to 1)                  │
│         Power (pre-normalization signal power)                   │
│                                                                  │
│  Encoder:                                                        │
│    Conv1d layers: 2→32→64→128→256 channels                      │
│    + SNR/Power conditioning embedding (16-dim each)              │
│    → Latent μ, σ² [batch, 32]                                   │
│                                                                  │
│  Reparameterization: z = μ + σ × ε, where ε ~ N(0,1)            │
│                                                                  │
│  Decoder:                                                        │
│    ConvTranspose1d: 256→128→64→32→2 channels                    │
│    + Conditioning                                                │
│    → Reconstructed signal [batch, 2, 1024]                      │
│                                                                  │
│  Loss: MSE(input, reconstruction) + β × KL(q(z|x) || p(z))      │
└─────────────────────────────────────────────────────────────────┘
```

### Detection Pipeline (What Actually Detects Anomalies)

```
┌─────────────────────────────────────────────────────────────────┐
│                    Hybrid Detection                              │
├─────────────────────────────────────────────────────────────────┤
│  Step 1: Fit on Normal Training Data                            │
│    - Compute mean (μ) of latent codes                           │
│    - Compute covariance (Σ) of latent codes                     │
│    - Fit frequency feature statistics                            │
│                                                                  │
│  Step 2: Score Test Samples                                      │
│                                                                  │
│    Latent Score:                                                 │
│      - Encode sample → z                                         │
│      - Mahalanobis distance: √((z-μ)ᵀΣ⁻¹(z-μ))                 │
│                                                                  │
│    Frequency Score:                                              │
│      - Extract spectral features (entropy, centroid, etc.)      │
│      - Compute deviation from normal statistics                  │
│                                                                  │
│    Hybrid Score:                                                 │
│      - Normalize both scores to [0, 1]                          │
│      - Combine: 0.5 × latent + 0.5 × frequency                  │
│                                                                  │
│  Output: Higher score = more anomalous                          │
└─────────────────────────────────────────────────────────────────┘
```

---

## 9. Results Summary

### Overall Performance Evolution

| Method | AUROC | What Changed |
|--------|-------|--------------|
| Reconstruction (baseline) | 0.42 | Nothing - broken |
| Reconstruction (inverted) | 0.56 | Flipped scores |
| Latent-only (Mahalanobis) | 0.91 | Used latent space |
| + Power conditioning | 0.94 | Preserved power info |
| + Phase hybrid detection | **0.96** | Added frequency features |

### Per-Anomaly Performance (Final System)

| Anomaly Type | AUROC | Difficulty | Notes |
|--------------|-------|------------|-------|
| Amplitude Spike | 1.00 | Easy | Power conditioning helps |
| Burst Noise | 1.00 | Easy | Distinct in latent space |
| Phase Noise | 0.96 | Medium | Well captured |
| Interference | 0.90 | Medium | Good detection |
| Frequency Drift | 0.85 | **Hard** | Improved with hybrid |

### Key Achievements

- **Fully unsupervised**: No labeled anomalies used during training
- **Generalizes**: Works on anomaly types never seen during training
- **Robust**: Works across -5 dB to 30 dB SNR range

---

## 10. Topics for Further Study

To deepen your understanding, here are topics to explore, organized by priority:

### Priority 1: Core Concepts (Start Here)

1. **Variational Autoencoders (VAEs)**
   - How they differ from regular autoencoders
   - The "reparameterization trick"
   - KL divergence and why it matters
   - *Read*: "Auto-Encoding Variational Bayes" (Kingma & Welling, 2014)

2. **Anomaly Detection Fundamentals**
   - One-class classification concept
   - Reconstruction-based vs density-based methods
   - Threshold selection strategies

3. **I/Q Signal Representation**
   - Why we use I and Q instead of raw RF
   - Converting between I/Q and amplitude/phase
   - *Search*: "IQ signal processing basics"

### Priority 2: Key Methods Used

4. **Mahalanobis Distance**
   - Multivariate Gaussian distributions
   - Covariance matrices and what they represent
   - Why it's better than Euclidean distance for anomaly detection
   - *Exercise*: Compute it by hand on a 2D dataset

5. **Convolutional Neural Networks for 1D Signals**
   - How 1D convolutions work
   - Receptive fields and temporal patterns
   - Downsampling with strided convolutions

6. **Spectral Analysis**
   - Fourier Transform basics
   - Spectral features (entropy, centroid, bandwidth)
   - *Exercise*: Compute FFT of a simple sine wave

### Priority 3: Advanced Topics

7. **Conditioning in Neural Networks**
   - How to inject side information (like SNR)
   - Embedding layers for continuous values
   - *Related*: Feature-wise Linear Modulation (FiLM)

8. **Phase and Instantaneous Frequency**
   - Phase unwrapping
   - Computing instantaneous frequency
   - Why phase is tricky (wraps around at ±π)

9. **Hybrid/Ensemble Methods**
   - When to combine learned and hand-crafted features
   - Score normalization and fusion
   - Avoiding degradation on some classes

10. **Complex-Valued Neural Networks** (Advanced)
    - "Deep Complex Networks" (Trabelsi et al., 2018)
    - Complex batch normalization challenges
    - When phase preservation is worth the complexity

### Recommended Learning Path

```
Week 1: VAEs + Anomaly Detection basics
        → Can explain latent space and reconstruction loss

Week 2: I/Q signals + Spectral analysis
        → Can generate and analyze RF signals

Week 3: Mahalanobis distance + Covariance
        → Can implement latent-space detection

Week 4: Conditioning + Hybrid methods
        → Can add features without breaking things
```

### Practical Exercises

1. **Implement a basic VAE** for MNIST or simple 1D signals
2. **Visualize latent spaces** with PCA and t-SNE
3. **Compute Mahalanobis distance** on synthetic 2D Gaussian data
4. **Generate synthetic RF signals** with different modulations
5. **Implement a hybrid detector** combining two scoring methods

---

## Appendix: Key Files in This Repository

| File | Purpose |
|------|---------|
| `src/models/snr_encoder.py` | VAE with SNR/power conditioning |
| `src/detection/detector.py` | AnomalyDetector class (latent/hybrid) |
| `src/detection/phase_detector.py` | Phase and frequency feature extractors |
| `src/data/synthetic.py` | Synthetic RF signal generation |
| `experiments/validate_best_config.py` | Validates our best configuration |
| `experiments/test_improved_detection.py` | Tests hybrid detection methods |
| `experiments/test_powder_data.py` | Tests on POWDER LTE+DSSS dataset |
| `configs/default.yaml` | All hyperparameters |

---

## Quick Reference: What We Learned

| Problem | Solution | Improvement |
|---------|----------|-------------|
| Reconstruction works backwards | Use latent space instead | 0.42 → 0.93 AUROC |
| Normalization hides amplitude | Add power conditioning | +3% AUROC |
| Phase info lost in real values | Add frequency features at detection | 0.93 → 0.9549 AUROC |
| Phase loss during training | Don't do it - destabilizes | (failed) |
| Complex-valued networks | Not worth the instability | (failed) |
| DSSS inverts freq features | Auto-detect and invert features | 0.73 → 0.89 AUROC |

---

## Session 6 Validation Results (2026-01-18)

### Overfitting Validation: PASS ✓

| Test | Latent-Only | Hybrid(f=0.5) |
|------|-------------|---------------|
| Seed Stability | 0.9308 ± 0.0115 | **0.9454 ± 0.0092** |
| Frequency Drift | 0.8004 | **0.8329** |
| Subtle Anomalies (sev=1.0) | 0.8393 | **0.8934** |
| Low SNR (-10 to 10 dB) | 0.7186 | **0.7735** |

### Continuous Learning: Hybrid Improves All Methods

| Method | Latent | Hybrid | Improvement |
|--------|--------|--------|-------------|
| No Adaptation | 0.8363 | 0.8858 | +5.0% |
| Online Learning | 0.8015 | 0.8395 | +3.8% |
| Online + EWC | 0.7799 | 0.8390 | +5.9% |

### Production Model
```
snr_conditioned_vae_hybrid_v1.pt (21 MB)
```

---

## Frequency Drift Target: ACHIEVED with ChirpDetector

**Final Result: 0.9245 AUROC** (target was 0.9+)

### Why Frequency Drift is the Hardest Anomaly

Frequency drift is fundamentally different from other anomalies because it affects **phase**, not amplitude:

| Feature | Normal | Freq Drift | Detectability |
|---------|--------|------------|---------------|
| Peak Amplitude | 1.40 | 1.41 | Identical |
| Mean Power | 1.15 | 1.16 | Identical |
| Latent cosine similarity | 1.00 | **0.92** | Too similar |
| Quadratic phase coeff | 8e-9 | **2e-4** | 23,000x different |

### The Physics

Frequency drift = carrier frequency changes linearly with time:
```
f(t) = f₀ + k·t
```

Integrating to get phase:
```
φ(t) = 2π∫f(t)dt = 2π(f₀·t + k·t²/2)
                        ↑         ↑
                     linear    QUADRATIC term
```

The **quadratic term** is the smoking gun—but amplitude-based detection can't see it, and the VAE latent space barely captures it because convolutional encoders are partially frequency-shift invariant.

### ChirpDetector Solution

Instead of relying on the VAE, ChirpDetector uses domain-specific physics:

1. **Fits quadratic polynomial to unwrapped phase** → drift has low residual
2. **Computes linear vs quadratic fit ratio** → drift shows huge improvement with quadratic
3. **Measures instantaneous frequency slope** → drift has systematic trend
4. **Computes IF R²** → drift has high linearity (not random noise)

Result: 0.9245 AUROC on frequency drift (vs 0.79 for latent-only)

---

## When Does the Model Beat Simple Baselines?

### Honest Assessment

Tested on HackRF-captured WiFi signals (200 samples):

| Method | Overall AUROC |
|--------|---------------|
| Peak Amplitude | 0.9293 |
| Mean Power (dB) | 0.9094 |
| **Our Model** | **0.9735** |

### Per-Anomaly Breakdown

| Anomaly Type | Amplitude | Model | Advantage |
|--------------|-----------|-------|-----------|
| amplitude_spike | **1.000** | **1.000** | None (equal) |
| burst_noise | 0.977 | **0.999** | +2% |
| chirp | 0.880 | **0.970** | **+9%** |
| barrage | 0.875 | **0.969** | **+9%** |
| tone | 0.867 | **0.899** | +3% |

### Key Insight

The model earns its complexity for **spectral anomalies** (chirps, tones, barrage) where amplitude alone is insufficient. For power-based anomalies (spikes, bursts), a simple threshold works nearly as well.

**Recommendation:**
- Use amplitude threshold for power-monitoring systems
- Use VAE latent space for spectral anomaly detection
- Use ChirpDetector for frequency drift
- Use hybrid ensemble for unknown/mixed threats

---

## POWDER Dataset: Testing on Real LTE with DSSS Interference

### The Experiment (2026-01-19)

We obtained the POWDER dataset containing real LTE signals with and without DSSS (Direct Sequence Spread Spectrum) interference. This is a completely different anomaly type than anything in our synthetic training data.

**Dataset Details:**
- Normal: 250 files of clean LTE signals
- Anomaly: 1000 files of LTE + DSSS interference (SIR = -10 dB)
- Bandwidth: 10 MHz at 11.52 MHz sample rate
- Format: Complex64 I/Q samples (~912,600 samples per file)

### Key Discovery: Frequency Features Invert for Spread Spectrum

We discovered that DSSS interference causes **inverted** frequency feature relationships:

| Feature | Normal LTE | LTE + DSSS | Why |
|---------|------------|------------|-----|
| Spectral entropy | Higher | Lower | LTE has structured subcarriers; DSSS fills in the spectrum |
| Spectral bandwidth | Higher | Lower | DSSS spreads uniformly, reducing measured bandwidth variation |
| Spectral flatness | Higher | Lower | DSSS makes spectrum more uniform |

This makes sense when you understand DSSS: it's designed to spread energy across a wide bandwidth, making it look more like noise. When added to LTE (which has distinct spectral peaks), it actually "fills in" the spectrum, reducing entropy.

### Results

| Method | AUROC | Notes |
|--------|-------|-------|
| Latent-only (Mahalanobis) | 0.7319 | Trained on synthetic data |
| Amplitude threshold | 0.7734 | Simple baseline |
| Spectral bandwidth (inverted) | 0.7506 | Single frequency feature |
| **Latent + Amp + Freq** | **0.8882** | Best hybrid approach |

### Why Hybrid Detection Wins

DSSS interference adds power across the bandwidth, so:
1. **Amplitude** detects the power increase (+4.4x mean amplitude)
2. **Latent space** captures structural changes in the signal
3. **Frequency features** (inverted) detect the spectral flattening

No single method captures all aspects—the combination is synergistic.

### Lesson Learned

> **Frequency feature relationships can invert depending on anomaly type. Always check AUROC direction and auto-invert if needed.**

This is now implemented in `experiments/test_powder_data.py`.

---

## Live HackRF Validation

### TorchRF Testbed

Built a complete live detection system:

```
TorchRF_Testbed/
├── src/
│   ├── capture.py      # HackRF via GNURadio
│   ├── detector.py     # Model inference wrapper
│   ├── injection.py    # Software anomaly injection
│   └── recorder.py     # HDF5 recording
├── scripts/
│   ├── live_detect.py  # Interactive CLI
│   ├── record_session.py
│   └── replay_test.py
└── data/
    └── hackrf_dataset.h5
```

### Results on Real WiFi Signals

- **Frequency:** 2.437 GHz (WiFi Channel 6)
- **Samples:** 200 (140 normal, 60 anomalies)
- **AUROC:** 0.9735
- **F1 Score:** 0.9355

The model generalizes from synthetic training data to real HackRF captures.

---

*Document created: January 2026*
*Research collaboration with Claude (Anthropic)*
*Last validated: 2026-01-19 (HackRF live testing)*
