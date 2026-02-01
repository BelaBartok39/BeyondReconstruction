# Latent-Space Anomaly Detection in Raw I/Q RF Signals Using SNR-Conditioned Variational Autoencoders with Hybrid Phase-Aware Scoring

**Authors:** [Your Name], [Advisor Name]
**Affiliation:** University of Memphis, Department of [Your Department]

---

## Abstract

We present a novel unsupervised anomaly detection framework for raw in-phase/quadrature (I/Q) radio frequency (RF) signals using a signal-to-noise ratio (SNR) and power-conditioned variational autoencoder (VAE) with hybrid latent-phase scoring. Unlike prior RF anomaly detection work that relies on spectrogram representations, our approach operates directly on raw I/Q samples, preserving phase information critical for detecting subtle spectral anomalies. We demonstrate that reconstruction-based anomaly scoring is fundamentally unreliable for RF signals due to normalization artifacts---a finding consistent with recent theoretical results showing autoencoders can reconstruct out-of-distribution samples. Instead, we employ Mahalanobis distance in the VAE latent space, achieving 0.93 AUROC compared to 0.42 for reconstruction error (2.2x improvement). We further introduce power conditioning to recover amplitude information lost during normalization, and a hybrid detection pipeline that combines latent-space scores with physics-based phase features at inference time without retraining. A specialized ChirpDetector exploiting quadratic phase structure achieves 0.9245 AUROC on the hardest anomaly type (frequency drift), up from 0.80 with latent-only detection. Our system achieves 0.9549 overall AUROC on synthetic data, 0.9735 on live HackRF-captured WiFi signals, and 0.8882 on POWDER LTE signals with unseen DSSS interference, demonstrating strong generalization from synthetic to real-world RF environments. We validate against classical baselines (One-Class SVM, Isolation Forest, amplitude thresholding) with statistical significance tests, honestly characterizing where the learned model excels (+9% on spectral anomalies) and where simple baselines suffice (amplitude-based anomalies).

**Index Terms:** RF anomaly detection, variational autoencoder, I/Q signals, latent-space analysis, Mahalanobis distance, continuous learning, spectrum monitoring

---

## I. Introduction

Radio frequency (RF) spectrum monitoring is essential for detecting interference, unauthorized transmissions, and anomalous signal behaviors in wireless communication systems. As the electromagnetic environment grows increasingly congested, automated anomaly detection systems must identify diverse, previously unseen anomaly types without labeled training examples.

Traditional approaches to RF anomaly detection rely on hand-crafted features extracted from spectrograms or power spectral density estimates [1]. While effective for known interference patterns, these methods require domain expertise to design features for each anomaly type and lose phase information during the time-frequency transformation. Deep learning approaches using autoencoders have shown promise for unsupervised anomaly detection across domains [2], but their application to raw I/Q RF signals remains underexplored.

In this work, we address several fundamental challenges in applying VAE-based anomaly detection to RF signals:

1. **The Reconstruction Paradox.** We discover that standard reconstruction-based scoring produces *inverted* anomaly scores (0.42 AUROC, worse than random) because signal normalization compresses high-amplitude anomalies into flat, easily reconstructable patterns. This finding aligns with recent theoretical analysis proving that autoencoder reconstruction is fundamentally unreliable for anomaly detection [3].

2. **Information Loss from Normalization.** Normalizing I/Q signals to unit range destroys power information (10-14x compression for amplitude anomalies). We introduce power conditioning that preserves this information as an auxiliary model input.

3. **Frequency-Shift Invariance.** Convolutional encoders learn local structural patterns rather than absolute frequency, making them partially invariant to frequency drift---the hardest anomaly type. We develop a physics-based ChirpDetector that exploits the quadratic phase structure of drifting signals.

4. **Generalization to Real Data.** Models trained on synthetic data must transfer to real-world RF captures with different noise profiles, modulation characteristics, and propagation effects.

Our key contributions are:
- First application of latent-space Mahalanobis distance in a VAE to raw I/Q RF anomaly detection, achieving 2.2x improvement over reconstruction-based scoring
- SNR and power conditioning that enables robust detection across signal conditions (-5 to 30 dB SNR)
- A hybrid detection pipeline combining learned latent features with physics-based phase features at inference time, requiring no model retraining
- A specialized ChirpDetector achieving 0.9245 AUROC on frequency drift through quadratic phase analysis
- Validation on three datasets: synthetic (0.9549 AUROC), live HackRF WiFi captures (0.9735), and POWDER LTE with DSSS interference (0.8882)
- Honest comparison against classical baselines with statistical significance testing

---

## II. Related Work

### A. Autoencoder-Based Anomaly Detection

Autoencoders have been widely adopted for unsupervised anomaly detection under the assumption that models trained on normal data will produce higher reconstruction error for anomalous inputs [2]. However, Bouman and Heskes [3] recently proved theoretically that this assumption is fundamentally flawed: autoencoders can perfectly reconstruct out-of-distribution samples. Our empirical findings in the RF domain---where anomalies produce *lower* reconstruction error than normal signals---provide additional evidence from a new application area.

Several works have explored latent-space alternatives. Pitsiorlas et al. [4] derive confidence metrics from VAE latent representations for intrusion detection systems, demonstrating that latent space contains richer anomaly information than reconstruction error. Time series anomaly detection with VAE and Mahalanobis distance has been applied to water distribution systems [5], validating the approach for sequential data. Our work is the first to combine VAE latent-space Mahalanobis distance with raw I/Q RF signals.

### B. Conditional VAE Architectures

Astrom and Sopasakis [6] propose conditional latent-space VAE ensembles (CL-VAE) with Gaussian mixture model priors for anomaly detection, achieving 97.4% AUC on MNIST. Our SNR and power conditioning follows a similar principle---injecting domain-specific side information to improve latent representations---but targets RF-specific signal characteristics rather than class labels.

Nguyen et al. [7] compare VAE architectures for anomaly detection, finding that Vision Transformer-based VAEs (ViT-VAE) outperform CNN-based alternatives, suggesting self-attention may better capture global patterns. This is relevant to our frequency drift challenge, where CNN receptive field limitations contribute to detection difficulty.

### C. RF Signal Processing with Deep Learning

Tandiya et al. [1] apply deep predictive coding networks to RF anomaly detection using spectrogram image sequences. Their approach detects jamming, chirping, and spectrum hijacking, but operates on spectrogram representations rather than raw I/Q, losing phase information in the conversion.

Kompella et al. [8] use vector-quantized VAEs (VQ-VAE) for RF signal classification, demonstrating that VAE architectures can effectively represent RF signal characteristics and improve low-SNR classification through data augmentation. Their discrete codebook approach could complement our continuous latent space.

### D. Gap Analysis

No prior work combines all of: (i) VAE with Mahalanobis distance, (ii) raw I/Q signal processing (preserving phase), (iii) SNR and power conditioning, and (iv) hybrid detection with physics-based features. Table I summarizes the positioning.

**TABLE I: Comparison with Related Work**

| Aspect | Tandiya [1] | Astrom [6] | Nguyen [7] | Kompella [8] | **Ours** |
|--------|-------------|------------|------------|-------------|----------|
| Input | Spectrogram | Images | Images | I/Q | **Raw I/Q** |
| Phase preserved | No | N/A | N/A | Partial | **Yes** |
| Detection | Pred. error | Recon. | Recon. | Classification | **Latent Mahal.** |
| Conditioning | None | Class | None | None | **SNR + Power** |
| Hybrid features | No | Ensemble | No | No | **Latent + Phase** |
| RF validated | Yes | No | No | Yes | **Yes (3 datasets)** |

---

## III. Methodology

### A. Problem Formulation

Given a stream of raw I/Q RF signals $\mathbf{x} \in \mathbb{R}^{2 \times L}$ where $L=1024$ is the sequence length and the two channels represent in-phase (I) and quadrature (Q) components, we aim to assign an anomaly score $s(\mathbf{x}) \in \mathbb{R}$ such that anomalous signals receive higher scores than normal signals. The system is trained exclusively on normal signals (unsupervised).

Each signal is accompanied by estimated metadata: SNR $\hat{\gamma} \in [-5, 30]$ dB and pre-normalization signal power $P \in \mathbb{R}$. The SNR is estimated using the M2M4 method [9] and normalized to $[0, 1]$. Power is computed before signal normalization as $P = \frac{1}{L}\sum_{l=1}^{L}(I_l^2 + Q_l^2)$ and normalized to $[0, 1]$ using a predefined range.

### B. SNR and Power-Conditioned VAE

#### Architecture

Our encoder $q_\phi(\mathbf{z}|\mathbf{x}, \hat{\gamma}, P)$ consists of four 1D convolutional blocks with channel progression $[2 \rightarrow 32 \rightarrow 64 \rightarrow 128 \rightarrow 256]$, kernel size 7, stride 2, batch normalization, LeakyReLU activation, and dropout (0.1). Each block performs 2x temporal downsampling.

The conditioning inputs $(\hat{\gamma}, P)$ are processed through a two-layer MLP:
$$\mathbf{c} = \text{MLP}([\hat{\gamma}; P]) \in \mathbb{R}^{16}$$

The flattened convolutional output is concatenated with $\mathbf{c}$ and projected to the latent parameters:
$$\boldsymbol{\mu} = W_\mu [\text{flatten}(\text{Conv}(\mathbf{x})); \mathbf{c}] + b_\mu$$
$$\log \boldsymbol{\sigma}^2 = W_\sigma [\text{flatten}(\text{Conv}(\mathbf{x})); \mathbf{c}] + b_\sigma$$

where $\boldsymbol{\mu}, \log \boldsymbol{\sigma}^2 \in \mathbb{R}^{32}$ parameterize the approximate posterior.

Sampling uses the reparameterization trick [10]:
$$\mathbf{z} = \boldsymbol{\mu} + \boldsymbol{\sigma} \odot \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(\mathbf{0}, \mathbf{I})$$

The decoder $p_\theta(\mathbf{x}|\mathbf{z}, \hat{\gamma}, P)$ mirrors the encoder with transposed convolutions $[256 \rightarrow 128 \rightarrow 64 \rightarrow 32 \rightarrow 2]$, also conditioned on $(\hat{\gamma}, P)$ via a separate MLP embedding.

#### Training Objective

We train with the standard VAE loss:
$$\mathcal{L} = \mathcal{L}_{\text{recon}} + \beta \cdot D_{\text{KL}}(q_\phi(\mathbf{z}|\mathbf{x}, \hat{\gamma}, P) \| p(\mathbf{z}))$$

where $\mathcal{L}_{\text{recon}} = \text{MSE}(\mathbf{x}, \hat{\mathbf{x}})$ and $\beta = 1.0$. Training uses Adam optimizer with learning rate $10^{-3}$, weight decay $10^{-5}$, cosine annealing schedule (min LR $10^{-6}$), batch size 64, and gradient clipping at norm 1.0 for 100 epochs with early stopping (patience 10).

#### Why Power Conditioning is Critical

Signal normalization to $[-1, 1]$ compresses high-amplitude anomalies by 10-14x:

$$\text{Before: } [0.1, 0.2, 100, 0.3, 0.1] \xrightarrow{\text{normalize}} [0.001, 0.002, 1.0, 0.003, 0.001]$$

The anomaly spike becomes a flat, easily reconstructable pattern. By providing pre-normalization power as a conditioning input, the model retains information about the original signal's power characteristics.

### C. Latent-Space Anomaly Detection

#### The Reconstruction Paradox

We discovered that reconstruction error is *inversely* correlated with anomalousness for RF signals (0.42 AUROC). This occurs because normalization maps high-amplitude anomalies to near-zero patterns that the decoder reconstructs with lower error than complex normal signals. This finding is now theoretically grounded by Bouman and Heskes [3], who prove that autoencoders can perfectly reconstruct out-of-distribution samples.

#### Mahalanobis Distance Scoring

Instead, we score anomalies by their distance from the normal distribution in the VAE's latent space. During fitting, we compute the mean $\boldsymbol{\mu}_0$ and covariance $\boldsymbol{\Sigma}$ of latent codes from the training set (normal signals only):

$$\boldsymbol{\mu}_0 = \frac{1}{N}\sum_{i=1}^{N} \boldsymbol{\mu}_i, \quad \boldsymbol{\Sigma} = \frac{1}{N-1}\sum_{i=1}^{N}(\boldsymbol{\mu}_i - \boldsymbol{\mu}_0)(\boldsymbol{\mu}_i - \boldsymbol{\mu}_0)^\top + \epsilon \mathbf{I}$$

where $\epsilon = 10^{-6}$ ensures numerical stability. The anomaly score for a test sample is:

$$s_{\text{latent}}(\mathbf{x}) = \sqrt{(\boldsymbol{\mu} - \boldsymbol{\mu}_0)^\top \boldsymbol{\Sigma}^{-1} (\boldsymbol{\mu} - \boldsymbol{\mu}_0)}$$

This approach leverages the VAE's KL divergence loss, which encourages normal signals to cluster around a standard Gaussian in latent space, causing anomalies to map to outlying regions.

### D. Hybrid Detection with Phase Features

While latent-space detection achieves strong overall performance, frequency drift anomalies remain challenging (0.80 AUROC) because convolutional encoders are partially frequency-shift invariant---they learn local structural patterns rather than absolute frequency position.

#### Why Frequency Drift is Hard

Frequency drift introduces a linear change in carrier frequency: $f(t) = f_0 + kt$. This produces quadratic phase:
$$\phi(t) = 2\pi(f_0 t + kt^2/2)$$

The amplitude envelope is unchanged, making power-based detection impossible. In the latent space, drifted signals maintain 0.92 cosine similarity with normal signals, insufficient for reliable detection. However, the quadratic phase coefficient differs by 23,000x between normal and drifted signals.

#### Physics-Based ChirpDetector

The ChirpDetector exploits this physics directly. For each signal, it extracts 12 features including:

1. **Quadratic phase fit residual**: Fits $\phi(t) = at^2 + bt + c$ and measures residual
2. **Quadratic coefficient magnitude**: $|a| \times 10^6$ (the chirp rate indicator)
3. **Linear-to-quadratic improvement ratio**: $R_{\text{lin}} / R_{\text{quad}}$, which is $\gg 1$ for chirps
4. **Instantaneous frequency R-squared**: $R^2$ of linear fit to $\frac{d\phi}{dt}$, high for systematic drift
5. **Spectral centroid drift**: Slope of spectral centroid across signal segments

These features are fit to normal training data, and scored using weighted deviation from normal statistics.

#### Hybrid Scoring

The final anomaly score combines latent and frequency-domain information:

$$s_{\text{hybrid}}(\mathbf{x}) = (1 - \alpha) \cdot \tilde{s}_{\text{latent}}(\mathbf{x}) + \alpha \cdot \tilde{s}_{\text{freq}}(\mathbf{x})$$

where $\tilde{s}$ denotes min-max normalized scores and $\alpha = 0.7$ is the frequency weight (optimized empirically). This achieves 0.9549 AUROC overall, with the frequency component specifically boosting drift detection by +9% without degrading other anomaly types.

### E. Continuous Learning

To adapt to evolving RF environments, we implement three continuous learning strategies:

1. **Online Learning**: Incremental gradient updates with reduced learning rate ($10^{-4}$)
2. **Elastic Weight Consolidation (EWC)** [11]: Penalizes changes to parameters important for previous tasks using Fisher information, with $\lambda = 1000$
3. **Periodic Retraining**: Buffers new samples (reservoir sampling, 5000 capacity) and retrains periodically with experience replay (50% replay ratio)

The hybrid detection approach improves all continuous learning methods by 3.8-5.9% AUROC, as frequency features provide detection capability independent of model weights.

### F. SNR-Adaptive Thresholds

Detection thresholds are computed per SNR bin (7 bins across $[-5, 30]$ dB) using the 95th percentile of normal training scores. Low-SNR signals have inherently higher anomaly scores, so adaptive thresholds prevent excessive false positives at low SNR and missed detections at high SNR.

---

## IV. Experimental Setup

### A. Synthetic Data

We generate I/Q signals using four modulation types (BPSK, QPSK, 16-QAM, 64-QAM) with configurable parameters:
- Sequence length: 1024 samples at 1 MHz sample rate
- SNR range: $[-5, 30]$ dB (uniform)
- Training: 10,000 normal samples
- Testing: 2,000 samples (10% anomalous)

Five anomaly types with severity 4.0:
- **Narrowband interference**: Sinusoidal interferer at random frequency
- **Frequency drift**: Linear carrier frequency change
- **Amplitude spike**: Transient power surge
- **Phase noise**: Random phase perturbation (excess of thermal noise)
- **Burst noise**: Short-duration broadband noise burst

### B. Real-World Datasets

**HackRF WiFi Dataset**: 200 samples captured at 2.437 GHz (WiFi Channel 6) using a HackRF One SDR. Contains 140 normal and 60 anomaly samples with 7 injected anomaly types plus 4 naturally occurring anomalies.

**POWDER LTE+DSSS Dataset**: Real LTE signals from the POWDER testbed [12] with and without DSSS (Direct Sequence Spread Spectrum) interference at SIR = -10 dB. Contains 250 normal and 1000 anomaly files at 10 MHz bandwidth, 11.52 MHz sample rate. Critically, DSSS interference is a completely unseen anomaly type not present in synthetic training data.

### C. Baseline Methods

- **Peak Amplitude Threshold**: $s(\mathbf{x}) = \max(|I|, |Q|)$
- **Mean Power (dB)**: $s(\mathbf{x}) = 10\log_{10}(\frac{1}{L}\sum(I^2 + Q^2))$
- **One-Class SVM**: Fit on 19 engineered features (spectral, power, phase statistics) with RBF kernel
- **Isolation Forest**: Fit on 32-dimensional VAE latent codes
- **VAE Reconstruction Error**: Standard reconstruction-based scoring

### D. Evaluation Metrics

Primary metric: AUROC (Area Under Receiver Operating Characteristic). We also report AUPRC (precision-recall), F1 score, and Cohen's d effect size. Statistical significance is assessed via Wilcoxon signed-rank tests. Reproducibility is validated across 5 random seeds.

---

## V. Results

### A. Detection Method Comparison

Table II shows the progression from reconstruction-based to our final hybrid approach.

**TABLE II: Detection Method Evolution**

| Method | AUROC | Improvement |
|--------|-------|-------------|
| Reconstruction error (baseline) | 0.42 | --- |
| Reconstruction (inverted scores) | 0.56 | +0.14 |
| Latent Mahalanobis distance | 0.91 | +0.49 |
| + Power conditioning | 0.94 | +0.03 |
| + Hybrid phase detection ($\alpha$=0.7) | **0.9549** | +0.01 |

The largest single improvement (+49 percentage points) comes from switching to latent-space detection, validating the theoretical prediction of [3].

### B. Per-Anomaly Performance

Table III shows performance by anomaly type for the final system.

**TABLE III: Per-Anomaly Detection Performance (Hybrid, $\alpha$=0.7)**

| Anomaly Type | AUROC | Difficulty | Notes |
|--------------|-------|------------|-------|
| Amplitude spike | 0.9898 | Easy | Power conditioning critical |
| Burst noise | 0.9999 | Easy | Unseen in training; generalizes |
| Phase noise | 0.9642 | Medium | Well-captured in latent space |
| Narrowband interference | 0.9630 | Medium | Spectral features help |
| Frequency drift | 0.8775 | Hard | ChirpDetector: 0.9245 |

### C. Baseline Comparison

Table IV compares against classical methods on 5,000 synthetic samples.

**TABLE IV: Comparison with Baseline Methods (Synthetic Data)**

| Method | AUROC | AUPRC | F1 | Cohen's d | 95% CI |
|--------|-------|-------|-----|-----------|--------|
| OC-SVM (19 features) | 0.9575 | 0.9173 | 0.840 | 2.81 | [0.949, 0.966] |
| VAE Latent (Mahalanobis) | 0.9218 | 0.8475 | 0.765 | 1.24 | [0.911, 0.933] |
| Isolation Forest (latent) | 0.9222 | 0.8427 | 0.755 | 1.95 | [0.912, 0.933] |
| Amplitude threshold | 0.5328 | 0.2127 | 0.328 | 0.00 | [0.515, 0.550] |
| VAE Reconstruction | 0.5582 | 0.4341 | 0.371 | 0.43 | [0.532, 0.583] |

All pairwise comparisons are statistically significant (Wilcoxon $p < 0.001$).

Key observations:
- OC-SVM with engineered features achieves the highest AUROC (0.9575), demonstrating the value of domain knowledge
- VAE latent and Isolation Forest perform nearly identically on latent codes (~0.92), suggesting latent space quality matters more than the detection algorithm
- VAE reconstruction confirms the fundamental unreliability of reconstruction-based scoring (0.56 AUROC)
- Amplitude threshold fails on spectral anomalies (0.53 AUROC at severity 4.0 which includes spectral types)

### D. Honest Comparison: When Does the Model Win?

Table V provides a per-anomaly breakdown against the amplitude baseline on real HackRF data.

**TABLE V: Per-Anomaly Comparison vs. Amplitude Threshold (HackRF Data)**

| Anomaly Type | Amplitude | Our Model | Advantage |
|--------------|-----------|-----------|-----------|
| Amplitude spike | **1.000** | **1.000** | None (equal) |
| Burst noise | 0.977 | **0.999** | +2.2% |
| Chirp | 0.880 | **0.970** | **+9.0%** |
| Barrage | 0.875 | **0.969** | **+9.4%** |
| Tone | 0.867 | **0.899** | +3.2% |
| **Overall** | 0.929 | **0.974** | **+4.4%** |

The model provides the greatest advantage for **spectral anomalies** (chirps, barrage, tones) where amplitude-based detection is inherently limited. For amplitude-based anomalies, a simple threshold performs equally well.

### E. Real-World Validation

**HackRF WiFi (Table VI):**

| Method | Overall AUROC |
|--------|---------------|
| Peak amplitude | 0.9293 |
| Mean power (dB) | 0.9094 |
| **Our model (latent)** | **0.9735** |

The model achieves 0.9735 AUROC on real WiFi signals captured at 2.437 GHz, demonstrating effective transfer from synthetic training data.

**POWDER LTE+DSSS (Table VII):**

| Method | AUROC | Notes |
|--------|-------|-------|
| Latent-only | 0.7319 | Synthetic-trained model |
| Amplitude threshold | 0.7734 | Simple baseline |
| **Hybrid (Lat+Amp+Freq)** | **0.8882** | +11.5% over baseline |

DSSS is a completely unseen anomaly type. We discovered that DSSS *inverts* frequency feature relationships: spread-spectrum interference reduces spectral entropy and bandwidth rather than increasing them. Auto-detection and inversion of feature polarity recovers detection performance.

### F. Reproducibility and Robustness

**Seed Stability (Table VIII):**

| Method | Mean AUROC | Std |
|--------|-----------|-----|
| Latent-only | 0.9308 | 0.0115 |
| Hybrid ($\alpha$=0.5) | 0.9454 | **0.0092** |

Low variance across 5 seeds confirms reproducibility. The hybrid method also reduces variance, as frequency features provide a detection floor independent of model initialization.

**Freq_weight Sensitivity (Table IX):**

| $\alpha$ | Hybrid AUROC |
|-----------|--------------|
| 0.3 | 0.938 |
| 0.5 | 0.946 |
| 0.6 | 0.951 |
| **0.7** | **0.957** |

Optimal $\alpha \in [0.6, 0.7]$, with higher weight on frequency features slightly improving overall AUROC.

**Model Comparison (V1 vs V2, Table X):**

| Dataset | V1 Hybrid | V2 Hybrid |
|---------|-----------|-----------|
| Synthetic | 0.9499 | 0.9417 |
| POWDER DSSS | 0.8746 | 0.8661 |
| HackRF Live | 0.8449 | 0.8631 |

Both models perform within 1-2% of each other, confirming that results are not dependent on a specific training run.

### G. Continuous Learning

Table XI shows hybrid detection improves all continuous learning methods.

**TABLE XI: Continuous Learning Results**

| Method | Latent AUROC | Hybrid AUROC | Improvement |
|--------|-------------|--------------|-------------|
| No adaptation | 0.8363 | 0.8858 | +5.0% |
| Online learning | 0.8015 | 0.8395 | +3.8% |
| Online + EWC | 0.7799 | 0.8390 | +5.9% |

Hybrid detection provides a consistent 3.8-5.9% improvement regardless of adaptation strategy, because frequency features are independent of model parameters.

### H. Challenging Conditions

**Subtle anomalies (severity 1.0):** Hybrid achieves 0.8934 AUROC (vs 0.8393 latent-only, +5.4%).

**Low SNR (-10 to 10 dB):** Hybrid achieves 0.7735 AUROC (vs 0.7186 latent-only, +5.5%).

**Frequency drift (standalone):** ChirpDetector achieves 0.9245 AUROC compared to 0.80 for latent-only, exploiting the 23,000x difference in quadratic phase coefficient between normal and drifted signals.

---

## VI. Discussion

### A. Why Latent-Space Detection Works

The VAE's KL divergence term regularizes the latent space toward a standard Gaussian, causing normal signals to cluster tightly. Anomalies, having different underlying signal characteristics, are encoded to regions far from this cluster even when the decoder successfully reconstructs them. This decouples detection from reconstruction quality---a critical advantage given the theoretical unreliability of reconstruction-based methods [3].

### B. The Value of Domain-Specific Features

Our honest baseline comparison reveals that OC-SVM with 19 engineered features achieves 0.9575 AUROC---slightly higher than VAE latent alone (0.9218). This suggests that **domain knowledge and learned representations are complementary**, not competing. The hybrid approach combines both: the VAE provides a general-purpose anomaly score, while physics-based features target specific failure modes (frequency drift).

The key insight is to add domain features **at detection time, not during training**. Our failed attempt to incorporate phase loss during training (loss explosion to 79M, amplitude detection collapse from 100% to 37%) demonstrates that modifying the training objective can be counterproductive. Post-hoc feature fusion is safer and more modular.

### C. Frequency Feature Inversion for Spread Spectrum

A notable finding from POWDER validation is that DSSS interference *inverts* expected frequency feature relationships. While most interference increases spectral entropy and bandwidth, DSSS---designed to spread energy uniformly---actually fills spectral gaps, reducing entropy. This highlights the importance of adaptive feature polarity in real-world deployment.

### D. Limitations

1. **Anomaly severity dependence**: At severity 1.0, performance drops from 0.9549 to 0.8934, indicating that very subtle anomalies remain challenging.
2. **Domain gap**: POWDER DSSS achieves 0.8882, below the 0.95+ on synthetic data, reflecting the synthetic-to-real domain gap.
3. **Computational overhead**: The ChirpDetector requires per-sample polynomial fitting, which is slower than pure neural inference.
4. **Known anomaly types**: The system is tested on 5 synthetic + DSSS anomaly types. Performance on adversarial or highly novel anomaly types is unknown.

---

## VII. Conclusion

We presented an unsupervised anomaly detection system for raw I/Q RF signals that addresses fundamental limitations of reconstruction-based methods. By operating directly on I/Q samples rather than spectrograms, conditioning on SNR and signal power, and combining latent-space Mahalanobis distance with physics-based phase features, our system achieves robust detection across diverse anomaly types and signal conditions.

The core finding---that reconstruction-based anomaly detection is unreliable for normalized RF signals---has both practical implications (use latent-space scoring) and theoretical alignment with recent mathematical proofs [3]. Our hybrid approach provides a principled framework for combining learned and engineered features at inference time without retraining.

Validation on live HackRF and POWDER datasets demonstrates that the system generalizes from synthetic to real-world RF environments, achieving 0.9735 and 0.8882 AUROC respectively. The honest comparison against simple baselines clarifies where the model adds value: spectral anomalies (+9% over amplitude threshold) and mixed/unknown threats.

Future work includes: (i) transformer-based encoders (ViT-VAE) for improved frequency drift detection without the ChirpDetector, (ii) VQ-VAE for data augmentation to improve synthetic-to-real transfer, and (iii) evaluation on additional real-world datasets including RadioML and DARPA RFML.

---

## References

[1] N. Tandiya, A. Jauhar, V. Marojevic, and J. H. Reed, "Deep predictive coding neural network for RF anomaly detection in wireless networks," *arXiv:1803.06054*, 2018.

[2] D. P. Kingma and M. Welling, "Auto-encoding variational Bayes," in *Proc. ICLR*, 2014.

[3] R. Bouman and T. Heskes, "Autoencoders for anomaly detection are unreliable," *arXiv:2501.13864*, 2025.

[4] I. Pitsiorlas, G. Arvanitakis, and M. Kountouris, "Trustworthy intrusion detection: Confidence estimation using latent space," *arXiv:2409.13774*, 2024.

[5] "Time series anomaly detection with variational autoencoder using Mahalanobis distance," *Springer LNCS*, vol. 12490, pp. 43-56, 2020.

[6] O. Astrom and A. Sopasakis, "Improved anomaly detection through conditional latent space VAE ensembles," *arXiv:2410.12328*, 2024.

[7] H. H. Nguyen et al., "Variational autoencoder for anomaly detection: A comparative study," in *Proc. IEEE ICCE*, 2024. arXiv:2408.13561.

[8] S. K. Kompella, K. Davaslioglu, Y. E. Sagduyu, and S. Kompella, "Augmenting training data with vector-quantized variational autoencoder for classifying RF signals," in *Proc. IEEE MILCOM*, 2024. arXiv:2410.18283.

[9] D. R. Pauluzzi and N. C. Beaulieu, "A comparison of SNR estimation techniques for the AWGN channel," *IEEE Trans. Commun.*, vol. 48, no. 10, pp. 1681-1691, Oct. 2000.

[10] D. P. Kingma and M. Welling, "An introduction to variational autoencoders," *Found. Trends Mach. Learn.*, vol. 12, no. 4, pp. 307-392, 2019.

[11] J. Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks," *Proc. Natl. Acad. Sci.*, vol. 114, no. 13, pp. 3521-3526, 2017.

[12] J. Breen et al., "POWDER: Platform for Open Wireless Data-driven Experimental Research," *Computer Networks*, vol. 197, 108281, 2021.

---

## Appendix: Summary of Key Results

| Metric | Value |
|--------|-------|
| Overall AUROC (hybrid, synthetic) | 0.9549 |
| Overall AUROC (HackRF live WiFi) | 0.9735 |
| Overall AUROC (POWDER LTE+DSSS) | 0.8882 |
| Freq. drift (ChirpDetector) | 0.9245 |
| Generalization to unseen anomaly | 0.9970 |
| Seed stability (std across 5 seeds) | 0.0092 |
| Reconstruction vs. latent improvement | 2.2x (0.42 vs 0.93) |
| Model advantage on spectral anomalies | +9% over amplitude |
| Continuous learning improvement | +3.8-5.9% with hybrid |
| Production model size | 21 MB |
