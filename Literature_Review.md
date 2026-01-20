# Literature Review: VAE Architecture with Mahalanobis Distance for RF Signal Anomaly Detection

**Date:** 2026-01-20
**Topic:** VAE-based anomaly detection using Mahalanobis distance for raw I/Q RF signals
**Researcher:** Claude (Research Assistant)

---

## Executive Summary

This literature review examines the intersection of VAE-based anomaly detection, Mahalanobis distance in latent space, and RF signal processing. The review identifies that this combination represents a **relatively unexplored but highly promising research area**. While VAEs are widely used for anomaly detection across domains, and Mahalanobis distance has been validated as superior to reconstruction error, specific applications to raw I/Q RF signals are sparse in the literature.

**Key Finding:** Our project's approach—latent-only detection achieving 0.93 AUROC vs 0.42 for reconstruction—aligns with recent theoretical work (Bouman & Heskes, 2025) showing that reconstruction-based methods are fundamentally unreliable.

---

## Search Methodology

### Sources Searched
- arXiv.org (cs.LG, eess.SP, stat.ML categories)
- Google Scholar
- IEEE Xplore (via search results)
- Springer Link

### Search Queries Used
1. `site:arxiv.org "variational autoencoder" "Mahalanobis distance" "anomaly detection" 2024 2025`
2. `site:arxiv.org VAE latent space anomaly detection RF signal 2024`
3. `site:arxiv.org "RF anomaly detection" "deep learning" "I/Q signal" 2024 2025`
4. `site:scholar.google.com VAE Mahalanobis distance anomaly detection time series`
5. `site:arxiv.org "spectrum sensing" "autoencoder" anomaly detection 2024`
6. `site:arxiv.org "cognitive radio" "variational autoencoder" signal detection 2024`

---

## Key Papers

### 1. Autoencoders for Anomaly Detection are Unreliable (Bouman & Heskes, 2025)

**Citation:** Bouman, R., & Heskes, T. (2025). Autoencoders for Anomaly Detection are Unreliable. arXiv:2501.13864.

**Link:** [https://arxiv.org/abs/2501.13864](https://arxiv.org/abs/2501.13864)

**Abstract Summary:**
The paper challenges a fundamental assumption in autoencoder-based anomaly detection. The authors prove theoretically and empirically that autoencoders can "perfectly reconstruct" out-of-distribution samples, undermining the core premise that reconstruction error distinguishes normal from anomalous samples.

**Key Contributions:**
- Demonstrates that anomalies lying far from normal data can be perfectly reconstructed in practice
- Revisits linear autoencoder theory, showing how these models can "perfectly reconstruct out of bounds, or extrapolate undesirably"
- Extends findings to both tabular and image datasets
- Highlights dangers for safety-critical applications

**Relevance to Our Work:**
- **DIRECTLY VALIDATES** our empirical discovery that reconstruction error is inverted for anomalies
- Our finding that anomalies have *lower* reconstruction error due to normalization artifacts is consistent with their theoretical analysis
- Strengthens the case for our latent-space Mahalanobis approach over reconstruction-based methods
- Provides theoretical backing for our 0.93 vs 0.42 AUROC comparison

**Citation for Paper:**
> "Recent theoretical analysis demonstrates that reconstruction-based anomaly detection is unreliable [Bouman & Heskes, 2025]. Our latent-space approach, using Mahalanobis distance, avoids this pitfall and achieves 2.2x higher AUROC (0.93 vs 0.42)."

---

### 2. Improved Anomaly Detection through Conditional Latent Space VAE Ensembles (Astrom & Sopasakis, 2024)

**Citation:** Astrom, O., & Sopasakis, A. (2024). Improved Anomaly Detection through Conditional Latent Space VAE Ensembles. arXiv:2410.12328.

**Link:** [https://arxiv.org/abs/2410.12328](https://arxiv.org/abs/2410.12328)

**Abstract Summary:**
Proposes a novel Conditional Latent space Variational Autoencoder (CL-VAE) to perform improved pre-processing for anomaly detection on data with known inlier classes and unknown outlier patterns.

**Key Contributions:**
- Fits unique prior distribution to each class, implementing a Gaussian Mixture Model (GMM) in latent space
- Ensemble of CL-VAEs merged in latent space for group consensus
- Achieves 97.4% AUC on MNIST (vs 95.7% for second-best)
- More interpretable latent space

**Methodology:**
```
Standard VAE: Single Gaussian prior N(0, I)
CL-VAE: GMM prior with K components, one per known class
        p(z) = sum_k pi_k * N(z | mu_k, Sigma_k)
```

**Relevance to Our Work:**
- Our SNR/power conditioning is conceptually similar—conditioning on signal characteristics to improve latent representation
- GMM prior could help separate different anomaly types in latent space
- Ensemble approach could complement our hybrid detection method

**Potential Improvement:**
- [ ] Implement GMM prior in latent space for better separation between signal types
- [ ] Ensemble multiple VAEs trained with different seeds

---

### 3. Trustworthy Intrusion Detection: Confidence Estimation Using Latent Space (Pitsiorlas et al., 2024)

**Citation:** Pitsiorlas, I., Arvanitakis, G., & Kountouris, M. (2024). Trustworthy Intrusion Detection: Confidence Estimation Using Latent Space. arXiv:2409.13774.

**Link:** [https://arxiv.org/abs/2409.13774](https://arxiv.org/abs/2409.13774)

**Abstract Summary:**
Introduces a novel method for enhancing confidence in anomaly detection for Intrusion Detection Systems (IDS) using VAE latent space representations. Develops confidence metrics derived from latent space to gauge trustworthiness of detections.

**Key Contributions:**
- Confidence metric derived from latent space representations
- Correlation of 0.45 between reconstruction error and proposed metric
- Reduces false positives through confidence-aware decision making
- Applied to NSL-KDD network intrusion dataset

**Relevance to Our Work:**
- Demonstrates that latent space contains richer information than reconstruction error
- Confidence estimation could improve our threshold selection
- Network anomaly detection parallels RF anomaly detection conceptually

---

### 4. Deep Predictive Coding Neural Network for RF Anomaly Detection in Wireless Networks (Tandiya et al., 2018)

**Citation:** Tandiya, N., Jauhar, A., Marojevic, V., & Reed, J. H. (2018). Deep Predictive Coding Neural Network for RF Anomaly Detection in Wireless Networks. arXiv:1803.06054.

**Link:** [https://arxiv.org/abs/1803.06054](https://arxiv.org/abs/1803.06054)

**Abstract Summary:**
Proposes anomaly detection for wireless systems based on monitoring and analyzing RF spectrum activities. Leverages video prediction techniques on image sequences generated from wireless spectrum monitoring.

**Key Contributions:**
- Uses time-frequency spectrograms and spectral correlation functions as images
- Deep predictive coding network trained on normal behavior images
- Detects anomalies via deviation between actual and predicted behavior
- Tests on jamming, transmitter chirping, spectrum hijacking, node failure

**Methodology:**
```
RF Signal -> Spectrogram Image -> Predictive Coding Network -> Prediction Error
                                       |
                                  Trained on normal only
```

**Relevance to Our Work:**
- **Closest prior work** to our approach in RF anomaly detection
- Uses image-based (spectrogram) representation vs our raw I/Q
- Our direct I/Q processing preserves phase information that spectrograms lose
- Our approach is more lightweight (no spectrogram computation)

**Comparison:**

| Aspect | Tandiya et al. | Our Approach |
|--------|----------------|--------------|
| Input | Spectrograms | Raw I/Q |
| Phase Info | Lost in conversion | Preserved |
| Model | Predictive coding | VAE |
| Detection | Prediction error | Latent Mahalanobis |
| Complexity | Higher (image processing) | Lower |

---

### 5. Variational Autoencoder for Anomaly Detection: A Comparative Study (Nguyen et al., 2024)

**Citation:** Nguyen, H. H., et al. (2024). Variational Autoencoder for Anomaly Detection: A Comparative Study. arXiv:2408.13561. IEEE ICCE 2024.

**Link:** [https://arxiv.org/abs/2408.13561](https://arxiv.org/abs/2408.13561)

**Abstract Summary:**
Conducts comparative analysis of contemporary VAE architectures for anomaly detection: original VAE, VAE-GRF (Gaussian Random Field prior), and ViT-VAE (Vision Transformer-based VAE).

**Key Findings:**
- **ViT-VAE exhibits exemplary performance** across various scenarios
- VAE-GRF requires more intricate hyperparameter tuning
- Uses MiAD dataset in addition to MVTec for more generalizable results

**ViT-VAE Architecture:**
```
Input -> Patch Embedding -> Transformer Encoder -> Latent -> Transformer Decoder -> Output
         (16x16 patches)   (Self-attention)        (z)      (Cross-attention)
```

**Relevance to Our Work:**
- Transformer-based encoder may better capture long-range frequency dependencies
- Could improve frequency drift detection where our CNN shows limitations (0.80 vs 0.92 with ChirpDetector)
- Self-attention naturally captures global patterns vs CNN's local receptive fields

**Potential Improvement:**
- [ ] Implement ViT-VAE architecture for 1D I/Q signals
- [ ] Compare attention patterns to understand what the model learns

---

### 6. Augmenting Training Data with Vector-Quantized VAE for Classifying RF Signals (Kompella et al., 2024)

**Citation:** Kompella, S. K., Davaslioglu, K., Sagduyu, Y. E., & Kompella, S. (2024). Augmenting Training Data with Vector-Quantized Variational Autoencoder for Classifying RF Signals. arXiv:2410.18283. IEEE MILCOM 2024.

**Link:** [https://arxiv.org/abs/2410.18283](https://arxiv.org/abs/2410.18283)

**Abstract Summary:**
Proposes using VQ-VAE to generate high-fidelity synthetic RF signals for data augmentation, particularly addressing classification under low SNR conditions.

**Key Contributions:**
- VQ-VAE generates synthetic RF signals capturing subtle variations
- Noise injection in latent space for diversity
- Significantly improves classification accuracy at low SNR
- Applied to spectrum management and signal interception

**VQ-VAE Architecture:**
```
Input -> Encoder -> z_e -> Quantize -> z_q -> Decoder -> Output
                      |         |
                      v         v
                   Codebook lookup: z_q = e_k where k = argmin||z_e - e_j||
```

**Key Insight:** Discrete latent codes may be more robust for RF signal representation than continuous Gaussian latents.

**Relevance to Our Work:**
- Could generate diverse synthetic anomalies for training
- VQ-VAE's discrete latents may improve robustness at low SNR
- Validates VAE effectiveness for RF signal representation
- Could help with domain adaptation (synthetic -> real data)

**Potential Improvement:**
- [ ] Use VQ-VAE to augment anomaly training data
- [ ] Compare discrete vs continuous latents for anomaly detection

---

### 7. Time Series Anomaly Detection with Variational Autoencoder Using Mahalanobis Distance (2020)

**Citation:** (Author names not in search results). Time Series Anomaly Detection with Variational Autoencoder Using Mahalanobis Distance. Springer, 2020.

**Link:** [https://link.springer.com/chapter/10.1007/978-3-030-62098-1_4](https://link.springer.com/chapter/10.1007/978-3-030-62098-1_4)

**Abstract Summary:**
Proposes VAE architectures for detecting cyber-attacks on water distribution systems (BATADAL challenge). Examines impact of using Mahalanobis distance as reconstruction error metric.

**Relevance to Our Work:**
- **Directly relevant methodology** - Mahalanobis in VAE for time series
- Different domain (water systems) but similar approach
- Validates Mahalanobis for sequential data anomaly detection

---

### 8. A Real-time Anomaly Detection Using Convolutional Autoencoder with Dynamic Threshold (2024)

**Citation:** arXiv:2404.04311, 2024.

**Link:** [https://arxiv.org/abs/2404.04311](https://arxiv.org/abs/2404.04311)

**Abstract Summary:**
Introduces hybrid modeling combining statistics and Convolutional Autoencoder with dynamic threshold determined by Mahalanobis distance and moving averages.

**Relevance to Our Work:**
- Mahalanobis for adaptive thresholding
- Could improve our SNR-stratified detection thresholds

---

## Comparison to Our Approach

| Aspect | Our Method | Related Work | Gap/Opportunity |
|--------|------------|--------------|-----------------|
| **Detection Method** | Latent Mahalanobis (0.93 AUROC) | Reconstruction error (standard) | We outperform; validated by Bouman 2025 theory |
| **Input Representation** | Raw I/Q [2, 1024] | Spectrograms [Tandiya 2018] | Our approach preserves phase; novel contribution |
| **Conditioning** | SNR + Power embedding | Class-conditional [Astrom 2024] | Similar principle; domain-specific conditioning is novel |
| **Hybrid Detection** | Latent + Frequency features | Single method typical | No direct comparisons in RF domain |
| **Frequency Drift** | ChirpDetector (0.92 AUROC) | Not addressed in VAE literature | **Novel contribution**—phase-based physics detection |
| **RF Focus** | Yes, raw I/Q signals | Minimal VAE+RF literature | **Underexplored area**—publication opportunity |
| **Architecture** | CNN-based VAE | ViT-VAE emerging | Potential improvement direction |

---

## Literature Gaps Identified

1. **VAE + Mahalanobis + RF Signals:** No papers combine all three specifically for RF anomaly detection
2. **Raw I/Q Processing:** Most RF work uses spectrograms; direct I/Q VAE is novel
3. **Frequency Drift Detection:** No VAE-based solutions; our ChirpDetector is novel
4. **Hybrid Detection:** Combining learned + engineered features at inference is rare
5. **Power Conditioning:** Preserving amplitude through conditioning is not addressed elsewhere

---

## Recommendations for Architecture Improvements

### Priority 1: ViT-VAE for 1D Signals (High Impact)

**Rationale:** CNNs have limited receptive fields, causing frequency-shift invariance that hurts drift detection. Transformers naturally capture global patterns.

**Implementation Path:**
1. Adapt ViT architecture for 1D I/Q sequences
2. Patch embedding: [2, 1024] -> patches of [2, 64] x 16 patches
3. Self-attention over patches
4. Compare attention patterns to understand learned features

**Expected Benefit:** Improved frequency drift detection without ChirpDetector (~0.80 -> 0.90+ AUROC)

### Priority 2: VQ-VAE for Data Augmentation (Medium Impact)

**Rationale:** Limited anomaly diversity in training; VQ-VAE can generate diverse, realistic anomalies.

**Implementation Path:**
1. Train VQ-VAE on normal + known anomaly signals
2. Use codebook interpolation to generate novel anomalies
3. Noise injection in latent space for diversity

**Expected Benefit:** Better generalization to unseen anomaly types

### Priority 3: Ensemble of Conditional VAEs (Medium Impact)

**Rationale:** Ensemble methods consistently outperform single models; conditioning improves latent separation.

**Implementation Path:**
1. Train multiple VAEs with different seeds
2. Implement SNR-conditional priors (GMM-style)
3. Merge in latent space for consensus

**Expected Benefit:** More robust detection, reduced variance

---

## Potential Improvements Checklist

From this literature review, the following improvements are recommended:

- [ ] **ViT-VAE architecture** (Nguyen 2024): Transformer-based encoder for long-range dependencies
- [ ] **VQ-VAE for augmentation** (Kompella 2024): Generate diverse synthetic anomalies
- [ ] **Conditional latent priors** (Astrom 2024): GMM prior for better anomaly separation
- [ ] **Ensemble of VAEs** (Astrom 2024): Multiple models merged in latent space
- [ ] **Cite Bouman 2025** in paper: Theoretical backing for reconstruction failure finding
- [ ] **Compare to spectrogram methods** (Tandiya 2018): Establish baseline comparison
- [ ] **Confidence estimation** (Pitsiorlas 2024): Add trustworthiness metrics to detections

---

## Publication Positioning

### Novel Contributions (vs Literature)

1. **First to apply latent-only VAE detection to raw I/Q** (not spectrograms)
2. **Power conditioning** to preserve amplitude information lost in normalization
3. **Hybrid detection** combining latent + physics-based features at inference
4. **ChirpDetector** for frequency drift using phase-physics approach
5. **Validated on real HackRF and POWDER datasets**

### Suggested Venues

| Venue | Focus | Fit |
|-------|-------|-----|
| IEEE TCCN | Cognitive Communications | High - RF focus |
| IEEE TNNLS | Neural Networks | Medium - Method focus |
| ICASSP 2026 | Signal Processing | High - Signal + ML |
| IEEE WCNC | Wireless Communications | Medium - Application focus |

### Key Claims with Evidence

| Claim | Evidence | Supporting Literature |
|-------|----------|----------------------|
| Latent > reconstruction | 0.93 vs 0.42 AUROC | Bouman 2025 (theoretical) |
| Model > amplitude threshold (spectral) | +9% on chirp/barrage | Novel comparison |
| Generalizes to unseen anomalies | 0.9970 AUROC on burst_noise | Standard VAE property |
| ChirpDetector solves frequency drift | 0.92 AUROC (vs 0.79) | Novel - no prior work |
| Validated on real RF | 0.9735 HackRF, 0.8882 POWDER | Novel validation |

---

## References

1. Bouman, R., & Heskes, T. (2025). Autoencoders for Anomaly Detection are Unreliable. arXiv:2501.13864. https://arxiv.org/abs/2501.13864

2. Astrom, O., & Sopasakis, A. (2024). Improved Anomaly Detection through Conditional Latent Space VAE Ensembles. arXiv:2410.12328. https://arxiv.org/abs/2410.12328

3. Pitsiorlas, I., Arvanitakis, G., & Kountouris, M. (2024). Trustworthy Intrusion Detection: Confidence Estimation Using Latent Space. arXiv:2409.13774. https://arxiv.org/abs/2409.13774

4. Tandiya, N., Jauhar, A., Marojevic, V., & Reed, J. H. (2018). Deep Predictive Coding Neural Network for RF Anomaly Detection in Wireless Networks. arXiv:1803.06054. https://arxiv.org/abs/1803.06054

5. Nguyen, H. H., et al. (2024). Variational Autoencoder for Anomaly Detection: A Comparative Study. arXiv:2408.13561. https://arxiv.org/abs/2408.13561

6. Kompella, S. K., Davaslioglu, K., Sagduyu, Y. E., & Kompella, S. (2024). Augmenting Training Data with Vector-Quantized Variational Autoencoder for Classifying RF Signals. arXiv:2410.18283. https://arxiv.org/abs/2410.18283

7. Time Series Anomaly Detection with Variational Autoencoder Using Mahalanobis Distance. (2020). Springer. https://link.springer.com/chapter/10.1007/978-3-030-62098-1_4

8. A Real-time Anomaly Detection Using Convolutional Autoencoder with Dynamic Threshold. (2024). arXiv:2404.04311. https://arxiv.org/abs/2404.04311

---

## Appendix: Detailed Architecture Comparison

### Our Current Architecture: SNRConditionedVAE

**File:** `src/models/snr_encoder.py`

```
Input: I/Q Signal [batch, 2, 1024]
       SNR [batch] (normalized 0-1)
       Power [batch] (optional, normalized)

Encoder:
  ├─ Conv1d layers: 2 → 32 → 64 → 128 → 256 channels
  │   (kernel=7, stride=2, BatchNorm, LeakyReLU, Dropout=0.1)
  ├─ Conditioning: MLP([SNR, Power]) → 16-dim embedding
  ├─ Concat: [flattened_conv, conditioning] → Linear → μ, σ²
  └─ Output: Latent [batch, 32]

Decoder:
  ├─ Linear: [z, conditioning] → 256 × 16
  ├─ ConvTranspose1d: 256 → 128 → 64 → 32 → 2 channels
  └─ Interpolate to 1024 if needed

Loss: MSE + β × KL (β=1.0)
```

**Key Features:**
- SNR + Power conditioning via MLP embedding
- Kernel size 7 (medium receptive field)
- 4 conv layers with stride 2 (16x downsampling)
- Latent dim 32
- Optional: Bayesian last layer, phase loss, probabilistic decoder

### ViT-VAE Architecture (Nguyen et al., 2024)

**Source:** [arXiv:2408.13561](https://arxiv.org/abs/2408.13561)

```
Input: Image [batch, C, H, W]

Encoder:
  ├─ Patch Embedding: Split into 14×14 patches → Linear embedding
  ├─ Position Embedding: Learnable positional encoding
  ├─ Transformer Encoder: Multiple self-attention + FFN layers
  │   └─ Each layer: LayerNorm → MultiHead Attention → Residual → FFN
  ├─ Extract intermediate features (not class token)
  └─ Output: Latent [batch, 384]

Decoder:
  ├─ Conv layers to reconstruct from features
  └─ Output: Reconstructed image

Loss: Standard VAE loss
```

**Key Hyperparameters:**
- Latent dim: 384
- Latent image size: 14
- Epochs: 100
- Batch size: 8
- Learning rate: 1e-4
- β = 1

**Why ViT-VAE May Help Us:**

| Aspect | CNN (Our Method) | ViT-VAE | Impact |
|--------|------------------|---------|--------|
| Receptive field | Local (kernel 7) | Global (self-attention) | ViT sees full sequence |
| Freq drift detection | Limited (0.80 AUROC) | May improve | Global patterns help |
| Position awareness | Implicit via convolution | Explicit positional encoding | Better for drift |
| Computational cost | Lower | Higher | Trade-off |

### VQ-VAE Architecture (Kompella et al., 2024)

**Source:** [arXiv:2410.18283](https://arxiv.org/abs/2410.18283)

```
Input: I/Q Signal [batch, 2, 2048]

Encoder:
  ├─ Conv2d layers: 3 layers + 1 residual stack
  │   (Progressive compression with ReLU)
  ├─ Pre-quantizer Conv layer → reshape to embedding dim
  └─ Output: Continuous latent z_e

Quantization:
  ├─ Codebook: e ∈ ℝ^(K×D), K discrete codes
  ├─ Nearest neighbor: z_q = e_k where k = argmin ||z_e - e_j||₂
  └─ Output: Discrete latent z_q

Decoder:
  ├─ Post-quantizer Conv layer
  ├─ Conv2d Transpose: 3 layers (progressive upsampling)
  └─ Output: Reconstructed signal

Loss:
  ℓ = log p(x|z_q) + ||sg[z_e] - e||² + β||z_e - sg[e]||²

  Where sg = stop-gradient
```

**Key Innovation - Latent Noise Injection:**
```python
# After VQ-VAE training, inject noise before quantization
w ~ N(0, σ_s²)  # σ_s² = 1.5 works best
z_e_noisy = z_e + w
z_q_noisy = quantize(z_e_noisy)  # Different codebook entry
x_synthetic = decode(z_q_noisy)  # New diverse sample
```

**Why VQ-VAE May Help Us:**

| Aspect | Continuous VAE (Our Method) | VQ-VAE | Impact |
|--------|----------------------------|--------|--------|
| Latent space | Gaussian, continuous | Discrete codebook | More structured |
| Low SNR robustness | Good | Better (+15.86% at -10dB) | Significant |
| Data augmentation | Limited | Excellent (noise injection) | More diverse training |
| Anomaly detection | Mahalanobis distance | Codebook usage patterns | Alternative approach |

### Adaptation Plan for 1D I/Q Signals

#### Option 1: ViT-VAE for I/Q (Recommended for frequency drift)

```python
class IQViTVAE(nn.Module):
    """Vision Transformer VAE adapted for 1D I/Q signals."""

    def __init__(
        self,
        seq_length: int = 1024,
        patch_size: int = 64,      # 1024 / 64 = 16 patches
        embed_dim: int = 256,       # Per-patch embedding
        num_heads: int = 8,
        num_layers: int = 6,
        latent_dim: int = 32,
        snr_embedding_dim: int = 16,
    ):
        super().__init__()
        num_patches = seq_length // patch_size  # 16

        # Patch embedding: [batch, 2, 1024] → [batch, 16, 256]
        self.patch_embed = nn.Conv1d(2, embed_dim, patch_size, stride=patch_size)

        # Positional encoding
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        # SNR conditioning (inject into transformer via cross-attention or concat)
        self.snr_embed = nn.Sequential(
            nn.Linear(1, snr_embedding_dim),
            nn.ReLU(),
            nn.Linear(snr_embedding_dim, embed_dim),
        )

        # Latent projection
        self.mu_proj = nn.Linear(num_patches * embed_dim, latent_dim)
        self.logvar_proj = nn.Linear(num_patches * embed_dim, latent_dim)

    def forward(self, x, snr):
        # x: [batch, 2, 1024], snr: [batch]
        patches = self.patch_embed(x).transpose(1, 2)  # [batch, 16, 256]
        patches = patches + self.pos_embed

        # Add SNR as extra token or via cross-attention
        snr_token = self.snr_embed(snr.unsqueeze(1)).unsqueeze(1)  # [batch, 1, 256]
        tokens = torch.cat([snr_token, patches], dim=1)  # [batch, 17, 256]

        h = self.transformer(tokens)
        h = h[:, 1:, :].flatten(1)  # Remove SNR token, flatten

        mu = self.mu_proj(h)
        logvar = self.logvar_proj(h)
        return mu, logvar
```

**Expected Benefits:**
- Self-attention captures global frequency relationships
- May detect frequency drift without ChirpDetector (target: 0.90+ AUROC)
- Position encoding makes model phase-aware

**Training Requirements:**
- Epochs: 100-200 (transformers need more data)
- Batch size: 32-64 (larger for transformer stability)
- Learning rate: 1e-4 with warmup
- GPU: ~4GB VRAM (manageable on cluster)

#### Option 2: VQ-VAE for Data Augmentation

```python
class IQVectorQuantizer(nn.Module):
    """Vector quantization for I/Q signals."""

    def __init__(self, num_embeddings: int = 512, embedding_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)

    def forward(self, z_e):
        # z_e: [batch, embedding_dim, seq_len]
        # Find nearest codebook entry
        z_e_flat = z_e.permute(0, 2, 1).contiguous().view(-1, z_e.size(1))
        distances = torch.cdist(z_e_flat, self.embedding.weight)
        indices = distances.argmin(dim=1)
        z_q_flat = self.embedding(indices)
        z_q = z_q_flat.view(z_e.size(0), z_e.size(2), -1).permute(0, 2, 1)

        # Straight-through estimator
        z_q = z_e + (z_q - z_e).detach()
        return z_q, indices

def generate_augmented_data(vqvae, x, noise_std=1.5):
    """Generate synthetic I/Q signals via latent noise injection."""
    with torch.no_grad():
        z_e = vqvae.encode(x)
        z_e_noisy = z_e + torch.randn_like(z_e) * noise_std
        z_q, _ = vqvae.quantize(z_e_noisy)
        x_synthetic = vqvae.decode(z_q)
    return x_synthetic
```

**Expected Benefits:**
- Generate diverse anomaly types for training
- Improve generalization to unseen anomalies
- Better low-SNR performance (+15% at -10dB reported)

### Implementation Recommendation

**Phase 1 (Immediate): ViT-VAE for Frequency Drift**
1. Implement `IQViTVAE` class
2. Train on cluster with current data
3. Compare frequency drift AUROC: target 0.90+ (vs current 0.80 without ChirpDetector)
4. If successful, may eliminate need for ChirpDetector

**Phase 2 (If Phase 1 succeeds): Hybrid Architecture**
1. Use ViT encoder + CNN decoder (common pattern)
2. Add VQ-VAE augmentation for training data diversity
3. Compare ensemble (ViT-VAE + CNN-VAE) vs single model

**Cluster Training Estimate:**
- ViT-VAE: ~2-4 hours on single GPU (A100)
- Data: Use existing synthetic data (10,000+ samples)
- Validation: HackRF dataset for real-world comparison

### Summary Comparison Table

| Feature | Our CNN-VAE | ViT-VAE | VQ-VAE |
|---------|-------------|---------|--------|
| Receptive field | Local | Global | Local |
| Latent space | Continuous | Continuous | Discrete |
| Freq drift potential | Limited | High | Medium |
| Low SNR robustness | Good | Medium | Excellent |
| Data augmentation | None | None | Built-in |
| Training complexity | Low | Medium | Medium |
| Inference speed | Fast | Medium | Fast |
| **Recommendation** | Baseline | Frequency drift | Augmentation |

---

*Literature review conducted: 2026-01-20*
*Next review recommended: After implementing ViT-VAE architecture*
*Cluster experiment priority: ViT-VAE for frequency drift*
