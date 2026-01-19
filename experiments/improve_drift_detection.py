"""Experiment: Improve frequency drift detection to 0.9+ AUROC.

Current: 0.8467 AUROC | Target: 0.9+

The challenge: Frequency drift overlaps with normal signals in latent space
(Mahalanobis ~8 vs ~5.5 for normal). We need specialized features.

Frequency drift characteristics:
- Quadratic phase: phase = π * drift_rate * t² / sample_rate
- Linear instantaneous frequency change (chirp)
- FM index changes over time
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.metrics import roc_auc_score

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.synthetic import SyntheticRFGenerator, AnomalyType
from src.models.snr_encoder import SNRConditionedVAE


class ChirpDetector:
    """Specialized detector for frequency drift (chirp) signals.

    Key insight: Frequency drift creates quadratic phase, which means:
    1. Phase fits a parabola well
    2. Second derivative of phase is approximately constant (non-zero)
    3. FM index changes linearly over time
    """

    def __init__(self):
        self.means: dict[str, float] = {}
        self.stds: dict[str, float] = {}
        self._fitted = False

    def extract_chirp_features(self, iq: NDArray[np.float32]) -> NDArray[np.float32]:
        """Extract features optimized for chirp/frequency drift detection."""
        if iq.ndim == 2:
            iq = iq[np.newaxis, ...]

        batch_size = iq.shape[0]
        seq_len = iq.shape[2]
        n_features = 12
        features = np.zeros((batch_size, n_features), dtype=np.float32)

        for i in range(batch_size):
            # Convert to complex
            complex_sig = iq[i, 0, :] + 1j * iq[i, 1, :]

            # Unwrap phase
            phase = np.unwrap(np.angle(complex_sig))
            t = np.arange(len(phase))

            # Feature 0: Quadratic fit residual
            # For frequency drift, phase fits a quadratic well
            # For normal signals, phase is more linear
            coeffs_quad = np.polyfit(t, phase, 2)
            quad_fit = np.polyval(coeffs_quad, t)
            quad_residual = np.mean((phase - quad_fit) ** 2)
            features[i, 0] = quad_residual

            # Feature 1: Quadratic coefficient magnitude (chirp rate indicator)
            # This directly measures the curvature of phase
            features[i, 1] = np.abs(coeffs_quad[0]) * 1e6  # Scale up

            # Feature 2: Linear vs quadratic fit improvement
            coeffs_lin = np.polyfit(t, phase, 1)
            lin_fit = np.polyval(coeffs_lin, t)
            lin_residual = np.mean((phase - lin_fit) ** 2)
            # Ratio > 1 means quadratic is much better (suggests chirp)
            features[i, 2] = lin_residual / (quad_residual + 1e-10)

            # Instantaneous frequency (derivative of phase)
            inst_freq = np.diff(phase)

            # Feature 3: Linear fit of instantaneous frequency
            # For chirp, inst_freq is linear; for normal, more random
            t_if = np.arange(len(inst_freq))
            coeffs_if = np.polyfit(t_if, inst_freq, 1)
            if_fit = np.polyval(coeffs_if, t_if)
            if_residual = np.mean((inst_freq - if_fit) ** 2)
            features[i, 3] = if_residual

            # Feature 4: Slope of instantaneous frequency (drift rate)
            features[i, 4] = np.abs(coeffs_if[0]) * 1e6

            # Feature 5: Inst freq R² (how linear is the frequency change?)
            ss_res = np.sum((inst_freq - if_fit) ** 2)
            ss_tot = np.sum((inst_freq - np.mean(inst_freq)) ** 2) + 1e-10
            r_squared = 1 - (ss_res / ss_tot)
            features[i, 5] = r_squared

            # Feature 6: Second derivative of phase (frequency acceleration)
            # For chirp, this should be approximately constant
            freq_accel = np.diff(inst_freq)
            features[i, 6] = np.std(freq_accel)  # Low std = constant acceleration

            # Feature 7: Mean of frequency acceleration
            features[i, 7] = np.abs(np.mean(freq_accel)) * 1e6

            # Feature 8: Phase variance (also in EnhancedFrequencyDetector)
            features[i, 8] = np.var(phase)

            # Feature 9: Instantaneous frequency std
            features[i, 9] = np.std(inst_freq)

            # Feature 10: FM index estimation
            # FM index = frequency deviation / modulating frequency
            # For chirp, frequency deviation grows over time
            half = len(inst_freq) // 2
            first_half_std = np.std(inst_freq[:half])
            second_half_std = np.std(inst_freq[half:])
            fm_asymmetry = np.abs(second_half_std - first_half_std) / (first_half_std + 1e-10)
            features[i, 10] = fm_asymmetry

            # Feature 11: Spectral spread over time
            # Split signal, compute spectrum centroid for each segment
            n_segments = 4
            seg_len = seq_len // n_segments
            centroids = []
            for j in range(n_segments):
                seg = complex_sig[j*seg_len:(j+1)*seg_len]
                spec = np.abs(np.fft.fft(seg))
                freqs = np.arange(len(spec))
                centroid = np.sum(spec * freqs) / (np.sum(spec) + 1e-10)
                centroids.append(centroid)
            centroid_drift = np.polyfit(np.arange(n_segments), centroids, 1)[0]
            features[i, 11] = np.abs(centroid_drift)

        return features

    def fit(self, normal_iq: NDArray[np.float32]) -> "ChirpDetector":
        """Fit on normal signals."""
        features = self.extract_chirp_features(normal_iq)

        feature_names = [
            "quad_residual", "quad_coeff", "quad_improvement",
            "if_residual", "if_slope", "if_r_squared",
            "freq_accel_std", "freq_accel_mean", "phase_var",
            "inst_freq_std", "fm_asymmetry", "centroid_drift"
        ]

        for i, name in enumerate(feature_names):
            self.means[name] = float(np.mean(features[:, i]))
            self.stds[name] = float(np.std(features[:, i])) + 1e-8

        self._fitted = True
        return self

    def score(self, iq: NDArray[np.float32]) -> NDArray[np.float32]:
        """Compute chirp anomaly scores."""
        if not self._fitted:
            raise RuntimeError("Detector not fitted.")

        features = self.extract_chirp_features(iq)

        feature_names = [
            "quad_residual", "quad_coeff", "quad_improvement",
            "if_residual", "if_slope", "if_r_squared",
            "freq_accel_std", "freq_accel_mean", "phase_var",
            "inst_freq_std", "fm_asymmetry", "centroid_drift"
        ]

        # Weights optimized for chirp/drift detection
        # Key features: quadratic improvement, if_slope, r_squared, freq_accel
        weights = [
            0.5,  # quad_residual (lower = more parabolic, but this is residual)
            2.0,  # quad_coeff (direct chirp rate)
            3.0,  # quad_improvement (KEY: high = phase is parabolic not linear)
            0.5,  # if_residual
            2.5,  # if_slope (drift rate)
            2.0,  # if_r_squared (high = linear inst_freq = chirp)
            1.0,  # freq_accel_std (low = constant acceleration = chirp)
            1.5,  # freq_accel_mean
            1.0,  # phase_var
            1.5,  # inst_freq_std
            1.5,  # fm_asymmetry
            2.0,  # centroid_drift
        ]

        scores = np.zeros(features.shape[0], dtype=np.float32)
        for i, (name, weight) in enumerate(zip(feature_names, weights)):
            # Note: for some features, high value = anomaly; for others, low = anomaly
            # For simplicity, use deviation from normal mean
            deviation = np.abs(features[:, i] - self.means[name]) / self.stds[name]
            scores += weight * deviation

        return scores / sum(weights)


def evaluate_detector(
    detector_name: str,
    detector,
    train_iq: NDArray,
    test_iq: NDArray,
    test_labels: NDArray,
    test_types: list[str],
    latent_scores: NDArray | None = None,
    freq_weight: float = 0.5,
) -> dict:
    """Evaluate a detector and return results."""
    # Fit detector
    normal_train = train_iq[:]  # All training data is normal
    detector.fit(normal_train)

    # Get detector scores
    if hasattr(detector, 'score'):
        detector_scores = detector.score(test_iq)
    else:
        raise ValueError(f"Detector {detector_name} has no score method")

    # Combine with latent scores if provided
    if latent_scores is not None:
        def normalize(s):
            return (s - s.min()) / (s.max() - s.min() + 1e-8)

        combined_scores = (1 - freq_weight) * normalize(latent_scores) + freq_weight * normalize(detector_scores)
    else:
        combined_scores = detector_scores

    # Calculate AUROC overall
    overall_auroc = roc_auc_score(test_labels, combined_scores)

    # Calculate per-type AUROC
    type_aurocs = {}
    for anom_type in set(test_types):
        if anom_type == "normal":
            continue

        # Create mask for this type vs normal
        mask = np.array([(t == anom_type or t == "normal") for t in test_types])
        type_labels = (np.array(test_types)[mask] == anom_type).astype(int)
        type_scores = combined_scores[mask]

        if len(np.unique(type_labels)) > 1:
            type_aurocs[anom_type] = roc_auc_score(type_labels, type_scores)

    return {
        "detector": detector_name,
        "overall_auroc": overall_auroc,
        "per_type": type_aurocs,
        "frequency_drift_auroc": type_aurocs.get("frequency_drift", 0.0),
    }


def main():
    parser = argparse.ArgumentParser(description="Improve frequency drift detection")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to trained VAE checkpoint")
    parser.add_argument("--num-samples", type=int, default=2000,
                        help="Number of test samples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print("=" * 60)
    print("FREQUENCY DRIFT DETECTION IMPROVEMENT EXPERIMENT")
    print("=" * 60)
    print(f"Target: Improve frequency_drift from 0.8467 to 0.9+ AUROC")
    print(f"Device: {args.device}")
    print(f"Seed: {args.seed}")
    print("=" * 60)

    # Generate data
    print("\nGenerating test data...")
    generator = SyntheticRFGenerator(sequence_length=1024, seed=args.seed)

    # Training data (normal only) - match config settings
    train_iq, train_meta = generator.generate_batch(
        num_samples=2000,  # More training data
        anomaly_ratio=0.0,
        snr_range=(-5, 30),  # Match config
    )

    # Test data - use config's anomaly_severity (4.0 by default)
    # Only frequency_drift anomalies for focused testing
    test_iq, test_meta = generator.generate_batch(
        num_samples=args.num_samples,
        anomaly_ratio=0.1,  # 10% anomaly like original test
        snr_range=(-5, 30),  # Match config
        anomaly_types=["frequency_drift"],  # Only frequency drift
        anomaly_severity=4.0,  # Key: use severity 4.0!
    )

    test_labels = np.array([1 if m.is_anomaly else 0 for m in test_meta])
    test_types = [m.anomaly_type if m.is_anomaly else "normal" for m in test_meta]

    print(f"Training samples: {len(train_iq)}")
    print(f"Test samples: {len(test_iq)} ({sum(test_labels)} anomalies)")

    # Count anomaly types
    type_counts = {}
    for t in test_types:
        type_counts[t] = type_counts.get(t, 0) + 1
    print(f"Anomaly distribution: {type_counts}")

    # Load VAE and get latent scores
    latent_scores = None
    if args.checkpoint:
        print(f"\nLoading VAE from {args.checkpoint}...")
        checkpoint = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        config = checkpoint.get("config", {})

        model = SNRConditionedVAE(
            sequence_length=config.get("sequence_length", 1024),
            latent_dim=config.get("latent_dim", 32),
            hidden_channels=config.get("hidden_channels", [32, 64, 128, 256]),
            snr_embedding_dim=config.get("snr_embedding_dim", 16),
            dropout=config.get("dropout", 0.1),
            use_power_conditioning=config.get("use_power_conditioning", True),
            probabilistic_decoder=config.get("probabilistic_decoder", True),
        ).to(args.device)

        # Trigger lazy initialization with dummy forward pass
        dummy_iq = torch.randn(1, 2, config.get("sequence_length", 1024), device=args.device)
        dummy_snr = torch.rand(1, device=args.device)
        dummy_power = torch.rand(1, device=args.device)
        with torch.no_grad():
            _ = model(dummy_iq, dummy_snr, dummy_power)

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # Get latent scores using Mahalanobis distance
        print("Computing latent scores...")

        # Extract latent representations
        with torch.no_grad():
            # Training data - process in batches
            batch_size = 128
            train_mus = []
            for i in range(0, len(train_iq), batch_size):
                batch = torch.from_numpy(train_iq[i:i+batch_size]).to(args.device)
                snr = torch.zeros(len(batch), device=args.device)
                power = torch.zeros(len(batch), device=args.device)
                # Returns: (x_mean, x_logvar, mu, logvar, z) for probabilistic decoder
                out = model(batch, snr, power)
                mu = out[2]  # mu is the 3rd output
                train_mus.append(mu.cpu().numpy())
            train_mu = np.concatenate(train_mus, axis=0)

            # Test data
            test_mus = []
            for i in range(0, len(test_iq), batch_size):
                batch = torch.from_numpy(test_iq[i:i+batch_size]).to(args.device)
                snr = torch.zeros(len(batch), device=args.device)
                power = torch.zeros(len(batch), device=args.device)
                out = model(batch, snr, power)
                mu = out[2]
                test_mus.append(mu.cpu().numpy())
            test_mu = np.concatenate(test_mus, axis=0)

        # Compute Mahalanobis distance
        # Fit mean and covariance on training (normal) data
        train_mean = np.mean(train_mu, axis=0)
        train_cov = np.cov(train_mu.T) + np.eye(train_mu.shape[1]) * 1e-6

        # Mahalanobis distance for test data
        from scipy.spatial.distance import mahalanobis
        cov_inv = np.linalg.inv(train_cov)
        latent_scores = np.array([
            mahalanobis(test_mu[i], train_mean, cov_inv)
            for i in range(len(test_mu))
        ])

        # Baseline: latent-only
        latent_auroc = roc_auc_score(test_labels, latent_scores)
        print(f"\nLatent-only AUROC: {latent_auroc:.4f}")

        # Per-type latent scores
        for anom_type in set(test_types):
            if anom_type == "normal":
                continue
            mask = np.array([(t == anom_type or t == "normal") for t in test_types])
            type_labels = (np.array(test_types)[mask] == anom_type).astype(int)
            type_scores = latent_scores[mask]
            auroc = roc_auc_score(type_labels, type_scores)
            print(f"  {anom_type}: {auroc:.4f}")

    # Test different detectors
    print("\n" + "=" * 60)
    print("TESTING DETECTORS")
    print("=" * 60)

    # Import existing detectors
    from src.detection.phase_detector import (
        EnhancedFrequencyDetector,
        PhaseAnomalyDetector,
    )

    detectors = [
        ("ChirpDetector (new)", ChirpDetector()),
        ("EnhancedFrequencyDetector", EnhancedFrequencyDetector()),
        ("PhaseAnomalyDetector", PhaseAnomalyDetector()),
    ]

    # Test each detector standalone and combined with latent
    results = []

    for name, detector in detectors:
        print(f"\n{name}:")

        # Standalone
        result_standalone = evaluate_detector(
            f"{name} (standalone)",
            detector,
            train_iq,
            test_iq,
            test_labels,
            test_types,
        )
        print(f"  Standalone: {result_standalone['overall_auroc']:.4f}")
        print(f"    frequency_drift: {result_standalone['frequency_drift_auroc']:.4f}")
        results.append(result_standalone)

        # Combined with latent (if available)
        if latent_scores is not None:
            # Re-fit detector (it was fit in standalone evaluation)
            detector.fit(train_iq)

            for fw in [0.3, 0.5, 0.7]:
                result_combined = evaluate_detector(
                    f"{name} + Latent (fw={fw})",
                    detector,
                    train_iq,
                    test_iq,
                    test_labels,
                    test_types,
                    latent_scores=latent_scores,
                    freq_weight=fw,
                )
                print(f"  + Latent (fw={fw}): {result_combined['overall_auroc']:.4f}")
                print(f"    frequency_drift: {result_combined['frequency_drift_auroc']:.4f}")
                results.append(result_combined)

    # Summary table
    print("\n" + "=" * 60)
    print("SUMMARY: FREQUENCY DRIFT AUROC")
    print("=" * 60)
    print(f"{'Detector':<50} {'Overall':>10} {'Freq Drift':>12}")
    print("-" * 72)

    for r in sorted(results, key=lambda x: x['frequency_drift_auroc'], reverse=True):
        print(f"{r['detector']:<50} {r['overall_auroc']:>10.4f} {r['frequency_drift_auroc']:>12.4f}")

    # Best result
    best = max(results, key=lambda x: x['frequency_drift_auroc'])
    print("\n" + "=" * 60)
    print(f"BEST for frequency_drift: {best['detector']}")
    print(f"  Overall AUROC: {best['overall_auroc']:.4f}")
    print(f"  Frequency Drift AUROC: {best['frequency_drift_auroc']:.4f}")

    if best['frequency_drift_auroc'] >= 0.9:
        print("\n✓ TARGET ACHIEVED: frequency_drift >= 0.9 AUROC!")
    else:
        gap = 0.9 - best['frequency_drift_auroc']
        print(f"\n✗ Target not yet achieved. Gap: {gap:.4f}")

    print("=" * 60)

    return results


if __name__ == "__main__":
    main()
