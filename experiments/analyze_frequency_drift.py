#!/usr/bin/env python3
"""Analyze why frequency_drift detection is weaker (0.8004 AUROC).

Investigations:
1. Latent space visualization - where do freq drift anomalies fall?
2. Compare latent distributions across anomaly types
3. Check if frequency drift is closer to normal in latent space
4. Test different severity levels for frequency drift
5. Analyze spectral characteristics
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.data.synthetic import SyntheticRFGenerator
from src.data.datasets import RFDataset
from src.models.snr_encoder import create_model
from src.detection.detector import AnomalyDetector
from src.detection.metrics import compute_metrics


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(checkpoint_path: str, config, device: torch.device):
    """Load model with lazy layer initialization."""
    model = create_model(config)
    model = model.to(device)

    # Initialize lazy layers
    dummy_iq = torch.randn(1, 2, config.data.sequence_length, device=device)
    dummy_snr = torch.rand(1, device=device)
    dummy_power = torch.rand(1, device=device)
    with torch.no_grad():
        _ = model(dummy_iq, dummy_snr, dummy_power)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def extract_latent_by_type(model, generator, config, device, anomaly_type, num_samples=500):
    """Extract latent representations for a specific anomaly type."""
    # Generate anomaly samples
    dataset = RFDataset.from_generator(
        generator=generator,
        num_samples=num_samples,
        anomaly_ratio=1.0,  # All anomalies
        snr_range=tuple(config.data.snr_range),
        anomaly_types=[anomaly_type],
        anomaly_severity=config.data.anomaly_severity,
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    model.eval()
    all_latents = []
    all_snrs = []

    with torch.no_grad():
        for batch in loader:
            iq = batch["iq"].to(device)
            snr = batch.get("snr")
            if snr is not None:
                snr = snr.to(device)
            power = batch.get("power")
            if power is not None:
                power = power.to(device)

            mu, _ = model.encoder(iq, snr, power)
            all_latents.append(mu.cpu().numpy())
            if batch.get("snr_db") is not None:
                all_snrs.append(batch["snr_db"].numpy())

    latents = np.concatenate(all_latents)
    snrs = np.concatenate(all_snrs) if all_snrs else None
    return latents, snrs


def extract_normal_latent(model, generator, config, device, num_samples=1000):
    """Extract latent representations for normal samples."""
    dataset = RFDataset.from_generator(
        generator=generator,
        num_samples=num_samples,
        anomaly_ratio=0.0,
        snr_range=tuple(config.data.snr_range),
    )
    loader = DataLoader(dataset, batch_size=64, shuffle=False)

    model.eval()
    all_latents = []

    with torch.no_grad():
        for batch in loader:
            iq = batch["iq"].to(device)
            snr = batch.get("snr")
            if snr is not None:
                snr = snr.to(device)
            power = batch.get("power")
            if power is not None:
                power = power.to(device)

            mu, _ = model.encoder(iq, snr, power)
            all_latents.append(mu.cpu().numpy())

    return np.concatenate(all_latents)


def compute_mahalanobis_distance(latents, mean, cov_inv):
    """Compute Mahalanobis distance from training distribution."""
    diff = latents - mean
    left = np.dot(diff, cov_inv)
    distances = np.sum(left * diff, axis=1)
    return np.sqrt(distances)


def analyze_latent_distributions(normal_latents, anomaly_latents_dict):
    """Compare latent distributions across anomaly types."""
    # Compute normal distribution statistics
    normal_mean = np.mean(normal_latents, axis=0)
    normal_cov = np.cov(normal_latents.T)
    # Regularize covariance
    normal_cov += np.eye(normal_cov.shape[0]) * 1e-6
    normal_cov_inv = np.linalg.inv(normal_cov)

    print("\n" + "="*70)
    print("LATENT SPACE ANALYSIS")
    print("="*70)

    # Compute distances for each anomaly type
    results = []
    for atype, latents in anomaly_latents_dict.items():
        distances = compute_mahalanobis_distance(latents, normal_mean, normal_cov_inv)
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        min_dist = np.min(distances)
        max_dist = np.max(distances)

        results.append({
            "type": atype,
            "mean_dist": mean_dist,
            "std_dist": std_dist,
            "min_dist": min_dist,
            "max_dist": max_dist,
        })

    # Sort by mean distance (ascending - lower distance = harder to detect)
    results_sorted = sorted(results, key=lambda x: x["mean_dist"])

    print(f"\n{'Anomaly Type':<20} {'Mean Dist':>12} {'Std Dist':>12} {'Min':>10} {'Max':>10}")
    print("-"*70)

    for r in results_sorted:
        marker = " <-- CLOSEST TO NORMAL" if r["type"] == results_sorted[0]["type"] else ""
        print(f"{r['type']:<20} {r['mean_dist']:>12.2f} {r['std_dist']:>12.2f} {r['min_dist']:>10.2f} {r['max_dist']:>10.2f}{marker}")

    # Compare normal to itself
    normal_distances = compute_mahalanobis_distance(normal_latents, normal_mean, normal_cov_inv)
    print(f"\n{'NORMAL (baseline)':<20} {np.mean(normal_distances):>12.2f} {np.std(normal_distances):>12.2f}")

    return results_sorted, normal_mean, normal_cov_inv


def test_frequency_drift_severity(model, generator, config, device):
    """Test how detection changes with frequency drift severity."""
    print("\n" + "="*70)
    print("FREQUENCY DRIFT: SEVERITY ANALYSIS")
    print("="*70)

    # Training data for fitting detector
    train_dataset = RFDataset.from_generator(
        generator=generator,
        num_samples=2000,
        anomaly_ratio=0.0,
        snr_range=tuple(config.data.snr_range),
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

    severities = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]
    results = []

    for severity in severities:
        # Test data with frequency drift only
        test_dataset = RFDataset.from_generator(
            generator=generator,
            num_samples=1000,
            anomaly_ratio=0.1,
            snr_range=tuple(config.data.snr_range),
            anomaly_types=["frequency_drift"],
            anomaly_severity=severity,
        )
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        detector = AnomalyDetector(
            model=model,
            method="latent",
            threshold_method="percentile",
            threshold_percentile=95,
            snr_adaptive=True,
            snr_bins=7,
            device=device,
        )
        detector.fit(train_loader, num_batches=50)

        all_scores, all_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                iq = batch["iq"].to(device)
                snr = batch.get("snr")
                if snr is not None:
                    snr = snr.to(device)
                snr_db = batch.get("snr_db")
                power = batch.get("power")
                if power is not None:
                    power = power.to(device)

                result = detector.detect(iq, snr, snr_db, power)
                all_scores.append(result.scores)
                all_labels.append(batch["label"].numpy())

        scores = np.concatenate(all_scores)
        labels = np.concatenate(all_labels)
        metrics = compute_metrics(scores, labels)

        results.append({
            "severity": severity,
            "auroc": metrics.auroc,
            "auprc": metrics.auprc,
        })
        print(f"  Severity {severity:>5.1f}: AUROC={metrics.auroc:.4f}, AUPRC={metrics.auprc:.4f}")

    return results


def analyze_frequency_drift_characteristics(generator, config):
    """Analyze the spectral characteristics of frequency drift anomalies."""
    print("\n" + "="*70)
    print("FREQUENCY DRIFT: SIGNAL CHARACTERISTICS")
    print("="*70)

    # Generate normal and freq drift samples using generate_anomaly
    np.random.seed(42)

    normal_samples = []
    freq_drift_samples = []

    for _ in range(100):
        # Normal sample
        normal_iq, _ = generator.generate_normal_signal(snr_db=15.0)
        normal_samples.append(normal_iq)

        # Frequency drift sample
        drift_iq, _ = generator.generate_anomaly(
            anomaly_type="frequency_drift",
            snr_db=15.0,
            severity=config.data.anomaly_severity,
        )
        freq_drift_samples.append(drift_iq)

    normal_samples = np.array(normal_samples)
    freq_drift_samples = np.array(freq_drift_samples)

    # Compute spectral characteristics
    def compute_spectral_features(samples):
        ffts = np.fft.fft(samples[:, 0, :] + 1j * samples[:, 1, :], axis=1)
        magnitudes = np.abs(ffts)

        # Features
        peak_freq = np.argmax(magnitudes, axis=1)
        spectral_centroid = np.sum(magnitudes * np.arange(magnitudes.shape[1]), axis=1) / np.sum(magnitudes, axis=1)
        spectral_spread = np.sqrt(np.sum(magnitudes * (np.arange(magnitudes.shape[1]) - spectral_centroid[:, None])**2, axis=1) / np.sum(magnitudes, axis=1))

        return {
            "peak_freq_mean": np.mean(peak_freq),
            "peak_freq_std": np.std(peak_freq),
            "centroid_mean": np.mean(spectral_centroid),
            "centroid_std": np.std(spectral_centroid),
            "spread_mean": np.mean(spectral_spread),
            "spread_std": np.std(spectral_spread),
        }

    normal_features = compute_spectral_features(normal_samples)
    drift_features = compute_spectral_features(freq_drift_samples)

    print(f"\n{'Feature':<25} {'Normal':>15} {'Freq Drift':>15} {'Diff %':>10}")
    print("-"*70)

    for key in normal_features:
        normal_val = normal_features[key]
        drift_val = drift_features[key]
        diff_pct = (drift_val - normal_val) / (normal_val + 1e-8) * 100
        print(f"{key:<25} {normal_val:>15.2f} {drift_val:>15.2f} {diff_pct:>+10.1f}%")

    # Time-domain analysis
    print("\n--- Time Domain Analysis ---")

    def compute_time_features(samples):
        # Instantaneous frequency (phase derivative)
        complex_signal = samples[:, 0, :] + 1j * samples[:, 1, :]
        phase = np.unwrap(np.angle(complex_signal), axis=1)
        inst_freq = np.diff(phase, axis=1)

        return {
            "inst_freq_mean": np.mean(inst_freq),
            "inst_freq_std": np.mean(np.std(inst_freq, axis=1)),  # Variance over time
            "phase_variance": np.mean(np.var(phase, axis=1)),
        }

    normal_time = compute_time_features(normal_samples)
    drift_time = compute_time_features(freq_drift_samples)

    for key in normal_time:
        normal_val = normal_time[key]
        drift_val = drift_time[key]
        diff_pct = (drift_val - normal_val) / (abs(normal_val) + 1e-8) * 100
        print(f"{key:<25} {normal_val:>15.4f} {drift_val:>15.4f} {diff_pct:>+10.1f}%")


def test_alternative_detection_for_freq_drift(model, generator, config, device):
    """Test if alternative detection methods work better for freq drift."""
    print("\n" + "="*70)
    print("ALTERNATIVE DETECTION METHODS FOR FREQUENCY DRIFT")
    print("="*70)

    # Training data
    train_dataset = RFDataset.from_generator(
        generator=generator,
        num_samples=2000,
        anomaly_ratio=0.0,
        snr_range=tuple(config.data.snr_range),
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

    # Test data with freq drift only
    test_dataset = RFDataset.from_generator(
        generator=generator,
        num_samples=1000,
        anomaly_ratio=0.1,
        snr_range=tuple(config.data.snr_range),
        anomaly_types=["frequency_drift"],
        anomaly_severity=config.data.anomaly_severity,
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    methods = ["latent", "reconstruction", "hybrid"]
    hybrid_weights_list = [[0.5, 0.5], [0.3, 0.7], [0.7, 0.3]]

    results = []

    for method in methods:
        if method == "hybrid":
            for weights in hybrid_weights_list:
                detector = AnomalyDetector(
                    model=model,
                    method=method,
                    hybrid_weights=weights,
                    threshold_method="percentile",
                    threshold_percentile=95,
                    snr_adaptive=True,
                    snr_bins=7,
                    device=device,
                )
                detector.fit(train_loader, num_batches=50)

                all_scores, all_labels = [], []
                with torch.no_grad():
                    for batch in test_loader:
                        iq = batch["iq"].to(device)
                        snr = batch.get("snr")
                        if snr is not None:
                            snr = snr.to(device)
                        snr_db = batch.get("snr_db")
                        power = batch.get("power")
                        if power is not None:
                            power = power.to(device)

                        result = detector.detect(iq, snr, snr_db, power)
                        all_scores.append(result.scores)
                        all_labels.append(batch["label"].numpy())

                scores = np.concatenate(all_scores)
                labels = np.concatenate(all_labels)
                metrics = compute_metrics(scores, labels)

                name = f"hybrid {weights}"
                results.append({"method": name, "auroc": metrics.auroc})
                print(f"  {name:<25}: AUROC={metrics.auroc:.4f}")
        else:
            detector = AnomalyDetector(
                model=model,
                method=method,
                threshold_method="percentile",
                threshold_percentile=95,
                snr_adaptive=True,
                snr_bins=7,
                device=device,
                invert_scores=(method == "reconstruction"),
            )
            detector.fit(train_loader, num_batches=50)

            all_scores, all_labels = [], []
            with torch.no_grad():
                for batch in test_loader:
                    iq = batch["iq"].to(device)
                    snr = batch.get("snr")
                    if snr is not None:
                        snr = snr.to(device)
                    snr_db = batch.get("snr_db")
                    power = batch.get("power")
                    if power is not None:
                        power = power.to(device)

                    result = detector.detect(iq, snr, snr_db, power)
                    all_scores.append(result.scores)
                    all_labels.append(batch["label"].numpy())

            scores = np.concatenate(all_scores)
            labels = np.concatenate(all_labels)
            metrics = compute_metrics(scores, labels)

            results.append({"method": method, "auroc": metrics.auroc})
            print(f"  {method:<25}: AUROC={metrics.auroc:.4f}")

    return results


def main():
    config = load_config("configs/default.yaml")
    device = get_device()
    print(f"Using device: {device}")

    checkpoint_path = "checkpoints/20260118_184144/best_model.pt"
    model = load_model(checkpoint_path, config, device)
    model.eval()

    generator = SyntheticRFGenerator(
        sequence_length=config.data.sequence_length,
        sample_rate=config.data.sample_rate,
        seed=42,
    )

    print("\n" + "="*70)
    print("FREQUENCY DRIFT DETECTION ANALYSIS")
    print("="*70)
    print(f"\nInvestigating why frequency_drift has lower AUROC (0.8004)")

    # 1. Extract latent representations for all anomaly types
    anomaly_types = ["interference", "frequency_drift", "amplitude_spike", "phase_noise", "burst_noise"]
    anomaly_latents = {}

    print("\nExtracting latent representations...")
    for atype in anomaly_types:
        latents, _ = extract_latent_by_type(model, generator, config, device, atype)
        anomaly_latents[atype] = latents
        print(f"  {atype}: {latents.shape}")

    normal_latents = extract_normal_latent(model, generator, config, device)
    print(f"  normal: {normal_latents.shape}")

    # 2. Analyze latent distributions
    results, normal_mean, normal_cov_inv = analyze_latent_distributions(normal_latents, anomaly_latents)

    # 3. Test frequency drift with different severities
    severity_results = test_frequency_drift_severity(model, generator, config, device)

    # 4. Analyze signal characteristics
    analyze_frequency_drift_characteristics(generator, config)

    # 5. Test alternative detection methods
    alt_results = test_alternative_detection_for_freq_drift(model, generator, config, device)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY: FREQUENCY DRIFT DETECTION")
    print("="*70)

    freq_drift_result = [r for r in results if r["type"] == "frequency_drift"][0]
    closest = results[0]

    print(f"\n1. Latent Space Analysis:")
    print(f"   - Frequency drift mean Mahalanobis distance: {freq_drift_result['mean_dist']:.2f}")
    print(f"   - Closest to normal: {closest['type']} ({closest['mean_dist']:.2f})")

    if closest["type"] == "frequency_drift":
        print(f"   --> Frequency drift is CLOSEST to normal in latent space!")
        print(f"       This explains the lower AUROC.")

    print(f"\n2. Severity Required:")
    for r in severity_results:
        if r["auroc"] >= 0.90:
            print(f"   - Need severity >= {r['severity']} for 0.90+ AUROC")
            break

    print(f"\n3. Recommendations:")
    print(f"   - Increase frequency drift rate in synthetic generator")
    print(f"   - Add frequency-domain features to latent space")
    print(f"   - Consider phase-based detection for this anomaly type")


if __name__ == "__main__":
    main()
