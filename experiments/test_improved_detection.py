#!/usr/bin/env python3
"""Test improved hybrid detection methods.

Compares:
1. Latent-only (baseline)
2. Basic hybrid (latent + phase)
3. Enhanced frequency detector
4. Adaptive hybrid (latent + phase + frequency with adaptive weights)

Goal: Improve frequency_drift AUROC without hurting other anomaly types.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.data.synthetic import SyntheticRFGenerator
from src.data.datasets import RFDataset
from src.models.snr_encoder import create_model
from src.detection.detector import AnomalyDetector
from src.detection.phase_detector import (
    PhaseAnomalyDetector,
    EnhancedFrequencyDetector,
    AdaptiveHybridDetector,
    ChirpDetector,
)
from src.detection.metrics import compute_metrics


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(checkpoint_path: str, config, device: torch.device):
    """Load trained model."""
    model = create_model(config)
    model = model.to(device)

    dummy_iq = torch.randn(1, 2, config.data.sequence_length, device=device)
    dummy_snr = torch.rand(1, device=device)
    dummy_power = torch.rand(1, device=device)
    with torch.no_grad():
        _ = model(dummy_iq, dummy_snr, dummy_power)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def get_latent_scores(model, train_loader, test_loader, device):
    """Get latent-based anomaly scores."""
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

    return np.concatenate(all_scores), np.concatenate(all_labels)


def test_detection_methods(model, train_loader, test_loader, device, anomaly_type="all"):
    """Test all detection methods on a dataset."""
    # Get raw IQ data
    train_iq = np.concatenate([b["iq"].numpy() for b in train_loader])
    test_iq = np.concatenate([b["iq"].numpy() for b in test_loader])

    # Get latent scores
    latent_scores, labels = get_latent_scores(model, train_loader, test_loader, device)

    results = {}

    # 1. Latent-only
    metrics = compute_metrics(latent_scores, labels)
    results["Latent-only"] = metrics.auroc

    # 2. Phase-only
    phase_det = PhaseAnomalyDetector()
    phase_det.fit(train_iq)
    phase_scores = phase_det.score(test_iq)
    metrics = compute_metrics(phase_scores, labels)
    results["Phase-only"] = metrics.auroc

    # 3. Enhanced frequency-only
    freq_det = EnhancedFrequencyDetector()
    freq_det.fit(train_iq)
    freq_scores = freq_det.score(test_iq)
    metrics = compute_metrics(freq_scores, labels)
    results["Freq-only"] = metrics.auroc

    # 3b. Chirp detector (optimized for frequency drift)
    chirp_det = ChirpDetector()
    chirp_det.fit(train_iq)
    chirp_scores = chirp_det.score(test_iq)
    metrics = compute_metrics(chirp_scores, labels)
    results["Chirp-only"] = metrics.auroc

    # 4. Basic hybrid (latent + phase)
    def normalize(s):
        return (s - s.min()) / (s.max() - s.min() + 1e-8)

    for pw in [0.3, 0.5, 0.7]:
        hybrid = (1 - pw) * normalize(latent_scores) + pw * normalize(phase_scores)
        metrics = compute_metrics(hybrid, labels)
        results[f"Hybrid(p={pw})"] = metrics.auroc

    # 5. Enhanced hybrid (latent + freq)
    for fw in [0.3, 0.5, 0.7]:
        hybrid = (1 - fw) * normalize(latent_scores) + fw * normalize(freq_scores)
        metrics = compute_metrics(hybrid, labels)
        results[f"Hybrid(f={fw})"] = metrics.auroc

    # 5b. Chirp hybrid (latent + chirp)
    for cw in [0.3, 0.5, 0.7]:
        hybrid = (1 - cw) * normalize(latent_scores) + cw * normalize(chirp_scores)
        metrics = compute_metrics(hybrid, labels)
        results[f"Hybrid(c={cw})"] = metrics.auroc

    # 6. Adaptive hybrid (latent + phase + freq)
    adaptive_det = AdaptiveHybridDetector(
        base_latent_weight=0.5,
        base_freq_weight=0.3,
        base_phase_weight=0.2,
    )
    adaptive_det.fit(train_iq)

    adaptive_scores = adaptive_det.score(test_iq, latent_scores, adaptive=False)
    metrics = compute_metrics(adaptive_scores, labels)
    results["Adaptive(fixed)"] = metrics.auroc

    adaptive_scores = adaptive_det.score(test_iq, latent_scores, adaptive=True)
    metrics = compute_metrics(adaptive_scores, labels)
    results["Adaptive(dynamic)"] = metrics.auroc

    # 7. Try different adaptive weights
    for lw, fw, pw in [(0.4, 0.4, 0.2), (0.3, 0.5, 0.2), (0.3, 0.4, 0.3)]:
        adaptive_det = AdaptiveHybridDetector(
            base_latent_weight=lw,
            base_freq_weight=fw,
            base_phase_weight=pw,
        )
        adaptive_det.fit(train_iq)
        adaptive_scores = adaptive_det.score(test_iq, latent_scores, adaptive=True)
        metrics = compute_metrics(adaptive_scores, labels)
        results[f"Adapt({lw},{fw},{pw})"] = metrics.auroc

    return results


def main():
    config = load_config("configs/default.yaml")
    device = get_device()
    print(f"Device: {device}")

    checkpoint_path = "checkpoints/20260118_184144/best_model.pt"
    if not Path(checkpoint_path).exists():
        print(f"Checkpoint not found: {checkpoint_path}")
        return

    print(f"Loading model from: {checkpoint_path}")
    model = load_model(checkpoint_path, config, device)

    generator = SyntheticRFGenerator(
        sequence_length=config.data.sequence_length,
        sample_rate=config.data.sample_rate,
        seed=42,
    )

    # Training data
    train_dataset = RFDataset.from_generator(
        generator=generator,
        num_samples=5000,
        anomaly_ratio=0.0,
        snr_range=tuple(config.data.snr_range),
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

    print("\n" + "="*80)
    print("IMPROVED DETECTION METHODS COMPARISON")
    print("="*80)

    # Test on each anomaly type
    anomaly_types = ["frequency_drift", "interference", "amplitude_spike", "phase_noise", "burst_noise"]

    all_results = {}
    for atype in anomaly_types:
        print(f"\n--- Testing: {atype.upper()} ---")

        test_dataset = RFDataset.from_generator(
            generator=generator,
            num_samples=1000,
            anomaly_ratio=0.1,
            snr_range=tuple(config.data.snr_range),
            anomaly_types=[atype],
            anomaly_severity=config.data.anomaly_severity,
        )
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        results = test_detection_methods(model, train_loader, test_loader, device, atype)
        all_results[atype] = results

        # Print sorted by AUROC
        sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
        print(f"{'Method':<25} {'AUROC':>10}")
        print("-"*37)
        for method, auroc in sorted_results[:8]:  # Top 8
            marker = " <-- BEST" if auroc == sorted_results[0][1] else ""
            print(f"{method:<25} {auroc:>10.4f}{marker}")

    # Summary table
    print("\n" + "="*80)
    print("SUMMARY: ALL ANOMALY TYPES")
    print("="*80)

    # Get all methods
    all_methods = list(all_results["frequency_drift"].keys())

    # Print header
    header = f"{'Method':<20}"
    for atype in anomaly_types:
        header += f" {atype[:8]:>10}"
    header += f" {'Average':>10}"
    print(header)
    print("-"*90)

    # Key methods to highlight
    key_methods = ["Latent-only", "Phase-only", "Chirp-only", "Hybrid(p=0.5)", "Hybrid(f=0.5)", "Hybrid(c=0.5)", "Adaptive(dynamic)"]

    for method in key_methods:
        if method not in all_methods:
            continue
        row = f"{method:<20}"
        total = 0
        for atype in anomaly_types:
            auroc = all_results[atype].get(method, 0)
            total += auroc
            row += f" {auroc:>10.4f}"
        avg = total / len(anomaly_types)
        row += f" {avg:>10.4f}"
        print(row)

    # Find best for frequency_drift
    print("\n" + "="*80)
    print("FREQUENCY DRIFT FOCUS (Our Target)")
    print("="*80)

    fd_results = all_results["frequency_drift"]
    sorted_fd = sorted(fd_results.items(), key=lambda x: x[1], reverse=True)

    print(f"\nTop 5 methods for frequency_drift:")
    for i, (method, auroc) in enumerate(sorted_fd[:5], 1):
        print(f"  {i}. {method}: {auroc:.4f}")

    # Improvement analysis
    baseline = fd_results["Latent-only"]
    best_method, best_auroc = sorted_fd[0]

    print(f"\nImprovement over latent-only:")
    print(f"  Baseline (Latent-only): {baseline:.4f}")
    print(f"  Best ({best_method}): {best_auroc:.4f}")
    print(f"  Improvement: {best_auroc - baseline:+.4f}")

    # Check if other anomalies degraded
    print("\n" + "="*80)
    print("DEGRADATION CHECK")
    print("="*80)
    print(f"\nComparing '{best_method}' vs 'Latent-only':")

    for atype in anomaly_types:
        baseline_auroc = all_results[atype]["Latent-only"]
        best_auroc = all_results[atype].get(best_method, 0)
        delta = best_auroc - baseline_auroc
        status = "OK" if delta >= -0.02 else "DEGRADED"
        print(f"  {atype:<20}: {baseline_auroc:.4f} -> {best_auroc:.4f} ({delta:+.4f}) [{status}]")


if __name__ == "__main__":
    main()
