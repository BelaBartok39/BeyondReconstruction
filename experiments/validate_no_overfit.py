#!/usr/bin/env python3
"""Validate that the model is not overfitting.

Tests:
1. Different random seeds - results should be consistent
2. Held-out anomaly types - model should generalize
3. Different SNR ranges - robustness check
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


def evaluate_auroc(model, train_loader, test_loader, device):
    """Evaluate AUROC using latent-only detection."""
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
    return metrics.auroc, metrics.auprc


def test_different_seeds(model, config, device, seeds=[42, 123, 456, 789, 2024]):
    """Test model on data generated with different random seeds."""
    print("\n" + "="*60)
    print("TEST 1: Different Random Seeds")
    print("="*60)

    results = []
    for seed in seeds:
        generator = SyntheticRFGenerator(
            sequence_length=config.data.sequence_length,
            sample_rate=config.data.sample_rate,
            seed=seed,
        )

        # Training data (normal only)
        train_dataset = RFDataset.from_generator(
            generator=generator,
            num_samples=2000,
            anomaly_ratio=0.0,
            snr_range=tuple(config.data.snr_range),
            anomaly_severity=config.data.anomaly_severity,
        )
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

        # Test data (with anomalies)
        test_dataset = RFDataset.from_generator(
            generator=generator,
            num_samples=2000,
            anomaly_ratio=0.1,
            snr_range=tuple(config.data.snr_range),
            anomaly_severity=config.data.anomaly_severity,
        )
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        auroc, auprc = evaluate_auroc(model, train_loader, test_loader, device)
        results.append((seed, auroc, auprc))
        print(f"  Seed {seed}: AUROC={auroc:.4f}, AUPRC={auprc:.4f}")

    aurocs = [r[1] for r in results]
    print(f"\n  Mean AUROC: {np.mean(aurocs):.4f} ± {np.std(aurocs):.4f}")
    print(f"  Min: {np.min(aurocs):.4f}, Max: {np.max(aurocs):.4f}")

    return results


def test_held_out_anomalies(model, config, device):
    """Test on anomaly types not seen during training."""
    print("\n" + "="*60)
    print("TEST 2: Held-Out Anomaly Types")
    print("="*60)

    all_anomaly_types = ["interference", "frequency_drift", "amplitude_spike", "phase_noise", "burst_noise"]

    generator = SyntheticRFGenerator(
        sequence_length=config.data.sequence_length,
        sample_rate=config.data.sample_rate,
        seed=42,
    )

    # Training data (normal only - same for all)
    train_dataset = RFDataset.from_generator(
        generator=generator,
        num_samples=2000,
        anomaly_ratio=0.0,
        snr_range=tuple(config.data.snr_range),
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

    results = []
    for anomaly_type in all_anomaly_types:
        # Test with single anomaly type
        test_dataset = RFDataset.from_generator(
            generator=generator,
            num_samples=2000,
            anomaly_ratio=0.1,
            snr_range=tuple(config.data.snr_range),
            anomaly_types=[anomaly_type],
            anomaly_severity=config.data.anomaly_severity,
        )
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        auroc, auprc = evaluate_auroc(model, train_loader, test_loader, device)
        in_training = anomaly_type in config.data.anomaly_types
        marker = "✓" if in_training else "✗ (held-out)"
        results.append((anomaly_type, auroc, in_training))
        print(f"  {anomaly_type:20s}: AUROC={auroc:.4f} {marker}")

    seen = [r[1] for r in results if r[2]]
    unseen = [r[1] for r in results if not r[2]]

    if seen:
        print(f"\n  Seen anomaly types:   Mean AUROC = {np.mean(seen):.4f}")
    if unseen:
        print(f"  Unseen anomaly types: Mean AUROC = {np.mean(unseen):.4f}")

    return results


def test_different_snr_ranges(model, config, device):
    """Test on different SNR ranges to check robustness."""
    print("\n" + "="*60)
    print("TEST 3: Different SNR Ranges")
    print("="*60)

    snr_ranges = [
        (-5, 30),   # Original
        (-10, 10),  # Low SNR (harder)
        (10, 30),   # High SNR (easier)
        (0, 20),    # Mid range
    ]

    generator = SyntheticRFGenerator(
        sequence_length=config.data.sequence_length,
        sample_rate=config.data.sample_rate,
        seed=42,
    )

    results = []
    for snr_range in snr_ranges:
        # Training data
        train_dataset = RFDataset.from_generator(
            generator=generator,
            num_samples=2000,
            anomaly_ratio=0.0,
            snr_range=snr_range,
        )
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

        # Test data
        test_dataset = RFDataset.from_generator(
            generator=generator,
            num_samples=2000,
            anomaly_ratio=0.1,
            snr_range=snr_range,
            anomaly_severity=config.data.anomaly_severity,
        )
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        auroc, auprc = evaluate_auroc(model, train_loader, test_loader, device)
        is_original = snr_range == tuple(config.data.snr_range)
        marker = "← training range" if is_original else ""
        results.append((snr_range, auroc))
        print(f"  SNR {snr_range[0]:3d} to {snr_range[1]:3d} dB: AUROC={auroc:.4f} {marker}")

    return results


def test_severity_sensitivity(model, config, device):
    """Test how AUROC changes with anomaly severity."""
    print("\n" + "="*60)
    print("TEST 4: Anomaly Severity Sensitivity")
    print("="*60)

    severities = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]

    generator = SyntheticRFGenerator(
        sequence_length=config.data.sequence_length,
        sample_rate=config.data.sample_rate,
        seed=42,
    )

    # Training data (normal only - same for all)
    train_dataset = RFDataset.from_generator(
        generator=generator,
        num_samples=2000,
        anomaly_ratio=0.0,
        snr_range=tuple(config.data.snr_range),
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

    results = []
    for severity in severities:
        test_dataset = RFDataset.from_generator(
            generator=generator,
            num_samples=2000,
            anomaly_ratio=0.1,
            snr_range=tuple(config.data.snr_range),
            anomaly_severity=severity,
        )
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        auroc, auprc = evaluate_auroc(model, train_loader, test_loader, device)
        is_training = severity == config.data.anomaly_severity
        marker = "← training severity" if is_training else ""
        results.append((severity, auroc))
        print(f"  Severity {severity:.1f}: AUROC={auroc:.4f} {marker}")

    return results


def main():
    config = load_config("configs/default.yaml")
    device = get_device()
    print(f"Using device: {device}")

    checkpoint_path = "checkpoints/20260118_184144/best_model.pt"
    model = load_model(checkpoint_path, config, device)
    model.eval()

    print("\n" + "="*60)
    print("OVERFITTING VALIDATION TESTS")
    print("="*60)
    print(f"Model: {checkpoint_path}")
    print(f"Training anomaly types: {config.data.anomaly_types}")
    print(f"Training SNR range: {config.data.snr_range}")
    print(f"Training severity: {config.data.anomaly_severity}")

    # Run all tests
    seed_results = test_different_seeds(model, config, device)
    anomaly_results = test_held_out_anomalies(model, config, device)
    snr_results = test_different_snr_ranges(model, config, device)
    severity_results = test_severity_sensitivity(model, config, device)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    seed_aurocs = [r[1] for r in seed_results]
    print(f"\n1. Seed Stability: {np.mean(seed_aurocs):.4f} ± {np.std(seed_aurocs):.4f}")
    if np.std(seed_aurocs) < 0.05:
        print("   ✓ PASS: Results are stable across random seeds")
    else:
        print("   ✗ FAIL: High variance across seeds suggests overfitting")

    seen_aurocs = [r[1] for r in anomaly_results if r[2]]
    unseen_aurocs = [r[1] for r in anomaly_results if not r[2]]
    if unseen_aurocs:
        gap = np.mean(seen_aurocs) - np.mean(unseen_aurocs)
        print(f"\n2. Generalization Gap: {gap:.4f} (seen - unseen)")
        if gap < 0.1:
            print("   ✓ PASS: Model generalizes to unseen anomaly types")
        else:
            print("   ✗ FAIL: Large gap suggests overfitting to training anomaly types")

    original_snr = [r[1] for r in snr_results if r[0] == tuple(config.data.snr_range)][0]
    other_snr = [r[1] for r in snr_results if r[0] != tuple(config.data.snr_range)]
    snr_gap = original_snr - np.mean(other_snr)
    print(f"\n3. SNR Robustness Gap: {snr_gap:.4f}")
    if snr_gap < 0.15:
        print("   ✓ PASS: Model is robust to different SNR ranges")
    else:
        print("   ✗ FAIL: Performance drops significantly outside training SNR range")

    severity_1 = [r[1] for r in severity_results if r[0] == 1.0][0]
    print(f"\n4. Severity=1.0 AUROC: {severity_1:.4f}")
    if severity_1 > 0.7:
        print("   ✓ PASS: Model detects subtle anomalies (severity=1.0)")
    else:
        print("   ✗ WARNING: Model struggles with subtle anomalies")


if __name__ == "__main__":
    main()
