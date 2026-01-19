#!/usr/bin/env python3
"""Test phase-aware detection for improved frequency drift detection.

This experiment tests whether adding phase-based features can improve
detection of frequency_drift anomalies (currently the weakest at 0.77 AUROC).
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
from src.detection.phase_detector import PhaseAnomalyDetector, HybridPhaseLatentDetector
from src.detection.metrics import compute_metrics


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(checkpoint_path: str, config, device: torch.device):
    """Load model with lazy layer initialization."""
    model = create_model(config)
    model = model.to(device)

    dummy_iq = torch.randn(1, 2, config.data.sequence_length, device=device)
    dummy_snr = torch.rand(1, device=device)
    dummy_power = torch.rand(1, device=device)
    with torch.no_grad():
        _ = model(dummy_iq, dummy_snr, dummy_power)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def extract_iq_numpy(loader):
    """Extract raw I/Q data as numpy array."""
    all_iq = []
    all_labels = []
    for batch in loader:
        all_iq.append(batch["iq"].numpy())
        all_labels.append(batch["label"].numpy())
    return np.concatenate(all_iq), np.concatenate(all_labels)


def test_phase_only(train_iq, test_iq, test_labels):
    """Test phase-only detection."""
    detector = PhaseAnomalyDetector(percentile_threshold=95.0)
    detector.fit(train_iq)

    scores = detector.score(test_iq)
    metrics = compute_metrics(scores, test_labels)

    return {
        "method": "Phase-Only",
        "auroc": metrics.auroc,
        "auprc": metrics.auprc,
        "f1": metrics.f1,
    }


def test_latent_only(model, train_loader, test_loader, device):
    """Test latent-only detection."""
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

    return {
        "method": "Latent-Only",
        "auroc": metrics.auroc,
        "auprc": metrics.auprc,
        "f1": metrics.f1,
        "scores": scores,
        "labels": labels,
    }


def test_hybrid_phase_latent(model, train_loader, test_loader, train_iq, test_iq, device, phase_weight=0.3):
    """Test hybrid phase + latent detection."""
    # Get latent scores
    latent_result = test_latent_only(model, train_loader, test_loader, device)
    latent_scores = latent_result["scores"]
    labels = latent_result["labels"]

    # Fit phase detector
    phase_detector = PhaseAnomalyDetector(percentile_threshold=95.0)
    phase_detector.fit(train_iq)
    phase_scores = phase_detector.score(test_iq)

    # Normalize and combine
    latent_norm = (latent_scores - latent_scores.min()) / (latent_scores.max() - latent_scores.min() + 1e-8)
    phase_norm = (phase_scores - phase_scores.min()) / (phase_scores.max() - phase_scores.min() + 1e-8)

    hybrid_scores = (1 - phase_weight) * latent_norm + phase_weight * phase_norm

    metrics = compute_metrics(hybrid_scores, labels)

    return {
        "method": f"Hybrid (phase={phase_weight})",
        "auroc": metrics.auroc,
        "auprc": metrics.auprc,
        "f1": metrics.f1,
    }


def test_on_anomaly_type(model, generator, config, device, anomaly_type):
    """Test detection performance on a specific anomaly type."""
    # Training data (normal only)
    train_dataset = RFDataset.from_generator(
        generator=generator,
        num_samples=2000,
        anomaly_ratio=0.0,
        snr_range=tuple(config.data.snr_range),
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    train_iq, _ = extract_iq_numpy(train_loader)

    # Test data (specific anomaly type)
    test_dataset = RFDataset.from_generator(
        generator=generator,
        num_samples=1000,
        anomaly_ratio=0.1,
        snr_range=tuple(config.data.snr_range),
        anomaly_types=[anomaly_type],
        anomaly_severity=config.data.anomaly_severity,
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    test_iq, test_labels = extract_iq_numpy(test_loader)

    results = []

    # Phase-only
    results.append(test_phase_only(train_iq, test_iq, test_labels))

    # Latent-only
    latent_result = test_latent_only(model, train_loader, test_loader, device)
    results.append({
        "method": "Latent-Only",
        "auroc": latent_result["auroc"],
        "auprc": latent_result["auprc"],
        "f1": latent_result["f1"],
    })

    # Hybrid with different weights
    for weight in [0.1, 0.2, 0.3, 0.4, 0.5]:
        results.append(test_hybrid_phase_latent(
            model, train_loader, test_loader, train_iq, test_iq, device, phase_weight=weight
        ))

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
    print("PHASE-AWARE DETECTION EXPERIMENT")
    print("="*70)

    anomaly_types = ["interference", "frequency_drift", "amplitude_spike", "phase_noise", "burst_noise"]

    all_results = {}

    for atype in anomaly_types:
        print(f"\n--- {atype.upper()} ---")
        results = test_on_anomaly_type(model, generator, config, device, atype)
        all_results[atype] = results

        print(f"{'Method':<25} {'AUROC':>10} {'AUPRC':>10} {'F1':>10}")
        print("-"*60)

        best_result = max(results, key=lambda x: x["auroc"])
        for r in results:
            marker = " <-- BEST" if r["method"] == best_result["method"] else ""
            print(f"{r['method']:<25} {r['auroc']:>10.4f} {r['auprc']:>10.4f} {r['f1']:>10.4f}{marker}")

    # Summary focusing on frequency_drift
    print("\n" + "="*70)
    print("SUMMARY: FREQUENCY DRIFT IMPROVEMENT")
    print("="*70)

    freq_results = all_results["frequency_drift"]
    latent_only = [r for r in freq_results if r["method"] == "Latent-Only"][0]
    best_hybrid = max([r for r in freq_results if "Hybrid" in r["method"]], key=lambda x: x["auroc"])

    print(f"\n  Latent-Only AUROC:  {latent_only['auroc']:.4f}")
    print(f"  Best Hybrid AUROC:  {best_hybrid['auroc']:.4f} ({best_hybrid['method']})")
    print(f"  Improvement:        {best_hybrid['auroc'] - latent_only['auroc']:+.4f}")

    # Overall comparison
    print("\n" + "="*70)
    print("OVERALL COMPARISON (ALL ANOMALY TYPES)")
    print("="*70)

    print(f"\n{'Anomaly Type':<20} {'Latent':>10} {'Best Hybrid':>12} {'Δ':>8}")
    print("-"*55)

    total_latent = 0
    total_hybrid = 0

    for atype in anomaly_types:
        results = all_results[atype]
        latent = [r for r in results if r["method"] == "Latent-Only"][0]["auroc"]
        best = max([r for r in results if "Hybrid" in r["method"]], key=lambda x: x["auroc"])

        total_latent += latent
        total_hybrid += best["auroc"]

        delta = best["auroc"] - latent
        print(f"{atype:<20} {latent:>10.4f} {best['auroc']:>12.4f} {delta:>+8.4f}")

    print("-"*55)
    avg_latent = total_latent / len(anomaly_types)
    avg_hybrid = total_hybrid / len(anomaly_types)
    print(f"{'AVERAGE':<20} {avg_latent:>10.4f} {avg_hybrid:>12.4f} {avg_hybrid - avg_latent:>+8.4f}")


if __name__ == "__main__":
    main()
