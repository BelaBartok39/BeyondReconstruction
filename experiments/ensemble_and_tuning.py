#!/usr/bin/env python3
"""Ensemble methods and hyperparameter tuning for improvement.

Experiments:
1. Ensemble of Mahalanobis + Isolation Forest
2. Different latent dimensions
3. Threshold optimization
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
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

    dummy_iq = torch.randn(1, 2, config.data.sequence_length, device=device)
    dummy_snr = torch.rand(1, device=device)
    dummy_power = torch.rand(1, device=device)
    with torch.no_grad():
        _ = model(dummy_iq, dummy_snr, dummy_power)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def extract_latents_and_scores(model, loader, device, detector=None):
    """Extract latent representations and optionally compute Mahalanobis scores."""
    model.eval()
    all_latents = []
    all_labels = []
    all_mahal_scores = []

    with torch.no_grad():
        for batch in loader:
            iq = batch["iq"].to(device)
            snr = batch.get("snr")
            if snr is not None:
                snr = snr.to(device)
            power = batch.get("power")
            if power is not None:
                power = power.to(device)
            snr_db = batch.get("snr_db")

            mu, _ = model.encoder(iq, snr, power)
            all_latents.append(mu.cpu().numpy())
            all_labels.append(batch["label"].numpy())

            if detector is not None:
                result = detector.detect(iq, snr, snr_db, power)
                all_mahal_scores.append(result.scores)

    latents = np.concatenate(all_latents)
    labels = np.concatenate(all_labels)
    mahal_scores = np.concatenate(all_mahal_scores) if all_mahal_scores else None

    return latents, labels, mahal_scores


def ensemble_scoring(mahal_scores, iforest_scores, method="average", weights=None):
    """Combine Mahalanobis and Isolation Forest scores."""
    # Normalize scores to [0, 1] range
    mahal_norm = (mahal_scores - mahal_scores.min()) / (mahal_scores.max() - mahal_scores.min() + 1e-8)
    iforest_norm = (iforest_scores - iforest_scores.min()) / (iforest_scores.max() - iforest_scores.min() + 1e-8)

    if method == "average":
        return (mahal_norm + iforest_norm) / 2
    elif method == "weighted":
        weights = weights or [0.5, 0.5]
        return weights[0] * mahal_norm + weights[1] * iforest_norm
    elif method == "max":
        return np.maximum(mahal_norm, iforest_norm)
    elif method == "product":
        return mahal_norm * iforest_norm
    else:
        raise ValueError(f"Unknown method: {method}")


def test_ensemble_methods(model, train_loader, test_loader, device):
    """Test different ensemble combinations."""
    print("\n" + "="*70)
    print("ENSEMBLE METHOD COMPARISON")
    print("="*70)

    # Fit Mahalanobis detector
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

    # Extract latents and Mahalanobis scores
    train_latents, _, _ = extract_latents_and_scores(model, train_loader, device)
    test_latents, test_labels, mahal_scores = extract_latents_and_scores(model, test_loader, device, detector)

    # Fit Isolation Forest
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_latents)
    test_scaled = scaler.transform(test_latents)

    iforest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    iforest.fit(train_scaled)
    iforest_scores = -iforest.decision_function(test_scaled)

    # Test different ensemble methods
    results = []

    # Individual methods
    mahal_metrics = compute_metrics(mahal_scores, test_labels)
    results.append(("Mahalanobis (Ours)", mahal_metrics.auroc, mahal_metrics.auprc))

    iforest_metrics = compute_metrics(iforest_scores, test_labels)
    results.append(("Isolation Forest", iforest_metrics.auroc, iforest_metrics.auprc))

    # Ensemble methods
    for method in ["average", "max", "product"]:
        ensemble_scores = ensemble_scoring(mahal_scores, iforest_scores, method=method)
        metrics = compute_metrics(ensemble_scores, test_labels)
        results.append((f"Ensemble ({method})", metrics.auroc, metrics.auprc))

    # Weighted ensembles
    for w1 in [0.3, 0.4, 0.5, 0.6, 0.7]:
        w2 = 1 - w1
        ensemble_scores = ensemble_scoring(mahal_scores, iforest_scores, method="weighted", weights=[w1, w2])
        metrics = compute_metrics(ensemble_scores, test_labels)
        results.append((f"Weighted ({w1:.1f}/{w2:.1f})", metrics.auroc, metrics.auprc))

    # Sort by AUROC
    results_sorted = sorted(results, key=lambda x: x[1], reverse=True)

    print(f"\n{'Method':<30} {'AUROC':>10} {'AUPRC':>10}")
    print("-"*55)

    for name, auroc, auprc in results_sorted:
        marker = " <-- BEST" if name == results_sorted[0][0] else ""
        print(f"{name:<30} {auroc:>10.4f} {auprc:>10.4f}{marker}")

    return results_sorted


def test_snr_bin_variations(model, train_loader, test_loader, device):
    """Test different SNR bin configurations."""
    print("\n" + "="*70)
    print("SNR BIN VARIATIONS")
    print("="*70)

    results = []
    bin_configs = [1, 3, 5, 7, 10, 15]  # 1 = no SNR binning

    for n_bins in bin_configs:
        detector = AnomalyDetector(
            model=model,
            method="latent",
            threshold_method="percentile",
            threshold_percentile=95,
            snr_adaptive=(n_bins > 1),
            snr_bins=n_bins,
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

        results.append((n_bins, metrics.auroc, metrics.auprc))

    print(f"\n{'SNR Bins':>10} {'AUROC':>10} {'AUPRC':>10}")
    print("-"*35)

    best_bins = max(results, key=lambda x: x[1])
    for n_bins, auroc, auprc in results:
        marker = " <-- BEST" if n_bins == best_bins[0] else ""
        print(f"{n_bins:>10} {auroc:>10.4f} {auprc:>10.4f}{marker}")

    return results


def test_percentile_thresholds(model, train_loader, test_loader, device):
    """Test different percentile thresholds."""
    print("\n" + "="*70)
    print("THRESHOLD PERCENTILE VARIATIONS")
    print("="*70)

    results = []
    percentiles = [90, 92, 94, 95, 96, 97, 98, 99]

    for pct in percentiles:
        detector = AnomalyDetector(
            model=model,
            method="latent",
            threshold_method="percentile",
            threshold_percentile=pct,
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

        results.append((pct, metrics.auroc, metrics.f1, metrics.precision, metrics.recall))

    print(f"\n{'Percentile':>12} {'AUROC':>10} {'F1':>10} {'Precision':>12} {'Recall':>10}")
    print("-"*60)

    best = max(results, key=lambda x: x[2])  # Best by F1
    for pct, auroc, f1, prec, rec in results:
        marker = " <-- BEST F1" if pct == best[0] else ""
        print(f"{pct:>12} {auroc:>10.4f} {f1:>10.4f} {prec:>12.4f} {rec:>10.4f}{marker}")

    return results


def test_per_anomaly_improvement(model, train_loader, device, config):
    """Test if improvements help specific anomaly types."""
    print("\n" + "="*70)
    print("PER-ANOMALY TYPE IMPROVEMENT CHECK")
    print("="*70)

    generator = SyntheticRFGenerator(
        sequence_length=config.data.sequence_length,
        sample_rate=config.data.sample_rate,
        seed=42,
    )

    # Fit combined detector (Mahalanobis + IForest ensemble)
    train_latents, _, _ = extract_latents_and_scores(model, train_loader, device)

    # Fit Mahalanobis detector
    mahal_detector = AnomalyDetector(
        model=model,
        method="latent",
        threshold_method="percentile",
        threshold_percentile=95,
        snr_adaptive=True,
        snr_bins=7,
        device=device,
    )
    mahal_detector.fit(train_loader, num_batches=50)

    # Fit Isolation Forest
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_latents)
    iforest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    iforest.fit(train_scaled)

    anomaly_types = ["interference", "frequency_drift", "amplitude_spike", "phase_noise", "burst_noise"]

    results = []
    for atype in anomaly_types:
        test_dataset = RFDataset.from_generator(
            generator=generator,
            num_samples=1000,
            anomaly_ratio=0.1,
            snr_range=tuple(config.data.snr_range),
            anomaly_types=[atype],
            anomaly_severity=config.data.anomaly_severity,
        )
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        # Get Mahalanobis scores
        test_latents, test_labels, mahal_scores = extract_latents_and_scores(model, test_loader, device, mahal_detector)

        # Get IForest scores
        test_scaled = scaler.transform(test_latents)
        iforest_scores = -iforest.decision_function(test_scaled)

        # Ensemble (average)
        ensemble_scores = ensemble_scoring(mahal_scores, iforest_scores, method="average")

        # Compute metrics
        mahal_metrics = compute_metrics(mahal_scores, test_labels)
        iforest_metrics = compute_metrics(iforest_scores, test_labels)
        ensemble_metrics = compute_metrics(ensemble_scores, test_labels)

        results.append({
            "type": atype,
            "mahal_auroc": mahal_metrics.auroc,
            "iforest_auroc": iforest_metrics.auroc,
            "ensemble_auroc": ensemble_metrics.auroc,
            "improvement": ensemble_metrics.auroc - mahal_metrics.auroc,
        })

    print(f"\n{'Anomaly Type':<20} {'Mahalanobis':>12} {'IForest':>12} {'Ensemble':>12} {'Δ':>8}")
    print("-"*70)

    for r in results:
        delta_str = f"{r['improvement']:+.4f}"
        print(f"{r['type']:<20} {r['mahal_auroc']:>12.4f} {r['iforest_auroc']:>12.4f} {r['ensemble_auroc']:>12.4f} {delta_str:>8}")

    # Overall improvement
    avg_mahal = np.mean([r["mahal_auroc"] for r in results])
    avg_iforest = np.mean([r["iforest_auroc"] for r in results])
    avg_ensemble = np.mean([r["ensemble_auroc"] for r in results])

    print("-"*70)
    print(f"{'AVERAGE':<20} {avg_mahal:>12.4f} {avg_iforest:>12.4f} {avg_ensemble:>12.4f} {avg_ensemble - avg_mahal:>+8.4f}")

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

    # Training data
    train_dataset = RFDataset.from_generator(
        generator=generator,
        num_samples=3000,
        anomaly_ratio=0.0,
        snr_range=tuple(config.data.snr_range),
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

    # Test data
    test_dataset = RFDataset.from_generator(
        generator=generator,
        num_samples=3000,
        anomaly_ratio=0.1,
        snr_range=tuple(config.data.snr_range),
        anomaly_severity=config.data.anomaly_severity,
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print("\n" + "="*70)
    print("IMPROVEMENT AND TUNING EXPERIMENTS")
    print("="*70)

    # Run experiments
    ensemble_results = test_ensemble_methods(model, train_loader, test_loader, device)
    snr_results = test_snr_bin_variations(model, train_loader, test_loader, device)
    threshold_results = test_percentile_thresholds(model, train_loader, test_loader, device)
    per_anomaly_results = test_per_anomaly_improvement(model, train_loader, device, config)

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    best_ensemble = ensemble_results[0]
    print(f"\n1. Best Ensemble: {best_ensemble[0]} (AUROC={best_ensemble[1]:.4f})")

    best_snr = max(snr_results, key=lambda x: x[1])
    print(f"2. Best SNR Bins: {best_snr[0]} (AUROC={best_snr[1]:.4f})")

    best_thresh = max(threshold_results, key=lambda x: x[2])
    print(f"3. Best Threshold: {best_thresh[0]}th percentile (F1={best_thresh[2]:.4f})")

    # Frequency drift improvement
    freq_drift = [r for r in per_anomaly_results if r["type"] == "frequency_drift"][0]
    print(f"\n4. Frequency Drift Improvement: {freq_drift['mahal_auroc']:.4f} -> {freq_drift['ensemble_auroc']:.4f}")


if __name__ == "__main__":
    main()
