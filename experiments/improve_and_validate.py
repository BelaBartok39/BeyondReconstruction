#!/usr/bin/env python3
"""Additional experiments to improve and validate results.

Tests:
1. Ensemble detection (multiple latent samples)
2. SNR-stratified analysis (identify weak spots)
3. Confusion analysis (which anomalies are missed)
4. Threshold optimization
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix

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


def snr_stratified_analysis(model, config, device):
    """Analyze performance across SNR bins."""
    print("\n" + "="*60)
    print("SNR-STRATIFIED ANALYSIS")
    print("="*60)

    generator = SyntheticRFGenerator(
        sequence_length=config.data.sequence_length,
        sample_rate=config.data.sample_rate,
        seed=42,
    )

    # Generate data with known SNR values
    snr_bins = [(-5, 5), (5, 15), (15, 25), (25, 35)]

    for snr_min, snr_max in snr_bins:
        train_dataset = RFDataset.from_generator(
            generator=generator,
            num_samples=1000,
            anomaly_ratio=0.0,
            snr_range=(snr_min, snr_max),
        )
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

        test_dataset = RFDataset.from_generator(
            generator=generator,
            num_samples=1000,
            anomaly_ratio=0.1,
            snr_range=(snr_min, snr_max),
            anomaly_severity=config.data.anomaly_severity,
        )
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        detector = AnomalyDetector(
            model=model,
            method="latent",
            threshold_method="percentile",
            threshold_percentile=95,
            snr_adaptive=False,  # Disabled for per-bin analysis
            device=device,
        )
        detector.fit(train_loader, num_batches=20)

        all_scores, all_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                iq = batch["iq"].to(device)
                snr = batch.get("snr")
                if snr is not None:
                    snr = snr.to(device)
                power = batch.get("power")
                if power is not None:
                    power = power.to(device)

                result = detector.detect(iq, snr, batch.get("snr_db"), power)
                all_scores.append(result.scores)
                all_labels.append(batch["label"].numpy())

        scores = np.concatenate(all_scores)
        labels = np.concatenate(all_labels)
        auroc = roc_auc_score(labels, scores)

        print(f"  SNR {snr_min:3d} to {snr_max:3d} dB: AUROC = {auroc:.4f}")


def per_anomaly_analysis(model, config, device):
    """Detailed per-anomaly-type analysis."""
    print("\n" + "="*60)
    print("PER-ANOMALY DETAILED ANALYSIS")
    print("="*60)

    generator = SyntheticRFGenerator(
        sequence_length=config.data.sequence_length,
        sample_rate=config.data.sample_rate,
        seed=42,
    )

    all_anomaly_types = ["interference", "frequency_drift", "amplitude_spike", "phase_noise", "burst_noise"]

    train_dataset = RFDataset.from_generator(
        generator=generator,
        num_samples=2000,
        anomaly_ratio=0.0,
        snr_range=tuple(config.data.snr_range),
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

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

    print(f"\n  {'Anomaly Type':<20} {'AUROC':>8} {'AUPRC':>8} {'F1@95':>8}")
    print("  " + "-"*50)

    for anomaly_type in all_anomaly_types:
        test_dataset = RFDataset.from_generator(
            generator=generator,
            num_samples=2000,
            anomaly_ratio=0.1,
            snr_range=tuple(config.data.snr_range),
            anomaly_types=[anomaly_type],
            anomaly_severity=config.data.anomaly_severity,
        )
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        all_scores, all_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                iq = batch["iq"].to(device)
                snr = batch.get("snr")
                if snr is not None:
                    snr = snr.to(device)
                power = batch.get("power")
                if power is not None:
                    power = power.to(device)

                result = detector.detect(iq, snr, batch.get("snr_db"), power)
                all_scores.append(result.scores)
                all_labels.append(batch["label"].numpy())

        scores = np.concatenate(all_scores)
        labels = np.concatenate(all_labels)

        metrics = compute_metrics(scores, labels)
        in_training = "✓" if anomaly_type in config.data.anomaly_types else "✗"

        print(f"  {anomaly_type:<20} {metrics.auroc:>8.4f} {metrics.auprc:>8.4f} {metrics.f1:>8.4f} {in_training}")


def ensemble_detection(model, config, device, n_samples=10):
    """Test if ensemble/multiple forward passes improves detection."""
    print("\n" + "="*60)
    print(f"ENSEMBLE DETECTION (n={n_samples} forward passes)")
    print("="*60)

    generator = SyntheticRFGenerator(
        sequence_length=config.data.sequence_length,
        sample_rate=config.data.sample_rate,
        seed=42,
    )

    train_dataset = RFDataset.from_generator(
        generator=generator,
        num_samples=2000,
        anomaly_ratio=0.0,
        snr_range=tuple(config.data.snr_range),
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

    test_dataset = RFDataset.from_generator(
        generator=generator,
        num_samples=2000,
        anomaly_ratio=0.1,
        snr_range=tuple(config.data.snr_range),
        anomaly_severity=config.data.anomaly_severity,
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Single pass baseline
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

    # Collect latent statistics from training data
    all_latents = []
    model.eval()
    with torch.no_grad():
        for batch in train_loader:
            iq = batch["iq"].to(device)
            snr = batch.get("snr")
            if snr is not None:
                snr = snr.to(device)
            power = batch.get("power")
            if power is not None:
                power = power.to(device)

            # Get latent representation
            if hasattr(model, 'encode'):
                mu, logvar = model.encode(iq, snr, power)
            else:
                mu, logvar = model.encoder(iq, snr, power)
            all_latents.append(mu.cpu().numpy())

    latents = np.concatenate(all_latents, axis=0)
    latent_mean = latents.mean(axis=0)
    latent_cov = np.cov(latents.T)
    latent_cov_inv = np.linalg.pinv(latent_cov + 1e-6 * np.eye(latent_cov.shape[0]))

    def mahalanobis_score(z):
        diff = z - latent_mean
        return np.sqrt(np.sum(diff @ latent_cov_inv * diff, axis=-1))

    # Single pass scores
    single_scores, ensemble_scores, all_labels = [], [], []

    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            iq = batch["iq"].to(device)
            snr = batch.get("snr")
            if snr is not None:
                snr = snr.to(device)
            power = batch.get("power")
            if power is not None:
                power = power.to(device)

            # Single pass
            if hasattr(model, 'encode'):
                mu, logvar = model.encode(iq, snr, power)
            else:
                mu, logvar = model.encoder(iq, snr, power)
            single_scores.append(mahalanobis_score(mu.cpu().numpy()))

            # Ensemble: sample from posterior multiple times
            batch_ensemble_scores = []
            for _ in range(n_samples):
                std = torch.exp(0.5 * logvar)
                eps = torch.randn_like(std)
                z = mu + eps * std
                batch_ensemble_scores.append(mahalanobis_score(z.cpu().numpy()))

            # Average ensemble scores
            ensemble_scores.append(np.mean(batch_ensemble_scores, axis=0))
            all_labels.append(batch["label"].numpy())

    single_scores = np.concatenate(single_scores)
    ensemble_scores = np.concatenate(ensemble_scores)
    labels = np.concatenate(all_labels)

    single_auroc = roc_auc_score(labels, single_scores)
    ensemble_auroc = roc_auc_score(labels, ensemble_scores)

    print(f"  Single pass AUROC:   {single_auroc:.4f}")
    print(f"  Ensemble AUROC:      {ensemble_auroc:.4f}")
    print(f"  Improvement:         {ensemble_auroc - single_auroc:+.4f}")


def threshold_analysis(model, config, device):
    """Analyze optimal threshold selection."""
    print("\n" + "="*60)
    print("THRESHOLD ANALYSIS")
    print("="*60)

    generator = SyntheticRFGenerator(
        sequence_length=config.data.sequence_length,
        sample_rate=config.data.sample_rate,
        seed=42,
    )

    train_dataset = RFDataset.from_generator(
        generator=generator,
        num_samples=2000,
        anomaly_ratio=0.0,
        snr_range=tuple(config.data.snr_range),
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

    test_dataset = RFDataset.from_generator(
        generator=generator,
        num_samples=2000,
        anomaly_ratio=0.1,
        snr_range=tuple(config.data.snr_range),
        anomaly_severity=config.data.anomaly_severity,
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
            power = batch.get("power")
            if power is not None:
                power = power.to(device)

            result = detector.detect(iq, snr, batch.get("snr_db"), power)
            all_scores.append(result.scores)
            all_labels.append(batch["label"].numpy())

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)

    # Find optimal threshold for different metrics
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_f1_idx = np.argmax(f1_scores)

    print(f"\n  Optimal F1 threshold: {thresholds[best_f1_idx]:.4f}")
    print(f"  Best F1 score:        {f1_scores[best_f1_idx]:.4f}")
    print(f"  At this threshold:")
    print(f"    Precision: {precision[best_f1_idx]:.4f}")
    print(f"    Recall:    {recall[best_f1_idx]:.4f}")

    # Different percentile thresholds
    print(f"\n  Percentile-based thresholds:")
    for pct in [90, 95, 97, 99]:
        thresh = np.percentile(scores[labels == 0], pct)
        preds = (scores > thresh).astype(int)
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
        print(f"    {pct}th percentile: Precision={prec:.3f}, Recall={rec:.3f}, F1={f1:.3f}")


def test_lower_severity(model, config, device):
    """Test detection at very low severity to find limits."""
    print("\n" + "="*60)
    print("LOW SEVERITY DETECTION LIMITS")
    print("="*60)

    generator = SyntheticRFGenerator(
        sequence_length=config.data.sequence_length,
        sample_rate=config.data.sample_rate,
        seed=42,
    )

    train_dataset = RFDataset.from_generator(
        generator=generator,
        num_samples=2000,
        anomaly_ratio=0.0,
        snr_range=tuple(config.data.snr_range),
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

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

    severities = [0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]

    print(f"\n  {'Severity':>10} {'AUROC':>8} {'Detection Level':<20}")
    print("  " + "-"*45)

    for severity in severities:
        test_dataset = RFDataset.from_generator(
            generator=generator,
            num_samples=2000,
            anomaly_ratio=0.1,
            snr_range=tuple(config.data.snr_range),
            anomaly_severity=severity,
        )
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

        all_scores, all_labels = [], []
        with torch.no_grad():
            for batch in test_loader:
                iq = batch["iq"].to(device)
                snr = batch.get("snr")
                if snr is not None:
                    snr = snr.to(device)
                power = batch.get("power")
                if power is not None:
                    power = power.to(device)

                result = detector.detect(iq, snr, batch.get("snr_db"), power)
                all_scores.append(result.scores)
                all_labels.append(batch["label"].numpy())

        scores = np.concatenate(all_scores)
        labels = np.concatenate(all_labels)
        auroc = roc_auc_score(labels, scores)

        if auroc > 0.9:
            level = "Excellent"
        elif auroc > 0.8:
            level = "Good"
        elif auroc > 0.7:
            level = "Moderate"
        elif auroc > 0.6:
            level = "Weak"
        else:
            level = "Near random"

        print(f"  {severity:>10.2f} {auroc:>8.4f} {level:<20}")


def main():
    config = load_config("configs/default.yaml")
    device = get_device()
    print(f"Using device: {device}")

    checkpoint_path = "checkpoints/20260118_184144/best_model.pt"
    model = load_model(checkpoint_path, config, device)
    model.eval()

    print("\n" + "="*60)
    print("ADDITIONAL VALIDATION & IMPROVEMENT EXPERIMENTS")
    print("="*60)

    # Run all analyses
    snr_stratified_analysis(model, config, device)
    per_anomaly_analysis(model, config, device)
    ensemble_detection(model, config, device)
    threshold_analysis(model, config, device)
    test_lower_severity(model, config, device)

    print("\n" + "="*60)
    print("EXPERIMENTS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
