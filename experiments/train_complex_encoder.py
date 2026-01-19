#!/usr/bin/env python3
"""Train VAE with complex-valued encoder for phase-preserving RF processing.

This experiment compares the complex-valued encoder against the standard
real-valued encoder to test if complex convolutions better capture phase
information and reduce the Mahalanobis distance overlap in the 5-15 range.
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.data.synthetic import SyntheticRFGenerator
from src.data.datasets import RFDataset
from src.models.complex_encoder import ComplexVAE
from src.models.snr_encoder import SNRConditionedVAE
from src.detection.detector import AnomalyDetector
from src.detection.metrics import compute_metrics


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def create_complex_model(config):
    """Create ComplexVAE model with SNR/power conditioning."""
    return ComplexVAE(
        latent_dim=config.model.latent_dim,
        sequence_length=config.data.sequence_length,
        hidden_channels=config.model.hidden_channels,
        kernel_size=config.model.kernel_size,
        use_batch_norm=config.model.use_batch_norm,
        dropout=config.model.dropout,
        beta=config.model.beta,
        snr_embedding_dim=config.model.snr_embedding_dim,
        use_power_conditioning=getattr(config.model, 'use_power_conditioning', False),
    )


def create_baseline_model(config):
    """Create standard SNRConditionedVAE for comparison."""
    return SNRConditionedVAE(
        latent_dim=config.model.latent_dim,
        sequence_length=config.data.sequence_length,
        hidden_channels=config.model.hidden_channels,
        snr_embedding_dim=config.model.snr_embedding_dim,
        kernel_size=config.model.kernel_size,
        use_batch_norm=config.model.use_batch_norm,
        dropout=config.model.dropout,
        beta=config.model.beta,
        use_power_conditioning=getattr(config.model, 'use_power_conditioning', False),
    )


def train_epoch(model, train_loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    n_batches = 0

    for batch in train_loader:
        iq = batch["iq"].to(device)
        snr = batch.get("snr")
        if snr is not None:
            snr = snr.to(device)
        power = batch.get("power")
        if power is not None:
            power = power.to(device)

        optimizer.zero_grad()

        # Forward pass
        x_recon, mu, logvar, z = model(iq, snr, power)

        # Compute loss
        loss, recon_loss, kl_loss = model.loss(iq, x_recon, mu, logvar)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "recon": total_recon / n_batches,
        "kl": total_kl / n_batches,
    }


def compute_mahalanobis_stats(model, train_loader, test_loader, device):
    """Compute Mahalanobis distance statistics for overlap analysis."""
    model.eval()

    # Collect training latents to compute mean and covariance
    train_latents = []
    with torch.no_grad():
        for batch in train_loader:
            iq = batch["iq"].to(device)
            snr = batch.get("snr")
            if snr is not None:
                snr = snr.to(device)
            power = batch.get("power")
            if power is not None:
                power = power.to(device)

            mu, _ = model.encode(iq, snr, power)
            train_latents.append(mu.cpu())

    train_latents = torch.cat(train_latents, dim=0)

    # Check for NaN in latents
    if torch.isnan(train_latents).any():
        print("WARNING: NaN in training latents, returning empty stats")
        return {"normal": [], "anomaly": {}}

    mean = train_latents.mean(dim=0)
    cov = torch.cov(train_latents.T) + 1e-4 * torch.eye(train_latents.size(1))  # Stronger regularization

    # Use pseudo-inverse for numerical stability
    try:
        cov_inv = torch.linalg.inv(cov)
    except RuntimeError:
        cov_inv = torch.linalg.pinv(cov)

    # Compute Mahalanobis distances for test set
    distances_by_type = {"normal": [], "anomaly": {}}

    with torch.no_grad():
        for batch in test_loader:
            iq = batch["iq"].to(device)
            snr = batch.get("snr")
            if snr is not None:
                snr = snr.to(device)
            power = batch.get("power")
            if power is not None:
                power = power.to(device)
            labels = batch["label"].numpy()
            anomaly_types = batch.get("anomaly_type", [None] * len(labels))

            mu, _ = model.encode(iq, snr, power)
            mu = mu.cpu()

            # Check for NaN
            if torch.isnan(mu).any():
                continue

            # Compute Mahalanobis distance with numerical stability
            diff = mu - mean
            mahal_sq = torch.sum(diff @ cov_inv * diff, dim=1)
            mahal_sq = torch.clamp(mahal_sq, min=0)  # Ensure non-negative
            dist = torch.sqrt(mahal_sq).numpy()
            dist = np.nan_to_num(dist, nan=0.0, posinf=100.0, neginf=0.0)

            for i, (label, atype) in enumerate(zip(labels, anomaly_types)):
                if label == 0:
                    distances_by_type["normal"].append(dist[i])
                else:
                    if atype not in distances_by_type["anomaly"]:
                        distances_by_type["anomaly"][atype] = []
                    distances_by_type["anomaly"][atype].append(dist[i])

    return distances_by_type


def evaluate_detection(model, train_loader, test_loader, device):
    """Evaluate anomaly detection performance."""
    model.eval()

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
            # Handle NaN scores
            scores_clean = np.nan_to_num(result.scores, nan=0.0, posinf=100.0, neginf=0.0)
            all_scores.append(scores_clean)
            all_labels.append(batch["label"].numpy())

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)

    # Final NaN check
    if np.isnan(scores).any():
        print("WARNING: NaN in scores after detection, returning 0.5 AUROC")
        return 0.5, 0.5

    metrics = compute_metrics(scores, labels)

    return metrics.auroc, metrics.auprc


def train_and_evaluate(
    config,
    device,
    model_type: str = "complex",
    num_epochs: int = 50,
):
    """Train model and evaluate."""
    generator = SyntheticRFGenerator(
        sequence_length=config.data.sequence_length,
        sample_rate=config.data.sample_rate,
        seed=42,
    )

    # Training data (normal only)
    train_dataset = RFDataset.from_generator(
        generator=generator,
        num_samples=5000,
        anomaly_ratio=0.0,
        snr_range=tuple(config.data.snr_range),
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)

    # Test data for all anomalies
    test_all = RFDataset.from_generator(
        generator=generator,
        num_samples=2000,
        anomaly_ratio=0.1,
        snr_range=tuple(config.data.snr_range),
        anomaly_severity=config.data.anomaly_severity,
    )
    test_all_loader = DataLoader(test_all, batch_size=64, shuffle=False)

    # Create model
    if model_type == "complex":
        model = create_complex_model(config)
    else:
        model = create_baseline_model(config)
    model = model.to(device)

    # Initialize lazy layers
    dummy_batch = next(iter(train_loader))
    with torch.no_grad():
        _ = model(
            dummy_batch["iq"].to(device),
            dummy_batch["snr"].to(device) if dummy_batch.get("snr") is not None else None,
            dummy_batch["power"].to(device) if dummy_batch.get("power") is not None else None,
        )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    print(f"\nTraining {model_type.upper()} encoder")
    print("-" * 60)

    for epoch in range(num_epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device)

        # Evaluate every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            auroc, auprc = evaluate_detection(model, train_loader, test_all_loader, device)
            print(f"  Epoch {epoch+1:3d}: loss={train_metrics['loss']:.4f}, "
                  f"recon={train_metrics['recon']:.4f}, kl={train_metrics['kl']:.4f}, "
                  f"AUROC={auroc:.4f}")

            scheduler.step(train_metrics['loss'])

    # Final evaluation on all anomaly types
    anomaly_types = ["interference", "frequency_drift", "amplitude_spike", "phase_noise", "burst_noise"]
    final_results = {}

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
        auroc, _ = evaluate_detection(model, train_loader, test_loader, device)
        final_results[atype] = auroc

    # Compute Mahalanobis distance stats
    mahal_stats = compute_mahalanobis_stats(model, train_loader, test_all_loader, device)

    return {
        "model_type": model_type,
        "per_anomaly": final_results,
        "mahal_stats": mahal_stats,
        "model": model,
    }


def analyze_overlap(mahal_stats):
    """Analyze the 5-15 overlap region."""
    normal_dists = np.array(mahal_stats["normal"])

    # Count samples in overlap region
    normal_in_overlap = np.sum((normal_dists >= 5) & (normal_dists <= 15))
    normal_pct = 100 * normal_in_overlap / len(normal_dists)

    print(f"\n  Normal signals:")
    print(f"    Mean distance: {np.mean(normal_dists):.2f}")
    print(f"    In 5-15 overlap: {normal_in_overlap}/{len(normal_dists)} ({normal_pct:.1f}%)")

    for atype, dists in mahal_stats["anomaly"].items():
        dists = np.array(dists)
        in_overlap = np.sum((dists >= 5) & (dists <= 15))
        pct = 100 * in_overlap / len(dists)
        print(f"\n  {atype}:")
        print(f"    Mean distance: {np.mean(dists):.2f}")
        print(f"    In 5-15 overlap: {in_overlap}/{len(dists)} ({pct:.1f}%)")


def main():
    config = load_config("configs/default.yaml")
    device = get_device()
    print(f"Using device: {device}")

    print("\n" + "="*70)
    print("COMPLEX ENCODER VS STANDARD ENCODER EXPERIMENT")
    print("="*70)

    # Train and evaluate both models
    results = {}

    for model_type in ["baseline", "complex"]:
        result = train_and_evaluate(
            config, device,
            model_type=model_type,
            num_epochs=50,
        )
        results[model_type] = result

    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    print(f"\n{'Model':>15} {'Interference':>12} {'FreqDrift':>12} {'AmpSpike':>12} {'PhaseNoise':>12} {'BurstNoise':>12} {'Average':>10}")
    print("-"*95)

    for model_type in ["baseline", "complex"]:
        r = results[model_type]
        avg = np.mean(list(r["per_anomaly"].values()))
        print(f"{model_type:>15} "
              f"{r['per_anomaly']['interference']:>12.4f} "
              f"{r['per_anomaly']['frequency_drift']:>12.4f} "
              f"{r['per_anomaly']['amplitude_spike']:>12.4f} "
              f"{r['per_anomaly']['phase_noise']:>12.4f} "
              f"{r['per_anomaly']['burst_noise']:>12.4f} "
              f"{avg:>10.4f}")

    # Improvement analysis
    baseline_avg = np.mean(list(results["baseline"]["per_anomaly"].values()))
    complex_avg = np.mean(list(results["complex"]["per_anomaly"].values()))
    improvement = complex_avg - baseline_avg
    print(f"\nComplex encoder improvement: {improvement:+.4f} AUROC")

    # Frequency drift specific
    fd_baseline = results["baseline"]["per_anomaly"]["frequency_drift"]
    fd_complex = results["complex"]["per_anomaly"]["frequency_drift"]
    fd_improvement = fd_complex - fd_baseline
    print(f"Frequency drift improvement: {fd_improvement:+.4f} AUROC")

    # Overlap analysis
    print("\n" + "="*70)
    print("MAHALANOBIS DISTANCE OVERLAP ANALYSIS (5-15 region)")
    print("="*70)

    for model_type in ["baseline", "complex"]:
        print(f"\n{model_type.upper()} model:")
        analyze_overlap(results[model_type]["mahal_stats"])

    # Save best model
    best_model_type = "complex" if complex_avg > baseline_avg else "baseline"
    save_dir = Path("checkpoints") / f"complex_encoder_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": results[best_model_type]["model"].state_dict(),
        "model_type": best_model_type,
        "results": {k: v for k, v in results[best_model_type].items() if k != "model"},
    }, save_dir / "best_model.pt")
    print(f"\nBest model ({best_model_type}) saved to: {save_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
