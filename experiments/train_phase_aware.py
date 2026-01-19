#!/usr/bin/env python3
"""Train VAE with phase-sensitive loss terms.

This experiment trains a model with phase and instantaneous frequency loss
to force the latent space to capture phase information better.
"""

from __future__ import annotations

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.data.synthetic import SyntheticRFGenerator
from src.data.datasets import RFDataset
from src.models.snr_encoder import SNRConditionedVAE
from src.detection.detector import AnomalyDetector
from src.detection.metrics import compute_metrics


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def create_phase_aware_model(config, phase_weight: float, inst_freq_weight: float):
    """Create model with phase-sensitive loss."""
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
        probabilistic_decoder=False,
        phase_loss_weight=phase_weight,
        inst_freq_loss_weight=inst_freq_weight,
    )


def train_epoch(model, train_loader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_recon = 0
    total_kl = 0
    total_phase = 0
    total_inst_freq = 0
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
        loss_tuple = model.loss(iq, x_recon, mu, logvar)

        if len(loss_tuple) == 3:
            loss, recon_loss, kl_loss = loss_tuple
            phase_loss = inst_freq_loss = torch.tensor(0.0)
        else:
            loss, recon_loss, kl_loss, phase_loss, inst_freq_loss = loss_tuple

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item()
        total_phase += phase_loss.item()
        total_inst_freq += inst_freq_loss.item()
        n_batches += 1

    return {
        "loss": total_loss / n_batches,
        "recon": total_recon / n_batches,
        "kl": total_kl / n_batches,
        "phase": total_phase / n_batches,
        "inst_freq": total_inst_freq / n_batches,
    }


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
            all_scores.append(result.scores)
            all_labels.append(batch["label"].numpy())

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    metrics = compute_metrics(scores, labels)

    return metrics.auroc, metrics.auprc


def train_and_evaluate(
    config,
    device,
    phase_weight: float,
    inst_freq_weight: float,
    num_epochs: int = 50,
):
    """Train model and evaluate on frequency drift."""
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

    # Test data for frequency drift specifically
    test_freq_drift = RFDataset.from_generator(
        generator=generator,
        num_samples=1000,
        anomaly_ratio=0.1,
        snr_range=tuple(config.data.snr_range),
        anomaly_types=["frequency_drift"],
        anomaly_severity=config.data.anomaly_severity,
    )
    test_freq_drift_loader = DataLoader(test_freq_drift, batch_size=64, shuffle=False)

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
    model = create_phase_aware_model(config, phase_weight, inst_freq_weight)
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

    best_auroc = 0
    best_epoch = 0

    print(f"\nTraining with phase_weight={phase_weight}, inst_freq_weight={inst_freq_weight}")
    print("-" * 60)

    for epoch in range(num_epochs):
        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device)

        # Evaluate every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            auroc_fd, _ = evaluate_detection(model, train_loader, test_freq_drift_loader, device)
            auroc_all, _ = evaluate_detection(model, train_loader, test_all_loader, device)

            print(f"  Epoch {epoch+1:3d}: loss={train_metrics['loss']:.4f}, "
                  f"phase={train_metrics['phase']:.4f}, inst_freq={train_metrics['inst_freq']:.4f}, "
                  f"AUROC(fd)={auroc_fd:.4f}, AUROC(all)={auroc_all:.4f}")

            if auroc_fd > best_auroc:
                best_auroc = auroc_fd
                best_epoch = epoch + 1

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

    return {
        "phase_weight": phase_weight,
        "inst_freq_weight": inst_freq_weight,
        "best_auroc_fd": best_auroc,
        "best_epoch": best_epoch,
        "per_anomaly": final_results,
        "model": model,
    }


def main():
    config = load_config("configs/default.yaml")
    device = get_device()
    print(f"Using device: {device}")

    print("\n" + "="*70)
    print("PHASE-AWARE TRAINING EXPERIMENT")
    print("="*70)

    # Test different weight combinations
    weight_configs = [
        (0.0, 0.0),    # Baseline (no phase loss)
        (0.1, 0.0),    # Phase only
        (0.0, 0.1),    # Inst freq only
        (0.1, 0.1),    # Both equal
        (0.2, 0.1),    # More phase
        (0.1, 0.2),    # More inst_freq
        (0.5, 0.5),    # Strong both
    ]

    all_results = []

    for phase_w, inst_freq_w in weight_configs:
        result = train_and_evaluate(
            config, device,
            phase_weight=phase_w,
            inst_freq_weight=inst_freq_w,
            num_epochs=30,  # Shorter for quick comparison
        )
        all_results.append(result)

    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    print(f"\n{'Phase λ':>10} {'InstFreq λ':>12} {'FD AUROC':>10} {'Overall':>10}")
    print("-"*50)

    baseline_fd = None
    best_result = None

    for r in all_results:
        avg_auroc = np.mean(list(r["per_anomaly"].values()))
        fd_auroc = r["per_anomaly"]["frequency_drift"]

        if r["phase_weight"] == 0.0 and r["inst_freq_weight"] == 0.0:
            baseline_fd = fd_auroc
            marker = " (baseline)"
        elif best_result is None or fd_auroc > best_result["per_anomaly"]["frequency_drift"]:
            best_result = r
            marker = " <-- BEST FD"
        else:
            marker = ""

        print(f"{r['phase_weight']:>10.1f} {r['inst_freq_weight']:>12.1f} {fd_auroc:>10.4f} {avg_auroc:>10.4f}{marker}")

    if best_result and baseline_fd:
        improvement = best_result["per_anomaly"]["frequency_drift"] - baseline_fd
        print(f"\nFrequency Drift Improvement: {improvement:+.4f}")

    # Detailed per-anomaly for best config
    if best_result:
        print(f"\nBest Configuration: phase_weight={best_result['phase_weight']}, "
              f"inst_freq_weight={best_result['inst_freq_weight']}")
        print("\nPer-Anomaly AUROC:")
        for atype, auroc in best_result["per_anomaly"].items():
            print(f"  {atype:<20}: {auroc:.4f}")

        # Save best model
        save_dir = Path("checkpoints") / f"phase_aware_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        save_dir.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": best_result["model"].state_dict(),
            "config": {
                "phase_weight": best_result["phase_weight"],
                "inst_freq_weight": best_result["inst_freq_weight"],
            },
        }, save_dir / "best_model.pt")
        print(f"\nBest model saved to: {save_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
