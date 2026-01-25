#!/usr/bin/env python3
"""Reproduce production model achieving 0.95+ AUROC.

This script:
1. Trains model with default.yaml configuration
2. Evaluates with latent-only detection (expect ~0.93 AUROC)
3. Evaluates with hybrid detection using ChirpDetector (expect ~0.95 AUROC)
4. Prints comparison table
5. Saves results to results/reproduction_test.json

Usage:
    python experiments/reproduce_production.py --device cuda
    python experiments/reproduce_production.py --checkpoint checkpoints/existing/best_model.pt  # Skip training
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.datasets import RFDataset, create_dataloaders
from src.data.synthetic import SyntheticRFGenerator
from src.detection.detector import AnomalyDetector
from src.detection.metrics import compute_metrics
from src.detection.phase_detector import ChirpDetector
from src.models.snr_encoder import create_model
from src.utils.config import load_config, save_config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Reproduce production model")
    parser.add_argument(
        "--config", default="configs/default.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--checkpoint", default=None, help="Path to existing checkpoint (skip training)"
    )
    parser.add_argument(
        "--output-dir",
        default="checkpoints/reproduce_production",
        help="Output directory for model and results",
    )
    parser.add_argument(
        "--device", default="auto", help="Device to use (auto, cuda, cpu)"
    )
    parser.add_argument(
        "--freq-weight", type=float, default=0.6, help="Weight for frequency detector in hybrid (0.6-0.7 for 0.95+ AUROC)"
    )
    return parser.parse_args()


def get_device(device_str: str) -> torch.device:
    """Get torch device from string."""
    if device_str != "auto":
        return torch.device(device_str)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_model_capabilities(model: torch.nn.Module) -> tuple[bool, bool, bool, bool]:
    """Detect model capabilities."""
    is_snr_conditioned = hasattr(model, "encoder") and hasattr(model.encoder, "cond_embed")
    is_vae = hasattr(model, "reparameterize")
    uses_power = hasattr(model, "use_power_conditioning") and model.use_power_conditioning
    is_probabilistic = hasattr(model, "probabilistic_decoder") and model.probabilistic_decoder
    return is_snr_conditioned, is_vae, uses_power, is_probabilistic


def extract_batch_tensors(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Extract and move batch tensors to device."""
    iq = batch["iq"].to(device)
    snr = batch.get("snr")
    power = batch.get("power")
    if snr is not None:
        snr = snr.to(device)
    if power is not None:
        power = power.to(device)
    return iq, snr, power


def model_forward(
    model: torch.nn.Module,
    iq: torch.Tensor,
    snr: torch.Tensor | None,
    power: torch.Tensor | None,
    is_snr_conditioned: bool,
    is_vae: bool,
    uses_power: bool,
    is_probabilistic: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
    """Unified forward pass handling all model types."""
    if is_snr_conditioned and snr is not None:
        if uses_power and power is not None:
            out = model(iq, snr, power)
        else:
            out = model(iq, snr)
    elif is_vae:
        out = model(iq)
    else:
        x_recon, latent = model(iq)
        return x_recon, None, None, None, latent

    if is_probabilistic:
        x_mean, x_logvar, mu, logvar, z = out
        return x_mean, x_logvar, mu, logvar, z
    else:
        x_recon, mu, logvar, z = out
        return x_recon, None, mu, logvar, z


def compute_loss(
    model: torch.nn.Module,
    iq: torch.Tensor,
    x_recon: torch.Tensor,
    x_recon_logvar: torch.Tensor | None,
    mu: torch.Tensor | None,
    logvar: torch.Tensor | None,
    is_vae: bool,
    is_probabilistic: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute loss for any model type."""
    if is_vae and mu is not None and logvar is not None:
        loss_out = model.loss(iq, x_recon, mu, logvar, x_recon_logvar)
        if len(loss_out) == 4:
            return loss_out
        return loss_out[0], loss_out[1], loss_out[2], torch.tensor(0.0)
    loss = model.reconstruction_loss(iq, x_recon, x_recon_logvar)
    return loss, loss, torch.tensor(0.0), torch.tensor(0.0)


def train_epoch(
    model: torch.nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip_norm: float = 1.0,
) -> dict[str, float]:
    """Train for one epoch."""
    model.train()
    is_snr_conditioned, is_vae, uses_power, is_probabilistic = get_model_capabilities(model)

    total_loss = total_recon = total_kl = 0.0
    num_batches = 0

    for batch in tqdm(train_loader, desc="Training", leave=False):
        iq, snr, power = extract_batch_tensors(batch, device)
        optimizer.zero_grad()

        x_recon, x_recon_logvar, mu, logvar, _ = model_forward(
            model, iq, snr, power, is_snr_conditioned, is_vae, uses_power, is_probabilistic
        )
        loss, recon_loss, kl_loss, _ = compute_loss(
            model, iq, x_recon, x_recon_logvar, mu, logvar, is_vae, is_probabilistic
        )

        loss.backward()
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss
        num_batches += 1

    return {
        "loss": total_loss / num_batches,
        "recon_loss": total_recon / num_batches,
        "kl_loss": total_kl / num_batches,
    }


@torch.no_grad()
def validate(
    model: torch.nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Validate model."""
    model.eval()
    is_snr_conditioned, is_vae, uses_power, is_probabilistic = get_model_capabilities(model)

    total_loss = 0.0

    for batch in val_loader:
        iq, snr, power = extract_batch_tensors(batch, device)
        x_recon, x_recon_logvar, mu, logvar, _ = model_forward(
            model, iq, snr, power, is_snr_conditioned, is_vae, uses_power, is_probabilistic
        )
        loss, _, _, _ = compute_loss(
            model, iq, x_recon, x_recon_logvar, mu, logvar, is_vae, is_probabilistic
        )
        total_loss += loss.item()

    return {"val_loss": total_loss / len(val_loader)}


def train_model(
    config,
    device: torch.device,
    output_dir: Path,
) -> tuple[torch.nn.Module, DataLoader, DataLoader]:
    """Train the model and return trained model with data loaders."""
    torch.manual_seed(config.experiment.seed)

    # Create data generator and loaders
    logger.info("Creating synthetic data generator...")
    generator = SyntheticRFGenerator(
        sequence_length=config.data.sequence_length,
        sample_rate=config.data.sample_rate,
        seed=config.experiment.seed,
    )

    logger.info("Creating data loaders...")
    train_loader, val_loader, test_loader = create_dataloaders(config, generator)
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")

    # Create model
    logger.info(f"Creating model: {config.model.type}")
    model = create_model(config)
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")

    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.training.scheduler.T_max,
        eta_min=config.training.scheduler.min_lr,
    )

    # Training loop
    logger.info("Starting training...")
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(1, config.training.num_epochs + 1):
        train_metrics = train_epoch(
            model, train_loader, optimizer, device, config.training.gradient_clip_norm
        )
        val_metrics = validate(model, val_loader, device)
        scheduler.step()

        logger.info(
            f"Epoch {epoch}/{config.training.num_epochs} - "
            f"Train Loss: {train_metrics['loss']:.4f} - "
            f"Val Loss: {val_metrics['val_loss']:.4f} - "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            patience_counter = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_metrics["val_loss"],
                },
                output_dir / "best_model.pt",
            )
        else:
            patience_counter += 1

        if patience_counter >= config.training.early_stopping_patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

    # Load best model
    checkpoint = torch.load(output_dir / "best_model.pt", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    logger.info(f"Loaded best model from epoch {checkpoint['epoch']}")

    return model, train_loader, test_loader


def load_existing_model(
    checkpoint_path: Path,
    config,
    device: torch.device,
) -> tuple[torch.nn.Module, DataLoader, DataLoader]:
    """Load an existing model checkpoint."""
    logger.info(f"Loading existing model from {checkpoint_path}")

    # Create model and initialize
    model = create_model(config)
    model = model.to(device)

    # Initialize lazy layers with dummy forward pass
    dummy_iq = torch.randn(1, 2, config.data.sequence_length, device=device)
    dummy_snr = torch.rand(1, device=device)
    dummy_power = torch.rand(1, device=device)
    with torch.no_grad():
        _ = model(dummy_iq, dummy_snr, dummy_power)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Create data loaders
    generator = SyntheticRFGenerator(
        sequence_length=config.data.sequence_length,
        sample_rate=config.data.sample_rate,
        seed=config.experiment.seed,
    )
    train_loader, _, test_loader = create_dataloaders(config, generator)

    return model, train_loader, test_loader


def evaluate_latent_only(
    model: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    config,
) -> dict[str, float]:
    """Evaluate using latent-only (Mahalanobis) detection."""
    logger.info("Evaluating with latent-only detection...")

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
    scores, predictions, labels = detector.detect_batch(test_loader)

    if labels is None:
        return {"error": "No labels in test data"}

    metrics = compute_metrics(scores, labels, predictions)
    return {
        "auroc": metrics.auroc,
        "auprc": metrics.auprc,
        "f1": metrics.f1,
        "precision": metrics.precision,
        "recall": metrics.recall,
    }


def evaluate_hybrid(
    model: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    freq_weight: float = 0.5,
) -> dict[str, float]:
    """Evaluate using hybrid detection (latent + ChirpDetector)."""
    logger.info(f"Evaluating with hybrid detection (freq_weight={freq_weight})...")

    # Get latent scores
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

    # Collect latent scores and IQ data
    latent_scores_list = []
    labels_list = []
    test_iq_list = []

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
            latent_scores_list.append(result.scores)
            labels_list.append(batch["label"].numpy())
            test_iq_list.append(batch["iq"].numpy())

    latent_scores = np.concatenate(latent_scores_list)
    labels = np.concatenate(labels_list)
    test_iq = np.concatenate(test_iq_list)

    # Get training IQ for ChirpDetector fitting
    train_iq = np.concatenate([b["iq"].numpy() for b in train_loader])

    # Get chirp scores
    chirp_det = ChirpDetector()
    chirp_det.fit(train_iq)
    chirp_scores = chirp_det.score(test_iq)

    # Normalize and combine
    def normalize(s):
        return (s - s.min()) / (s.max() - s.min() + 1e-8)

    hybrid_scores = (1 - freq_weight) * normalize(latent_scores) + freq_weight * normalize(
        chirp_scores
    )

    metrics = compute_metrics(hybrid_scores, labels)
    return {
        "auroc": metrics.auroc,
        "auprc": metrics.auprc,
        "f1": metrics.f1,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "freq_weight": freq_weight,
    }


def evaluate_per_anomaly_type(
    model: torch.nn.Module,
    train_loader: DataLoader,
    device: torch.device,
    config,
    freq_weight: float = 0.5,
) -> dict[str, dict[str, float]]:
    """Evaluate on each anomaly type separately."""
    logger.info("Evaluating per anomaly type...")

    generator = SyntheticRFGenerator(
        sequence_length=config.data.sequence_length,
        sample_rate=config.data.sample_rate,
        seed=config.experiment.seed + 2000,
    )

    anomaly_types = ["frequency_drift", "interference", "amplitude_spike", "phase_noise"]
    results = {}

    # Setup detectors
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

    train_iq = np.concatenate([b["iq"].numpy() for b in train_loader])
    chirp_det = ChirpDetector()
    chirp_det.fit(train_iq)

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

        # Collect scores
        latent_scores_list = []
        labels_list = []
        test_iq_list = []

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
                latent_scores_list.append(result.scores)
                labels_list.append(batch["label"].numpy())
                test_iq_list.append(batch["iq"].numpy())

        latent_scores = np.concatenate(latent_scores_list)
        labels = np.concatenate(labels_list)
        test_iq = np.concatenate(test_iq_list)

        # Latent-only
        latent_metrics = compute_metrics(latent_scores, labels)

        # Hybrid
        chirp_scores = chirp_det.score(test_iq)

        def normalize(s):
            return (s - s.min()) / (s.max() - s.min() + 1e-8)

        hybrid_scores = (1 - freq_weight) * normalize(latent_scores) + freq_weight * normalize(
            chirp_scores
        )
        hybrid_metrics = compute_metrics(hybrid_scores, labels)

        results[atype] = {
            "latent_auroc": latent_metrics.auroc,
            "hybrid_auroc": hybrid_metrics.auroc,
        }

    return results


def print_results_table(
    latent_results: dict[str, float],
    hybrid_results: dict[str, float],
    per_type_results: dict[str, dict[str, float]],
) -> None:
    """Print formatted results table."""
    print("\n" + "=" * 70)
    print("REPRODUCTION RESULTS")
    print("=" * 70)

    print("\n--- Overall Detection Performance ---")
    print(f"{'Method':<25} {'AUROC':>10} {'AUPRC':>10} {'F1':>10}")
    print("-" * 55)
    print(
        f"{'Latent-only':<25} {latent_results['auroc']:>10.4f} "
        f"{latent_results['auprc']:>10.4f} {latent_results['f1']:>10.4f}"
    )
    freq_w = hybrid_results["freq_weight"]
    print(
        f"{'Hybrid (freq=' + str(freq_w) + ')':<25} {hybrid_results['auroc']:>10.4f} "
        f"{hybrid_results['auprc']:>10.4f} {hybrid_results['f1']:>10.4f}"
    )

    print("\n--- Per Anomaly Type AUROC ---")
    print(f"{'Anomaly Type':<20} {'Latent':>10} {'Hybrid':>10} {'Delta':>10}")
    print("-" * 55)

    for atype, metrics in per_type_results.items():
        delta = metrics["hybrid_auroc"] - metrics["latent_auroc"]
        print(
            f"{atype:<20} {metrics['latent_auroc']:>10.4f} "
            f"{metrics['hybrid_auroc']:>10.4f} {delta:>+10.4f}"
        )

    # Verification
    print("\n--- Verification ---")
    latent_pass = latent_results["auroc"] >= 0.92
    hybrid_pass = hybrid_results["auroc"] >= 0.95

    print(f"Latent AUROC >= 0.92: {'PASS' if latent_pass else 'FAIL'} ({latent_results['auroc']:.4f})")
    print(f"Hybrid AUROC >= 0.95: {'PASS' if hybrid_pass else 'FAIL'} ({hybrid_results['auroc']:.4f})")

    if latent_pass and hybrid_pass:
        print("\nProduction model successfully reproduced!")
    else:
        print("\nWarning: Results below expected thresholds.")


def main() -> None:
    """Main function."""
    args = parse_args()

    # Load config
    config = load_config(args.config)
    device = get_device(args.device)
    logger.info(f"Using device: {device}")

    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train or load model
    if args.checkpoint:
        model, train_loader, test_loader = load_existing_model(
            Path(args.checkpoint), config, device
        )
    else:
        save_config(config, output_dir / "config.yaml")
        model, train_loader, test_loader = train_model(config, device, output_dir)

    model.eval()

    # Evaluate with both methods
    latent_results = evaluate_latent_only(model, train_loader, test_loader, device, config)
    hybrid_results = evaluate_hybrid(model, train_loader, test_loader, device, args.freq_weight)
    per_type_results = evaluate_per_anomaly_type(
        model, train_loader, device, config, args.freq_weight
    )

    # Print results
    print_results_table(latent_results, hybrid_results, per_type_results)

    # Save results
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    results = {
        "timestamp": datetime.now().isoformat(),
        "config_path": args.config,
        "checkpoint": args.checkpoint or str(output_dir / "best_model.pt"),
        "device": str(device),
        "latent_detection": latent_results,
        "hybrid_detection": hybrid_results,
        "per_anomaly_type": per_type_results,
        "verification": {
            "latent_pass": latent_results["auroc"] >= 0.92,
            "hybrid_pass": hybrid_results["auroc"] >= 0.95,
        },
    }

    results_path = results_dir / "reproduction_test.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
