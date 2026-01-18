#!/usr/bin/env python3
"""Baseline training script for RF anomaly detection autoencoder."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config, save_config
from src.data.synthetic import SyntheticRFGenerator
from src.data.datasets import RFDataset, create_dataloaders
from src.models.snr_encoder import create_model
from src.detection.detector import AnomalyDetector
from src.detection.metrics import compute_metrics


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train RF anomaly detection model")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/default.yaml",
        help="Path to config file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: checkpoints/<timestamp>)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (auto, cuda, cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (overrides config)",
    )
    return parser.parse_args()


def get_device(device_str: str) -> torch.device:
    """Get torch device from string."""
    if device_str == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    return torch.device(device_str)


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip_norm: float | None = 1.0,
) -> dict[str, float]:
    """Train for one epoch.

    Args:
        model: Model to train.
        train_loader: Training data loader.
        optimizer: Optimizer.
        device: Training device.
        gradient_clip_norm: Gradient clipping threshold.

    Returns:
        Dictionary with training metrics.
    """
    model.train()

    total_loss = 0.0
    total_recon_loss = 0.0
    total_kl_loss = 0.0
    num_batches = 0

    is_snr_conditioned = hasattr(model, "encoder") and hasattr(model.encoder, "snr_embed")
    is_vae = hasattr(model, "reparameterize")

    pbar = tqdm(train_loader, desc="Training", leave=False)
    for batch in pbar:
        iq = batch["iq"].to(device)
        snr = batch.get("snr")
        if snr is not None:
            snr = snr.to(device)

        optimizer.zero_grad()

        # Forward pass
        if is_snr_conditioned and snr is not None:
            x_recon, mu, logvar, _ = model(iq, snr)
            loss, recon_loss, kl_loss = model.loss(iq, x_recon, mu, logvar)
        elif is_vae:
            x_recon, mu, logvar, _ = model(iq)
            loss, recon_loss, kl_loss = model.loss(iq, x_recon, mu, logvar)
        else:
            x_recon, _ = model(iq)
            loss = model.reconstruction_loss(iq, x_recon)
            recon_loss = loss
            kl_loss = torch.tensor(0.0)

        # Backward pass
        loss.backward()

        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)

        optimizer.step()

        # Accumulate metrics
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl_loss += kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss
        num_batches += 1

        pbar.set_postfix({"loss": loss.item()})

    return {
        "loss": total_loss / num_batches,
        "recon_loss": total_recon_loss / num_batches,
        "kl_loss": total_kl_loss / num_batches,
    }


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Validate model.

    Args:
        model: Model to validate.
        val_loader: Validation data loader.
        device: Device.

    Returns:
        Dictionary with validation metrics.
    """
    model.eval()

    total_loss = 0.0
    num_batches = 0

    is_snr_conditioned = hasattr(model, "encoder") and hasattr(model.encoder, "snr_embed")
    is_vae = hasattr(model, "reparameterize")

    for batch in val_loader:
        iq = batch["iq"].to(device)
        snr = batch.get("snr")
        if snr is not None:
            snr = snr.to(device)

        if is_snr_conditioned and snr is not None:
            x_recon, mu, logvar, _ = model(iq, snr)
            loss, _, _ = model.loss(iq, x_recon, mu, logvar)
        elif is_vae:
            x_recon, mu, logvar, _ = model(iq)
            loss, _, _ = model.loss(iq, x_recon, mu, logvar)
        else:
            x_recon, _ = model(iq)
            loss = model.reconstruction_loss(iq, x_recon)

        total_loss += loss.item()
        num_batches += 1

    return {"val_loss": total_loss / num_batches}


def evaluate_detection(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    config,
) -> dict[str, float]:
    """Evaluate anomaly detection performance.

    Args:
        model: Trained model.
        test_loader: Test data loader with anomalies.
        device: Device.
        config: Configuration.

    Returns:
        Dictionary with detection metrics.
    """
    # Create and fit detector
    detector = AnomalyDetector(
        model=model,
        method=config.detection.method,
        threshold_method=config.detection.threshold_method,
        threshold_percentile=config.detection.threshold_percentile,
        snr_adaptive=config.detection.snr_adaptive,
        snr_bins=config.detection.snr_bins,
        device=device,
    )

    # Fit on normal data from test set
    detector.fit(test_loader, num_batches=50)

    # Detect anomalies
    scores, predictions, labels = detector.detect_batch(test_loader)

    if labels is None:
        return {"error": "No labels in test data"}

    # Compute metrics
    metrics = compute_metrics(scores, labels, predictions)

    return {
        "auroc": metrics.auroc,
        "auprc": metrics.auprc,
        "f1": metrics.f1,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "threshold": metrics.threshold,
    }


def main():
    """Main training function."""
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Override seed if provided
    if args.seed is not None:
        config.experiment.seed = args.seed

    # Set random seeds
    torch.manual_seed(config.experiment.seed)

    # Setup device
    device = get_device(args.device if args.device != "auto" else config.experiment.device)
    logger.info(f"Using device: {device}")

    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(config.experiment.save_dir) / timestamp

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Save configuration
    save_config(config, output_dir / "config.yaml")

    # Create data generator
    logger.info("Creating synthetic data generator...")
    generator = SyntheticRFGenerator(
        sequence_length=config.data.sequence_length,
        sample_rate=config.data.sample_rate,
        seed=config.experiment.seed,
    )

    # Create data loaders
    logger.info("Creating data loaders...")
    train_loader, val_loader, test_loader = create_dataloaders(config, generator)
    logger.info(f"Training samples: {len(train_loader.dataset)}")
    logger.info(f"Validation samples: {len(val_loader.dataset)}")
    logger.info(f"Test samples: {len(test_loader.dataset)}")

    # Create model
    logger.info(f"Creating model: {config.model.type}")
    model = create_model(config)
    model = model.to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {num_params:,}")

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    # Create scheduler
    if config.training.scheduler.type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.training.scheduler.T_max,
            eta_min=config.training.scheduler.min_lr,
        )
    elif config.training.scheduler.type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=30, gamma=0.1
        )
    else:
        scheduler = None

    # Training loop
    logger.info("Starting training...")
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train": [], "val": []}

    for epoch in range(1, config.training.num_epochs + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            gradient_clip_norm=config.training.gradient_clip_norm,
        )

        # Validate
        val_metrics = validate(model, val_loader, device)

        # Update scheduler
        if scheduler is not None:
            scheduler.step()

        # Log progress
        lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch {epoch}/{config.training.num_epochs} - "
            f"Train Loss: {train_metrics['loss']:.4f} - "
            f"Val Loss: {val_metrics['val_loss']:.4f} - "
            f"LR: {lr:.2e}"
        )

        # Save history
        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        # Check for improvement
        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            patience_counter = 0

            # Save best model
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_metrics["val_loss"],
            }, output_dir / "best_model.pt")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config.training.early_stopping_patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

        # Periodic checkpoint
        if epoch % config.experiment.checkpoint_interval == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, output_dir / f"checkpoint_epoch_{epoch}.pt")

    # Load best model for evaluation
    checkpoint = torch.load(output_dir / "best_model.pt", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Evaluate detection performance
    logger.info("Evaluating detection performance...")
    detection_metrics = evaluate_detection(model, test_loader, device, config)

    logger.info("Detection Results:")
    for key, value in detection_metrics.items():
        logger.info(f"  {key}: {value:.4f}")

    # Save final results
    results = {
        "best_val_loss": best_val_loss,
        "best_epoch": checkpoint["epoch"],
        "detection_metrics": detection_metrics,
        "config": config.to_dict(),
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save training history
    with open(output_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"Training complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
