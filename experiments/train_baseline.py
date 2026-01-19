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


def get_model_capabilities(model: nn.Module) -> tuple[bool, bool, bool, bool]:
    """Detect model capabilities once.

    Returns:
        Tuple of (is_snr_conditioned, is_vae, uses_power_conditioning, is_probabilistic).
    """
    is_snr_conditioned = hasattr(model, "encoder") and hasattr(model.encoder, "cond_embed")
    is_vae = hasattr(model, "reparameterize")
    uses_power_conditioning = hasattr(model, "use_power_conditioning") and model.use_power_conditioning
    is_probabilistic = hasattr(model, "probabilistic_decoder") and model.probabilistic_decoder
    return is_snr_conditioned, is_vae, uses_power_conditioning, is_probabilistic


def model_forward(
    model: nn.Module,
    iq: torch.Tensor,
    snr: torch.Tensor | None,
    power: torch.Tensor | None,
    is_snr_conditioned: bool,
    is_vae: bool,
    uses_power_conditioning: bool,
    is_probabilistic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
    """Unified forward pass handling all model types.

    Returns:
        Tuple of (x_recon, x_recon_logvar, mu, logvar, latent).
        x_recon_logvar is None for non-probabilistic models.
    """
    if is_snr_conditioned and snr is not None:
        if uses_power_conditioning and power is not None:
            out = model(iq, snr, power)
        else:
            out = model(iq, snr)
    elif is_vae:
        out = model(iq)
    else:
        x_recon, latent = model(iq)
        return x_recon, None, None, None, latent

    # Handle probabilistic decoder output
    if is_probabilistic:
        # Output: (x_mean, x_logvar, mu, logvar, z)
        x_mean, x_logvar, mu, logvar, z = out
        return x_mean, x_logvar, mu, logvar, z
    else:
        # Output: (x_recon, mu, logvar, z)
        x_recon, mu, logvar, z = out
        return x_recon, None, mu, logvar, z


def compute_loss(
    model: nn.Module,
    iq: torch.Tensor,
    x_recon: torch.Tensor,
    x_recon_logvar: torch.Tensor | None,
    mu: torch.Tensor | None,
    logvar: torch.Tensor | None,
    is_vae: bool,
    is_probabilistic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute loss for any model type.

    Returns:
        Tuple of (total_loss, recon_loss, kl_loss, smooth_loss).
    """
    if is_vae and mu is not None and logvar is not None:
        loss_out = model.loss(iq, x_recon, mu, logvar, x_recon_logvar)
        if len(loss_out) == 4:
            # Probabilistic with smoothness: (total, recon, kl, smooth)
            return loss_out
        else:
            # Standard: (total, recon, kl)
            return loss_out[0], loss_out[1], loss_out[2], torch.tensor(0.0)
    loss = model.reconstruction_loss(iq, x_recon, x_recon_logvar)
    return loss, loss, torch.tensor(0.0), torch.tensor(0.0)


def compute_contrastive_loss(
    iq: torch.Tensor,
    x_recon: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 0.5,
) -> torch.Tensor:
    """Compute contrastive loss that encourages higher error for anomalies.

    Args:
        iq: Original signals [batch, 2, seq_len].
        x_recon: Reconstructed signals.
        labels: Binary labels (0=normal, 1=anomaly).
        margin: Minimum difference between anomaly and normal error.

    Returns:
        Contrastive loss encouraging separation.
    """
    # Per-sample reconstruction error
    recon_error = ((iq - x_recon) ** 2).mean(dim=(1, 2))

    normal_mask = labels == 0
    anomaly_mask = labels == 1

    if normal_mask.sum() == 0 or anomaly_mask.sum() == 0:
        return torch.tensor(0.0, device=iq.device)

    # Normal samples: minimize reconstruction error
    normal_loss = recon_error[normal_mask].mean()

    # Anomaly samples: maximize reconstruction error (encourage error > margin)
    # Use hinge loss: max(0, margin - anomaly_error)
    anomaly_errors = recon_error[anomaly_mask]
    anomaly_loss = torch.relu(margin - anomaly_errors).mean()

    return normal_loss + anomaly_loss


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train RF anomaly detection model")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--output-dir", default=None, help="Output directory (default: checkpoints/<timestamp>)")
    parser.add_argument("--device", default="auto", help="Device to use (auto, cuda, cpu)")
    parser.add_argument("--seed", type=int, default=None, help="Random seed (overrides config)")
    parser.add_argument("--semi-supervised", action="store_true", help="Use semi-supervised training with anomalies")
    parser.add_argument("--train-anomaly-ratio", type=float, default=0.1, help="Anomaly ratio in training for semi-supervised")
    parser.add_argument("--contrastive-weight", type=float, default=1.0, help="Weight for contrastive loss in semi-supervised")
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


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip_norm: float | None = 1.0,
    contrastive_weight: float = 0.0,
) -> dict[str, float]:
    """Train for one epoch.

    Args:
        model: Model to train.
        train_loader: Training data loader.
        optimizer: Optimizer.
        device: Training device.
        gradient_clip_norm: Gradient clipping threshold.
        contrastive_weight: Weight for contrastive loss (0 = disabled).
    """
    model.train()
    is_snr_conditioned, is_vae, uses_power, is_probabilistic = get_model_capabilities(model)

    total_loss = total_recon = total_kl = total_contrastive = total_smooth = 0.0
    num_batches = 0

    for batch in tqdm(train_loader, desc="Training", leave=False):
        iq = batch["iq"].to(device)
        snr = batch.get("snr")
        power = batch.get("power")
        labels = batch.get("label")
        if snr is not None:
            snr = snr.to(device)
        if power is not None:
            power = power.to(device)
        if labels is not None:
            labels = labels.to(device)

        optimizer.zero_grad()

        # Forward and compute loss
        x_recon, x_recon_logvar, mu, logvar, _ = model_forward(
            model, iq, snr, power, is_snr_conditioned, is_vae, uses_power, is_probabilistic
        )
        loss, recon_loss, kl_loss, smooth_loss = compute_loss(
            model, iq, x_recon, x_recon_logvar, mu, logvar, is_vae, is_probabilistic
        )

        # Add contrastive loss if semi-supervised
        contrastive_loss = torch.tensor(0.0, device=device)
        if contrastive_weight > 0 and labels is not None:
            contrastive_loss = compute_contrastive_loss(iq, x_recon, labels)
            loss = loss + contrastive_weight * contrastive_loss

        loss.backward()
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
        optimizer.step()

        total_loss += loss.item()
        total_recon += recon_loss.item()
        total_kl += kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss
        total_contrastive += contrastive_loss.item() if isinstance(contrastive_loss, torch.Tensor) else contrastive_loss
        total_smooth += smooth_loss.item() if isinstance(smooth_loss, torch.Tensor) else smooth_loss
        num_batches += 1

    return {
        "loss": total_loss / num_batches,
        "recon_loss": total_recon / num_batches,
        "kl_loss": total_kl / num_batches,
        "contrastive_loss": total_contrastive / num_batches,
        "smooth_loss": total_smooth / num_batches,
    }


@torch.no_grad()
def validate(model: nn.Module, val_loader: DataLoader, device: torch.device) -> dict[str, float]:
    """Validate model."""
    model.eval()
    is_snr_conditioned, is_vae, uses_power, is_probabilistic = get_model_capabilities(model)

    total_loss = 0.0

    for batch in val_loader:
        iq = batch["iq"].to(device)
        snr = batch.get("snr")
        power = batch.get("power")
        if snr is not None:
            snr = snr.to(device)
        if power is not None:
            power = power.to(device)

        x_recon, x_recon_logvar, mu, logvar, _ = model_forward(
            model, iq, snr, power, is_snr_conditioned, is_vae, uses_power, is_probabilistic
        )
        loss, _, _, _ = compute_loss(model, iq, x_recon, x_recon_logvar, mu, logvar, is_vae, is_probabilistic)
        total_loss += loss.item()

    return {"val_loss": total_loss / len(val_loader)}


def evaluate_detection(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    config
) -> dict[str, float]:
    """Evaluate anomaly detection performance.

    Args:
        model: Trained model.
        train_loader: Training data loader (for fitting detector on normal data).
        test_loader: Test data loader (for evaluating detection).
        device: Device to run inference on.
        config: Configuration object.

    Returns:
        Dictionary of detection metrics.
    """
    # Get config values with defaults for backwards compatibility
    invert_scores = config.detection.get("invert_scores", False)
    hybrid_weights = tuple(config.detection.get("hybrid_weights", [0.5, 0.5]))
    scoring_method = config.detection.get("scoring_method", "auto")

    detector = AnomalyDetector(
        model=model,
        method=config.detection.method,
        threshold_method=config.detection.threshold_method,
        threshold_percentile=config.detection.threshold_percentile,
        snr_adaptive=config.detection.snr_adaptive,
        snr_bins=config.detection.snr_bins,
        invert_scores=invert_scores,
        hybrid_weights=hybrid_weights,
        device=device,
        scoring_method=scoring_method,
    )
    # Fit on training data (assumed normal) to learn threshold and latent statistics
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
        "threshold": metrics.threshold,
        "scoring_method": scoring_method,
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

    # Setup device and output directory
    device = get_device(args.device if args.device != "auto" else config.experiment.device)
    logger.info(f"Using device: {device}")

    output_dir = Path(args.output_dir) if args.output_dir else (
        Path(config.experiment.save_dir) / datetime.now().strftime("%Y%m%d_%H%M%S")
    )
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
    # Get anomaly severity from config (default 1.0)
    anomaly_severity = getattr(config.data, "anomaly_severity", 1.0)
    logger.info(f"Anomaly severity: {anomaly_severity}")

    if args.semi_supervised:
        # For semi-supervised, include some anomalies in training data
        logger.info(f"Semi-supervised mode: {args.train_anomaly_ratio:.0%} anomalies in training")
        train_dataset = RFDataset.from_generator(
            generator=generator,
            num_samples=config.data.num_train_samples,
            anomaly_ratio=args.train_anomaly_ratio,
            snr_range=tuple(config.data.snr_range),
            modulations=config.data.modulations,
            anomaly_types=config.data.anomaly_types,
            anomaly_severity=anomaly_severity,
        )
        val_dataset = RFDataset.from_generator(
            generator=generator,
            num_samples=config.data.num_val_samples,
            anomaly_ratio=config.data.anomaly_ratio,
            snr_range=tuple(config.data.snr_range),
            anomaly_severity=anomaly_severity,
        )
        test_dataset = RFDataset.from_generator(
            generator=generator,
            num_samples=config.data.num_test_samples,
            anomaly_ratio=config.data.anomaly_ratio,
            snr_range=tuple(config.data.snr_range),
            anomaly_severity=anomaly_severity,
        )
        train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=config.training.batch_size)
        test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size)
    else:
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
    scheduler = None
    if config.training.scheduler.type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config.training.scheduler.T_max, eta_min=config.training.scheduler.min_lr
        )
    elif config.training.scheduler.type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # Training loop
    logger.info("Starting training...")
    best_val_loss = float("inf")
    patience_counter = 0
    history = {"train": [], "val": []}

    contrastive_weight = args.contrastive_weight if args.semi_supervised else 0.0

    for epoch in range(1, config.training.num_epochs + 1):
        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            config.training.gradient_clip_norm, contrastive_weight
        )
        val_metrics = validate(model, val_loader, device)

        if scheduler is not None:
            scheduler.step()

        logger.info(
            f"Epoch {epoch}/{config.training.num_epochs} - "
            f"Train Loss: {train_metrics['loss']:.4f} - "
            f"Val Loss: {val_metrics['val_loss']:.4f} - "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        history["train"].append(train_metrics)
        history["val"].append(val_metrics)

        if val_metrics["val_loss"] < best_val_loss:
            best_val_loss = val_metrics["val_loss"]
            patience_counter = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_metrics["val_loss"],
            }, output_dir / "best_model.pt")
        else:
            patience_counter += 1

        if patience_counter >= config.training.early_stopping_patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break

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
    detection_metrics = evaluate_detection(model, train_loader, test_loader, device, config)

    logger.info("Detection Results:")
    for key, value in detection_metrics.items():
        if isinstance(value, (int, float)):
            logger.info(f"  {key}: {value:.4f}")
        else:
            logger.info(f"  {key}: {value}")

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
