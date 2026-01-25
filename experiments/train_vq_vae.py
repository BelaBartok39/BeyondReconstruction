#!/usr/bin/env python3
"""Train VQ-VAE for RF anomaly detection.

This script trains an SNR-conditioned Vector Quantized VAE on synthetic RF signals.
Key differences from train_baseline.py:
- Uses discrete codebook instead of continuous latent space
- Tracks codebook utilization during training
- Supports both reconstruction and quantization-based anomaly detection

Usage:
    python experiments/train_vq_vae.py --config configs/vq_vae.yaml
    python experiments/train_vq_vae.py --config configs/vq_vae.yaml --num-embeddings 1024
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.datasets import create_dataloaders
from src.models.vq_vae import SNRConditionedVQVAE, create_vq_model
from src.utils.config import load_config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train VQ-VAE for RF anomaly detection")
    parser.add_argument(
        "--config", type=str, default="configs/vq_vae.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--num-embeddings", type=int, default=None,
        help="Override codebook size (K)"
    )
    parser.add_argument(
        "--latent-dim", type=int, default=None,
        help="Override embedding dimension (D)"
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override number of training epochs"
    )
    parser.add_argument(
        "--commitment-cost", type=float, default=None,
        help="Override commitment cost (beta)"
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default=None,
        help="Override checkpoint directory"
    )
    parser.add_argument(
        "--resume", type=str, default=None,
        help="Resume training from checkpoint"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override random seed"
    )
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def extract_batch_tensors(
    batch: dict[str, torch.Tensor],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Extract and move batch tensors to device.

    Returns:
        Tuple of (iq, snr, power, labels).
    """
    iq = batch["iq"].to(device)
    snr = batch.get("snr")
    power = batch.get("power")
    labels = batch.get("label")

    if snr is not None:
        snr = snr.to(device)
    if power is not None:
        power = power.to(device)

    return iq, snr, power, labels


def train_epoch(
    model: SNRConditionedVQVAE,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    config,
) -> dict:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_vq_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(train_loader):
        iq, snr, power, _ = extract_batch_tensors(batch, device)

        optimizer.zero_grad()

        # Forward pass
        x_recon, vq_loss, _ = model(iq, snr, power)

        # Compute loss
        loss, recon_loss, vq_loss_val = model.loss(iq, x_recon, vq_loss)

        # Backward pass
        loss.backward()

        # Gradient clipping
        if hasattr(config.training, "gradient_clip_norm"):
            nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip_norm)

        optimizer.step()

        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_vq_loss += vq_loss_val.item()
        num_batches += 1

        # Log progress
        if batch_idx % config.experiment.log_interval == 0:
            print(f"  Batch {batch_idx}/{len(train_loader)}: "
                  f"Loss={loss.item():.4f} (Recon={recon_loss.item():.4f}, VQ={vq_loss_val.item():.4f})")

    # Get codebook usage
    codebook_stats = model.get_codebook_usage()

    return {
        "loss": total_loss / num_batches,
        "recon_loss": total_recon_loss / num_batches,
        "vq_loss": total_vq_loss / num_batches,
        "codebook_utilization": codebook_stats["utilization"],
        "codebook_perplexity": codebook_stats["perplexity"],
        "active_codes": codebook_stats["active_codes"],
    }


def validate(
    model: SNRConditionedVQVAE,
    val_loader: DataLoader,
    device: torch.device,
) -> dict:
    """Validate the model."""
    model.eval()
    total_loss = 0.0
    total_recon_loss = 0.0
    total_vq_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch in val_loader:
            iq, snr, power, _ = extract_batch_tensors(batch, device)

            x_recon, vq_loss, _ = model(iq, snr, power)
            loss, recon_loss, vq_loss_val = model.loss(iq, x_recon, vq_loss)

            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_vq_loss += vq_loss_val.item()
            num_batches += 1

    return {
        "loss": total_loss / num_batches,
        "recon_loss": total_recon_loss / num_batches,
        "vq_loss": total_vq_loss / num_batches,
    }


def compute_mahalanobis_scores(latents: np.ndarray, train_latents: np.ndarray) -> np.ndarray:
    """Compute Mahalanobis distance scores from training distribution."""
    # Compute mean and covariance from training data
    mean = train_latents.mean(axis=0)
    cov = np.cov(train_latents.T)

    # Add small regularization for numerical stability
    cov += np.eye(cov.shape[0]) * 1e-6

    # Compute inverse covariance
    try:
        cov_inv = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        # If matrix is singular, use pseudo-inverse
        cov_inv = np.linalg.pinv(cov)

    # Compute Mahalanobis distance for each sample
    diff = latents - mean
    scores = np.sqrt(np.sum(diff @ cov_inv * diff, axis=1))

    return scores


def evaluate_detection(
    model: SNRConditionedVQVAE,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate anomaly detection performance."""
    model.eval()

    # Collect training latent codes for detector fitting
    train_latents = []
    with torch.no_grad():
        for batch in train_loader:
            iq, snr, power, _ = extract_batch_tensors(batch, device)

            # Get continuous encoding (before quantization)
            z_e = model.encode_continuous(iq, snr, power)
            train_latents.append(z_e.cpu().numpy())

    train_latents = np.concatenate(train_latents, axis=0)

    # Evaluate on test set
    all_scores = []
    all_labels = []
    all_recon_errors = []
    all_quant_errors = []

    with torch.no_grad():
        for batch in test_loader:
            iq, snr, power, labels = extract_batch_tensors(batch, device)

            # Get continuous encoding
            z_e = model.encode_continuous(iq, snr, power)

            # Compute Mahalanobis scores
            latent_scores = compute_mahalanobis_scores(z_e.cpu().numpy(), train_latents)
            recon_errors = model.get_reconstruction_error(iq, snr, power).cpu().numpy()
            quant_errors = model.get_quantization_error(iq, snr, power).cpu().numpy()

            all_scores.append(latent_scores)
            if labels is not None:
                all_labels.append(labels.numpy())
            all_recon_errors.append(recon_errors)
            all_quant_errors.append(quant_errors)

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels) if all_labels else None
    recon_errors = np.concatenate(all_recon_errors)
    quant_errors = np.concatenate(all_quant_errors)

    if labels is None or labels.sum() == 0:
        return {"error": "No anomaly labels in test set"}

    # Compute metrics for different scoring methods
    results = {}

    # Latent-based (Mahalanobis)
    auroc_latent = roc_auc_score(labels, scores)
    auprc_latent = average_precision_score(labels, scores)
    results["latent"] = {"auroc": auroc_latent, "auprc": auprc_latent}

    # Reconstruction error
    auroc_recon = roc_auc_score(labels, recon_errors)
    auprc_recon = average_precision_score(labels, recon_errors)
    results["reconstruction"] = {"auroc": auroc_recon, "auprc": auprc_recon}

    # Quantization error
    auroc_quant = roc_auc_score(labels, quant_errors)
    auprc_quant = average_precision_score(labels, quant_errors)
    results["quantization"] = {"auroc": auroc_quant, "auprc": auprc_quant}

    # Hybrid (latent + quant)
    scores_norm = (scores - scores.mean()) / (scores.std() + 1e-8)
    quant_norm = (quant_errors - quant_errors.mean()) / (quant_errors.std() + 1e-8)
    hybrid_scores = scores_norm + 0.5 * quant_norm
    auroc_hybrid = roc_auc_score(labels, hybrid_scores)
    auprc_hybrid = average_precision_score(labels, hybrid_scores)
    results["hybrid"] = {"auroc": auroc_hybrid, "auprc": auprc_hybrid}

    return results


def save_checkpoint(
    model: SNRConditionedVQVAE,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    metrics: dict,
    checkpoint_dir: Path,
    is_best: bool = False,
):
    """Save training checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "metrics": metrics,
        "codebook_usage": model.get_codebook_usage(),
    }

    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save latest
    torch.save(checkpoint, checkpoint_dir / "latest_model.pt")

    # Save best
    if is_best:
        torch.save(checkpoint, checkpoint_dir / "best_model.pt")

    # Save epoch checkpoint
    if epoch % 25 == 0:  # Save every 25 epochs
        torch.save(checkpoint, checkpoint_dir / f"epoch_{epoch:03d}.pt")


def main():
    """Main training function."""
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

    # Apply command line overrides
    if args.num_embeddings is not None:
        config.model.num_embeddings = args.num_embeddings
    if args.latent_dim is not None:
        config.model.latent_dim = args.latent_dim
    if args.epochs is not None:
        config.training.num_epochs = args.epochs
    if args.commitment_cost is not None:
        config.model.commitment_cost = args.commitment_cost
    if args.seed is not None:
        config.experiment.seed = args.seed

    # Set random seed
    set_seed(config.experiment.seed)

    # Create checkpoint directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    checkpoint_dir = Path(args.checkpoint_dir or config.experiment.save_dir)
    checkpoint_dir = checkpoint_dir / f"vq_vae_{timestamp}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration
    with open(checkpoint_dir / "config.json", "w") as f:
        json.dump({
            "model": {
                "type": "vq_vae",
                "latent_dim": config.model.latent_dim,
                "num_embeddings": getattr(config.model, "num_embeddings", 512),
                "commitment_cost": getattr(config.model, "commitment_cost", 0.25),
                "hidden_channels": config.model.hidden_channels,
                "use_power_conditioning": getattr(config.model, "use_power_conditioning", False),
            },
            "training": {
                "num_epochs": config.training.num_epochs,
                "batch_size": config.training.batch_size,
                "learning_rate": config.training.learning_rate,
            },
            "data": {
                "sequence_length": config.data.sequence_length,
                "num_train_samples": config.data.num_train_samples,
                "anomaly_severity": config.data.anomaly_severity,
            }
        }, f, indent=2)

    # Set device
    if config.experiment.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.experiment.device)
    print(f"Using device: {device}")

    # Create dataloaders using the standard helper
    train_loader, val_loader, test_loader = create_dataloaders(config)
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")

    # Create model
    model = create_vq_model(config)
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    print(f"Codebook size: {model.num_embeddings} x {model.embedding_dim}")

    # Create optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.training.num_epochs,
        eta_min=config.training.scheduler.min_lr,
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float("inf")

    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint["scheduler_state_dict"]:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["metrics"].get("val_loss", float("inf"))
        print(f"Resumed from epoch {start_epoch}")

    # Training loop
    print(f"\n{'='*60}")
    print(f"Starting VQ-VAE training for {config.training.num_epochs} epochs")
    print(f"{'='*60}\n")

    patience_counter = 0
    training_history = []

    for epoch in range(start_epoch, config.training.num_epochs):
        epoch_start = time.time()

        # Train
        train_metrics = train_epoch(model, train_loader, optimizer, device, epoch, config)

        # Validate
        val_metrics = validate(model, val_loader, device)

        # Step scheduler
        scheduler.step()

        epoch_time = time.time() - epoch_start

        # Log progress
        print(f"\nEpoch {epoch + 1}/{config.training.num_epochs} ({epoch_time:.1f}s)")
        print(f"  Train: Loss={train_metrics['loss']:.4f} (Recon={train_metrics['recon_loss']:.4f}, VQ={train_metrics['vq_loss']:.4f})")
        print(f"  Valid: Loss={val_metrics['loss']:.4f} (Recon={val_metrics['recon_loss']:.4f}, VQ={val_metrics['vq_loss']:.4f})")
        print(f"  Codebook: {train_metrics['active_codes']}/{model.num_embeddings} active "
              f"({train_metrics['codebook_utilization']*100:.1f}%), perplexity={train_metrics['codebook_perplexity']:.1f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.2e}")

        # Save metrics
        epoch_metrics = {
            "epoch": epoch + 1,
            "train": train_metrics,
            "val": val_metrics,
            "lr": scheduler.get_last_lr()[0],
        }
        training_history.append(epoch_metrics)

        # Check for improvement
        is_best = val_metrics["loss"] < best_val_loss
        if is_best:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            print(f"  *** New best validation loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1

        # Save checkpoint
        save_checkpoint(
            model, optimizer, scheduler, epoch + 1,
            {"train": train_metrics, "val": val_metrics, "val_loss": val_metrics["loss"]},
            checkpoint_dir, is_best
        )

        # Early stopping
        if patience_counter >= config.training.early_stopping_patience:
            print(f"\nEarly stopping at epoch {epoch + 1} (no improvement for {patience_counter} epochs)")
            break

        # Periodic evaluation
        if (epoch + 1) % 25 == 0:
            print("\n  Running detection evaluation...")
            detection_results = evaluate_detection(model, train_loader, test_loader, device)
            if "error" not in detection_results:
                for method, metrics in detection_results.items():
                    print(f"    {method}: AUROC={metrics['auroc']:.4f}, AUPRC={metrics['auprc']:.4f}")

    # Final evaluation
    print(f"\n{'='*60}")
    print("Final Evaluation")
    print(f"{'='*60}")

    # Load best model
    best_checkpoint = torch.load(checkpoint_dir / "best_model.pt", map_location=device)
    model.load_state_dict(best_checkpoint["model_state_dict"])

    # Full evaluation
    detection_results = evaluate_detection(model, train_loader, test_loader, device)

    print("\nDetection Performance:")
    if "error" in detection_results:
        print(f"  {detection_results['error']}")
    else:
        for method, metrics in detection_results.items():
            print(f"  {method}: AUROC={metrics['auroc']:.4f}, AUPRC={metrics['auprc']:.4f}")

    # Save final results
    final_results = {
        "training_history": training_history,
        "best_epoch": best_checkpoint["epoch"],
        "best_val_loss": best_val_loss,
        "detection_results": detection_results,
        "codebook_usage": model.get_codebook_usage(),
    }

    with open(checkpoint_dir / "results.json", "w") as f:
        json.dump(final_results, f, indent=2)

    print(f"\nResults saved to: {checkpoint_dir}")
    print(f"Best model: {checkpoint_dir / 'best_model.pt'}")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    if "error" not in detection_results:
        print(f"Best AUROC (latent): {detection_results['latent']['auroc']:.4f}")
        print(f"Best AUROC (hybrid): {detection_results['hybrid']['auroc']:.4f}")
    print(f"Codebook utilization: {model.get_codebook_usage()['utilization']*100:.1f}%")

    return detection_results


if __name__ == "__main__":
    main()
