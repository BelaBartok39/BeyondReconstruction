#!/usr/bin/env python3
"""Full evaluation pipeline for trained models."""

from __future__ import annotations

import argparse
import json
import logging
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
from src.detection.metrics import (
    compute_metrics,
    compute_snr_stratified_metrics,
    compute_reconstruction_stats,
)
from src.utils.visualization import (
    plot_detection_curves,
    plot_snr_performance,
    plot_score_distribution,
    plot_latent_space,
    plot_reconstruction,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_model_capabilities(model: nn.Module) -> tuple[bool, bool]:
    """Detect model capabilities once."""
    is_snr_conditioned = hasattr(model, "encoder") and hasattr(model.encoder, "snr_embed")
    is_vae = hasattr(model, "reparameterize")
    return is_snr_conditioned, is_vae


def model_forward(
    model: nn.Module,
    iq: torch.Tensor,
    snr: torch.Tensor | None,
    is_snr_conditioned: bool,
    is_vae: bool,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, torch.Tensor]:
    """Unified forward pass handling all model types."""
    if is_snr_conditioned and snr is not None:
        return model(iq, snr)
    if is_vae:
        return model(iq)
    x_recon, latent = model(iq)
    return x_recon, None, None, latent


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate trained model")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--config", default=None, help="Path to config (default: same directory as checkpoint)")
    parser.add_argument("--output-dir", default=None, help="Output directory for results")
    parser.add_argument("--num-test-samples", type=int, default=5000, help="Number of test samples")
    parser.add_argument("--anomaly-ratio", type=float, default=0.2, help="Fraction of anomalous test samples")
    parser.add_argument("--save-plots", action="store_true", help="Save visualization plots")
    parser.add_argument("--invert-scores", action="store_true", help="Invert scores (use when anomalies have lower error)")
    return parser.parse_args()


def get_device() -> torch.device:
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def collect_latents_and_scores(
    model, dataloader: DataLoader, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Collect latent representations and scores from model.

    Returns:
        Tuple of (latents, scores, labels, snr_db).
    """
    model.eval()
    is_snr_conditioned, is_vae = get_model_capabilities(model)

    all_latents, all_scores, all_labels, all_snr_db = [], [], [], []

    with torch.no_grad():
        for batch in dataloader:
            iq = batch["iq"].to(device)
            snr = batch.get("snr")
            if snr is not None:
                snr = snr.to(device)

            # Get latent and reconstruction
            if is_snr_conditioned and snr is not None:
                mu, _ = model.encode(iq, snr)
            elif is_vae:
                mu, _ = model.encode(iq)
            else:
                mu = model.encode(iq)

            x_recon, _, _, _ = model_forward(model, iq, snr, is_snr_conditioned, is_vae)
            error = ((iq - x_recon) ** 2).mean(dim=(1, 2))

            all_latents.append(mu.cpu().numpy())
            all_scores.append(error.cpu().numpy())
            all_labels.append(batch["label"].numpy())
            all_snr_db.append(batch["snr_db"].numpy())

    return (
        np.concatenate(all_latents),
        np.concatenate(all_scores),
        np.concatenate(all_labels),
        np.concatenate(all_snr_db),
    )


def main():
    """Main evaluation function."""
    args = parse_args()
    device = get_device()

    # Find config and setup directories
    checkpoint_dir = Path(args.checkpoint).parent
    config_path = args.config or checkpoint_dir / "config.yaml"
    if not Path(config_path).exists():
        config_path = "configs/default.yaml"

    config = load_config(config_path)
    logger.info(f"Using config: {config_path}")

    output_dir = Path(args.output_dir) if args.output_dir else checkpoint_dir / "evaluation"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model = create_model(config)
    model = model.to(device)

    # Initialize lazy layers with dummy forward pass
    dummy_iq = torch.randn(1, 2, config.data.sequence_length, device=device)
    dummy_snr = torch.tensor([15.0], device=device)
    with torch.no_grad():
        _ = model(dummy_iq, dummy_snr)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Create test data
    logger.info("Generating test data...")
    generator = SyntheticRFGenerator(
        sequence_length=config.data.sequence_length,
        sample_rate=config.data.sample_rate,
        seed=config.experiment.seed + 1000,  # Different seed from training
    )

    test_dataset = RFDataset.from_generator(
        generator=generator,
        num_samples=args.num_test_samples,
        anomaly_ratio=args.anomaly_ratio,
        snr_range=tuple(config.data.snr_range),
        anomaly_types=config.data.anomaly_types,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.training.batch_size,
        shuffle=False,
    )

    # Collect predictions
    logger.info("Running inference...")
    latents, scores, labels, snr_db = collect_latents_and_scores(
        model, test_loader, device
    )

    # Optionally invert scores (when anomalies have lower reconstruction error)
    if args.invert_scores:
        logger.info("Inverting scores (assuming anomalies have lower error)...")
        scores = -scores

    # Overall metrics
    logger.info("\nComputing metrics...")
    overall_metrics = compute_metrics(scores, labels)

    logger.info("\n" + "=" * 50)
    logger.info("OVERALL DETECTION METRICS")
    logger.info("=" * 50)
    for key, value in overall_metrics.to_dict().items():
        logger.info(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

    # SNR-stratified metrics
    snr_metrics = compute_snr_stratified_metrics(
        scores, labels, snr_db,
        num_bins=config.detection.snr_bins,
        snr_range=tuple(config.data.snr_range),
    )

    logger.info("\n" + "=" * 50)
    logger.info("SNR-STRATIFIED METRICS")
    logger.info("=" * 50)

    for i, (snr_bin, metrics) in enumerate(zip(snr_metrics.snr_bins, snr_metrics.metrics_per_bin)):
        logger.info(f"\n  SNR {snr_bin[0]:.0f} to {snr_bin[1]:.0f} dB (n={snr_metrics.sample_counts[i]}):")
        logger.info(f"    AUROC: {metrics.auroc:.4f}, F1: {metrics.f1:.4f}")

    logger.info(f"\n  Summary:")
    summary = snr_metrics.summary()
    logger.info(f"    Mean AUROC: {summary['mean_auroc']:.4f} +/- {summary['std_auroc']:.4f}")
    logger.info(f"    Mean F1: {summary['mean_f1']:.4f} +/- {summary['std_f1']:.4f}")

    # Reconstruction statistics
    recon_stats = compute_reconstruction_stats(scores, labels)
    logger.info("\n" + "=" * 50)
    logger.info("RECONSTRUCTION ERROR STATISTICS")
    logger.info("=" * 50)
    logger.info(f"  Normal: mean={recon_stats['normal']['mean']:.4f}, std={recon_stats['normal']['std']:.4f}")
    logger.info(f"  Anomaly: mean={recon_stats['anomaly']['mean']:.4f}, std={recon_stats['anomaly']['std']:.4f}")
    logger.info(f"  Cohen's d (effect size): {recon_stats.get('cohens_d', 0):.4f}")

    # Save results
    results = {
        "overall_metrics": overall_metrics.to_dict(),
        "snr_stratified_metrics": snr_metrics.to_dict(),
        "snr_summary": snr_metrics.summary(),
        "reconstruction_stats": recon_stats,
        "num_test_samples": args.num_test_samples,
        "anomaly_ratio": args.anomaly_ratio,
    }

    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Generate plots
    if args.save_plots:
        logger.info("\nGenerating plots...")
        save_kwargs = {"dpi": 150, "bbox_inches": "tight"}

        # ROC and PR curves
        plot_detection_curves(scores, labels).savefig(output_dir / "detection_curves.png", **save_kwargs)

        # Score distribution
        plot_score_distribution(scores, labels, threshold=overall_metrics.threshold).savefig(
            output_dir / "score_distribution.png", **save_kwargs
        )

        # SNR performance
        aurocs = [m.auroc for m in snr_metrics.metrics_per_bin]
        f1s = [m.f1 for m in snr_metrics.metrics_per_bin]
        plot_snr_performance(snr_metrics.snr_bins, aurocs, f1s).savefig(
            output_dir / "snr_performance.png", **save_kwargs
        )

        # Latent space
        plot_latent_space(latents, labels, method="pca").savefig(
            output_dir / "latent_space_pca.png", **save_kwargs
        )

        # Sample reconstructions
        with torch.no_grad():
            sample_batch = next(iter(test_loader))
            iq = sample_batch["iq"][:1].to(device)
            snr = sample_batch.get("snr")
            if snr is not None:
                snr = snr[:1].to(device)

            is_snr_conditioned, is_vae = get_model_capabilities(model)
            x_recon, _, _, _ = model_forward(model, iq, snr, is_snr_conditioned, is_vae)
            plot_reconstruction(iq[0], x_recon[0]).savefig(
                output_dir / "sample_reconstruction.png", **save_kwargs
            )

        logger.info(f"Plots saved to {output_dir}")

    logger.info(f"\nEvaluation complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
