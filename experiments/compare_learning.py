#!/usr/bin/env python3
"""Compare online vs periodic learning approaches."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config, save_config
from src.data.synthetic import SyntheticRFGenerator
from src.data.datasets import RFDataset, StreamingRFDataset
from src.models.snr_encoder import create_model
from src.detection.detector import AnomalyDetector
from src.detection.metrics import compute_metrics
from src.learning.online import OnlineLearner
from src.learning.periodic import PeriodicRetrainer, RetrainingTrigger
from src.learning.ewc import EWCLearner
from src.learning.ucl import UCLLearner
from src.learning.replay_buffer import ReplayBuffer


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
) -> torch.Tensor:
    """Unified forward pass returning reconstruction only."""
    if is_snr_conditioned and snr is not None:
        x_recon, _, _, _ = model(iq, snr)
    elif is_vae:
        x_recon, _, _, _ = model(iq)
    else:
        x_recon, _ = model(iq)
    return x_recon


def compute_task_loss(
    model: nn.Module,
    iq: torch.Tensor,
    snr: torch.Tensor | None,
    is_snr_conditioned: bool,
    is_vae: bool,
) -> torch.Tensor:
    """Compute task loss for any model type."""
    if is_snr_conditioned and snr is not None:
        x_recon, mu, logvar, _ = model(iq, snr)
        loss, _, _ = model.loss(iq, x_recon, mu, logvar)
    elif is_vae:
        x_recon, mu, logvar, _ = model(iq)
        loss, _, _ = model.loss(iq, x_recon, mu, logvar)
    else:
        x_recon, _ = model(iq)
        loss = model.reconstruction_loss(iq, x_recon)
    return loss


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare continuous learning methods")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--baseline-checkpoint", required=True, help="Path to trained baseline model checkpoint")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--num-streaming-samples", type=int, default=10000, help="Number of streaming samples")
    parser.add_argument("--eval-interval", type=int, default=500, help="Evaluation interval in samples")
    parser.add_argument("--concept-drift", action="store_true", help="Enable concept drift in streaming data")
    return parser.parse_args()


def get_device() -> torch.device:
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_baseline_model(checkpoint_path: str, config, device: torch.device) -> nn.Module:
    """Load trained baseline model."""
    model = create_model(config)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    return model


def evaluate_model(model: nn.Module, test_loader: DataLoader, device: torch.device) -> dict[str, float]:
    """Evaluate model on test set."""
    model.eval()
    is_snr_conditioned, is_vae = get_model_capabilities(model)

    all_scores, all_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            iq = batch["iq"].to(device)
            snr = batch.get("snr")
            if snr is not None:
                snr = snr.to(device)

            x_recon = model_forward(model, iq, snr, is_snr_conditioned, is_vae)
            error = ((iq - x_recon) ** 2).mean(dim=(1, 2))

            all_scores.append(error.cpu())
            all_labels.append(batch["label"])

    scores = torch.cat(all_scores).numpy()
    labels = torch.cat(all_labels).numpy()
    metrics = compute_metrics(scores, labels)

    return {
        "auroc": metrics.auroc,
        "auprc": metrics.auprc,
        "f1": metrics.f1,
        "loss": float(scores.mean()),
    }


def run_online_learning(
    model: nn.Module,
    stream_loader: DataLoader,
    test_loader: DataLoader,
    config,
    device: torch.device,
    eval_interval: int,
) -> dict:
    """Run online learning experiment."""
    logger.info("Running Online Learning...")

    learner = OnlineLearner(
        model=model,
        learning_rate=config.continuous_learning.online.learning_rate,
        update_frequency=config.continuous_learning.online.update_frequency,
        device=device,
    )

    metrics_history = []
    sample_count = 0

    for batch in tqdm(stream_loader, desc="Online Learning"):
        # Update model
        learner.update(batch)
        sample_count += batch["iq"].size(0)

        # Periodic evaluation
        if sample_count % eval_interval == 0:
            eval_metrics = evaluate_model(model, test_loader, device)
            eval_metrics["sample_count"] = sample_count
            metrics_history.append(eval_metrics)

    return {
        "method": "online",
        "final_metrics": metrics_history[-1] if metrics_history else {},
        "history": metrics_history,
    }


def run_online_ewc_learning(
    model: nn.Module,
    stream_loader: DataLoader,
    test_loader: DataLoader,
    config,
    device: torch.device,
    eval_interval: int,
    initial_data_loader: DataLoader,
) -> dict:
    """Run online learning with EWC."""
    logger.info("Running Online Learning with EWC...")

    ewc = EWCLearner(
        model=model,
        ewc_lambda=config.continuous_learning.ewc.lambda_,
        fisher_samples=config.continuous_learning.ewc.fisher_samples,
        device=device,
    )
    ewc.compute_fisher(initial_data_loader)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.continuous_learning.online.learning_rate)
    is_snr_conditioned, is_vae = get_model_capabilities(model)

    metrics_history = []
    sample_count = 0

    for batch in tqdm(stream_loader, desc="Online+EWC Learning"):
        model.train()

        iq = batch["iq"].to(device)
        snr = batch.get("snr")
        if snr is not None:
            snr = snr.to(device)

        optimizer.zero_grad()
        task_loss = compute_task_loss(model, iq, snr, is_snr_conditioned, is_vae)
        loss = task_loss + ewc.penalty()
        loss.backward()
        optimizer.step()

        sample_count += batch["iq"].size(0)

        if sample_count % eval_interval == 0:
            eval_metrics = evaluate_model(model, test_loader, device)
            eval_metrics["sample_count"] = sample_count
            metrics_history.append(eval_metrics)

    return {
        "method": "online_ewc",
        "final_metrics": metrics_history[-1] if metrics_history else {},
        "history": metrics_history,
    }


def run_periodic_retraining(
    model: nn.Module,
    stream_loader: DataLoader,
    test_loader: DataLoader,
    config,
    device: torch.device,
    eval_interval: int,
) -> dict:
    """Run periodic retraining experiment."""
    logger.info("Running Periodic Retraining...")

    retrainer = PeriodicRetrainer(
        model=model,
        interval=config.continuous_learning.periodic.interval,
        epochs_per_retrain=config.continuous_learning.periodic.epochs_per_retrain,
        learning_rate=config.training.learning_rate,
        buffer_size=config.continuous_learning.replay.buffer_size,
        replay_ratio=config.continuous_learning.replay.replay_batch_ratio,
        device=device,
    )

    metrics_history = []
    sample_count = 0

    for batch in tqdm(stream_loader, desc="Periodic Retraining"):
        retrainer.add_samples(batch)
        sample_count += batch["iq"].size(0)

        if retrainer.should_retrain():
            retrainer.retrain(validation_fn=lambda m: evaluate_model(m, test_loader, device))

        if sample_count % eval_interval == 0:
            eval_metrics = evaluate_model(model, test_loader, device)
            eval_metrics["sample_count"] = sample_count
            metrics_history.append(eval_metrics)

    return {
        "method": "periodic",
        "final_metrics": metrics_history[-1] if metrics_history else {},
        "history": metrics_history,
        "retraining_events": len(retrainer.get_history()),
    }


def run_online_ucl_learning(
    model: nn.Module,
    stream_loader: DataLoader,
    test_loader: DataLoader,
    config,
    device: torch.device,
    eval_interval: int,
    initial_data_loader: DataLoader,
) -> dict:
    """Run online learning with UCL (Uncertainty-based Continual Learning).

    UCL is designed for models with Bayesian layers, where it uses posterior
    variance to determine weight importance.
    """
    logger.info("Running Online Learning with UCL...")

    # Get UCL config with defaults
    ucl_lambda = config.continuous_learning.get("ucl", {}).get("lambda", 100.0)

    ucl = UCLLearner(
        model=model,
        ucl_lambda=ucl_lambda,
        device=device,
    )

    # Initialize UCL with initial data (take snapshot after initial training)
    logger.info("Computing initial importance for UCL...")
    # Do a forward pass to initialize lazy layers
    for batch in initial_data_loader:
        iq = batch["iq"].to(device)
        snr = batch.get("snr")
        snr = snr.to(device) if snr is not None else None
        with torch.no_grad():
            _ = model(iq, snr) if snr is not None else model(iq)
        break

    ucl.snapshot()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.continuous_learning.online.learning_rate)
    is_snr_conditioned, is_vae = get_model_capabilities(model)

    metrics_history = []
    sample_count = 0

    for batch in tqdm(stream_loader, desc="Online+UCL Learning"):
        model.train()

        iq = batch["iq"].to(device)
        snr = batch.get("snr")
        if snr is not None:
            snr = snr.to(device)

        optimizer.zero_grad()
        task_loss = compute_task_loss(model, iq, snr, is_snr_conditioned, is_vae)
        loss = task_loss + ucl.penalty()
        loss.backward()
        optimizer.step()

        sample_count += batch["iq"].size(0)

        if sample_count % eval_interval == 0:
            eval_metrics = evaluate_model(model, test_loader, device)
            eval_metrics["sample_count"] = sample_count
            metrics_history.append(eval_metrics)

    return {
        "method": "online_ucl",
        "final_metrics": metrics_history[-1] if metrics_history else {},
        "history": metrics_history,
    }


def run_no_adaptation(
    model: nn.Module,
    stream_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    eval_interval: int,
) -> dict:
    """Run baseline without any adaptation (for comparison)."""
    logger.info("Running No Adaptation Baseline...")

    metrics_history = []
    sample_count = 0

    for batch in tqdm(stream_loader, desc="No Adaptation"):
        sample_count += batch["iq"].size(0)

        # Periodic evaluation (no updates)
        if sample_count % eval_interval == 0:
            eval_metrics = evaluate_model(model, test_loader, device)
            eval_metrics["sample_count"] = sample_count
            metrics_history.append(eval_metrics)

    return {
        "method": "no_adaptation",
        "final_metrics": metrics_history[-1] if metrics_history else {},
        "history": metrics_history,
    }


def main():
    """Main comparison function."""
    args = parse_args()

    # Load configuration and setup
    config = load_config(args.config)
    device = get_device()
    logger.info(f"Using device: {device}")

    output_dir = Path(args.output_dir) if args.output_dir else (
        Path("results") / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    # Create data generator
    generator = SyntheticRFGenerator(
        sequence_length=config.data.sequence_length,
        sample_rate=config.data.sample_rate,
        seed=config.experiment.seed,
    )

    # Create test loader (fixed for all experiments)
    test_dataset = RFDataset.from_generator(
        generator=generator,
        num_samples=2000,
        anomaly_ratio=config.data.anomaly_ratio,
        snr_range=tuple(config.data.snr_range),
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Create streaming data loader
    stream_dataset = StreamingRFDataset(
        generator=generator,
        samples_per_epoch=args.num_streaming_samples,
        anomaly_ratio=0.0,  # Online learning on normal data
        snr_range=tuple(config.data.snr_range),
        concept_drift=args.concept_drift,
        drift_rate=0.001 if args.concept_drift else 0.0,
    )
    stream_loader = DataLoader(stream_dataset, batch_size=config.training.batch_size)

    # Create initial data loader (for EWC Fisher computation)
    initial_dataset = RFDataset.from_generator(
        generator=generator,
        num_samples=1000,
        anomaly_ratio=0.0,
        snr_range=tuple(config.data.snr_range),
    )
    initial_loader = DataLoader(initial_dataset, batch_size=64, shuffle=True)

    # Run experiments
    results = {}

    # 1. No adaptation baseline
    model_no_adapt = load_baseline_model(args.baseline_checkpoint, config, device)
    results["no_adaptation"] = run_no_adaptation(
        model_no_adapt, stream_loader, test_loader, device, args.eval_interval
    )

    # 2. Online learning
    model_online = load_baseline_model(args.baseline_checkpoint, config, device)
    results["online"] = run_online_learning(
        model_online, stream_loader, test_loader, config, device, args.eval_interval
    )

    # 3. Online + EWC
    model_ewc = load_baseline_model(args.baseline_checkpoint, config, device)
    results["online_ewc"] = run_online_ewc_learning(
        model_ewc, stream_loader, test_loader, config, device, args.eval_interval, initial_loader
    )

    # 4. Periodic retraining
    model_periodic = load_baseline_model(args.baseline_checkpoint, config, device)
    results["periodic"] = run_periodic_retraining(
        model_periodic, stream_loader, test_loader, config, device, args.eval_interval
    )

    # 5. Online + UCL (if model has Bayesian layers or we want to compare anyway)
    ucl_enabled = config.continuous_learning.get("ucl", {}).get("enabled", False)
    if ucl_enabled:
        model_ucl = load_baseline_model(args.baseline_checkpoint, config, device)
        results["online_ucl"] = run_online_ucl_learning(
            model_ucl, stream_loader, test_loader, config, device, args.eval_interval, initial_loader
        )

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON RESULTS")
    logger.info("=" * 60)

    for method, result in results.items():
        final = result.get("final_metrics", {})
        logger.info(f"\n{method.upper()}:")
        logger.info(f"  AUROC: {final.get('auroc', 0):.4f}")
        logger.info(f"  AUPRC: {final.get('auprc', 0):.4f}")
        logger.info(f"  F1: {final.get('f1', 0):.4f}")

    # Save results
    serializable_results = {
        method: {
            "method": result["method"],
            "final_metrics": result["final_metrics"],
            "history": result["history"],
        }
        for method, result in results.items()
    }
    with open(output_dir / "comparison_results.json", "w") as f:
        json.dump(serializable_results, f, indent=2)

    logger.info(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
