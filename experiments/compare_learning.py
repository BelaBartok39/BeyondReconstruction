#!/usr/bin/env python3
"""Compare online vs periodic learning approaches."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.datasets import RFDataset, StreamingRFDataset
from src.data.synthetic import SyntheticRFGenerator
from src.detection.detector import AnomalyDetector
from src.detection.metrics import compute_metrics
from src.detection.phase_detector import EnhancedFrequencyDetector
from src.learning.ewc import EWCLearner
from src.learning.online import OnlineLearner
from src.learning.periodic import PeriodicRetrainer
from src.learning.ucl import UCLLearner
from src.models.snr_encoder import create_model
from src.utils.config import load_config


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_model_capabilities(model: nn.Module) -> tuple[bool, bool]:
    """Detect model capabilities once.

    Returns:
        Tuple of (is_snr_conditioned, is_vae)
    """
    is_snr_conditioned = hasattr(model, "encoder") and hasattr(model.encoder, "cond_embed")
    is_vae = hasattr(model, "reparameterize")
    return is_snr_conditioned, is_vae


def extract_batch_tensors(
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Extract and move batch tensors to device.

    Returns:
        Tuple of (iq, snr, power) tensors
    """
    iq = batch["iq"].to(device)
    snr = batch.get("snr")
    if snr is not None:
        snr = snr.to(device)
    power = batch.get("power")
    if power is not None:
        power = power.to(device)
    return iq, snr, power


def compute_task_loss(
    model: nn.Module,
    iq: torch.Tensor,
    snr: torch.Tensor | None,
    power: torch.Tensor | None,
    is_snr_conditioned: bool,
    is_vae: bool,
) -> torch.Tensor:
    """Compute task loss for any model type."""
    if is_snr_conditioned:
        if snr is None:
            snr = torch.full((iq.size(0),), 0.5, device=iq.device)
        if power is None:
            power = torch.full((iq.size(0),), 0.5, device=iq.device)
        result = model(iq, snr, power)
        if len(result) == 5:
            x_mean, x_logvar, mu, logvar, _ = result
            loss, _, _ = model.loss(iq, x_mean, mu, logvar, x_logvar)
        else:
            x_recon, mu, logvar, _ = result
            loss, _, _ = model.loss(iq, x_recon, mu, logvar)
    elif is_vae:
        x_recon, mu, logvar, _ = model(iq)
        loss, _, _ = model.loss(iq, x_recon, mu, logvar)
    else:
        x_recon, _ = model(iq)
        loss = model.reconstruction_loss(iq, x_recon)
    return loss


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare continuous learning methods")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--baseline-checkpoint", required=True, help="Path to trained baseline model checkpoint")
    parser.add_argument("--output-dir", default=None, help="Output directory")
    parser.add_argument("--num-streaming-samples", type=int, default=10000, help="Number of streaming samples")
    parser.add_argument("--eval-interval", type=int, default=500, help="Evaluation interval in samples")
    parser.add_argument("--concept-drift", action="store_true", help="Enable concept drift in streaming data")
    parser.add_argument(
        "--detection-method",
        choices=["latent", "hybrid"],
        default="latent",
        help="Detection method: latent (Mahalanobis) or hybrid (latent + freq features)",
    )
    parser.add_argument(
        "--freq-weight",
        type=float,
        default=0.5,
        help="Weight for frequency features in hybrid mode (default: 0.5)",
    )
    return parser.parse_args()


def get_device() -> torch.device:
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_baseline_model(checkpoint_path: str, config: Any, device: torch.device) -> nn.Module:
    """Load trained baseline model."""
    model = create_model(config)
    model = model.to(device)

    dummy_iq = torch.randn(1, 2, config.data.sequence_length, device=device)
    dummy_snr = torch.rand(1, device=device)
    dummy_power = torch.rand(1, device=device) if getattr(config.model, "use_power_conditioning", False) else None

    with torch.no_grad():
        if dummy_power is not None:
            _ = model(dummy_iq, dummy_snr, dummy_power)
        else:
            _ = model(dummy_iq, dummy_snr)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def normalize_scores(scores: np.ndarray) -> np.ndarray:
    """Normalize scores to [0, 1] range."""
    return (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)


def evaluate_model(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    fit_loader: DataLoader | None = None,
    detection_method: str = "latent",
    freq_weight: float = 0.5,
) -> dict[str, float]:
    """Evaluate model on test set using specified detection method.

    Args:
        model: The model to evaluate.
        test_loader: DataLoader with test data (including anomalies).
        device: Device to run on.
        fit_loader: DataLoader with normal-only data for fitting detector.
                   If None, uses test_loader (less accurate).
        detection_method: "latent" or "hybrid"
        freq_weight: Weight for frequency features in hybrid mode
    """
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

    loader_for_fit = fit_loader if fit_loader is not None else test_loader
    detector.fit(loader_for_fit, num_batches=50)

    freq_detector = None
    if detection_method == "hybrid":
        train_iq = np.concatenate([b["iq"].numpy() for b in loader_for_fit])
        freq_detector = EnhancedFrequencyDetector()
        freq_detector.fit(train_iq)

    all_scores = []
    all_labels = []
    all_iq_for_freq = [] if detection_method == "hybrid" else None

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
            all_scores.append(torch.from_numpy(result.scores))
            all_labels.append(batch["label"])

            if detection_method == "hybrid":
                all_iq_for_freq.append(batch["iq"].numpy())

    latent_scores = torch.cat(all_scores).numpy()
    labels = torch.cat(all_labels).numpy()

    if detection_method == "hybrid":
        test_iq = np.concatenate(all_iq_for_freq)
        freq_scores = freq_detector.score(test_iq)
        scores = (1 - freq_weight) * normalize_scores(latent_scores) + freq_weight * normalize_scores(freq_scores)
    else:
        scores = latent_scores

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
    config: Any,
    device: torch.device,
    eval_interval: int,
    fit_loader: DataLoader,
    detection_method: str = "latent",
    freq_weight: float = 0.5,
) -> dict[str, Any]:
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
        learner.update(batch)
        sample_count += batch["iq"].size(0)

        if sample_count % eval_interval == 0:
            eval_metrics = evaluate_model(model, test_loader, device, fit_loader, detection_method, freq_weight)
            eval_metrics["sample_count"] = sample_count
            metrics_history.append(eval_metrics)

    return {
        "method": "online",
        "final_metrics": metrics_history[-1] if metrics_history else {},
        "history": metrics_history,
    }


def run_regularized_learning(
    model: nn.Module,
    stream_loader: DataLoader,
    test_loader: DataLoader,
    config: Any,
    device: torch.device,
    eval_interval: int,
    initial_data_loader: DataLoader,
    detection_method: str = "latent",
    freq_weight: float = 0.5,
    regularizer_type: str = "ewc",
) -> dict[str, Any]:
    """Run online learning with regularization (EWC or UCL).

    Args:
        regularizer_type: "ewc" for Elastic Weight Consolidation, "ucl" for Uncertainty-based CL
    """
    method_name = f"online_{regularizer_type}"
    logger.info(f"Running Online Learning with {regularizer_type.upper()}...")

    if regularizer_type == "ewc":
        regularizer = EWCLearner(
            model=model,
            ewc_lambda=getattr(config.continuous_learning.ewc, "lambda", 1000.0),
            fisher_samples=config.continuous_learning.ewc.fisher_samples,
            device=device,
        )
        regularizer.compute_fisher(initial_data_loader)
    else:
        ucl_lambda = config.continuous_learning.get("ucl", {}).get("lambda", 100.0)
        regularizer = UCLLearner(
            model=model,
            ucl_lambda=ucl_lambda,
            device=device,
        )
        logger.info("Computing initial importance for UCL...")
        for batch in initial_data_loader:
            iq = batch["iq"].to(device)
            snr = batch.get("snr")
            snr = snr.to(device) if snr is not None else None
            with torch.no_grad():
                _ = model(iq, snr) if snr is not None else model(iq)
            break
        regularizer.snapshot()

    optimizer = torch.optim.AdamW(model.parameters(), lr=config.continuous_learning.online.learning_rate)
    is_snr_conditioned, is_vae = get_model_capabilities(model)

    metrics_history = []
    sample_count = 0

    for batch in tqdm(stream_loader, desc=f"Online+{regularizer_type.upper()} Learning"):
        model.train()

        iq, snr, power = extract_batch_tensors(batch, device)

        optimizer.zero_grad()
        task_loss = compute_task_loss(model, iq, snr, power, is_snr_conditioned, is_vae)
        loss = task_loss + regularizer.penalty()
        loss.backward()
        optimizer.step()

        sample_count += batch["iq"].size(0)

        if sample_count % eval_interval == 0:
            eval_metrics = evaluate_model(
                model, test_loader, device, initial_data_loader, detection_method, freq_weight
            )
            eval_metrics["sample_count"] = sample_count
            metrics_history.append(eval_metrics)

    return {
        "method": method_name,
        "final_metrics": metrics_history[-1] if metrics_history else {},
        "history": metrics_history,
    }


def run_periodic_retraining(
    model: nn.Module,
    stream_loader: DataLoader,
    test_loader: DataLoader,
    config: Any,
    device: torch.device,
    eval_interval: int,
    fit_loader: DataLoader,
    detection_method: str = "latent",
    freq_weight: float = 0.5,
) -> dict[str, Any]:
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

    def validation_fn(m: nn.Module) -> dict[str, float]:
        return evaluate_model(m, test_loader, device, fit_loader, detection_method, freq_weight)

    metrics_history = []
    sample_count = 0

    for batch in tqdm(stream_loader, desc="Periodic Retraining"):
        retrainer.add_samples(batch)
        sample_count += batch["iq"].size(0)

        if retrainer.should_retrain():
            retrainer.retrain(validation_fn=validation_fn)

        if sample_count % eval_interval == 0:
            eval_metrics = evaluate_model(model, test_loader, device, fit_loader, detection_method, freq_weight)
            eval_metrics["sample_count"] = sample_count
            metrics_history.append(eval_metrics)

    return {
        "method": "periodic",
        "final_metrics": metrics_history[-1] if metrics_history else {},
        "history": metrics_history,
        "retraining_events": len(retrainer.get_history()),
    }


def run_no_adaptation(
    model: nn.Module,
    stream_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    eval_interval: int,
    fit_loader: DataLoader,
    detection_method: str = "latent",
    freq_weight: float = 0.5,
) -> dict[str, Any]:
    """Run baseline without any adaptation (for comparison)."""
    logger.info("Running No Adaptation Baseline...")

    metrics_history = []
    sample_count = 0

    for batch in tqdm(stream_loader, desc="No Adaptation"):
        sample_count += batch["iq"].size(0)

        if sample_count % eval_interval == 0:
            eval_metrics = evaluate_model(model, test_loader, device, fit_loader, detection_method, freq_weight)
            eval_metrics["sample_count"] = sample_count
            metrics_history.append(eval_metrics)

    return {
        "method": "no_adaptation",
        "final_metrics": metrics_history[-1] if metrics_history else {},
        "history": metrics_history,
    }


def create_data_loaders(
    config: Any,
    args: argparse.Namespace,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create test, stream, and initial data loaders.

    Returns:
        Tuple of (test_loader, stream_loader, initial_loader)
    """
    generator = SyntheticRFGenerator(
        sequence_length=config.data.sequence_length,
        sample_rate=config.data.sample_rate,
        seed=config.experiment.seed,
    )

    test_dataset = RFDataset.from_generator(
        generator=generator,
        num_samples=2000,
        anomaly_ratio=config.data.anomaly_ratio,
        snr_range=tuple(config.data.snr_range),
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    stream_dataset = StreamingRFDataset(
        generator=generator,
        samples_per_epoch=args.num_streaming_samples,
        anomaly_ratio=0.0,
        snr_range=tuple(config.data.snr_range),
        concept_drift=args.concept_drift,
        drift_rate=0.001 if args.concept_drift else 0.0,
    )
    stream_loader = DataLoader(stream_dataset, batch_size=config.training.batch_size)

    initial_dataset = RFDataset.from_generator(
        generator=generator,
        num_samples=1000,
        anomaly_ratio=0.0,
        snr_range=tuple(config.data.snr_range),
    )
    initial_loader = DataLoader(initial_dataset, batch_size=64, shuffle=True)

    return test_loader, stream_loader, initial_loader


def run_experiment(
    name: str,
    run_fn: Callable[..., dict[str, Any]],
    checkpoint_path: str,
    config: Any,
    device: torch.device,
    **kwargs: Any,
) -> dict[str, Any]:
    """Run a single experiment with a fresh model copy."""
    model = load_baseline_model(checkpoint_path, config, device)
    return run_fn(model, **kwargs)


def print_results_summary(results: dict[str, dict[str, Any]]) -> None:
    """Print formatted results summary."""
    logger.info("\n" + "=" * 60)
    logger.info("COMPARISON RESULTS")
    logger.info("=" * 60)

    for method, result in results.items():
        final = result.get("final_metrics", {})
        logger.info(f"\n{method.upper()}:")
        logger.info(f"  AUROC: {final.get('auroc', 0):.4f}")
        logger.info(f"  AUPRC: {final.get('auprc', 0):.4f}")
        logger.info(f"  F1: {final.get('f1', 0):.4f}")


def save_results(results: dict[str, dict[str, Any]], output_dir: Path) -> None:
    """Save results to JSON file."""
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


def main() -> None:
    """Main comparison function."""
    args = parse_args()

    config = load_config(args.config)
    device = get_device()
    logger.info(f"Using device: {device}")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path("results") / f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    test_loader, stream_loader, initial_loader = create_data_loaders(config, args)

    detection_method = args.detection_method
    freq_weight = args.freq_weight
    detection_info = f" (freq_weight={freq_weight})" if detection_method == "hybrid" else ""
    logger.info(f"Detection method: {detection_method}{detection_info}")

    common_kwargs = {
        "stream_loader": stream_loader,
        "test_loader": test_loader,
        "device": device,
        "eval_interval": args.eval_interval,
        "detection_method": detection_method,
        "freq_weight": freq_weight,
    }

    results = {}

    # 1. No adaptation baseline
    model = load_baseline_model(args.baseline_checkpoint, config, device)
    results["no_adaptation"] = run_no_adaptation(
        model, fit_loader=initial_loader, **common_kwargs
    )

    # 2. Online learning
    model = load_baseline_model(args.baseline_checkpoint, config, device)
    results["online"] = run_online_learning(
        model, config=config, fit_loader=initial_loader, **common_kwargs
    )

    # 3. Online + EWC
    model = load_baseline_model(args.baseline_checkpoint, config, device)
    results["online_ewc"] = run_regularized_learning(
        model, config=config, initial_data_loader=initial_loader,
        regularizer_type="ewc", **common_kwargs
    )

    # 4. Periodic retraining
    model = load_baseline_model(args.baseline_checkpoint, config, device)
    results["periodic"] = run_periodic_retraining(
        model, config=config, fit_loader=initial_loader, **common_kwargs
    )

    # 5. Online + UCL (if enabled)
    ucl_enabled = config.continuous_learning.get("ucl", {}).get("enabled", False)
    if ucl_enabled:
        model = load_baseline_model(args.baseline_checkpoint, config, device)
        results["online_ucl"] = run_regularized_learning(
            model, config=config, initial_data_loader=initial_loader,
            regularizer_type="ucl", **common_kwargs
        )

    print_results_summary(results)
    save_results(results, output_dir)
    logger.info(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
