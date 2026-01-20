#!/usr/bin/env python3
"""Test model on recorded HDF5 files.

Evaluates the anomaly detection model on previously recorded datasets,
computing metrics like AUROC, AUPRC, and F1 score.

Usage:
    python scripts/replay_test.py --input session.h5 --model ../snr_conditioned_vae_hybrid_v1.pt
    python scripts/replay_test.py --input session.h5 --save-plots
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# Add CLP_Project root first, then testbed
_CLP_ROOT = Path(__file__).parent.parent.parent
_TESTBED_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_CLP_ROOT))
sys.path.insert(0, str(_TESTBED_ROOT))

from TorchRF_Testbed.src.detector import LiveDetector, load_detector
from TorchRF_Testbed.src.recorder import SessionReader


def compute_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float | None = None) -> dict:
    """Compute detection metrics.

    Args:
        labels: Ground truth labels (True = anomaly).
        scores: Detection scores.
        threshold: Decision threshold. If None, uses optimal threshold.

    Returns:
        Dict with metrics.
    """
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        confusion_matrix,
    )

    # AUROC and AUPRC
    auroc = roc_auc_score(labels, scores)
    auprc = average_precision_score(labels, scores)

    # Find optimal threshold if not provided
    if threshold is None:
        # Use Youden's J statistic
        from sklearn.metrics import roc_curve
        fpr, tpr, thresholds = roc_curve(labels, scores)
        j_scores = tpr - fpr
        threshold = thresholds[np.argmax(j_scores)]

    # Binary predictions
    predictions = scores > threshold

    # Classification metrics
    f1 = f1_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()

    return {
        "auroc": auroc,
        "auprc": auprc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "threshold": threshold,
        "true_positives": int(tp),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
    }


def compute_per_type_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    anomaly_types: list[str],
    threshold: float,
) -> dict:
    """Compute metrics per anomaly type.

    Args:
        labels: Ground truth labels.
        scores: Detection scores.
        anomaly_types: Anomaly type for each sample.
        threshold: Decision threshold.

    Returns:
        Dict mapping anomaly type to metrics.
    """
    results = {}

    # Get unique types (excluding empty string for normal)
    unique_types = set(anomaly_types) - {""}

    for atype in unique_types:
        # Get indices for this type and normal samples
        type_mask = np.array([t == atype for t in anomaly_types])
        normal_mask = np.array([t == "" for t in anomaly_types])

        # Create subset with this type vs normal
        subset_mask = type_mask | normal_mask
        subset_labels = labels[subset_mask]
        subset_scores = scores[subset_mask]

        if len(subset_labels) > 0 and subset_labels.sum() > 0:
            try:
                metrics = compute_metrics(subset_labels, subset_scores, threshold)
                results[atype] = metrics
            except:
                results[atype] = {"error": "Could not compute metrics"}

    return results


def replay_test(
    input_path: str,
    model_path: str | None = None,
    config_path: str | None = None,
    device: str = "cpu",
    save_plots: bool = False,
    output_dir: str | None = None,
    verbose: bool = True,
) -> dict:
    """Test model on recorded dataset.

    Args:
        input_path: Path to input HDF5 file.
        model_path: Path to model checkpoint.
        config_path: Path to model config.
        device: Device for inference.
        save_plots: Whether to save ROC/PR curves.
        output_dir: Directory for output plots.
        verbose: Print progress and results.

    Returns:
        Dict with evaluation results.
    """
    # Load detector
    detector = load_detector(model_path, config_path, device)

    # Load dataset
    reader = SessionReader(input_path)
    metadata = reader.get_metadata()

    if verbose:
        print(f"\nEvaluating on: {input_path}")
        print(f"  Samples: {len(reader)}")
        print(f"  Sample rate: {metadata.get('sample_rate', 'N/A')} Hz")
        print(f"  Center freq: {metadata.get('center_freq', 'N/A')} Hz")
        print()

    # Collect predictions
    all_scores = []
    all_labels = []
    all_types = []
    all_snrs = []

    for i, sample in enumerate(reader):
        signal = sample["signal"]
        label = sample["label"]
        atype = sample["anomaly_type"]
        if isinstance(atype, bytes):
            atype = atype.decode("utf-8")

        # Run detection
        result = detector.detect(signal)

        all_scores.append(result.score)
        all_labels.append(label)
        all_types.append(atype)
        all_snrs.append(sample["snr_db"])

        # Progress
        if verbose and (i + 1) % max(1, len(reader) // 10) == 0:
            print(f"  Progress: {100*(i+1)/len(reader):.0f}%")

    reader.close()

    # Convert to arrays
    scores = np.array(all_scores)
    labels = np.array(all_labels)
    snrs = np.array(all_snrs)

    # Compute overall metrics
    overall_metrics = compute_metrics(labels, scores)

    # Compute per-type metrics
    per_type_metrics = compute_per_type_metrics(labels, scores, all_types, overall_metrics["threshold"])

    # Compute per-SNR metrics
    snr_bins = [(-5, 5), (5, 15), (15, 25), (25, 40)]
    per_snr_metrics = {}
    for low, high in snr_bins:
        mask = (snrs >= low) & (snrs < high)
        if mask.sum() > 0 and labels[mask].sum() > 0:
            try:
                per_snr_metrics[f"{low}-{high}dB"] = compute_metrics(
                    labels[mask], scores[mask], overall_metrics["threshold"]
                )
            except:
                pass

    results = {
        "input_path": input_path,
        "num_samples": len(labels),
        "num_anomalies": int(labels.sum()),
        "num_normal": int((~labels).sum()),
        "overall": overall_metrics,
        "per_type": per_type_metrics,
        "per_snr": per_snr_metrics,
    }

    # Print results
    if verbose:
        print("\n" + "=" * 60)
        print("Evaluation Results")
        print("=" * 60)
        print(f"\nOverall Metrics:")
        print(f"  AUROC: {overall_metrics['auroc']:.4f}")
        print(f"  AUPRC: {overall_metrics['auprc']:.4f}")
        print(f"  F1 Score: {overall_metrics['f1']:.4f}")
        print(f"  Precision: {overall_metrics['precision']:.4f}")
        print(f"  Recall: {overall_metrics['recall']:.4f}")
        print(f"  Threshold: {overall_metrics['threshold']:.4f}")
        print(f"\n  Confusion Matrix:")
        print(f"    TP: {overall_metrics['true_positives']:5d}  FN: {overall_metrics['false_negatives']:5d}")
        print(f"    FP: {overall_metrics['false_positives']:5d}  TN: {overall_metrics['true_negatives']:5d}")

        if per_type_metrics:
            print(f"\nPer Anomaly Type:")
            for atype, metrics in per_type_metrics.items():
                if "error" not in metrics:
                    print(f"  {atype}:")
                    print(f"    AUROC: {metrics['auroc']:.4f}, F1: {metrics['f1']:.4f}")

        if per_snr_metrics:
            print(f"\nPer SNR Range:")
            for snr_range, metrics in per_snr_metrics.items():
                print(f"  {snr_range}:")
                print(f"    AUROC: {metrics['auroc']:.4f}, F1: {metrics['f1']:.4f}")

    # Save plots if requested
    if save_plots:
        try:
            _save_plots(labels, scores, overall_metrics, output_dir, input_path)
            if verbose:
                print(f"\nPlots saved to: {output_dir or Path(input_path).parent}")
        except ImportError:
            if verbose:
                print("\nWarning: matplotlib not available for plotting")

    return results


def _save_plots(
    labels: np.ndarray,
    scores: np.ndarray,
    metrics: dict,
    output_dir: str | None,
    input_path: str,
) -> None:
    """Save ROC and PR curves."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, precision_recall_curve

    output_dir = Path(output_dir) if output_dir else Path(input_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    base_name = Path(input_path).stem

    # ROC Curve
    fpr, tpr, _ = roc_curve(labels, scores)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'AUROC = {metrics["auroc"]:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / f"{base_name}_roc.png", dpi=150, bbox_inches='tight')
    plt.close()

    # PR Curve
    precision, recall, _ = precision_recall_curve(labels, scores)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'AUPRC = {metrics["auprc"]:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / f"{base_name}_pr.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Score distribution
    plt.figure(figsize=(10, 6))
    plt.hist(scores[~labels], bins=50, alpha=0.5, label='Normal', density=True)
    plt.hist(scores[labels], bins=50, alpha=0.5, label='Anomaly', density=True)
    plt.axvline(metrics["threshold"], color='r', linestyle='--', label=f'Threshold = {metrics["threshold"]:.2f}')
    plt.xlabel('Detection Score')
    plt.ylabel('Density')
    plt.title('Score Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(output_dir / f"{base_name}_scores.png", dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test model on recorded HDF5 files",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input settings
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Input HDF5 file path",
    )

    # Model settings
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to model config",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for inference",
    )

    # Output settings
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save ROC/PR curves and score distribution",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory for output plots",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress output",
    )

    args = parser.parse_args()

    # Validate
    if not Path(args.input).exists():
        parser.error(f"Input file not found: {args.input}")

    # Run evaluation
    replay_test(
        input_path=args.input,
        model_path=args.model,
        config_path=args.config,
        device=args.device,
        save_plots=args.save_plots,
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
