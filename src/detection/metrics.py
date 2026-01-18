"""Detection metrics for anomaly detection evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
)


@dataclass
class MetricsResult:
    """Container for detection metrics."""

    # Primary metrics
    auroc: float
    auprc: float
    f1: float
    precision: float
    recall: float

    # At operating point
    threshold: float
    accuracy: float

    # Confusion matrix
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int

    # Additional
    fpr_at_95_tpr: float | None = None  # False positive rate at 95% true positive rate

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "auroc": self.auroc,
            "auprc": self.auprc,
            "f1": self.f1,
            "precision": self.precision,
            "recall": self.recall,
            "threshold": self.threshold,
            "accuracy": self.accuracy,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "true_negatives": self.true_negatives,
            "false_negatives": self.false_negatives,
            "fpr_at_95_tpr": self.fpr_at_95_tpr,
        }


def compute_metrics(
    scores: NDArray[np.float32],
    labels: NDArray[np.int64],
    predictions: NDArray[np.bool_] | None = None,
    threshold: float | None = None,
) -> MetricsResult:
    """Compute comprehensive detection metrics.

    Args:
        scores: Anomaly scores (higher = more anomalous).
        labels: Ground truth labels (1 = anomaly, 0 = normal).
        predictions: Binary predictions. If None, computed from threshold.
        threshold: Detection threshold. If None, optimal F1 threshold is used.

    Returns:
        MetricsResult with all metrics.
    """
    # Ensure arrays
    scores = np.asarray(scores).flatten()
    labels = np.asarray(labels).flatten()

    # Handle edge cases
    if len(np.unique(labels)) < 2:
        # All same class - metrics undefined
        return MetricsResult(
            auroc=0.5,
            auprc=float(np.mean(labels)),
            f1=0.0,
            precision=0.0,
            recall=0.0,
            threshold=0.0,
            accuracy=float(np.mean(labels == 0)),
            true_positives=0,
            false_positives=0,
            true_negatives=int(np.sum(labels == 0)),
            false_negatives=int(np.sum(labels == 1)),
        )

    # Compute ROC-AUC
    auroc = roc_auc_score(labels, scores)

    # Compute PR-AUC
    auprc = average_precision_score(labels, scores)

    # Find optimal threshold if not provided
    if threshold is None:
        threshold = _find_optimal_threshold(scores, labels)

    # Get predictions
    if predictions is None:
        predictions = scores > threshold

    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()

    # Compute metrics
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    accuracy = (tp + tn) / (tp + tn + fp + fn)

    # FPR at 95% TPR
    fpr_at_95_tpr = _compute_fpr_at_tpr(scores, labels, target_tpr=0.95)

    return MetricsResult(
        auroc=float(auroc),
        auprc=float(auprc),
        f1=float(f1),
        precision=float(precision),
        recall=float(recall),
        threshold=float(threshold),
        accuracy=float(accuracy),
        true_positives=int(tp),
        false_positives=int(fp),
        true_negatives=int(tn),
        false_negatives=int(fn),
        fpr_at_95_tpr=fpr_at_95_tpr,
    )


def _find_optimal_threshold(
    scores: NDArray, labels: NDArray, metric: str = "f1"
) -> float:
    """Find threshold that optimizes given metric.

    Args:
        scores: Anomaly scores.
        labels: Ground truth labels.
        metric: Metric to optimize ("f1", "youden").

    Returns:
        Optimal threshold value.
    """
    if metric == "f1":
        precisions, recalls, thresholds = precision_recall_curve(labels, scores)
        # Avoid division by zero
        f1_scores = np.where(
            (precisions + recalls) > 0,
            2 * precisions * recalls / (precisions + recalls),
            0,
        )
        # Best threshold (thresholds array is one shorter)
        if len(thresholds) > 0:
            best_idx = np.argmax(f1_scores[:-1])
            return float(thresholds[best_idx])
        return float(np.median(scores))

    elif metric == "youden":
        # Youden's J statistic = TPR - FPR
        fpr, tpr, thresholds = roc_curve(labels, scores)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        return float(thresholds[best_idx])

    else:
        raise ValueError(f"Unknown metric: {metric}")


def _compute_fpr_at_tpr(
    scores: NDArray, labels: NDArray, target_tpr: float = 0.95
) -> float | None:
    """Compute FPR at specified TPR level.

    Args:
        scores: Anomaly scores.
        labels: Ground truth labels.
        target_tpr: Target true positive rate.

    Returns:
        False positive rate at target TPR, or None if not achievable.
    """
    fpr, tpr, _ = roc_curve(labels, scores)

    # Find FPR at target TPR
    idx = np.searchsorted(tpr, target_tpr)
    if idx < len(fpr):
        return float(fpr[idx])
    return None


@dataclass
class SNRStratifiedMetrics:
    """Metrics stratified by SNR bins."""

    snr_bins: list[tuple[float, float]]
    metrics_per_bin: list[MetricsResult]
    sample_counts: list[int]

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "snr_bins": self.snr_bins,
            "metrics_per_bin": [m.to_dict() for m in self.metrics_per_bin],
            "sample_counts": self.sample_counts,
        }

    def summary(self) -> dict:
        """Get summary statistics across bins."""
        aurocs = [m.auroc for m in self.metrics_per_bin if m.auroc > 0]
        f1s = [m.f1 for m in self.metrics_per_bin if m.f1 > 0]

        return {
            "mean_auroc": float(np.mean(aurocs)) if aurocs else 0.0,
            "std_auroc": float(np.std(aurocs)) if aurocs else 0.0,
            "mean_f1": float(np.mean(f1s)) if f1s else 0.0,
            "std_f1": float(np.std(f1s)) if f1s else 0.0,
            "min_auroc": float(np.min(aurocs)) if aurocs else 0.0,
            "max_auroc": float(np.max(aurocs)) if aurocs else 0.0,
        }


def compute_snr_stratified_metrics(
    scores: NDArray[np.float32],
    labels: NDArray[np.int64],
    snr_db: NDArray[np.float32],
    num_bins: int = 7,
    snr_range: tuple[float, float] = (-5, 30),
    predictions: NDArray[np.bool_] | None = None,
) -> SNRStratifiedMetrics:
    """Compute metrics stratified by SNR level.

    This helps understand detector performance across different
    signal quality conditions.

    Args:
        scores: Anomaly scores.
        labels: Ground truth labels.
        snr_db: SNR values in dB.
        num_bins: Number of SNR bins.
        snr_range: SNR range (min, max).
        predictions: Binary predictions.

    Returns:
        SNRStratifiedMetrics with per-bin results.
    """
    snr_min, snr_max = snr_range
    bin_edges = np.linspace(snr_min, snr_max, num_bins + 1)

    snr_bins = []
    metrics_per_bin = []
    sample_counts = []

    for i in range(num_bins):
        bin_low, bin_high = bin_edges[i], bin_edges[i + 1]
        snr_bins.append((float(bin_low), float(bin_high)))

        # Get samples in this bin
        mask = (snr_db >= bin_low) & (snr_db < bin_high)
        sample_counts.append(int(np.sum(mask)))

        if np.sum(mask) < 10 or len(np.unique(labels[mask])) < 2:
            # Not enough samples or all same class
            metrics_per_bin.append(
                MetricsResult(
                    auroc=0.5,
                    auprc=0.0,
                    f1=0.0,
                    precision=0.0,
                    recall=0.0,
                    threshold=0.0,
                    accuracy=0.0,
                    true_positives=0,
                    false_positives=0,
                    true_negatives=0,
                    false_negatives=0,
                )
            )
        else:
            bin_preds = predictions[mask] if predictions is not None else None
            metrics = compute_metrics(scores[mask], labels[mask], bin_preds)
            metrics_per_bin.append(metrics)

    return SNRStratifiedMetrics(
        snr_bins=snr_bins,
        metrics_per_bin=metrics_per_bin,
        sample_counts=sample_counts,
    )


def compute_reconstruction_stats(
    reconstruction_errors: NDArray[np.float32],
    labels: NDArray[np.int64],
) -> dict:
    """Compute statistics on reconstruction errors.

    Args:
        reconstruction_errors: Per-sample reconstruction errors.
        labels: Ground truth labels.

    Returns:
        Dictionary with statistics.
    """
    normal_mask = labels == 0
    anomaly_mask = labels == 1

    normal_errors = reconstruction_errors[normal_mask]
    anomaly_errors = reconstruction_errors[anomaly_mask]

    stats = {
        "normal": {
            "mean": float(np.mean(normal_errors)),
            "std": float(np.std(normal_errors)),
            "median": float(np.median(normal_errors)),
            "min": float(np.min(normal_errors)),
            "max": float(np.max(normal_errors)),
            "count": int(np.sum(normal_mask)),
        },
        "anomaly": {
            "mean": float(np.mean(anomaly_errors)) if len(anomaly_errors) > 0 else 0,
            "std": float(np.std(anomaly_errors)) if len(anomaly_errors) > 0 else 0,
            "median": float(np.median(anomaly_errors)) if len(anomaly_errors) > 0 else 0,
            "min": float(np.min(anomaly_errors)) if len(anomaly_errors) > 0 else 0,
            "max": float(np.max(anomaly_errors)) if len(anomaly_errors) > 0 else 0,
            "count": int(np.sum(anomaly_mask)),
        },
    }

    # Separation metrics
    if len(anomaly_errors) > 0 and len(normal_errors) > 0:
        # Cohen's d (effect size)
        pooled_std = np.sqrt(
            (np.std(normal_errors) ** 2 + np.std(anomaly_errors) ** 2) / 2
        )
        if pooled_std > 0:
            stats["cohens_d"] = float(
                (np.mean(anomaly_errors) - np.mean(normal_errors)) / pooled_std
            )
        else:
            stats["cohens_d"] = 0.0

        # Overlap coefficient
        from scipy.stats import gaussian_kde

        try:
            kde_normal = gaussian_kde(normal_errors)
            kde_anomaly = gaussian_kde(anomaly_errors)
            x = np.linspace(
                min(normal_errors.min(), anomaly_errors.min()),
                max(normal_errors.max(), anomaly_errors.max()),
                1000,
            )
            overlap = np.trapz(np.minimum(kde_normal(x), kde_anomaly(x)), x)
            stats["distribution_overlap"] = float(overlap)
        except Exception:
            stats["distribution_overlap"] = None

    return stats
