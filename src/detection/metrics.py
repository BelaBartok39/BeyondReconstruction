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
    scores = np.asarray(scores).flatten()
    labels = np.asarray(labels).flatten()

    # Handle edge case: all same class
    if len(np.unique(labels)) < 2:
        return MetricsResult(
            auroc=0.5,
            auprc=float(labels.mean()),
            f1=0.0, precision=0.0, recall=0.0,
            threshold=0.0,
            accuracy=float((labels == 0).mean()),
            true_positives=0, false_positives=0,
            true_negatives=int((labels == 0).sum()),
            false_negatives=int((labels == 1).sum()),
        )

    # Compute ROC-AUC and PR-AUC
    auroc = roc_auc_score(labels, scores)
    auprc = average_precision_score(labels, scores)

    # Get threshold and predictions
    threshold = threshold or _find_optimal_threshold(scores, labels)
    predictions = predictions if predictions is not None else scores > threshold

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(labels, predictions).ravel()

    return MetricsResult(
        auroc=float(auroc),
        auprc=float(auprc),
        f1=float(f1_score(labels, predictions, zero_division=0)),
        precision=float(precision_score(labels, predictions, zero_division=0)),
        recall=float(recall_score(labels, predictions, zero_division=0)),
        threshold=float(threshold),
        accuracy=float((tp + tn) / (tp + tn + fp + fn)),
        true_positives=int(tp),
        false_positives=int(fp),
        true_negatives=int(tn),
        false_negatives=int(fn),
        fpr_at_95_tpr=_compute_fpr_at_tpr(scores, labels, 0.95),
    )


def _find_optimal_threshold(scores: NDArray, labels: NDArray, metric: str = "f1") -> float:
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
        if len(thresholds) == 0:
            return float(np.median(scores))
        f1_scores = np.where(
            (precisions + recalls) > 0,
            2 * precisions * recalls / (precisions + recalls),
            0
        )
        return float(thresholds[np.argmax(f1_scores[:-1])])

    if metric == "youden":
        fpr, tpr, thresholds = roc_curve(labels, scores)
        return float(thresholds[np.argmax(tpr - fpr)])

    raise ValueError(f"Unknown metric: {metric}")


def _compute_fpr_at_tpr(scores: NDArray, labels: NDArray, target_tpr: float = 0.95) -> float | None:
    """Compute FPR at specified TPR level.

    Args:
        scores: Anomaly scores.
        labels: Ground truth labels.
        target_tpr: Target true positive rate.

    Returns:
        False positive rate at target TPR, or None if not achievable.
    """
    fpr, tpr, _ = roc_curve(labels, scores)
    idx = np.searchsorted(tpr, target_tpr)
    return float(fpr[idx]) if idx < len(fpr) else None


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

        def safe_stat(arr, func):
            return float(func(arr)) if arr else 0.0

        return {
            "mean_auroc": safe_stat(aurocs, np.mean),
            "std_auroc": safe_stat(aurocs, np.std),
            "mean_f1": safe_stat(f1s, np.mean),
            "std_f1": safe_stat(f1s, np.std),
            "min_auroc": safe_stat(aurocs, np.min),
            "max_auroc": safe_stat(aurocs, np.max),
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
    bin_edges = np.linspace(*snr_range, num_bins + 1)
    snr_bins, metrics_per_bin, sample_counts = [], [], []

    for i in range(num_bins):
        bin_low, bin_high = bin_edges[i], bin_edges[i + 1]
        snr_bins.append((float(bin_low), float(bin_high)))

        mask = (snr_db >= bin_low) & (snr_db < bin_high)
        sample_counts.append(int(mask.sum()))

        if mask.sum() < 10 or len(np.unique(labels[mask])) < 2:
            metrics_per_bin.append(MetricsResult(
                auroc=0.5, auprc=0.0, f1=0.0, precision=0.0, recall=0.0,
                threshold=0.0, accuracy=0.0,
                true_positives=0, false_positives=0, true_negatives=0, false_negatives=0,
            ))
        else:
            bin_preds = predictions[mask] if predictions is not None else None
            metrics_per_bin.append(compute_metrics(scores[mask], labels[mask], bin_preds))

    return SNRStratifiedMetrics(snr_bins, metrics_per_bin, sample_counts)


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
    normal_errors = reconstruction_errors[labels == 0]
    anomaly_errors = reconstruction_errors[labels == 1]

    def error_stats(errors):
        """Compute stats for error array."""
        if len(errors) == 0:
            return {"mean": 0.0, "std": 0.0, "median": 0.0, "min": 0.0, "max": 0.0, "count": 0}
        return {
            "mean": float(errors.mean()),
            "std": float(errors.std()),
            "median": float(np.median(errors)),
            "min": float(errors.min()),
            "max": float(errors.max()),
            "count": len(errors),
        }

    stats = {
        "normal": error_stats(normal_errors),
        "anomaly": error_stats(anomaly_errors),
    }

    # Separation metrics
    if len(anomaly_errors) > 0 and len(normal_errors) > 0:
        pooled_std = np.sqrt((normal_errors.std() ** 2 + anomaly_errors.std() ** 2) / 2)
        stats["cohens_d"] = (
            float((anomaly_errors.mean() - normal_errors.mean()) / pooled_std)
            if pooled_std > 0 else 0.0
        )

        # Overlap coefficient
        from scipy.stats import gaussian_kde
        try:
            x = np.linspace(
                min(normal_errors.min(), anomaly_errors.min()),
                max(normal_errors.max(), anomaly_errors.max()),
                1000
            )
            overlap = np.trapz(
                np.minimum(gaussian_kde(normal_errors)(x), gaussian_kde(anomaly_errors)(x)),
                x
            )
            stats["distribution_overlap"] = float(overlap)
        except Exception:
            stats["distribution_overlap"] = None

    return stats
