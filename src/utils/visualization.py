"""Visualization utilities for RF signal analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
import torch
from torch import Tensor

# Set style
plt.style.use("seaborn-v0_8-whitegrid")


def plot_signals(
    signals: NDArray | Tensor,
    titles: Sequence[str] | None = None,
    figsize: tuple[int, int] = (12, 8),
    max_signals: int = 4,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot IQ signals in time domain.

    Args:
        signals: Signals to plot [N, 2, seq_len] or [2, seq_len].
        titles: Optional titles for each subplot.
        figsize: Figure size.
        max_signals: Maximum number of signals to plot.
        save_path: Optional path to save figure.

    Returns:
        Matplotlib figure.
    """
    if isinstance(signals, Tensor):
        signals = signals.detach().cpu().numpy()

    if signals.ndim == 2:
        signals = signals[np.newaxis, ...]

    n_signals = min(len(signals), max_signals)

    fig, axes = plt.subplots(n_signals, 2, figsize=figsize)
    if n_signals == 1:
        axes = axes[np.newaxis, :]

    for i in range(n_signals):
        sig = signals[i]

        # I channel
        axes[i, 0].plot(sig[0], linewidth=0.5, color="blue")
        axes[i, 0].set_ylabel("Amplitude")
        axes[i, 0].set_title(f"I Channel" + (f" - {titles[i]}" if titles else ""))

        # Q channel
        axes[i, 1].plot(sig[1], linewidth=0.5, color="orange")
        axes[i, 1].set_ylabel("Amplitude")
        axes[i, 1].set_title(f"Q Channel" + (f" - {titles[i]}" if titles else ""))

    axes[-1, 0].set_xlabel("Sample")
    axes[-1, 1].set_xlabel("Sample")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_reconstruction(
    original: NDArray | Tensor,
    reconstructed: NDArray | Tensor,
    figsize: tuple[int, int] = (14, 6),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot original vs reconstructed signal.

    Args:
        original: Original signal [2, seq_len].
        reconstructed: Reconstructed signal [2, seq_len].
        figsize: Figure size.
        save_path: Optional save path.

    Returns:
        Matplotlib figure.
    """
    if isinstance(original, Tensor):
        original = original.detach().cpu().numpy()
    if isinstance(reconstructed, Tensor):
        reconstructed = reconstructed.detach().cpu().numpy()

    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # I channel
    axes[0, 0].plot(original[0], label="Original", alpha=0.7)
    axes[0, 0].plot(reconstructed[0], label="Reconstructed", alpha=0.7)
    axes[0, 0].set_title("I Channel")
    axes[0, 0].legend()

    # Q channel
    axes[0, 1].plot(original[1], label="Original", alpha=0.7)
    axes[0, 1].plot(reconstructed[1], label="Reconstructed", alpha=0.7)
    axes[0, 1].set_title("Q Channel")
    axes[0, 1].legend()

    # Reconstruction error
    error_i = original[0] - reconstructed[0]
    error_q = original[1] - reconstructed[1]

    axes[1, 0].plot(error_i, color="red", alpha=0.7)
    axes[1, 0].set_title("I Channel Error")
    axes[1, 0].set_xlabel("Sample")

    axes[1, 1].plot(error_q, color="red", alpha=0.7)
    axes[1, 1].set_title("Q Channel Error")
    axes[1, 1].set_xlabel("Sample")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_constellation(
    signal: NDArray | Tensor,
    title: str = "Constellation Diagram",
    figsize: tuple[int, int] = (8, 8),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot IQ constellation diagram.

    Args:
        signal: IQ signal [2, seq_len].
        title: Plot title.
        figsize: Figure size.
        save_path: Optional save path.

    Returns:
        Matplotlib figure.
    """
    if isinstance(signal, Tensor):
        signal = signal.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=figsize)

    ax.scatter(signal[0], signal[1], alpha=0.5, s=1)
    ax.set_xlabel("In-phase (I)")
    ax.set_ylabel("Quadrature (Q)")
    ax.set_title(title)
    ax.set_aspect("equal")
    ax.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_latent_space(
    latents: NDArray | Tensor,
    labels: NDArray | Tensor | None = None,
    method: str = "pca",
    figsize: tuple[int, int] = (10, 8),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot latent space visualization.

    Args:
        latents: Latent vectors [N, latent_dim].
        labels: Optional labels for coloring (0=normal, 1=anomaly).
        method: Dimensionality reduction method ("pca", "tsne").
        figsize: Figure size.
        save_path: Optional save path.

    Returns:
        Matplotlib figure.
    """
    if isinstance(latents, Tensor):
        latents = latents.detach().cpu().numpy()
    if isinstance(labels, Tensor):
        labels = labels.detach().cpu().numpy()

    # Dimensionality reduction
    if latents.shape[1] > 2:
        if method == "pca":
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2)
            latents_2d = reducer.fit_transform(latents)
        elif method == "tsne":
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42)
            latents_2d = reducer.fit_transform(latents)
        else:
            raise ValueError(f"Unknown method: {method}")
    else:
        latents_2d = latents

    fig, ax = plt.subplots(figsize=figsize)

    if labels is not None:
        normal_mask = labels == 0
        anomaly_mask = labels == 1

        ax.scatter(
            latents_2d[normal_mask, 0],
            latents_2d[normal_mask, 1],
            c="blue",
            label="Normal",
            alpha=0.5,
            s=10,
        )
        ax.scatter(
            latents_2d[anomaly_mask, 0],
            latents_2d[anomaly_mask, 1],
            c="red",
            label="Anomaly",
            alpha=0.5,
            s=10,
        )
        ax.legend()
    else:
        ax.scatter(latents_2d[:, 0], latents_2d[:, 1], alpha=0.5, s=10)

    ax.set_xlabel(f"{method.upper()} Component 1")
    ax.set_ylabel(f"{method.upper()} Component 2")
    ax.set_title("Latent Space Visualization")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_learning_curves(
    train_losses: Sequence[float],
    val_losses: Sequence[float] | None = None,
    figsize: tuple[int, int] = (10, 6),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot training and validation loss curves.

    Args:
        train_losses: Training losses per epoch.
        val_losses: Optional validation losses per epoch.
        figsize: Figure size.
        save_path: Optional save path.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    epochs = range(1, len(train_losses) + 1)

    ax.plot(epochs, train_losses, label="Training Loss", marker="o", markersize=3)
    if val_losses is not None:
        ax.plot(epochs, val_losses, label="Validation Loss", marker="s", markersize=3)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Learning Curves")
    ax.legend()
    ax.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_detection_curves(
    scores: NDArray,
    labels: NDArray,
    figsize: tuple[int, int] = (14, 5),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot ROC and Precision-Recall curves.

    Args:
        scores: Anomaly scores.
        labels: Ground truth labels.
        figsize: Figure size.
        save_path: Optional save path.

    Returns:
        Matplotlib figure.
    """
    from sklearn.metrics import roc_curve, precision_recall_curve, auc

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # ROC curve
    fpr, tpr, _ = roc_curve(labels, scores)
    roc_auc = auc(fpr, tpr)

    axes[0].plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
    axes[0].plot([0, 1], [0, 1], "k--", label="Random")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend()
    axes[0].grid(True)

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(labels, scores)
    pr_auc = auc(recall, precision)

    axes[1].plot(recall, precision, label=f"PR (AUC = {pr_auc:.3f})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_snr_performance(
    snr_bins: Sequence[tuple[float, float]],
    aurocs: Sequence[float],
    f1_scores: Sequence[float],
    figsize: tuple[int, int] = (12, 5),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot performance metrics vs SNR.

    Args:
        snr_bins: SNR bin ranges.
        aurocs: AUROC per bin.
        f1_scores: F1 score per bin.
        figsize: Figure size.
        save_path: Optional save path.

    Returns:
        Matplotlib figure.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    bin_centers = [(b[0] + b[1]) / 2 for b in snr_bins]
    bin_labels = [f"{b[0]:.0f}-{b[1]:.0f}" for b in snr_bins]

    # AUROC
    axes[0].bar(range(len(aurocs)), aurocs, color="steelblue")
    axes[0].set_xticks(range(len(aurocs)))
    axes[0].set_xticklabels(bin_labels, rotation=45)
    axes[0].set_xlabel("SNR Range (dB)")
    axes[0].set_ylabel("AUROC")
    axes[0].set_title("AUROC vs SNR")
    axes[0].set_ylim(0, 1)

    # F1 Score
    axes[1].bar(range(len(f1_scores)), f1_scores, color="darkorange")
    axes[1].set_xticks(range(len(f1_scores)))
    axes[1].set_xticklabels(bin_labels, rotation=45)
    axes[1].set_xlabel("SNR Range (dB)")
    axes[1].set_ylabel("F1 Score")
    axes[1].set_title("F1 Score vs SNR")
    axes[1].set_ylim(0, 1)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_score_distribution(
    scores: NDArray,
    labels: NDArray,
    threshold: float | None = None,
    figsize: tuple[int, int] = (10, 6),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot distribution of anomaly scores.

    Args:
        scores: Anomaly scores.
        labels: Ground truth labels.
        threshold: Optional detection threshold.
        figsize: Figure size.
        save_path: Optional save path.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]

    ax.hist(normal_scores, bins=50, alpha=0.6, label="Normal", density=True)
    ax.hist(anomaly_scores, bins=50, alpha=0.6, label="Anomaly", density=True)

    if threshold is not None:
        ax.axvline(threshold, color="red", linestyle="--", label=f"Threshold ({threshold:.4f})")

    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Density")
    ax.set_title("Score Distribution")
    ax.legend()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_continuous_learning_metrics(
    metrics_history: list[dict],
    figsize: tuple[int, int] = (14, 8),
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot metrics over continuous learning.

    Args:
        metrics_history: List of metric dictionaries over time.
        figsize: Figure size.
        save_path: Optional save path.

    Returns:
        Matplotlib figure.
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)

    # Extract metrics
    steps = range(len(metrics_history))
    losses = [m.get("loss", 0) for m in metrics_history]
    aurocs = [m.get("auroc", 0) for m in metrics_history]
    f1s = [m.get("f1", 0) for m in metrics_history]

    # Loss
    axes[0, 0].plot(steps, losses)
    axes[0, 0].set_xlabel("Update Step")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].set_title("Training Loss")

    # AUROC
    axes[0, 1].plot(steps, aurocs)
    axes[0, 1].set_xlabel("Update Step")
    axes[0, 1].set_ylabel("AUROC")
    axes[0, 1].set_title("Detection AUROC")
    axes[0, 1].set_ylim(0, 1)

    # F1 Score
    axes[1, 0].plot(steps, f1s)
    axes[1, 0].set_xlabel("Update Step")
    axes[1, 0].set_ylabel("F1 Score")
    axes[1, 0].set_title("Detection F1")
    axes[1, 0].set_ylim(0, 1)

    # Learning rate if available
    lrs = [m.get("learning_rate", None) for m in metrics_history]
    if lrs[0] is not None:
        axes[1, 1].plot(steps, lrs)
        axes[1, 1].set_xlabel("Update Step")
        axes[1, 1].set_ylabel("Learning Rate")
        axes[1, 1].set_title("Learning Rate Schedule")
        axes[1, 1].set_yscale("log")
    else:
        axes[1, 1].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
