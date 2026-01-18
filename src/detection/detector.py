"""Anomaly detection logic for RF signals."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn
from torch import Tensor
from scipy import stats


@dataclass
class DetectionResult:
    """Result of anomaly detection."""

    scores: NDArray[np.float32]  # Anomaly scores
    predictions: NDArray[np.bool_]  # Binary predictions
    threshold: float  # Detection threshold used


class AnomalyDetector:
    """Anomaly detector using autoencoder reconstruction error.

    Supports multiple detection methods:
    - reconstruction: Threshold on reconstruction error
    - latent: Mahalanobis distance in latent space
    - hybrid: Combination of both methods

    Example:
        detector = AnomalyDetector(model, method="hybrid")
        detector.fit(train_loader)  # Learn threshold from training data
        result = detector.detect(test_batch)
    """

    def __init__(
        self,
        model: nn.Module,
        method: Literal["reconstruction", "latent", "hybrid"] = "hybrid",
        threshold_method: Literal["fixed", "percentile", "adaptive"] = "percentile",
        threshold_percentile: float = 95.0,
        fixed_threshold: float | None = None,
        snr_adaptive: bool = True,
        snr_bins: int = 7,
        snr_range: tuple[float, float] = (-5, 30),
        device: torch.device | str | None = None,
    ):
        """Initialize anomaly detector.

        Args:
            model: Trained autoencoder model.
            method: Detection method.
            threshold_method: How to determine threshold.
            threshold_percentile: Percentile for threshold (if using percentile method).
            fixed_threshold: Fixed threshold value (if using fixed method).
            snr_adaptive: Whether to use SNR-adaptive thresholds.
            snr_bins: Number of SNR bins for adaptive thresholds.
            snr_range: SNR range in dB.
            device: Device to run inference on.
        """
        self.model = model
        self.method = method
        self.threshold_method = threshold_method
        self.threshold_percentile = threshold_percentile
        self.fixed_threshold = fixed_threshold
        self.snr_adaptive = snr_adaptive
        self.snr_bins = snr_bins
        self.snr_range = snr_range

        if device is None:
            device = next(model.parameters()).device
        self.device = torch.device(device) if isinstance(device, str) else device

        # Fitted parameters
        self._threshold = None
        self._snr_thresholds = None  # Thresholds per SNR bin
        self._latent_mean = None
        self._latent_cov_inv = None
        self._is_fitted = False

        # Model type detection
        self._is_snr_conditioned = hasattr(model, "encoder") and hasattr(
            model.encoder, "snr_embed"
        )
        self._is_vae = hasattr(model, "reparameterize")

    def fit(
        self,
        dataloader: torch.utils.data.DataLoader,
        num_batches: int | None = None,
    ) -> "AnomalyDetector":
        """Fit detector on training data (assumed normal).

        Learns thresholds and latent space statistics from normal data.

        Args:
            dataloader: DataLoader with normal training samples.
            num_batches: Max batches to use (None = all).

        Returns:
            Self for chaining.
        """
        self.model.eval()

        all_scores = []
        all_snrs = []
        all_latents = []

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if num_batches is not None and i >= num_batches:
                    break

                iq = batch["iq"].to(self.device)
                snr = batch.get("snr")
                if snr is not None:
                    snr = snr.to(self.device)

                # Get anomaly scores
                scores = self._compute_scores(iq, snr)
                all_scores.append(scores.cpu().numpy())

                if snr is not None:
                    all_snrs.append(batch["snr_db"].numpy())

                # Get latent representations for latent-space method
                if self.method in ["latent", "hybrid"]:
                    latent = self._get_latent(iq, snr)
                    all_latents.append(latent.cpu().numpy())

        all_scores = np.concatenate(all_scores)

        # Compute global threshold
        if self.threshold_method == "fixed":
            self._threshold = self.fixed_threshold
        elif self.threshold_method == "percentile":
            self._threshold = float(np.percentile(all_scores, self.threshold_percentile))
        elif self.threshold_method == "adaptive":
            # Use mean + k*std
            self._threshold = float(np.mean(all_scores) + 3 * np.std(all_scores))

        # Compute SNR-adaptive thresholds
        if self.snr_adaptive and len(all_snrs) > 0:
            all_snrs = np.concatenate(all_snrs)
            self._fit_snr_thresholds(all_scores, all_snrs)

        # Fit latent space statistics
        if self.method in ["latent", "hybrid"] and len(all_latents) > 0:
            all_latents = np.concatenate(all_latents)
            self._fit_latent_statistics(all_latents)

        self._is_fitted = True
        return self

    def _fit_snr_thresholds(
        self, scores: NDArray, snrs: NDArray
    ) -> None:
        """Fit per-SNR-bin thresholds.

        Args:
            scores: Anomaly scores.
            snrs: SNR values in dB.
        """
        snr_min, snr_max = self.snr_range
        bin_edges = np.linspace(snr_min, snr_max, self.snr_bins + 1)

        self._snr_thresholds = np.zeros(self.snr_bins)
        self._snr_bin_edges = bin_edges

        for i in range(self.snr_bins):
            mask = (snrs >= bin_edges[i]) & (snrs < bin_edges[i + 1])
            if np.sum(mask) > 0:
                bin_scores = scores[mask]
                if self.threshold_method == "percentile":
                    self._snr_thresholds[i] = np.percentile(
                        bin_scores, self.threshold_percentile
                    )
                else:
                    self._snr_thresholds[i] = np.mean(bin_scores) + 3 * np.std(bin_scores)
            else:
                self._snr_thresholds[i] = self._threshold

    def _fit_latent_statistics(self, latents: NDArray) -> None:
        """Fit latent space mean and covariance.

        Args:
            latents: Latent representations from training data.
        """
        self._latent_mean = np.mean(latents, axis=0)

        # Regularized covariance for numerical stability
        cov = np.cov(latents, rowvar=False)
        cov += np.eye(cov.shape[0]) * 1e-6
        self._latent_cov_inv = np.linalg.inv(cov)

    def _compute_scores(
        self, iq: Tensor, snr: Tensor | None = None
    ) -> Tensor:
        """Compute anomaly scores for a batch.

        Args:
            iq: IQ signals [batch, 2, seq_len].
            snr: Normalized SNR values [batch].

        Returns:
            Anomaly scores [batch].
        """
        if self.method == "reconstruction":
            return self._reconstruction_score(iq, snr)
        elif self.method == "latent":
            return self._latent_score(iq, snr)
        elif self.method == "hybrid":
            recon = self._reconstruction_score(iq, snr)
            latent = self._latent_score(iq, snr)
            # Normalize and combine
            return 0.5 * recon + 0.5 * latent
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _reconstruction_score(
        self, iq: Tensor, snr: Tensor | None = None
    ) -> Tensor:
        """Compute reconstruction-based anomaly score.

        Args:
            iq: IQ signals.
            snr: Normalized SNR values.

        Returns:
            Reconstruction error per sample.
        """
        if self._is_snr_conditioned and snr is not None:
            x_recon, _, _, _ = self.model(iq, snr)
        elif self._is_vae:
            x_recon, _, _, _ = self.model(iq)
        else:
            x_recon, _ = self.model(iq)

        # MSE per sample
        error = ((iq - x_recon) ** 2).mean(dim=(1, 2))
        return error

    def _latent_score(
        self, iq: Tensor, snr: Tensor | None = None
    ) -> Tensor:
        """Compute latent-space anomaly score (Mahalanobis distance).

        Args:
            iq: IQ signals.
            snr: Normalized SNR values.

        Returns:
            Mahalanobis distance per sample.
        """
        latent = self._get_latent(iq, snr)

        if self._latent_mean is None:
            # Not fitted yet, return zeros
            return torch.zeros(iq.size(0), device=iq.device)

        # Compute Mahalanobis distance
        latent_np = latent.cpu().numpy()
        diff = latent_np - self._latent_mean

        # Mahalanobis: sqrt((x-mu)^T * Cov^-1 * (x-mu))
        mahal = np.sqrt(np.sum(diff @ self._latent_cov_inv * diff, axis=1))

        return torch.from_numpy(mahal).float().to(iq.device)

    def _get_latent(self, iq: Tensor, snr: Tensor | None = None) -> Tensor:
        """Get latent representation from model.

        Args:
            iq: IQ signals.
            snr: Normalized SNR values.

        Returns:
            Latent vectors [batch, latent_dim].
        """
        if self._is_snr_conditioned and snr is not None:
            mu, _ = self.model.encode(iq, snr)
            return mu
        elif self._is_vae:
            mu, _ = self.model.encode(iq)
            return mu
        else:
            return self.model.encode(iq)

    def _get_threshold(self, snr_db: NDArray | None = None) -> NDArray:
        """Get threshold(s), optionally SNR-adaptive.

        Args:
            snr_db: SNR values in dB for adaptive thresholds.

        Returns:
            Threshold value(s).
        """
        if not self.snr_adaptive or snr_db is None or self._snr_thresholds is None:
            return np.full(len(snr_db) if snr_db is not None else 1, self._threshold)

        # Get bin index for each sample
        bin_indices = np.digitize(snr_db, self._snr_bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, self.snr_bins - 1)

        return self._snr_thresholds[bin_indices]

    def detect(
        self,
        iq: Tensor | NDArray,
        snr: Tensor | NDArray | None = None,
        snr_db: Tensor | NDArray | None = None,
    ) -> DetectionResult:
        """Detect anomalies in signals.

        Args:
            iq: IQ signals [batch, 2, seq_len].
            snr: Normalized SNR values [batch] (for model input).
            snr_db: SNR in dB [batch] (for adaptive thresholds).

        Returns:
            DetectionResult with scores, predictions, and threshold.
        """
        if not self._is_fitted:
            raise RuntimeError("Detector must be fitted before detection. Call fit() first.")

        self.model.eval()

        # Convert to tensor
        if isinstance(iq, np.ndarray):
            iq = torch.from_numpy(iq).float()
        iq = iq.to(self.device)

        if snr is not None:
            if isinstance(snr, np.ndarray):
                snr = torch.from_numpy(snr).float()
            snr = snr.to(self.device)

        # Compute scores
        with torch.no_grad():
            scores = self._compute_scores(iq, snr)

        scores_np = scores.cpu().numpy()

        # Get thresholds
        if snr_db is not None:
            if isinstance(snr_db, torch.Tensor):
                snr_db = snr_db.numpy()
            thresholds = self._get_threshold(snr_db)
        else:
            thresholds = np.full(len(scores_np), self._threshold)

        # Make predictions
        predictions = scores_np > thresholds

        return DetectionResult(
            scores=scores_np,
            predictions=predictions,
            threshold=float(self._threshold),
        )

    def detect_batch(
        self,
        dataloader: torch.utils.data.DataLoader,
    ) -> tuple[NDArray, NDArray, NDArray]:
        """Detect anomalies in a full dataloader.

        Args:
            dataloader: DataLoader with test samples.

        Returns:
            Tuple of (scores, predictions, labels).
        """
        all_scores = []
        all_predictions = []
        all_labels = []

        for batch in dataloader:
            iq = batch["iq"]
            snr = batch.get("snr")
            snr_db = batch.get("snr_db")
            labels = batch.get("label")

            result = self.detect(iq, snr, snr_db)

            all_scores.append(result.scores)
            all_predictions.append(result.predictions)
            if labels is not None:
                all_labels.append(labels.numpy())

        scores = np.concatenate(all_scores)
        predictions = np.concatenate(all_predictions)
        labels = np.concatenate(all_labels) if all_labels else None

        return scores, predictions, labels

    def update_threshold(self, new_threshold: float) -> None:
        """Update detection threshold.

        Args:
            new_threshold: New threshold value.
        """
        self._threshold = new_threshold

    def get_stats(self) -> dict:
        """Get detector statistics.

        Returns:
            Dictionary with detector statistics.
        """
        stats = {
            "method": self.method,
            "threshold_method": self.threshold_method,
            "threshold": self._threshold,
            "snr_adaptive": self.snr_adaptive,
            "is_fitted": self._is_fitted,
        }

        if self._snr_thresholds is not None:
            stats["snr_thresholds"] = self._snr_thresholds.tolist()
            stats["snr_bin_edges"] = self._snr_bin_edges.tolist()

        return stats
