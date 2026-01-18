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
        invert_scores: bool = False,
        hybrid_weights: tuple[float, float] = (0.5, 0.5),
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
            invert_scores: If True, invert scores (use when anomalies have lower scores).
            hybrid_weights: Weights for (reconstruction, latent) in hybrid mode.
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
        self.invert_scores = invert_scores
        self.hybrid_weights = hybrid_weights
        self.device = (
            next(model.parameters()).device if device is None
            else torch.device(device) if isinstance(device, str)
            else device
        )

        # Fitted parameters
        self._threshold = None
        self._snr_thresholds = None
        self._latent_mean = None
        self._latent_cov_inv = None
        self._score_mean = None  # For normalization
        self._score_std = None
        self._is_fitted = False

        # Model type detection
        self._is_snr_conditioned = hasattr(model, "encoder") and hasattr(model.encoder, "cond_embed")
        self._is_vae = hasattr(model, "reparameterize")
        self._uses_power = hasattr(model, "use_power_conditioning") and model.use_power_conditioning

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
        all_scores, all_snrs, all_latents = [], [], []

        with torch.no_grad():
            for i, batch in enumerate(dataloader):
                if num_batches and i >= num_batches:
                    break

                iq = batch["iq"].to(self.device)
                snr = batch.get("snr")
                power = batch.get("power")
                snr = snr.to(self.device) if snr is not None else None
                power = power.to(self.device) if power is not None else None

                all_scores.append(self._compute_scores(iq, snr, power).cpu().numpy())

                if snr is not None:
                    all_snrs.append(batch["snr_db"].numpy())

                if self.method in ["latent", "hybrid"]:
                    all_latents.append(self._get_latent(iq, snr, power).cpu().numpy())

        all_scores = np.concatenate(all_scores)

        # Store score statistics for potential normalization
        self._score_mean = np.mean(all_scores)
        self._score_std = np.std(all_scores)

        # Compute global threshold
        self._threshold = {
            "fixed": self.fixed_threshold,
            "percentile": np.percentile(all_scores, self.threshold_percentile),
            "adaptive": self._score_mean + 3 * self._score_std
        }[self.threshold_method]

        # Compute SNR-adaptive thresholds
        if self.snr_adaptive and all_snrs:
            self._fit_snr_thresholds(all_scores, np.concatenate(all_snrs))

        # Fit latent space statistics
        if self.method in ["latent", "hybrid"] and all_latents:
            self._fit_latent_statistics(np.concatenate(all_latents))

        self._is_fitted = True
        return self

    def _fit_snr_thresholds(self, scores: NDArray, snrs: NDArray) -> None:
        """Fit per-SNR-bin thresholds.

        Args:
            scores: Anomaly scores.
            snrs: SNR values in dB.
        """
        self._snr_bin_edges = np.linspace(*self.snr_range, self.snr_bins + 1)
        self._snr_thresholds = np.zeros(self.snr_bins)

        for i in range(self.snr_bins):
            mask = (snrs >= self._snr_bin_edges[i]) & (snrs < self._snr_bin_edges[i + 1])
            if mask.sum() == 0:
                self._snr_thresholds[i] = self._threshold
                continue

            bin_scores = scores[mask]
            self._snr_thresholds[i] = (
                np.percentile(bin_scores, self.threshold_percentile)
                if self.threshold_method == "percentile"
                else np.mean(bin_scores) + 3 * np.std(bin_scores)
            )

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

    def _compute_scores(self, iq: Tensor, snr: Tensor | None = None, power: Tensor | None = None) -> Tensor:
        """Compute anomaly scores for a batch.

        Args:
            iq: IQ signals [batch, 2, seq_len].
            snr: Normalized SNR values [batch].
            power: Normalized power values [batch].

        Returns:
            Anomaly scores [batch].
        """
        if self.method == "reconstruction":
            scores = self._reconstruction_score(iq, snr, power)
        elif self.method == "latent":
            scores = self._latent_score(iq, snr, power)
        elif self.method == "hybrid":
            w_recon, w_latent = self.hybrid_weights
            recon_scores = self._reconstruction_score(iq, snr, power)
            latent_scores = self._latent_score(iq, snr, power)
            # Normalize both to [0, 1] range before combining
            recon_norm = recon_scores / (recon_scores.max() + 1e-8)
            latent_norm = latent_scores / (latent_scores.max() + 1e-8)
            scores = w_recon * recon_norm + w_latent * latent_norm
        else:
            raise ValueError(f"Unknown method: {self.method}")

        # Invert scores if anomalies have lower reconstruction error
        if self.invert_scores:
            scores = -scores

        return scores

    def _reconstruction_score(self, iq: Tensor, snr: Tensor | None = None, power: Tensor | None = None) -> Tensor:
        """Compute reconstruction-based anomaly score.

        Args:
            iq: IQ signals.
            snr: Normalized SNR values.
            power: Normalized power values.

        Returns:
            Reconstruction error per sample.
        """
        # Get reconstruction based on model type
        if self._is_snr_conditioned and snr is not None:
            if self._uses_power and power is not None:
                x_recon = self.model(iq, snr, power)[0]
            else:
                x_recon = self.model(iq, snr)[0]
        elif self._is_vae:
            x_recon = self.model(iq)[0]
        else:
            x_recon = self.model(iq)[0]

        return ((iq - x_recon) ** 2).mean(dim=(1, 2))

    def _latent_score(self, iq: Tensor, snr: Tensor | None = None, power: Tensor | None = None) -> Tensor:
        """Compute latent-space anomaly score (Mahalanobis distance).

        Args:
            iq: IQ signals.
            snr: Normalized SNR values.
            power: Normalized power values.

        Returns:
            Mahalanobis distance per sample.
        """
        if self._latent_mean is None:
            return torch.zeros(iq.size(0), device=iq.device)

        latent_np = self._get_latent(iq, snr, power).cpu().numpy()
        diff = latent_np - self._latent_mean
        mahal = np.sqrt(np.sum(diff @ self._latent_cov_inv * diff, axis=1))

        return torch.from_numpy(mahal).float().to(iq.device)

    def _get_latent(self, iq: Tensor, snr: Tensor | None = None, power: Tensor | None = None) -> Tensor:
        """Get latent representation from model.

        Args:
            iq: IQ signals.
            snr: Normalized SNR values.
            power: Normalized power values.

        Returns:
            Latent vectors [batch, latent_dim].
        """
        if self._is_snr_conditioned and snr is not None:
            if self._uses_power and power is not None:
                return self.model.encode(iq, snr, power)[0]
            return self.model.encode(iq, snr)[0]
        if self._is_vae:
            return self.model.encode(iq)[0]
        return self.model.encode(iq)

    def _get_threshold(self, snr_db: NDArray | None = None) -> NDArray:
        """Get threshold(s), optionally SNR-adaptive.

        Args:
            snr_db: SNR values in dB for adaptive thresholds.

        Returns:
            Threshold value(s).
        """
        size = len(snr_db) if snr_db is not None else 1

        if not self.snr_adaptive or snr_db is None or self._snr_thresholds is None:
            return np.full(size, self._threshold)

        bin_indices = np.clip(np.digitize(snr_db, self._snr_bin_edges) - 1, 0, self.snr_bins - 1)
        return self._snr_thresholds[bin_indices]

    def detect(
        self,
        iq: Tensor | NDArray,
        snr: Tensor | NDArray | None = None,
        snr_db: Tensor | NDArray | None = None,
        power: Tensor | NDArray | None = None,
    ) -> DetectionResult:
        """Detect anomalies in signals.

        Args:
            iq: IQ signals [batch, 2, seq_len].
            snr: Normalized SNR values [batch] (for model input).
            snr_db: SNR in dB [batch] (for adaptive thresholds).
            power: Normalized power values [batch] (for model input).

        Returns:
            DetectionResult with scores, predictions, and threshold.
        """
        if not self._is_fitted:
            raise RuntimeError("Detector must be fitted before detection. Call fit() first.")

        self.model.eval()

        # Convert to tensors
        iq = self._to_tensor(iq)
        snr = self._to_tensor(snr) if snr is not None else None
        power = self._to_tensor(power) if power is not None else None

        # Compute scores
        with torch.no_grad():
            scores_np = self._compute_scores(iq, snr, power).cpu().numpy()

        # Get thresholds and make predictions
        snr_db_np = snr_db.numpy() if isinstance(snr_db, torch.Tensor) else snr_db
        thresholds = self._get_threshold(snr_db_np) if snr_db is not None else np.full(len(scores_np), self._threshold)

        return DetectionResult(
            scores=scores_np,
            predictions=scores_np > thresholds,
            threshold=float(self._threshold),
        )

    def _to_tensor(self, x: Tensor | NDArray) -> Tensor:
        """Convert numpy array to tensor on device."""
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x).float()
        return x.to(self.device)

    def detect_batch(self, dataloader: torch.utils.data.DataLoader) -> tuple[NDArray, NDArray, NDArray]:
        """Detect anomalies in a full dataloader.

        Args:
            dataloader: DataLoader with test samples.

        Returns:
            Tuple of (scores, predictions, labels).
        """
        all_scores, all_predictions, all_labels = [], [], []

        for batch in dataloader:
            result = self.detect(batch["iq"], batch.get("snr"), batch.get("snr_db"), batch.get("power"))
            all_scores.append(result.scores)
            all_predictions.append(result.predictions)
            if (labels := batch.get("label")) is not None:
                all_labels.append(labels.numpy())

        return (
            np.concatenate(all_scores),
            np.concatenate(all_predictions),
            np.concatenate(all_labels) if all_labels else None
        )

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
            "invert_scores": self.invert_scores,
            "hybrid_weights": self.hybrid_weights,
            "is_fitted": self._is_fitted,
            "score_mean": self._score_mean,
            "score_std": self._score_std,
        }

        if self._snr_thresholds is not None:
            stats.update({
                "snr_thresholds": self._snr_thresholds.tolist(),
                "snr_bin_edges": self._snr_bin_edges.tolist()
            })

        return stats
