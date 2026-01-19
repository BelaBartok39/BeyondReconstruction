"""Phase-aware anomaly detection for frequency drift.

Frequency drift causes:
- 1245% increase in phase variance
- Changes in instantaneous frequency pattern
- Spectral centroid shifts

This module provides phase-based features that complement latent-space detection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass
class PhaseFeatures:
    """Phase-based features extracted from I/Q signal."""

    inst_freq_mean: float
    inst_freq_std: float
    phase_variance: float
    phase_unwrap_range: float
    freq_drift_rate: float
    spectral_centroid_std: float


class PhaseAnomalyDetector:
    """Detect anomalies using phase-based features.

    Particularly effective for frequency drift detection where
    latent-space methods struggle.
    """

    def __init__(self, percentile_threshold: float = 95.0):
        """Initialize phase detector.

        Args:
            percentile_threshold: Percentile for anomaly threshold.
        """
        self.percentile_threshold = percentile_threshold
        self.thresholds: dict[str, float] = {}
        self.means: dict[str, float] = {}
        self.stds: dict[str, float] = {}
        self._fitted = False

    def extract_phase_features(self, iq: NDArray[np.float32]) -> PhaseFeatures:
        """Extract phase-based features from I/Q signal.

        Args:
            iq: I/Q signal of shape [2, seq_len] or [batch, 2, seq_len]

        Returns:
            PhaseFeatures dataclass with extracted features.
        """
        if iq.ndim == 2:
            iq = iq[np.newaxis, ...]  # Add batch dim

        # Convert to complex signal
        complex_signal = iq[:, 0, :] + 1j * iq[:, 1, :]

        # Unwrap phase
        phase = np.unwrap(np.angle(complex_signal), axis=1)

        # Instantaneous frequency (phase derivative)
        inst_freq = np.diff(phase, axis=1)

        # Compute features (averaged over batch)
        inst_freq_mean = float(np.mean(inst_freq))
        inst_freq_std = float(np.mean(np.std(inst_freq, axis=1)))
        phase_variance = float(np.mean(np.var(phase, axis=1)))
        phase_unwrap_range = float(np.mean(np.ptp(phase, axis=1)))

        # Frequency drift rate (linear fit slope of inst_freq)
        t = np.arange(inst_freq.shape[1])
        drift_rates = []
        for i in range(inst_freq.shape[0]):
            slope, _ = np.polyfit(t, inst_freq[i], 1)
            drift_rates.append(abs(slope))
        freq_drift_rate = float(np.mean(drift_rates))

        # Spectral centroid variation
        fft = np.fft.fft(complex_signal, axis=1)
        magnitudes = np.abs(fft)
        freqs = np.arange(magnitudes.shape[1])
        centroids = np.sum(magnitudes * freqs, axis=1) / (np.sum(magnitudes, axis=1) + 1e-8)
        spectral_centroid_std = float(np.std(centroids))

        return PhaseFeatures(
            inst_freq_mean=inst_freq_mean,
            inst_freq_std=inst_freq_std,
            phase_variance=phase_variance,
            phase_unwrap_range=phase_unwrap_range,
            freq_drift_rate=freq_drift_rate,
            spectral_centroid_std=spectral_centroid_std,
        )

    def extract_batch_features(self, iq_batch: NDArray[np.float32]) -> NDArray[np.float32]:
        """Extract phase features for a batch of signals.

        Args:
            iq_batch: Batch of I/Q signals [batch, 2, seq_len]

        Returns:
            Feature matrix [batch, n_features]
        """
        if iq_batch.ndim == 2:
            iq_batch = iq_batch[np.newaxis, ...]

        batch_size = iq_batch.shape[0]
        features = np.zeros((batch_size, 6), dtype=np.float32)

        # Convert to complex signal
        complex_signal = iq_batch[:, 0, :] + 1j * iq_batch[:, 1, :]

        # Unwrap phase
        phase = np.unwrap(np.angle(complex_signal), axis=1)

        # Instantaneous frequency
        inst_freq = np.diff(phase, axis=1)

        # Feature 0: inst_freq_mean (per sample)
        features[:, 0] = np.mean(inst_freq, axis=1)

        # Feature 1: inst_freq_std
        features[:, 1] = np.std(inst_freq, axis=1)

        # Feature 2: phase_variance
        features[:, 2] = np.var(phase, axis=1)

        # Feature 3: phase_unwrap_range
        features[:, 3] = np.ptp(phase, axis=1)

        # Feature 4: freq_drift_rate (slope of inst_freq)
        t = np.arange(inst_freq.shape[1])
        for i in range(batch_size):
            slope, _ = np.polyfit(t, inst_freq[i], 1)
            features[i, 4] = abs(slope)

        # Feature 5: spectral properties
        fft = np.fft.fft(complex_signal, axis=1)
        magnitudes = np.abs(fft)
        freqs = np.arange(magnitudes.shape[1])
        features[:, 5] = np.std(magnitudes, axis=1)

        return features

    def fit(self, normal_iq: NDArray[np.float32]) -> "PhaseAnomalyDetector":
        """Fit detector on normal signals.

        Args:
            normal_iq: Normal I/Q signals [n_samples, 2, seq_len]

        Returns:
            Self for chaining.
        """
        features = self.extract_batch_features(normal_iq)

        feature_names = [
            "inst_freq_mean", "inst_freq_std", "phase_variance",
            "phase_unwrap_range", "freq_drift_rate", "spectral_std"
        ]

        for i, name in enumerate(feature_names):
            self.means[name] = float(np.mean(features[:, i]))
            self.stds[name] = float(np.std(features[:, i])) + 1e-8
            # Use absolute deviation from mean for threshold
            deviations = np.abs(features[:, i] - self.means[name])
            self.thresholds[name] = float(np.percentile(deviations, self.percentile_threshold))

        self._fitted = True
        return self

    def score(self, iq: NDArray[np.float32]) -> NDArray[np.float32]:
        """Compute anomaly scores based on phase features.

        Args:
            iq: I/Q signals [batch, 2, seq_len] or [2, seq_len]

        Returns:
            Anomaly scores [batch] - higher means more anomalous.
        """
        if not self._fitted:
            raise RuntimeError("Detector not fitted. Call fit() first.")

        if iq.ndim == 2:
            iq = iq[np.newaxis, ...]

        features = self.extract_batch_features(iq)

        feature_names = [
            "inst_freq_mean", "inst_freq_std", "phase_variance",
            "phase_unwrap_range", "freq_drift_rate", "spectral_std"
        ]

        # Compute normalized deviation for each feature
        scores = np.zeros(features.shape[0], dtype=np.float32)

        # Weight phase_variance and freq_drift_rate higher (most discriminative for freq drift)
        weights = [0.5, 1.0, 2.0, 1.0, 2.0, 0.5]

        for i, (name, weight) in enumerate(zip(feature_names, weights)):
            deviation = np.abs(features[:, i] - self.means[name]) / self.stds[name]
            scores += weight * deviation

        return scores / sum(weights)


class HybridPhaseLatentDetector:
    """Combines phase-based and latent-based detection.

    Uses phase features specifically to boost frequency drift detection
    while maintaining high performance on other anomaly types.
    """

    def __init__(
        self,
        latent_detector,
        phase_weight: float = 0.3,
        percentile_threshold: float = 95.0,
    ):
        """Initialize hybrid detector.

        Args:
            latent_detector: Fitted AnomalyDetector instance.
            phase_weight: Weight for phase scores (0-1). Latent weight = 1 - phase_weight.
            percentile_threshold: Threshold for phase detector.
        """
        self.latent_detector = latent_detector
        self.phase_detector = PhaseAnomalyDetector(percentile_threshold)
        self.phase_weight = phase_weight
        self._fitted = False

    def fit(self, normal_iq: NDArray[np.float32]) -> "HybridPhaseLatentDetector":
        """Fit phase detector on normal signals.

        Args:
            normal_iq: Normal I/Q signals [n_samples, 2, seq_len]

        Returns:
            Self for chaining.
        """
        self.phase_detector.fit(normal_iq)
        self._fitted = True
        return self

    def score(
        self,
        iq: NDArray[np.float32],
        latent_scores: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Compute hybrid anomaly scores.

        Args:
            iq: I/Q signals [batch, 2, seq_len]
            latent_scores: Pre-computed latent space scores [batch]

        Returns:
            Hybrid anomaly scores [batch]
        """
        if not self._fitted:
            raise RuntimeError("Detector not fitted. Call fit() first.")

        phase_scores = self.phase_detector.score(iq)

        # Normalize both score types to similar range
        latent_norm = (latent_scores - latent_scores.min()) / (latent_scores.max() - latent_scores.min() + 1e-8)
        phase_norm = (phase_scores - phase_scores.min()) / (phase_scores.max() - phase_scores.min() + 1e-8)

        # Weighted combination
        hybrid_scores = (1 - self.phase_weight) * latent_norm + self.phase_weight * phase_norm

        return hybrid_scores
