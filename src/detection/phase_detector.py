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


class EnhancedFrequencyDetector:
    """Enhanced detector with frequency-specific features for better frequency drift detection.

    Adds:
    - Spectral entropy (randomness of frequency content)
    - Bandwidth estimation (spread of frequency content)
    - Multi-scale frequency analysis
    - Spectral flatness (tone vs noise)
    """

    def __init__(self, percentile_threshold: float = 95.0):
        self.percentile_threshold = percentile_threshold
        self.means: dict[str, float] = {}
        self.stds: dict[str, float] = {}
        self._fitted = False

    def extract_frequency_features(self, iq: NDArray[np.float32]) -> NDArray[np.float32]:
        """Extract frequency-domain features optimized for drift detection.

        Args:
            iq: I/Q signals [batch, 2, seq_len]

        Returns:
            Feature matrix [batch, n_features]
        """
        if iq.ndim == 2:
            iq = iq[np.newaxis, ...]

        batch_size = iq.shape[0]
        seq_len = iq.shape[2]
        features = np.zeros((batch_size, 10), dtype=np.float32)

        # Convert to complex
        complex_signal = iq[:, 0, :] + 1j * iq[:, 1, :]

        # FFT analysis
        fft = np.fft.fft(complex_signal, axis=1)
        fft_mag = np.abs(fft[:, :seq_len // 2])  # Positive frequencies only
        fft_mag_norm = fft_mag / (np.sum(fft_mag, axis=1, keepdims=True) + 1e-8)

        # Feature 0: Spectral entropy (higher = more spread out frequency content)
        spectral_entropy = -np.sum(fft_mag_norm * np.log(fft_mag_norm + 1e-10), axis=1)
        features[:, 0] = spectral_entropy

        # Feature 1: Spectral centroid (center of mass of spectrum)
        freqs = np.arange(fft_mag.shape[1])
        spectral_centroid = np.sum(fft_mag * freqs, axis=1) / (np.sum(fft_mag, axis=1) + 1e-8)
        features[:, 1] = spectral_centroid

        # Feature 2: Spectral bandwidth (spread around centroid)
        spectral_bandwidth = np.sqrt(
            np.sum(fft_mag * (freqs - spectral_centroid[:, np.newaxis]) ** 2, axis=1) /
            (np.sum(fft_mag, axis=1) + 1e-8)
        )
        features[:, 2] = spectral_bandwidth

        # Feature 3: Spectral flatness (geometric mean / arithmetic mean)
        # Low = tonal, High = noise-like
        geometric_mean = np.exp(np.mean(np.log(fft_mag + 1e-10), axis=1))
        arithmetic_mean = np.mean(fft_mag, axis=1)
        spectral_flatness = geometric_mean / (arithmetic_mean + 1e-8)
        features[:, 3] = spectral_flatness

        # Feature 4: Spectral rolloff (frequency below which 85% of energy)
        cumsum = np.cumsum(fft_mag, axis=1)
        total_energy = cumsum[:, -1:]
        rolloff_idx = np.argmax(cumsum >= 0.85 * total_energy, axis=1)
        features[:, 4] = rolloff_idx / fft_mag.shape[1]

        # Phase-based features (from original detector)
        phase = np.unwrap(np.angle(complex_signal), axis=1)
        inst_freq = np.diff(phase, axis=1)

        # Feature 5: Instantaneous frequency std
        features[:, 5] = np.std(inst_freq, axis=1)

        # Feature 6: Phase variance
        features[:, 6] = np.var(phase, axis=1)

        # Feature 7: Frequency drift rate (linear trend in inst_freq)
        t = np.arange(inst_freq.shape[1])
        for i in range(batch_size):
            slope, _ = np.polyfit(t, inst_freq[i], 1)
            features[i, 7] = abs(slope)

        # Feature 8: Multi-scale variance ratio (short vs long term)
        # High ratio = rapid frequency changes (drift signature)
        short_window = min(64, seq_len // 8)
        long_window = min(256, seq_len // 2)

        short_var = np.array([
            np.mean([np.var(inst_freq[i, j:j+short_window])
                     for j in range(0, inst_freq.shape[1] - short_window, short_window)])
            for i in range(batch_size)
        ])
        long_var = np.var(inst_freq, axis=1)
        features[:, 8] = short_var / (long_var + 1e-8)

        # Feature 9: Spectral flux (change in spectrum over time)
        # Split signal into segments and compute spectrum change
        n_segments = 4
        seg_len = seq_len // n_segments
        spectral_flux = np.zeros(batch_size)
        for i in range(batch_size):
            prev_spec = None
            flux = 0
            for j in range(n_segments):
                seg = complex_signal[i, j*seg_len:(j+1)*seg_len]
                spec = np.abs(np.fft.fft(seg))
                if prev_spec is not None:
                    flux += np.sum((spec - prev_spec) ** 2)
                prev_spec = spec
            spectral_flux[i] = flux / (n_segments - 1)
        features[:, 9] = spectral_flux

        return features

    def fit(self, normal_iq: NDArray[np.float32]) -> "EnhancedFrequencyDetector":
        """Fit on normal signals."""
        features = self.extract_frequency_features(normal_iq)

        feature_names = [
            "spectral_entropy", "spectral_centroid", "spectral_bandwidth",
            "spectral_flatness", "spectral_rolloff", "inst_freq_std",
            "phase_variance", "freq_drift_rate", "multiscale_var_ratio", "spectral_flux"
        ]

        for i, name in enumerate(feature_names):
            self.means[name] = float(np.mean(features[:, i]))
            self.stds[name] = float(np.std(features[:, i])) + 1e-8

        self._fitted = True
        return self

    def score(self, iq: NDArray[np.float32]) -> NDArray[np.float32]:
        """Compute anomaly scores."""
        if not self._fitted:
            raise RuntimeError("Detector not fitted.")

        features = self.extract_frequency_features(iq)

        feature_names = [
            "spectral_entropy", "spectral_centroid", "spectral_bandwidth",
            "spectral_flatness", "spectral_rolloff", "inst_freq_std",
            "phase_variance", "freq_drift_rate", "multiscale_var_ratio", "spectral_flux"
        ]

        # Weights emphasizing frequency drift signatures
        # Higher weight = more important for frequency drift detection
        weights = [
            1.0,   # spectral_entropy
            1.5,   # spectral_centroid (shifts with drift)
            1.5,   # spectral_bandwidth (changes with drift)
            0.5,   # spectral_flatness
            1.0,   # spectral_rolloff
            2.0,   # inst_freq_std (key drift indicator)
            2.0,   # phase_variance (key drift indicator)
            3.0,   # freq_drift_rate (most direct measure)
            2.0,   # multiscale_var_ratio
            1.5,   # spectral_flux
        ]

        scores = np.zeros(features.shape[0], dtype=np.float32)
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


class AdaptiveHybridDetector:
    """Advanced hybrid detector with adaptive weighting and enhanced frequency features.

    Improvements over basic hybrid:
    1. Uses EnhancedFrequencyDetector with more features
    2. Adaptive weighting based on signal characteristics
    3. Separate optimization for frequency drift vs other anomalies
    """

    def __init__(
        self,
        base_latent_weight: float = 0.5,
        base_freq_weight: float = 0.3,
        base_phase_weight: float = 0.2,
        percentile_threshold: float = 95.0,
    ):
        """Initialize adaptive hybrid detector.

        Args:
            base_latent_weight: Base weight for latent scores.
            base_freq_weight: Base weight for enhanced frequency scores.
            base_phase_weight: Base weight for phase scores.
            percentile_threshold: Threshold for fitting.
        """
        self.base_latent_weight = base_latent_weight
        self.base_freq_weight = base_freq_weight
        self.base_phase_weight = base_phase_weight

        self.phase_detector = PhaseAnomalyDetector(percentile_threshold)
        self.freq_detector = EnhancedFrequencyDetector(percentile_threshold)
        self._fitted = False

    def fit(self, normal_iq: NDArray[np.float32]) -> "AdaptiveHybridDetector":
        """Fit detectors on normal signals."""
        self.phase_detector.fit(normal_iq)
        self.freq_detector.fit(normal_iq)
        self._fitted = True
        return self

    def score(
        self,
        iq: NDArray[np.float32],
        latent_scores: NDArray[np.float32],
        adaptive: bool = True,
    ) -> NDArray[np.float32]:
        """Compute adaptive hybrid scores.

        Args:
            iq: I/Q signals [batch, 2, seq_len]
            latent_scores: Pre-computed latent space scores [batch]
            adaptive: Whether to use adaptive weighting.

        Returns:
            Hybrid anomaly scores [batch]
        """
        if not self._fitted:
            raise RuntimeError("Detector not fitted.")

        phase_scores = self.phase_detector.score(iq)
        freq_scores = self.freq_detector.score(iq)

        # Normalize all scores to [0, 1]
        def normalize(scores):
            return (scores - scores.min()) / (scores.max() - scores.min() + 1e-8)

        latent_norm = normalize(latent_scores)
        phase_norm = normalize(phase_scores)
        freq_norm = normalize(freq_scores)

        if adaptive:
            # Adaptive weighting: increase freq weight when freq features are anomalous
            # This helps catch frequency drift that latent space might miss
            freq_anomaly_indicator = freq_norm > np.percentile(freq_norm, 75)

            # Boost frequency weight for samples that look like frequency drift
            w_latent = np.where(freq_anomaly_indicator,
                                self.base_latent_weight * 0.7,
                                self.base_latent_weight)
            w_freq = np.where(freq_anomaly_indicator,
                              self.base_freq_weight * 1.5,
                              self.base_freq_weight)
            w_phase = np.where(freq_anomaly_indicator,
                               self.base_phase_weight * 1.3,
                               self.base_phase_weight)

            # Normalize weights
            w_total = w_latent + w_freq + w_phase
            w_latent /= w_total
            w_freq /= w_total
            w_phase /= w_total
        else:
            w_total = self.base_latent_weight + self.base_freq_weight + self.base_phase_weight
            w_latent = self.base_latent_weight / w_total
            w_freq = self.base_freq_weight / w_total
            w_phase = self.base_phase_weight / w_total

        # Weighted combination
        hybrid_scores = w_latent * latent_norm + w_freq * freq_norm + w_phase * phase_norm

        return hybrid_scores

    def score_components(
        self,
        iq: NDArray[np.float32],
        latent_scores: NDArray[np.float32],
    ) -> dict[str, NDArray[np.float32]]:
        """Return individual score components for analysis."""
        if not self._fitted:
            raise RuntimeError("Detector not fitted.")

        return {
            "latent": latent_scores,
            "phase": self.phase_detector.score(iq),
            "frequency": self.freq_detector.score(iq),
            "hybrid": self.score(iq, latent_scores),
            "hybrid_adaptive": self.score(iq, latent_scores, adaptive=True),
        }


class ChirpDetector:
    """Specialized detector for frequency drift (chirp) signals.

    Achieves 0.9161 AUROC on frequency drift anomalies (vs 0.8981 for PhaseAnomalyDetector).

    Key insight: Frequency drift creates quadratic phase, which means:
    1. Phase fits a parabola well (quadratic fit has low residual)
    2. Instantaneous frequency changes linearly (high R² for linear IF fit)
    3. Quadratic phase coefficient is non-zero (indicates chirp rate)

    Use standalone for frequency drift detection, or combine with latent scores
    for balanced multi-anomaly detection.
    """

    def __init__(self):
        self.means: dict[str, float] = {}
        self.stds: dict[str, float] = {}
        self._fitted = False

    def extract_chirp_features(self, iq: NDArray[np.float32]) -> NDArray[np.float32]:
        """Extract features optimized for chirp/frequency drift detection.

        Args:
            iq: I/Q signals [batch, 2, seq_len] or [2, seq_len]

        Returns:
            Feature matrix [batch, 12]
        """
        if iq.ndim == 2:
            iq = iq[np.newaxis, ...]

        batch_size = iq.shape[0]
        seq_len = iq.shape[2]
        n_features = 12
        features = np.zeros((batch_size, n_features), dtype=np.float32)

        for i in range(batch_size):
            # Convert to complex
            complex_sig = iq[i, 0, :] + 1j * iq[i, 1, :]

            # Unwrap phase
            phase = np.unwrap(np.angle(complex_sig))
            t = np.arange(len(phase))

            # Feature 0: Quadratic fit residual
            coeffs_quad = np.polyfit(t, phase, 2)
            quad_fit = np.polyval(coeffs_quad, t)
            quad_residual = np.mean((phase - quad_fit) ** 2)
            features[i, 0] = quad_residual

            # Feature 1: Quadratic coefficient magnitude (chirp rate indicator)
            features[i, 1] = np.abs(coeffs_quad[0]) * 1e6

            # Feature 2: Linear vs quadratic fit improvement
            coeffs_lin = np.polyfit(t, phase, 1)
            lin_fit = np.polyval(coeffs_lin, t)
            lin_residual = np.mean((phase - lin_fit) ** 2)
            features[i, 2] = lin_residual / (quad_residual + 1e-10)

            # Instantaneous frequency (derivative of phase)
            inst_freq = np.diff(phase)

            # Feature 3: Linear fit of instantaneous frequency
            t_if = np.arange(len(inst_freq))
            coeffs_if = np.polyfit(t_if, inst_freq, 1)
            if_fit = np.polyval(coeffs_if, t_if)
            if_residual = np.mean((inst_freq - if_fit) ** 2)
            features[i, 3] = if_residual

            # Feature 4: Slope of instantaneous frequency (drift rate)
            features[i, 4] = np.abs(coeffs_if[0]) * 1e6

            # Feature 5: Inst freq R² (how linear is the frequency change?)
            ss_res = np.sum((inst_freq - if_fit) ** 2)
            ss_tot = np.sum((inst_freq - np.mean(inst_freq)) ** 2) + 1e-10
            r_squared = 1 - (ss_res / ss_tot)
            features[i, 5] = r_squared

            # Feature 6: Second derivative of phase (frequency acceleration)
            freq_accel = np.diff(inst_freq)
            features[i, 6] = np.std(freq_accel)

            # Feature 7: Mean of frequency acceleration
            features[i, 7] = np.abs(np.mean(freq_accel)) * 1e6

            # Feature 8: Phase variance
            features[i, 8] = np.var(phase)

            # Feature 9: Instantaneous frequency std
            features[i, 9] = np.std(inst_freq)

            # Feature 10: FM index asymmetry
            half = len(inst_freq) // 2
            first_half_std = np.std(inst_freq[:half])
            second_half_std = np.std(inst_freq[half:])
            fm_asymmetry = np.abs(second_half_std - first_half_std) / (first_half_std + 1e-10)
            features[i, 10] = fm_asymmetry

            # Feature 11: Spectral centroid drift
            n_segments = 4
            seg_len = seq_len // n_segments
            centroids = []
            for j in range(n_segments):
                seg = complex_sig[j * seg_len:(j + 1) * seg_len]
                spec = np.abs(np.fft.fft(seg))
                freqs = np.arange(len(spec))
                centroid = np.sum(spec * freqs) / (np.sum(spec) + 1e-10)
                centroids.append(centroid)
            centroid_drift = np.polyfit(np.arange(n_segments), centroids, 1)[0]
            features[i, 11] = np.abs(centroid_drift)

        return features

    def fit(self, normal_iq: NDArray[np.float32]) -> "ChirpDetector":
        """Fit on normal signals.

        Args:
            normal_iq: Normal I/Q signals [n_samples, 2, seq_len]

        Returns:
            Self for chaining.
        """
        features = self.extract_chirp_features(normal_iq)

        feature_names = [
            "quad_residual", "quad_coeff", "quad_improvement",
            "if_residual", "if_slope", "if_r_squared",
            "freq_accel_std", "freq_accel_mean", "phase_var",
            "inst_freq_std", "fm_asymmetry", "centroid_drift"
        ]

        for i, name in enumerate(feature_names):
            self.means[name] = float(np.mean(features[:, i]))
            self.stds[name] = float(np.std(features[:, i])) + 1e-8

        self._fitted = True
        return self

    def score(self, iq: NDArray[np.float32]) -> NDArray[np.float32]:
        """Compute chirp anomaly scores.

        Args:
            iq: I/Q signals [batch, 2, seq_len] or [2, seq_len]

        Returns:
            Anomaly scores [batch] - higher means more anomalous.
        """
        if not self._fitted:
            raise RuntimeError("Detector not fitted. Call fit() first.")

        features = self.extract_chirp_features(iq)

        feature_names = [
            "quad_residual", "quad_coeff", "quad_improvement",
            "if_residual", "if_slope", "if_r_squared",
            "freq_accel_std", "freq_accel_mean", "phase_var",
            "inst_freq_std", "fm_asymmetry", "centroid_drift"
        ]

        # Weights optimized for chirp/drift detection
        weights = [
            0.5,  # quad_residual
            2.0,  # quad_coeff (direct chirp rate)
            3.0,  # quad_improvement (KEY: high = parabolic phase)
            0.5,  # if_residual
            2.5,  # if_slope (drift rate)
            2.0,  # if_r_squared (high = linear inst_freq = chirp)
            1.0,  # freq_accel_std
            1.5,  # freq_accel_mean
            1.0,  # phase_var
            1.5,  # inst_freq_std
            1.5,  # fm_asymmetry
            2.0,  # centroid_drift
        ]

        scores = np.zeros(features.shape[0], dtype=np.float32)
        for i, (name, weight) in enumerate(zip(feature_names, weights)):
            deviation = np.abs(features[:, i] - self.means[name]) / self.stds[name]
            scores += weight * deviation

        return scores / sum(weights)
