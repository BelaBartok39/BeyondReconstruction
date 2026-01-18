"""SNR estimation utilities for RF IQ signals."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import torch


def _to_complex(iq: NDArray[np.float32] | torch.Tensor) -> NDArray[np.complex128]:
    """Convert IQ signal to complex numpy array.

    Args:
        iq: IQ signal as tensor or numpy array.

    Returns:
        Complex numpy array.
    """
    if isinstance(iq, torch.Tensor):
        iq = iq.detach().cpu().numpy()

    if iq.ndim == 2 and iq.shape[0] == 2:
        return iq[0] + 1j * iq[1]

    return iq.astype(np.complex128)


def estimate_snr(
    iq: NDArray[np.float32] | torch.Tensor,
    method: str = "m2m4",
) -> float:
    """Estimate SNR from IQ samples.

    Args:
        iq: IQ signal array of shape [2, seq_len] or complex array [seq_len].
        method: Estimation method. Options: "m2m4", "wavelet", "spectral".

    Returns:
        Estimated SNR in dB.
    """
    signal = _to_complex(iq)

    estimators = {
        "m2m4": _estimate_snr_m2m4,
        "wavelet": _estimate_snr_wavelet,
        "spectral": _estimate_snr_spectral,
    }

    if method not in estimators:
        raise ValueError(f"Unknown SNR estimation method: {method}")

    return estimators[method](signal)


def estimate_snr_batch(
    iq_batch: NDArray[np.float32] | torch.Tensor,
    method: str = "m2m4",
) -> NDArray[np.float32]:
    """Estimate SNR for a batch of IQ signals.

    Args:
        iq_batch: Batch of IQ signals [batch, 2, seq_len].
        method: Estimation method.

    Returns:
        Array of estimated SNR values in dB [batch].
    """
    if isinstance(iq_batch, torch.Tensor):
        iq_batch = iq_batch.detach().cpu().numpy()

    return np.array([estimate_snr(iq_batch[i], method) for i in range(len(iq_batch))], dtype=np.float32)


def _estimate_snr_m2m4(signal: NDArray[np.complex128]) -> float:
    """Estimate SNR using M2M4 moment-based method.

    This method uses the second and fourth moments of the signal
    to estimate SNR without requiring knowledge of the modulation.

    Reference:
        Pauluzzi & Beaulieu, "A comparison of SNR estimation techniques
        for the AWGN channel," IEEE Trans. Comm., 2000.

    Args:
        signal: Complex signal array.

    Returns:
        Estimated SNR in dB.
    """
    m2 = np.mean(np.abs(signal) ** 2)
    m4 = np.mean(np.abs(signal) ** 4)
    kappa = m4 / (m2 ** 2)

    # Kurtosis-based SNR estimation
    # For QPSK: signal kurtosis ≈ 1, noise kurtosis = 2
    if kappa >= 2:
        snr_linear = 0.01  # Very low SNR
    elif kappa <= 1:
        snr_linear = 100  # Very high SNR
    else:
        snr_linear = max(2 / (kappa - 1) - 1, 0.01)

    return float(np.clip(10 * np.log10(snr_linear), -10, 40))


def _estimate_snr_wavelet(signal: NDArray[np.complex128]) -> float:
    """Estimate SNR using wavelet-based noise estimation.

    Uses the median absolute deviation of wavelet coefficients
    to estimate noise floor.

    Args:
        signal: Complex signal array.

    Returns:
        Estimated SNR in dB.
    """
    # Haar wavelet decomposition (difference of adjacent samples)
    detail = signal[1:] - signal[:-1]

    # Estimate noise std using MAD (robust to outliers)
    mad = np.median(np.abs(detail - np.median(detail)))
    noise_power = (mad / 0.6745) ** 2 / 2  # Scale for Gaussian, adjust for differencing

    if noise_power < 1e-10:
        return 40.0

    signal_power = np.mean(np.abs(signal) ** 2)
    snr_linear = max((signal_power - noise_power) / noise_power, 0.01)

    return float(np.clip(10 * np.log10(snr_linear), -10, 40))


def _estimate_snr_spectral(signal: NDArray[np.complex128]) -> float:
    """Estimate SNR using spectral analysis.

    Estimates noise from spectral bins outside the main signal bandwidth.

    Args:
        signal: Complex signal array.

    Returns:
        Estimated SNR in dB.
    """
    n = len(signal)
    power_spectrum = np.abs(np.fft.fft(signal)) ** 2 / n
    sorted_power = np.sort(power_spectrum)

    # Estimate noise from lower 25%, signal from upper 25%
    noise_floor = np.mean(sorted_power[: n // 4])

    if noise_floor < 1e-10:
        return 40.0

    signal_power = np.mean(sorted_power[3 * n // 4 :]) - noise_floor
    snr_linear = max(signal_power / noise_floor, 0.01)

    return float(np.clip(10 * np.log10(snr_linear), -10, 40))


def normalize_snr(snr_db: float | NDArray, snr_range: tuple[float, float] = (-5, 30)) -> float | NDArray:
    """Normalize SNR to [0, 1] range for model input.

    Args:
        snr_db: SNR value(s) in dB.
        snr_range: Expected SNR range (min, max).

    Returns:
        Normalized SNR value(s) in [0, 1].
    """
    return np.clip((snr_db - snr_range[0]) / (snr_range[1] - snr_range[0]), 0, 1)


def denormalize_snr(snr_norm: float | NDArray, snr_range: tuple[float, float] = (-5, 30)) -> float | NDArray:
    """Convert normalized SNR back to dB.

    Args:
        snr_norm: Normalized SNR value(s) in [0, 1].
        snr_range: SNR range (min, max).

    Returns:
        SNR value(s) in dB.
    """
    return snr_norm * (snr_range[1] - snr_range[0]) + snr_range[0]
