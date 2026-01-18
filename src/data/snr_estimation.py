"""SNR estimation utilities for RF IQ signals."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import torch


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
    # Convert to numpy if tensor
    if isinstance(iq, torch.Tensor):
        iq = iq.detach().cpu().numpy()

    # Convert to complex if [2, seq_len] format
    if iq.ndim == 2 and iq.shape[0] == 2:
        signal = iq[0] + 1j * iq[1]
    else:
        signal = iq

    if method == "m2m4":
        return _estimate_snr_m2m4(signal)
    elif method == "wavelet":
        return _estimate_snr_wavelet(signal)
    elif method == "spectral":
        return _estimate_snr_spectral(signal)
    else:
        raise ValueError(f"Unknown SNR estimation method: {method}")


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

    snr_values = np.array([
        estimate_snr(iq_batch[i], method=method)
        for i in range(len(iq_batch))
    ], dtype=np.float32)

    return snr_values


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
    # Compute moments
    m2 = np.mean(np.abs(signal) ** 2)
    m4 = np.mean(np.abs(signal) ** 4)

    # For constant modulus signals (BPSK, QPSK)
    # Kurtosis-based estimate
    kappa = m4 / (m2 ** 2)

    # For QPSK, signal kurtosis is 1, noise kurtosis is 2
    # kappa = (S^2 + 2*S*N + 2*N^2) / (S + N)^2
    # Solving for SNR = S/N

    # Simplified estimation assuming signal kurtosis ≈ 1
    if kappa >= 2:
        # Very low SNR regime
        snr_linear = 0.01
    elif kappa <= 1:
        # Very high SNR regime
        snr_linear = 100
    else:
        # Estimate based on kurtosis
        # Derived from: kappa = (1 + 2/SNR + 2/SNR^2) / (1 + 1/SNR)^2
        # Approximate solution
        snr_linear = 2 / (kappa - 1) - 1
        snr_linear = max(snr_linear, 0.01)

    snr_db = 10 * np.log10(snr_linear)
    return float(np.clip(snr_db, -10, 40))


def _estimate_snr_wavelet(signal: NDArray[np.complex128]) -> float:
    """Estimate SNR using wavelet-based noise estimation.

    Uses the median absolute deviation of wavelet coefficients
    to estimate noise floor.

    Args:
        signal: Complex signal array.

    Returns:
        Estimated SNR in dB.
    """
    # Simple Haar wavelet decomposition (difference of adjacent samples)
    detail = signal[1:] - signal[:-1]

    # Estimate noise std using MAD (robust to outliers)
    mad = np.median(np.abs(detail - np.median(detail)))
    noise_std = mad / 0.6745  # Scale factor for Gaussian

    # Signal power
    signal_power = np.mean(np.abs(signal) ** 2)

    # Noise power (factor of 2 due to differencing)
    noise_power = (noise_std ** 2) / 2

    if noise_power < 1e-10:
        return 40.0  # Very high SNR

    snr_linear = (signal_power - noise_power) / noise_power
    snr_linear = max(snr_linear, 0.01)

    snr_db = 10 * np.log10(snr_linear)
    return float(np.clip(snr_db, -10, 40))


def _estimate_snr_spectral(signal: NDArray[np.complex128]) -> float:
    """Estimate SNR using spectral analysis.

    Estimates noise from spectral bins outside the main signal bandwidth.

    Args:
        signal: Complex signal array.

    Returns:
        Estimated SNR in dB.
    """
    # Compute FFT
    n = len(signal)
    spectrum = np.fft.fft(signal)
    power_spectrum = np.abs(spectrum) ** 2 / n

    # Sort power values
    sorted_power = np.sort(power_spectrum)

    # Estimate noise from lower 25% of spectrum (assumed noise floor)
    noise_floor = np.mean(sorted_power[: n // 4])

    # Estimate signal power from upper 25% minus noise
    signal_power = np.mean(sorted_power[3 * n // 4 :]) - noise_floor

    if noise_floor < 1e-10:
        return 40.0

    snr_linear = max(signal_power / noise_floor, 0.01)
    snr_db = 10 * np.log10(snr_linear)

    return float(np.clip(snr_db, -10, 40))


def normalize_snr(snr_db: float | NDArray, snr_range: tuple[float, float] = (-5, 30)) -> float | NDArray:
    """Normalize SNR to [0, 1] range for model input.

    Args:
        snr_db: SNR value(s) in dB.
        snr_range: Expected SNR range (min, max).

    Returns:
        Normalized SNR value(s) in [0, 1].
    """
    min_snr, max_snr = snr_range
    normalized = (snr_db - min_snr) / (max_snr - min_snr)
    return np.clip(normalized, 0, 1)


def denormalize_snr(snr_norm: float | NDArray, snr_range: tuple[float, float] = (-5, 30)) -> float | NDArray:
    """Convert normalized SNR back to dB.

    Args:
        snr_norm: Normalized SNR value(s) in [0, 1].
        snr_range: SNR range (min, max).

    Returns:
        SNR value(s) in dB.
    """
    min_snr, max_snr = snr_range
    return snr_norm * (max_snr - min_snr) + min_snr
