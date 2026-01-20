"""Signal processing utilities for TorchRF Testbed."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def complex_to_iq(signal: NDArray[np.complex64]) -> NDArray[np.float32]:
    """Convert complex signal to I/Q format [2, N].

    Args:
        signal: Complex signal array of shape [N].

    Returns:
        I/Q array of shape [2, N] with I in row 0 and Q in row 1.
    """
    return np.stack([signal.real, signal.imag], axis=0).astype(np.float32)


def iq_to_complex(iq: NDArray[np.float32]) -> NDArray[np.complex64]:
    """Convert I/Q format [2, N] to complex signal.

    Args:
        iq: I/Q array of shape [2, N].

    Returns:
        Complex signal array of shape [N].
    """
    return (iq[0] + 1j * iq[1]).astype(np.complex64)


def normalize_signal(signal: NDArray[np.complex64]) -> tuple[NDArray[np.complex64], float]:
    """Normalize signal amplitude to unit maximum.

    Args:
        signal: Complex signal array.

    Returns:
        Tuple of (normalized signal, original power in dB).
    """
    power = np.mean(np.abs(signal) ** 2)
    power_db = float(10 * np.log10(power + 1e-10))

    max_amp = np.max(np.abs(signal))
    if max_amp > 0:
        normalized = signal / max_amp
    else:
        normalized = signal

    return normalized.astype(np.complex64), power_db


def estimate_power(signal: NDArray[np.complex64]) -> float:
    """Estimate signal power in dB.

    Args:
        signal: Complex signal array.

    Returns:
        Power in dB.
    """
    power = np.mean(np.abs(signal) ** 2)
    return float(10 * np.log10(power + 1e-10))


def estimate_snr(
    signal: NDArray[np.complex64],
    method: str = "m2m4",
) -> float:
    """Estimate SNR from complex signal.

    Args:
        signal: Complex signal array.
        method: Estimation method ("m2m4", "wavelet", "spectral").

    Returns:
        Estimated SNR in dB.
    """
    estimators = {
        "m2m4": _estimate_snr_m2m4,
        "wavelet": _estimate_snr_wavelet,
        "spectral": _estimate_snr_spectral,
    }

    if method not in estimators:
        raise ValueError(f"Unknown SNR estimation method: {method}")

    return estimators[method](signal)


def _estimate_snr_m2m4(signal: NDArray[np.complex64]) -> float:
    """Estimate SNR using M2M4 moment-based method.

    Uses the second and fourth moments of the signal to estimate SNR
    without requiring knowledge of the modulation.

    Args:
        signal: Complex signal array.

    Returns:
        Estimated SNR in dB.
    """
    signal = signal.astype(np.complex128)
    m2 = np.mean(np.abs(signal) ** 2)
    m4 = np.mean(np.abs(signal) ** 4)
    kappa = m4 / (m2 ** 2 + 1e-10)

    if kappa >= 2:
        snr_linear = 0.01
    elif kappa <= 1:
        snr_linear = 100
    else:
        snr_linear = max(2 / (kappa - 1) - 1, 0.01)

    return float(np.clip(10 * np.log10(snr_linear), -10, 40))


def _estimate_snr_wavelet(signal: NDArray[np.complex64]) -> float:
    """Estimate SNR using wavelet-based noise estimation.

    Args:
        signal: Complex signal array.

    Returns:
        Estimated SNR in dB.
    """
    signal = signal.astype(np.complex128)
    detail = signal[1:] - signal[:-1]

    mad = np.median(np.abs(detail - np.median(detail)))
    noise_power = (mad / 0.6745) ** 2 / 2

    if noise_power < 1e-10:
        return 40.0

    signal_power = np.mean(np.abs(signal) ** 2)
    snr_linear = max((signal_power - noise_power) / noise_power, 0.01)

    return float(np.clip(10 * np.log10(snr_linear), -10, 40))


def _estimate_snr_spectral(signal: NDArray[np.complex64]) -> float:
    """Estimate SNR using spectral analysis.

    Args:
        signal: Complex signal array.

    Returns:
        Estimated SNR in dB.
    """
    signal = signal.astype(np.complex128)
    n = len(signal)
    power_spectrum = np.abs(np.fft.fft(signal)) ** 2 / n
    sorted_power = np.sort(power_spectrum)

    noise_floor = np.mean(sorted_power[: n // 4])

    if noise_floor < 1e-10:
        return 40.0

    signal_power = np.mean(sorted_power[3 * n // 4 :]) - noise_floor
    snr_linear = max(signal_power / noise_floor, 0.01)

    return float(np.clip(10 * np.log10(snr_linear), -10, 40))


def normalize_snr_value(snr_db: float, snr_range: tuple[float, float] = (-5, 30)) -> float:
    """Normalize SNR to [0, 1] range for model input.

    Args:
        snr_db: SNR value in dB.
        snr_range: Expected SNR range (min, max).

    Returns:
        Normalized SNR value in [0, 1].
    """
    return float(np.clip((snr_db - snr_range[0]) / (snr_range[1] - snr_range[0]), 0, 1))


def normalize_power_value(power_db: float, power_range: tuple[float, float] = (-20, 10)) -> float:
    """Normalize power to [0, 1] range for model input.

    Args:
        power_db: Power value in dB.
        power_range: Expected power range (min, max).

    Returns:
        Normalized power value in [0, 1].
    """
    return float(np.clip((power_db - power_range[0]) / (power_range[1] - power_range[0]), 0, 1))


def segment_signal(
    signal: NDArray[np.complex64],
    segment_length: int = 1024,
    overlap: int = 0,
) -> list[NDArray[np.complex64]]:
    """Segment a long signal into fixed-length segments.

    Args:
        signal: Complex signal array.
        segment_length: Length of each segment.
        overlap: Number of overlapping samples between segments.

    Returns:
        List of signal segments.
    """
    step = segment_length - overlap
    segments = []

    for start in range(0, len(signal) - segment_length + 1, step):
        segments.append(signal[start : start + segment_length])

    return segments


def compute_instantaneous_frequency(signal: NDArray[np.complex64]) -> NDArray[np.float32]:
    """Compute instantaneous frequency of a complex signal.

    Args:
        signal: Complex signal array.

    Returns:
        Instantaneous frequency array (normalized, length N-1).
    """
    phase = np.angle(signal)
    freq = np.diff(np.unwrap(phase)) / (2 * np.pi)
    return freq.astype(np.float32)


def compute_envelope(signal: NDArray[np.complex64]) -> NDArray[np.float32]:
    """Compute amplitude envelope of a complex signal.

    Args:
        signal: Complex signal array.

    Returns:
        Amplitude envelope array.
    """
    return np.abs(signal).astype(np.float32)
