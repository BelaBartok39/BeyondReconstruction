"""Software anomaly injection for live RF signals.

Mirrors the anomaly types from the CLP_Project synthetic data generator,
with additional MIT dataset compatible types.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np
from numpy.typing import NDArray


class AnomalyType(Enum):
    """Supported anomaly types for injection."""

    TONE = "tone"
    MULTI_TONE = "multi_tone"
    CHIRP = "chirp"
    SWEEP = "sweep"
    BARRAGE = "barrage"
    PULSE = "pulse"
    # CLP_Project compatible types
    INTERFERENCE = "interference"
    FREQUENCY_DRIFT = "frequency_drift"
    AMPLITUDE_SPIKE = "amplitude_spike"
    PHASE_NOISE = "phase_noise"
    BURST_NOISE = "burst_noise"


@dataclass
class InjectionMetadata:
    """Metadata about an injected anomaly."""

    anomaly_type: str
    severity: float
    params: dict


def inject_tone(
    signal: NDArray[np.complex64],
    frequency_offset: float = 0.1,
    sir_db: float = 0,
    sample_rate: float = 2e6,
) -> tuple[NDArray[np.complex64], dict]:
    """Inject narrowband tone interference.

    Args:
        signal: Complex input signal.
        frequency_offset: Normalized frequency offset (-0.5 to 0.5).
        sir_db: Signal-to-interference ratio in dB.
        sample_rate: Sample rate in Hz (for metadata only).

    Returns:
        Tuple of (corrupted signal, parameters dict).
    """
    n = len(signal)
    t = np.arange(n) / sample_rate

    # Convert normalized frequency to Hz
    freq_hz = frequency_offset * sample_rate

    # Calculate interference power
    signal_power = np.mean(np.abs(signal) ** 2)
    interference_power = signal_power / (10 ** (sir_db / 10))

    # Generate tone
    tone = np.sqrt(interference_power) * np.exp(2j * np.pi * freq_hz * t)

    params = {
        "frequency_offset": frequency_offset,
        "frequency_hz": freq_hz,
        "sir_db": sir_db,
    }

    return (signal + tone).astype(np.complex64), params


def inject_multi_tone(
    signal: NDArray[np.complex64],
    num_tones: int = 3,
    sir_db: float = 0,
    sample_rate: float = 2e6,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.complex64], dict]:
    """Inject multiple narrowband tone interferers.

    Args:
        signal: Complex input signal.
        num_tones: Number of tones to inject.
        sir_db: Overall signal-to-interference ratio in dB.
        sample_rate: Sample rate in Hz.
        rng: Random number generator.

    Returns:
        Tuple of (corrupted signal, parameters dict).
    """
    rng = rng or np.random.default_rng()
    n = len(signal)
    t = np.arange(n) / sample_rate

    # Calculate total interference power
    signal_power = np.mean(np.abs(signal) ** 2)
    total_interference_power = signal_power / (10 ** (sir_db / 10))
    per_tone_power = total_interference_power / num_tones

    # Generate random frequencies
    freqs = rng.uniform(-0.4, 0.4, num_tones) * sample_rate

    # Generate and sum tones
    interference = np.zeros(n, dtype=np.complex128)
    for freq in freqs:
        interference += np.sqrt(per_tone_power) * np.exp(2j * np.pi * freq * t)

    params = {
        "num_tones": num_tones,
        "frequencies_hz": freqs.tolist(),
        "sir_db": sir_db,
    }

    return (signal + interference).astype(np.complex64), params


def inject_chirp(
    signal: NDArray[np.complex64],
    start_freq: float = -0.3,
    end_freq: float = 0.3,
    sir_db: float = 0,
    sample_rate: float = 2e6,
) -> tuple[NDArray[np.complex64], dict]:
    """Inject chirp/sweep jamming signal.

    Args:
        signal: Complex input signal.
        start_freq: Starting normalized frequency.
        end_freq: Ending normalized frequency.
        sir_db: Signal-to-interference ratio in dB.
        sample_rate: Sample rate in Hz.

    Returns:
        Tuple of (corrupted signal, parameters dict).
    """
    n = len(signal)
    t = np.arange(n) / sample_rate

    # Convert to Hz
    start_hz = start_freq * sample_rate
    end_hz = end_freq * sample_rate

    # Calculate chirp rate and interference power
    duration = n / sample_rate
    chirp_rate = (end_hz - start_hz) / duration

    signal_power = np.mean(np.abs(signal) ** 2)
    interference_power = signal_power / (10 ** (sir_db / 10))

    # Generate chirp: phase = 2*pi*(f0*t + 0.5*chirp_rate*t^2)
    phase = 2 * np.pi * (start_hz * t + 0.5 * chirp_rate * t**2)
    chirp = np.sqrt(interference_power) * np.exp(1j * phase)

    params = {
        "start_freq": start_freq,
        "end_freq": end_freq,
        "start_hz": start_hz,
        "end_hz": end_hz,
        "chirp_rate": chirp_rate,
        "sir_db": sir_db,
    }

    return (signal + chirp).astype(np.complex64), params


def inject_sweep(
    signal: NDArray[np.complex64],
    start_freq: float = -0.3,
    end_freq: float = 0.3,
    sir_db: float = 0,
    sample_rate: float = 2e6,
) -> tuple[NDArray[np.complex64], dict]:
    """Inject frequency sweep (alias for chirp).

    Args:
        signal: Complex input signal.
        start_freq: Starting normalized frequency.
        end_freq: Ending normalized frequency.
        sir_db: Signal-to-interference ratio in dB.
        sample_rate: Sample rate in Hz.

    Returns:
        Tuple of (corrupted signal, parameters dict).
    """
    return inject_chirp(signal, start_freq, end_freq, sir_db, sample_rate)


def inject_barrage(
    signal: NDArray[np.complex64],
    bandwidth: float = 0.8,
    jsr_db: float = 0,
    sample_rate: float = 2e6,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.complex64], dict]:
    """Inject wideband barrage noise jamming.

    Args:
        signal: Complex input signal.
        bandwidth: Normalized bandwidth (0-1).
        jsr_db: Jammer-to-signal ratio in dB.
        sample_rate: Sample rate in Hz.
        rng: Random number generator.

    Returns:
        Tuple of (corrupted signal, parameters dict).
    """
    rng = rng or np.random.default_rng()
    n = len(signal)

    # Calculate interference power
    signal_power = np.mean(np.abs(signal) ** 2)
    jammer_power = signal_power * (10 ** (jsr_db / 10))

    # Generate wideband noise
    noise = np.sqrt(jammer_power / 2) * (
        rng.standard_normal(n) + 1j * rng.standard_normal(n)
    )

    # Apply bandwidth filter in frequency domain
    if bandwidth < 1.0:
        noise_fft = np.fft.fft(noise)
        freqs = np.fft.fftfreq(n)
        mask = np.abs(freqs) <= bandwidth / 2
        noise_fft[~mask] = 0
        noise = np.fft.ifft(noise_fft)
        # Rescale to maintain power
        noise *= np.sqrt(jammer_power / (np.mean(np.abs(noise) ** 2) + 1e-10))

    params = {
        "bandwidth": bandwidth,
        "bandwidth_hz": bandwidth * sample_rate,
        "jsr_db": jsr_db,
    }

    return (signal + noise).astype(np.complex64), params


def inject_pulse(
    signal: NDArray[np.complex64],
    duty_cycle: float = 0.2,
    jsr_db: float = 3,
    pulse_frequency: float = 10,
    sample_rate: float = 2e6,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.complex64], dict]:
    """Inject pulsed jamming interference.

    Args:
        signal: Complex input signal.
        duty_cycle: Fraction of time the jammer is on (0-1).
        jsr_db: Jammer-to-signal ratio during on periods.
        pulse_frequency: Pulse repetition frequency in Hz.
        sample_rate: Sample rate in Hz.
        rng: Random number generator.

    Returns:
        Tuple of (corrupted signal, parameters dict).
    """
    rng = rng or np.random.default_rng()
    n = len(signal)
    duration = n / sample_rate

    # Calculate interference power (during on time)
    signal_power = np.mean(np.abs(signal) ** 2)
    jammer_power = signal_power * (10 ** (jsr_db / 10))

    # Generate pulse envelope
    t = np.arange(n) / sample_rate
    period = 1 / pulse_frequency
    pulse_envelope = ((t % period) / period) < duty_cycle

    # Generate wideband noise
    noise = np.sqrt(jammer_power / 2) * (
        rng.standard_normal(n) + 1j * rng.standard_normal(n)
    )

    # Apply pulse envelope
    pulsed_noise = noise * pulse_envelope

    params = {
        "duty_cycle": duty_cycle,
        "jsr_db": jsr_db,
        "pulse_frequency": pulse_frequency,
        "num_pulses": int(duration * pulse_frequency),
    }

    return (signal + pulsed_noise).astype(np.complex64), params


def inject_interference(
    signal: NDArray[np.complex64],
    sir_db: float | None = None,
    severity: float = 1.0,
    sample_rate: float = 2e6,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.complex64], dict]:
    """Inject narrowband interference (CLP_Project compatible).

    Args:
        signal: Complex input signal.
        sir_db: Signal-to-interference ratio in dB.
        severity: Anomaly severity multiplier.
        sample_rate: Sample rate in Hz.
        rng: Random number generator.

    Returns:
        Tuple of (corrupted signal, parameters dict).
    """
    rng = rng or np.random.default_rng()

    if sir_db is None:
        sir_db = rng.uniform(-10 * severity, 5)

    freq_offset = rng.uniform(-0.4, 0.4)

    return inject_tone(signal, freq_offset, sir_db, sample_rate)


def inject_frequency_drift(
    signal: NDArray[np.complex64],
    drift_rate: float | None = None,
    severity: float = 1.0,
    sample_rate: float = 2e6,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.complex64], dict]:
    """Inject frequency drift (CLP_Project compatible).

    Args:
        signal: Complex input signal.
        drift_rate: Drift rate in Hz/sample.
        severity: Anomaly severity multiplier.
        sample_rate: Sample rate in Hz.
        rng: Random number generator.

    Returns:
        Tuple of (corrupted signal, parameters dict).
    """
    rng = rng or np.random.default_rng()

    if drift_rate is None:
        drift_rate = rng.uniform(-30 * severity, 30 * severity)

    n = len(signal)
    t = np.arange(n)
    phase = np.pi * drift_rate * t**2 / sample_rate

    corrupted = signal * np.exp(1j * phase)

    params = {
        "drift_rate_hz_per_sample": drift_rate,
        "severity": severity,
    }

    return corrupted.astype(np.complex64), params


def inject_amplitude_spike(
    signal: NDArray[np.complex64],
    spike_amplitude: float | None = None,
    spike_duration: int | None = None,
    severity: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.complex64], dict]:
    """Inject amplitude spike/burst (CLP_Project compatible).

    Args:
        signal: Complex input signal.
        spike_amplitude: Relative spike amplitude.
        spike_duration: Duration in samples.
        severity: Anomaly severity multiplier.
        rng: Random number generator.

    Returns:
        Tuple of (corrupted signal, parameters dict).
    """
    rng = rng or np.random.default_rng()
    n = len(signal)

    if spike_amplitude is None:
        spike_amplitude = rng.uniform(3 * severity, 10 * severity)
    if spike_duration is None:
        spike_duration = rng.integers(20, min(200, n // 4))

    spike_start = rng.integers(0, max(1, n - spike_duration))

    corrupted = signal.copy()
    corrupted[spike_start : spike_start + spike_duration] *= spike_amplitude

    params = {
        "spike_amplitude": spike_amplitude,
        "spike_duration": spike_duration,
        "spike_start": spike_start,
        "severity": severity,
    }

    return corrupted.astype(np.complex64), params


def inject_phase_noise(
    signal: NDArray[np.complex64],
    noise_std: float | None = None,
    severity: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.complex64], dict]:
    """Inject random phase noise (CLP_Project compatible).

    Args:
        signal: Complex input signal.
        noise_std: Std deviation of phase noise in radians.
        severity: Anomaly severity multiplier.
        rng: Random number generator.

    Returns:
        Tuple of (corrupted signal, parameters dict).
    """
    rng = rng or np.random.default_rng()
    n = len(signal)

    if noise_std is None:
        noise_std = rng.uniform(0.5 * severity, 2.0 * severity)

    # Generate correlated phase noise (random walk)
    phase_noise = np.cumsum(rng.normal(0, noise_std, n))

    corrupted = signal * np.exp(1j * phase_noise)

    params = {
        "phase_noise_std": noise_std,
        "severity": severity,
    }

    return corrupted.astype(np.complex64), params


def inject_burst_noise(
    signal: NDArray[np.complex64],
    burst_snr: float | None = None,
    num_bursts: int | None = None,
    severity: float = 1.0,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.complex64], dict]:
    """Inject impulsive burst noise (CLP_Project compatible).

    Args:
        signal: Complex input signal.
        burst_snr: SNR during bursts in dB.
        num_bursts: Number of burst events.
        severity: Anomaly severity multiplier.
        rng: Random number generator.

    Returns:
        Tuple of (corrupted signal, parameters dict).
    """
    rng = rng or np.random.default_rng()
    n = len(signal)

    if burst_snr is None:
        burst_snr = rng.uniform(-20 * severity, -5)
    if num_bursts is None:
        num_bursts = rng.integers(2, 8)

    signal_power = np.mean(np.abs(signal) ** 2)
    burst_power = signal_power / (10 ** (burst_snr / 10))

    corrupted = signal.copy()
    burst_params = []

    for _ in range(num_bursts):
        burst_duration = rng.integers(5, 50)
        burst_start = rng.integers(0, max(1, n - burst_duration))

        burst_noise = np.sqrt(burst_power / 2) * (
            rng.standard_normal(burst_duration)
            + 1j * rng.standard_normal(burst_duration)
        )

        corrupted[burst_start : burst_start + burst_duration] += burst_noise
        burst_params.append({"start": int(burst_start), "duration": int(burst_duration)})

    params = {
        "burst_snr_db": burst_snr,
        "num_bursts": num_bursts,
        "bursts": burst_params,
        "severity": severity,
    }

    return corrupted.astype(np.complex64), params


# Mapping of anomaly types to injection functions
_INJECTION_FUNCTIONS: dict[AnomalyType, Callable] = {
    AnomalyType.TONE: inject_tone,
    AnomalyType.MULTI_TONE: inject_multi_tone,
    AnomalyType.CHIRP: inject_chirp,
    AnomalyType.SWEEP: inject_sweep,
    AnomalyType.BARRAGE: inject_barrage,
    AnomalyType.PULSE: inject_pulse,
    AnomalyType.INTERFERENCE: inject_interference,
    AnomalyType.FREQUENCY_DRIFT: inject_frequency_drift,
    AnomalyType.AMPLITUDE_SPIKE: inject_amplitude_spike,
    AnomalyType.PHASE_NOISE: inject_phase_noise,
    AnomalyType.BURST_NOISE: inject_burst_noise,
}


def inject_anomaly(
    signal: NDArray[np.complex64],
    anomaly_type: str | AnomalyType | None = None,
    severity: float = 1.0,
    sample_rate: float = 2e6,
    rng: np.random.Generator | None = None,
    **kwargs,
) -> tuple[NDArray[np.complex64], InjectionMetadata]:
    """Main entry point for anomaly injection.

    Args:
        signal: Complex input signal.
        anomaly_type: Type of anomaly to inject. If None, randomly selected.
        severity: Anomaly severity multiplier (1.0=default, higher=stronger).
        sample_rate: Sample rate in Hz.
        rng: Random number generator.
        **kwargs: Additional parameters for specific anomaly type.

    Returns:
        Tuple of (corrupted signal, InjectionMetadata).
    """
    rng = rng or np.random.default_rng()

    # Parse anomaly type
    if anomaly_type is None:
        anomaly_type = rng.choice(list(AnomalyType))
    elif isinstance(anomaly_type, str):
        anomaly_type = AnomalyType(anomaly_type.lower())

    # Get injection function
    inject_fn = _INJECTION_FUNCTIONS[anomaly_type]

    # Build kwargs based on function signature
    func_params = inject_fn.__code__.co_varnames[:inject_fn.__code__.co_argcount]
    inject_kwargs = {}
    if "sample_rate" in func_params:
        inject_kwargs["sample_rate"] = sample_rate
    if "rng" in func_params:
        inject_kwargs["rng"] = rng
    if "severity" in func_params:
        inject_kwargs["severity"] = severity

    # Add user-provided kwargs (filtered to valid parameters)
    for key, value in kwargs.items():
        if key in func_params:
            inject_kwargs[key] = value

    # Inject anomaly
    corrupted, params = inject_fn(signal, **inject_kwargs)

    metadata = InjectionMetadata(
        anomaly_type=anomaly_type.value,
        severity=severity,
        params=params,
    )

    return corrupted, metadata


def get_anomaly_types() -> list[str]:
    """Get list of available anomaly types.

    Returns:
        List of anomaly type names.
    """
    return [t.value for t in AnomalyType]
