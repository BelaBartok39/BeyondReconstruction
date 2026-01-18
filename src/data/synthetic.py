"""Synthetic RF signal generation for training and testing."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Callable

import numpy as np
from numpy.typing import NDArray


class Modulation(Enum):
    """Supported modulation types."""

    BPSK = "bpsk"
    QPSK = "qpsk"
    QAM16 = "qam16"
    QAM64 = "qam64"


class AnomalyType(Enum):
    """Types of signal anomalies."""

    INTERFERENCE = "interference"
    FREQUENCY_DRIFT = "frequency_drift"
    AMPLITUDE_SPIKE = "amplitude_spike"
    PHASE_NOISE = "phase_noise"
    BURST_NOISE = "burst_noise"


@dataclass
class SignalMetadata:
    """Metadata for a generated signal."""

    modulation: str
    snr_db: float
    is_anomaly: bool
    anomaly_type: str | None = None
    anomaly_params: dict | None = None


class SyntheticRFGenerator:
    """Generate synthetic RF IQ signals with controllable parameters.

    Generates normal signals with various modulations and anomalous signals
    with different types of interference and distortions.

    Example:
        generator = SyntheticRFGenerator(sequence_length=1024, sample_rate=1e6)
        iq, metadata = generator.generate_normal_signal(modulation="qpsk", snr_db=15)
        iq_anom, metadata_anom = generator.generate_anomaly(anomaly_type="interference")
    """

    def __init__(
        self,
        sequence_length: int = 1024,
        sample_rate: float = 1e6,
        symbol_rate: float = 1e5,
        carrier_freq: float = 0.0,
        seed: int | None = None,
    ):
        """Initialize signal generator.

        Args:
            sequence_length: Number of IQ samples per signal.
            sample_rate: Sample rate in Hz.
            symbol_rate: Symbol rate in symbols/second.
            carrier_freq: Carrier frequency offset in Hz.
            seed: Random seed for reproducibility.
        """
        self.sequence_length = sequence_length
        self.sample_rate = sample_rate
        self.symbol_rate = symbol_rate
        self.carrier_freq = carrier_freq
        self.rng = np.random.default_rng(seed)

        self.samples_per_symbol = int(sample_rate / symbol_rate)

        # Constellation mappings
        self._constellations = {
            Modulation.BPSK: self._bpsk_constellation(),
            Modulation.QPSK: self._qpsk_constellation(),
            Modulation.QAM16: self._qam16_constellation(),
            Modulation.QAM64: self._qam64_constellation(),
        }

        # Anomaly generators
        self._anomaly_generators: dict[AnomalyType, Callable] = {
            AnomalyType.INTERFERENCE: self._add_interference,
            AnomalyType.FREQUENCY_DRIFT: self._add_frequency_drift,
            AnomalyType.AMPLITUDE_SPIKE: self._add_amplitude_spike,
            AnomalyType.PHASE_NOISE: self._add_phase_noise,
            AnomalyType.BURST_NOISE: self._add_burst_noise,
        }

    @staticmethod
    def _bpsk_constellation() -> NDArray[np.complex128]:
        """BPSK constellation points."""
        return np.array([-1, 1], dtype=np.complex128)

    @staticmethod
    def _qpsk_constellation() -> NDArray[np.complex128]:
        """QPSK constellation points."""
        return np.exp(1j * np.pi * np.array([0.25, 0.75, 1.25, 1.75]))

    @staticmethod
    def _qam16_constellation() -> NDArray[np.complex128]:
        """16-QAM constellation points."""
        levels = np.array([-3, -1, 1, 3])
        constellation = []
        for i in levels:
            for q in levels:
                constellation.append(i + 1j * q)
        return np.array(constellation) / np.sqrt(10)  # Normalize power

    @staticmethod
    def _qam64_constellation() -> NDArray[np.complex128]:
        """64-QAM constellation points."""
        levels = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
        constellation = []
        for i in levels:
            for q in levels:
                constellation.append(i + 1j * q)
        return np.array(constellation) / np.sqrt(42)  # Normalize power

    def _generate_symbols(
        self, modulation: Modulation, num_symbols: int
    ) -> NDArray[np.complex128]:
        """Generate random modulated symbols.

        Args:
            modulation: Modulation type.
            num_symbols: Number of symbols to generate.

        Returns:
            Array of complex symbols.
        """
        constellation = self._constellations[modulation]
        indices = self.rng.integers(0, len(constellation), num_symbols)
        return constellation[indices]

    def _pulse_shape(
        self, symbols: NDArray[np.complex128], beta: float = 0.35
    ) -> NDArray[np.complex128]:
        """Apply raised cosine pulse shaping.

        Args:
            symbols: Complex symbols to shape.
            beta: Roll-off factor (0-1).

        Returns:
            Pulse-shaped signal.
        """
        # Upsample symbols
        upsampled = np.zeros(len(symbols) * self.samples_per_symbol, dtype=np.complex128)
        upsampled[:: self.samples_per_symbol] = symbols

        # Create raised cosine filter
        span = 6  # Filter spans 6 symbols
        t = np.arange(
            -span * self.samples_per_symbol // 2,
            span * self.samples_per_symbol // 2 + 1,
        )
        t = t / self.samples_per_symbol

        # Avoid division by zero
        eps = 1e-10
        t_safe = np.where(np.abs(t) < eps, eps, t)
        denom = 1 - (2 * beta * t_safe) ** 2
        denom = np.where(np.abs(denom) < eps, eps, denom)

        h = np.sinc(t) * np.cos(np.pi * beta * t) / denom
        h /= np.sqrt(np.sum(h**2))  # Normalize energy

        # Apply filter
        shaped = np.convolve(upsampled, h, mode="same")
        return shaped[: self.sequence_length]

    def _add_carrier(self, signal: NDArray[np.complex128]) -> NDArray[np.complex128]:
        """Add carrier frequency offset.

        Args:
            signal: Baseband signal.

        Returns:
            Signal with carrier frequency.
        """
        if self.carrier_freq == 0:
            return signal

        t = np.arange(len(signal)) / self.sample_rate
        carrier = np.exp(2j * np.pi * self.carrier_freq * t)
        return signal * carrier

    def _add_awgn(
        self, signal: NDArray[np.complex128], snr_db: float
    ) -> NDArray[np.complex128]:
        """Add white Gaussian noise to achieve target SNR.

        Args:
            signal: Input signal.
            snr_db: Target SNR in dB.

        Returns:
            Noisy signal.
        """
        signal_power = np.mean(np.abs(signal) ** 2)
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear

        noise = np.sqrt(noise_power / 2) * (
            self.rng.standard_normal(len(signal))
            + 1j * self.rng.standard_normal(len(signal))
        )

        return signal + noise

    def generate_normal_signal(
        self,
        modulation: str | Modulation = "qpsk",
        snr_db: float | None = None,
        snr_range: tuple[float, float] = (-5, 30),
    ) -> tuple[NDArray[np.float32], SignalMetadata]:
        """Generate a normal (non-anomalous) modulated signal.

        Args:
            modulation: Modulation type (bpsk, qpsk, qam16, qam64).
            snr_db: Target SNR in dB. If None, random from snr_range.
            snr_range: Range for random SNR selection.

        Returns:
            Tuple of (IQ array [2, seq_len], metadata).
        """
        if isinstance(modulation, str):
            modulation = Modulation(modulation.lower())

        if snr_db is None:
            snr_db = self.rng.uniform(snr_range[0], snr_range[1])

        # Generate enough symbols
        num_symbols = self.sequence_length // self.samples_per_symbol + 10
        symbols = self._generate_symbols(modulation, num_symbols)

        # Pulse shaping
        signal = self._pulse_shape(symbols)

        # Add carrier
        signal = self._add_carrier(signal)

        # Add noise
        signal = self._add_awgn(signal, snr_db)

        # Ensure correct length
        signal = signal[: self.sequence_length]
        if len(signal) < self.sequence_length:
            signal = np.pad(signal, (0, self.sequence_length - len(signal)))

        # Normalize
        signal = signal / (np.max(np.abs(signal)) + 1e-8)

        # Convert to [2, seq_len] format (I and Q channels)
        iq = np.stack([signal.real, signal.imag], axis=0).astype(np.float32)

        metadata = SignalMetadata(
            modulation=modulation.value,
            snr_db=float(snr_db),
            is_anomaly=False,
        )

        return iq, metadata

    def _add_interference(
        self,
        signal: NDArray[np.complex128],
        snr_db: float,
        sir_db: float | None = None,
    ) -> tuple[NDArray[np.complex128], dict]:
        """Add narrowband interference.

        Args:
            signal: Input signal.
            snr_db: Signal SNR (unused, kept for interface consistency).
            sir_db: Signal-to-interference ratio in dB.

        Returns:
            Tuple of (corrupted signal, anomaly parameters).
        """
        if sir_db is None:
            sir_db = self.rng.uniform(-5, 10)

        # Random interference frequency
        freq_offset = self.rng.uniform(-0.4, 0.4) * self.sample_rate

        # Generate interference
        t = np.arange(len(signal)) / self.sample_rate
        interference = np.exp(2j * np.pi * freq_offset * t)

        # Scale to desired SIR
        signal_power = np.mean(np.abs(signal) ** 2)
        sir_linear = 10 ** (sir_db / 10)
        interference_power = signal_power / sir_linear
        interference *= np.sqrt(interference_power)

        params = {"sir_db": sir_db, "frequency_offset_hz": freq_offset}
        return signal + interference, params

    def _add_frequency_drift(
        self,
        signal: NDArray[np.complex128],
        snr_db: float,
        drift_rate: float | None = None,
    ) -> tuple[NDArray[np.complex128], dict]:
        """Add frequency drift over time.

        Args:
            signal: Input signal.
            snr_db: Signal SNR (unused).
            drift_rate: Drift rate in Hz/sample.

        Returns:
            Tuple of (corrupted signal, anomaly parameters).
        """
        if drift_rate is None:
            drift_rate = self.rng.uniform(-10, 10)

        t = np.arange(len(signal))
        phase = 2 * np.pi * drift_rate * t**2 / (2 * self.sample_rate)
        drift = np.exp(1j * phase)

        params = {"drift_rate_hz_per_sample": drift_rate}
        return signal * drift, params

    def _add_amplitude_spike(
        self,
        signal: NDArray[np.complex128],
        snr_db: float,
        spike_amplitude: float | None = None,
        spike_duration: int | None = None,
    ) -> tuple[NDArray[np.complex128], dict]:
        """Add amplitude spike/burst.

        Args:
            signal: Input signal.
            snr_db: Signal SNR (unused).
            spike_amplitude: Relative spike amplitude.
            spike_duration: Duration of spike in samples.

        Returns:
            Tuple of (corrupted signal, anomaly parameters).
        """
        if spike_amplitude is None:
            spike_amplitude = self.rng.uniform(2, 5)
        if spike_duration is None:
            spike_duration = self.rng.integers(10, 100)

        # Random spike location
        spike_start = self.rng.integers(0, len(signal) - spike_duration)

        corrupted = signal.copy()
        corrupted[spike_start : spike_start + spike_duration] *= spike_amplitude

        params = {
            "spike_amplitude": spike_amplitude,
            "spike_duration": spike_duration,
            "spike_start": spike_start,
        }
        return corrupted, params

    def _add_phase_noise(
        self,
        signal: NDArray[np.complex128],
        snr_db: float,
        noise_std: float | None = None,
    ) -> tuple[NDArray[np.complex128], dict]:
        """Add random phase noise.

        Args:
            signal: Input signal.
            snr_db: Signal SNR (unused).
            noise_std: Standard deviation of phase noise in radians.

        Returns:
            Tuple of (corrupted signal, anomaly parameters).
        """
        if noise_std is None:
            noise_std = self.rng.uniform(0.3, 1.0)

        # Generate correlated phase noise (random walk)
        phase_increments = self.rng.normal(0, noise_std, len(signal))
        phase_noise = np.cumsum(phase_increments)

        corrupted = signal * np.exp(1j * phase_noise)

        params = {"phase_noise_std": noise_std}
        return corrupted, params

    def _add_burst_noise(
        self,
        signal: NDArray[np.complex128],
        snr_db: float,
        burst_snr: float | None = None,
        num_bursts: int | None = None,
    ) -> tuple[NDArray[np.complex128], dict]:
        """Add impulsive burst noise.

        Args:
            signal: Input signal.
            snr_db: Signal SNR (unused).
            burst_snr: SNR during bursts in dB.
            num_bursts: Number of burst events.

        Returns:
            Tuple of (corrupted signal, anomaly parameters).
        """
        if burst_snr is None:
            burst_snr = self.rng.uniform(-10, 0)
        if num_bursts is None:
            num_bursts = self.rng.integers(1, 5)

        corrupted = signal.copy()
        burst_params = []

        for _ in range(num_bursts):
            burst_duration = self.rng.integers(5, 50)
            burst_start = self.rng.integers(0, len(signal) - burst_duration)

            # Add burst noise
            signal_power = np.mean(np.abs(signal) ** 2)
            burst_snr_linear = 10 ** (burst_snr / 10)
            burst_power = signal_power / burst_snr_linear

            burst_noise = np.sqrt(burst_power / 2) * (
                self.rng.standard_normal(burst_duration)
                + 1j * self.rng.standard_normal(burst_duration)
            )

            corrupted[burst_start : burst_start + burst_duration] += burst_noise
            burst_params.append(
                {"start": burst_start, "duration": burst_duration}
            )

        params = {
            "burst_snr_db": burst_snr,
            "num_bursts": num_bursts,
            "bursts": burst_params,
        }
        return corrupted, params

    def generate_anomaly(
        self,
        anomaly_type: str | AnomalyType | None = None,
        base_modulation: str | Modulation = "qpsk",
        snr_db: float | None = None,
        snr_range: tuple[float, float] = (-5, 30),
        **anomaly_kwargs,
    ) -> tuple[NDArray[np.float32], SignalMetadata]:
        """Generate an anomalous signal.

        Args:
            anomaly_type: Type of anomaly. If None, randomly selected.
            base_modulation: Base modulation for the normal signal.
            snr_db: Target SNR in dB.
            snr_range: Range for random SNR selection.
            **anomaly_kwargs: Additional parameters for specific anomaly type.

        Returns:
            Tuple of (IQ array [2, seq_len], metadata).
        """
        if anomaly_type is None:
            anomaly_type = self.rng.choice(list(AnomalyType))
        elif isinstance(anomaly_type, str):
            anomaly_type = AnomalyType(anomaly_type.lower())

        if isinstance(base_modulation, str):
            base_modulation = Modulation(base_modulation.lower())

        if snr_db is None:
            snr_db = self.rng.uniform(snr_range[0], snr_range[1])

        # Generate base signal
        num_symbols = self.sequence_length // self.samples_per_symbol + 10
        symbols = self._generate_symbols(base_modulation, num_symbols)
        signal = self._pulse_shape(symbols)
        signal = self._add_carrier(signal)

        # Apply anomaly before noise (more realistic)
        anomaly_func = self._anomaly_generators[anomaly_type]
        signal, anomaly_params = anomaly_func(signal, snr_db, **anomaly_kwargs)

        # Add noise
        signal = self._add_awgn(signal, snr_db)

        # Ensure correct length and normalize
        signal = signal[: self.sequence_length]
        if len(signal) < self.sequence_length:
            signal = np.pad(signal, (0, self.sequence_length - len(signal)))
        signal = signal / (np.max(np.abs(signal)) + 1e-8)

        # Convert to [2, seq_len] format
        iq = np.stack([signal.real, signal.imag], axis=0).astype(np.float32)

        metadata = SignalMetadata(
            modulation=base_modulation.value,
            snr_db=float(snr_db),
            is_anomaly=True,
            anomaly_type=anomaly_type.value,
            anomaly_params=anomaly_params,
        )

        return iq, metadata

    def generate_batch(
        self,
        num_samples: int,
        anomaly_ratio: float = 0.0,
        modulations: list[str] | None = None,
        snr_range: tuple[float, float] = (-5, 30),
        anomaly_types: list[str] | None = None,
    ) -> tuple[NDArray[np.float32], list[SignalMetadata]]:
        """Generate a batch of signals.

        Args:
            num_samples: Total number of samples to generate.
            anomaly_ratio: Fraction of anomalous samples (0-1).
            modulations: List of modulations to use. If None, uses all.
            snr_range: Range for random SNR selection.
            anomaly_types: List of anomaly types. If None, uses all.

        Returns:
            Tuple of (IQ array [N, 2, seq_len], list of metadata).
        """
        if modulations is None:
            modulations = [m.value for m in Modulation]
        if anomaly_types is None:
            anomaly_types = [a.value for a in AnomalyType]

        num_anomalies = int(num_samples * anomaly_ratio)
        num_normal = num_samples - num_anomalies

        signals = []
        metadata_list = []

        # Generate normal signals
        for _ in range(num_normal):
            mod = self.rng.choice(modulations)
            iq, meta = self.generate_normal_signal(
                modulation=mod, snr_range=snr_range
            )
            signals.append(iq)
            metadata_list.append(meta)

        # Generate anomalous signals
        for _ in range(num_anomalies):
            mod = self.rng.choice(modulations)
            anom_type = self.rng.choice(anomaly_types)
            iq, meta = self.generate_anomaly(
                anomaly_type=anom_type,
                base_modulation=mod,
                snr_range=snr_range,
            )
            signals.append(iq)
            metadata_list.append(meta)

        # Shuffle
        indices = self.rng.permutation(num_samples)
        signals = [signals[i] for i in indices]
        metadata_list = [metadata_list[i] for i in indices]

        return np.stack(signals, axis=0), metadata_list
