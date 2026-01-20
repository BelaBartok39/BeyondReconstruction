"""HackRF signal capture via GNURadio/osmosdr.

This module provides a high-level interface for capturing I/Q samples
from HackRF devices using the osmosdr source block.

Note: Requires gnuradio and gr-osmosdr to be installed via system
package manager (not pip).
"""

from __future__ import annotations

import sys
import time
import threading
from dataclasses import dataclass
from queue import Queue, Empty
from typing import Callable

import numpy as np
from numpy.typing import NDArray

# Try to find gnuradio in system site-packages if not directly available
_SYSTEM_SITE_PACKAGES = [
    "/usr/lib/python3.13/site-packages",
    "/usr/lib/python3.12/site-packages",
    "/usr/lib/python3.11/site-packages",
    "/usr/lib/python3/dist-packages",
]

# GNURadio imports - these may fail if not installed
try:
    from gnuradio import gr
    from gnuradio import blocks
    import osmosdr

    GNURADIO_AVAILABLE = True
except ImportError:
    # Try adding system site-packages
    GNURADIO_AVAILABLE = False
    for path in _SYSTEM_SITE_PACKAGES:
        if path not in sys.path:
            sys.path.insert(0, path)
        try:
            from gnuradio import gr
            from gnuradio import blocks
            import osmosdr
            GNURADIO_AVAILABLE = True
            break
        except ImportError:
            continue

    if not GNURADIO_AVAILABLE:
        gr = None
        blocks = None
        osmosdr = None


@dataclass
class CaptureConfig:
    """Configuration for HackRF capture."""

    center_freq: float = 915e6  # Hz
    sample_rate: float = 2e6  # Hz
    gain: float = 40  # dB (RF gain)
    if_gain: float = 20  # dB (IF gain)
    bb_gain: float = 20  # dB (Baseband gain)
    buffer_size: int = 1024  # samples per capture
    bandwidth: float = 0  # Hz (0 = automatic)


class HackRFCapture:
    """Interface for capturing I/Q samples from HackRF via GNURadio.

    This class sets up a GNURadio flowgraph with an osmosdr source
    and provides methods to read samples synchronously.

    Example:
        capture = HackRFCapture(center_freq=915e6, sample_rate=2e6, gain=40)
        capture.start()
        samples = capture.read_samples(1024)
        capture.stop()
    """

    def __init__(
        self,
        center_freq: float = 915e6,
        sample_rate: float = 2e6,
        gain: float = 40,
        if_gain: float = 20,
        bb_gain: float = 20,
        buffer_size: int = 1024,
        bandwidth: float = 0,
        device_args: str = "",
    ):
        """Initialize HackRF capture.

        Args:
            center_freq: Center frequency in Hz.
            sample_rate: Sample rate in Hz.
            gain: RF gain in dB.
            if_gain: IF gain in dB.
            bb_gain: Baseband gain in dB.
            buffer_size: Default number of samples per read.
            bandwidth: Bandwidth in Hz (0 = automatic).
            device_args: Additional device arguments for osmosdr.
        """
        if not GNURADIO_AVAILABLE:
            raise RuntimeError(
                "GNURadio and/or osmosdr not available. "
                "Install via system package manager (e.g., apt install gnuradio gr-osmosdr)"
            )

        self.config = CaptureConfig(
            center_freq=center_freq,
            sample_rate=sample_rate,
            gain=gain,
            if_gain=if_gain,
            bb_gain=bb_gain,
            buffer_size=buffer_size,
            bandwidth=bandwidth,
        )
        self.device_args = device_args

        self._flowgraph = None
        self._source = None
        self._sink = None
        self._is_running = False
        self._sample_queue: Queue[NDArray[np.complex64]] = Queue(maxsize=100)
        self._thread = None

    def _create_flowgraph(self) -> None:
        """Create the GNURadio flowgraph."""

        class CaptureFlowgraph(gr.top_block):
            """GNURadio flowgraph for HackRF capture."""

            def __init__(self, config: CaptureConfig, device_args: str, queue: Queue):
                gr.top_block.__init__(self, "HackRF Capture")

                # Create osmosdr source
                # Device string format: "hackrf=0" or custom
                device_str = device_args if device_args else "hackrf=0"
                self.source = osmosdr.source(args=f"numchan=1 {device_str}")

                # Configure source
                self.source.set_sample_rate(config.sample_rate)
                self.source.set_center_freq(config.center_freq, 0)
                self.source.set_freq_corr(0, 0)
                self.source.set_gain(config.gain, 0)
                self.source.set_if_gain(config.if_gain, 0)
                self.source.set_bb_gain(config.bb_gain, 0)

                if config.bandwidth > 0:
                    self.source.set_bandwidth(config.bandwidth, 0)

                # Create custom sink that pushes to queue
                self.sink = QueueSink(queue, config.buffer_size)

                # Connect source to sink
                self.connect(self.source, self.sink)

        self._flowgraph = CaptureFlowgraph(self.config, self.device_args, self._sample_queue)
        self._source = self._flowgraph.source

    def start(self) -> None:
        """Start the capture stream."""
        if self._is_running:
            return

        self._create_flowgraph()
        self._flowgraph.start()
        self._is_running = True

    def stop(self) -> None:
        """Stop the capture stream and release device."""
        if not self._is_running:
            return

        self._flowgraph.stop()
        self._flowgraph.wait()
        self._is_running = False

        # Clear any remaining samples in queue
        while not self._sample_queue.empty():
            try:
                self._sample_queue.get_nowait()
            except Empty:
                break

    def read_samples(self, num_samples: int | None = None, timeout: float = 1.0) -> NDArray[np.complex64]:
        """Read samples from the capture stream.

        Args:
            num_samples: Number of samples to read. If None, uses buffer_size.
            timeout: Timeout in seconds for blocking read.

        Returns:
            Complex64 array of I/Q samples.

        Raises:
            RuntimeError: If capture is not running.
            TimeoutError: If no samples available within timeout.
        """
        if not self._is_running:
            raise RuntimeError("Capture not running. Call start() first.")

        num_samples = num_samples or self.config.buffer_size

        try:
            samples = self._sample_queue.get(timeout=timeout)
            if len(samples) >= num_samples:
                return samples[:num_samples]
            return samples
        except Empty:
            raise TimeoutError(f"No samples received within {timeout}s timeout")

    def read_samples_nonblocking(self, num_samples: int | None = None) -> NDArray[np.complex64] | None:
        """Non-blocking read of samples.

        Args:
            num_samples: Number of samples to read.

        Returns:
            Complex64 array or None if no samples available.
        """
        if not self._is_running:
            return None

        num_samples = num_samples or self.config.buffer_size

        try:
            samples = self._sample_queue.get_nowait()
            if len(samples) >= num_samples:
                return samples[:num_samples]
            return samples
        except Empty:
            return None

    def set_center_freq(self, freq: float) -> None:
        """Change center frequency while running.

        Args:
            freq: New center frequency in Hz.
        """
        self.config.center_freq = freq
        if self._source is not None:
            self._source.set_center_freq(freq, 0)

    def set_gain(self, gain: float) -> None:
        """Change RF gain while running.

        Args:
            gain: New gain in dB.
        """
        self.config.gain = gain
        if self._source is not None:
            self._source.set_gain(gain, 0)

    def get_info(self) -> dict:
        """Get current capture configuration.

        Returns:
            Dict with capture settings.
        """
        return {
            "center_freq": self.config.center_freq,
            "sample_rate": self.config.sample_rate,
            "gain": self.config.gain,
            "if_gain": self.config.if_gain,
            "bb_gain": self.config.bb_gain,
            "buffer_size": self.config.buffer_size,
            "is_running": self._is_running,
        }

    @property
    def is_running(self) -> bool:
        """Whether capture is currently running."""
        return self._is_running

    def __enter__(self) -> "HackRFCapture":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()


# QueueSink is only defined when GNURadio is available
if GNURADIO_AVAILABLE:

    class QueueSink(gr.sync_block):
        """GNURadio sink block that pushes samples to a queue."""

        def __init__(self, queue: Queue, chunk_size: int):
            gr.sync_block.__init__(
                self,
                name="Queue Sink",
                in_sig=[np.complex64],
                out_sig=None,
            )
            self._queue = queue
            self._chunk_size = chunk_size
            self._buffer = np.zeros(0, dtype=np.complex64)

        def work(self, input_items, output_items):
            """Process incoming samples."""
            samples = input_items[0]

            # Accumulate samples
            self._buffer = np.concatenate([self._buffer, samples])

            # Push complete chunks to queue
            while len(self._buffer) >= self._chunk_size:
                chunk = self._buffer[: self._chunk_size].copy()
                self._buffer = self._buffer[self._chunk_size :]

                try:
                    self._queue.put_nowait(chunk)
                except:
                    # Queue full, drop oldest
                    try:
                        self._queue.get_nowait()
                        self._queue.put_nowait(chunk)
                    except:
                        pass

            return len(samples)


class SimulatedCapture:
    """Simulated capture source for testing without hardware.

    Generates synthetic signals that mimic real RF captures.
    Useful for development and testing when HackRF is not available.
    """

    def __init__(
        self,
        center_freq: float = 915e6,
        sample_rate: float = 2e6,
        gain: float = 40,
        buffer_size: int = 1024,
        snr_db: float = 15.0,
        modulation: str = "qpsk",
    ):
        """Initialize simulated capture.

        Args:
            center_freq: Simulated center frequency in Hz.
            sample_rate: Sample rate in Hz.
            gain: Simulated gain in dB (affects noise level).
            buffer_size: Samples per read.
            snr_db: Signal-to-noise ratio in dB.
            modulation: Simulated modulation type.
        """
        self.config = CaptureConfig(
            center_freq=center_freq,
            sample_rate=sample_rate,
            gain=gain,
            buffer_size=buffer_size,
        )
        self.snr_db = snr_db
        self.modulation = modulation
        self._is_running = False
        self._rng = np.random.default_rng()

    def start(self) -> None:
        """Start simulated capture."""
        self._is_running = True

    def stop(self) -> None:
        """Stop simulated capture."""
        self._is_running = False

    def read_samples(self, num_samples: int | None = None, timeout: float = 1.0) -> NDArray[np.complex64]:
        """Generate simulated samples.

        Args:
            num_samples: Number of samples to generate.
            timeout: Ignored (for API compatibility).

        Returns:
            Complex64 array of simulated samples.
        """
        if not self._is_running:
            raise RuntimeError("Capture not running. Call start() first.")

        num_samples = num_samples or self.config.buffer_size

        # Generate QPSK-like signal
        symbols_per_sample = 10
        num_symbols = num_samples // symbols_per_sample + 1

        # Random QPSK symbols
        symbols = np.exp(1j * np.pi * (self._rng.integers(0, 4, num_symbols) * 2 + 1) / 4)

        # Upsample
        signal = np.repeat(symbols, symbols_per_sample)[:num_samples]

        # Add noise
        snr_linear = 10 ** (self.snr_db / 10)
        signal_power = np.mean(np.abs(signal) ** 2)
        noise_power = signal_power / snr_linear
        noise = np.sqrt(noise_power / 2) * (
            self._rng.standard_normal(num_samples)
            + 1j * self._rng.standard_normal(num_samples)
        )

        return (signal + noise).astype(np.complex64)

    def read_samples_nonblocking(self, num_samples: int | None = None) -> NDArray[np.complex64] | None:
        """Non-blocking read (always returns samples for simulation)."""
        if not self._is_running:
            return None
        return self.read_samples(num_samples)

    def set_center_freq(self, freq: float) -> None:
        """Change simulated center frequency."""
        self.config.center_freq = freq

    def set_gain(self, gain: float) -> None:
        """Change simulated gain."""
        self.config.gain = gain

    def get_info(self) -> dict:
        """Get simulated capture configuration."""
        return {
            "center_freq": self.config.center_freq,
            "sample_rate": self.config.sample_rate,
            "gain": self.config.gain,
            "buffer_size": self.config.buffer_size,
            "is_running": self._is_running,
            "simulated": True,
            "snr_db": self.snr_db,
        }

    @property
    def is_running(self) -> bool:
        """Whether simulated capture is running."""
        return self._is_running

    def __enter__(self) -> "SimulatedCapture":
        """Context manager entry."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.stop()


def create_capture(
    use_simulation: bool = False,
    **kwargs,
) -> HackRFCapture | SimulatedCapture:
    """Factory function to create appropriate capture source.

    Args:
        use_simulation: If True, use simulated capture.
        **kwargs: Arguments passed to capture constructor.

    Returns:
        HackRFCapture or SimulatedCapture instance.
    """
    if use_simulation or not GNURADIO_AVAILABLE:
        return SimulatedCapture(**kwargs)
    return HackRFCapture(**kwargs)


def is_gnuradio_available() -> bool:
    """Check if GNURadio is available.

    Returns:
        True if gnuradio and osmosdr are installed.
    """
    return GNURADIO_AVAILABLE
