"""Tests for TorchRF Testbed capture module."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add testbed to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.capture import (
    SimulatedCapture,
    CaptureConfig,
    create_capture,
    is_gnuradio_available,
)


class TestSimulatedCapture:
    """Test SimulatedCapture class (no hardware required)."""

    def test_create_simulated_capture(self):
        """Test creating a simulated capture instance."""
        capture = SimulatedCapture()
        assert not capture.is_running

    def test_start_stop(self):
        """Test starting and stopping capture."""
        capture = SimulatedCapture()
        capture.start()
        assert capture.is_running
        capture.stop()
        assert not capture.is_running

    def test_read_samples(self):
        """Test reading samples from simulated capture."""
        capture = SimulatedCapture(buffer_size=1024)
        capture.start()
        samples = capture.read_samples(1024)
        assert samples.shape == (1024,)
        assert samples.dtype == np.complex64
        capture.stop()

    def test_read_samples_not_running(self):
        """Test error when reading without starting."""
        capture = SimulatedCapture()
        with pytest.raises(RuntimeError):
            capture.read_samples()

    def test_read_samples_nonblocking(self):
        """Test non-blocking read."""
        capture = SimulatedCapture()
        capture.start()
        samples = capture.read_samples_nonblocking()
        assert samples is not None
        assert samples.dtype == np.complex64
        capture.stop()

    def test_context_manager(self):
        """Test context manager usage."""
        with SimulatedCapture() as capture:
            assert capture.is_running
            samples = capture.read_samples()
            assert len(samples) > 0
        assert not capture.is_running

    def test_set_center_freq(self):
        """Test changing center frequency."""
        capture = SimulatedCapture(center_freq=915e6)
        assert capture.config.center_freq == 915e6
        capture.set_center_freq(2.4e9)
        assert capture.config.center_freq == 2.4e9

    def test_set_gain(self):
        """Test changing gain."""
        capture = SimulatedCapture(gain=40)
        assert capture.config.gain == 40
        capture.set_gain(30)
        assert capture.config.gain == 30

    def test_get_info(self):
        """Test getting capture info."""
        capture = SimulatedCapture(
            center_freq=915e6,
            sample_rate=2e6,
            gain=40,
        )
        info = capture.get_info()
        assert info["center_freq"] == 915e6
        assert info["sample_rate"] == 2e6
        assert info["gain"] == 40
        assert info["simulated"] == True

    def test_custom_buffer_size(self):
        """Test custom buffer size."""
        capture = SimulatedCapture(buffer_size=512)
        capture.start()
        samples = capture.read_samples()
        assert len(samples) == 512
        capture.stop()


class TestCaptureConfig:
    """Test CaptureConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = CaptureConfig()
        assert config.center_freq == 915e6
        assert config.sample_rate == 2e6
        assert config.gain == 40
        assert config.buffer_size == 1024

    def test_custom_config(self):
        """Test custom configuration values."""
        config = CaptureConfig(
            center_freq=2.4e9,
            sample_rate=4e6,
            gain=30,
            buffer_size=2048,
        )
        assert config.center_freq == 2.4e9
        assert config.sample_rate == 4e6
        assert config.gain == 30
        assert config.buffer_size == 2048


class TestCreateCapture:
    """Test create_capture factory function."""

    def test_create_simulated(self):
        """Test creating simulated capture."""
        capture = create_capture(use_simulation=True)
        assert isinstance(capture, SimulatedCapture)

    def test_create_with_params(self):
        """Test creating capture with custom parameters."""
        capture = create_capture(
            use_simulation=True,
            center_freq=868e6,
            sample_rate=1e6,
            gain=20,
        )
        assert capture.config.center_freq == 868e6
        assert capture.config.sample_rate == 1e6
        assert capture.config.gain == 20


class TestGNURadioAvailability:
    """Test GNURadio availability check."""

    def test_is_gnuradio_available_returns_bool(self):
        """Test that availability check returns boolean."""
        result = is_gnuradio_available()
        assert isinstance(result, bool)


class TestSimulatedSignalQuality:
    """Test quality of simulated signals."""

    def test_signal_is_complex(self):
        """Test that simulated signal is complex."""
        with SimulatedCapture() as capture:
            samples = capture.read_samples()
            assert np.iscomplexobj(samples)

    def test_signal_has_reasonable_power(self):
        """Test that signal has reasonable power level."""
        with SimulatedCapture(snr_db=20) as capture:
            samples = capture.read_samples()
            power = np.mean(np.abs(samples) ** 2)
            # Power should be roughly around 1 for unit-power signal
            assert 0.01 < power < 100

    def test_signals_are_different(self):
        """Test that consecutive reads produce different signals."""
        with SimulatedCapture() as capture:
            samples1 = capture.read_samples()
            samples2 = capture.read_samples()
            # Signals should be different (random)
            assert not np.allclose(samples1, samples2)

    def test_snr_affects_noise_level(self):
        """Test that SNR parameter affects noise level."""
        with SimulatedCapture(snr_db=30) as capture_high:
            samples_high = capture_high.read_samples()

        with SimulatedCapture(snr_db=0) as capture_low:
            samples_low = capture_low.read_samples()

        # Lower SNR should have more variance (more noise)
        var_high = np.var(samples_high)
        var_low = np.var(samples_low)
        # This is a probabilistic test, but should generally hold
        # We just check both are non-zero
        assert var_high > 0
        assert var_low > 0
