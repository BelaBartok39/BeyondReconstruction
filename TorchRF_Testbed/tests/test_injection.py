"""Tests for TorchRF Testbed anomaly injection."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add testbed to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.injection import (
    inject_anomaly,
    inject_tone,
    inject_multi_tone,
    inject_chirp,
    inject_sweep,
    inject_barrage,
    inject_pulse,
    inject_interference,
    inject_frequency_drift,
    inject_amplitude_spike,
    inject_phase_noise,
    inject_burst_noise,
    get_anomaly_types,
    AnomalyType,
)


@pytest.fixture
def sample_signal():
    """Generate a sample complex signal for testing."""
    rng = np.random.default_rng(42)
    signal = rng.standard_normal(1024) + 1j * rng.standard_normal(1024)
    return signal.astype(np.complex64)


class TestInjectionFunctions:
    """Test individual injection functions."""

    def test_inject_tone_shape(self, sample_signal):
        """Test that tone injection preserves signal shape."""
        corrupted, params = inject_tone(sample_signal)
        assert corrupted.shape == sample_signal.shape
        assert corrupted.dtype == np.complex64
        assert "frequency_offset" in params

    def test_inject_multi_tone_shape(self, sample_signal):
        """Test that multi-tone injection preserves signal shape."""
        corrupted, params = inject_multi_tone(sample_signal, num_tones=3)
        assert corrupted.shape == sample_signal.shape
        assert params["num_tones"] == 3
        assert len(params["frequencies_hz"]) == 3

    def test_inject_chirp_shape(self, sample_signal):
        """Test that chirp injection preserves signal shape."""
        corrupted, params = inject_chirp(sample_signal)
        assert corrupted.shape == sample_signal.shape
        assert "chirp_rate" in params

    def test_inject_sweep_is_chirp_alias(self, sample_signal):
        """Test that sweep is an alias for chirp."""
        rng = np.random.default_rng(42)
        chirp_result, chirp_params = inject_chirp(
            sample_signal.copy(), start_freq=-0.2, end_freq=0.2, sir_db=0
        )
        sweep_result, sweep_params = inject_sweep(
            sample_signal.copy(), start_freq=-0.2, end_freq=0.2, sir_db=0
        )
        # Should have same parameter structure
        assert set(chirp_params.keys()) == set(sweep_params.keys())

    def test_inject_barrage_shape(self, sample_signal):
        """Test that barrage injection preserves signal shape."""
        corrupted, params = inject_barrage(sample_signal, bandwidth=0.5)
        assert corrupted.shape == sample_signal.shape
        assert params["bandwidth"] == 0.5

    def test_inject_pulse_shape(self, sample_signal):
        """Test that pulse injection preserves signal shape."""
        corrupted, params = inject_pulse(sample_signal, duty_cycle=0.3)
        assert corrupted.shape == sample_signal.shape
        assert params["duty_cycle"] == 0.3

    def test_inject_interference_shape(self, sample_signal):
        """Test that interference injection preserves signal shape."""
        corrupted, params = inject_interference(sample_signal)
        assert corrupted.shape == sample_signal.shape

    def test_inject_frequency_drift_shape(self, sample_signal):
        """Test that frequency drift injection preserves signal shape."""
        corrupted, params = inject_frequency_drift(sample_signal)
        assert corrupted.shape == sample_signal.shape
        assert "drift_rate_hz_per_sample" in params

    def test_inject_amplitude_spike_shape(self, sample_signal):
        """Test that amplitude spike injection preserves signal shape."""
        corrupted, params = inject_amplitude_spike(sample_signal)
        assert corrupted.shape == sample_signal.shape
        assert "spike_amplitude" in params
        assert "spike_start" in params

    def test_inject_phase_noise_shape(self, sample_signal):
        """Test that phase noise injection preserves signal shape."""
        corrupted, params = inject_phase_noise(sample_signal)
        assert corrupted.shape == sample_signal.shape
        assert "phase_noise_std" in params

    def test_inject_burst_noise_shape(self, sample_signal):
        """Test that burst noise injection preserves signal shape."""
        corrupted, params = inject_burst_noise(sample_signal)
        assert corrupted.shape == sample_signal.shape
        assert "num_bursts" in params
        assert "bursts" in params


class TestInjectAnomaly:
    """Test the main inject_anomaly function."""

    def test_inject_anomaly_random_type(self, sample_signal):
        """Test random anomaly type selection."""
        corrupted, metadata = inject_anomaly(sample_signal)
        assert corrupted.shape == sample_signal.shape
        assert metadata.anomaly_type in get_anomaly_types()

    def test_inject_anomaly_specific_type(self, sample_signal):
        """Test specific anomaly type injection."""
        corrupted, metadata = inject_anomaly(sample_signal, anomaly_type="tone")
        assert corrupted.shape == sample_signal.shape
        assert metadata.anomaly_type == "tone"

    def test_inject_anomaly_with_severity(self, sample_signal):
        """Test anomaly severity parameter."""
        _, metadata_low = inject_anomaly(sample_signal.copy(), anomaly_type="amplitude_spike", severity=0.5)
        _, metadata_high = inject_anomaly(sample_signal.copy(), anomaly_type="amplitude_spike", severity=2.0)
        assert metadata_low.severity == 0.5
        assert metadata_high.severity == 2.0

    def test_inject_anomaly_all_types(self, sample_signal):
        """Test that all anomaly types can be injected."""
        for atype in get_anomaly_types():
            corrupted, metadata = inject_anomaly(sample_signal.copy(), anomaly_type=atype)
            assert corrupted.shape == sample_signal.shape
            assert metadata.anomaly_type == atype

    def test_inject_anomaly_enum_type(self, sample_signal):
        """Test anomaly type as enum."""
        corrupted, metadata = inject_anomaly(sample_signal, anomaly_type=AnomalyType.CHIRP)
        assert corrupted.shape == sample_signal.shape
        assert metadata.anomaly_type == "chirp"


class TestAnomalyTypes:
    """Test anomaly type enumeration."""

    def test_get_anomaly_types_not_empty(self):
        """Test that anomaly types list is not empty."""
        types = get_anomaly_types()
        assert len(types) > 0

    def test_anomaly_type_enum_values(self):
        """Test that all enum values are strings."""
        for atype in AnomalyType:
            assert isinstance(atype.value, str)

    def test_expected_anomaly_types(self):
        """Test that expected anomaly types are present."""
        types = get_anomaly_types()
        expected = ["tone", "chirp", "barrage", "pulse", "interference"]
        for exp in expected:
            assert exp in types


class TestInjectionQuality:
    """Test that injections modify signals appropriately."""

    def test_tone_changes_signal(self, sample_signal):
        """Test that tone injection modifies the signal."""
        corrupted, _ = inject_tone(sample_signal, sir_db=-10)  # Strong interference
        diff = np.abs(corrupted - sample_signal)
        assert np.mean(diff) > 0.01

    def test_amplitude_spike_creates_spike(self, sample_signal):
        """Test that amplitude spike creates noticeable spike."""
        corrupted, params = inject_amplitude_spike(
            sample_signal, spike_amplitude=10, spike_duration=100
        )
        start = params["spike_start"]
        dur = params["spike_duration"]
        # Spike region should have higher amplitude
        spike_power = np.mean(np.abs(corrupted[start : start + dur]) ** 2)
        normal_power = np.mean(np.abs(sample_signal) ** 2)
        assert spike_power > normal_power

    def test_frequency_drift_changes_phase(self, sample_signal):
        """Test that frequency drift modifies phase."""
        # Create a clean signal for this test
        t = np.arange(1024)
        clean = np.exp(1j * 0.1 * t).astype(np.complex64)
        corrupted, _ = inject_frequency_drift(clean, drift_rate=10)
        # Phase difference should grow over time
        phase_diff = np.angle(corrupted) - np.angle(clean)
        assert np.abs(phase_diff[-1]) > np.abs(phase_diff[0])
