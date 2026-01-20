"""Tests for TorchRF Testbed utilities."""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add testbed to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import (
    complex_to_iq,
    iq_to_complex,
    normalize_signal,
    estimate_power,
    estimate_snr,
    normalize_snr_value,
    normalize_power_value,
    segment_signal,
    compute_instantaneous_frequency,
    compute_envelope,
)


class TestComplexIQConversion:
    """Test complex to I/Q conversion functions."""

    def test_complex_to_iq_shape(self):
        """Test that complex_to_iq produces correct shape."""
        signal = np.random.randn(1024) + 1j * np.random.randn(1024)
        signal = signal.astype(np.complex64)
        iq = complex_to_iq(signal)
        assert iq.shape == (2, 1024)
        assert iq.dtype == np.float32

    def test_iq_to_complex_shape(self):
        """Test that iq_to_complex produces correct shape."""
        iq = np.random.randn(2, 1024).astype(np.float32)
        signal = iq_to_complex(iq)
        assert signal.shape == (1024,)
        assert signal.dtype == np.complex64

    def test_roundtrip_conversion(self):
        """Test that conversion is reversible."""
        original = np.random.randn(1024) + 1j * np.random.randn(1024)
        original = original.astype(np.complex64)
        iq = complex_to_iq(original)
        recovered = iq_to_complex(iq)
        np.testing.assert_array_almost_equal(original, recovered, decimal=5)


class TestNormalization:
    """Test signal normalization functions."""

    def test_normalize_signal_max_amplitude(self):
        """Test that normalized signal has unit max amplitude."""
        signal = 10 * (np.random.randn(1024) + 1j * np.random.randn(1024))
        signal = signal.astype(np.complex64)
        normalized, power_db = normalize_signal(signal)
        assert np.max(np.abs(normalized)) <= 1.0 + 1e-6

    def test_normalize_signal_returns_power(self):
        """Test that power_db is returned correctly."""
        signal = np.ones(1024, dtype=np.complex64)
        normalized, power_db = normalize_signal(signal)
        assert isinstance(power_db, float)
        # Power of constant unit amplitude signal is 0 dB
        assert abs(power_db - 0.0) < 0.1

    def test_normalize_snr_value_range(self):
        """Test SNR normalization to [0, 1] range."""
        snr_range = (-5, 30)
        assert normalize_snr_value(-5, snr_range) == 0.0
        assert normalize_snr_value(30, snr_range) == 1.0
        assert 0 < normalize_snr_value(15, snr_range) < 1

    def test_normalize_power_value_clipping(self):
        """Test power normalization clips out-of-range values."""
        power_range = (-20, 10)
        assert normalize_power_value(-30, power_range) == 0.0
        assert normalize_power_value(20, power_range) == 1.0


class TestSNREstimation:
    """Test SNR estimation functions."""

    def test_estimate_snr_m2m4(self):
        """Test M2M4 SNR estimation returns reasonable value."""
        signal = np.random.randn(1024) + 1j * np.random.randn(1024)
        signal = signal.astype(np.complex64)
        snr = estimate_snr(signal, method="m2m4")
        assert isinstance(snr, float)
        assert -10 <= snr <= 40

    def test_estimate_snr_wavelet(self):
        """Test wavelet SNR estimation returns reasonable value."""
        signal = np.random.randn(1024) + 1j * np.random.randn(1024)
        signal = signal.astype(np.complex64)
        snr = estimate_snr(signal, method="wavelet")
        assert isinstance(snr, float)
        assert -10 <= snr <= 40

    def test_estimate_snr_spectral(self):
        """Test spectral SNR estimation returns reasonable value."""
        signal = np.random.randn(1024) + 1j * np.random.randn(1024)
        signal = signal.astype(np.complex64)
        snr = estimate_snr(signal, method="spectral")
        assert isinstance(snr, float)
        assert -10 <= snr <= 40

    def test_estimate_snr_invalid_method(self):
        """Test that invalid method raises ValueError."""
        signal = np.random.randn(1024).astype(np.complex64)
        with pytest.raises(ValueError):
            estimate_snr(signal, method="invalid")


class TestPowerEstimation:
    """Test power estimation functions."""

    def test_estimate_power_unit_signal(self):
        """Test power estimation for unit amplitude signal."""
        signal = np.ones(1024, dtype=np.complex64)
        power_db = estimate_power(signal)
        assert abs(power_db - 0.0) < 0.1

    def test_estimate_power_scaled_signal(self):
        """Test power estimation scales with amplitude."""
        signal = np.ones(1024, dtype=np.complex64)
        power_1 = estimate_power(signal)
        power_2 = estimate_power(signal * 10)
        # 10x amplitude = 20 dB increase
        assert abs((power_2 - power_1) - 20.0) < 0.1


class TestSegmentation:
    """Test signal segmentation functions."""

    def test_segment_signal_no_overlap(self):
        """Test segmentation without overlap."""
        signal = np.random.randn(4096).astype(np.complex64)
        segments = segment_signal(signal, segment_length=1024, overlap=0)
        assert len(segments) == 4
        assert all(len(s) == 1024 for s in segments)

    def test_segment_signal_with_overlap(self):
        """Test segmentation with overlap."""
        signal = np.random.randn(2048).astype(np.complex64)
        segments = segment_signal(signal, segment_length=1024, overlap=512)
        assert len(segments) == 3
        assert all(len(s) == 1024 for s in segments)

    def test_segment_signal_short_signal(self):
        """Test segmentation of signal shorter than segment length."""
        signal = np.random.randn(500).astype(np.complex64)
        segments = segment_signal(signal, segment_length=1024)
        assert len(segments) == 0


class TestFeatureExtraction:
    """Test feature extraction functions."""

    def test_compute_instantaneous_frequency_shape(self):
        """Test instantaneous frequency output shape."""
        signal = np.exp(1j * np.linspace(0, 10 * np.pi, 1024)).astype(np.complex64)
        freq = compute_instantaneous_frequency(signal)
        assert freq.shape == (1023,)
        assert freq.dtype == np.float32

    def test_compute_envelope_shape(self):
        """Test envelope output shape."""
        signal = np.random.randn(1024) + 1j * np.random.randn(1024)
        signal = signal.astype(np.complex64)
        envelope = compute_envelope(signal)
        assert envelope.shape == (1024,)
        assert envelope.dtype == np.float32
        assert np.all(envelope >= 0)
