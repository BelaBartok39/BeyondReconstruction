"""Tests for synthetic signal generation."""

import numpy as np
import pytest

from src.data.synthetic import SyntheticRFGenerator, Modulation, AnomalyType


class TestSyntheticRFGenerator:
    """Test synthetic signal generator."""

    @pytest.fixture
    def generator(self):
        """Create generator fixture."""
        return SyntheticRFGenerator(
            sequence_length=1024,
            sample_rate=1e6,
            seed=42,
        )

    def test_generator_init(self, generator):
        """Test generator initialization."""
        assert generator.sequence_length == 1024
        assert generator.sample_rate == 1e6

    def test_normal_signal_shape(self, generator):
        """Test normal signal has correct shape."""
        iq, metadata = generator.generate_normal_signal(modulation="qpsk")

        assert iq.shape == (2, 1024)
        assert iq.dtype == np.float32

    def test_normal_signal_metadata(self, generator):
        """Test normal signal metadata."""
        iq, metadata = generator.generate_normal_signal(
            modulation="bpsk", snr_db=20.0
        )

        assert metadata.modulation == "bpsk"
        assert metadata.snr_db == 20.0
        assert metadata.is_anomaly is False
        assert metadata.anomaly_type is None

    def test_all_modulations(self, generator):
        """Test all modulation types generate valid signals."""
        for mod in Modulation:
            iq, metadata = generator.generate_normal_signal(modulation=mod.value)
            assert iq.shape == (2, 1024)
            assert np.isfinite(iq).all()
            assert metadata.modulation == mod.value

    def test_snr_range(self, generator):
        """Test SNR is within specified range."""
        for _ in range(20):
            iq, metadata = generator.generate_normal_signal(snr_range=(0, 20))
            assert 0 <= metadata.snr_db <= 20

    def test_anomaly_signal(self, generator):
        """Test anomaly signal generation."""
        iq, metadata = generator.generate_anomaly(anomaly_type="interference")

        assert iq.shape == (2, 1024)
        assert metadata.is_anomaly is True
        assert metadata.anomaly_type == "interference"
        assert metadata.anomaly_params is not None

    def test_all_anomaly_types(self, generator):
        """Test all anomaly types generate valid signals."""
        for anom in AnomalyType:
            iq, metadata = generator.generate_anomaly(anomaly_type=anom.value)
            assert iq.shape == (2, 1024)
            assert np.isfinite(iq).all()
            assert metadata.is_anomaly is True
            assert metadata.anomaly_type == anom.value

    def test_batch_generation(self, generator):
        """Test batch generation."""
        iq_batch, metadata_list = generator.generate_batch(
            num_samples=100,
            anomaly_ratio=0.2,
        )

        assert iq_batch.shape == (100, 2, 1024)
        assert len(metadata_list) == 100

        # Check anomaly ratio
        num_anomalies = sum(1 for m in metadata_list if m.is_anomaly)
        assert 15 <= num_anomalies <= 25  # Allow some variance

    def test_reproducibility(self):
        """Test reproducibility with same seed."""
        gen1 = SyntheticRFGenerator(seed=123)
        gen2 = SyntheticRFGenerator(seed=123)

        iq1, _ = gen1.generate_normal_signal()
        iq2, _ = gen2.generate_normal_signal()

        np.testing.assert_array_equal(iq1, iq2)

    def test_signal_normalization(self, generator):
        """Test signals are normalized."""
        for _ in range(10):
            iq, _ = generator.generate_normal_signal()
            assert np.abs(iq).max() <= 1.0 + 1e-6
