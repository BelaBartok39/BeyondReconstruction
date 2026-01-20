"""Tests for TorchRF Testbed HDF5 recorder."""

import numpy as np
import pytest
import tempfile
import sys
from pathlib import Path

# Add testbed to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.recorder import SessionRecorder, SessionReader


@pytest.fixture
def temp_hdf5_file():
    """Create a temporary HDF5 file path."""
    import os
    fd, path = tempfile.mkstemp(suffix=".h5")
    os.close(fd)
    os.unlink(path)  # Remove the file so SessionRecorder can create it
    yield path
    # Cleanup
    if os.path.exists(path):
        os.unlink(path)


@pytest.fixture
def sample_signal():
    """Generate a sample complex signal."""
    rng = np.random.default_rng(42)
    return (rng.standard_normal(1024) + 1j * rng.standard_normal(1024)).astype(np.complex64)


class TestSessionRecorder:
    """Test SessionRecorder class."""

    def test_create_recorder(self, temp_hdf5_file):
        """Test creating a session recorder."""
        recorder = SessionRecorder(temp_hdf5_file)
        assert recorder.is_open
        assert recorder.sample_count == 0
        recorder.close()
        assert not recorder.is_open

    def test_add_sample(self, temp_hdf5_file, sample_signal):
        """Test adding a single sample."""
        recorder = SessionRecorder(temp_hdf5_file)
        recorder.add_sample(sample_signal, label=False)
        assert recorder.sample_count == 1
        recorder.close()

    def test_add_anomaly_sample(self, temp_hdf5_file, sample_signal):
        """Test adding an anomaly sample."""
        recorder = SessionRecorder(temp_hdf5_file)
        recorder.add_sample(
            sample_signal,
            label=True,
            anomaly_type="tone",
            snr_db=15.0,
            power_db=-5.0,
            score=8.5,
        )
        assert recorder.sample_count == 1
        recorder.close()

    def test_add_multiple_samples(self, temp_hdf5_file, sample_signal):
        """Test adding multiple samples."""
        recorder = SessionRecorder(temp_hdf5_file)
        for i in range(10):
            recorder.add_sample(sample_signal, label=(i % 2 == 0))
        assert recorder.sample_count == 10
        recorder.close()

    def test_add_batch(self, temp_hdf5_file):
        """Test adding a batch of samples."""
        rng = np.random.default_rng(42)
        signals = (rng.standard_normal((50, 1024)) + 1j * rng.standard_normal((50, 1024))).astype(np.complex64)
        labels = np.array([i % 2 == 0 for i in range(50)])

        recorder = SessionRecorder(temp_hdf5_file)
        recorder.add_batch(signals, labels)
        assert recorder.sample_count == 50
        recorder.close()

    def test_context_manager(self, temp_hdf5_file, sample_signal):
        """Test context manager usage."""
        with SessionRecorder(temp_hdf5_file) as recorder:
            recorder.add_sample(sample_signal, label=False)
            assert recorder.is_open
        # Should be closed after exiting context
        assert not recorder.is_open

    def test_get_stats(self, temp_hdf5_file, sample_signal):
        """Test getting recorder statistics."""
        recorder = SessionRecorder(temp_hdf5_file)
        recorder.add_sample(sample_signal, label=False)
        recorder.add_sample(sample_signal, label=True)

        stats = recorder.get_stats()
        assert stats["sample_count"] == 2
        assert stats["num_anomalies"] == 1
        assert stats["is_open"] == True
        recorder.close()

    def test_overwrite_protection(self, temp_hdf5_file):
        """Test that overwriting is protected."""
        recorder = SessionRecorder(temp_hdf5_file)
        recorder.close()

        # Should raise error without overwrite flag
        with pytest.raises(FileExistsError):
            SessionRecorder(temp_hdf5_file, overwrite=False)

        # Should work with overwrite flag
        recorder2 = SessionRecorder(temp_hdf5_file, overwrite=True)
        recorder2.close()


class TestSessionReader:
    """Test SessionReader class."""

    def test_read_recorded_session(self, temp_hdf5_file, sample_signal):
        """Test reading a recorded session."""
        # Write some data
        with SessionRecorder(temp_hdf5_file) as recorder:
            recorder.add_sample(sample_signal, label=False, snr_db=10.0)
            recorder.add_sample(sample_signal, label=True, anomaly_type="tone", snr_db=15.0)

        # Read it back
        reader = SessionReader(temp_hdf5_file)
        assert len(reader) == 2
        reader.close()

    def test_read_sample_by_index(self, temp_hdf5_file, sample_signal):
        """Test reading individual samples by index."""
        with SessionRecorder(temp_hdf5_file) as recorder:
            recorder.add_sample(sample_signal, label=False, snr_db=10.0)
            recorder.add_sample(sample_signal, label=True, anomaly_type="chirp", snr_db=20.0)

        reader = SessionReader(temp_hdf5_file)
        sample = reader[0]
        assert sample["label"] == False
        assert abs(sample["snr_db"] - 10.0) < 0.01

        sample = reader[1]
        assert sample["label"] == True
        reader.close()

    def test_iterate_samples(self, temp_hdf5_file, sample_signal):
        """Test iterating over samples."""
        with SessionRecorder(temp_hdf5_file) as recorder:
            for i in range(5):
                recorder.add_sample(sample_signal, label=(i % 2 == 0))

        reader = SessionReader(temp_hdf5_file)
        count = 0
        for sample in reader:
            count += 1
            assert "signal" in sample
            assert "label" in sample
        assert count == 5
        reader.close()

    def test_get_all_signals(self, temp_hdf5_file, sample_signal):
        """Test getting all signals at once."""
        with SessionRecorder(temp_hdf5_file) as recorder:
            for _ in range(10):
                recorder.add_sample(sample_signal, label=False)

        reader = SessionReader(temp_hdf5_file)
        signals = reader.get_signals()
        assert signals.shape == (10, 1024)
        assert signals.dtype == np.complex64
        reader.close()

    def test_get_all_labels(self, temp_hdf5_file, sample_signal):
        """Test getting all labels at once."""
        with SessionRecorder(temp_hdf5_file) as recorder:
            for i in range(10):
                recorder.add_sample(sample_signal, label=(i < 5))

        reader = SessionReader(temp_hdf5_file)
        labels = reader.get_labels()
        assert labels.shape == (10,)
        assert labels.sum() == 5
        reader.close()

    def test_get_metadata(self, temp_hdf5_file, sample_signal):
        """Test reading metadata."""
        with SessionRecorder(temp_hdf5_file, sample_rate=2e6, center_freq=915e6) as recorder:
            recorder.add_sample(sample_signal, label=False)

        reader = SessionReader(temp_hdf5_file)
        metadata = reader.get_metadata()
        assert "sample_rate" in metadata
        assert abs(metadata["sample_rate"] - 2e6) < 1
        assert "center_freq" in metadata
        reader.close()

    def test_context_manager(self, temp_hdf5_file, sample_signal):
        """Test context manager usage for reader."""
        with SessionRecorder(temp_hdf5_file) as recorder:
            recorder.add_sample(sample_signal, label=False)

        with SessionReader(temp_hdf5_file) as reader:
            assert len(reader) == 1

    def test_file_not_found(self):
        """Test error when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            SessionReader("/nonexistent/path/file.h5")


class TestRoundTrip:
    """Test complete write-read roundtrip."""

    def test_signal_integrity(self, temp_hdf5_file):
        """Test that signals are preserved exactly."""
        rng = np.random.default_rng(42)
        original = (rng.standard_normal(1024) + 1j * rng.standard_normal(1024)).astype(np.complex64)

        with SessionRecorder(temp_hdf5_file) as recorder:
            recorder.add_sample(original, label=True, anomaly_type="test", snr_db=15.5)

        with SessionReader(temp_hdf5_file) as reader:
            recovered = reader[0]["signal"]
            np.testing.assert_array_almost_equal(original, recovered, decimal=5)

    def test_metadata_integrity(self, temp_hdf5_file, sample_signal):
        """Test that metadata is preserved correctly."""
        with SessionRecorder(temp_hdf5_file) as recorder:
            recorder.add_sample(
                sample_signal,
                label=True,
                anomaly_type="chirp",
                snr_db=12.5,
                power_db=-3.2,
                score=7.8,
            )

        with SessionReader(temp_hdf5_file) as reader:
            sample = reader[0]
            assert sample["label"] == True
            # Handle both string and bytes for anomaly_type
            atype = sample["anomaly_type"]
            if isinstance(atype, bytes):
                atype = atype.decode("utf-8")
            assert atype == "chirp"
            assert abs(sample["snr_db"] - 12.5) < 0.01
            assert abs(sample["power_db"] - (-3.2)) < 0.01
            assert abs(sample["score"] - 7.8) < 0.01
