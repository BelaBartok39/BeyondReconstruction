"""Tests for anomaly detection."""

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.vae import ConvVAE
from src.detection.detector import AnomalyDetector
from src.detection.metrics import compute_metrics, compute_snr_stratified_metrics


class TestAnomalyDetector:
    """Test anomaly detector."""

    @pytest.fixture
    def model(self):
        """Create model fixture."""
        return ConvVAE(
            latent_dim=16,
            sequence_length=256,
            hidden_channels=[16, 32],
        )

    @pytest.fixture
    def dataloader(self):
        """Create sample dataloader."""
        # Normal samples
        iq = torch.randn(100, 2, 256) * 0.1  # Low variance = "normal"
        snr = torch.rand(100)
        labels = torch.zeros(100, dtype=torch.long)
        snr_db = torch.rand(100) * 30 - 5

        dataset = TensorDataset(iq, snr, labels, snr_db)

        class DictDataLoader(DataLoader):
            def __iter__(self):
                for iq, snr, label, snr_db in super().__iter__():
                    yield {"iq": iq, "snr": snr, "label": label, "snr_db": snr_db}

        return DictDataLoader(dataset, batch_size=32)

    def test_detector_init(self, model):
        """Test detector initialization."""
        detector = AnomalyDetector(model, method="reconstruction")
        assert detector.method == "reconstruction"
        assert not detector._is_fitted

    def test_detector_fit(self, model, dataloader):
        """Test detector fitting."""
        detector = AnomalyDetector(model, method="reconstruction")
        detector.fit(dataloader)

        assert detector._is_fitted
        assert detector._threshold is not None

    def test_detector_detect(self, model, dataloader):
        """Test anomaly detection."""
        detector = AnomalyDetector(model, method="reconstruction")
        detector.fit(dataloader)

        # Create test batch
        test_iq = torch.randn(10, 2, 256)
        test_snr = torch.rand(10)

        result = detector.detect(test_iq, test_snr)

        assert len(result.scores) == 10
        assert len(result.predictions) == 10
        assert result.threshold > 0

    def test_detector_methods(self, model, dataloader):
        """Test different detection methods."""
        for method in ["reconstruction", "latent", "hybrid"]:
            detector = AnomalyDetector(model, method=method)
            detector.fit(dataloader)

            test_iq = torch.randn(5, 2, 256)
            test_snr = torch.rand(5)
            result = detector.detect(test_iq, test_snr)

            assert len(result.scores) == 5

    def test_snr_adaptive_threshold(self, model, dataloader):
        """Test SNR-adaptive thresholding."""
        detector = AnomalyDetector(
            model,
            method="reconstruction",
            snr_adaptive=True,
            snr_bins=5,
        )
        detector.fit(dataloader)

        assert detector._snr_thresholds is not None
        assert len(detector._snr_thresholds) == 5


class TestMetrics:
    """Test detection metrics."""

    def test_compute_metrics(self):
        """Test metric computation."""
        # Perfect predictions
        scores = np.array([0.1, 0.2, 0.3, 0.8, 0.9, 1.0])
        labels = np.array([0, 0, 0, 1, 1, 1])

        metrics = compute_metrics(scores, labels)

        assert 0 <= metrics.auroc <= 1
        assert 0 <= metrics.auprc <= 1
        assert 0 <= metrics.f1 <= 1
        assert metrics.true_positives + metrics.false_negatives == 3
        assert metrics.true_negatives + metrics.false_positives == 3

    def test_compute_metrics_perfect(self):
        """Test perfect classification."""
        scores = np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
        labels = np.array([0, 0, 0, 1, 1, 1])

        metrics = compute_metrics(scores, labels, threshold=0.5)

        assert metrics.auroc == 1.0
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0

    def test_compute_metrics_random(self):
        """Test random predictions give ~0.5 AUROC."""
        np.random.seed(42)
        scores = np.random.rand(1000)
        labels = np.random.randint(0, 2, 1000)

        metrics = compute_metrics(scores, labels)

        # Random should be close to 0.5
        assert 0.4 <= metrics.auroc <= 0.6

    def test_snr_stratified_metrics(self):
        """Test SNR-stratified metric computation."""
        np.random.seed(42)
        scores = np.random.rand(500)
        labels = np.random.randint(0, 2, 500)
        snr_db = np.random.uniform(-5, 30, 500)

        snr_metrics = compute_snr_stratified_metrics(
            scores, labels, snr_db, num_bins=5
        )

        assert len(snr_metrics.snr_bins) == 5
        assert len(snr_metrics.metrics_per_bin) == 5
        assert len(snr_metrics.sample_counts) == 5

    def test_metrics_edge_cases(self):
        """Test metrics with edge cases."""
        # All normal
        scores = np.array([0.1, 0.2, 0.3])
        labels = np.array([0, 0, 0])

        metrics = compute_metrics(scores, labels)
        assert metrics.auroc == 0.5  # Default for single class

        # All anomaly
        scores = np.array([0.7, 0.8, 0.9])
        labels = np.array([1, 1, 1])

        metrics = compute_metrics(scores, labels)
        assert metrics.auroc == 0.5
