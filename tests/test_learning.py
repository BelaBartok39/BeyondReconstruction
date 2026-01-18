"""Tests for continuous learning modules."""

import torch
import pytest
from torch.utils.data import DataLoader, TensorDataset

from src.models.vae import ConvVAE
from src.learning.online import OnlineLearner
from src.learning.ewc import EWCLearner
from src.learning.replay_buffer import ReplayBuffer, StratifiedReplayBuffer


class TestOnlineLearner:
    """Test online learning module."""

    @pytest.fixture
    def model(self):
        """Create model fixture."""
        return ConvVAE(
            latent_dim=16,
            sequence_length=256,
            hidden_channels=[16, 32],
        )

    @pytest.fixture
    def batch(self):
        """Create sample batch."""
        return {
            "iq": torch.randn(16, 2, 256),
            "snr": torch.rand(16),
        }

    def test_learner_init(self, model):
        """Test learner initialization."""
        learner = OnlineLearner(model, learning_rate=1e-4)
        assert learner.learning_rate == 1e-4

    def test_learner_update(self, model, batch):
        """Test online update."""
        learner = OnlineLearner(model, learning_rate=1e-4)

        # Get initial parameters
        initial_params = [p.clone() for p in model.parameters()]

        # Perform update
        metrics = learner.update(batch)

        assert "loss" in metrics
        assert metrics["loss"] > 0

        # Parameters should change
        current_params = list(model.parameters())
        changed = any(
            not torch.allclose(init, curr)
            for init, curr in zip(initial_params, current_params)
        )
        assert changed

    def test_learner_update_frequency(self, model, batch):
        """Test update frequency control."""
        learner = OnlineLearner(model, update_frequency=3)

        # First two updates should be skipped
        metrics1 = learner.update(batch)
        metrics2 = learner.update(batch)
        assert metrics1.get("skipped", False)
        assert metrics2.get("skipped", False)

        # Third should perform update
        metrics3 = learner.update(batch)
        assert "loss" in metrics3

    def test_set_learning_rate(self, model):
        """Test learning rate adjustment."""
        learner = OnlineLearner(model, learning_rate=1e-3)
        learner.set_learning_rate(1e-5)

        assert learner.learning_rate == 1e-5
        assert learner.optimizer.param_groups[0]["lr"] == 1e-5


class TestEWCLearner:
    """Test Elastic Weight Consolidation."""

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
        iq = torch.randn(50, 2, 256)
        snr = torch.rand(50)

        dataset = TensorDataset(iq, snr)

        class DictDataLoader(DataLoader):
            def __iter__(self):
                for iq, snr in super().__iter__():
                    yield {"iq": iq, "snr": snr}

        return DictDataLoader(dataset, batch_size=16)

    def test_ewc_init(self, model):
        """Test EWC initialization."""
        ewc = EWCLearner(model, ewc_lambda=1000)
        assert ewc.ewc_lambda == 1000
        assert not ewc._is_initialized

    def test_compute_fisher(self, model, dataloader):
        """Test Fisher information computation."""
        ewc = EWCLearner(model, fisher_samples=50)
        ewc.compute_fisher(dataloader)

        assert ewc._is_initialized
        assert len(ewc._fisher) > 0
        assert len(ewc._params_snapshot) > 0

    def test_penalty_before_init(self, model):
        """Test penalty is zero before initialization."""
        ewc = EWCLearner(model)
        penalty = ewc.penalty()

        assert penalty.item() == 0.0

    def test_penalty_after_init(self, model, dataloader):
        """Test penalty is non-zero after parameter change."""
        ewc = EWCLearner(model, ewc_lambda=1000)
        ewc.compute_fisher(dataloader)

        # Modify parameters
        for p in model.parameters():
            p.data += 0.1

        penalty = ewc.penalty()
        assert penalty.item() > 0


class TestReplayBuffer:
    """Test replay buffer."""

    def test_buffer_add(self):
        """Test adding samples to buffer."""
        buffer = ReplayBuffer(capacity=100)

        for i in range(50):
            buffer.add({"iq": torch.randn(2, 256), "value": i})

        assert len(buffer) == 50

    def test_buffer_capacity(self):
        """Test buffer respects capacity."""
        buffer = ReplayBuffer(capacity=100, strategy="fifo")

        for i in range(200):
            buffer.add({"value": i})

        assert len(buffer) == 100

    def test_buffer_sample(self):
        """Test sampling from buffer."""
        buffer = ReplayBuffer(capacity=100)

        for i in range(50):
            buffer.add({"value": i})

        samples = buffer.sample(10)
        assert len(samples) == 10

    def test_buffer_sample_larger_than_size(self):
        """Test sampling more than available."""
        buffer = ReplayBuffer(capacity=100)

        for i in range(5):
            buffer.add({"value": i})

        samples = buffer.sample(10)
        assert len(samples) == 5  # Returns available samples

    def test_reservoir_sampling(self):
        """Test reservoir sampling strategy."""
        buffer = ReplayBuffer(capacity=100, strategy="reservoir", seed=42)

        for i in range(1000):
            buffer.add({"value": i})

        assert len(buffer) == 100
        # Values should include both early and late additions
        values = [s["value"] for s in buffer.sample(100)]
        assert min(values) < 100  # Some early values
        assert max(values) > 900  # Some late values

    def test_sample_tensors(self):
        """Test tensor sampling."""
        buffer = ReplayBuffer(capacity=100)

        for i in range(20):
            buffer.add({"iq": torch.randn(2, 256)})

        batch = buffer.sample_tensors(10)
        assert batch["iq"].shape == (10, 2, 256)


class TestStratifiedReplayBuffer:
    """Test stratified replay buffer."""

    def test_stratified_add(self):
        """Test stratified adding."""
        buffer = StratifiedReplayBuffer(capacity=100, num_bins=5)

        for snr in [-3, 5, 15, 25, 35]:
            buffer.add({"snr_db": snr})

        assert len(buffer) == 5

    def test_stratified_sample_balanced(self):
        """Test balanced sampling."""
        buffer = StratifiedReplayBuffer(capacity=500, num_bins=5)

        # Add samples to different bins
        for _ in range(100):
            for snr in [0, 10, 20]:
                buffer.add({"snr_db": snr, "iq": torch.randn(2, 256)})

        samples = buffer.sample(50, balanced=True)
        # Should have samples from multiple bins
        assert len(samples) <= 50
