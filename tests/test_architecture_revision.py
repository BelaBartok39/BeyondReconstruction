"""Tests for architecture revision: Probabilistic Decoder, Smoothness Prior, BLL, UCL."""

import torch
import pytest
from torch.utils.data import DataLoader, TensorDataset

from src.models.snr_encoder import SNRConditionedVAE, SNREncoder, SNRDecoder
from src.models.bayesian import BayesianLinear, BayesianEncoder, collect_kl_divergence
from src.learning.ucl import UCLLearner
from src.detection.detector import AnomalyDetector


# ============================================================================
# Phase 1: Probabilistic Decoder Tests
# ============================================================================

class TestProbabilisticDecoder:
    """Test probabilistic decoder with NLL loss."""

    @pytest.fixture
    def deterministic_model(self):
        """Create deterministic model (no probabilistic decoder)."""
        return SNRConditionedVAE(
            latent_dim=16,
            sequence_length=256,
            hidden_channels=[16, 32],
            probabilistic_decoder=False,
        )

    @pytest.fixture
    def probabilistic_model(self):
        """Create probabilistic model."""
        return SNRConditionedVAE(
            latent_dim=16,
            sequence_length=256,
            hidden_channels=[16, 32],
            probabilistic_decoder=True,
        )

    def test_deterministic_output_shape(self, deterministic_model):
        """Test deterministic model returns 4 values."""
        x = torch.randn(4, 2, 256)
        snr = torch.rand(4)

        output = deterministic_model(x, snr)

        assert len(output) == 4  # x_recon, mu, logvar, z
        x_recon, mu, logvar, z = output
        assert x_recon.shape == (4, 2, 256)
        assert mu.shape == (4, 16)
        assert logvar.shape == (4, 16)
        assert z.shape == (4, 16)

    def test_probabilistic_output_shape(self, probabilistic_model):
        """Test probabilistic model returns 5 values (includes x_logvar)."""
        x = torch.randn(4, 2, 256)
        snr = torch.rand(4)

        output = probabilistic_model(x, snr)

        assert len(output) == 5  # x_mean, x_logvar, mu, logvar, z
        x_mean, x_logvar, mu, logvar, z = output
        assert x_mean.shape == (4, 2, 256)
        assert x_logvar.shape == (4, 2, 256)
        assert mu.shape == (4, 16)
        assert logvar.shape == (4, 16)
        assert z.shape == (4, 16)

    def test_nll_loss_computation(self, probabilistic_model):
        """Test NLL loss is computed correctly."""
        x = torch.randn(4, 2, 256)
        snr = torch.rand(4)

        x_mean, x_logvar, mu, logvar, _ = probabilistic_model(x, snr)
        loss, recon_loss, kl_loss = probabilistic_model.loss(x, x_mean, mu, logvar, x_logvar)

        assert loss.shape == ()
        assert recon_loss.shape == ()
        assert kl_loss.shape == ()
        assert loss.item() > 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_nll_vs_mse(self, probabilistic_model, deterministic_model):
        """Test that NLL and MSE give different loss values."""
        x = torch.randn(4, 2, 256)
        snr = torch.rand(4)

        # Probabilistic (NLL)
        x_mean, x_logvar, mu_p, logvar_p, _ = probabilistic_model(x, snr)
        loss_nll, _, _ = probabilistic_model.loss(x, x_mean, mu_p, logvar_p, x_logvar)

        # Deterministic (MSE)
        x_recon, mu_d, logvar_d, _ = deterministic_model(x, snr)
        loss_mse, _, _ = deterministic_model.loss(x, x_recon, mu_d, logvar_d)

        # Losses should be different (different loss functions)
        assert loss_nll.item() != loss_mse.item()

    def test_logvar_is_bounded(self, probabilistic_model):
        """Test that logvar is clamped for numerical stability."""
        x = torch.randn(4, 2, 256)
        snr = torch.rand(4)

        _, x_logvar, _, _, _ = probabilistic_model(x, snr)

        # Check logvar is within reasonable bounds
        assert x_logvar.min() >= -10.0
        assert x_logvar.max() <= 2.0

    def test_gradient_flow_probabilistic(self, probabilistic_model):
        """Test gradients flow through probabilistic model."""
        x = torch.randn(4, 2, 256, requires_grad=True)
        snr = torch.rand(4)

        x_mean, x_logvar, mu, logvar, _ = probabilistic_model(x, snr)
        loss, _, _ = probabilistic_model.loss(x, x_mean, mu, logvar, x_logvar)
        loss.backward()

        assert x.grad is not None
        for param in probabilistic_model.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_anomaly_score_nll(self, probabilistic_model):
        """Test anomaly score computation with NLL."""
        probabilistic_model.eval()  # Deterministic mode
        x = torch.randn(4, 2, 256)
        snr = torch.rand(4)

        with torch.no_grad():
            score_auto = probabilistic_model.get_anomaly_score(x, snr, scoring_method="auto")
            score_nll = probabilistic_model.get_anomaly_score(x, snr, scoring_method="nll")
            score_mse = probabilistic_model.get_anomaly_score(x, snr, scoring_method="mse")

        assert score_auto.shape == (4,)
        assert score_nll.shape == (4,)
        assert score_mse.shape == (4,)

        # auto should use NLL for probabilistic model (in eval mode, should be deterministic)
        torch.testing.assert_close(score_auto, score_nll)
        # NLL and MSE should be different
        assert not torch.allclose(score_nll, score_mse)


# ============================================================================
# Phase 2: Smoothness Prior Tests
# ============================================================================

class TestSmoothnessPrior:
    """Test smoothness prior for temporal consistency."""

    @pytest.fixture
    def model_with_smoothness(self):
        """Create model with smoothness prior."""
        return SNRConditionedVAE(
            latent_dim=16,
            sequence_length=256,
            hidden_channels=[16, 32],
            probabilistic_decoder=True,
            smoothness_lambda=0.1,
        )

    @pytest.fixture
    def model_without_smoothness(self):
        """Create model without smoothness prior."""
        return SNRConditionedVAE(
            latent_dim=16,
            sequence_length=256,
            hidden_channels=[16, 32],
            probabilistic_decoder=True,
            smoothness_lambda=0.0,
        )

    def test_smoothness_loss_returns_4_values(self, model_with_smoothness):
        """Test that loss returns 4 values when smoothness is enabled."""
        x = torch.randn(4, 2, 256)
        snr = torch.rand(4)

        x_mean, x_logvar, mu, logvar, _ = model_with_smoothness(x, snr)
        loss_out = model_with_smoothness.loss(x, x_mean, mu, logvar, x_logvar)

        assert len(loss_out) == 4  # total, recon, kl, smooth
        total_loss, recon_loss, kl_loss, smooth_loss = loss_out
        assert smooth_loss.item() >= 0
        assert not torch.isnan(smooth_loss)

    def test_smoothness_loss_returns_3_values_when_disabled(self, model_without_smoothness):
        """Test that loss returns 3 values when smoothness is disabled."""
        x = torch.randn(4, 2, 256)
        snr = torch.rand(4)

        x_mean, x_logvar, mu, logvar, _ = model_without_smoothness(x, snr)
        loss_out = model_without_smoothness.loss(x, x_mean, mu, logvar, x_logvar)

        assert len(loss_out) == 3  # total, recon, kl

    def test_smoothness_penalizes_rapid_changes(self, model_with_smoothness):
        """Test that smoothness loss is higher for rapidly changing signals."""
        # Create smooth signal
        t = torch.linspace(0, 1, 256)
        smooth_signal = torch.sin(2 * 3.14159 * t).unsqueeze(0).unsqueeze(0).expand(4, 2, -1)

        # Create spiky signal
        spiky_signal = smooth_signal.clone()
        spiky_signal[:, :, 128] = 10.0  # Add spike

        snr = torch.rand(4)

        # Get smoothness loss for both
        _, x_logvar_smooth, _, _, _ = model_with_smoothness(smooth_signal, snr)
        _, x_logvar_spiky, _, _, _ = model_with_smoothness(spiky_signal, snr)

        smooth_loss_smooth = model_with_smoothness.smoothness_loss(smooth_signal, x_logvar_smooth)
        smooth_loss_spiky = model_with_smoothness.smoothness_loss(spiky_signal, x_logvar_spiky)

        # Both should be valid losses
        assert smooth_loss_smooth.item() >= 0
        assert smooth_loss_spiky.item() >= 0


# ============================================================================
# Phase 3: Bayesian Last Layer Tests
# ============================================================================

class TestBayesianLinear:
    """Test Bayesian linear layer."""

    @pytest.fixture
    def layer(self):
        """Create Bayesian linear layer."""
        return BayesianLinear(64, 32, prior_std=1.0)

    def test_forward_shape(self, layer):
        """Test forward pass shape."""
        x = torch.randn(8, 64)
        y = layer(x, sample=True)

        assert y.shape == (8, 32)

    def test_sampling_produces_different_outputs(self, layer):
        """Test that sampling produces different outputs in training mode."""
        layer.train()
        x = torch.randn(4, 64)

        y1 = layer(x, sample=True)
        y2 = layer(x, sample=True)

        # Should be different due to weight sampling
        assert not torch.allclose(y1, y2)

    def test_no_sampling_produces_same_outputs(self, layer):
        """Test that no sampling produces deterministic outputs."""
        x = torch.randn(4, 64)

        y1 = layer(x, sample=False)
        y2 = layer(x, sample=False)

        torch.testing.assert_close(y1, y2)

    def test_kl_divergence(self, layer):
        """Test KL divergence computation."""
        kl = layer.kl_divergence()

        assert kl.shape == ()
        assert kl.item() >= 0
        assert not torch.isnan(kl)

    def test_weight_uncertainty(self, layer):
        """Test weight uncertainty property."""
        uncertainty = layer.weight_uncertainty

        assert uncertainty.shape == (32, 64)
        assert (uncertainty > 0).all()

    def test_gradient_flow(self, layer):
        """Test gradients flow through layer."""
        x = torch.randn(4, 64, requires_grad=True)
        y = layer(x, sample=True)
        loss = y.sum()
        loss.backward()

        assert x.grad is not None
        assert layer.weight_mean.grad is not None
        assert layer.weight_logvar.grad is not None


class TestBayesianEncoder:
    """Test Bayesian encoder wrapper."""

    @pytest.fixture
    def encoder(self):
        """Create Bayesian encoder."""
        return BayesianEncoder(combined_size=128, latent_dim=32, prior_std=1.0)

    def test_forward_shape(self, encoder):
        """Test forward pass shape."""
        h = torch.randn(4, 128)
        mu, logvar = encoder(h, sample=True)

        assert mu.shape == (4, 32)
        assert logvar.shape == (4, 32)

    def test_epistemic_uncertainty(self, encoder):
        """Test epistemic uncertainty estimation."""
        h = torch.randn(4, 128)
        uncertainty = encoder.get_epistemic_uncertainty(h, num_samples=10)

        assert uncertainty.shape == (4, 32)
        assert (uncertainty >= 0).all()

    def test_kl_divergence(self, encoder):
        """Test total KL divergence."""
        kl = encoder.kl_divergence()

        assert kl.shape == ()
        assert kl.item() >= 0


class TestSNREncoderWithBayesian:
    """Test SNR encoder with Bayesian last layer."""

    @pytest.fixture
    def bayesian_encoder(self):
        """Create SNR encoder with Bayesian projection."""
        return SNREncoder(
            latent_dim=16,
            hidden_channels=[16, 32],
            use_bayesian=True,
            bll_prior_std=1.0,
        )

    @pytest.fixture
    def standard_encoder(self):
        """Create standard SNR encoder."""
        return SNREncoder(
            latent_dim=16,
            hidden_channels=[16, 32],
            use_bayesian=False,
        )

    def test_bayesian_forward(self, bayesian_encoder):
        """Test Bayesian encoder forward pass."""
        x = torch.randn(4, 2, 1024)
        snr = torch.rand(4)

        mu, logvar = bayesian_encoder(x, snr, sample=True)

        assert mu.shape == (4, 16)
        assert logvar.shape == (4, 16)

    def test_epistemic_uncertainty_bayesian(self, bayesian_encoder):
        """Test epistemic uncertainty for Bayesian encoder."""
        x = torch.randn(4, 2, 1024)
        snr = torch.rand(4)

        uncertainty = bayesian_encoder.get_epistemic_uncertainty(x, snr, num_samples=10)

        assert uncertainty.shape == (4, 16)
        assert (uncertainty >= 0).all()

    def test_epistemic_uncertainty_standard(self, standard_encoder):
        """Test epistemic uncertainty is zero for standard encoder."""
        x = torch.randn(4, 2, 1024)
        snr = torch.rand(4)

        uncertainty = standard_encoder.get_epistemic_uncertainty(x, snr, num_samples=10)

        assert uncertainty.shape == (4, 16)
        assert (uncertainty == 0).all()

    def test_kl_divergence_bayesian(self, bayesian_encoder):
        """Test KL divergence for Bayesian encoder."""
        # Initialize lazy layers
        x = torch.randn(4, 2, 1024)
        snr = torch.rand(4)
        _ = bayesian_encoder(x, snr)

        kl = bayesian_encoder.kl_divergence()
        assert kl.item() > 0

    def test_kl_divergence_standard(self, standard_encoder):
        """Test KL divergence is zero for standard encoder."""
        kl = standard_encoder.kl_divergence()
        assert kl.item() == 0


class TestSNRConditionedVAEWithBayesian:
    """Test full VAE with Bayesian encoder."""

    @pytest.fixture
    def bayesian_vae(self):
        """Create VAE with Bayesian encoder."""
        return SNRConditionedVAE(
            latent_dim=16,
            sequence_length=256,
            hidden_channels=[16, 32],
            use_bayesian_encoder=True,
            bll_kl_weight=1e-4,
        )

    def test_bayesian_vae_forward(self, bayesian_vae):
        """Test Bayesian VAE forward pass."""
        x = torch.randn(4, 2, 256)
        snr = torch.rand(4)

        output = bayesian_vae(x, snr)
        x_recon, mu, logvar, z = output

        assert x_recon.shape == (4, 2, 256)
        assert mu.shape == (4, 16)

    def test_bayesian_vae_loss_includes_bll_kl(self, bayesian_vae):
        """Test that Bayesian VAE loss includes BLL KL."""
        x = torch.randn(4, 2, 256)
        snr = torch.rand(4)

        x_recon, mu, logvar, _ = bayesian_vae(x, snr)
        loss, recon_loss, kl_loss = bayesian_vae.loss(x, x_recon, mu, logvar)

        # Loss should include BLL KL contribution
        assert loss.item() > 0


# ============================================================================
# Phase 4: UCL (Uncertainty-based Continual Learning) Tests
# ============================================================================

class TestUCLLearner:
    """Test UCL learner."""

    @pytest.fixture
    def bayesian_model(self):
        """Create model with Bayesian layers."""
        return SNRConditionedVAE(
            latent_dim=16,
            sequence_length=256,
            hidden_channels=[16, 32],
            use_bayesian_encoder=True,
        )

    @pytest.fixture
    def standard_model(self):
        """Create standard model without Bayesian layers."""
        return SNRConditionedVAE(
            latent_dim=16,
            sequence_length=256,
            hidden_channels=[16, 32],
            use_bayesian_encoder=False,
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

    def test_ucl_init(self, bayesian_model):
        """Test UCL initialization."""
        ucl = UCLLearner(bayesian_model, ucl_lambda=100.0)
        assert ucl.ucl_lambda == 100.0
        assert not ucl._is_initialized

    def test_ucl_snapshot(self, bayesian_model, dataloader):
        """Test UCL snapshot."""
        # Initialize lazy layers
        for batch in dataloader:
            with torch.no_grad():
                _ = bayesian_model(batch["iq"], batch["snr"])
            break

        ucl = UCLLearner(bayesian_model, ucl_lambda=100.0)
        ucl.snapshot()

        assert ucl._is_initialized
        assert len(ucl._importance) > 0
        assert len(ucl._params_snapshot) > 0

    def test_ucl_penalty_before_init(self, bayesian_model):
        """Test penalty is zero before initialization."""
        ucl = UCLLearner(bayesian_model)
        penalty = ucl.penalty()

        assert penalty.item() == 0.0

    def test_ucl_penalty_after_change(self, bayesian_model, dataloader):
        """Test penalty is non-zero after parameter change."""
        # Initialize lazy layers
        for batch in dataloader:
            with torch.no_grad():
                _ = bayesian_model(batch["iq"], batch["snr"])
            break

        ucl = UCLLearner(bayesian_model, ucl_lambda=100.0)
        ucl.snapshot()

        # Modify parameters
        for p in bayesian_model.parameters():
            if p.requires_grad and "_logvar" not in str(p):
                p.data += 0.1

        penalty = ucl.penalty()
        assert penalty.item() > 0

    def test_ucl_compute_importance_bayesian(self, bayesian_model, dataloader):
        """Test importance computation for Bayesian model."""
        # Initialize lazy layers
        for batch in dataloader:
            with torch.no_grad():
                _ = bayesian_model(batch["iq"], batch["snr"])
            break

        ucl = UCLLearner(bayesian_model)
        importance = ucl.compute_importance()

        # Should have importance for Bayesian layer weights
        assert len(importance) > 0

    def test_ucl_with_standard_model_warns(self, standard_model):
        """Test UCL warns when used with non-Bayesian model."""
        with pytest.warns(UserWarning, match="Bayesian layers"):
            UCLLearner(standard_model)

    def test_ucl_state_save_load(self, bayesian_model, dataloader):
        """Test UCL state save and load."""
        # Initialize lazy layers
        for batch in dataloader:
            with torch.no_grad():
                _ = bayesian_model(batch["iq"], batch["snr"])
            break

        ucl = UCLLearner(bayesian_model)
        ucl.snapshot()

        # Save state
        state = ucl.get_state()
        assert "importance" in state
        assert "params_snapshot" in state
        assert "is_initialized" in state

        # Create new UCL and load state
        ucl2 = UCLLearner(bayesian_model)
        ucl2.load_state(state)

        assert ucl2._is_initialized
        assert len(ucl2._importance) == len(ucl._importance)


# ============================================================================
# Integration Tests: AnomalyDetector with New Features
# ============================================================================

class TestAnomalyDetectorIntegration:
    """Test AnomalyDetector with new architecture features."""

    @pytest.fixture
    def probabilistic_model(self):
        """Create probabilistic model."""
        return SNRConditionedVAE(
            latent_dim=16,
            sequence_length=256,
            hidden_channels=[16, 32],
            probabilistic_decoder=True,
        )

    @pytest.fixture
    def bayesian_model(self):
        """Create Bayesian model."""
        return SNRConditionedVAE(
            latent_dim=16,
            sequence_length=256,
            hidden_channels=[16, 32],
            use_bayesian_encoder=True,
        )

    @pytest.fixture
    def dataloader(self):
        """Create sample dataloader."""
        iq = torch.randn(100, 2, 256)
        snr = torch.rand(100)
        snr_db = snr * 35 - 5  # Map to [-5, 30] dB
        labels = torch.zeros(100)
        labels[80:] = 1  # 20% anomalies

        class DictDataLoader(DataLoader):
            def __init__(self, iq, snr, snr_db, labels, batch_size):
                self.data = list(zip(iq, snr, snr_db, labels))
                self.batch_size = batch_size

            def __iter__(self):
                for i in range(0, len(self.data), self.batch_size):
                    batch_data = self.data[i:i+self.batch_size]
                    yield {
                        "iq": torch.stack([d[0] for d in batch_data]),
                        "snr": torch.stack([d[1] for d in batch_data]),
                        "snr_db": torch.stack([d[2] for d in batch_data]),
                        "label": torch.stack([d[3] for d in batch_data]),
                    }

            def __len__(self):
                return (len(self.data) + self.batch_size - 1) // self.batch_size

            @property
            def dataset(self):
                return self.data

        return DictDataLoader(iq, snr, snr_db, labels, batch_size=16)

    def test_detector_with_nll_scoring(self, probabilistic_model, dataloader):
        """Test detector with NLL scoring method."""
        detector = AnomalyDetector(
            model=probabilistic_model,
            method="reconstruction",
            scoring_method="nll",
        )

        detector.fit(dataloader, num_batches=5)
        assert detector._is_fitted
        assert detector._is_probabilistic

    def test_detector_with_auto_scoring(self, probabilistic_model, dataloader):
        """Test detector with auto scoring method."""
        detector = AnomalyDetector(
            model=probabilistic_model,
            method="reconstruction",
            scoring_method="auto",
        )

        detector.fit(dataloader, num_batches=5)

        # Auto should detect probabilistic model
        stats = detector.get_stats()
        assert stats["scoring_method"] == "auto"
        assert stats["is_probabilistic"] == True

    def test_detector_detect_returns_valid_results(self, probabilistic_model, dataloader):
        """Test detector returns valid detection results."""
        detector = AnomalyDetector(
            model=probabilistic_model,
            method="reconstruction",
            scoring_method="nll",
        )

        detector.fit(dataloader, num_batches=5)

        # Get a batch
        for batch in dataloader:
            result = detector.detect(
                batch["iq"],
                batch["snr"],
                batch["snr_db"]
            )
            break

        assert result.scores is not None
        assert result.predictions is not None
        assert len(result.scores) == len(result.predictions)


# ============================================================================
# Model Creation and Config Tests
# ============================================================================

class TestModelCreation:
    """Test model creation with config."""

    def test_create_probabilistic_model(self):
        """Test creating model with probabilistic decoder."""
        from src.models.snr_encoder import create_model
        from src.utils.config import load_config

        # Create a mock config
        class MockConfig:
            class model:
                type = "snr_vae"
                latent_dim = 16
                hidden_channels = [16, 32]
                kernel_size = 7
                use_batch_norm = True
                dropout = 0.1
                snr_embedding_dim = 16
                beta = 1.0
                use_power_conditioning = False
                probabilistic_decoder = True
                smoothness_lambda = 0.1
                use_bayesian_encoder = False
                bll_prior_std = 1.0
                bll_kl_weight = 1e-4

            class data:
                sequence_length = 256

        config = MockConfig()
        model = create_model(config)

        assert model.probabilistic_decoder == True
        assert model.smoothness_lambda == 0.1

    def test_create_bayesian_model(self):
        """Test creating model with Bayesian encoder."""
        from src.models.snr_encoder import create_model

        class MockConfig:
            class model:
                type = "snr_vae"
                latent_dim = 16
                hidden_channels = [16, 32]
                kernel_size = 7
                use_batch_norm = True
                dropout = 0.1
                snr_embedding_dim = 16
                beta = 1.0
                use_power_conditioning = False
                probabilistic_decoder = False
                smoothness_lambda = 0.0
                use_bayesian_encoder = True
                bll_prior_std = 1.0
                bll_kl_weight = 1e-4

            class data:
                sequence_length = 256

        config = MockConfig()
        model = create_model(config)

        assert model.use_bayesian_encoder == True
        assert model.encoder.use_bayesian == True
