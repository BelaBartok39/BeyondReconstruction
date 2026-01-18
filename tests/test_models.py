"""Tests for model architectures."""

import torch
import pytest

from src.models.autoencoder import ConvAutoencoder
from src.models.vae import ConvVAE
from src.models.snr_encoder import SNRConditionedVAE


class TestConvAutoencoder:
    """Test convolutional autoencoder."""

    @pytest.fixture
    def model(self):
        """Create model fixture."""
        return ConvAutoencoder(
            latent_dim=32,
            sequence_length=1024,
            hidden_channels=[32, 64],
        )

    def test_forward_shape(self, model):
        """Test forward pass output shapes."""
        x = torch.randn(8, 2, 1024)
        x_recon, z = model(x)

        assert x_recon.shape == (8, 2, 1024)
        assert z.shape == (8, 32)

    def test_encode(self, model):
        """Test encoding."""
        x = torch.randn(4, 2, 1024)
        z = model.encode(x)

        assert z.shape == (4, 32)

    def test_decode(self, model):
        """Test decoding."""
        z = torch.randn(4, 32)
        x_recon = model.decode(z)

        assert x_recon.shape == (4, 2, 1024)

    def test_reconstruction_loss(self, model):
        """Test reconstruction loss computation."""
        x = torch.randn(8, 2, 1024)
        x_recon, _ = model(x)
        loss = model.reconstruction_loss(x, x_recon)

        assert loss.shape == ()
        assert loss.item() >= 0

    def test_reconstruction_error_per_sample(self, model):
        """Test per-sample reconstruction error."""
        x = torch.randn(8, 2, 1024)
        error = model.get_reconstruction_error(x)

        assert error.shape == (8,)
        assert (error >= 0).all()


class TestConvVAE:
    """Test variational autoencoder."""

    @pytest.fixture
    def model(self):
        """Create model fixture."""
        return ConvVAE(
            latent_dim=32,
            sequence_length=1024,
            hidden_channels=[32, 64],
            beta=1.0,
        )

    def test_forward_shape(self, model):
        """Test forward pass output shapes."""
        x = torch.randn(8, 2, 1024)
        x_recon, mu, logvar, z = model(x)

        assert x_recon.shape == (8, 2, 1024)
        assert mu.shape == (8, 32)
        assert logvar.shape == (8, 32)
        assert z.shape == (8, 32)

    def test_reparameterize_training(self, model):
        """Test reparameterization adds noise during training."""
        model.train()
        mu = torch.zeros(4, 32)
        logvar = torch.zeros(4, 32)

        z1 = model.reparameterize(mu, logvar)
        z2 = model.reparameterize(mu, logvar)

        # Should be different due to random sampling
        assert not torch.allclose(z1, z2)

    def test_reparameterize_eval(self, model):
        """Test reparameterization returns mean during eval."""
        model.eval()
        mu = torch.randn(4, 32)
        logvar = torch.zeros(4, 32)

        z = model.reparameterize(mu, logvar)

        torch.testing.assert_close(z, mu)

    def test_loss_components(self, model):
        """Test loss computation returns all components."""
        x = torch.randn(8, 2, 1024)
        x_recon, mu, logvar, _ = model(x)

        total_loss, recon_loss, kl_loss = model.loss(x, x_recon, mu, logvar)

        assert total_loss.shape == ()
        assert recon_loss.shape == ()
        assert kl_loss.shape == ()
        assert total_loss.item() >= 0
        assert recon_loss.item() >= 0
        assert kl_loss.item() >= 0

    def test_sample(self, model):
        """Test sampling from prior."""
        samples = model.sample(num_samples=4)

        assert samples.shape == (4, 2, 1024)


class TestSNRConditionedVAE:
    """Test SNR-conditioned VAE."""

    @pytest.fixture
    def model(self):
        """Create model fixture."""
        return SNRConditionedVAE(
            latent_dim=32,
            sequence_length=1024,
            hidden_channels=[32, 64],
            snr_embedding_dim=16,
        )

    def test_forward_shape(self, model):
        """Test forward pass with SNR conditioning."""
        x = torch.randn(8, 2, 1024)
        snr = torch.rand(8)

        x_recon, mu, logvar, z = model(x, snr)

        assert x_recon.shape == (8, 2, 1024)
        assert mu.shape == (8, 32)
        assert logvar.shape == (8, 32)
        assert z.shape == (8, 32)

    def test_snr_shape_handling(self, model):
        """Test SNR can be 1D or 2D."""
        x = torch.randn(4, 2, 1024)

        # 1D SNR
        snr_1d = torch.rand(4)
        x_recon_1d, _, _, _ = model(x, snr_1d)

        # 2D SNR
        snr_2d = torch.rand(4, 1)
        x_recon_2d, _, _, _ = model(x, snr_2d)

        assert x_recon_1d.shape == x_recon_2d.shape

    def test_snr_affects_output(self, model):
        """Test different SNR produces different outputs."""
        x = torch.randn(4, 2, 1024)

        snr_low = torch.zeros(4)
        snr_high = torch.ones(4)

        model.eval()
        with torch.no_grad():
            x_recon_low, _, _, _ = model(x, snr_low)
            x_recon_high, _, _, _ = model(x, snr_high)

        # Outputs should be different
        assert not torch.allclose(x_recon_low, x_recon_high)

    def test_gradient_flow(self, model):
        """Test gradients flow through model."""
        x = torch.randn(4, 2, 1024, requires_grad=True)
        snr = torch.rand(4)

        x_recon, mu, logvar, _ = model(x, snr)
        loss, _, _ = model.loss(x, x_recon, mu, logvar)

        loss.backward()

        assert x.grad is not None
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None


class TestModelDeviceHandling:
    """Test models work on different devices."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_forward(self):
        """Test forward pass on CUDA."""
        model = ConvVAE(latent_dim=16, sequence_length=256).cuda()
        x = torch.randn(2, 2, 256).cuda()

        x_recon, mu, logvar, z = model(x)

        assert x_recon.device.type == "cuda"
        assert mu.device.type == "cuda"
