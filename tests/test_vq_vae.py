"""Tests for VQ-VAE model."""

import pytest
import torch
import numpy as np

from src.models.vq_vae import (
    VectorQuantizer,
    VQEncoder,
    VQDecoder,
    SNRConditionedVQVAE,
    create_vq_model,
)


class TestVectorQuantizer:
    """Tests for the VectorQuantizer layer."""

    def test_init(self):
        """Test VectorQuantizer initialization."""
        vq = VectorQuantizer(num_embeddings=512, embedding_dim=32)
        assert vq.num_embeddings == 512
        assert vq.embedding_dim == 32
        assert vq.embedding.weight.shape == (512, 32)

    def test_forward_2d(self):
        """Test quantization with 2D input [batch, dim]."""
        vq = VectorQuantizer(num_embeddings=64, embedding_dim=16)
        z = torch.randn(8, 16)

        z_q, loss, indices = vq(z)

        assert z_q.shape == z.shape
        assert loss.shape == ()
        assert indices.shape == (8,)
        assert indices.max() < 64
        assert indices.min() >= 0

    def test_forward_3d(self):
        """Test quantization with 3D input [batch, dim, seq]."""
        vq = VectorQuantizer(num_embeddings=64, embedding_dim=16)
        z = torch.randn(8, 16, 32)

        z_q, loss, indices = vq(z)

        assert z_q.shape == z.shape
        assert loss.shape == ()
        assert indices.shape == (8, 32)

    def test_straight_through_gradient(self):
        """Test that gradients flow through quantization."""
        vq = VectorQuantizer(num_embeddings=64, embedding_dim=16)
        z = torch.randn(8, 16, requires_grad=True)

        z_q, loss, _ = vq(z)
        (z_q.sum() + loss).backward()

        assert z.grad is not None
        assert not torch.isnan(z.grad).any()

    def test_ema_update(self):
        """Test that EMA updates codebook during training."""
        vq = VectorQuantizer(num_embeddings=64, embedding_dim=16, decay=0.99)
        vq.train()

        initial_weight = vq.embedding.weight.clone()

        # Run several forward passes
        for _ in range(10):
            z = torch.randn(32, 16)
            vq(z)

        # Weights should have changed due to EMA updates
        assert not torch.allclose(initial_weight, vq.embedding.weight)

    def test_codebook_usage(self):
        """Test codebook usage statistics."""
        vq = VectorQuantizer(num_embeddings=32, embedding_dim=8)
        vq.train()

        # Run forward passes to update EMA statistics
        for _ in range(100):
            z = torch.randn(64, 8)
            vq(z)

        stats = vq.get_codebook_usage()
        assert "active_codes" in stats
        assert "utilization" in stats
        assert "perplexity" in stats
        assert 0 <= stats["utilization"] <= 1


class TestVQEncoder:
    """Tests for the VQ-VAE encoder."""

    def test_forward(self):
        """Test encoder forward pass."""
        encoder = VQEncoder(
            embedding_dim=32,
            use_power_conditioning=True,
        )
        x = torch.randn(4, 2, 1024)
        snr = torch.rand(4)
        power = torch.rand(4)

        z = encoder(x, snr, power)

        assert z.shape == (4, 32)

    def test_without_power(self):
        """Test encoder without power conditioning."""
        encoder = VQEncoder(
            embedding_dim=32,
            use_power_conditioning=False,
        )
        x = torch.randn(4, 2, 1024)
        snr = torch.rand(4)

        z = encoder(x, snr)

        assert z.shape == (4, 32)


class TestVQDecoder:
    """Tests for the VQ-VAE decoder."""

    def test_forward(self):
        """Test decoder forward pass."""
        decoder = VQDecoder(
            embedding_dim=32,
            output_length=1024,
            use_power_conditioning=True,
        )
        z_q = torch.randn(4, 32)
        snr = torch.rand(4)
        power = torch.rand(4)

        x_recon = decoder(z_q, snr, power)

        assert x_recon.shape == (4, 2, 1024)


class TestSNRConditionedVQVAE:
    """Tests for the full VQ-VAE model."""

    @pytest.fixture
    def model(self):
        """Create a VQ-VAE model for testing."""
        return SNRConditionedVQVAE(
            embedding_dim=32,
            num_embeddings=64,
            sequence_length=1024,
            hidden_channels=[16, 32, 64, 128],
            use_power_conditioning=True,
        )

    def test_forward(self, model):
        """Test full forward pass."""
        x = torch.randn(4, 2, 1024)
        snr = torch.rand(4)
        power = torch.rand(4)

        x_recon, vq_loss, indices = model(x, snr, power)

        assert x_recon.shape == x.shape
        assert vq_loss.shape == ()
        assert indices.shape == (4,)

    def test_encode(self, model):
        """Test encoding."""
        x = torch.randn(4, 2, 1024)
        snr = torch.rand(4)
        power = torch.rand(4)

        z_q, indices = model.encode(x, snr, power)

        assert z_q.shape == (4, 32)
        assert indices.shape == (4,)

    def test_encode_continuous(self, model):
        """Test continuous encoding (before quantization)."""
        x = torch.randn(4, 2, 1024)
        snr = torch.rand(4)
        power = torch.rand(4)

        z_e = model.encode_continuous(x, snr, power)

        assert z_e.shape == (4, 32)

    def test_decode_from_indices(self, model):
        """Test decoding from codebook indices."""
        indices = torch.randint(0, 64, (4,))
        snr = torch.rand(4)
        power = torch.rand(4)

        x_recon = model.decode_from_indices(indices, snr, power)

        assert x_recon.shape == (4, 2, 1024)

    def test_loss(self, model):
        """Test loss computation."""
        x = torch.randn(4, 2, 1024)
        snr = torch.rand(4)
        power = torch.rand(4)

        x_recon, vq_loss, _ = model(x, snr, power)
        total_loss, recon_loss, vq_loss_out = model.loss(x, x_recon, vq_loss)

        assert total_loss.shape == ()
        assert recon_loss.shape == ()
        assert vq_loss_out.shape == ()
        assert total_loss == recon_loss + vq_loss_out

    def test_reconstruction_error(self, model):
        """Test reconstruction error computation."""
        x = torch.randn(4, 2, 1024)
        snr = torch.rand(4)
        power = torch.rand(4)

        recon_error = model.get_reconstruction_error(x, snr, power)

        assert recon_error.shape == (4,)
        assert (recon_error >= 0).all()

    def test_quantization_error(self, model):
        """Test quantization error computation."""
        x = torch.randn(4, 2, 1024)
        snr = torch.rand(4)
        power = torch.rand(4)

        quant_error = model.get_quantization_error(x, snr, power)

        assert quant_error.shape == (4,)
        assert (quant_error >= 0).all()

    def test_anomaly_score_methods(self, model):
        """Test different anomaly scoring methods."""
        x = torch.randn(4, 2, 1024)
        snr = torch.rand(4)
        power = torch.rand(4)

        for method in ["recon", "quantization", "hybrid"]:
            scores = model.get_anomaly_score(x, snr, power, method=method)
            assert scores.shape == (4,)

    def test_training_step(self, model):
        """Test a full training step."""
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        x = torch.randn(8, 2, 1024)
        snr = torch.rand(8)
        power = torch.rand(8)

        optimizer.zero_grad()
        x_recon, vq_loss, _ = model(x, snr, power)
        total_loss, _, _ = model.loss(x, x_recon, vq_loss)
        total_loss.backward()
        optimizer.step()

        # Check no NaN gradients
        for param in model.parameters():
            if param.grad is not None:
                assert not torch.isnan(param.grad).any()

    def test_latent_dim_compatibility(self, model):
        """Test that latent_dim attribute is set for detector compatibility."""
        assert hasattr(model, "latent_dim")
        assert model.latent_dim == 32


class TestCreateVQModel:
    """Tests for the model factory function."""

    def test_create_vq_model(self):
        """Test creating VQ-VAE from config."""
        class MockConfig:
            class model:
                latent_dim = 32
                num_embeddings = 256
                hidden_channels = [16, 32, 64, 128]
                snr_embedding_dim = 16
                kernel_size = 7
                use_batch_norm = True
                dropout = 0.1
                commitment_cost = 0.25
                vq_decay = 0.99
                use_power_conditioning = True

            class data:
                sequence_length = 1024

        model = create_vq_model(MockConfig)

        assert isinstance(model, SNRConditionedVQVAE)
        assert model.embedding_dim == 32
        assert model.num_embeddings == 256


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
