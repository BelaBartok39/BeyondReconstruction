"""Vector Quantized VAE (VQ-VAE) for RF anomaly detection.

This module implements VQ-VAE with SNR and power conditioning for RF signals.
VQ-VAE uses discrete latent codes instead of continuous Gaussians, which can
provide better clustering for anomaly detection.

References:
    - van den Oord et al. "Neural Discrete Representation Learning" (2017)
    - Kompella et al. "VQ-VAE for RF Signals" (arXiv:2410.18283, 2024)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .autoencoder import ConvBlock, ConvTransposeBlock
from .snr_encoder import (
    _create_conditioning_embedding,
    _combine_conditioning,
    _normalize_conditioning,
    _interpolate_to_length,
)


class VectorQuantizer(nn.Module):
    """Vector Quantization layer with EMA codebook updates.

    Maps continuous encoder outputs to discrete codebook entries using
    nearest neighbor lookup. Supports EMA updates for stable training.

    Args:
        num_embeddings: Size of the codebook (K).
        embedding_dim: Dimension of each codebook entry (D).
        commitment_cost: Weight for commitment loss (beta in the paper).
        decay: EMA decay rate for codebook updates (0.99 typical).
        epsilon: Small constant for numerical stability in EMA.
    """

    def __init__(
        self,
        num_embeddings: int = 512,
        embedding_dim: int = 32,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        epsilon: float = 1e-5,
    ):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        self.decay = decay
        self.epsilon = epsilon

        # Codebook: [num_embeddings, embedding_dim]
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

        # EMA tracking
        self.register_buffer("ema_cluster_size", torch.zeros(num_embeddings))
        self.register_buffer("ema_weight", self.embedding.weight.data.clone())
        self._ema_initialized = False

    def forward(self, z: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Quantize continuous latent vectors.

        Args:
            z: Encoder output [batch, embedding_dim] or [batch, embedding_dim, seq_len].

        Returns:
            Tuple of (quantized, loss, encoding_indices):
            - quantized: Quantized vectors (same shape as z).
            - loss: VQ loss (commitment + codebook loss).
            - encoding_indices: Codebook indices [batch] or [batch, seq_len].
        """
        # Handle both 2D [batch, dim] and 3D [batch, dim, seq] inputs
        input_shape = z.shape
        is_3d = z.dim() == 3

        if is_3d:
            # Reshape: [batch, dim, seq] -> [batch * seq, dim]
            z = z.permute(0, 2, 1).contiguous()
            z_flat = z.view(-1, self.embedding_dim)
        else:
            z_flat = z

        # Compute distances to all codebook entries
        # d(z, e) = ||z||² + ||e||² - 2 * z·e
        distances = (
            z_flat.pow(2).sum(dim=1, keepdim=True)
            + self.embedding.weight.pow(2).sum(dim=1)
            - 2 * z_flat @ self.embedding.weight.t()
        )

        # Find nearest codebook entry
        encoding_indices = distances.argmin(dim=1)
        quantized_flat = self.embedding(encoding_indices)

        # EMA codebook update (only during training)
        if self.training:
            self._update_codebook_ema(z_flat, encoding_indices)

        # Compute losses
        # Commitment loss: encourages encoder to commit to codebook entries
        commitment_loss = F.mse_loss(z_flat, quantized_flat.detach())

        # Codebook loss: encourages codebook to stay close to encoder outputs
        # (only for non-EMA mode, EMA handles this implicitly)
        codebook_loss = F.mse_loss(quantized_flat, z_flat.detach())

        loss = codebook_loss + self.commitment_cost * commitment_loss

        # Straight-through estimator: copy gradients from quantized to z
        quantized_flat = z_flat + (quantized_flat - z_flat).detach()

        # Reshape back to input shape
        if is_3d:
            batch_size, seq_len = input_shape[0], input_shape[2]
            quantized = quantized_flat.view(batch_size, seq_len, self.embedding_dim)
            quantized = quantized.permute(0, 2, 1).contiguous()
            encoding_indices = encoding_indices.view(batch_size, seq_len)
        else:
            quantized = quantized_flat

        return quantized, loss, encoding_indices

    def _update_codebook_ema(self, z_flat: Tensor, encoding_indices: Tensor) -> None:
        """Update codebook using exponential moving average."""
        # One-hot encodings
        encodings = F.one_hot(encoding_indices, self.num_embeddings).float()

        # Cluster sizes (how many inputs map to each codebook entry)
        cluster_size = encodings.sum(dim=0)

        # Sum of inputs for each cluster
        dw = encodings.t() @ z_flat

        # EMA update for cluster sizes
        self.ema_cluster_size.data.mul_(self.decay).add_(
            cluster_size, alpha=1 - self.decay
        )

        # EMA update for codebook weights
        self.ema_weight.data.mul_(self.decay).add_(dw, alpha=1 - self.decay)

        # Normalize by cluster size (Laplace smoothing)
        n = self.ema_cluster_size.sum()
        cluster_size_normalized = (
            (self.ema_cluster_size + self.epsilon)
            / (n + self.num_embeddings * self.epsilon)
            * n
        )

        # Update embedding weights
        self.embedding.weight.data.copy_(
            self.ema_weight / cluster_size_normalized.unsqueeze(1)
        )

    def get_codebook_usage(self) -> dict:
        """Get codebook utilization statistics."""
        usage = (self.ema_cluster_size > 0.1).sum().item()
        return {
            "active_codes": usage,
            "total_codes": self.num_embeddings,
            "utilization": usage / self.num_embeddings,
            "perplexity": torch.exp(
                -(self.ema_cluster_size / self.ema_cluster_size.sum() + 1e-10).log()
                @ (self.ema_cluster_size / self.ema_cluster_size.sum() + 1e-10)
            ).item(),
        }


class VQEncoder(nn.Module):
    """Encoder for VQ-VAE with SNR/power conditioning.

    Outputs continuous latent vectors that will be quantized by VectorQuantizer.
    """

    def __init__(
        self,
        in_channels: int = 2,
        hidden_channels: list[int] | None = None,
        embedding_dim: int = 32,
        snr_embedding_dim: int = 16,
        kernel_size: int = 7,
        use_batch_norm: bool = True,
        dropout: float = 0.1,
        use_power_conditioning: bool = False,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels or [32, 64, 128, 256]
        self.embedding_dim = embedding_dim
        self.snr_embedding_dim = snr_embedding_dim
        self.use_power_conditioning = use_power_conditioning

        # Conditioning embedding
        cond_input_dim = 2 if use_power_conditioning else 1
        self.cond_embed = _create_conditioning_embedding(cond_input_dim, snr_embedding_dim)

        # Encoder conv layers
        channels = [in_channels] + self.hidden_channels
        self.conv_layers = nn.Sequential(*[
            ConvBlock(
                channels[i], channels[i + 1], kernel_size, stride=2,
                use_batch_norm=use_batch_norm,
                dropout=dropout if i < len(self.hidden_channels) - 1 else 0
            )
            for i in range(len(self.hidden_channels))
        ])

        # Projection to embedding dimension (lazy init)
        self._latent_proj = None
        self._combined_size = None

    def forward(self, x: Tensor, snr: Tensor, power: Tensor | None = None) -> Tensor:
        """Encode input to continuous latent vector.

        Args:
            x: Input signal [batch, channels, seq_len].
            snr: Normalized SNR values [batch].
            power: Optional normalized power values [batch].

        Returns:
            Continuous latent representation [batch, embedding_dim].
        """
        cond = _combine_conditioning(snr, power if self.use_power_conditioning else None)
        cond_emb = self.cond_embed(cond)
        h = self.conv_layers(x).flatten(1)
        h_combined = torch.cat([h, cond_emb], dim=1)

        # Lazy initialization
        if self._latent_proj is None:
            self._combined_size = h_combined.size(1)
            self._latent_proj = nn.Linear(self._combined_size, self.embedding_dim).to(x.device)

        return self._latent_proj(h_combined)


class VQDecoder(nn.Module):
    """Decoder for VQ-VAE with SNR/power conditioning."""

    def __init__(
        self,
        embedding_dim: int = 32,
        hidden_channels: list[int] | None = None,
        out_channels: int = 2,
        output_length: int = 1024,
        snr_embedding_dim: int = 16,
        kernel_size: int = 7,
        use_batch_norm: bool = True,
        dropout: float = 0.1,
        use_power_conditioning: bool = False,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels or [256, 128, 64, 32]
        self.output_length = output_length
        self.use_power_conditioning = use_power_conditioning

        # Conditioning embedding
        cond_input_dim = 2 if use_power_conditioning else 1
        self.cond_embed = _create_conditioning_embedding(cond_input_dim, snr_embedding_dim)

        # Calculate initial size
        self._init_length = output_length
        for _ in self.hidden_channels:
            self._init_length = (self._init_length + 1) // 2

        self._init_channels = self.hidden_channels[0]
        self.latent_proj = nn.Sequential(
            nn.Linear(embedding_dim + snr_embedding_dim, self._init_channels * self._init_length),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Decoder conv layers
        channels = self.hidden_channels
        self.conv_layers = nn.Sequential(*[
            ConvTransposeBlock(
                channels[i], channels[i + 1], kernel_size, stride=2,
                use_batch_norm=use_batch_norm,
                dropout=dropout if i < len(channels) - 2 else 0
            )
            for i in range(len(channels) - 1)
        ])

        # Final layer
        self.final = nn.ConvTranspose1d(
            channels[-1], out_channels, kernel_size, stride=2,
            padding=kernel_size // 2, output_padding=1
        )

    def forward(self, z_q: Tensor, snr: Tensor, power: Tensor | None = None) -> Tensor:
        """Decode quantized latent to reconstruction.

        Args:
            z_q: Quantized latent [batch, embedding_dim].
            snr: Normalized SNR values [batch].
            power: Optional normalized power values [batch].

        Returns:
            Reconstruction [batch, channels, seq_len].
        """
        cond = _combine_conditioning(snr, power if self.use_power_conditioning else None)
        cond_emb = self.cond_embed(cond)
        z_combined = torch.cat([z_q, cond_emb], dim=1)
        h = self.latent_proj(z_combined).view(-1, self._init_channels, self._init_length)
        h = self.conv_layers(h)
        x_recon = self.final(h)
        return _interpolate_to_length(x_recon, self.output_length)


class SNRConditionedVQVAE(nn.Module):
    """SNR and Power-Conditioned Vector Quantized VAE.

    Uses discrete codebook entries instead of continuous latent distributions.
    This can provide better clustering for anomaly detection and avoids
    posterior collapse issues common in standard VAEs.

    Key differences from standard VAE:
    - No KL divergence loss (discrete codes don't need regularization)
    - Commitment loss encourages encoder to commit to codebook entries
    - EMA updates for stable codebook learning
    - Codebook perplexity as a training diagnostic

    Example:
        model = SNRConditionedVQVAE(
            embedding_dim=32,
            num_embeddings=512,
            sequence_length=1024,
            use_power_conditioning=True
        )
        x = torch.randn(16, 2, 1024)
        snr = torch.rand(16)
        power = torch.rand(16)
        x_recon, vq_loss, indices = model(x, snr, power)
    """

    def __init__(
        self,
        embedding_dim: int = 32,
        num_embeddings: int = 512,
        sequence_length: int = 1024,
        hidden_channels: list[int] | None = None,
        snr_embedding_dim: int = 16,
        kernel_size: int = 7,
        use_batch_norm: bool = True,
        dropout: float = 0.1,
        commitment_cost: float = 0.25,
        decay: float = 0.99,
        use_power_conditioning: bool = False,
    ):
        super().__init__()
        hidden_channels = hidden_channels or [32, 64, 128, 256]
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.sequence_length = sequence_length
        self.use_power_conditioning = use_power_conditioning
        # For compatibility with detection code
        self.latent_dim = embedding_dim

        self.encoder = VQEncoder(
            in_channels=2,
            hidden_channels=hidden_channels,
            embedding_dim=embedding_dim,
            snr_embedding_dim=snr_embedding_dim,
            kernel_size=kernel_size,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            use_power_conditioning=use_power_conditioning,
        )

        self.quantizer = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost,
            decay=decay,
        )

        self.decoder = VQDecoder(
            embedding_dim=embedding_dim,
            hidden_channels=hidden_channels[::-1],
            out_channels=2,
            output_length=sequence_length,
            snr_embedding_dim=snr_embedding_dim,
            kernel_size=kernel_size,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            use_power_conditioning=use_power_conditioning,
        )

    def forward(
        self, x: Tensor, snr: Tensor, power: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Forward pass with SNR and optional power conditioning.

        Args:
            x: Input signal [batch, 2, seq_len].
            snr: Normalized SNR values [batch].
            power: Optional normalized power values [batch].

        Returns:
            Tuple of (x_recon, vq_loss, encoding_indices).
        """
        z_e = self.encoder(x, snr, power)
        z_q, vq_loss, indices = self.quantizer(z_e)
        x_recon = self.decoder(z_q, snr, power)
        return x_recon, vq_loss, indices

    def encode(self, x: Tensor, snr: Tensor, power: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Encode and quantize input.

        Returns:
            Tuple of (z_quantized, encoding_indices).
        """
        z_e = self.encoder(x, snr, power)
        z_q, _, indices = self.quantizer(z_e)
        return z_q, indices

    def encode_continuous(self, x: Tensor, snr: Tensor, power: Tensor | None = None) -> Tensor:
        """Encode to continuous representation (before quantization).

        Useful for anomaly detection using distance to nearest codebook entry.
        """
        return self.encoder(x, snr, power)

    def decode(self, z_q: Tensor, snr: Tensor, power: Tensor | None = None) -> Tensor:
        """Decode quantized latent."""
        return self.decoder(z_q, snr, power)

    def decode_from_indices(
        self, indices: Tensor, snr: Tensor, power: Tensor | None = None
    ) -> Tensor:
        """Decode from codebook indices."""
        z_q = self.quantizer.embedding(indices)
        return self.decoder(z_q, snr, power)

    def loss(
        self, x: Tensor, x_recon: Tensor, vq_loss: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute total VQ-VAE loss.

        Args:
            x: Original input.
            x_recon: Reconstruction.
            vq_loss: Vector quantization loss from forward pass.

        Returns:
            Tuple of (total_loss, recon_loss, vq_loss).
        """
        recon_loss = F.mse_loss(x_recon, x)
        total_loss = recon_loss + vq_loss
        return total_loss, recon_loss, vq_loss

    def get_reconstruction_error(
        self, x: Tensor, snr: Tensor, power: Tensor | None = None
    ) -> Tensor:
        """Get per-sample reconstruction error."""
        x_recon, _, _ = self(x, snr, power)
        return ((x - x_recon) ** 2).mean(dim=(1, 2))

    def get_quantization_error(
        self, x: Tensor, snr: Tensor, power: Tensor | None = None
    ) -> Tensor:
        """Get per-sample distance to nearest codebook entry.

        This measures how well the input fits the learned codebook.
        High values indicate unusual inputs (potential anomalies).
        """
        z_e = self.encoder(x, snr, power)

        # Distance to nearest codebook entry
        distances = (
            z_e.pow(2).sum(dim=1, keepdim=True)
            + self.quantizer.embedding.weight.pow(2).sum(dim=1)
            - 2 * z_e @ self.quantizer.embedding.weight.t()
        )
        min_distances = distances.min(dim=1)[0]
        return min_distances

    def get_anomaly_score(
        self,
        x: Tensor,
        snr: Tensor,
        power: Tensor | None = None,
        method: str = "hybrid",
    ) -> Tensor:
        """Compute anomaly score.

        Args:
            x: Input signals [batch, 2, seq_len].
            snr: Normalized SNR values [batch].
            power: Optional normalized power values [batch].
            method: Scoring method - "recon", "quantization", or "hybrid".

        Returns:
            Anomaly scores [batch].
        """
        if method == "recon":
            return self.get_reconstruction_error(x, snr, power)
        elif method == "quantization":
            return self.get_quantization_error(x, snr, power)
        else:  # hybrid
            recon_error = self.get_reconstruction_error(x, snr, power)
            quant_error = self.get_quantization_error(x, snr, power)
            # Normalize and combine
            return recon_error + 0.5 * quant_error

    def get_codebook_usage(self) -> dict:
        """Get codebook utilization statistics."""
        return self.quantizer.get_codebook_usage()


def create_vq_model(config) -> SNRConditionedVQVAE:
    """Create VQ-VAE model from configuration."""
    return SNRConditionedVQVAE(
        embedding_dim=getattr(config.model, "latent_dim", 32),
        num_embeddings=getattr(config.model, "num_embeddings", 512),
        sequence_length=config.data.sequence_length,
        hidden_channels=config.model.hidden_channels,
        snr_embedding_dim=config.model.snr_embedding_dim,
        kernel_size=config.model.kernel_size,
        use_batch_norm=config.model.use_batch_norm,
        dropout=config.model.dropout,
        commitment_cost=getattr(config.model, "commitment_cost", 0.25),
        decay=getattr(config.model, "vq_decay", 0.99),
        use_power_conditioning=getattr(config.model, "use_power_conditioning", False),
    )
