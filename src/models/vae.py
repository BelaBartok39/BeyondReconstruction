"""Variational Autoencoder for RF signal anomaly detection."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .autoencoder import ConvBlock, ConvTransposeBlock


class VAEEncoder(nn.Module):
    """VAE encoder that outputs mean and log-variance."""

    def __init__(
        self,
        in_channels: int = 2,
        hidden_channels: list[int] | None = None,
        latent_dim: int = 32,
        kernel_size: int = 7,
        use_batch_norm: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels or [32, 64, 128, 256]
        self.latent_dim = latent_dim

        # Build encoder layers
        channels = [in_channels] + self.hidden_channels
        self.conv_layers = nn.Sequential(*[
            ConvBlock(
                channels[i], channels[i + 1], kernel_size, stride=2,
                use_batch_norm=use_batch_norm,
                dropout=dropout if i < len(self.hidden_channels) - 1 else 0
            )
            for i in range(len(self.hidden_channels))
        ])

        # Projection layers (lazy initialization)
        self._mu_proj = None
        self._logvar_proj = None

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encode input to latent distribution parameters."""
        h = self.conv_layers(x).flatten(1)

        # Lazy initialization
        if self._mu_proj is None:
            flat_size = h.size(1)
            self._mu_proj = nn.Linear(flat_size, self.latent_dim).to(x.device)
            self._logvar_proj = nn.Linear(flat_size, self.latent_dim).to(x.device)

        return self._mu_proj(h), self._logvar_proj(h)


class VAEDecoder(nn.Module):
    """VAE decoder for IQ signal reconstruction."""

    def __init__(
        self,
        latent_dim: int = 32,
        hidden_channels: list[int] | None = None,
        out_channels: int = 2,
        output_length: int = 1024,
        kernel_size: int = 7,
        use_batch_norm: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels or [256, 128, 64, 32]
        self.output_length = output_length

        # Calculate initial size
        self._init_length = output_length
        for _ in self.hidden_channels:
            self._init_length = (self._init_length + 1) // 2

        self._init_channels = self.hidden_channels[0]
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, self._init_channels * self._init_length),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Build decoder layers
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
        self.final_conv = nn.ConvTranspose1d(
            channels[-1], out_channels, kernel_size, stride=2,
            padding=kernel_size // 2, output_padding=1
        )

    def forward(self, z: Tensor) -> Tensor:
        """Decode latent sample to signal."""
        h = self.latent_proj(z).view(-1, self._init_channels, self._init_length)
        h = self.conv_layers(h)
        x_recon = self.final_conv(h)

        # Ensure output length matches
        if x_recon.size(2) != self.output_length:
            x_recon = nn.functional.interpolate(
                x_recon, size=self.output_length, mode="linear", align_corners=False
            )

        return x_recon


def _compute_kl_divergence(mu: Tensor, logvar: Tensor, reduce: bool = True) -> Tensor:
    """Compute KL divergence from standard normal: KL(q(z|x) || N(0, I))."""
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    return kl.mean() if reduce else kl


class ConvVAE(nn.Module):
    """Convolutional Variational Autoencoder for RF signals.

    The VAE adds KL divergence regularization to create a smooth,
    continuous latent space which improves anomaly detection.

    Example:
        model = ConvVAE(latent_dim=32, sequence_length=1024)
        x = torch.randn(16, 2, 1024)
        x_recon, mu, logvar, z = model(x)
        loss = model.loss(x, x_recon, mu, logvar)
    """

    def __init__(
        self,
        latent_dim: int = 32,
        sequence_length: int = 1024,
        hidden_channels: list[int] | None = None,
        kernel_size: int = 7,
        use_batch_norm: bool = True,
        dropout: float = 0.1,
        beta: float = 1.0,
    ):
        super().__init__()
        hidden_channels = hidden_channels or [32, 64, 128, 256]
        self.latent_dim = latent_dim
        self.sequence_length = sequence_length
        self.beta = beta

        self.encoder = VAEEncoder(
            in_channels=2,
            hidden_channels=hidden_channels,
            latent_dim=latent_dim,
            kernel_size=kernel_size,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
        )

        self.decoder = VAEDecoder(
            latent_dim=latent_dim,
            hidden_channels=hidden_channels[::-1],
            out_channels=2,
            output_length=sequence_length,
            kernel_size=kernel_size,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
        )

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterization trick for sampling."""
        if self.training:
            return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return mu  # Use mean during inference

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward pass through VAE."""
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar, z

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encode input to latent distribution."""
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        """Decode latent sample."""
        return self.decoder(z)

    def sample(self, num_samples: int, device: torch.device | None = None) -> Tensor:
        """Sample from the prior distribution."""
        device = device or next(self.parameters()).device
        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)

    def reconstruction_loss(self, x: Tensor, x_recon: Tensor) -> Tensor:
        """Compute reconstruction loss (MSE)."""
        return nn.functional.mse_loss(x_recon, x, reduction="mean")

    def kl_divergence(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Compute KL divergence from standard normal."""
        return _compute_kl_divergence(mu, logvar, reduce=True)

    def loss(self, x: Tensor, x_recon: Tensor, mu: Tensor, logvar: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Compute total VAE loss."""
        recon_loss = self.reconstruction_loss(x, x_recon)
        kl_loss = self.kl_divergence(mu, logvar)
        return recon_loss + self.beta * kl_loss, recon_loss, kl_loss

    def get_reconstruction_error(self, x: Tensor) -> Tensor:
        """Get per-sample reconstruction error."""
        x_recon, _, _, _ = self(x)
        return ((x - x_recon) ** 2).mean(dim=(1, 2))

    def get_anomaly_score(self, x: Tensor, include_kl: bool = True, num_samples: int = 1) -> Tensor:
        """Compute anomaly score combining reconstruction and KL divergence."""
        mu, logvar = self.encoder(x)

        # Compute reconstruction error with optional Monte Carlo sampling
        if num_samples == 1:
            z = self.reparameterize(mu, logvar)
            x_recon = self.decoder(z)
            recon_error = ((x - x_recon) ** 2).mean(dim=(1, 2))
        else:
            # Vectorized Monte Carlo estimate
            batch_size = x.size(0)
            mu_expanded = mu.unsqueeze(1).expand(-1, num_samples, -1).reshape(-1, mu.size(1))
            logvar_expanded = logvar.unsqueeze(1).expand(-1, num_samples, -1).reshape(-1, logvar.size(1))
            z = self.reparameterize(mu_expanded, logvar_expanded)

            x_expanded = x.unsqueeze(1).expand(-1, num_samples, -1, -1).reshape(-1, x.size(1), x.size(2))
            x_recon = self.decoder(z)
            recon_error = ((x_expanded - x_recon) ** 2).mean(dim=(1, 2)).view(batch_size, num_samples).mean(dim=1)

        if include_kl:
            return recon_error + self.beta * _compute_kl_divergence(mu, logvar, reduce=False)
        return recon_error
