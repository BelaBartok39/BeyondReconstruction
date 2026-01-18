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
        """Initialize VAE encoder.

        Args:
            in_channels: Number of input channels (2 for IQ).
            hidden_channels: List of hidden channel sizes.
            latent_dim: Dimension of latent space.
            kernel_size: Convolution kernel size.
            use_batch_norm: Whether to use batch normalization.
            dropout: Dropout probability.
        """
        super().__init__()

        if hidden_channels is None:
            hidden_channels = [32, 64, 128, 256]

        self.hidden_channels = hidden_channels
        self.latent_dim = latent_dim

        # Build encoder layers
        layers = []
        prev_channels = in_channels

        for i, out_channels in enumerate(hidden_channels):
            layers.append(
                ConvBlock(
                    prev_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=2,
                    use_batch_norm=use_batch_norm,
                    dropout=dropout if i < len(hidden_channels) - 1 else 0,
                )
            )
            prev_channels = out_channels

        self.conv_layers = nn.Sequential(*layers)

        # Projection layers for mean and log-variance
        self._mu_proj = None
        self._logvar_proj = None
        self._flat_size = None

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encode input to latent distribution parameters.

        Args:
            x: Input tensor [batch, 2, seq_len].

        Returns:
            Tuple of (mean, log_variance) each [batch, latent_dim].
        """
        h = self.conv_layers(x)

        batch_size = h.size(0)
        h_flat = h.view(batch_size, -1)

        # Lazy initialization
        if self._mu_proj is None:
            self._flat_size = h_flat.size(1)
            self._mu_proj = nn.Linear(self._flat_size, self.latent_dim).to(x.device)
            self._logvar_proj = nn.Linear(self._flat_size, self.latent_dim).to(x.device)

        mu = self._mu_proj(h_flat)
        logvar = self._logvar_proj(h_flat)

        return mu, logvar


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
        """Initialize VAE decoder.

        Args:
            latent_dim: Dimension of latent space.
            hidden_channels: List of hidden channel sizes.
            out_channels: Number of output channels (2 for IQ).
            output_length: Target output sequence length.
            kernel_size: Convolution kernel size.
            use_batch_norm: Whether to use batch normalization.
            dropout: Dropout probability.
        """
        super().__init__()

        if hidden_channels is None:
            hidden_channels = [256, 128, 64, 32]

        self.hidden_channels = hidden_channels
        self.output_length = output_length

        # Calculate initial size
        self._init_length = output_length
        for _ in hidden_channels:
            self._init_length = (self._init_length + 1) // 2

        self._init_channels = hidden_channels[0]
        flat_size = self._init_channels * self._init_length

        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim, flat_size),
            nn.LeakyReLU(0.2, inplace=True),
        )

        # Build decoder layers
        layers = []
        prev_channels = hidden_channels[0]

        for i, out_ch in enumerate(hidden_channels[1:]):
            layers.append(
                ConvTransposeBlock(
                    prev_channels,
                    out_ch,
                    kernel_size=kernel_size,
                    stride=2,
                    use_batch_norm=use_batch_norm,
                    dropout=dropout if i < len(hidden_channels) - 2 else 0,
                )
            )
            prev_channels = out_ch

        self.conv_layers = nn.Sequential(*layers)

        # Final layer
        self.final_conv = nn.ConvTranspose1d(
            prev_channels, out_channels, kernel_size, stride=2,
            padding=kernel_size // 2, output_padding=1
        )

    def forward(self, z: Tensor) -> Tensor:
        """Decode latent sample to signal.

        Args:
            z: Latent tensor [batch, latent_dim].

        Returns:
            Reconstructed signal [batch, 2, seq_len].
        """
        h = self.latent_proj(z)
        h = h.view(-1, self._init_channels, self._init_length)

        h = self.conv_layers(h)
        x_recon = self.final_conv(h)

        # Ensure output length matches
        if x_recon.size(2) != self.output_length:
            x_recon = nn.functional.interpolate(
                x_recon, size=self.output_length, mode="linear", align_corners=False
            )

        return x_recon


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
        """Initialize VAE.

        Args:
            latent_dim: Dimension of latent space.
            sequence_length: Input sequence length.
            hidden_channels: List of hidden channel sizes.
            kernel_size: Convolution kernel size.
            use_batch_norm: Whether to use batch normalization.
            dropout: Dropout probability.
            beta: Weight for KL divergence (beta-VAE).
        """
        super().__init__()

        if hidden_channels is None:
            hidden_channels = [32, 64, 128, 256]

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
        """Reparameterization trick for sampling.

        Args:
            mu: Mean of latent distribution [batch, latent_dim].
            logvar: Log variance of latent distribution [batch, latent_dim].

        Returns:
            Sampled latent vector [batch, latent_dim].
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu  # Use mean during inference

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward pass through VAE.

        Args:
            x: Input IQ signal [batch, 2, seq_len].

        Returns:
            Tuple of (reconstructed, mean, log_variance, latent_sample).
        """
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar, z

    def encode(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Encode input to latent distribution.

        Args:
            x: Input IQ signal [batch, 2, seq_len].

        Returns:
            Tuple of (mean, log_variance).
        """
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        """Decode latent sample.

        Args:
            z: Latent tensor [batch, latent_dim].

        Returns:
            Reconstructed signal [batch, 2, seq_len].
        """
        return self.decoder(z)

    def sample(self, num_samples: int, device: torch.device | None = None) -> Tensor:
        """Sample from the prior distribution.

        Args:
            num_samples: Number of samples to generate.
            device: Device to create samples on.

        Returns:
            Generated signals [num_samples, 2, seq_len].
        """
        if device is None:
            device = next(self.parameters()).device

        z = torch.randn(num_samples, self.latent_dim, device=device)
        return self.decode(z)

    def reconstruction_loss(self, x: Tensor, x_recon: Tensor) -> Tensor:
        """Compute reconstruction loss (MSE).

        Args:
            x: Original signal.
            x_recon: Reconstructed signal.

        Returns:
            Reconstruction loss.
        """
        return nn.functional.mse_loss(x_recon, x, reduction="mean")

    def kl_divergence(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Compute KL divergence from standard normal.

        Args:
            mu: Mean of latent distribution.
            logvar: Log variance of latent distribution.

        Returns:
            KL divergence loss.
        """
        # KL(q(z|x) || p(z)) where p(z) = N(0, I)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kl.mean()

    def loss(
        self,
        x: Tensor,
        x_recon: Tensor,
        mu: Tensor,
        logvar: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute total VAE loss.

        Args:
            x: Original signal.
            x_recon: Reconstructed signal.
            mu: Latent mean.
            logvar: Latent log variance.

        Returns:
            Tuple of (total_loss, reconstruction_loss, kl_loss).
        """
        recon_loss = self.reconstruction_loss(x, x_recon)
        kl_loss = self.kl_divergence(mu, logvar)
        total_loss = recon_loss + self.beta * kl_loss
        return total_loss, recon_loss, kl_loss

    def get_reconstruction_error(self, x: Tensor) -> Tensor:
        """Get per-sample reconstruction error.

        Args:
            x: Input IQ signal [batch, 2, seq_len].

        Returns:
            Reconstruction error per sample [batch].
        """
        x_recon, _, _, _ = self(x)
        error = ((x - x_recon) ** 2).mean(dim=(1, 2))
        return error

    def get_anomaly_score(
        self,
        x: Tensor,
        include_kl: bool = True,
        num_samples: int = 1,
    ) -> Tensor:
        """Compute anomaly score combining reconstruction and KL divergence.

        Args:
            x: Input IQ signal [batch, 2, seq_len].
            include_kl: Whether to include KL divergence in score.
            num_samples: Number of samples for Monte Carlo estimate.

        Returns:
            Anomaly score per sample [batch].
        """
        mu, logvar = self.encoder(x)

        if num_samples == 1:
            z = self.reparameterize(mu, logvar)
            x_recon = self.decoder(z)
            recon_error = ((x - x_recon) ** 2).mean(dim=(1, 2))
        else:
            # Monte Carlo estimate
            recon_errors = []
            for _ in range(num_samples):
                z = self.reparameterize(mu, logvar)
                x_recon = self.decoder(z)
                recon_errors.append(((x - x_recon) ** 2).mean(dim=(1, 2)))
            recon_error = torch.stack(recon_errors).mean(dim=0)

        if include_kl:
            # Per-sample KL
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            return recon_error + self.beta * kl
        else:
            return recon_error
