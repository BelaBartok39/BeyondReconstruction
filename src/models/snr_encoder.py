"""SNR-conditioned VAE for adaptive anomaly detection."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .autoencoder import ConvBlock, ConvTransposeBlock


class SNREncoder(nn.Module):
    """VAE encoder with SNR conditioning."""

    def __init__(
        self,
        in_channels: int = 2,
        hidden_channels: list[int] | None = None,
        latent_dim: int = 32,
        snr_embedding_dim: int = 16,
        kernel_size: int = 7,
        use_batch_norm: bool = True,
        dropout: float = 0.1,
    ):
        """Initialize SNR-conditioned encoder.

        Args:
            in_channels: Number of input channels (2 for IQ).
            hidden_channels: List of hidden channel sizes.
            latent_dim: Dimension of latent space.
            snr_embedding_dim: Dimension of SNR embedding.
            kernel_size: Convolution kernel size.
            use_batch_norm: Whether to use batch normalization.
            dropout: Dropout probability.
        """
        super().__init__()

        if hidden_channels is None:
            hidden_channels = [32, 64, 128, 256]

        self.hidden_channels = hidden_channels
        self.latent_dim = latent_dim
        self.snr_embedding_dim = snr_embedding_dim

        # SNR embedding network
        self.snr_embed = nn.Sequential(
            nn.Linear(1, snr_embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(snr_embedding_dim, snr_embedding_dim),
            nn.ReLU(inplace=True),
        )

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

        # Projection layers (will be initialized lazily)
        self._mu_proj = None
        self._logvar_proj = None
        self._flat_size = None

    def forward(self, x: Tensor, snr: Tensor) -> tuple[Tensor, Tensor]:
        """Encode input with SNR conditioning.

        Args:
            x: Input tensor [batch, 2, seq_len].
            snr: Normalized SNR values [batch] or [batch, 1].

        Returns:
            Tuple of (mean, log_variance) each [batch, latent_dim].
        """
        # Ensure SNR has correct shape
        if snr.dim() == 1:
            snr = snr.unsqueeze(1)

        # Get SNR embedding
        snr_emb = self.snr_embed(snr)  # [batch, snr_embedding_dim]

        # Convolutional encoding
        h = self.conv_layers(x)

        batch_size = h.size(0)
        h_flat = h.view(batch_size, -1)

        # Concatenate SNR embedding
        h_combined = torch.cat([h_flat, snr_emb], dim=1)

        # Lazy initialization
        if self._mu_proj is None:
            combined_size = h_combined.size(1)
            self._flat_size = h_flat.size(1)
            self._mu_proj = nn.Linear(combined_size, self.latent_dim).to(x.device)
            self._logvar_proj = nn.Linear(combined_size, self.latent_dim).to(x.device)

        mu = self._mu_proj(h_combined)
        logvar = self._logvar_proj(h_combined)

        return mu, logvar


class SNRDecoder(nn.Module):
    """VAE decoder with SNR conditioning."""

    def __init__(
        self,
        latent_dim: int = 32,
        hidden_channels: list[int] | None = None,
        out_channels: int = 2,
        output_length: int = 1024,
        snr_embedding_dim: int = 16,
        kernel_size: int = 7,
        use_batch_norm: bool = True,
        dropout: float = 0.1,
    ):
        """Initialize SNR-conditioned decoder.

        Args:
            latent_dim: Dimension of latent space.
            hidden_channels: List of hidden channel sizes.
            out_channels: Number of output channels (2 for IQ).
            output_length: Target output sequence length.
            snr_embedding_dim: Dimension of SNR embedding.
            kernel_size: Convolution kernel size.
            use_batch_norm: Whether to use batch normalization.
            dropout: Dropout probability.
        """
        super().__init__()

        if hidden_channels is None:
            hidden_channels = [256, 128, 64, 32]

        self.hidden_channels = hidden_channels
        self.output_length = output_length
        self.snr_embedding_dim = snr_embedding_dim

        # SNR embedding network
        self.snr_embed = nn.Sequential(
            nn.Linear(1, snr_embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(snr_embedding_dim, snr_embedding_dim),
            nn.ReLU(inplace=True),
        )

        # Calculate initial size
        self._init_length = output_length
        for _ in hidden_channels:
            self._init_length = (self._init_length + 1) // 2

        self._init_channels = hidden_channels[0]
        flat_size = self._init_channels * self._init_length

        # Latent projection includes SNR embedding
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim + snr_embedding_dim, flat_size),
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

    def forward(self, z: Tensor, snr: Tensor) -> Tensor:
        """Decode latent sample with SNR conditioning.

        Args:
            z: Latent tensor [batch, latent_dim].
            snr: Normalized SNR values [batch] or [batch, 1].

        Returns:
            Reconstructed signal [batch, 2, seq_len].
        """
        # Ensure SNR has correct shape
        if snr.dim() == 1:
            snr = snr.unsqueeze(1)

        # Get SNR embedding
        snr_emb = self.snr_embed(snr)

        # Concatenate and project
        z_combined = torch.cat([z, snr_emb], dim=1)
        h = self.latent_proj(z_combined)
        h = h.view(-1, self._init_channels, self._init_length)

        # Convolutional decoding
        h = self.conv_layers(h)
        x_recon = self.final_conv(h)

        # Ensure output length matches
        if x_recon.size(2) != self.output_length:
            x_recon = nn.functional.interpolate(
                x_recon, size=self.output_length, mode="linear", align_corners=False
            )

        return x_recon


class SNRConditionedVAE(nn.Module):
    """SNR-Conditioned Variational Autoencoder.

    This model conditions both encoder and decoder on SNR, allowing it to:
    - Learn SNR-dependent reconstruction patterns
    - Tolerate higher reconstruction error in low-SNR conditions
    - Be more sensitive to anomalies in high-SNR signals

    Example:
        model = SNRConditionedVAE(latent_dim=32, sequence_length=1024)
        x = torch.randn(16, 2, 1024)
        snr = torch.rand(16)  # Normalized SNR [0, 1]
        x_recon, mu, logvar, z = model(x, snr)
    """

    def __init__(
        self,
        latent_dim: int = 32,
        sequence_length: int = 1024,
        hidden_channels: list[int] | None = None,
        snr_embedding_dim: int = 16,
        kernel_size: int = 7,
        use_batch_norm: bool = True,
        dropout: float = 0.1,
        beta: float = 1.0,
    ):
        """Initialize SNR-conditioned VAE.

        Args:
            latent_dim: Dimension of latent space.
            sequence_length: Input sequence length.
            hidden_channels: List of hidden channel sizes.
            snr_embedding_dim: Dimension of SNR embedding.
            kernel_size: Convolution kernel size.
            use_batch_norm: Whether to use batch normalization.
            dropout: Dropout probability.
            beta: Weight for KL divergence.
        """
        super().__init__()

        if hidden_channels is None:
            hidden_channels = [32, 64, 128, 256]

        self.latent_dim = latent_dim
        self.sequence_length = sequence_length
        self.beta = beta

        self.encoder = SNREncoder(
            in_channels=2,
            hidden_channels=hidden_channels,
            latent_dim=latent_dim,
            snr_embedding_dim=snr_embedding_dim,
            kernel_size=kernel_size,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
        )

        self.decoder = SNRDecoder(
            latent_dim=latent_dim,
            hidden_channels=hidden_channels[::-1],
            out_channels=2,
            output_length=sequence_length,
            snr_embedding_dim=snr_embedding_dim,
            kernel_size=kernel_size,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
        )

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterization trick for sampling."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu

    def forward(
        self, x: Tensor, snr: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward pass with SNR conditioning.

        Args:
            x: Input IQ signal [batch, 2, seq_len].
            snr: Normalized SNR values [batch].

        Returns:
            Tuple of (reconstructed, mean, log_variance, latent_sample).
        """
        mu, logvar = self.encoder(x, snr)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z, snr)
        return x_recon, mu, logvar, z

    def encode(self, x: Tensor, snr: Tensor) -> tuple[Tensor, Tensor]:
        """Encode with SNR conditioning."""
        return self.encoder(x, snr)

    def decode(self, z: Tensor, snr: Tensor) -> Tensor:
        """Decode with SNR conditioning."""
        return self.decoder(z, snr)

    def reconstruction_loss(self, x: Tensor, x_recon: Tensor) -> Tensor:
        """Compute reconstruction loss."""
        return nn.functional.mse_loss(x_recon, x, reduction="mean")

    def kl_divergence(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Compute KL divergence from standard normal."""
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kl.mean()

    def loss(
        self,
        x: Tensor,
        x_recon: Tensor,
        mu: Tensor,
        logvar: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute total VAE loss."""
        recon_loss = self.reconstruction_loss(x, x_recon)
        kl_loss = self.kl_divergence(mu, logvar)
        total_loss = recon_loss + self.beta * kl_loss
        return total_loss, recon_loss, kl_loss

    def get_reconstruction_error(self, x: Tensor, snr: Tensor) -> Tensor:
        """Get per-sample reconstruction error.

        Args:
            x: Input IQ signal [batch, 2, seq_len].
            snr: Normalized SNR values [batch].

        Returns:
            Reconstruction error per sample [batch].
        """
        x_recon, _, _, _ = self(x, snr)
        error = ((x - x_recon) ** 2).mean(dim=(1, 2))
        return error

    def get_anomaly_score(
        self,
        x: Tensor,
        snr: Tensor,
        include_kl: bool = True,
        num_samples: int = 1,
    ) -> Tensor:
        """Compute anomaly score with SNR conditioning.

        Args:
            x: Input IQ signal [batch, 2, seq_len].
            snr: Normalized SNR values [batch].
            include_kl: Whether to include KL divergence.
            num_samples: Number of samples for Monte Carlo estimate.

        Returns:
            Anomaly score per sample [batch].
        """
        mu, logvar = self.encoder(x, snr)

        if num_samples == 1:
            z = self.reparameterize(mu, logvar)
            x_recon = self.decoder(z, snr)
            recon_error = ((x - x_recon) ** 2).mean(dim=(1, 2))
        else:
            recon_errors = []
            for _ in range(num_samples):
                z = self.reparameterize(mu, logvar)
                x_recon = self.decoder(z, snr)
                recon_errors.append(((x - x_recon) ** 2).mean(dim=(1, 2)))
            recon_error = torch.stack(recon_errors).mean(dim=0)

        if include_kl:
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
            return recon_error + self.beta * kl
        else:
            return recon_error


def create_model(config) -> nn.Module:
    """Create model from configuration.

    Args:
        config: Configuration object with model settings.

    Returns:
        Model instance.
    """
    model_type = config.model.type.lower()

    common_args = {
        "latent_dim": config.model.latent_dim,
        "sequence_length": config.data.sequence_length,
        "hidden_channels": config.model.hidden_channels,
        "kernel_size": config.model.kernel_size,
        "use_batch_norm": config.model.use_batch_norm,
        "dropout": config.model.dropout,
    }

    if model_type == "autoencoder":
        from .autoencoder import ConvAutoencoder
        return ConvAutoencoder(**common_args)

    elif model_type == "vae":
        from .vae import ConvVAE
        return ConvVAE(**common_args, beta=config.model.beta)

    elif model_type == "snr_vae":
        return SNRConditionedVAE(
            **common_args,
            snr_embedding_dim=config.model.snr_embedding_dim,
            beta=config.model.beta,
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")
