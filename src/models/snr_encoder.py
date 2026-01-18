"""SNR and power-conditioned VAE for adaptive anomaly detection."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

from .autoencoder import ConvBlock, ConvTransposeBlock


def _create_conditioning_embedding(input_dim: int, embedding_dim: int) -> nn.Sequential:
    """Create conditioning embedding network shared by encoder and decoder.

    Args:
        input_dim: Number of conditioning inputs (e.g., 1 for SNR only, 2 for SNR+power).
        embedding_dim: Output embedding dimension.
    """
    return nn.Sequential(
        nn.Linear(input_dim, embedding_dim),
        nn.ReLU(inplace=True),
        nn.Linear(embedding_dim, embedding_dim),
        nn.ReLU(inplace=True),
    )


def _normalize_conditioning(cond: Tensor) -> Tensor:
    """Ensure conditioning tensor has shape [batch, N]."""
    if cond.dim() == 1:
        return cond.unsqueeze(1)
    return cond


def _combine_conditioning(snr: Tensor, power: Tensor | None = None) -> Tensor:
    """Combine SNR and optional power into a single conditioning tensor.

    Args:
        snr: SNR values [batch] or [batch, 1].
        power: Optional power values [batch] or [batch, 1].

    Returns:
        Combined conditioning tensor [batch, N] where N is 1 or 2.
    """
    snr = _normalize_conditioning(snr)
    if power is None:
        return snr
    power = _normalize_conditioning(power)
    return torch.cat([snr, power], dim=1)


class SNREncoder(nn.Module):
    """VAE encoder with SNR and optional power conditioning."""

    def __init__(
        self,
        in_channels: int = 2,
        hidden_channels: list[int] | None = None,
        latent_dim: int = 32,
        snr_embedding_dim: int = 16,
        kernel_size: int = 7,
        use_batch_norm: bool = True,
        dropout: float = 0.1,
        use_power_conditioning: bool = False,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels or [32, 64, 128, 256]
        self.latent_dim = latent_dim
        self.snr_embedding_dim = snr_embedding_dim
        self.use_power_conditioning = use_power_conditioning

        # Conditioning input dimension: 1 for SNR only, 2 for SNR+power
        cond_input_dim = 2 if use_power_conditioning else 1
        self.cond_embed = _create_conditioning_embedding(cond_input_dim, snr_embedding_dim)

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

    def forward(self, x: Tensor, snr: Tensor, power: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Encode input with SNR and optional power conditioning."""
        cond = _combine_conditioning(snr, power if self.use_power_conditioning else None)
        cond_emb = self.cond_embed(cond)
        h = self.conv_layers(x).flatten(1)
        h_combined = torch.cat([h, cond_emb], dim=1)

        # Lazy initialization
        if self._mu_proj is None:
            combined_size = h_combined.size(1)
            self._mu_proj = nn.Linear(combined_size, self.latent_dim).to(x.device)
            self._logvar_proj = nn.Linear(combined_size, self.latent_dim).to(x.device)

        return self._mu_proj(h_combined), self._logvar_proj(h_combined)


class SNRDecoder(nn.Module):
    """VAE decoder with SNR and optional power conditioning."""

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
        use_power_conditioning: bool = False,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels or [256, 128, 64, 32]
        self.output_length = output_length
        self.use_power_conditioning = use_power_conditioning

        # Conditioning input dimension: 1 for SNR only, 2 for SNR+power
        cond_input_dim = 2 if use_power_conditioning else 1
        self.cond_embed = _create_conditioning_embedding(cond_input_dim, snr_embedding_dim)

        # Calculate initial size
        self._init_length = output_length
        for _ in self.hidden_channels:
            self._init_length = (self._init_length + 1) // 2

        self._init_channels = self.hidden_channels[0]
        self.latent_proj = nn.Sequential(
            nn.Linear(latent_dim + snr_embedding_dim, self._init_channels * self._init_length),
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

    def forward(self, z: Tensor, snr: Tensor, power: Tensor | None = None) -> Tensor:
        """Decode latent sample with SNR and optional power conditioning."""
        cond = _combine_conditioning(snr, power if self.use_power_conditioning else None)
        cond_emb = self.cond_embed(cond)
        z_combined = torch.cat([z, cond_emb], dim=1)
        h = self.latent_proj(z_combined).view(-1, self._init_channels, self._init_length)
        h = self.conv_layers(h)
        x_recon = self.final_conv(h)

        # Ensure output length matches
        if x_recon.size(2) != self.output_length:
            x_recon = nn.functional.interpolate(
                x_recon, size=self.output_length, mode="linear", align_corners=False
            )

        return x_recon


class SNRConditionedVAE(nn.Module):
    """SNR and Power-Conditioned Variational Autoencoder.

    This model conditions both encoder and decoder on SNR and optionally on
    signal power, allowing it to:
    - Learn SNR-dependent reconstruction patterns
    - Tolerate higher reconstruction error in low-SNR conditions
    - Be more sensitive to anomalies in high-SNR signals
    - Distinguish anomalies by their unusual power characteristics

    Example:
        model = SNRConditionedVAE(latent_dim=32, sequence_length=1024, use_power_conditioning=True)
        x = torch.randn(16, 2, 1024)
        snr = torch.rand(16)  # Normalized SNR [0, 1]
        power = torch.rand(16)  # Normalized power [0, 1]
        x_recon, mu, logvar, z = model(x, snr, power)
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
        use_power_conditioning: bool = False,
    ):
        super().__init__()
        hidden_channels = hidden_channels or [32, 64, 128, 256]
        self.latent_dim = latent_dim
        self.sequence_length = sequence_length
        self.beta = beta
        self.use_power_conditioning = use_power_conditioning

        self.encoder = SNREncoder(
            in_channels=2,
            hidden_channels=hidden_channels,
            latent_dim=latent_dim,
            snr_embedding_dim=snr_embedding_dim,
            kernel_size=kernel_size,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            use_power_conditioning=use_power_conditioning,
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
            use_power_conditioning=use_power_conditioning,
        )

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterization trick for sampling."""
        if self.training:
            return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return mu

    def forward(
        self, x: Tensor, snr: Tensor, power: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward pass with SNR and optional power conditioning."""
        mu, logvar = self.encoder(x, snr, power)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z, snr, power), mu, logvar, z

    def encode(self, x: Tensor, snr: Tensor, power: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Encode with SNR and optional power conditioning."""
        return self.encoder(x, snr, power)

    def decode(self, z: Tensor, snr: Tensor, power: Tensor | None = None) -> Tensor:
        """Decode with SNR and optional power conditioning."""
        return self.decoder(z, snr, power)

    def reconstruction_loss(self, x: Tensor, x_recon: Tensor) -> Tensor:
        """Compute reconstruction loss."""
        return nn.functional.mse_loss(x_recon, x, reduction="mean")

    def kl_divergence(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Compute KL divergence from standard normal."""
        from .vae import _compute_kl_divergence
        return _compute_kl_divergence(mu, logvar, reduce=True)

    def loss(self, x: Tensor, x_recon: Tensor, mu: Tensor, logvar: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Compute total VAE loss."""
        recon_loss = self.reconstruction_loss(x, x_recon)
        kl_loss = self.kl_divergence(mu, logvar)
        return recon_loss + self.beta * kl_loss, recon_loss, kl_loss

    def get_reconstruction_error(self, x: Tensor, snr: Tensor, power: Tensor | None = None) -> Tensor:
        """Get per-sample reconstruction error."""
        x_recon, _, _, _ = self(x, snr, power)
        return ((x - x_recon) ** 2).mean(dim=(1, 2))

    def get_anomaly_score(
        self,
        x: Tensor,
        snr: Tensor,
        power: Tensor | None = None,
        include_kl: bool = True,
        num_samples: int = 1,
    ) -> Tensor:
        """Compute anomaly score with SNR and optional power conditioning."""
        mu, logvar = self.encoder(x, snr, power)

        # Compute reconstruction error with optional Monte Carlo sampling
        if num_samples == 1:
            z = self.reparameterize(mu, logvar)
            x_recon = self.decoder(z, snr, power)
            recon_error = ((x - x_recon) ** 2).mean(dim=(1, 2))
        else:
            # Vectorized Monte Carlo estimate
            batch_size = x.size(0)
            mu_expanded = mu.unsqueeze(1).expand(-1, num_samples, -1).reshape(-1, mu.size(1))
            logvar_expanded = logvar.unsqueeze(1).expand(-1, num_samples, -1).reshape(-1, logvar.size(1))
            z = self.reparameterize(mu_expanded, logvar_expanded)

            snr_expanded = _normalize_conditioning(snr).unsqueeze(1).expand(-1, num_samples, -1).reshape(-1, 1)
            x_expanded = x.unsqueeze(1).expand(-1, num_samples, -1, -1).reshape(-1, x.size(1), x.size(2))

            if power is not None:
                power_expanded = _normalize_conditioning(power).unsqueeze(1).expand(-1, num_samples, -1).reshape(-1, 1)
                x_recon = self.decoder(z, snr_expanded.squeeze(1), power_expanded.squeeze(1))
            else:
                x_recon = self.decoder(z, snr_expanded.squeeze(1), None)

            recon_error = ((x_expanded - x_recon) ** 2).mean(dim=(1, 2)).view(batch_size, num_samples).mean(dim=1)

        if include_kl:
            from .vae import _compute_kl_divergence
            return recon_error + self.beta * _compute_kl_divergence(mu, logvar, reduce=False)
        return recon_error


def create_model(config) -> nn.Module:
    """Create model from configuration."""
    common_args = {
        "latent_dim": config.model.latent_dim,
        "sequence_length": config.data.sequence_length,
        "hidden_channels": config.model.hidden_channels,
        "kernel_size": config.model.kernel_size,
        "use_batch_norm": config.model.use_batch_norm,
        "dropout": config.model.dropout,
    }

    model_type = config.model.type.lower()

    if model_type == "autoencoder":
        from .autoencoder import ConvAutoencoder
        return ConvAutoencoder(**common_args)

    if model_type == "vae":
        from .vae import ConvVAE
        return ConvVAE(**common_args, beta=config.model.beta)

    if model_type == "snr_vae":
        use_power = getattr(config.model, 'use_power_conditioning', False)
        return SNRConditionedVAE(
            **common_args,
            snr_embedding_dim=config.model.snr_embedding_dim,
            beta=config.model.beta,
            use_power_conditioning=use_power,
        )

    raise ValueError(f"Unknown model type: {model_type}")
