"""Base convolutional autoencoder for RF signal reconstruction."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


def _build_conv_layers(
    conv_cls: type,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    padding: int,
    output_padding: int = 0,
    use_batch_norm: bool = True,
    dropout: float = 0.0,
    activation: nn.Module | None = None,
) -> nn.Sequential:
    """Build convolutional layer sequence with normalization and dropout.

    Args:
        conv_cls: Conv1d or ConvTranspose1d class.
        in_channels: Input channels.
        out_channels: Output channels.
        kernel_size: Convolution kernel size.
        stride: Convolution stride.
        padding: Padding size.
        output_padding: Output padding for transpose convolutions.
        use_batch_norm: Whether to use batch normalization.
        dropout: Dropout probability.
        activation: Activation function (default: LeakyReLU).

    Returns:
        Sequential module with conv, optional batchnorm, activation, and dropout.
    """
    layers = []

    if conv_cls == nn.ConvTranspose1d:
        layers.append(conv_cls(in_channels, out_channels, kernel_size, stride, padding, output_padding))
    else:
        layers.append(conv_cls(in_channels, out_channels, kernel_size, stride, padding))

    if use_batch_norm:
        layers.append(nn.BatchNorm1d(out_channels))

    layers.append(activation or nn.LeakyReLU(0.2, inplace=True))

    if dropout > 0:
        layers.append(nn.Dropout(dropout))

    return nn.Sequential(*layers)


class ConvBlock(nn.Module):
    """Convolutional block with optional batch norm and dropout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 2,
        padding: int | None = None,
        use_batch_norm: bool = True,
        dropout: float = 0.0,
        activation: nn.Module | None = None,
    ):
        super().__init__()
        self.block = _build_conv_layers(
            nn.Conv1d, in_channels, out_channels, kernel_size, stride,
            padding or kernel_size // 2, 0, use_batch_norm, dropout, activation
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class ConvTransposeBlock(nn.Module):
    """Transposed convolutional block for decoder."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 2,
        padding: int | None = None,
        output_padding: int = 1,
        use_batch_norm: bool = True,
        dropout: float = 0.0,
        activation: nn.Module | None = None,
    ):
        super().__init__()
        self.block = _build_conv_layers(
            nn.ConvTranspose1d, in_channels, out_channels, kernel_size, stride,
            padding or kernel_size // 2, output_padding, use_batch_norm, dropout, activation
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class Encoder(nn.Module):
    """1D Convolutional encoder for IQ signals."""

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

        # Latent projection (lazy initialization)
        self._latent_proj = None
        self._flat_size = None

    def forward(self, x: Tensor) -> Tensor:
        """Encode input to latent space."""
        h = self.conv_layers(x).flatten(1)

        # Lazy initialization of projection layer
        if self._latent_proj is None:
            self._flat_size = h.size(1)
            self._latent_proj = nn.Linear(self._flat_size, self.latent_dim).to(x.device)

        return self._latent_proj(h)

    def get_output_shape(self, input_length: int) -> tuple[int, int]:
        """Calculate encoder output shape before flattening."""
        length = input_length
        for _ in self.hidden_channels:
            length = (length + 1) // 2  # stride=2 with padding
        return (self.hidden_channels[-1], length)


class Decoder(nn.Module):
    """1D Convolutional decoder for IQ signal reconstruction."""

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

        # Calculate initial size for latent projection
        self._init_length = output_length
        for _ in self.hidden_channels:
            self._init_length = (self._init_length + 1) // 2

        self._init_channels = self.hidden_channels[0]
        self.latent_proj = nn.Linear(latent_dim, self._init_channels * self._init_length)

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

        # Final layer without activation
        self.final_conv = nn.ConvTranspose1d(
            channels[-1], out_channels, kernel_size, stride=2,
            padding=kernel_size // 2, output_padding=1
        )

    def forward(self, z: Tensor) -> Tensor:
        """Decode latent representation to signal."""
        h = self.latent_proj(z).view(-1, self._init_channels, self._init_length)
        h = self.conv_layers(h)
        x_recon = self.final_conv(h)

        # Ensure output length matches target
        if x_recon.size(2) != self.output_length:
            x_recon = nn.functional.interpolate(
                x_recon, size=self.output_length, mode="linear", align_corners=False
            )

        return x_recon


class ConvAutoencoder(nn.Module):
    """1D Convolutional autoencoder for RF IQ signals.

    Architecture:
        Encoder: IQ [batch, 2, seq_len] -> Latent [batch, latent_dim]
        Decoder: Latent [batch, latent_dim] -> IQ [batch, 2, seq_len]

    Example:
        model = ConvAutoencoder(latent_dim=32, sequence_length=1024)
        x = torch.randn(16, 2, 1024)  # Batch of 16 IQ signals
        x_recon, z = model(x)
        loss = model.reconstruction_loss(x, x_recon)
    """

    def __init__(
        self,
        latent_dim: int = 32,
        sequence_length: int = 1024,
        hidden_channels: list[int] | None = None,
        kernel_size: int = 7,
        use_batch_norm: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_channels = hidden_channels or [32, 64, 128, 256]
        self.latent_dim = latent_dim
        self.sequence_length = sequence_length

        self.encoder = Encoder(
            in_channels=2,
            hidden_channels=hidden_channels,
            latent_dim=latent_dim,
            kernel_size=kernel_size,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
        )

        self.decoder = Decoder(
            latent_dim=latent_dim,
            hidden_channels=hidden_channels[::-1],
            out_channels=2,
            output_length=sequence_length,
            kernel_size=kernel_size,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass through autoencoder."""
        z = self.encoder(x)
        return self.decoder(z), z

    def encode(self, x: Tensor) -> Tensor:
        """Encode input to latent space."""
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        """Decode latent representation."""
        return self.decoder(z)

    def reconstruction_loss(self, x: Tensor, x_recon: Tensor, reduction: str = "mean") -> Tensor:
        """Compute reconstruction loss (MSE)."""
        return nn.functional.mse_loss(x_recon, x, reduction=reduction)

    def get_reconstruction_error(self, x: Tensor) -> Tensor:
        """Get per-sample reconstruction error."""
        x_recon, _ = self(x)
        return ((x - x_recon) ** 2).mean(dim=(1, 2))
