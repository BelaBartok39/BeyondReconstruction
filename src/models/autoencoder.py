"""Base convolutional autoencoder for RF signal reconstruction."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


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
        """Initialize convolutional block.

        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            kernel_size: Convolution kernel size.
            stride: Convolution stride.
            padding: Padding (auto-calculated if None).
            use_batch_norm: Whether to use batch normalization.
            dropout: Dropout probability.
            activation: Activation function (default: LeakyReLU).
        """
        super().__init__()

        if padding is None:
            padding = kernel_size // 2

        layers = [
            nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding)
        ]

        if use_batch_norm:
            layers.append(nn.BatchNorm1d(out_channels))

        if activation is None:
            activation = nn.LeakyReLU(0.2, inplace=True)
        layers.append(activation)

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
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
        """Initialize transposed convolutional block.

        Args:
            in_channels: Input channels.
            out_channels: Output channels.
            kernel_size: Convolution kernel size.
            stride: Convolution stride.
            padding: Padding (auto-calculated if None).
            output_padding: Output padding for size matching.
            use_batch_norm: Whether to use batch normalization.
            dropout: Dropout probability.
            activation: Activation function (default: LeakyReLU).
        """
        super().__init__()

        if padding is None:
            padding = kernel_size // 2

        layers = [
            nn.ConvTranspose1d(
                in_channels, out_channels, kernel_size, stride, padding, output_padding
            )
        ]

        if use_batch_norm:
            layers.append(nn.BatchNorm1d(out_channels))

        if activation is None:
            activation = nn.LeakyReLU(0.2, inplace=True)
        layers.append(activation)

        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
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
        """Initialize encoder.

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

        # Latent projection (will be set after first forward pass)
        self._latent_proj = None
        self._flat_size = None

    def forward(self, x: Tensor) -> Tensor:
        """Encode input to latent space.

        Args:
            x: Input tensor [batch, 2, seq_len].

        Returns:
            Latent representation [batch, latent_dim].
        """
        # Convolutional encoding
        h = self.conv_layers(x)

        # Flatten
        batch_size = h.size(0)
        h_flat = h.view(batch_size, -1)

        # Lazy initialization of projection layer
        if self._latent_proj is None:
            self._flat_size = h_flat.size(1)
            self._latent_proj = nn.Linear(self._flat_size, self.latent_dim).to(x.device)

        z = self._latent_proj(h_flat)
        return z

    def get_output_shape(self, input_length: int) -> tuple[int, int]:
        """Calculate encoder output shape before flattening.

        Args:
            input_length: Input sequence length.

        Returns:
            Tuple of (channels, length) after conv layers.
        """
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
        """Initialize decoder.

        Args:
            latent_dim: Dimension of latent space.
            hidden_channels: List of hidden channel sizes (reverse of encoder).
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

        # Calculate initial size for latent projection
        self._init_length = output_length
        for _ in hidden_channels:
            self._init_length = (self._init_length + 1) // 2

        self._init_channels = hidden_channels[0]
        flat_size = self._init_channels * self._init_length

        self.latent_proj = nn.Linear(latent_dim, flat_size)

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

        # Final layer without activation
        self.final_conv = nn.ConvTranspose1d(
            prev_channels, out_channels, kernel_size, stride=2, padding=kernel_size // 2, output_padding=1
        )

    def forward(self, z: Tensor) -> Tensor:
        """Decode latent representation to signal.

        Args:
            z: Latent tensor [batch, latent_dim].

        Returns:
            Reconstructed signal [batch, 2, seq_len].
        """
        # Project and reshape
        h = self.latent_proj(z)
        h = h.view(-1, self._init_channels, self._init_length)

        # Convolutional decoding
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
        """Initialize autoencoder.

        Args:
            latent_dim: Dimension of latent space.
            sequence_length: Input sequence length.
            hidden_channels: List of hidden channel sizes for encoder.
            kernel_size: Convolution kernel size.
            use_batch_norm: Whether to use batch normalization.
            dropout: Dropout probability.
        """
        super().__init__()

        if hidden_channels is None:
            hidden_channels = [32, 64, 128, 256]

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
            hidden_channels=hidden_channels[::-1],  # Reverse for decoder
            out_channels=2,
            output_length=sequence_length,
            kernel_size=kernel_size,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Forward pass through autoencoder.

        Args:
            x: Input IQ signal [batch, 2, seq_len].

        Returns:
            Tuple of (reconstructed signal, latent representation).
        """
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return x_recon, z

    def encode(self, x: Tensor) -> Tensor:
        """Encode input to latent space.

        Args:
            x: Input IQ signal [batch, 2, seq_len].

        Returns:
            Latent representation [batch, latent_dim].
        """
        return self.encoder(x)

    def decode(self, z: Tensor) -> Tensor:
        """Decode latent representation.

        Args:
            z: Latent tensor [batch, latent_dim].

        Returns:
            Reconstructed signal [batch, 2, seq_len].
        """
        return self.decoder(z)

    def reconstruction_loss(
        self,
        x: Tensor,
        x_recon: Tensor,
        reduction: str = "mean",
    ) -> Tensor:
        """Compute reconstruction loss (MSE).

        Args:
            x: Original signal [batch, 2, seq_len].
            x_recon: Reconstructed signal [batch, 2, seq_len].
            reduction: Loss reduction method.

        Returns:
            Reconstruction loss.
        """
        return nn.functional.mse_loss(x_recon, x, reduction=reduction)

    def get_reconstruction_error(self, x: Tensor) -> Tensor:
        """Get per-sample reconstruction error.

        Args:
            x: Input IQ signal [batch, 2, seq_len].

        Returns:
            Reconstruction error per sample [batch].
        """
        x_recon, _ = self(x)
        # MSE per sample
        error = ((x - x_recon) ** 2).mean(dim=(1, 2))
        return error
