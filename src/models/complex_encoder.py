"""Complex-valued neural network layers for RF signal processing.

Complex convolutions naturally preserve phase information, which is lost when
treating I/Q as separate real channels. This is critical for detecting
frequency drift anomalies.

References:
- Trabelsi et al., "Deep Complex Networks" (ICLR 2018)
- Virtue et al., "Better than Real: Complex-valued Neural Nets for MRI Fingerprinting"
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ComplexConv1d(nn.Module):
    """Complex-valued 1D convolution.

    Implements complex convolution as:
    (W_r + iW_i) * (x_r + ix_i) = (W_r*x_r - W_i*x_i) + i(W_r*x_i + W_i*x_r)

    This preserves phase relationships that are lost in real-valued convolutions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Real and imaginary weight matrices
        self.conv_real = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias
        )
        self.conv_imag = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=padding, bias=bias
        )

    def forward(self, x_real: Tensor, x_imag: Tensor) -> tuple[Tensor, Tensor]:
        """Apply complex convolution.

        Args:
            x_real: Real part [batch, channels, seq_len]
            x_imag: Imaginary part [batch, channels, seq_len]

        Returns:
            Tuple of (output_real, output_imag)
        """
        # (W_r + iW_i) * (x_r + ix_i) = (W_r*x_r - W_i*x_i) + i(W_r*x_i + W_i*x_r)
        out_real = self.conv_real(x_real) - self.conv_imag(x_imag)
        out_imag = self.conv_real(x_imag) + self.conv_imag(x_real)
        return out_real, out_imag


class ComplexBatchNorm1d(nn.Module):
    """Complex-valued batch normalization.

    Normalizes the complex signal while preserving phase information.
    Uses 2x2 covariance whitening as in Trabelsi et al.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps

        # Learnable parameters
        self.gamma_rr = nn.Parameter(torch.ones(num_features) / 1.41421)  # 1/sqrt(2)
        self.gamma_ri = nn.Parameter(torch.zeros(num_features))
        self.gamma_ii = nn.Parameter(torch.ones(num_features) / 1.41421)
        self.beta_real = nn.Parameter(torch.zeros(num_features))
        self.beta_imag = nn.Parameter(torch.zeros(num_features))

        # Running statistics
        self.register_buffer('running_mean_real', torch.zeros(num_features))
        self.register_buffer('running_mean_imag', torch.zeros(num_features))
        self.register_buffer('running_var_rr', torch.ones(num_features))
        self.register_buffer('running_var_ri', torch.zeros(num_features))
        self.register_buffer('running_var_ii', torch.ones(num_features))

        self.momentum = momentum

    def forward(self, x_real: Tensor, x_imag: Tensor) -> tuple[Tensor, Tensor]:
        """Apply complex batch normalization.

        Args:
            x_real: Real part [batch, channels, seq_len]
            x_imag: Imaginary part [batch, channels, seq_len]

        Returns:
            Normalized (real, imag) tuple
        """
        if self.training:
            # Compute batch statistics
            mean_real = x_real.mean(dim=(0, 2))
            mean_imag = x_imag.mean(dim=(0, 2))

            # Center
            x_real_c = x_real - mean_real.view(1, -1, 1)
            x_imag_c = x_imag - mean_imag.view(1, -1, 1)

            # Covariance
            n = x_real.numel() / x_real.size(1)
            var_rr = (x_real_c ** 2).mean(dim=(0, 2))
            var_ii = (x_imag_c ** 2).mean(dim=(0, 2))
            var_ri = (x_real_c * x_imag_c).mean(dim=(0, 2))

            # Update running statistics
            with torch.no_grad():
                self.running_mean_real = (1 - self.momentum) * self.running_mean_real + self.momentum * mean_real
                self.running_mean_imag = (1 - self.momentum) * self.running_mean_imag + self.momentum * mean_imag
                self.running_var_rr = (1 - self.momentum) * self.running_var_rr + self.momentum * var_rr
                self.running_var_ii = (1 - self.momentum) * self.running_var_ii + self.momentum * var_ii
                self.running_var_ri = (1 - self.momentum) * self.running_var_ri + self.momentum * var_ri
        else:
            mean_real = self.running_mean_real
            mean_imag = self.running_mean_imag
            x_real_c = x_real - mean_real.view(1, -1, 1)
            x_imag_c = x_imag - mean_imag.view(1, -1, 1)
            var_rr = self.running_var_rr
            var_ii = self.running_var_ii
            var_ri = self.running_var_ri

        # Whitening using inverse square root of covariance
        # det = var_rr * var_ii - var_ri^2
        # Clamp variances to prevent numerical instability
        var_rr = torch.clamp(var_rr, min=self.eps)
        var_ii = torch.clamp(var_ii, min=self.eps)
        det = var_rr * var_ii - var_ri ** 2
        det = torch.clamp(det, min=self.eps)  # Ensure positive definite
        s = torch.sqrt(det)
        s = torch.clamp(s, min=self.eps)  # Prevent division by zero
        t = torch.sqrt(var_rr + var_ii + 2 * s)
        t = torch.clamp(t, min=self.eps)
        inv_t = 1.0 / t

        # Inverse sqrt of covariance matrix
        w_rr = (var_ii + s) * inv_t / s
        w_ii = (var_rr + s) * inv_t / s
        w_ri = -var_ri * inv_t / s

        # Clamp whitening weights to prevent explosion
        w_rr = torch.clamp(w_rr, min=-100, max=100)
        w_ii = torch.clamp(w_ii, min=-100, max=100)
        w_ri = torch.clamp(w_ri, min=-100, max=100)

        # Apply whitening
        x_real_w = w_rr.view(1, -1, 1) * x_real_c + w_ri.view(1, -1, 1) * x_imag_c
        x_imag_w = w_ri.view(1, -1, 1) * x_real_c + w_ii.view(1, -1, 1) * x_imag_c

        # Apply learnable scale and shift
        out_real = (self.gamma_rr.view(1, -1, 1) * x_real_w +
                    self.gamma_ri.view(1, -1, 1) * x_imag_w +
                    self.beta_real.view(1, -1, 1))
        out_imag = (self.gamma_ri.view(1, -1, 1) * x_real_w +
                    self.gamma_ii.view(1, -1, 1) * x_imag_w +
                    self.beta_imag.view(1, -1, 1))

        return out_real, out_imag


class ComplexReLU(nn.Module):
    """Complex ReLU activation.

    Applies ReLU to the magnitude while preserving phase:
    z = |z| * exp(i*arg(z))
    CReLU(z) = ReLU(|z|) * exp(i*arg(z))
    """

    def forward(self, x_real: Tensor, x_imag: Tensor) -> tuple[Tensor, Tensor]:
        magnitude = torch.sqrt(x_real ** 2 + x_imag ** 2 + 1e-8)
        phase = torch.atan2(x_imag, x_real)

        # ReLU on magnitude
        magnitude_relu = F.relu(magnitude - 0.5) + 0.5  # Bias to avoid killing gradients

        # Reconstruct
        out_real = magnitude_relu * torch.cos(phase)
        out_imag = magnitude_relu * torch.sin(phase)

        return out_real, out_imag


class ModReLU(nn.Module):
    """Modulus ReLU activation (learnable bias on magnitude).

    ModReLU(z) = ReLU(|z| + b) * z/|z|

    This is generally better than CReLU as it has a learnable bias.
    """

    def __init__(self, num_features: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(num_features))

    def forward(self, x_real: Tensor, x_imag: Tensor) -> tuple[Tensor, Tensor]:
        magnitude = torch.sqrt(x_real ** 2 + x_imag ** 2 + 1e-8)

        # Apply ReLU with learnable bias
        bias = self.bias.view(1, -1, 1)
        magnitude_biased = F.relu(magnitude + bias)

        # Normalize by original magnitude to get unit direction, then scale
        scale = magnitude_biased / (magnitude + 1e-8)
        out_real = scale * x_real
        out_imag = scale * x_imag

        return out_real, out_imag


class ComplexConvBlock(nn.Module):
    """Complex convolutional block with normalization and activation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 7,
        stride: int = 2,
        use_batch_norm: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()
        padding = kernel_size // 2

        self.conv = ComplexConv1d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = ComplexBatchNorm1d(out_channels) if use_batch_norm else None
        self.activation = ModReLU(out_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else None

    def forward(self, x_real: Tensor, x_imag: Tensor) -> tuple[Tensor, Tensor]:
        x_real, x_imag = self.conv(x_real, x_imag)

        if self.bn is not None:
            x_real, x_imag = self.bn(x_real, x_imag)

        x_real, x_imag = self.activation(x_real, x_imag)

        if self.dropout is not None:
            x_real = self.dropout(x_real)
            x_imag = self.dropout(x_imag)

        return x_real, x_imag


def _create_cond_embedding(input_dim: int, embedding_dim: int) -> nn.Sequential:
    """Create conditioning embedding network."""
    return nn.Sequential(
        nn.Linear(input_dim, embedding_dim),
        nn.ReLU(inplace=True),
        nn.Linear(embedding_dim, embedding_dim),
        nn.ReLU(inplace=True),
    )


class ComplexEncoder(nn.Module):
    """Complex-valued encoder that preserves phase information.

    Takes I/Q signal as input and produces complex latent representation.
    Supports optional SNR and power conditioning.
    """

    def __init__(
        self,
        hidden_channels: list[int] | None = None,
        latent_dim: int = 32,
        kernel_size: int = 7,
        use_batch_norm: bool = True,
        dropout: float = 0.1,
        snr_embedding_dim: int = 16,
        use_power_conditioning: bool = False,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels or [32, 64, 128, 256]
        self.latent_dim = latent_dim
        self.snr_embedding_dim = snr_embedding_dim
        self.use_power_conditioning = use_power_conditioning

        # Conditioning embedding
        cond_input_dim = 2 if use_power_conditioning else 1
        self.cond_embed = _create_cond_embedding(cond_input_dim, snr_embedding_dim)

        # Build complex conv layers
        # Input: 1 complex channel (I + jQ)
        channels = [1] + self.hidden_channels
        self.conv_layers = nn.ModuleList([
            ComplexConvBlock(
                channels[i], channels[i + 1], kernel_size, stride=2,
                use_batch_norm=use_batch_norm,
                dropout=dropout if i < len(self.hidden_channels) - 1 else 0
            )
            for i in range(len(self.hidden_channels))
        ])

        # Projection to latent space (using real-valued projection on magnitude and phase)
        self._mu_proj = None
        self._logvar_proj = None

    def _init_projections(self, feature_size: int, device: torch.device) -> None:
        # Project magnitude, phase, and conditioning embedding
        combined_size = feature_size * 2 + self.snr_embedding_dim
        self._mu_proj = nn.Linear(combined_size, self.latent_dim).to(device)
        self._logvar_proj = nn.Linear(combined_size, self.latent_dim).to(device)

    def forward(
        self, x: Tensor, snr: Tensor | None = None, power: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        """Encode I/Q signal to complex latent space.

        Args:
            x: I/Q signal [batch, 2, seq_len] where x[:,0,:] is I and x[:,1,:] is Q.
            snr: Optional normalized SNR values [batch].
            power: Optional normalized power values [batch].

        Returns:
            Tuple of (mu, logvar) for the latent distribution.
        """
        # Split I/Q into real and imaginary parts
        x_real = x[:, 0:1, :]  # [batch, 1, seq_len]
        x_imag = x[:, 1:2, :]  # [batch, 1, seq_len]

        # Apply complex convolutions
        for conv_block in self.conv_layers:
            x_real, x_imag = conv_block(x_real, x_imag)

        # Flatten
        x_real_flat = x_real.flatten(1)
        x_imag_flat = x_imag.flatten(1)

        # Concatenate magnitude and phase features
        magnitude = torch.sqrt(x_real_flat ** 2 + x_imag_flat ** 2)
        phase = torch.atan2(x_imag_flat, x_real_flat)
        features = torch.cat([magnitude, phase], dim=1)

        # Add conditioning embedding
        if snr is not None:
            if snr.dim() == 1:
                snr = snr.unsqueeze(1)
            if self.use_power_conditioning and power is not None:
                if power.dim() == 1:
                    power = power.unsqueeze(1)
                cond = torch.cat([snr, power], dim=1)
            else:
                cond = snr
            cond_emb = self.cond_embed(cond)
            features = torch.cat([features, cond_emb], dim=1)
        else:
            # No conditioning - use zeros
            batch_size = x.size(0)
            cond_emb = torch.zeros(batch_size, self.snr_embedding_dim, device=x.device)
            features = torch.cat([features, cond_emb], dim=1)

        # Lazy initialization of projection layers
        if self._mu_proj is None:
            self._init_projections(x_real_flat.size(1), x.device)

        mu = self._mu_proj(features)
        logvar = self._logvar_proj(features)

        return mu, logvar


class ComplexVAE(nn.Module):
    """VAE with complex-valued encoder for phase-preserving RF processing.

    Uses complex convolutions in the encoder to naturally preserve phase
    information that would be lost with standard real-valued convolutions.
    The decoder remains real-valued as we ultimately reconstruct I/Q signals.

    Supports SNR and power conditioning like SNRConditionedVAE.
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
        snr_embedding_dim: int = 16,
        use_power_conditioning: bool = False,
    ):
        super().__init__()
        from .snr_encoder import SNRDecoder

        self.latent_dim = latent_dim
        self.sequence_length = sequence_length
        self.beta = beta
        self.use_power_conditioning = use_power_conditioning

        self.encoder = ComplexEncoder(
            hidden_channels=hidden_channels,
            latent_dim=latent_dim,
            kernel_size=kernel_size,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            snr_embedding_dim=snr_embedding_dim,
            use_power_conditioning=use_power_conditioning,
        )

        # Use standard real-valued decoder with conditioning
        self.decoder = SNRDecoder(
            latent_dim=latent_dim,
            hidden_channels=(hidden_channels or [32, 64, 128, 256])[::-1],
            out_channels=2,
            output_length=sequence_length,
            snr_embedding_dim=snr_embedding_dim,
            kernel_size=kernel_size,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            use_power_conditioning=use_power_conditioning,
            probabilistic=False,
        )

        # For compatibility with AnomalyDetector - mark as having cond_embed
        self.cond_embed = self.encoder.cond_embed

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        if self.training:
            return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return mu

    def forward(
        self, x: Tensor, snr: Tensor | None = None, power: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Forward pass with optional SNR and power conditioning.

        Args:
            x: I/Q signal [batch, 2, seq_len]
            snr: Optional normalized SNR values [batch].
            power: Optional normalized power values [batch].

        Returns:
            Tuple of (x_recon, mu, logvar, z)
        """
        mu, logvar = self.encoder(x, snr, power)
        z = self.reparameterize(mu, logvar)

        # Pass conditioning to decoder
        if snr is None:
            snr = torch.zeros(x.size(0), device=x.device)
        x_recon = self.decoder(z, snr, power if self.use_power_conditioning else None)

        return x_recon, mu, logvar, z

    def encode(self, x: Tensor, snr: Tensor | None = None, power: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Encode with optional conditioning."""
        return self.encoder(x, snr, power)

    def decode(self, z: Tensor, snr: Tensor | None = None, power: Tensor | None = None) -> Tensor:
        """Decode with optional conditioning."""
        if snr is None:
            snr = torch.zeros(z.size(0), device=z.device)
        return self.decoder(z, snr, power if self.use_power_conditioning else None)

    def loss(self, x: Tensor, x_recon: Tensor, mu: Tensor, logvar: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """Compute VAE loss."""
        recon_loss = F.mse_loss(x_recon, x)
        kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total_loss = recon_loss + self.beta * kl_loss
        return total_loss, recon_loss, kl_loss

    def get_reconstruction_error(self, x: Tensor, snr: Tensor | None = None, power: Tensor | None = None) -> Tensor:
        """Get per-sample reconstruction error."""
        x_recon, _, _, _ = self(x, snr, power)
        return ((x - x_recon) ** 2).mean(dim=(1, 2))
