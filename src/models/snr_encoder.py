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
    """VAE encoder with SNR and optional power conditioning.

    Optionally supports Bayesian last layers for epistemic uncertainty estimation.
    """

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
        use_bayesian: bool = False,
        bll_prior_std: float = 1.0,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels or [32, 64, 128, 256]
        self.latent_dim = latent_dim
        self.snr_embedding_dim = snr_embedding_dim
        self.use_power_conditioning = use_power_conditioning
        self.use_bayesian = use_bayesian
        self.bll_prior_std = bll_prior_std

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
        self._bayesian_proj = None  # For Bayesian last layer
        self._combined_size = None

    def _init_projections(self, combined_size: int, device: torch.device) -> None:
        """Initialize projection layers (lazy initialization)."""
        self._combined_size = combined_size

        if self.use_bayesian:
            from .bayesian import BayesianEncoder
            self._bayesian_proj = BayesianEncoder(
                combined_size=combined_size,
                latent_dim=self.latent_dim,
                prior_std=self.bll_prior_std,
            ).to(device)
        else:
            self._mu_proj = nn.Linear(combined_size, self.latent_dim).to(device)
            self._logvar_proj = nn.Linear(combined_size, self.latent_dim).to(device)

    def forward(
        self, x: Tensor, snr: Tensor, power: Tensor | None = None, sample: bool = True
    ) -> tuple[Tensor, Tensor]:
        """Encode input with SNR and optional power conditioning.

        Args:
            x: Input signal [batch, channels, seq_len].
            snr: Normalized SNR values [batch].
            power: Optional normalized power values [batch].
            sample: If True and using Bayesian layers, sample weights.

        Returns:
            Tuple of (mu, logvar) for the latent distribution.
        """
        cond = _combine_conditioning(snr, power if self.use_power_conditioning else None)
        cond_emb = self.cond_embed(cond)
        h = self.conv_layers(x).flatten(1)
        h_combined = torch.cat([h, cond_emb], dim=1)

        # Lazy initialization
        if self._mu_proj is None and self._bayesian_proj is None:
            self._init_projections(h_combined.size(1), x.device)

        if self.use_bayesian:
            return self._bayesian_proj(h_combined, sample=sample)
        return self._mu_proj(h_combined), self._logvar_proj(h_combined)

    def get_epistemic_uncertainty(
        self,
        x: Tensor,
        snr: Tensor,
        power: Tensor | None = None,
        num_samples: int = 10,
    ) -> Tensor:
        """Estimate epistemic uncertainty via Monte Carlo sampling.

        Only available when use_bayesian=True.

        Args:
            x: Input signal [batch, channels, seq_len].
            snr: Normalized SNR values [batch].
            power: Optional normalized power values [batch].
            num_samples: Number of MC samples for uncertainty estimation.

        Returns:
            Epistemic uncertainty [batch, latent_dim].
        """
        if not self.use_bayesian:
            # Return zeros for non-Bayesian encoders
            batch_size = x.size(0)
            return torch.zeros(batch_size, self.latent_dim, device=x.device)

        cond = _combine_conditioning(snr, power if self.use_power_conditioning else None)
        cond_emb = self.cond_embed(cond)
        h = self.conv_layers(x).flatten(1)
        h_combined = torch.cat([h, cond_emb], dim=1)

        # Lazy initialization if needed
        if self._bayesian_proj is None:
            self._init_projections(h_combined.size(1), x.device)

        return self._bayesian_proj.get_epistemic_uncertainty(h_combined, num_samples)

    def kl_divergence(self) -> Tensor:
        """Get KL divergence for Bayesian layers (for regularization).

        Returns:
            KL divergence scalar, or 0 if not using Bayesian layers.
        """
        if self.use_bayesian and self._bayesian_proj is not None:
            return self._bayesian_proj.kl_divergence()
        return torch.tensor(0.0)


class SNRDecoder(nn.Module):
    """VAE decoder with SNR and optional power conditioning.

    Supports probabilistic decoding where the decoder outputs both
    mean and log-variance for reconstruction uncertainty estimation.
    """

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
        probabilistic: bool = False,
    ):
        super().__init__()
        self.hidden_channels = hidden_channels or [256, 128, 64, 32]
        self.output_length = output_length
        self.use_power_conditioning = use_power_conditioning
        self.probabilistic = probabilistic

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

        # Final layer for mean
        self.final_mean = nn.ConvTranspose1d(
            channels[-1], out_channels, kernel_size, stride=2,
            padding=kernel_size // 2, output_padding=1
        )

        # Final layer for log-variance (only if probabilistic)
        if probabilistic:
            self.final_logvar = nn.ConvTranspose1d(
                channels[-1], out_channels, kernel_size, stride=2,
                padding=kernel_size // 2, output_padding=1
            )
            # Initialize logvar to small values for numerical stability
            nn.init.constant_(self.final_logvar.weight, 0.0)
            nn.init.constant_(self.final_logvar.bias, -3.0)  # exp(-3) ≈ 0.05 std

    def forward(self, z: Tensor, snr: Tensor, power: Tensor | None = None) -> Tensor | tuple[Tensor, Tensor]:
        """Decode latent sample with SNR and optional power conditioning.

        Args:
            z: Latent representation [batch, latent_dim].
            snr: Normalized SNR values [batch].
            power: Optional normalized power values [batch].

        Returns:
            If probabilistic=False: x_recon [batch, channels, seq_len]
            If probabilistic=True: (x_mean, x_logvar) tuple
        """
        cond = _combine_conditioning(snr, power if self.use_power_conditioning else None)
        cond_emb = self.cond_embed(cond)
        z_combined = torch.cat([z, cond_emb], dim=1)
        h = self.latent_proj(z_combined).view(-1, self._init_channels, self._init_length)
        h = self.conv_layers(h)

        # Compute mean
        x_mean = self.final_mean(h)
        if x_mean.size(2) != self.output_length:
            x_mean = nn.functional.interpolate(
                x_mean, size=self.output_length, mode="linear", align_corners=False
            )

        if not self.probabilistic:
            return x_mean

        # Compute log-variance for probabilistic decoding
        x_logvar = self.final_logvar(h)
        if x_logvar.size(2) != self.output_length:
            x_logvar = nn.functional.interpolate(
                x_logvar, size=self.output_length, mode="linear", align_corners=False
            )

        # Clamp logvar for numerical stability
        x_logvar = torch.clamp(x_logvar, min=-10.0, max=2.0)

        return x_mean, x_logvar


def _compute_phase_features(x: Tensor, eps: float = 1e-8) -> tuple[Tensor, Tensor, Tensor]:
    """Compute phase-based features from I/Q signal.

    Args:
        x: I/Q signal [batch, 2, seq_len] where x[:,0,:] is I and x[:,1,:] is Q.
        eps: Small value for numerical stability with near-zero amplitudes.

    Returns:
        Tuple of (phase, inst_freq, phase_variance):
        - phase: Unwrapped phase [batch, seq_len]
        - inst_freq: Instantaneous frequency (normalized to [-1, 1]) [batch, seq_len-1]
        - phase_variance: Per-sample phase variance [batch]
    """
    # Convert to complex signal
    complex_signal = torch.complex(x[:, 0, :], x[:, 1, :])

    # Compute amplitude for masking near-zero regions
    amplitude = torch.abs(complex_signal)

    # Compute phase (angle) - stable for near-zero amplitudes
    phase = torch.atan2(x[:, 1, :], x[:, 0, :] + eps)

    # Unwrap phase (approximate - PyTorch doesn't have unwrap)
    # Use diff to detect wraps and correct them
    phase_diff = phase[:, 1:] - phase[:, :-1]
    # Wrap diff to [-pi, pi]
    phase_diff = torch.remainder(phase_diff + torch.pi, 2 * torch.pi) - torch.pi

    # Cumsum to get unwrapped phase (starting from first value)
    phase_unwrapped = torch.cat([
        phase[:, :1],
        phase[:, :1] + torch.cumsum(phase_diff, dim=1)
    ], dim=1)

    # Instantaneous frequency (phase derivative), normalized to [-1, 1]
    # Original is in [-pi, pi], divide by pi to normalize
    inst_freq = phase_diff / torch.pi

    # Mask out near-zero amplitude regions where phase is unreliable
    amp_mask = (amplitude[:, :-1] > 0.01).float()
    inst_freq = inst_freq * amp_mask

    # Phase variance per sample
    phase_variance = torch.var(phase_unwrapped, dim=1)

    return phase_unwrapped, inst_freq, phase_variance


class SNRConditionedVAE(nn.Module):
    """SNR and Power-Conditioned Variational Autoencoder.

    This model conditions both encoder and decoder on SNR and optionally on
    signal power, allowing it to:
    - Learn SNR-dependent reconstruction patterns
    - Tolerate higher reconstruction error in low-SNR conditions
    - Be more sensitive to anomalies in high-SNR signals
    - Distinguish anomalies by their unusual power characteristics

    When probabilistic=True, the decoder outputs both mean and variance,
    enabling Gaussian NLL loss and reconstruction probability scoring.

    Phase-aware mode adds loss terms for phase and instantaneous frequency
    reconstruction, improving detection of frequency drift anomalies.

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
        probabilistic_decoder: bool = False,
        smoothness_lambda: float = 0.0,
        use_bayesian_encoder: bool = False,
        bll_prior_std: float = 1.0,
        bll_kl_weight: float = 1e-4,
        phase_loss_weight: float = 0.0,
        inst_freq_loss_weight: float = 0.0,
    ):
        super().__init__()
        hidden_channels = hidden_channels or [32, 64, 128, 256]
        self.latent_dim = latent_dim
        self.sequence_length = sequence_length
        self.beta = beta
        self.use_power_conditioning = use_power_conditioning
        self.probabilistic_decoder = probabilistic_decoder
        self.smoothness_lambda = smoothness_lambda
        self.use_bayesian_encoder = use_bayesian_encoder
        self.bll_kl_weight = bll_kl_weight
        self.phase_loss_weight = phase_loss_weight
        self.inst_freq_loss_weight = inst_freq_loss_weight

        self.encoder = SNREncoder(
            in_channels=2,
            hidden_channels=hidden_channels,
            latent_dim=latent_dim,
            snr_embedding_dim=snr_embedding_dim,
            kernel_size=kernel_size,
            use_batch_norm=use_batch_norm,
            dropout=dropout,
            use_power_conditioning=use_power_conditioning,
            use_bayesian=use_bayesian_encoder,
            bll_prior_std=bll_prior_std,
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
            probabilistic=probabilistic_decoder,
        )

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterization trick for sampling."""
        if self.training:
            return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
        return mu

    def forward(
        self, x: Tensor, snr: Tensor, power: Tensor | None = None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor] | tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Forward pass with SNR and optional power conditioning.

        Returns:
            If probabilistic_decoder=False:
                (x_recon, mu, logvar, z)
            If probabilistic_decoder=True:
                (x_recon_mean, x_recon_logvar, mu, logvar, z)
        """
        mu, logvar = self.encoder(x, snr, power)
        z = self.reparameterize(mu, logvar)
        decoder_out = self.decoder(z, snr, power)

        if self.probabilistic_decoder:
            x_mean, x_logvar = decoder_out
            return x_mean, x_logvar, mu, logvar, z
        return decoder_out, mu, logvar, z

    def encode(self, x: Tensor, snr: Tensor, power: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """Encode with SNR and optional power conditioning."""
        return self.encoder(x, snr, power)

    def decode(self, z: Tensor, snr: Tensor, power: Tensor | None = None) -> Tensor | tuple[Tensor, Tensor]:
        """Decode with SNR and optional power conditioning."""
        return self.decoder(z, snr, power)

    def reconstruction_loss(
        self, x: Tensor, x_recon: Tensor, x_logvar: Tensor | None = None
    ) -> Tensor:
        """Compute reconstruction loss.

        Args:
            x: Original input [batch, channels, seq_len].
            x_recon: Reconstruction mean [batch, channels, seq_len].
            x_logvar: Reconstruction log-variance (optional, for NLL loss).

        Returns:
            Reconstruction loss (MSE or Gaussian NLL).
        """
        if x_logvar is None:
            # Standard MSE loss
            return nn.functional.mse_loss(x_recon, x, reduction="mean")

        # Gaussian negative log-likelihood
        # NLL = 0.5 * (logvar + (x - mean)^2 / exp(logvar))
        var = torch.exp(x_logvar)
        nll = 0.5 * (x_logvar + (x - x_recon).pow(2) / var)
        return nll.mean()

    def smoothness_loss(self, x_mean: Tensor, x_logvar: Tensor) -> Tensor:
        """Compute smoothness prior loss (KL between adjacent time steps).

        Penalizes rapid changes in the reconstruction distribution to prevent
        overfitting to point anomalies like spikes and bursts.

        Args:
            x_mean: Reconstruction mean [batch, channels, seq_len].
            x_logvar: Reconstruction log-variance [batch, channels, seq_len].

        Returns:
            Smoothness loss (KL divergence between adjacent distributions).
        """
        if self.smoothness_lambda == 0.0:
            return torch.tensor(0.0, device=x_mean.device)

        # KL(N(μ_t, σ_t²) || N(μ_{t-1}, σ_{t-1}²)) between adjacent time steps
        mu_curr = x_mean[:, :, 1:]
        mu_prev = x_mean[:, :, :-1]
        logvar_curr = x_logvar[:, :, 1:]
        logvar_prev = x_logvar[:, :, :-1]

        var_curr = torch.exp(logvar_curr)
        var_prev = torch.exp(logvar_prev)

        # KL divergence: 0.5 * (var_ratio + mu_diff²/var_prev - 1 - log(var_ratio))
        var_ratio = var_curr / var_prev
        mu_diff_sq = (mu_curr - mu_prev).pow(2)
        kl = 0.5 * (var_ratio + mu_diff_sq / var_prev - 1 - torch.log(var_ratio))

        return kl.mean()

    def phase_loss(self, x: Tensor, x_recon: Tensor) -> tuple[Tensor, Tensor]:
        """Compute phase-sensitive loss terms.

        Forces the latent space to capture phase information by penalizing
        phase reconstruction error and instantaneous frequency error.

        Args:
            x: Original I/Q signal [batch, 2, seq_len].
            x_recon: Reconstructed I/Q signal [batch, 2, seq_len].

        Returns:
            Tuple of (phase_loss, inst_freq_loss).
        """
        if self.phase_loss_weight == 0.0 and self.inst_freq_loss_weight == 0.0:
            device = x.device
            return torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

        # Compute phase features for original and reconstruction
        phase_orig, inst_freq_orig, _ = _compute_phase_features(x)
        phase_recon, inst_freq_recon, _ = _compute_phase_features(x_recon)

        # Phase loss: circular distance (accounts for wrap-around)
        # Use cosine similarity for phase comparison
        phase_diff = phase_orig - phase_recon
        phase_cos_loss = 1.0 - torch.cos(phase_diff).mean()

        # Instantaneous frequency loss: MSE on frequency
        inst_freq_mse = nn.functional.mse_loss(inst_freq_recon, inst_freq_orig)

        return phase_cos_loss, inst_freq_mse

    def kl_divergence(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Compute KL divergence from standard normal."""
        from .vae import _compute_kl_divergence
        return _compute_kl_divergence(mu, logvar, reduce=True)

    def loss(
        self,
        x: Tensor,
        x_recon: Tensor,
        mu: Tensor,
        logvar: Tensor,
        x_recon_logvar: Tensor | None = None,
    ) -> tuple[Tensor, ...]:
        """Compute total VAE loss.

        Args:
            x: Original input.
            x_recon: Reconstruction mean.
            mu: Latent mean.
            logvar: Latent log-variance.
            x_recon_logvar: Reconstruction log-variance (for probabilistic decoder).

        Returns:
            Tuple containing (total_loss, recon_loss, kl_loss) plus optional:
            - smoothness_loss (if smoothness_lambda > 0)
            - phase_loss (if phase_loss_weight > 0)
            - inst_freq_loss (if inst_freq_loss_weight > 0)
        """
        recon_loss = self.reconstruction_loss(x, x_recon, x_recon_logvar)
        kl_loss = self.kl_divergence(mu, logvar)
        total_loss = recon_loss + self.beta * kl_loss

        # Add Bayesian encoder KL divergence if using BLL
        if self.use_bayesian_encoder:
            bll_kl = self.encoder.kl_divergence()
            total_loss = total_loss + self.bll_kl_weight * bll_kl

        # Track additional losses
        extra_losses = []

        # Smoothness loss
        if x_recon_logvar is not None and self.smoothness_lambda > 0:
            smooth_loss = self.smoothness_loss(x_recon, x_recon_logvar)
            total_loss = total_loss + self.smoothness_lambda * smooth_loss
            extra_losses.append(smooth_loss)

        # Phase-aware losses
        if self.phase_loss_weight > 0 or self.inst_freq_loss_weight > 0:
            phase_l, inst_freq_l = self.phase_loss(x, x_recon)
            total_loss = total_loss + self.phase_loss_weight * phase_l
            total_loss = total_loss + self.inst_freq_loss_weight * inst_freq_l
            extra_losses.extend([phase_l, inst_freq_l])

        if extra_losses:
            return (total_loss, recon_loss, kl_loss, *extra_losses)
        return total_loss, recon_loss, kl_loss

    def get_reconstruction_error(self, x: Tensor, snr: Tensor, power: Tensor | None = None) -> Tensor:
        """Get per-sample reconstruction error."""
        if self.probabilistic_decoder:
            x_mean, x_logvar, _, _, _ = self(x, snr, power)
            # Return NLL as reconstruction error
            var = torch.exp(x_logvar)
            nll = 0.5 * (x_logvar + (x - x_mean).pow(2) / var)
            return nll.mean(dim=(1, 2))
        else:
            x_recon, _, _, _ = self(x, snr, power)
            return ((x - x_recon) ** 2).mean(dim=(1, 2))

    def get_anomaly_score(
        self,
        x: Tensor,
        snr: Tensor,
        power: Tensor | None = None,
        include_kl: bool = True,
        num_samples: int = 1,
        scoring_method: str = "auto",
    ) -> Tensor:
        """Compute anomaly score with SNR and optional power conditioning.

        Args:
            x: Input signals [batch, 2, seq_len].
            snr: Normalized SNR values [batch].
            power: Optional normalized power values [batch].
            include_kl: Whether to include KL term in score.
            num_samples: Number of Monte Carlo samples.
            scoring_method: "mse", "nll", or "auto" (uses NLL if probabilistic).

        Returns:
            Anomaly scores [batch].
        """
        mu, logvar = self.encoder(x, snr, power)
        use_nll = scoring_method == "nll" or (scoring_method == "auto" and self.probabilistic_decoder)

        # Compute reconstruction error with optional Monte Carlo sampling
        if num_samples == 1:
            z = self.reparameterize(mu, logvar)
            decoder_out = self.decoder(z, snr, power)

            if use_nll and self.probabilistic_decoder:
                x_mean, x_logvar = decoder_out
                var = torch.exp(x_logvar)
                recon_error = 0.5 * (x_logvar + (x - x_mean).pow(2) / var).mean(dim=(1, 2))
            else:
                x_recon = decoder_out if not self.probabilistic_decoder else decoder_out[0]
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
                decoder_out = self.decoder(z, snr_expanded.squeeze(1), power_expanded.squeeze(1))
            else:
                decoder_out = self.decoder(z, snr_expanded.squeeze(1), None)

            if use_nll and self.probabilistic_decoder:
                x_mean, x_logvar = decoder_out
                var = torch.exp(x_logvar)
                recon_error = 0.5 * (x_logvar + (x_expanded - x_mean).pow(2) / var).mean(dim=(1, 2))
            else:
                x_recon = decoder_out if not self.probabilistic_decoder else decoder_out[0]
                recon_error = ((x_expanded - x_recon) ** 2).mean(dim=(1, 2))

            recon_error = recon_error.view(batch_size, num_samples).mean(dim=1)

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
        probabilistic = getattr(config.model, 'probabilistic_decoder', False)
        smoothness_lambda = getattr(config.model, 'smoothness_lambda', 0.0)
        use_bayesian = getattr(config.model, 'use_bayesian_encoder', False)
        bll_prior_std = getattr(config.model, 'bll_prior_std', 1.0)
        bll_kl_weight = getattr(config.model, 'bll_kl_weight', 1e-4)
        phase_loss_weight = getattr(config.model, 'phase_loss_weight', 0.0)
        inst_freq_loss_weight = getattr(config.model, 'inst_freq_loss_weight', 0.0)
        return SNRConditionedVAE(
            **common_args,
            snr_embedding_dim=config.model.snr_embedding_dim,
            beta=config.model.beta,
            use_power_conditioning=use_power,
            probabilistic_decoder=probabilistic,
            smoothness_lambda=smoothness_lambda,
            use_bayesian_encoder=use_bayesian,
            bll_prior_std=bll_prior_std,
            bll_kl_weight=bll_kl_weight,
            phase_loss_weight=phase_loss_weight,
            inst_freq_loss_weight=inst_freq_loss_weight,
        )

    raise ValueError(f"Unknown model type: {model_type}")
