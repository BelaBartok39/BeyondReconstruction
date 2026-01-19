"""Bayesian neural network layers for epistemic uncertainty estimation.

This module provides Bayesian linear layers using Pyro for probabilistic inference.
These layers learn a distribution over weights rather than point estimates,
enabling epistemic uncertainty quantification.

References:
    - Blundell et al., "Weight Uncertainty in Neural Networks" (2015)
    - Louizos & Welling, "Multiplicative Normalizing Flows for Variational
      Bayesian Neural Networks" (2017)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

try:
    import pyro
    import pyro.distributions as dist
    from pyro.nn import PyroModule, PyroSample
    PYRO_AVAILABLE = True
except ImportError:
    PYRO_AVAILABLE = False
    PyroModule = nn.Module
    PyroSample = None


class BayesianLinear(nn.Module):
    """Bayesian linear layer with weight uncertainty.

    Implements variational inference over weights using the local reparameterization
    trick for efficient sampling during training.

    This layer maintains a posterior distribution q(w) = N(w_mean, w_var) over
    weights and biases, and uses KL divergence against a prior p(w) = N(0, prior_std^2)
    for regularization.

    Args:
        in_features: Size of input features.
        out_features: Size of output features.
        prior_std: Standard deviation of the prior distribution (default: 1.0).
        bias: If True, adds a learnable bias (default: True).

    Example:
        layer = BayesianLinear(64, 32, prior_std=1.0)
        x = torch.randn(16, 64)

        # Training mode: samples weights
        y = layer(x, sample=True)
        kl = layer.kl_divergence()

        # Inference mode: uses mean weights
        y_mean = layer(x, sample=False)

        # Epistemic uncertainty via multiple forward passes
        ys = [layer(x, sample=True) for _ in range(10)]
        uncertainty = torch.stack(ys).var(dim=0)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        prior_std: float = 1.0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std
        self.use_bias = bias

        # Weight posterior parameters
        self.weight_mean = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_logvar = nn.Parameter(torch.empty(out_features, in_features))

        # Bias posterior parameters
        if bias:
            self.bias_mean = nn.Parameter(torch.empty(out_features))
            self.bias_logvar = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias_mean", None)
            self.register_parameter("bias_logvar", None)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize parameters."""
        # Initialize weight mean using Kaiming uniform
        nn.init.kaiming_uniform_(self.weight_mean, a=math.sqrt(5))
        # Initialize weight logvar to small values (exp(-5) ≈ 0.007 std)
        nn.init.constant_(self.weight_logvar, -5.0)

        if self.use_bias:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight_mean)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias_mean, -bound, bound)
            nn.init.constant_(self.bias_logvar, -5.0)

    def forward(self, x: Tensor, sample: bool = True) -> Tensor:
        """Forward pass with optional weight sampling.

        Args:
            x: Input tensor [batch, in_features].
            sample: If True, sample weights from posterior; if False, use mean.

        Returns:
            Output tensor [batch, out_features].
        """
        if sample and self.training:
            # Sample weights using reparameterization trick
            weight_std = torch.exp(0.5 * self.weight_logvar)
            weight = self.weight_mean + torch.randn_like(self.weight_mean) * weight_std

            if self.use_bias:
                bias_std = torch.exp(0.5 * self.bias_logvar)
                bias = self.bias_mean + torch.randn_like(self.bias_mean) * bias_std
            else:
                bias = None
        else:
            # Use mean weights (MAP estimate)
            weight = self.weight_mean
            bias = self.bias_mean if self.use_bias else None

        return F.linear(x, weight, bias)

    def kl_divergence(self) -> Tensor:
        """Compute KL divergence KL(q(w) || p(w)) for regularization.

        Assumes prior p(w) = N(0, prior_std^2).

        Returns:
            KL divergence scalar.
        """
        prior_var = self.prior_std ** 2

        # KL for weights: KL(N(μ, σ²) || N(0, prior_var))
        # = 0.5 * (σ²/prior_var + μ²/prior_var - 1 - log(σ²/prior_var))
        weight_var = torch.exp(self.weight_logvar)
        kl_weight = 0.5 * (
            weight_var / prior_var
            + self.weight_mean.pow(2) / prior_var
            - 1
            - self.weight_logvar
            + math.log(prior_var)
        ).sum()

        if self.use_bias:
            bias_var = torch.exp(self.bias_logvar)
            kl_bias = 0.5 * (
                bias_var / prior_var
                + self.bias_mean.pow(2) / prior_var
                - 1
                - self.bias_logvar
                + math.log(prior_var)
            ).sum()
            return kl_weight + kl_bias

        return kl_weight

    @property
    def weight_uncertainty(self) -> Tensor:
        """Get weight uncertainty (standard deviation)."""
        return torch.exp(0.5 * self.weight_logvar)

    @property
    def bias_uncertainty(self) -> Optional[Tensor]:
        """Get bias uncertainty (standard deviation)."""
        if self.use_bias:
            return torch.exp(0.5 * self.bias_logvar)
        return None

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"prior_std={self.prior_std}, bias={self.use_bias}"
        )


if PYRO_AVAILABLE:
    class PyroBayesianLinear(PyroModule):
        """Bayesian linear layer using Pyro for probabilistic inference.

        Uses Pyro's automatic guide generation for variational inference.
        Compatible with Pyro's SVI training loop.

        Args:
            in_features: Size of input features.
            out_features: Size of output features.
            prior_std: Standard deviation of the prior distribution.
            bias: If True, adds a learnable bias.
        """

        def __init__(
            self,
            in_features: int,
            out_features: int,
            prior_std: float = 1.0,
            bias: bool = True,
        ):
            super().__init__()
            self.in_features = in_features
            self.out_features = out_features
            self.prior_std = prior_std
            self.use_bias = bias

            # Register weight as a Pyro random variable
            self.weight = PyroSample(
                dist.Normal(
                    torch.zeros(out_features, in_features),
                    prior_std * torch.ones(out_features, in_features),
                ).to_event(2)
            )

            if bias:
                self.bias = PyroSample(
                    dist.Normal(
                        torch.zeros(out_features),
                        prior_std * torch.ones(out_features),
                    ).to_event(1)
                )
            else:
                self.bias = None

        def forward(self, x: Tensor) -> Tensor:
            """Forward pass samples weights from the posterior during training."""
            bias = self.bias if self.use_bias else None
            return F.linear(x, self.weight, bias)


class BayesianEncoder(nn.Module):
    """Wrapper that adds Bayesian last layers to an encoder.

    Replaces the final projection layers (mu and logvar) with Bayesian linear
    layers to capture epistemic uncertainty in the latent space.

    Args:
        base_encoder: The base encoder module (e.g., SNREncoder without projections).
        combined_size: Size of the combined features before projection.
        latent_dim: Dimension of the latent space.
        prior_std: Prior standard deviation for Bayesian layers.
    """

    def __init__(
        self,
        combined_size: int,
        latent_dim: int,
        prior_std: float = 1.0,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.prior_std = prior_std

        # Bayesian projection layers
        self.mu_proj = BayesianLinear(combined_size, latent_dim, prior_std=prior_std)
        self.logvar_proj = BayesianLinear(combined_size, latent_dim, prior_std=prior_std)

    def forward(self, h_combined: Tensor, sample: bool = True) -> tuple[Tensor, Tensor]:
        """Project combined features to latent distribution parameters.

        Args:
            h_combined: Combined encoder output [batch, combined_size].
            sample: Whether to sample weights (True during training).

        Returns:
            Tuple of (mu, logvar) for the latent distribution.
        """
        mu = self.mu_proj(h_combined, sample=sample)
        logvar = self.logvar_proj(h_combined, sample=sample)
        return mu, logvar

    def kl_divergence(self) -> Tensor:
        """Total KL divergence for both projection layers."""
        return self.mu_proj.kl_divergence() + self.logvar_proj.kl_divergence()

    def get_epistemic_uncertainty(
        self,
        h_combined: Tensor,
        num_samples: int = 10,
    ) -> Tensor:
        """Estimate epistemic uncertainty via Monte Carlo sampling.

        Performs multiple forward passes with sampled weights and measures
        the variance in the outputs.

        Args:
            h_combined: Combined encoder output [batch, combined_size].
            num_samples: Number of Monte Carlo samples.

        Returns:
            Epistemic uncertainty per sample [batch, latent_dim].
        """
        was_training = self.training
        self.train()  # Enable sampling

        mus = []
        for _ in range(num_samples):
            mu, _ = self.forward(h_combined, sample=True)
            mus.append(mu)

        # Restore training mode
        self.train(was_training)

        # Variance across samples represents epistemic uncertainty
        return torch.stack(mus).var(dim=0)

    @property
    def weight_uncertainty_stats(self) -> dict:
        """Get statistics about weight uncertainties."""
        return {
            "mu_proj_weight_std": self.mu_proj.weight_uncertainty.mean().item(),
            "mu_proj_weight_std_max": self.mu_proj.weight_uncertainty.max().item(),
            "logvar_proj_weight_std": self.logvar_proj.weight_uncertainty.mean().item(),
            "logvar_proj_weight_std_max": self.logvar_proj.weight_uncertainty.max().item(),
        }


def collect_kl_divergence(model: nn.Module) -> Tensor:
    """Collect KL divergence from all BayesianLinear layers in a model.

    Args:
        model: Model containing BayesianLinear layers.

    Returns:
        Total KL divergence.
    """
    total_kl = torch.tensor(0.0)
    for module in model.modules():
        if isinstance(module, BayesianLinear):
            total_kl = total_kl + module.kl_divergence()
    return total_kl
