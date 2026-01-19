"""Uncertainty-based Continual Learning (UCL) for preventing catastrophic forgetting.

UCL uses posterior variance from Bayesian neural networks to determine weight
importance. Uncertain weights (high variance) can adapt freely, while certain
weights (low variance) are protected from changes.

This approach naturally integrates with Bayesian Last Layers and provides a
theoretically grounded alternative to EWC (Elastic Weight Consolidation).

References:
    - Ahn et al., "Uncertainty-based Continual Learning with Adaptive
      Regularization" (NeurIPS 2019)
    - Nguyen et al., "Variational Continual Learning" (ICLR 2018)
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader

from ..models.bayesian import BayesianLinear, collect_kl_divergence


def _get_device(model: nn.Module, device: torch.device | str | None) -> torch.device:
    """Get device, defaulting to model's device."""
    if device is None:
        return next(model.parameters()).device
    return torch.device(device) if isinstance(device, str) else device


def _detect_model_type(model: nn.Module) -> tuple[bool, bool, bool]:
    """Detect if model is SNR-conditioned, VAE, and/or has Bayesian layers.

    Returns:
        Tuple of (is_snr_conditioned, is_vae, has_bayesian).
    """
    is_snr = hasattr(model, "encoder") and hasattr(model.encoder, "cond_embed")
    is_vae = hasattr(model, "reparameterize")
    has_bayesian = any(isinstance(m, BayesianLinear) for m in model.modules())
    return is_snr, is_vae, has_bayesian


class UCLLearner:
    """Uncertainty-based Continual Learning.

    UCL prevents catastrophic forgetting by using the posterior variance from
    Bayesian layers to determine weight importance. Weights with low variance
    (high certainty) are considered important and protected from changes.

    Unlike EWC which estimates importance via Fisher information after training,
    UCL directly uses the learned variance from Bayesian inference, providing
    a more natural integration with probabilistic models.

    Example:
        ucl = UCLLearner(model, ucl_lambda=100.0)
        ucl.snapshot()  # Save current parameters after initial training

        # During continuous learning
        for batch in new_data:
            loss = compute_loss(batch) + ucl.penalty()
            loss.backward()
            optimizer.step()

        # Periodically update the snapshot
        if should_consolidate:
            ucl.update_importance()
            ucl.snapshot()
    """

    def __init__(
        self,
        model: nn.Module,
        ucl_lambda: float = 100.0,
        min_variance: float = 1e-6,
        online: bool = False,
        gamma: float = 0.95,
        device: torch.device | str | None = None,
    ):
        """Initialize UCL learner.

        Args:
            model: Model with Bayesian layers to protect from forgetting.
            ucl_lambda: Regularization strength for UCL penalty.
            min_variance: Minimum variance to prevent division by zero.
            online: Use online UCL (running average of importance).
            gamma: Decay factor for online UCL.
            device: Device for computation.
        """
        self.model = model
        self.ucl_lambda = ucl_lambda
        self.min_variance = min_variance
        self.online = online
        self.gamma = gamma
        self.device = _get_device(model, device)

        # Storage for importance and parameter snapshots
        self._importance: Dict[str, Tensor] = {}
        self._params_snapshot: Dict[str, Tensor] = {}
        self._is_initialized = False
        self._is_snr_conditioned, self._is_vae, self._has_bayesian = _detect_model_type(model)

        if not self._has_bayesian:
            import warnings
            warnings.warn(
                "UCL is designed for models with Bayesian layers. "
                "For non-Bayesian models, consider using EWC instead."
            )

    def compute_importance(self) -> Dict[str, Tensor]:
        """Compute weight importance from Bayesian posterior variance.

        For Bayesian layers, importance = 1 / variance (certain weights are important).
        For non-Bayesian layers, importance is set to 1 (uniform importance).

        Returns:
            Dictionary mapping parameter names to importance tensors.
        """
        importance = {}

        for name, module in self.model.named_modules():
            if isinstance(module, BayesianLinear):
                # For Bayesian layers, importance = 1 / variance
                # Certain weights (low variance) are important
                weight_var = torch.exp(module.weight_logvar)
                importance[f"{name}.weight_mean"] = 1.0 / (weight_var + self.min_variance)

                if module.use_bias:
                    bias_var = torch.exp(module.bias_logvar)
                    importance[f"{name}.bias_mean"] = 1.0 / (bias_var + self.min_variance)

        # For non-Bayesian parameters, use uniform importance
        for name, param in self.model.named_parameters():
            if param.requires_grad and name not in importance:
                # Skip variance parameters of Bayesian layers
                if "_logvar" not in name:
                    importance[name] = torch.ones_like(param)

        return importance

    def snapshot(self) -> None:
        """Save current parameters and compute importance.

        Call this after training on a task to establish which parameters
        are important and should be protected.
        """
        # Compute importance from posterior variance
        new_importance = self.compute_importance()

        # Update importance (online UCL uses running average)
        if self.online and self._is_initialized:
            for name in self._importance:
                if name in new_importance:
                    self._importance[name] = (
                        self.gamma * self._importance[name]
                        + (1 - self.gamma) * new_importance[name]
                    )
            # Add any new parameters
            for name in new_importance:
                if name not in self._importance:
                    self._importance[name] = new_importance[name]
        else:
            self._importance = new_importance

        # Save parameter snapshot
        self._params_snapshot = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad and "_logvar" not in name:
                self._params_snapshot[name] = param.data.clone()

        self._is_initialized = True

    def update_importance(self) -> None:
        """Update importance without taking a new snapshot.

        Useful for periodically updating importance during training
        without resetting the reference parameters.
        """
        new_importance = self.compute_importance()

        if self._is_initialized:
            for name in self._importance:
                if name in new_importance:
                    self._importance[name] = (
                        self.gamma * self._importance[name]
                        + (1 - self.gamma) * new_importance[name]
                    )
        else:
            self._importance = new_importance

    def penalty(self) -> Tensor:
        """Compute UCL penalty term to add to the loss.

        Penalizes changes to important (certain) weights while allowing
        uncertain weights to adapt freely.

        Returns:
            UCL regularization penalty.
        """
        if not self._is_initialized:
            return torch.tensor(0.0, device=self.device)

        penalty = torch.tensor(0.0, device=self.device)

        for name, param in self.model.named_parameters():
            if (
                param.requires_grad
                and name in self._importance
                and name in self._params_snapshot
            ):
                # Weighted L2 penalty: importance * (w - w_old)^2
                diff = param - self._params_snapshot[name].to(param.device)
                imp = self._importance[name].to(param.device)
                penalty = penalty + (imp * diff.pow(2)).sum()

        return self.ucl_lambda * penalty / 2

    def get_importance_stats(self) -> dict:
        """Get statistics about weight importance.

        Returns:
            Dictionary with importance statistics.
        """
        if not self._is_initialized:
            return {"initialized": False}

        stats = {"initialized": True, "parameters": {}}

        for name, importance in self._importance.items():
            stats["parameters"][name] = {
                "mean": float(importance.mean()),
                "std": float(importance.std()),
                "min": float(importance.min()),
                "max": float(importance.max()),
                "high_importance_ratio": float((importance > importance.mean()).float().mean()),
            }

        return stats

    def get_state(self) -> dict:
        """Get state for checkpointing.

        Returns:
            State dictionary.
        """
        return {
            "importance": {k: v.cpu() for k, v in self._importance.items()},
            "params_snapshot": {k: v.cpu() for k, v in self._params_snapshot.items()},
            "is_initialized": self._is_initialized,
        }

    def load_state(self, state: dict) -> None:
        """Load state from checkpoint.

        Args:
            state: State dictionary.
        """
        self._importance = {k: v.to(self.device) for k, v in state["importance"].items()}
        self._params_snapshot = {
            k: v.to(self.device) for k, v in state["params_snapshot"].items()
        }
        self._is_initialized = state["is_initialized"]


class UCLTrainer:
    """Training loop with UCL regularization.

    Combines standard training with UCL penalty for continual learning.
    """

    def __init__(
        self,
        model: nn.Module,
        ucl: UCLLearner,
        optimizer: torch.optim.Optimizer,
        gradient_clip_norm: float | None = 1.0,
        include_bll_kl: bool = True,
        bll_kl_weight: float = 1e-4,
        device: torch.device | str | None = None,
    ):
        """Initialize trainer.

        Args:
            model: Model to train.
            ucl: UCLLearner instance.
            optimizer: Optimizer for training.
            gradient_clip_norm: Gradient clipping threshold.
            include_bll_kl: Include KL divergence from Bayesian layers in loss.
            bll_kl_weight: Weight for Bayesian layer KL divergence.
            device: Training device.
        """
        self.model = model
        self.ucl = ucl
        self.optimizer = optimizer
        self.gradient_clip_norm = gradient_clip_norm
        self.include_bll_kl = include_bll_kl
        self.bll_kl_weight = bll_kl_weight
        self.device = _get_device(model, device)
        self._is_snr_conditioned, self._is_vae, self._has_bayesian = _detect_model_type(model)

    def train_step(self, batch: dict[str, Tensor]) -> dict[str, float]:
        """Perform single training step with UCL.

        Args:
            batch: Training batch with 'iq', optional 'snr', 'power'.

        Returns:
            Dictionary with loss metrics.
        """
        self.model.train()

        iq = batch["iq"].to(self.device)
        snr = batch.get("snr")
        power = batch.get("power")
        snr = snr.to(self.device) if snr is not None else None
        power = power.to(self.device) if power is not None else None

        self.optimizer.zero_grad()

        # Compute task loss
        if self._is_vae:
            if self._is_snr_conditioned and snr is not None:
                out = self.model(iq, snr, power) if power is not None else self.model(iq, snr)
            else:
                out = self.model(iq)

            # Handle probabilistic decoder output
            if hasattr(self.model, 'probabilistic_decoder') and self.model.probabilistic_decoder:
                x_mean, x_logvar, mu, logvar, _ = out
                loss_out = self.model.loss(iq, x_mean, mu, logvar, x_logvar)
            else:
                x_recon, mu, logvar, _ = out
                loss_out = self.model.loss(iq, x_recon, mu, logvar)

            task_loss = loss_out[0]
            recon_loss = loss_out[1]
            kl_loss = loss_out[2]
        else:
            x_recon, _ = self.model(iq)
            task_loss = recon_loss = self.model.reconstruction_loss(iq, x_recon)
            kl_loss = torch.tensor(0.0)

        # Add Bayesian layer KL if requested
        bll_kl = torch.tensor(0.0)
        if self.include_bll_kl and self._has_bayesian:
            bll_kl = collect_kl_divergence(self.model)
            task_loss = task_loss + self.bll_kl_weight * bll_kl

        # Add UCL penalty
        ucl_penalty = self.ucl.penalty()
        total_loss = task_loss + ucl_penalty
        total_loss.backward()

        if self.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)

        self.optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "task_loss": task_loss.item(),
            "ucl_penalty": ucl_penalty.item(),
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item() if isinstance(kl_loss, Tensor) else kl_loss,
            "bll_kl": bll_kl.item() if isinstance(bll_kl, Tensor) else bll_kl,
        }

    def train_epoch(self, dataloader: DataLoader, log_interval: int = 100) -> dict[str, float]:
        """Train for one epoch and return average metrics.

        Args:
            dataloader: Training data loader.
            log_interval: Log every N batches (not used, for compatibility).

        Returns:
            Dictionary with average metrics.
        """
        metrics_sum = {}
        batch_count = 0

        for batch in dataloader:
            batch_metrics = self.train_step(batch)
            batch_count += 1
            for key, value in batch_metrics.items():
                metrics_sum[key] = metrics_sum.get(key, 0.0) + value

        return {k: v / batch_count for k, v in metrics_sum.items()}
