"""Elastic Weight Consolidation for preventing catastrophic forgetting."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader


def _get_device(model: nn.Module, device: torch.device | str | None) -> torch.device:
    """Get device, defaulting to model's device."""
    if device is None:
        return next(model.parameters()).device
    return torch.device(device) if isinstance(device, str) else device


def _detect_model_type(model: nn.Module) -> tuple[bool, bool]:
    """Detect if model is SNR-conditioned and/or VAE.

    Returns:
        Tuple of (is_snr_conditioned, is_vae).
    """
    # SNRConditionedVAE uses cond_embed in encoder for conditioning
    is_snr = hasattr(model, "encoder") and hasattr(model.encoder, "cond_embed")
    is_vae = hasattr(model, "reparameterize")
    return is_snr, is_vae


class EWCLearner:
    """Elastic Weight Consolidation (EWC) for continual learning.

    EWC prevents catastrophic forgetting by adding a penalty term
    that discourages changes to parameters important for previous tasks.

    Reference:
        Kirkpatrick et al., "Overcoming catastrophic forgetting in neural networks"
        PNAS, 2017.

    Example:
        ewc = EWCLearner(model, ewc_lambda=1000)
        ewc.compute_fisher(train_loader)  # After initial training

        # During continuous learning
        for batch in new_data:
            loss = compute_loss(batch) + ewc.penalty()
            loss.backward()
            optimizer.step()
    """

    def __init__(
        self,
        model: nn.Module,
        ewc_lambda: float = 1000.0,
        fisher_samples: int = 1000,
        online: bool = False,
        gamma: float = 0.95,
        device: torch.device | str | None = None,
    ):
        """Initialize EWC learner.

        Args:
            model: Model to protect from forgetting.
            ewc_lambda: Regularization strength for EWC penalty.
            fisher_samples: Number of samples for Fisher information estimation.
            online: Use online EWC (running average of Fisher).
            gamma: Decay factor for online EWC.
            device: Device for computation.
        """
        self.model = model
        self.ewc_lambda = ewc_lambda
        self.fisher_samples = fisher_samples
        self.online = online
        self.gamma = gamma
        self.device = _get_device(model, device)

        # Storage for Fisher information and parameter snapshots
        self._fisher: dict[str, Tensor] = {}
        self._params_snapshot: dict[str, Tensor] = {}
        self._is_initialized = False
        self._is_snr_conditioned, self._is_vae = _detect_model_type(model)

    def compute_fisher(
        self,
        dataloader: DataLoader,
        num_samples: int | None = None,
    ) -> None:
        """Compute Fisher information matrix diagonal.

        Should be called after training on a task to establish
        which parameters are important.

        Args:
            dataloader: DataLoader with representative samples.
            num_samples: Max samples to use (default: fisher_samples).
        """
        num_samples = num_samples or self.fisher_samples
        self.model.eval()

        # Do a dummy forward pass to initialize any lazy layers
        first_batch = next(iter(dataloader))
        dummy_iq = first_batch["iq"][:1].to(self.device)
        dummy_snr = first_batch.get("snr")
        dummy_snr = dummy_snr[:1].to(self.device) if dummy_snr is not None else None
        dummy_power = first_batch.get("power")
        dummy_power = dummy_power[:1].to(self.device) if dummy_power is not None else None
        with torch.no_grad():
            _ = self._compute_sample_loss(dummy_iq, dummy_snr, dummy_power)

        # Initialize Fisher accumulator (after lazy layers are initialized)
        fisher_accum = {name: torch.zeros_like(param) for name, param in self._named_parameters()}
        sample_count = 0

        for batch in dataloader:
            if sample_count >= num_samples:
                break

            iq = batch["iq"].to(self.device)
            snr = batch.get("snr")
            snr = snr.to(self.device) if snr is not None else None
            power = batch.get("power")
            power = power.to(self.device) if power is not None else None

            self.model.zero_grad()
            loss = self._compute_sample_loss(iq, snr, power)
            loss.backward()

            # Accumulate squared gradients (Fisher diagonal approximation)
            batch_size = iq.size(0)
            for name, param in self._named_parameters():
                if param.grad is not None:
                    fisher_accum[name] += param.grad.data.pow(2) * batch_size

            sample_count += batch_size

        # Normalize Fisher values
        for name in fisher_accum:
            fisher_accum[name] /= sample_count

        # Update Fisher (online EWC uses running average, otherwise replace)
        if self.online and self._is_initialized:
            for name in self._fisher:
                self._fisher[name] = self.gamma * self._fisher[name] + (1 - self.gamma) * fisher_accum[name]
        else:
            self._fisher = fisher_accum

        # Save parameter snapshot
        self._params_snapshot = {name: param.data.clone() for name, param in self._named_parameters()}
        self._is_initialized = True

    def _compute_sample_loss(self, iq: Tensor, snr: Tensor | None, power: Tensor | None = None) -> Tensor:
        """Compute loss for Fisher estimation (reconstruction loss only, not KL)."""
        if self._is_vae:
            if self._is_snr_conditioned:
                # SNR-conditioned models always need SNR and power
                if snr is None:
                    snr = torch.full((iq.size(0),), 0.5, device=iq.device)
                if power is None:
                    power = torch.full((iq.size(0),), 0.5, device=iq.device)
                x_recon, *_ = self.model(iq, snr, power)
            else:
                x_recon, *_ = self.model(iq)
        else:
            x_recon, _ = self.model(iq)
        return nn.functional.mse_loss(x_recon, iq, reduction="sum")

    def penalty(self) -> Tensor:
        """Compute EWC penalty term to add to the loss."""
        if not self._is_initialized:
            return torch.tensor(0.0, device=self.device)

        penalty = sum(
            (self._fisher[name] * (param - self._params_snapshot[name]).pow(2)).sum()
            for name, param in self._named_parameters()
            if name in self._fisher and name in self._params_snapshot
        )
        return self.ewc_lambda * penalty / 2

    def _named_parameters(self):
        """Get named parameters that require gradients."""
        return ((n, p) for n, p in self.model.named_parameters() if p.requires_grad)

    def get_fisher_stats(self) -> dict:
        """Get statistics about Fisher information."""
        if not self._is_initialized:
            return {"initialized": False}

        return {
            "initialized": True,
            "parameters": {
                name: {
                    "mean": float(fisher.mean()),
                    "std": float(fisher.std()),
                    "max": float(fisher.max()),
                    "sparsity": float((fisher < 1e-6).float().mean()),
                }
                for name, fisher in self._fisher.items()
            },
        }

    def get_state(self) -> dict:
        """Get state for checkpointing.

        Returns:
            State dictionary.
        """
        return {
            "fisher": {k: v.cpu() for k, v in self._fisher.items()},
            "params_snapshot": {k: v.cpu() for k, v in self._params_snapshot.items()},
            "is_initialized": self._is_initialized,
        }

    def load_state(self, state: dict) -> None:
        """Load state from checkpoint.

        Args:
            state: State dictionary.
        """
        self._fisher = {k: v.to(self.device) for k, v in state["fisher"].items()}
        self._params_snapshot = {
            k: v.to(self.device) for k, v in state["params_snapshot"].items()
        }
        self._is_initialized = state["is_initialized"]


class EWCTrainer:
    """Training loop with EWC regularization.

    Combines standard training with EWC penalty for continual learning.
    """

    def __init__(
        self,
        model: nn.Module,
        ewc: EWCLearner,
        optimizer: torch.optim.Optimizer,
        gradient_clip_norm: float | None = 1.0,
        device: torch.device | str | None = None,
    ):
        """Initialize trainer.

        Args:
            model: Model to train.
            ewc: EWCLearner instance.
            optimizer: Optimizer for training.
            gradient_clip_norm: Gradient clipping threshold.
            device: Training device.
        """
        self.model = model
        self.ewc = ewc
        self.optimizer = optimizer
        self.gradient_clip_norm = gradient_clip_norm
        self.device = _get_device(model, device)
        self._is_snr_conditioned, self._is_vae = _detect_model_type(model)

    def train_step(self, batch: dict[str, Tensor]) -> dict[str, float]:
        """Perform single training step with EWC."""
        self.model.train()

        iq = batch["iq"].to(self.device)
        snr = batch.get("snr")
        snr = snr.to(self.device) if snr is not None else None

        self.optimizer.zero_grad()

        # Compute task loss
        if self._is_vae:
            x_recon, mu, logvar, _ = self.model(iq, snr) if self._is_snr_conditioned and snr is not None else self.model(iq)
            task_loss, recon_loss, kl_loss = self.model.loss(iq, x_recon, mu, logvar)
        else:
            x_recon, _ = self.model(iq)
            task_loss = recon_loss = self.model.reconstruction_loss(iq, x_recon)
            kl_loss = 0.0

        # Add EWC penalty
        ewc_penalty = self.ewc.penalty()
        total_loss = task_loss + ewc_penalty
        total_loss.backward()

        if self.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)

        self.optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "task_loss": task_loss.item(),
            "ewc_penalty": ewc_penalty.item(),
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item() if isinstance(kl_loss, Tensor) else kl_loss,
        }

    def train_epoch(self, dataloader: DataLoader, log_interval: int = 100) -> dict[str, float]:
        """Train for one epoch and return average metrics."""
        metrics_sum = {}
        batch_count = 0

        for batch in dataloader:
            batch_metrics = self.train_step(batch)
            batch_count += 1
            for key, value in batch_metrics.items():
                metrics_sum[key] = metrics_sum.get(key, 0.0) + value

        return {k: v / batch_count for k, v in metrics_sum.items()}
