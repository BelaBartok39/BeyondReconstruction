"""Elastic Weight Consolidation for preventing catastrophic forgetting."""

from __future__ import annotations

from copy import deepcopy
from typing import Iterator

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader


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

        if device is None:
            device = next(model.parameters()).device
        self.device = torch.device(device) if isinstance(device, str) else device

        # Storage for Fisher information and parameter snapshots
        self._fisher: dict[str, Tensor] = {}
        self._params_snapshot: dict[str, Tensor] = {}
        self._is_initialized = False

        # Model type detection
        self._is_snr_conditioned = hasattr(model, "encoder") and hasattr(
            model.encoder, "snr_embed"
        )
        self._is_vae = hasattr(model, "reparameterize")

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
        if num_samples is None:
            num_samples = self.fisher_samples

        self.model.eval()

        # Initialize Fisher accumulator
        fisher_accum: dict[str, Tensor] = {}
        for name, param in self._named_parameters():
            fisher_accum[name] = torch.zeros_like(param)

        sample_count = 0

        for batch in dataloader:
            if sample_count >= num_samples:
                break

            iq = batch["iq"].to(self.device)
            snr = batch.get("snr")
            if snr is not None:
                snr = snr.to(self.device)

            batch_size = iq.size(0)

            # Compute loss and gradients
            self.model.zero_grad()
            loss = self._compute_sample_loss(iq, snr)
            loss.backward()

            # Accumulate squared gradients (Fisher diagonal approximation)
            for name, param in self._named_parameters():
                if param.grad is not None:
                    fisher_accum[name] += param.grad.data.pow(2) * batch_size

            sample_count += batch_size

        # Normalize
        for name in fisher_accum:
            fisher_accum[name] /= sample_count

        # Update Fisher (online or replace)
        if self.online and self._is_initialized:
            for name in self._fisher:
                self._fisher[name] = (
                    self.gamma * self._fisher[name]
                    + (1 - self.gamma) * fisher_accum[name]
                )
        else:
            self._fisher = fisher_accum

        # Save parameter snapshot
        self._params_snapshot = {}
        for name, param in self._named_parameters():
            self._params_snapshot[name] = param.data.clone()

        self._is_initialized = True

    def _compute_sample_loss(self, iq: Tensor, snr: Tensor | None) -> Tensor:
        """Compute loss for Fisher estimation.

        Uses log-likelihood (negative reconstruction loss) for proper
        Fisher information computation.
        """
        if self._is_snr_conditioned and snr is not None:
            x_recon, mu, logvar, _ = self.model(iq, snr)
            # Use reconstruction loss only (not KL) for Fisher
            loss = nn.functional.mse_loss(x_recon, iq, reduction="sum")
        elif self._is_vae:
            x_recon, mu, logvar, _ = self.model(iq)
            loss = nn.functional.mse_loss(x_recon, iq, reduction="sum")
        else:
            x_recon, _ = self.model(iq)
            loss = nn.functional.mse_loss(x_recon, iq, reduction="sum")

        return loss

    def penalty(self) -> Tensor:
        """Compute EWC penalty term.

        Returns:
            EWC penalty to add to the loss.
        """
        if not self._is_initialized:
            return torch.tensor(0.0, device=self.device)

        penalty = torch.tensor(0.0, device=self.device)

        for name, param in self._named_parameters():
            if name in self._fisher and name in self._params_snapshot:
                # Penalize deviation from snapshot weighted by Fisher
                diff = param - self._params_snapshot[name]
                penalty += (self._fisher[name] * diff.pow(2)).sum()

        return self.ewc_lambda * penalty / 2

    def _named_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        """Get named parameters that require gradients."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                yield name, param

    def get_fisher_stats(self) -> dict:
        """Get statistics about Fisher information.

        Returns:
            Dictionary with Fisher statistics.
        """
        if not self._is_initialized:
            return {"initialized": False}

        stats = {"initialized": True, "parameters": {}}

        for name, fisher in self._fisher.items():
            stats["parameters"][name] = {
                "mean": float(fisher.mean()),
                "std": float(fisher.std()),
                "max": float(fisher.max()),
                "sparsity": float((fisher < 1e-6).float().mean()),
            }

        return stats

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

        if device is None:
            device = next(model.parameters()).device
        self.device = torch.device(device) if isinstance(device, str) else device

        self._is_snr_conditioned = hasattr(model, "encoder") and hasattr(
            model.encoder, "snr_embed"
        )
        self._is_vae = hasattr(model, "reparameterize")

    def train_step(self, batch: dict[str, Tensor]) -> dict[str, float]:
        """Perform single training step with EWC.

        Args:
            batch: Training batch.

        Returns:
            Dictionary with loss components.
        """
        self.model.train()

        iq = batch["iq"].to(self.device)
        snr = batch.get("snr")
        if snr is not None:
            snr = snr.to(self.device)

        self.optimizer.zero_grad()

        # Compute task loss
        if self._is_snr_conditioned and snr is not None:
            x_recon, mu, logvar, _ = self.model(iq, snr)
            task_loss, recon_loss, kl_loss = self.model.loss(iq, x_recon, mu, logvar)
        elif self._is_vae:
            x_recon, mu, logvar, _ = self.model(iq)
            task_loss, recon_loss, kl_loss = self.model.loss(iq, x_recon, mu, logvar)
        else:
            x_recon, _ = self.model(iq)
            task_loss = self.model.reconstruction_loss(iq, x_recon)
            recon_loss = task_loss
            kl_loss = torch.tensor(0.0)

        # Add EWC penalty
        ewc_penalty = self.ewc.penalty()
        total_loss = task_loss + ewc_penalty

        total_loss.backward()

        if self.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_clip_norm
            )

        self.optimizer.step()

        return {
            "total_loss": total_loss.item(),
            "task_loss": task_loss.item(),
            "ewc_penalty": ewc_penalty.item(),
            "recon_loss": recon_loss.item(),
            "kl_loss": kl_loss.item() if isinstance(kl_loss, Tensor) else kl_loss,
        }

    def train_epoch(
        self, dataloader: DataLoader, log_interval: int = 100
    ) -> dict[str, float]:
        """Train for one epoch.

        Args:
            dataloader: Training data.
            log_interval: Logging interval.

        Returns:
            Average metrics for epoch.
        """
        metrics_sum = {}
        batch_count = 0

        for batch in dataloader:
            batch_metrics = self.train_step(batch)
            batch_count += 1

            for key, value in batch_metrics.items():
                metrics_sum[key] = metrics_sum.get(key, 0.0) + value

        # Average metrics
        return {k: v / batch_count for k, v in metrics_sum.items()}
