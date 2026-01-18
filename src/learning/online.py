"""Online/incremental learning for continuous model updates."""

from __future__ import annotations

from collections import deque
from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Optimizer


class OnlineLearner:
    """Online learning module for incremental model updates.

    Enables continuous learning from streaming data with features:
    - Reduced learning rate for stability
    - Gradient accumulation for larger effective batch sizes
    - Optional loss smoothing for noisy updates

    Example:
        learner = OnlineLearner(model, lr=1e-4)
        for batch in stream:
            metrics = learner.update(batch)
            if metrics["update_count"] % 100 == 0:
                print(f"Loss: {metrics['loss']:.4f}")
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        update_frequency: int = 1,
        gradient_accumulation_steps: int = 1,
        gradient_clip_norm: float | None = 1.0,
        loss_ema_decay: float = 0.99,
        optimizer: Optimizer | None = None,
        device: torch.device | str | None = None,
    ):
        """Initialize online learner.

        Args:
            model: Model to train.
            learning_rate: Learning rate for online updates.
            weight_decay: Weight decay for regularization.
            update_frequency: Update every N batches.
            gradient_accumulation_steps: Accumulate gradients over N steps.
            gradient_clip_norm: Max gradient norm (None to disable).
            loss_ema_decay: Decay for exponential moving average of loss.
            optimizer: Custom optimizer (default: AdamW).
            device: Device for training.
        """
        self.model = model
        self.learning_rate = learning_rate
        self.update_frequency = update_frequency
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.gradient_clip_norm = gradient_clip_norm
        self.loss_ema_decay = loss_ema_decay

        if device is None:
            device = next(model.parameters()).device
        self.device = torch.device(device) if isinstance(device, str) else device

        # Optimizer
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay,
            )
        else:
            self.optimizer = optimizer

        # State tracking
        self._update_count = 0
        self._batch_count = 0
        self._loss_ema = None
        self._accumulated_loss = 0.0

        # Model type detection
        self._is_snr_conditioned = hasattr(model, "encoder") and hasattr(
            model.encoder, "snr_embed"
        )
        self._is_vae = hasattr(model, "reparameterize")

    def update(self, batch: dict[str, Tensor]) -> dict[str, float]:
        """Perform online update with a batch.

        Args:
            batch: Dictionary with "iq" and optionally "snr" tensors.

        Returns:
            Dictionary with update metrics.
        """
        self.model.train()
        self._batch_count += 1

        # Check if we should update this batch
        if self._batch_count % self.update_frequency != 0:
            return {"skipped": True, "batch_count": self._batch_count}

        # Move data to device
        iq = batch["iq"].to(self.device)
        snr = batch.get("snr")
        if snr is not None:
            snr = snr.to(self.device)

        # Forward pass
        loss = self._compute_loss(iq, snr)

        # Scale loss for gradient accumulation
        scaled_loss = loss / self.gradient_accumulation_steps
        scaled_loss.backward()

        self._accumulated_loss += loss.item()

        # Perform optimizer step if accumulated enough
        if self._batch_count % (self.update_frequency * self.gradient_accumulation_steps) == 0:
            # Gradient clipping
            if self.gradient_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.gradient_clip_norm
                )

            self.optimizer.step()
            self.optimizer.zero_grad()

            # Update EMA loss
            avg_loss = self._accumulated_loss / self.gradient_accumulation_steps
            if self._loss_ema is None:
                self._loss_ema = avg_loss
            else:
                self._loss_ema = (
                    self.loss_ema_decay * self._loss_ema
                    + (1 - self.loss_ema_decay) * avg_loss
                )

            self._accumulated_loss = 0.0
            self._update_count += 1

        return {
            "loss": loss.item(),
            "loss_ema": self._loss_ema if self._loss_ema is not None else loss.item(),
            "update_count": self._update_count,
            "batch_count": self._batch_count,
        }

    def _compute_loss(self, iq: Tensor, snr: Tensor | None) -> Tensor:
        """Compute reconstruction loss.

        Args:
            iq: IQ signals.
            snr: SNR values.

        Returns:
            Loss tensor.
        """
        if self._is_snr_conditioned and snr is not None:
            x_recon, mu, logvar, _ = self.model(iq, snr)
            loss, _, _ = self.model.loss(iq, x_recon, mu, logvar)
        elif self._is_vae:
            x_recon, mu, logvar, _ = self.model(iq)
            loss, _, _ = self.model.loss(iq, x_recon, mu, logvar)
        else:
            x_recon, _ = self.model(iq)
            loss = self.model.reconstruction_loss(iq, x_recon)

        return loss

    def set_learning_rate(self, lr: float) -> None:
        """Update learning rate.

        Args:
            lr: New learning rate.
        """
        self.learning_rate = lr
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_state(self) -> dict:
        """Get learner state for checkpointing.

        Returns:
            State dictionary.
        """
        return {
            "optimizer_state": self.optimizer.state_dict(),
            "update_count": self._update_count,
            "batch_count": self._batch_count,
            "loss_ema": self._loss_ema,
        }

    def load_state(self, state: dict) -> None:
        """Load learner state.

        Args:
            state: State dictionary.
        """
        self.optimizer.load_state_dict(state["optimizer_state"])
        self._update_count = state["update_count"]
        self._batch_count = state["batch_count"]
        self._loss_ema = state["loss_ema"]


class AdaptiveLearningRateScheduler:
    """Adaptive learning rate based on loss statistics.

    Reduces learning rate when loss stops improving,
    increases when consistently improving.
    """

    def __init__(
        self,
        learner: OnlineLearner,
        patience: int = 100,
        factor: float = 0.5,
        min_lr: float = 1e-6,
        max_lr: float = 1e-3,
        threshold: float = 1e-4,
    ):
        """Initialize scheduler.

        Args:
            learner: OnlineLearner instance.
            patience: Updates to wait before reducing LR.
            factor: Factor to multiply LR when reducing.
            min_lr: Minimum learning rate.
            max_lr: Maximum learning rate.
            threshold: Minimum improvement to reset patience.
        """
        self.learner = learner
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.threshold = threshold

        self._best_loss = float("inf")
        self._wait = 0
        self._num_bad_updates = 0

    def step(self, loss: float) -> bool:
        """Update scheduler with current loss.

        Args:
            loss: Current loss value.

        Returns:
            True if learning rate was changed.
        """
        if loss < self._best_loss - self.threshold:
            self._best_loss = loss
            self._wait = 0
            self._num_bad_updates = 0
            return False

        self._wait += 1
        self._num_bad_updates += 1

        if self._wait >= self.patience:
            current_lr = self.learner.learning_rate
            new_lr = max(current_lr * self.factor, self.min_lr)

            if new_lr < current_lr:
                self.learner.set_learning_rate(new_lr)
                self._wait = 0
                return True

        return False


class GradientMonitor:
    """Monitor gradient statistics for debugging and analysis."""

    def __init__(self, model: nn.Module, window_size: int = 100):
        """Initialize monitor.

        Args:
            model: Model to monitor.
            window_size: Window for moving statistics.
        """
        self.model = model
        self.window_size = window_size

        self._grad_norms = deque(maxlen=window_size)
        self._param_norms = deque(maxlen=window_size)

    def record(self) -> dict[str, float]:
        """Record current gradient statistics.

        Returns:
            Dictionary with gradient statistics.
        """
        total_norm = 0.0
        param_norm = 0.0

        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
            param_norm += p.data.norm(2).item() ** 2

        total_norm = total_norm ** 0.5
        param_norm = param_norm ** 0.5

        self._grad_norms.append(total_norm)
        self._param_norms.append(param_norm)

        return {
            "grad_norm": total_norm,
            "param_norm": param_norm,
            "grad_norm_mean": sum(self._grad_norms) / len(self._grad_norms),
            "grad_norm_max": max(self._grad_norms),
        }
