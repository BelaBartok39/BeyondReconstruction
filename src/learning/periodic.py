"""Periodic retraining scheduler for continuous learning."""

from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from .replay_buffer import ReplayBuffer


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


class RetrainingTrigger(Enum):
    """Triggers for periodic retraining."""

    SAMPLE_COUNT = "sample_count"
    TIME_INTERVAL = "time_interval"
    PERFORMANCE_DROP = "performance_drop"
    DISTRIBUTION_SHIFT = "distribution_shift"


@dataclass
class RetrainingEvent:
    """Record of a retraining event."""

    timestamp: float
    trigger: RetrainingTrigger
    samples_processed: int
    metrics_before: dict
    metrics_after: dict
    duration: float


class PeriodicRetrainer:
    """Periodic retraining scheduler with data buffering.

    Collects new samples and periodically retrains the model,
    optionally combined with replay from historical data.

    Example:
        retrainer = PeriodicRetrainer(
            model, interval=1000, epochs_per_retrain=5
        )

        for batch in stream:
            retrainer.add_samples(batch)
            if retrainer.should_retrain():
                retrainer.retrain()
    """

    def __init__(
        self,
        model: nn.Module,
        interval: int = 1000,
        epochs_per_retrain: int = 5,
        batch_size: int = 64,
        learning_rate: float = 1e-4,
        trigger: RetrainingTrigger = RetrainingTrigger.SAMPLE_COUNT,
        time_interval_seconds: float | None = None,
        performance_threshold: float | None = None,
        buffer_size: int = 10000,
        replay_ratio: float = 0.5,
        use_replay: bool = True,
        gradient_clip_norm: float | None = 1.0,
        device: torch.device | str | None = None,
    ):
        """Initialize periodic retrainer.

        Args:
            model: Model to retrain.
            interval: Samples between retraining (for SAMPLE_COUNT trigger).
            epochs_per_retrain: Training epochs per retraining session.
            batch_size: Batch size for retraining.
            learning_rate: Learning rate for retraining.
            trigger: When to trigger retraining.
            time_interval_seconds: Seconds between retraining (for TIME_INTERVAL).
            performance_threshold: Performance drop threshold (for PERFORMANCE_DROP).
            buffer_size: Maximum samples to buffer.
            replay_ratio: Fraction of batch from replay buffer.
            use_replay: Whether to use experience replay.
            gradient_clip_norm: Gradient clipping threshold.
            device: Training device.
        """
        self.model = model
        self.interval = interval
        self.epochs_per_retrain = epochs_per_retrain
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.trigger = trigger
        self.time_interval_seconds = time_interval_seconds
        self.performance_threshold = performance_threshold
        self.replay_ratio = replay_ratio
        self.use_replay = use_replay
        self.gradient_clip_norm = gradient_clip_norm
        self.device = _get_device(model, device)

        # Data buffers
        self._new_samples_buffer: list[dict[str, Tensor]] = []
        self._replay_buffer = ReplayBuffer(capacity=buffer_size) if use_replay else None

        # State tracking
        self._samples_since_retrain = 0
        self._total_samples = 0
        self._last_retrain_time = time.time()
        self._last_performance = None
        self._retraining_history: list[RetrainingEvent] = []
        self._is_snr_conditioned, self._is_vae = _detect_model_type(model)

    def add_samples(self, batch: dict[str, Tensor]) -> None:
        """Add new samples to buffer."""
        batch_size = batch["iq"].size(0)
        self._samples_since_retrain += batch_size
        self._total_samples += batch_size

        # Add to new samples buffer
        self._new_samples_buffer.append({k: v.cpu().clone() for k, v in batch.items() if isinstance(v, Tensor)})

        # Add to replay buffer
        if self._replay_buffer is not None:
            for i in range(batch_size):
                self._replay_buffer.add({k: v[i].cpu() for k, v in batch.items() if isinstance(v, Tensor)})

    def should_retrain(self, current_performance: float | None = None) -> bool:
        """Check if retraining should be triggered based on configured trigger."""
        if not self._new_samples_buffer:
            return False

        if self.trigger == RetrainingTrigger.SAMPLE_COUNT:
            return self._samples_since_retrain >= self.interval

        if self.trigger == RetrainingTrigger.TIME_INTERVAL:
            return self.time_interval_seconds is not None and (
                time.time() - self._last_retrain_time >= self.time_interval_seconds
            )

        if self.trigger == RetrainingTrigger.PERFORMANCE_DROP:
            if current_performance is None or self.performance_threshold is None:
                return False
            if self._last_performance is None:
                self._last_performance = current_performance
                return False
            return (self._last_performance - current_performance) > self.performance_threshold

        # DISTRIBUTION_SHIFT uses simplified sample-based trigger
        return self._samples_since_retrain >= self.interval

    def retrain(self, validation_fn: Callable[[nn.Module], dict] | None = None) -> RetrainingEvent:
        """Perform retraining on buffered data."""
        start_time = time.time()
        metrics_before = validation_fn(self.model) if validation_fn else {}

        # Prepare training data and optimizer
        train_loader = self._prepare_dataloader()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)

        # Training loop
        self.model.train()
        for epoch in range(self.epochs_per_retrain):
            for batch in train_loader:
                self._train_step(batch, optimizer)

        metrics_after = validation_fn(self.model) if validation_fn else {}

        # Update state
        self._samples_since_retrain = 0
        self._last_retrain_time = time.time()
        self._new_samples_buffer = []
        if metrics_after.get("loss") is not None:
            self._last_performance = -metrics_after["loss"]  # Higher is better

        # Create and store event
        event = RetrainingEvent(
            timestamp=time.time(),
            trigger=self.trigger,
            samples_processed=self._total_samples,
            metrics_before=metrics_before,
            metrics_after=metrics_after,
            duration=time.time() - start_time,
        )
        self._retraining_history.append(event)
        return event

    def _prepare_dataloader(self) -> DataLoader:
        """Prepare dataloader from buffered samples with optional replay."""
        # Concatenate new samples
        all_iq = torch.cat([b["iq"] for b in self._new_samples_buffer], dim=0)
        all_snr = torch.cat([b["snr"] for b in self._new_samples_buffer], dim=0) if "snr" in self._new_samples_buffer[0] else None
        all_power = torch.cat([b["power"] for b in self._new_samples_buffer], dim=0) if "power" in self._new_samples_buffer[0] else None

        # Mix with replay buffer
        if self.use_replay and self._replay_buffer and len(self._replay_buffer) > 0:
            replay_size = min(int(len(all_iq) * self.replay_ratio), len(self._replay_buffer))
            if replay_size > 0:
                replay_samples = self._replay_buffer.sample(replay_size)
                replay_iq = torch.stack([s["iq"] for s in replay_samples])
                all_iq = torch.cat([all_iq, replay_iq], dim=0)

                if all_snr is not None and "snr" in replay_samples[0]:
                    replay_snr = torch.stack([s["snr"] for s in replay_samples])
                    all_snr = torch.cat([all_snr, replay_snr], dim=0)

                if all_power is not None and "power" in replay_samples[0]:
                    replay_power = torch.stack([s["power"] for s in replay_samples])
                    all_power = torch.cat([all_power, replay_power], dim=0)

        # Create dataset and loader
        tensors = [all_iq]
        if all_snr is not None:
            tensors.append(all_snr)
        if all_power is not None:
            tensors.append(all_power)
        dataset = TensorDataset(*tensors)
        return DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

    def _train_step(self, batch: tuple[Tensor, ...], optimizer: torch.optim.Optimizer) -> float:
        """Single training step."""
        iq = batch[0].to(self.device)
        snr = batch[1].to(self.device) if len(batch) > 1 else None
        power = batch[2].to(self.device) if len(batch) > 2 else None

        optimizer.zero_grad()

        if self._is_vae:
            if self._is_snr_conditioned:
                # SNR-conditioned models always need SNR and power
                if snr is None:
                    snr = torch.full((iq.size(0),), 0.5, device=iq.device)
                if power is None:
                    power = torch.full((iq.size(0),), 0.5, device=iq.device)
                result = self.model(iq, snr, power)
                # Handle probabilistic decoder (5 outputs) vs non-probabilistic (4 outputs)
                if len(result) == 5:
                    x_mean, x_logvar, mu, logvar, _ = result
                    loss, _, _ = self.model.loss(iq, x_mean, mu, logvar, x_logvar)
                else:
                    x_recon, mu, logvar, _ = result
                    loss, _, _ = self.model.loss(iq, x_recon, mu, logvar)
            else:
                x_recon, mu, logvar, _ = self.model(iq)
                loss, _, _ = self.model.loss(iq, x_recon, mu, logvar)
        else:
            x_recon, _ = self.model(iq)
            loss = self.model.reconstruction_loss(iq, x_recon)

        loss.backward()

        if self.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_norm)

        optimizer.step()
        return loss.item()

    def get_stats(self) -> dict:
        """Get retrainer statistics."""
        return {
            "total_samples": self._total_samples,
            "samples_since_retrain": self._samples_since_retrain,
            "buffer_size": sum(b["iq"].size(0) for b in self._new_samples_buffer),
            "replay_buffer_size": len(self._replay_buffer) if self._replay_buffer else 0,
            "num_retrainings": len(self._retraining_history),
            "last_retrain_time": self._last_retrain_time,
        }

    def get_history(self) -> list[RetrainingEvent]:
        """Get retraining history."""
        return self._retraining_history

    def get_state(self) -> dict:
        """Get state for checkpointing."""
        return {
            "samples_since_retrain": self._samples_since_retrain,
            "total_samples": self._total_samples,
            "last_retrain_time": self._last_retrain_time,
            "last_performance": self._last_performance,
            "replay_buffer": self._replay_buffer.get_state() if self._replay_buffer else None,
        }

    def load_state(self, state: dict) -> None:
        """Load state from checkpoint."""
        self._samples_since_retrain = state["samples_since_retrain"]
        self._total_samples = state["total_samples"]
        self._last_retrain_time = state["last_retrain_time"]
        self._last_performance = state["last_performance"]

        if state["replay_buffer"] and self._replay_buffer:
            self._replay_buffer.load_state(state["replay_buffer"])
