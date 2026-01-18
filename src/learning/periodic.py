"""Periodic retraining scheduler for continuous learning."""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from .replay_buffer import ReplayBuffer


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

        if device is None:
            device = next(model.parameters()).device
        self.device = torch.device(device) if isinstance(device, str) else device

        # Data buffers
        self._new_samples_buffer: list[dict[str, Tensor]] = []
        self._replay_buffer = ReplayBuffer(capacity=buffer_size) if use_replay else None

        # State tracking
        self._samples_since_retrain = 0
        self._total_samples = 0
        self._last_retrain_time = time.time()
        self._last_performance = None
        self._retraining_history: list[RetrainingEvent] = []

        # Model type detection
        self._is_snr_conditioned = hasattr(model, "encoder") and hasattr(
            model.encoder, "snr_embed"
        )
        self._is_vae = hasattr(model, "reparameterize")

    def add_samples(self, batch: dict[str, Tensor]) -> None:
        """Add new samples to buffer.

        Args:
            batch: Dictionary with "iq" and optionally "snr" tensors.
        """
        batch_size = batch["iq"].size(0)
        self._samples_since_retrain += batch_size
        self._total_samples += batch_size

        # Add to new samples buffer
        self._new_samples_buffer.append({
            k: v.cpu().clone() for k, v in batch.items()
            if isinstance(v, Tensor)
        })

        # Add to replay buffer
        if self._replay_buffer is not None:
            for i in range(batch_size):
                sample = {k: v[i].cpu() for k, v in batch.items() if isinstance(v, Tensor)}
                self._replay_buffer.add(sample)

    def should_retrain(self, current_performance: float | None = None) -> bool:
        """Check if retraining should be triggered.

        Args:
            current_performance: Current model performance metric.

        Returns:
            True if retraining should occur.
        """
        if len(self._new_samples_buffer) == 0:
            return False

        if self.trigger == RetrainingTrigger.SAMPLE_COUNT:
            return self._samples_since_retrain >= self.interval

        elif self.trigger == RetrainingTrigger.TIME_INTERVAL:
            if self.time_interval_seconds is None:
                return False
            return (time.time() - self._last_retrain_time) >= self.time_interval_seconds

        elif self.trigger == RetrainingTrigger.PERFORMANCE_DROP:
            if current_performance is None or self.performance_threshold is None:
                return False
            if self._last_performance is None:
                self._last_performance = current_performance
                return False
            drop = self._last_performance - current_performance
            return drop > self.performance_threshold

        elif self.trigger == RetrainingTrigger.DISTRIBUTION_SHIFT:
            # Would require distribution monitoring - simplified version
            return self._samples_since_retrain >= self.interval

        return False

    def retrain(
        self,
        validation_fn: Callable[[nn.Module], dict] | None = None,
    ) -> RetrainingEvent:
        """Perform retraining on buffered data.

        Args:
            validation_fn: Optional function to compute validation metrics.

        Returns:
            RetrainingEvent with details of the retraining.
        """
        start_time = time.time()

        # Get metrics before
        metrics_before = {}
        if validation_fn is not None:
            metrics_before = validation_fn(self.model)

        # Prepare training data
        train_loader = self._prepare_dataloader()

        # Create optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
        )

        # Training loop
        self.model.train()
        for epoch in range(self.epochs_per_retrain):
            for batch in train_loader:
                loss = self._train_step(batch, optimizer)

        # Get metrics after
        metrics_after = {}
        if validation_fn is not None:
            metrics_after = validation_fn(self.model)

        # Update state
        self._samples_since_retrain = 0
        self._last_retrain_time = time.time()
        self._new_samples_buffer = []

        if metrics_after.get("loss") is not None:
            self._last_performance = -metrics_after["loss"]  # Higher is better

        # Create event
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
        """Prepare dataloader from buffered samples.

        Returns:
            DataLoader for retraining.
        """
        # Concatenate new samples
        all_iq = torch.cat([b["iq"] for b in self._new_samples_buffer], dim=0)
        all_snr = None
        if "snr" in self._new_samples_buffer[0]:
            all_snr = torch.cat([b["snr"] for b in self._new_samples_buffer], dim=0)

        # Mix with replay buffer
        if self.use_replay and self._replay_buffer is not None:
            replay_size = int(len(all_iq) * self.replay_ratio)
            if replay_size > 0 and len(self._replay_buffer) > 0:
                replay_samples = self._replay_buffer.sample(
                    min(replay_size, len(self._replay_buffer))
                )
                replay_iq = torch.stack([s["iq"] for s in replay_samples])
                all_iq = torch.cat([all_iq, replay_iq], dim=0)

                if all_snr is not None and "snr" in replay_samples[0]:
                    replay_snr = torch.stack([s["snr"] for s in replay_samples])
                    all_snr = torch.cat([all_snr, replay_snr], dim=0)

        # Create dataset and loader
        if all_snr is not None:
            dataset = TensorDataset(all_iq, all_snr)
        else:
            dataset = TensorDataset(all_iq)

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

    def _train_step(
        self,
        batch: tuple[Tensor, ...],
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """Single training step.

        Args:
            batch: Tuple of tensors from dataloader.
            optimizer: Optimizer.

        Returns:
            Loss value.
        """
        iq = batch[0].to(self.device)
        snr = batch[1].to(self.device) if len(batch) > 1 else None

        optimizer.zero_grad()

        if self._is_snr_conditioned and snr is not None:
            x_recon, mu, logvar, _ = self.model(iq, snr)
            loss, _, _ = self.model.loss(iq, x_recon, mu, logvar)
        elif self._is_vae:
            x_recon, mu, logvar, _ = self.model(iq)
            loss, _, _ = self.model.loss(iq, x_recon, mu, logvar)
        else:
            x_recon, _ = self.model(iq)
            loss = self.model.reconstruction_loss(iq, x_recon)

        loss.backward()

        if self.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.gradient_clip_norm
            )

        optimizer.step()

        return loss.item()

    def get_stats(self) -> dict:
        """Get retrainer statistics.

        Returns:
            Dictionary with statistics.
        """
        return {
            "total_samples": self._total_samples,
            "samples_since_retrain": self._samples_since_retrain,
            "buffer_size": sum(b["iq"].size(0) for b in self._new_samples_buffer),
            "replay_buffer_size": len(self._replay_buffer) if self._replay_buffer else 0,
            "num_retrainings": len(self._retraining_history),
            "last_retrain_time": self._last_retrain_time,
        }

    def get_history(self) -> list[RetrainingEvent]:
        """Get retraining history.

        Returns:
            List of RetrainingEvent objects.
        """
        return self._retraining_history

    def get_state(self) -> dict:
        """Get state for checkpointing.

        Returns:
            State dictionary.
        """
        return {
            "samples_since_retrain": self._samples_since_retrain,
            "total_samples": self._total_samples,
            "last_retrain_time": self._last_retrain_time,
            "last_performance": self._last_performance,
            "replay_buffer": self._replay_buffer.get_state() if self._replay_buffer else None,
        }

    def load_state(self, state: dict) -> None:
        """Load state from checkpoint.

        Args:
            state: State dictionary.
        """
        self._samples_since_retrain = state["samples_since_retrain"]
        self._total_samples = state["total_samples"]
        self._last_retrain_time = state["last_retrain_time"]
        self._last_performance = state["last_performance"]

        if state["replay_buffer"] is not None and self._replay_buffer is not None:
            self._replay_buffer.load_state(state["replay_buffer"])
