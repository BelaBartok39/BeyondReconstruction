"""Experience replay buffer for continuous learning."""

from __future__ import annotations

import random
from collections import deque
from typing import Any

import numpy as np
import torch
from torch import Tensor


class ReplayBuffer:
    """Memory-efficient experience replay buffer.

    Stores samples for rehearsal during continuous learning to prevent
    catastrophic forgetting. Supports multiple sampling strategies.

    Example:
        buffer = ReplayBuffer(capacity=10000)

        # Add samples during training
        for batch in data_stream:
            for sample in batch:
                buffer.add(sample)

        # Sample for replay
        replay_batch = buffer.sample(batch_size=32)
    """

    def __init__(
        self,
        capacity: int = 10000,
        strategy: str = "reservoir",
        seed: int | None = None,
    ):
        """Initialize replay buffer.

        Args:
            capacity: Maximum number of samples to store.
            strategy: Sampling strategy ("reservoir", "fifo", "uniform").
            seed: Random seed for reproducibility.
        """
        self.capacity = capacity
        self.strategy = strategy
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        if strategy == "fifo":
            self._buffer: deque | list = deque(maxlen=capacity)
        else:
            self._buffer = []

        self._total_added = 0

    def add(self, sample: dict[str, Tensor | Any]) -> None:
        """Add a sample to the buffer using configured strategy."""
        self._total_added += 1

        if self.strategy == "reservoir":
            self._reservoir_add(sample)
        elif self.strategy == "fifo":
            self._buffer.append(sample)
        else:  # uniform - random replacement
            if len(self._buffer) < self.capacity:
                self._buffer.append(sample)
            else:
                self._buffer[self.rng.randint(0, len(self._buffer) - 1)] = sample

    def _reservoir_add(self, sample: dict[str, Tensor | Any]) -> None:
        """Add sample using reservoir sampling for uniform distribution over all seen samples."""
        if len(self._buffer) < self.capacity:
            self._buffer.append(sample)
        elif self.rng.random() < self.capacity / self._total_added:
            self._buffer[self.rng.randint(0, self.capacity - 1)] = sample

    def sample(self, batch_size: int) -> list[dict[str, Tensor | Any]]:
        """Sample a batch from the buffer."""
        if not self._buffer:
            return []
        actual_size = min(batch_size, len(self._buffer))
        indices = self.rng.sample(range(len(self._buffer)), actual_size)
        return [self._buffer[i] for i in indices]

    def sample_tensors(self, batch_size: int) -> dict[str, Tensor]:
        """Sample and stack into tensors."""
        samples = self.sample(batch_size)
        if not samples:
            return {}

        return {
            key: torch.stack([s[key] for s in samples]) if isinstance(samples[0][key], Tensor) else [s[key] for s in samples]
            for key in samples[0].keys()
        }

    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self._buffer)

    def clear(self) -> None:
        """Clear the buffer."""
        self._buffer.clear()
        if self.strategy != "fifo":
            self._buffer = []
        self._total_added = 0

    def get_state(self) -> dict:
        """Get buffer state for checkpointing."""
        # Convert tensors to CPU for serialization
        serializable_buffer = [
            {key: value.cpu() if isinstance(value, Tensor) else value for key, value in sample.items()}
            for sample in self._buffer
        ]

        return {
            "buffer": serializable_buffer,
            "total_added": self._total_added,
            "capacity": self.capacity,
            "strategy": self.strategy,
        }

    def load_state(self, state: dict) -> None:
        """Load buffer state."""
        self._buffer = list(state["buffer"])
        self._total_added = state["total_added"]


class PrioritizedReplayBuffer(ReplayBuffer):
    """Replay buffer with prioritized sampling.

    Samples are weighted by their reconstruction error,
    prioritizing harder examples.
    """

    def __init__(
        self,
        capacity: int = 10000,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        seed: int | None = None,
    ):
        """Initialize prioritized buffer.

        Args:
            capacity: Maximum samples.
            alpha: Priority exponent (0 = uniform, 1 = full prioritization).
            beta: Importance sampling exponent.
            beta_increment: Beta increase per sample.
            seed: Random seed.
        """
        super().__init__(capacity=capacity, strategy="uniform", seed=seed)
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment

        self._priorities = np.zeros(capacity, dtype=np.float32)
        self._max_priority = 1.0

    def add(self, sample: dict[str, Tensor | Any], priority: float | None = None) -> None:
        """Add sample with priority (defaults to max priority if not specified)."""
        priority = priority or self._max_priority
        idx = len(self._buffer) if len(self._buffer) < self.capacity else self._total_added % self.capacity

        super().add(sample)

        self._priorities[idx] = priority ** self.alpha
        self._max_priority = max(self._max_priority, priority)

    def sample(self, batch_size: int) -> tuple[list[dict], np.ndarray, np.ndarray]:
        """Sample with priorities and return (samples, indices, importance weights)."""
        if not self._buffer:
            return [], np.array([]), np.array([])

        actual_size = min(batch_size, len(self._buffer))

        # Compute sampling probabilities
        priorities = self._priorities[: len(self._buffer)]
        probs = priorities / priorities.sum()

        # Sample indices
        indices = self.np_rng.choice(len(self._buffer), size=actual_size, p=probs, replace=False)

        # Compute importance sampling weights
        self.beta = min(1.0, self.beta + self.beta_increment)
        weights = (len(self._buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        return [self._buffer[i] for i in indices], indices, weights.astype(np.float32)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Update priorities for sampled indices."""
        for idx, priority in zip(indices, priorities):
            self._priorities[idx] = priority ** self.alpha
            self._max_priority = max(self._max_priority, priority)


class StratifiedReplayBuffer:
    """Replay buffer with stratified sampling by SNR.

    Maintains separate buffers for different SNR ranges to ensure
    balanced sampling across signal quality levels.
    """

    def __init__(
        self,
        capacity: int = 10000,
        num_bins: int = 7,
        snr_range: tuple[float, float] = (-5, 30),
        seed: int | None = None,
    ):
        """Initialize stratified buffer.

        Args:
            capacity: Total capacity (divided among bins).
            num_bins: Number of SNR bins.
            snr_range: SNR range (min, max).
            seed: Random seed.
        """
        self.num_bins = num_bins
        self.snr_range = snr_range
        self.rng = random.Random(seed)

        bin_capacity = capacity // num_bins
        self._buffers = [ReplayBuffer(bin_capacity, seed=seed) for _ in range(num_bins)]
        self._bin_edges = np.linspace(snr_range[0], snr_range[1], num_bins + 1)

    def _get_bin(self, snr_db: float) -> int:
        """Get bin index for SNR value."""
        return int(np.clip(np.searchsorted(self._bin_edges, snr_db) - 1, 0, self.num_bins - 1))

    def add(self, sample: dict[str, Tensor | Any]) -> None:
        """Add sample to appropriate SNR bin."""
        snr_db = sample.get("snr_db")
        if snr_db is None:
            bin_idx = self.num_bins // 2  # Default to middle bin
        else:
            bin_idx = self._get_bin(snr_db.item() if isinstance(snr_db, Tensor) else snr_db)
        self._buffers[bin_idx].add(sample)

    def sample(self, batch_size: int, balanced: bool = True) -> list[dict[str, Tensor | Any]]:
        """Sample from buffer (balanced samples equally from bins, otherwise proportional)."""
        if balanced:
            samples_per_bin = batch_size // self.num_bins
            remainder = batch_size % self.num_bins
            samples = []
            for i, buffer in enumerate(self._buffers):
                if buffer:
                    n = samples_per_bin + (1 if i < remainder else 0)
                    samples.extend(buffer.sample(min(n, len(buffer))))
            return samples

        # Proportional to buffer sizes
        total = sum(len(b) for b in self._buffers)
        if total == 0:
            return []

        samples = []
        for buffer in self._buffers:
            if buffer:
                n = int(batch_size * len(buffer) / total)
                samples.extend(buffer.sample(min(n, len(buffer))))
        return samples

    def __len__(self) -> int:
        """Return total samples across all bins."""
        return sum(len(b) for b in self._buffers)

    def get_stats(self) -> dict:
        """Get per-bin statistics."""
        return {f"bin_{i}": len(self._buffers[i]) for i in range(self.num_bins)}
