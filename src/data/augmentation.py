"""Signal augmentation utilities for RF data."""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import NDArray
import torch


class RFAugmentor:
    """Apply random augmentations to RF IQ signals.

    Augmentations help improve model robustness by simulating
    real-world signal variations.

    Example:
        augmentor = RFAugmentor(p=0.5)
        augmentor.add("phase_rotation", p=0.3)
        augmentor.add("time_shift", max_shift=50)
        augmented = augmentor(iq_signal)
    """

    def __init__(self, p: float = 1.0, seed: int | None = None):
        """Initialize augmentor.

        Args:
            p: Global probability of applying any augmentation.
            seed: Random seed for reproducibility.
        """
        self.p = p
        self.rng = np.random.default_rng(seed)
        self.augmentations: list[tuple[Callable, float, dict]] = []

    def add(
        self,
        name: str,
        p: float = 0.5,
        **kwargs,
    ) -> "RFAugmentor":
        """Add an augmentation to the pipeline.

        Args:
            name: Augmentation name.
            p: Probability of applying this augmentation.
            **kwargs: Augmentation-specific parameters.

        Returns:
            Self for chaining.
        """
        aug_map = {
            "phase_rotation": self._phase_rotation,
            "time_shift": self._time_shift,
            "frequency_shift": self._frequency_shift,
            "amplitude_scale": self._amplitude_scale,
            "add_noise": self._add_noise,
            "time_reverse": self._time_reverse,
            "iq_swap": self._iq_swap,
            "channel_dropout": self._channel_dropout,
        }

        if name not in aug_map:
            raise ValueError(f"Unknown augmentation: {name}. Available: {list(aug_map.keys())}")

        self.augmentations.append((aug_map[name], p, kwargs))
        return self

    def __call__(
        self,
        iq: NDArray[np.float32] | torch.Tensor,
    ) -> NDArray[np.float32] | torch.Tensor:
        """Apply augmentations to signal.

        Args:
            iq: IQ signal [2, seq_len] or batch [batch, 2, seq_len].

        Returns:
            Augmented signal with same shape and type.
        """
        is_tensor = isinstance(iq, torch.Tensor)
        device = iq.device if is_tensor else None

        if is_tensor:
            iq = iq.detach().cpu().numpy()

        # Check if batch
        is_batch = iq.ndim == 3

        if is_batch:
            result = np.stack([self._augment_single(iq[i]) for i in range(len(iq))])
        else:
            result = self._augment_single(iq)

        if is_tensor:
            result = torch.from_numpy(result).to(device)

        return result

    def _augment_single(self, iq: NDArray[np.float32]) -> NDArray[np.float32]:
        """Apply augmentations to a single signal."""
        if self.rng.random() > self.p:
            return iq

        result = iq.copy()

        for aug_func, prob, kwargs in self.augmentations:
            if self.rng.random() < prob:
                result = aug_func(result, **kwargs)

        return result

    def _phase_rotation(
        self,
        iq: NDArray[np.float32],
        max_rotation: float = np.pi,
    ) -> NDArray[np.float32]:
        """Apply random phase rotation.

        Args:
            iq: IQ signal [2, seq_len].
            max_rotation: Maximum rotation in radians.

        Returns:
            Rotated signal.
        """
        theta = self.rng.uniform(-max_rotation, max_rotation)

        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        i_rot = cos_t * iq[0] - sin_t * iq[1]
        q_rot = sin_t * iq[0] + cos_t * iq[1]

        return np.stack([i_rot, q_rot], axis=0).astype(np.float32)

    def _time_shift(
        self,
        iq: NDArray[np.float32],
        max_shift: int = 100,
    ) -> NDArray[np.float32]:
        """Apply random circular time shift.

        Args:
            iq: IQ signal [2, seq_len].
            max_shift: Maximum shift in samples.

        Returns:
            Shifted signal.
        """
        shift = self.rng.integers(-max_shift, max_shift + 1)
        return np.roll(iq, shift, axis=1)

    def _frequency_shift(
        self,
        iq: NDArray[np.float32],
        max_shift: float = 0.1,
    ) -> NDArray[np.float32]:
        """Apply random frequency shift.

        Args:
            iq: IQ signal [2, seq_len].
            max_shift: Maximum normalized frequency shift.

        Returns:
            Frequency-shifted signal.
        """
        freq_shift = self.rng.uniform(-max_shift, max_shift)
        n = iq.shape[1]
        t = np.arange(n)
        phase = 2 * np.pi * freq_shift * t

        # Convert to complex, shift, convert back
        signal = iq[0] + 1j * iq[1]
        shifted = signal * np.exp(1j * phase)

        return np.stack([shifted.real, shifted.imag], axis=0).astype(np.float32)

    def _amplitude_scale(
        self,
        iq: NDArray[np.float32],
        scale_range: tuple[float, float] = (0.8, 1.2),
    ) -> NDArray[np.float32]:
        """Apply random amplitude scaling.

        Args:
            iq: IQ signal [2, seq_len].
            scale_range: Range of scale factors.

        Returns:
            Scaled signal.
        """
        scale = self.rng.uniform(scale_range[0], scale_range[1])
        return (iq * scale).astype(np.float32)

    def _add_noise(
        self,
        iq: NDArray[np.float32],
        noise_std_range: tuple[float, float] = (0.01, 0.1),
    ) -> NDArray[np.float32]:
        """Add random Gaussian noise.

        Args:
            iq: IQ signal [2, seq_len].
            noise_std_range: Range of noise standard deviations.

        Returns:
            Noisy signal.
        """
        noise_std = self.rng.uniform(noise_std_range[0], noise_std_range[1])
        noise = self.rng.normal(0, noise_std, iq.shape)
        return (iq + noise).astype(np.float32)

    def _time_reverse(
        self,
        iq: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Reverse signal in time.

        Args:
            iq: IQ signal [2, seq_len].

        Returns:
            Time-reversed signal.
        """
        return iq[:, ::-1].copy()

    def _iq_swap(
        self,
        iq: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Swap I and Q channels.

        Args:
            iq: IQ signal [2, seq_len].

        Returns:
            Signal with swapped channels.
        """
        return iq[[1, 0], :]

    def _channel_dropout(
        self,
        iq: NDArray[np.float32],
        dropout_value: float = 0.0,
    ) -> NDArray[np.float32]:
        """Randomly zero out one channel.

        Args:
            iq: IQ signal [2, seq_len].
            dropout_value: Value to replace channel with.

        Returns:
            Signal with one channel dropped.
        """
        result = iq.copy()
        channel = self.rng.integers(0, 2)
        result[channel] = dropout_value
        return result


def create_default_augmentor(seed: int | None = None) -> RFAugmentor:
    """Create augmentor with default settings for RF signals.

    Args:
        seed: Random seed.

    Returns:
        Configured RFAugmentor.
    """
    augmentor = RFAugmentor(p=0.8, seed=seed)
    augmentor.add("phase_rotation", p=0.5, max_rotation=np.pi)
    augmentor.add("time_shift", p=0.3, max_shift=50)
    augmentor.add("frequency_shift", p=0.3, max_shift=0.05)
    augmentor.add("amplitude_scale", p=0.3, scale_range=(0.9, 1.1))
    augmentor.add("add_noise", p=0.2, noise_std_range=(0.01, 0.05))

    return augmentor
