"""PyTorch Dataset classes for RF signal data."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from numpy.typing import NDArray
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset

from .synthetic import SyntheticRFGenerator, SignalMetadata
from .snr_estimation import estimate_snr, normalize_snr
from .augmentation import RFAugmentor


def normalize_power(power_db: float, power_range: tuple[float, float] = (-20, 10)) -> float:
    """Normalize power to [0, 1] range.

    Args:
        power_db: Power in dB.
        power_range: Min and max power in dB.

    Returns:
        Normalized power value in [0, 1].
    """
    min_power, max_power = power_range
    return max(0.0, min(1.0, (power_db - min_power) / (max_power - min_power)))


class RFDataset(Dataset):
    """PyTorch Dataset for RF IQ signals.

    Supports both pre-generated data and on-the-fly generation.

    Example:
        # From pre-generated data
        dataset = RFDataset(iq_data=signals, labels=labels, snr_values=snrs)

        # On-the-fly generation
        dataset = RFDataset.from_generator(
            generator=SyntheticRFGenerator(),
            num_samples=10000,
            anomaly_ratio=0.1,
        )
    """

    def __init__(
        self,
        iq_data: NDArray[np.float32] | torch.Tensor,
        labels: NDArray[np.int64] | torch.Tensor | None = None,
        snr_values: NDArray[np.float32] | torch.Tensor | None = None,
        power_values: NDArray[np.float32] | torch.Tensor | None = None,
        metadata: list[SignalMetadata] | None = None,
        augmentor: RFAugmentor | None = None,
        estimate_snr_online: bool = False,
        snr_range: tuple[float, float] = (-5, 30),
        power_range: tuple[float, float] = (-20, 10),
    ):
        """Initialize dataset.

        Args:
            iq_data: IQ signal data [N, 2, seq_len].
            labels: Binary labels (0=normal, 1=anomaly) [N].
            snr_values: SNR values in dB [N].
            power_values: Signal power values in dB [N].
            metadata: List of signal metadata.
            augmentor: Optional augmentation pipeline.
            estimate_snr_online: If True, estimate SNR from signal.
            snr_range: SNR range for normalization.
            power_range: Power range for normalization (dB).
        """
        self.iq_data = torch.as_tensor(iq_data, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.long) if labels is not None else torch.zeros(len(iq_data), dtype=torch.long)
        self.snr_values = torch.as_tensor(snr_values, dtype=torch.float32) if snr_values is not None else None
        self.power_values = torch.as_tensor(power_values, dtype=torch.float32) if power_values is not None else None
        self.metadata = metadata
        self.augmentor = augmentor
        self.estimate_snr_online = estimate_snr_online
        self.snr_range = snr_range
        self.power_range = power_range

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.iq_data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """Get a single sample.

        Args:
            idx: Sample index.

        Returns:
            Dictionary with keys:
                - iq: IQ signal [2, seq_len]
                - label: Binary label (0 or 1)
                - snr: Normalized SNR value [0, 1]
                - snr_db: Raw SNR in dB
                - power: Normalized power value [0, 1]
                - power_db: Raw power in dB
        """
        iq = self.iq_data[idx].clone()

        if self.augmentor is not None:
            iq = self.augmentor(iq)

        # Get or estimate SNR
        if self.snr_values is not None:
            snr_db = self.snr_values[idx].item()
        elif self.estimate_snr_online:
            snr_db = estimate_snr(iq)
        else:
            snr_db = 15.0

        # Get power (default to 0 dB if not available)
        if self.power_values is not None:
            power_db = self.power_values[idx].item()
        else:
            power_db = -10.0  # Default: typical normal signal power

        return {
            "iq": iq,
            "label": self.labels[idx],
            "snr": torch.tensor(normalize_snr(snr_db, self.snr_range), dtype=torch.float32),
            "snr_db": torch.tensor(snr_db, dtype=torch.float32),
            "power": torch.tensor(normalize_power(power_db, self.power_range), dtype=torch.float32),
            "power_db": torch.tensor(power_db, dtype=torch.float32),
        }

    @classmethod
    def from_generator(
        cls,
        generator: SyntheticRFGenerator,
        num_samples: int,
        anomaly_ratio: float = 0.0,
        modulations: list[str] | None = None,
        snr_range: tuple[float, float] = (-5, 30),
        anomaly_types: list[str] | None = None,
        augmentor: RFAugmentor | None = None,
        power_range: tuple[float, float] = (-20, 10),
        anomaly_severity: float = 1.0,
    ) -> "RFDataset":
        """Create dataset using synthetic generator.

        Args:
            generator: Signal generator instance.
            num_samples: Number of samples to generate.
            anomaly_ratio: Fraction of anomalous samples.
            modulations: List of modulations to use.
            snr_range: SNR range for generation.
            anomaly_types: List of anomaly types.
            augmentor: Optional augmentation pipeline.
            power_range: Power range for normalization (dB).
            anomaly_severity: Severity multiplier for anomalies (1.0=default).

        Returns:
            RFDataset instance.
        """
        iq_data, metadata = generator.generate_batch(
            num_samples=num_samples,
            anomaly_ratio=anomaly_ratio,
            modulations=modulations,
            snr_range=snr_range,
            anomaly_types=anomaly_types,
            anomaly_severity=anomaly_severity,
        )

        labels = np.array([1 if m.is_anomaly else 0 for m in metadata], dtype=np.int64)
        snr_values = np.array([m.snr_db for m in metadata], dtype=np.float32)
        power_values = np.array([m.signal_power_db or -10.0 for m in metadata], dtype=np.float32)

        return cls(
            iq_data=iq_data,
            labels=labels,
            snr_values=snr_values,
            power_values=power_values,
            metadata=metadata,
            augmentor=augmentor,
            snr_range=snr_range,
            power_range=power_range,
        )

    def save(self, path: str | Path) -> None:
        """Save dataset to disk.

        Args:
            path: Save path (.npz file).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            path,
            iq_data=self.iq_data.numpy(),
            labels=self.labels.numpy(),
            snr_values=self.snr_values.numpy() if self.snr_values is not None else np.array([]),
            power_values=self.power_values.numpy() if self.power_values is not None else np.array([]),
        )

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> "RFDataset":
        """Load dataset from disk.

        Args:
            path: Path to .npz file.
            **kwargs: Additional arguments for RFDataset.

        Returns:
            RFDataset instance.
        """
        data = np.load(path)

        return cls(
            iq_data=data["iq_data"],
            labels=data["labels"],
            snr_values=data["snr_values"] if data["snr_values"].size > 0 else None,
            power_values=data["power_values"] if "power_values" in data and data["power_values"].size > 0 else None,
            **kwargs,
        )


class StreamingRFDataset(IterableDataset):
    """Streaming dataset for continuous learning scenarios.

    Generates data on-the-fly, simulating a continuous stream of RF signals.

    Example:
        dataset = StreamingRFDataset(generator, samples_per_epoch=1000)
        for batch in DataLoader(dataset, batch_size=32):
            # Process streaming data
            pass
    """

    def __init__(
        self,
        generator: SyntheticRFGenerator,
        samples_per_epoch: int = 10000,
        anomaly_ratio: float = 0.1,
        modulations: list[str] | None = None,
        snr_range: tuple[float, float] = (-5, 30),
        anomaly_types: list[str] | None = None,
        augmentor: RFAugmentor | None = None,
        concept_drift: bool = False,
        drift_rate: float = 0.0,
        power_range: tuple[float, float] = (-20, 10),
    ):
        """Initialize streaming dataset.

        Args:
            generator: Signal generator instance.
            samples_per_epoch: Samples to generate per epoch.
            anomaly_ratio: Fraction of anomalous samples.
            modulations: List of modulations to use.
            snr_range: SNR range for generation.
            anomaly_types: List of anomaly types.
            augmentor: Optional augmentation pipeline.
            concept_drift: Enable gradual distribution shift.
            drift_rate: Rate of concept drift per sample.
            power_range: Power range for normalization (dB).
        """
        self.generator = generator
        self.samples_per_epoch = samples_per_epoch
        self.anomaly_ratio = anomaly_ratio
        self.modulations = modulations
        self.snr_range = snr_range
        self.anomaly_types = anomaly_types
        self.augmentor = augmentor
        self.concept_drift = concept_drift
        self.drift_rate = drift_rate
        self.power_range = power_range
        self._sample_count = 0

    def __iter__(self):
        """Iterate over streaming samples."""
        for _ in range(self.samples_per_epoch):
            yield self._generate_sample()

    def _generate_sample(self) -> dict[str, torch.Tensor]:
        """Generate a single streaming sample."""
        self._sample_count += 1

        # Apply concept drift by shifting SNR range
        drift_offset = self._sample_count * self.drift_rate if self.concept_drift else 0
        snr_range = (self.snr_range[0] + drift_offset, self.snr_range[1] + drift_offset)

        # Generate normal or anomalous sample
        is_anomaly = self.generator.rng.random() < self.anomaly_ratio
        modulation = self._random_modulation()

        if is_anomaly:
            iq, metadata = self.generator.generate_anomaly(
                anomaly_type=None,
                base_modulation=modulation,
                snr_range=snr_range,
            )
        else:
            iq, metadata = self.generator.generate_normal_signal(
                modulation=modulation,
                snr_range=snr_range,
            )

        iq = torch.from_numpy(iq)
        if self.augmentor is not None:
            iq = self.augmentor(iq)

        power_db = metadata.signal_power_db or -10.0

        return {
            "iq": iq,
            "label": torch.tensor(int(is_anomaly), dtype=torch.long),
            "snr": torch.tensor(normalize_snr(metadata.snr_db, self.snr_range), dtype=torch.float32),
            "snr_db": torch.tensor(metadata.snr_db, dtype=torch.float32),
            "power": torch.tensor(normalize_power(power_db, self.power_range), dtype=torch.float32),
            "power_db": torch.tensor(power_db, dtype=torch.float32),
        }

    def _random_modulation(self) -> str:
        """Get random modulation from available list."""
        return self.generator.rng.choice(self.modulations) if self.modulations else "qpsk"


def create_dataloaders(
    config,
    generator: SyntheticRFGenerator | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test dataloaders from config.

    Args:
        config: Configuration object with data settings.
        generator: Optional pre-configured generator.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    if generator is None:
        generator = SyntheticRFGenerator(
            sequence_length=config.data.sequence_length,
            sample_rate=config.data.sample_rate,
            seed=config.experiment.seed,
        )

    # Get anomaly severity from config (default 1.0)
    anomaly_severity = getattr(config.data, "anomaly_severity", 1.0)

    common_params = {
        "generator": generator,
        "modulations": config.data.modulations,
        "snr_range": tuple(config.data.snr_range),
        "anomaly_types": config.data.anomaly_types,
        "anomaly_severity": anomaly_severity,
    }

    # Create datasets
    train_dataset = RFDataset.from_generator(
        num_samples=config.data.num_train_samples,
        anomaly_ratio=0.0,  # Train on normal data only
        **common_params,
    )

    val_dataset = RFDataset.from_generator(
        num_samples=config.data.num_val_samples,
        anomaly_ratio=config.data.anomaly_ratio,
        **common_params,
    )

    test_dataset = RFDataset.from_generator(
        num_samples=config.data.num_test_samples,
        anomaly_ratio=config.data.anomaly_ratio,
        **common_params,
    )

    # Create dataloaders with common settings
    loader_params = {
        "batch_size": config.training.batch_size,
        "num_workers": config.experiment.num_workers,
        "pin_memory": True,
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_params)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_params)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_params)

    return train_loader, val_loader, test_loader
