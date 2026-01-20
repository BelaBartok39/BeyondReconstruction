"""HDF5 session recording for labeled RF datasets.

Records captured signals with labels and metadata in an HDF5 format
compatible with the MIT RF dataset structure.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from numpy.typing import NDArray


@dataclass
class RecordingConfig:
    """Configuration for HDF5 recording."""

    sample_rate: float = 2e6
    center_freq: float = 915e6
    chunk_size: int = 100
    compression: str = "gzip"
    compression_level: int = 4
    sequence_length: int = 1024


class SessionRecorder:
    """Record labeled RF sessions to HDF5 files.

    HDF5 Schema (MIT-compatible):
        /signals        - complex64 [N, seq_len]
        /labels         - bool [N] (True=anomaly)
        /anomaly_types  - string [N] (empty string for normal)
        /snr            - float32 [N]
        /power          - float32 [N]
        /timestamps     - float64 [N]
        /scores         - float32 [N] (detection scores, optional)
        /metadata       - group with capture settings

    Example:
        recorder = SessionRecorder("session.h5", sample_rate=2e6, center_freq=915e6)
        recorder.add_sample(signal, label=False)
        recorder.add_sample(anomaly_signal, label=True, anomaly_type="tone")
        recorder.close()
    """

    def __init__(
        self,
        output_path: str | Path,
        sample_rate: float = 2e6,
        center_freq: float = 915e6,
        sequence_length: int = 1024,
        chunk_size: int = 100,
        compression: str = "gzip",
        compression_level: int = 4,
        overwrite: bool = False,
    ):
        """Initialize HDF5 session recorder.

        Args:
            output_path: Path to output HDF5 file.
            sample_rate: Sample rate in Hz.
            center_freq: Center frequency in Hz.
            sequence_length: Number of samples per signal.
            chunk_size: HDF5 chunk size for datasets.
            compression: Compression algorithm (gzip, lzf, None).
            compression_level: Compression level (1-9 for gzip).
            overwrite: If True, overwrite existing file.
        """
        self.output_path = Path(output_path)
        self.config = RecordingConfig(
            sample_rate=sample_rate,
            center_freq=center_freq,
            chunk_size=chunk_size,
            compression=compression,
            compression_level=compression_level,
            sequence_length=sequence_length,
        )

        # Check for existing file
        if self.output_path.exists() and not overwrite:
            raise FileExistsError(f"File {self.output_path} already exists. Use overwrite=True to replace.")

        # Create parent directories if needed
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Open HDF5 file
        self._file = h5py.File(self.output_path, "w")
        self._sample_count = 0
        self._is_open = True
        self._start_time = time.time()

        # Initialize datasets
        self._init_datasets()
        self._write_metadata()

    def _init_datasets(self) -> None:
        """Initialize HDF5 datasets with proper chunking and compression."""
        seq_len = self.config.sequence_length
        chunk = self.config.chunk_size
        comp = self.config.compression if self.config.compression != "None" else None
        comp_level = self.config.compression_level

        compression_opts = {"compression": comp}
        if comp == "gzip":
            compression_opts["compression_opts"] = comp_level

        # Main signal dataset
        self._signals = self._file.create_dataset(
            "signals",
            shape=(0, seq_len),
            maxshape=(None, seq_len),
            dtype=np.complex64,
            chunks=(chunk, seq_len),
            **compression_opts,
        )

        # Label dataset (True = anomaly)
        self._labels = self._file.create_dataset(
            "labels",
            shape=(0,),
            maxshape=(None,),
            dtype=bool,
            chunks=(chunk,),
            **compression_opts,
        )

        # Anomaly type strings
        dt = h5py.string_dtype(encoding="utf-8")
        self._anomaly_types = self._file.create_dataset(
            "anomaly_types",
            shape=(0,),
            maxshape=(None,),
            dtype=dt,
            chunks=(chunk,),
            **compression_opts,
        )

        # SNR values (dB)
        self._snr = self._file.create_dataset(
            "snr",
            shape=(0,),
            maxshape=(None,),
            dtype=np.float32,
            chunks=(chunk,),
            **compression_opts,
        )

        # Power values (dB)
        self._power = self._file.create_dataset(
            "power",
            shape=(0,),
            maxshape=(None,),
            dtype=np.float32,
            chunks=(chunk,),
            **compression_opts,
        )

        # Timestamps (seconds since epoch)
        self._timestamps = self._file.create_dataset(
            "timestamps",
            shape=(0,),
            maxshape=(None,),
            dtype=np.float64,
            chunks=(chunk,),
            **compression_opts,
        )

        # Detection scores (optional, filled later or during recording)
        self._scores = self._file.create_dataset(
            "scores",
            shape=(0,),
            maxshape=(None,),
            dtype=np.float32,
            chunks=(chunk,),
            **compression_opts,
        )

    def _write_metadata(self) -> None:
        """Write capture metadata to HDF5."""
        meta = self._file.create_group("metadata")

        # Capture settings
        meta.attrs["sample_rate"] = self.config.sample_rate
        meta.attrs["center_freq"] = self.config.center_freq
        meta.attrs["sequence_length"] = self.config.sequence_length

        # Recording info
        meta.attrs["created"] = datetime.now().isoformat()
        meta.attrs["format_version"] = "1.0"
        meta.attrs["description"] = "TorchRF Testbed recording"

    def add_sample(
        self,
        signal: NDArray[np.complex64],
        label: bool = False,
        anomaly_type: str | None = None,
        snr_db: float | None = None,
        power_db: float | None = None,
        score: float | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Add a sample to the recording.

        Args:
            signal: Complex signal array [seq_len].
            label: True if anomaly, False if normal.
            anomaly_type: Type of anomaly (if label=True).
            snr_db: Estimated SNR in dB.
            power_db: Estimated power in dB.
            score: Detection score (optional).
            metadata: Additional per-sample metadata (stored as attributes).
        """
        if not self._is_open:
            raise RuntimeError("Recorder is closed. Cannot add samples.")

        # Ensure correct length
        seq_len = self.config.sequence_length
        if len(signal) > seq_len:
            signal = signal[:seq_len]
        elif len(signal) < seq_len:
            signal = np.pad(signal, (0, seq_len - len(signal)))

        # Resize datasets
        n = self._sample_count + 1
        self._signals.resize((n, seq_len))
        self._labels.resize((n,))
        self._anomaly_types.resize((n,))
        self._snr.resize((n,))
        self._power.resize((n,))
        self._timestamps.resize((n,))
        self._scores.resize((n,))

        # Write data
        idx = self._sample_count
        self._signals[idx] = signal.astype(np.complex64)
        self._labels[idx] = label
        self._anomaly_types[idx] = anomaly_type or ""
        self._snr[idx] = snr_db if snr_db is not None else 0.0
        self._power[idx] = power_db if power_db is not None else 0.0
        self._timestamps[idx] = time.time()
        self._scores[idx] = score if score is not None else 0.0

        self._sample_count += 1

        # Store per-sample metadata if provided
        if metadata:
            sample_meta = self._file.require_group("sample_metadata")
            sample_grp = sample_meta.create_group(str(idx))
            for key, value in metadata.items():
                sample_grp.attrs[key] = value

    def add_batch(
        self,
        signals: NDArray[np.complex64],
        labels: NDArray[np.bool_],
        anomaly_types: list[str] | None = None,
        snr_db: NDArray[np.float32] | None = None,
        power_db: NDArray[np.float32] | None = None,
        scores: NDArray[np.float32] | None = None,
    ) -> None:
        """Add a batch of samples efficiently.

        Args:
            signals: Complex signals [N, seq_len].
            labels: Labels [N].
            anomaly_types: Anomaly types list [N].
            snr_db: SNR values [N].
            power_db: Power values [N].
            scores: Detection scores [N].
        """
        if not self._is_open:
            raise RuntimeError("Recorder is closed. Cannot add samples.")

        batch_size = len(signals)
        seq_len = self.config.sequence_length

        # Validate shapes
        if signals.shape[1] != seq_len:
            raise ValueError(f"Signal length {signals.shape[1]} != expected {seq_len}")
        if len(labels) != batch_size:
            raise ValueError(f"Labels length {len(labels)} != batch size {batch_size}")

        # Resize datasets
        n = self._sample_count + batch_size
        self._signals.resize((n, seq_len))
        self._labels.resize((n,))
        self._anomaly_types.resize((n,))
        self._snr.resize((n,))
        self._power.resize((n,))
        self._timestamps.resize((n,))
        self._scores.resize((n,))

        # Write batch
        start_idx = self._sample_count
        end_idx = n
        self._signals[start_idx:end_idx] = signals.astype(np.complex64)
        self._labels[start_idx:end_idx] = labels

        if anomaly_types is not None:
            for i, atype in enumerate(anomaly_types):
                self._anomaly_types[start_idx + i] = atype or ""
        else:
            for i in range(batch_size):
                self._anomaly_types[start_idx + i] = ""

        if snr_db is not None:
            self._snr[start_idx:end_idx] = snr_db
        else:
            self._snr[start_idx:end_idx] = 0.0

        if power_db is not None:
            self._power[start_idx:end_idx] = power_db
        else:
            self._power[start_idx:end_idx] = 0.0

        self._timestamps[start_idx:end_idx] = time.time()

        if scores is not None:
            self._scores[start_idx:end_idx] = scores
        else:
            self._scores[start_idx:end_idx] = 0.0

        self._sample_count = n

    def flush(self) -> None:
        """Flush data to disk."""
        if self._is_open:
            self._file.flush()

    def close(self) -> None:
        """Finalize and close the HDF5 file."""
        if not self._is_open:
            return

        # Update final metadata
        meta = self._file["metadata"]
        meta.attrs["total_samples"] = self._sample_count
        meta.attrs["num_anomalies"] = int(np.sum(self._labels[:]))
        meta.attrs["num_normal"] = self._sample_count - meta.attrs["num_anomalies"]
        meta.attrs["duration_seconds"] = time.time() - self._start_time
        meta.attrs["closed"] = datetime.now().isoformat()

        self._file.close()
        self._is_open = False

    @property
    def sample_count(self) -> int:
        """Number of samples recorded."""
        return self._sample_count

    @property
    def is_open(self) -> bool:
        """Whether the file is open for writing."""
        return self._is_open

    def get_stats(self) -> dict:
        """Get recording statistics.

        Returns:
            Dict with recording statistics.
        """
        return {
            "output_path": str(self.output_path),
            "sample_count": self._sample_count,
            "num_anomalies": int(np.sum(self._labels[:])) if self._sample_count > 0 else 0,
            "duration_seconds": time.time() - self._start_time,
            "is_open": self._is_open,
            "sample_rate": self.config.sample_rate,
            "center_freq": self.config.center_freq,
        }

    def __enter__(self) -> "SessionRecorder":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()


class SessionReader:
    """Read recorded HDF5 sessions.

    Example:
        reader = SessionReader("session.h5")
        for signal, label, anomaly_type in reader:
            process(signal, label)
        reader.close()
    """

    def __init__(self, file_path: str | Path):
        """Initialize session reader.

        Args:
            file_path: Path to HDF5 file.
        """
        self.file_path = Path(file_path)
        if not self.file_path.exists():
            raise FileNotFoundError(f"File not found: {self.file_path}")

        self._file = h5py.File(self.file_path, "r")
        self._signals = self._file["signals"]
        self._labels = self._file["labels"]
        self._anomaly_types = self._file["anomaly_types"]
        self._snr = self._file["snr"]
        self._power = self._file["power"]
        self._timestamps = self._file["timestamps"]
        self._scores = self._file.get("scores")
        self._is_open = True

    def __len__(self) -> int:
        """Number of samples in the file."""
        return len(self._signals)

    def __getitem__(self, idx: int | slice) -> dict:
        """Get sample(s) by index.

        Args:
            idx: Sample index or slice.

        Returns:
            Dict with signal, label, anomaly_type, snr, power, timestamp, score.
        """
        return {
            "signal": self._signals[idx],
            "label": self._labels[idx],
            "anomaly_type": self._anomaly_types[idx],
            "snr_db": self._snr[idx],
            "power_db": self._power[idx],
            "timestamp": self._timestamps[idx],
            "score": self._scores[idx] if self._scores is not None else None,
        }

    def __iter__(self):
        """Iterate over samples."""
        for i in range(len(self)):
            yield self[i]

    def get_signals(self) -> NDArray[np.complex64]:
        """Get all signals.

        Returns:
            Complex signal array [N, seq_len].
        """
        return self._signals[:]

    def get_labels(self) -> NDArray[np.bool_]:
        """Get all labels.

        Returns:
            Label array [N].
        """
        return self._labels[:]

    def get_metadata(self) -> dict:
        """Get recording metadata.

        Returns:
            Dict with metadata attributes.
        """
        meta = self._file["metadata"]
        return dict(meta.attrs)

    def close(self) -> None:
        """Close the HDF5 file."""
        if self._is_open:
            self._file.close()
            self._is_open = False

    def __enter__(self) -> "SessionReader":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
