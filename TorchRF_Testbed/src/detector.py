"""Model inference wrapper for live RF anomaly detection."""

from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
import torch
import torch.nn as nn

# Add CLP_Project to path for model imports
CLP_PROJECT_PATH = Path(__file__).resolve().parent.parent.parent
if str(CLP_PROJECT_PATH) not in sys.path:
    sys.path.insert(0, str(CLP_PROJECT_PATH))

# Import from CLP_Project
from src.models.snr_encoder import SNRConditionedVAE, create_model
from src.detection.detector import AnomalyDetector
from src.utils.config import load_config

# Local utility functions (use relative import within this package)
from . import utils as testbed_utils


@dataclass
class DetectionOutput:
    """Output from live detection."""

    score: float
    is_anomaly: bool
    threshold: float
    snr_db: float
    power_db: float


class LiveDetector:
    """Wrapper for loading CLP_Project model and running inference on captured signals.

    Example:
        detector = LiveDetector("path/to/model.pt", "path/to/config.yaml")
        result = detector.detect(complex_signal)
        print(f"Score: {result.score}, Anomaly: {result.is_anomaly}")
    """

    def __init__(
        self,
        model_path: str | Path,
        config_path: str | Path,
        device: str = "cpu",
        snr_range: tuple[float, float] = (-5, 30),
        power_range: tuple[float, float] = (-20, 10),
        threshold_percentile: float = 95.0,
    ):
        """Initialize live detector.

        Args:
            model_path: Path to trained model checkpoint (.pt file).
            config_path: Path to model configuration (.yaml file).
            device: Device to run inference on ("cpu" or "cuda").
            snr_range: Expected SNR range in dB.
            power_range: Expected power range in dB.
            threshold_percentile: Percentile for detection threshold.
        """
        self.device = torch.device(device)
        self.snr_range = snr_range
        self.power_range = power_range
        self.threshold_percentile = threshold_percentile

        # Load configuration
        self.config = load_config(config_path)

        # Create and load model
        self.model = create_model(self.config)

        # Do a dummy forward pass to initialize lazy layers
        seq_len = getattr(self.config.data, "sequence_length", 1024)
        dummy_iq = torch.zeros(1, 2, seq_len)
        dummy_snr = torch.zeros(1)
        dummy_power = torch.zeros(1)
        with torch.no_grad():
            try:
                self.model(dummy_iq, dummy_snr, dummy_power)
            except:
                self.model(dummy_iq, dummy_snr)

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)

        # Handle different checkpoint formats
        if "model_state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            self.model.load_state_dict(checkpoint["state_dict"])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()

        # Create anomaly detector
        method = getattr(self.config.detection, "method", "latent")
        self._detector = AnomalyDetector(
            model=self.model,
            method=method,
            threshold_percentile=threshold_percentile,
            snr_adaptive=getattr(self.config.detection, "snr_adaptive", True),
            snr_range=snr_range,
            device=self.device,
        )

        # Initialize fitted state
        self._is_fitted = False
        self._threshold = 4.24  # Default threshold

        # Model properties
        self._uses_power = hasattr(self.model, "use_power_conditioning") and self.model.use_power_conditioning

    def fit(
        self,
        signals: list[NDArray[np.complex64]] | NDArray[np.complex64],
        num_samples: int | None = None,
    ) -> "LiveDetector":
        """Fit detector on normal signals to learn threshold.

        Args:
            signals: List of normal complex signals or array [N, seq_len].
            num_samples: Max number of samples to use.

        Returns:
            Self for chaining.
        """
        if isinstance(signals, np.ndarray) and signals.ndim == 1:
            signals = [signals]

        if num_samples:
            signals = signals[:num_samples]

        # Process signals and collect scores
        all_scores = []
        with torch.no_grad():
            for signal in signals:
                processed = self._preprocess(signal)
                iq, snr, power = processed["iq"], processed["snr"], processed["power"]
                iq_tensor = torch.from_numpy(iq).unsqueeze(0).to(self.device)
                snr_tensor = torch.tensor([snr]).to(self.device)
                power_tensor = torch.tensor([power]).to(self.device) if self._uses_power else None

                score = self._compute_score(iq_tensor, snr_tensor, power_tensor)
                all_scores.append(score)

        scores_arr = np.array(all_scores)
        self._threshold = float(np.percentile(scores_arr, self.threshold_percentile))
        self._is_fitted = True

        return self

    def _preprocess(self, signal: NDArray[np.complex64]) -> dict:
        """Convert complex signal to model input format.

        Args:
            signal: Complex signal array [seq_len].

        Returns:
            Dict with iq [2, seq_len], normalized snr, normalized power.
        """
        # Ensure correct length (pad/truncate to 1024)
        target_len = getattr(self.config.data, "sequence_length", 1024)
        if len(signal) > target_len:
            signal = signal[:target_len]
        elif len(signal) < target_len:
            signal = np.pad(signal, (0, target_len - len(signal)))

        # Estimate SNR and power before normalization
        snr_db = testbed_utils.estimate_snr(signal)
        power_db = testbed_utils.estimate_power(signal)

        # Normalize signal
        signal_norm, _ = testbed_utils.normalize_signal(signal)

        # Convert to I/Q format
        iq = testbed_utils.complex_to_iq(signal_norm)

        # Normalize SNR and power to [0, 1]
        snr_norm = testbed_utils.normalize_snr_value(snr_db, self.snr_range)
        power_norm = testbed_utils.normalize_power_value(power_db, self.power_range)

        return {
            "iq": iq,
            "snr": snr_norm,
            "power": power_norm,
            "snr_db": snr_db,
            "power_db": power_db,
        }

    def _compute_score(
        self,
        iq: torch.Tensor,
        snr: torch.Tensor,
        power: torch.Tensor | None = None,
    ) -> float:
        """Compute anomaly score for processed signal.

        Args:
            iq: I/Q tensor [1, 2, seq_len].
            snr: Normalized SNR tensor [1].
            power: Normalized power tensor [1] or None.

        Returns:
            Anomaly score.
        """
        # Get latent representation
        if self._uses_power and power is not None:
            mu, logvar = self.model.encode(iq, snr, power)
        else:
            mu, logvar = self.model.encode(iq, snr)

        # Compute Mahalanobis-like distance (simplified for single sample)
        # Use the latent representation directly
        latent = mu.squeeze(0).cpu().numpy()
        score = float(np.linalg.norm(latent))

        return score

    def detect(self, signal: NDArray[np.complex64]) -> DetectionOutput:
        """Run detection on a single signal.

        Args:
            signal: Complex signal array [seq_len].

        Returns:
            DetectionOutput with score, prediction, and metadata.
        """
        processed = self._preprocess(signal)

        with torch.no_grad():
            iq_tensor = torch.from_numpy(processed["iq"]).unsqueeze(0).to(self.device)
            snr_tensor = torch.tensor([processed["snr"]]).to(self.device)
            power_tensor = (
                torch.tensor([processed["power"]]).to(self.device)
                if self._uses_power
                else None
            )

            score = self._compute_score(iq_tensor, snr_tensor, power_tensor)

        return DetectionOutput(
            score=score,
            is_anomaly=score > self._threshold,
            threshold=self._threshold,
            snr_db=processed["snr_db"],
            power_db=processed["power_db"],
        )

    def detect_batch(
        self,
        signals: list[NDArray[np.complex64]] | NDArray[np.complex64],
    ) -> list[DetectionOutput]:
        """Run detection on multiple signals.

        Args:
            signals: List of complex signals or array [N, seq_len].

        Returns:
            List of DetectionOutput objects.
        """
        if isinstance(signals, np.ndarray) and signals.ndim == 1:
            signals = [signals]

        return [self.detect(s) for s in signals]

    def set_threshold(self, threshold: float) -> None:
        """Manually set detection threshold.

        Args:
            threshold: New threshold value.
        """
        self._threshold = threshold

    @property
    def threshold(self) -> float:
        """Current detection threshold."""
        return self._threshold

    @property
    def is_fitted(self) -> bool:
        """Whether detector has been fitted."""
        return self._is_fitted

    def get_info(self) -> dict:
        """Get detector information.

        Returns:
            Dict with model and detector info.
        """
        return {
            "model_type": type(self.model).__name__,
            "device": str(self.device),
            "threshold": self._threshold,
            "threshold_percentile": self.threshold_percentile,
            "is_fitted": self._is_fitted,
            "uses_power_conditioning": self._uses_power,
            "snr_range": self.snr_range,
            "power_range": self.power_range,
            "latent_dim": self.config.model.latent_dim,
        }


def load_detector(
    model_path: str | Path | None = None,
    config_path: str | Path | None = None,
    device: str = "cpu",
) -> LiveDetector:
    """Convenience function to load detector with default paths.

    Args:
        model_path: Path to model checkpoint. Defaults to production model.
        config_path: Path to config. Defaults to default.yaml.
        device: Device to run on.

    Returns:
        Initialized LiveDetector.
    """
    base_path = Path(__file__).parent.parent.parent

    if model_path is None:
        model_path = base_path / "snr_conditioned_vae_hybrid_v1.pt"
    if config_path is None:
        config_path = base_path / "configs" / "default.yaml"

    return LiveDetector(model_path, config_path, device)
