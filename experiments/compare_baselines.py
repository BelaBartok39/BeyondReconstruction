#!/usr/bin/env python3
"""Baseline comparison script for Phase 2 research validation.

Compares VAE-based anomaly detection with traditional baselines:
- One-Class SVM on engineered features
- Isolation Forest on latent space
- Amplitude threshold (simple baseline)

Includes statistical significance testing (Wilcoxon, bootstrap CI).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.datasets import RFDataset
from src.data.synthetic import SyntheticRFGenerator
from src.detection.detector import AnomalyDetector
from src.detection.metrics import compute_metrics, MetricsResult
from src.models.snr_encoder import create_model
from src.utils.config import load_config


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class BaselineResult:
    """Container for baseline comparison results."""

    name: str
    auroc: float
    auprc: float
    f1: float
    precision: float
    recall: float
    cohens_d: float
    auroc_ci_low: float
    auroc_ci_high: float
    scores: NDArray[np.float32]
    labels: NDArray[np.int64]

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "name": self.name,
            "auroc": float(self.auroc),
            "auprc": float(self.auprc),
            "f1": float(self.f1),
            "precision": float(self.precision),
            "recall": float(self.recall),
            "cohens_d": float(self.cohens_d),
            "auroc_ci_low": float(self.auroc_ci_low),
            "auroc_ci_high": float(self.auroc_ci_high),
        }


def extract_engineered_features(iq: NDArray[np.float32]) -> NDArray[np.float32]:
    """Extract engineered features from raw I/Q signals for One-Class SVM.

    Features are designed to be robust and avoid curse of dimensionality.

    Args:
        iq: I/Q signals [batch, 2, seq_len].

    Returns:
        Feature matrix [batch, num_features].
    """
    batch_size = iq.shape[0]
    features = []

    for i in range(batch_size):
        signal = iq[i]  # [2, seq_len]
        i_channel = signal[0]
        q_channel = signal[1]

        # Complex signal
        complex_signal = i_channel + 1j * q_channel
        amplitude = np.abs(complex_signal)
        phase = np.angle(complex_signal)

        # Amplitude statistics
        mean_amp = np.mean(amplitude)
        std_amp = np.std(amplitude)
        peak_amp = np.max(amplitude)
        min_amp = np.min(amplitude)

        # Power statistics
        power = amplitude ** 2
        mean_power = np.mean(power)
        power_db = 10 * np.log10(mean_power + 1e-10)
        power_var = np.var(power)

        # Spectral features (via FFT)
        fft = np.fft.fft(complex_signal)
        fft_mag = np.abs(fft)
        fft_mag_norm = fft_mag / (np.sum(fft_mag) + 1e-10)

        # Spectral centroid
        freqs = np.fft.fftfreq(len(complex_signal))
        spectral_centroid = np.sum(freqs * fft_mag_norm)

        # Spectral bandwidth
        spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * fft_mag_norm))

        # Spectral flatness (Wiener entropy)
        geo_mean = np.exp(np.mean(np.log(fft_mag + 1e-10)))
        arith_mean = np.mean(fft_mag)
        spectral_flatness = geo_mean / (arith_mean + 1e-10)

        # Higher-order statistics
        kurtosis = stats.kurtosis(amplitude)
        skewness = stats.skew(amplitude)

        # Phase statistics
        phase_unwrapped = np.unwrap(phase)
        phase_diff = np.diff(phase_unwrapped)
        phase_std = np.std(phase_diff)

        # Crest factor (peak to RMS ratio)
        rms = np.sqrt(mean_power)
        crest_factor = peak_amp / (rms + 1e-10)

        # Dynamic range
        dynamic_range = peak_amp / (min_amp + 1e-10)

        # Assemble feature vector
        sample_features = np.array([
            mean_amp,
            std_amp,
            peak_amp,
            power_db,
            power_var,
            spectral_centroid,
            spectral_bandwidth,
            spectral_flatness,
            kurtosis,
            skewness,
            phase_std,
            crest_factor,
            np.log(dynamic_range + 1),  # Log scale for stability
        ])

        features.append(sample_features)

    return np.array(features, dtype=np.float32)


def extract_latents(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    """Extract latent vectors from VAE encoder.

    Args:
        model: Trained VAE model.
        dataloader: Data loader with samples.
        device: Torch device.

    Returns:
        Tuple of (latents, labels).
    """
    model.eval()
    all_latents = []
    all_labels = []

    is_snr_conditioned = hasattr(model, "encoder") and hasattr(model.encoder, "cond_embed")
    uses_power = hasattr(model, "use_power_conditioning") and model.use_power_conditioning

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting latents", leave=False):
            iq = batch["iq"].to(device)
            snr = batch.get("snr")
            power = batch.get("power")

            if snr is not None:
                snr = snr.to(device)
            if power is not None:
                power = power.to(device)

            # Get latent representation
            if is_snr_conditioned and snr is not None:
                if uses_power and power is not None:
                    mu, _ = model.encode(iq, snr, power)
                else:
                    mu, _ = model.encode(iq, snr)
            else:
                mu, _ = model.encode(iq)

            all_latents.append(mu.cpu().numpy())
            all_labels.append(batch["label"].numpy())

    return np.concatenate(all_latents), np.concatenate(all_labels)


def extract_raw_iq(dataloader: DataLoader) -> tuple[NDArray[np.float32], NDArray[np.int64]]:
    """Extract raw I/Q data from dataloader.

    Args:
        dataloader: Data loader with samples.

    Returns:
        Tuple of (iq_data, labels).
    """
    all_iq = []
    all_labels = []

    for batch in tqdm(dataloader, desc="Extracting I/Q", leave=False):
        all_iq.append(batch["iq"].numpy())
        all_labels.append(batch["label"].numpy())

    return np.concatenate(all_iq), np.concatenate(all_labels)


def run_ocsvm_baseline(
    features_train: NDArray[np.float32],
    features_test: NDArray[np.float32],
    labels_test: NDArray[np.int64],
    nu: float = 0.1,
) -> BaselineResult:
    """Run One-Class SVM baseline on engineered features.

    Args:
        features_train: Training features (normal only).
        features_test: Test features.
        labels_test: Test labels.
        nu: Expected anomaly fraction (nu parameter for OC-SVM).

    Returns:
        BaselineResult with metrics.
    """
    logger.info("Training One-Class SVM...")

    # Scale features
    scaler = StandardScaler()
    features_train_scaled = scaler.fit_transform(features_train)
    features_test_scaled = scaler.transform(features_test)

    # Train OC-SVM
    ocsvm = OneClassSVM(kernel="rbf", nu=nu, gamma="auto")
    ocsvm.fit(features_train_scaled)

    # Get anomaly scores (negative of decision function, so higher = more anomalous)
    scores = -ocsvm.decision_function(features_test_scaled)

    return _compute_baseline_result("One-Class SVM (features)", scores, labels_test)


def run_isolation_forest(
    latents_train: NDArray[np.float32],
    latents_test: NDArray[np.float32],
    labels_test: NDArray[np.int64],
    contamination: float = 0.1,
    n_estimators: int = 100,
) -> BaselineResult:
    """Run Isolation Forest on latent space.

    Args:
        latents_train: Training latent vectors (normal only).
        latents_test: Test latent vectors.
        labels_test: Test labels.
        contamination: Expected anomaly ratio.
        n_estimators: Number of trees.

    Returns:
        BaselineResult with metrics.
    """
    logger.info("Training Isolation Forest on latent space...")

    iforest = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=42,
    )
    iforest.fit(latents_train)

    # Get anomaly scores (negative of decision function)
    scores = -iforest.decision_function(latents_test)

    return _compute_baseline_result("Isolation Forest (latent)", scores, labels_test)


def run_amplitude_threshold(
    iq_test: NDArray[np.float32],
    labels_test: NDArray[np.int64],
) -> BaselineResult:
    """Run simple amplitude threshold baseline.

    Args:
        iq_test: Test I/Q signals.
        labels_test: Test labels.

    Returns:
        BaselineResult with metrics.
    """
    logger.info("Computing amplitude threshold baseline...")

    # Peak amplitude as anomaly score
    scores = np.max(np.abs(iq_test[:, 0] + 1j * iq_test[:, 1]), axis=1)

    return _compute_baseline_result("Amplitude Threshold", scores, labels_test)


def run_vae_latent_detector(
    model: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> BaselineResult:
    """Run VAE latent-space detector (Mahalanobis distance).

    Args:
        model: Trained VAE model.
        train_loader: Training data loader (for fitting).
        test_loader: Test data loader.
        device: Torch device.

    Returns:
        BaselineResult with metrics.
    """
    logger.info("Running VAE Latent (Mahalanobis) detector...")

    detector = AnomalyDetector(
        model=model,
        method="latent",
        threshold_percentile=95.0,
        device=device,
    )

    # Fit on training data
    detector.fit(train_loader, num_batches=50)

    # Get scores on test data
    scores, _, labels = detector.detect_batch(test_loader)

    return _compute_baseline_result("VAE Latent (Mahalanobis)", scores, labels)


def run_vae_reconstruction_detector(
    model: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> BaselineResult:
    """Run VAE reconstruction-based detector.

    Args:
        model: Trained VAE model.
        train_loader: Training data loader (for fitting).
        test_loader: Test data loader.
        device: Torch device.

    Returns:
        BaselineResult with metrics.
    """
    logger.info("Running VAE Reconstruction detector...")

    detector = AnomalyDetector(
        model=model,
        method="reconstruction",
        threshold_percentile=95.0,
        device=device,
    )

    # Fit on training data
    detector.fit(train_loader, num_batches=50)

    # Get scores on test data
    scores, _, labels = detector.detect_batch(test_loader)

    return _compute_baseline_result("VAE Reconstruction", scores, labels)


def _compute_baseline_result(
    name: str,
    scores: NDArray[np.float32],
    labels: NDArray[np.int64],
) -> BaselineResult:
    """Compute metrics and create BaselineResult.

    Args:
        name: Baseline name.
        scores: Anomaly scores.
        labels: Ground truth labels.

    Returns:
        BaselineResult with all metrics.
    """
    metrics = compute_metrics(scores, labels)

    # Compute Cohen's d effect size
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]

    pooled_std = np.sqrt((np.var(normal_scores) + np.var(anomaly_scores)) / 2)
    cohens_d = (np.mean(anomaly_scores) - np.mean(normal_scores)) / (pooled_std + 1e-10)

    # Bootstrap 95% CI for AUROC
    auroc_ci_low, auroc_ci_high = bootstrap_auroc_ci(scores, labels)

    return BaselineResult(
        name=name,
        auroc=metrics.auroc,
        auprc=metrics.auprc,
        f1=metrics.f1,
        precision=metrics.precision,
        recall=metrics.recall,
        cohens_d=cohens_d,
        auroc_ci_low=auroc_ci_low,
        auroc_ci_high=auroc_ci_high,
        scores=scores,
        labels=labels,
    )


def bootstrap_auroc_ci(
    scores: NDArray[np.float32],
    labels: NDArray[np.int64],
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
) -> tuple[float, float]:
    """Compute bootstrap confidence interval for AUROC.

    Args:
        scores: Anomaly scores.
        labels: Ground truth labels.
        n_bootstrap: Number of bootstrap samples.
        confidence: Confidence level.

    Returns:
        Tuple of (lower_bound, upper_bound).
    """
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(42)
    aurocs = []

    n_samples = len(scores)
    for _ in range(n_bootstrap):
        idx = rng.choice(n_samples, size=n_samples, replace=True)
        boot_scores = scores[idx]
        boot_labels = labels[idx]

        # Skip if only one class in bootstrap sample
        if len(np.unique(boot_labels)) < 2:
            continue

        aurocs.append(roc_auc_score(boot_labels, boot_scores))

    alpha = (1 - confidence) / 2
    return float(np.percentile(aurocs, 100 * alpha)), float(np.percentile(aurocs, 100 * (1 - alpha)))


def run_statistical_tests(results: list[BaselineResult]) -> dict:
    """Run statistical significance tests between methods.

    Args:
        results: List of BaselineResult objects.

    Returns:
        Dictionary with test results.
    """
    logger.info("Running statistical significance tests...")

    tests = {}
    reference = results[0]  # VAE Latent (Mahalanobis) as reference

    for result in results[1:]:
        key = f"{reference.name} vs {result.name}"

        # Wilcoxon signed-rank test on scores
        # Test if score distributions are significantly different
        # We compare score differences for each sample
        score_diff = reference.scores - result.scores
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(score_diff, alternative="two-sided")

        # Mann-Whitney U test (non-parametric)
        # Compare anomaly scores between the two methods
        mw_stat, mw_p = stats.mannwhitneyu(
            reference.scores, result.scores, alternative="two-sided"
        )

        # Independent t-test on anomaly score distributions
        t_stat, t_p = stats.ttest_ind(reference.scores, result.scores)

        tests[key] = {
            "wilcoxon_statistic": float(wilcoxon_stat),
            "wilcoxon_p_value": float(wilcoxon_p),
            "mannwhitney_statistic": float(mw_stat),
            "mannwhitney_p_value": float(mw_p),
            "ttest_statistic": float(t_stat),
            "ttest_p_value": float(t_p),
            "auroc_diff": float(reference.auroc - result.auroc),
            "significant_at_0.05": bool(wilcoxon_p < 0.05),
        }

    return tests


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Compare baseline anomaly detectors")
    parser.add_argument(
        "--checkpoint",
        default="snr_conditioned_vae_hybrid_v1.pt",
        help="Path to trained VAE checkpoint",
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config file (default: same directory as checkpoint)",
    )
    parser.add_argument(
        "--output-dir",
        default="experiments/results/baseline_comparison",
        help="Output directory for results",
    )
    parser.add_argument(
        "--num-test-samples",
        type=int,
        default=5000,
        help="Number of test samples",
    )
    parser.add_argument(
        "--anomaly-ratio",
        type=float,
        default=0.2,
        help="Fraction of anomalous test samples",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    return parser.parse_args()


def get_device() -> torch.device:
    """Get best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def main() -> None:
    """Main comparison function."""
    args = parse_args()

    # Setup
    device = get_device()
    logger.info(f"Using device: {device}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find config
    checkpoint_path = Path(args.checkpoint)
    config_path = args.config or checkpoint_path.parent / "config.yaml"
    if not Path(config_path).exists():
        config_path = "configs/default.yaml"

    config = load_config(config_path)
    logger.info(f"Using config: {config_path}")

    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    model = create_model(config)
    model = model.to(device)

    # Initialize lazy layers with dummy forward pass
    dummy_iq = torch.randn(1, 2, config.data.sequence_length, device=device)
    dummy_snr = torch.tensor([0.5], device=device)  # Normalized SNR
    dummy_power = torch.tensor([0.5], device=device) if model.use_power_conditioning else None
    with torch.no_grad():
        if dummy_power is not None:
            _ = model(dummy_iq, dummy_snr, dummy_power)
        else:
            _ = model(dummy_iq, dummy_snr)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state_dict = checkpoint["model_state_dict"]

    # Check if legacy naming is used (snr_embed instead of cond_embed)
    uses_legacy_naming = any("snr_embed" in k for k in state_dict.keys())

    if uses_legacy_naming:
        # Handle legacy checkpoint naming (snr_embed -> cond_embed, final_conv -> final_mean)
        key_mapping = {
            "encoder.snr_embed": "encoder.cond_embed",
            "decoder.snr_embed": "decoder.cond_embed",
            "decoder.final_conv": "decoder.final_mean",
        }
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            for old_prefix, new_prefix in key_mapping.items():
                if key.startswith(old_prefix):
                    new_key = key.replace(old_prefix, new_prefix)
                    break
            new_state_dict[new_key] = value
        state_dict = new_state_dict
        logger.info("Converted legacy checkpoint naming to current format")

    model.load_state_dict(state_dict)
    model.eval()

    # Create data generator
    logger.info("Creating synthetic data...")
    generator = SyntheticRFGenerator(
        sequence_length=config.data.sequence_length,
        sample_rate=config.data.sample_rate,
        seed=args.seed + 1000,  # Different from training
    )

    anomaly_severity = getattr(config.data, "anomaly_severity", 4.0)

    # Create training data (normal only, for fitting baselines)
    train_dataset = RFDataset.from_generator(
        generator=generator,
        num_samples=2000,  # Enough for fitting
        anomaly_ratio=0.0,  # Normal only
        snr_range=tuple(config.data.snr_range),
        modulations=config.data.modulations,
        anomaly_severity=anomaly_severity,
    )

    # Create test data (mixed normal/anomaly)
    test_dataset = RFDataset.from_generator(
        generator=generator,
        num_samples=args.num_test_samples,
        anomaly_ratio=args.anomaly_ratio,
        snr_range=tuple(config.data.snr_range),
        modulations=config.data.modulations,
        anomaly_types=config.data.anomaly_types,
        anomaly_severity=anomaly_severity,
    )

    train_loader = DataLoader(train_dataset, batch_size=config.training.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.training.batch_size, shuffle=False)

    logger.info(f"Training samples: {len(train_dataset)} (normal only)")
    logger.info(f"Test samples: {len(test_dataset)} ({args.anomaly_ratio:.0%} anomalies)")

    # Extract data
    iq_train, labels_train = extract_raw_iq(train_loader)
    iq_test, labels_test = extract_raw_iq(test_loader)

    # Extract engineered features for OC-SVM
    logger.info("Extracting engineered features...")
    features_train = extract_engineered_features(iq_train)
    features_test = extract_engineered_features(iq_test)

    # Extract latent vectors for Isolation Forest
    latents_train, _ = extract_latents(model, train_loader, device)
    latents_test, _ = extract_latents(model, test_loader, device)

    # Run all baselines
    logger.info("\n" + "=" * 60)
    logger.info("RUNNING BASELINE COMPARISONS")
    logger.info("=" * 60)

    results = []

    # 1. VAE Latent (Mahalanobis) - our best method
    results.append(run_vae_latent_detector(model, train_loader, test_loader, device))

    # 2. Isolation Forest on latent space
    results.append(run_isolation_forest(
        latents_train, latents_test, labels_test,
        contamination=args.anomaly_ratio,
    ))

    # 3. One-Class SVM on engineered features
    results.append(run_ocsvm_baseline(
        features_train, features_test, labels_test,
        nu=args.anomaly_ratio,
    ))

    # 4. Simple amplitude threshold
    results.append(run_amplitude_threshold(iq_test, labels_test))

    # 5. VAE Reconstruction (known to be poor)
    results.append(run_vae_reconstruction_detector(model, train_loader, test_loader, device))

    # Run statistical tests
    stat_tests = run_statistical_tests(results)

    # Print results table
    logger.info("\n" + "=" * 80)
    logger.info("BASELINE COMPARISON RESULTS")
    logger.info("=" * 80)
    logger.info("")
    header = f"{'Method':<30} | {'AUROC':>8} | {'AUPRC':>8} | {'F1':>6} | {'Cohen d':>8} | {'95% CI':>15}"
    logger.info(header)
    logger.info("-" * 80)

    for r in results:
        ci_str = f"[{r.auroc_ci_low:.3f}, {r.auroc_ci_high:.3f}]"
        row = f"{r.name:<30} | {r.auroc:>8.4f} | {r.auprc:>8.4f} | {r.f1:>6.3f} | {r.cohens_d:>8.2f} | {ci_str:>15}"
        logger.info(row)

    logger.info("-" * 80)

    # Print statistical significance
    logger.info("\nStatistical Significance (Wilcoxon signed-rank test):")
    for comparison, test_result in stat_tests.items():
        sig_str = "***" if test_result["wilcoxon_p_value"] < 0.001 else (
            "**" if test_result["wilcoxon_p_value"] < 0.01 else (
                "*" if test_result["wilcoxon_p_value"] < 0.05 else ""
            )
        )
        logger.info(
            f"  {comparison}: p={test_result['wilcoxon_p_value']:.4f} "
            f"(ΔAUROC={test_result['auroc_diff']:+.4f}) {sig_str}"
        )

    # Save results
    output = {
        "results": [r.to_dict() for r in results],
        "statistical_tests": stat_tests,
        "config": {
            "checkpoint": str(args.checkpoint),
            "num_test_samples": args.num_test_samples,
            "anomaly_ratio": args.anomaly_ratio,
            "seed": args.seed,
        },
    }

    output_path = output_dir / "baseline_comparison_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logger.info(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
