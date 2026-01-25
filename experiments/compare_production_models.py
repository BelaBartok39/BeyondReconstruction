#!/usr/bin/env python3
"""Compare production models v1 vs v2 on multiple datasets.

Compares:
- snr_vae_hybrid_v1_20260118 (original production model)
- snr_vae_hybrid_v2_20260125 (reproduced model)

On datasets:
- Synthetic test data (all anomaly types)
- POWDER DSSS dataset (unseen anomaly type)
- HackRF live captured data

Saves visualizations to: figures/model_comparison_v2_20260125/
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

import glob
import h5py
import json
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.snr_encoder import create_model
from src.data.synthetic import SyntheticRFGenerator
from src.data.datasets import RFDataset
from src.data.snr_estimation import estimate_snr, normalize_snr
from src.detection.detector import AnomalyDetector
from src.detection.phase_detector import ChirpDetector
from src.detection.metrics import compute_metrics
from src.utils.config import load_config


# Paths
PROJECT_ROOT = Path(__file__).parent.parent
CHECKPOINT_V1 = PROJECT_ROOT / "checkpoints" / "snr_vae_hybrid_v1_20260118" / "best_model.pt"
CHECKPOINT_V2 = PROJECT_ROOT / "checkpoints" / "snr_vae_hybrid_v2_20260125" / "best_model.pt"
CONFIG_PATH = PROJECT_ROOT / "configs" / "default.yaml"
POWDER_PATH = PROJECT_ROOT / "RED_DATA" / "POWDER_Dataset"
HACKRF_PATH = PROJECT_ROOT / "TorchRF_Testbed" / "data" / "hackrf_dataset.h5"
OUTPUT_DIR = PROJECT_ROOT / "figures" / "model_comparison_v2_20260125"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(checkpoint_path: Path, config, device: torch.device) -> torch.nn.Module:
    """Load a trained model."""
    model = create_model(config)
    model = model.to(device)

    # Initialize lazy layers
    dummy_iq = torch.randn(1, 2, config.data.sequence_length, device=device)
    dummy_snr = torch.rand(1, device=device)
    dummy_power = torch.rand(1, device=device)
    with torch.no_grad():
        _ = model(dummy_iq, dummy_snr, dummy_power)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def normalize_power(power_db: np.ndarray, power_range: tuple[float, float] = (-40, 0)) -> np.ndarray:
    """Normalize power to [0, 1] range."""
    low, high = power_range
    return np.clip((power_db - low) / (high - low), 0, 1).astype(np.float32)


def get_latent_scores(
    model: torch.nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Get latent-based anomaly scores."""
    detector = AnomalyDetector(
        model=model,
        method="latent",
        threshold_method="percentile",
        threshold_percentile=95,
        snr_adaptive=True,
        snr_bins=7,
        device=device,
    )
    detector.fit(train_loader, num_batches=50)

    all_scores, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            iq = batch["iq"].to(device)
            snr = batch.get("snr")
            if snr is not None:
                snr = snr.to(device)
            snr_db = batch.get("snr_db")
            power = batch.get("power")
            if power is not None:
                power = power.to(device)

            result = detector.detect(iq, snr, snr_db, power)
            all_scores.append(result.scores)
            all_labels.append(batch["label"].numpy())

    return np.concatenate(all_scores), np.concatenate(all_labels)


def get_hybrid_scores(
    latent_scores: np.ndarray,
    test_iq: np.ndarray,
    train_iq: np.ndarray,
    freq_weight: float = 0.6,
) -> np.ndarray:
    """Combine latent and chirp detector scores."""
    chirp_det = ChirpDetector()
    chirp_det.fit(train_iq)
    chirp_scores = chirp_det.score(test_iq)

    def normalize(s):
        return (s - s.min()) / (s.max() - s.min() + 1e-8)

    return (1 - freq_weight) * normalize(latent_scores) + freq_weight * normalize(chirp_scores)


# ============ Synthetic Data Evaluation ============

def evaluate_synthetic(
    model_v1: torch.nn.Module,
    model_v2: torch.nn.Module,
    config,
    device: torch.device,
) -> dict:
    """Evaluate both models on synthetic data."""
    print("\n" + "=" * 60)
    print("SYNTHETIC DATA EVALUATION")
    print("=" * 60)

    generator = SyntheticRFGenerator(
        sequence_length=config.data.sequence_length,
        sample_rate=config.data.sample_rate,
        seed=42,
    )

    # Training data (normal only)
    train_dataset = RFDataset.from_generator(
        generator=generator,
        num_samples=5000,
        anomaly_ratio=0.0,
        snr_range=tuple(config.data.snr_range),
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    train_iq = np.concatenate([b["iq"].numpy() for b in train_loader])

    # Test data (with anomalies)
    test_dataset = RFDataset.from_generator(
        generator=generator,
        num_samples=2000,
        anomaly_ratio=0.2,
        snr_range=tuple(config.data.snr_range),
        anomaly_types=config.data.anomaly_types,
        anomaly_severity=config.data.anomaly_severity,
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    test_iq = np.concatenate([b["iq"].numpy() for b in test_loader])

    results = {"v1": {}, "v2": {}}

    for name, model in [("v1", model_v1), ("v2", model_v2)]:
        print(f"\n--- Model {name} ---")

        # Latent scores
        latent_scores, labels = get_latent_scores(model, train_loader, test_loader, device)
        latent_auroc = roc_auc_score(labels, latent_scores)

        # Hybrid scores
        hybrid_scores = get_hybrid_scores(latent_scores, test_iq, train_iq, freq_weight=0.6)
        hybrid_auroc = roc_auc_score(labels, hybrid_scores)

        results[name] = {
            "latent_auroc": latent_auroc,
            "hybrid_auroc": hybrid_auroc,
            "latent_scores": latent_scores,
            "hybrid_scores": hybrid_scores,
            "labels": labels,
        }

        print(f"  Latent AUROC: {latent_auroc:.4f}")
        print(f"  Hybrid AUROC: {hybrid_auroc:.4f}")

    return results


# ============ Per-Anomaly Type Evaluation ============

def evaluate_per_anomaly_type(
    model_v1: torch.nn.Module,
    model_v2: torch.nn.Module,
    config,
    device: torch.device,
) -> dict:
    """Evaluate both models on each anomaly type."""
    print("\n" + "=" * 60)
    print("PER-ANOMALY TYPE EVALUATION")
    print("=" * 60)

    generator = SyntheticRFGenerator(
        sequence_length=config.data.sequence_length,
        sample_rate=config.data.sample_rate,
        seed=42,
    )

    # Training data
    train_dataset = RFDataset.from_generator(
        generator=generator,
        num_samples=5000,
        anomaly_ratio=0.0,
        snr_range=tuple(config.data.snr_range),
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)
    train_iq = np.concatenate([b["iq"].numpy() for b in train_loader])

    anomaly_types = ["frequency_drift", "interference", "amplitude_spike", "phase_noise"]
    results = {atype: {"v1": {}, "v2": {}} for atype in anomaly_types}

    for atype in anomaly_types:
        print(f"\n--- {atype.upper()} ---")

        test_dataset = RFDataset.from_generator(
            generator=generator,
            num_samples=1000,
            anomaly_ratio=0.1,
            snr_range=tuple(config.data.snr_range),
            anomaly_types=[atype],
            anomaly_severity=config.data.anomaly_severity,
        )
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
        test_iq = np.concatenate([b["iq"].numpy() for b in test_loader])

        for name, model in [("v1", model_v1), ("v2", model_v2)]:
            latent_scores, labels = get_latent_scores(model, train_loader, test_loader, device)
            hybrid_scores = get_hybrid_scores(latent_scores, test_iq, train_iq, freq_weight=0.6)

            results[atype][name] = {
                "latent_auroc": roc_auc_score(labels, latent_scores),
                "hybrid_auroc": roc_auc_score(labels, hybrid_scores),
            }

        print(f"  V1: Latent={results[atype]['v1']['latent_auroc']:.4f}, Hybrid={results[atype]['v1']['hybrid_auroc']:.4f}")
        print(f"  V2: Latent={results[atype]['v2']['latent_auroc']:.4f}, Hybrid={results[atype]['v2']['hybrid_auroc']:.4f}")

    return results


# ============ POWDER Dataset Evaluation ============

def load_powder_file(filepath: str, window_size: int = 1024, max_windows: int = 100):
    """Load a single POWDER file."""
    data = np.fromfile(filepath, dtype=np.complex64)
    if np.abs(data[0]) < 1e-30:
        data = data[1:]

    n_windows = min(len(data) // window_size, max_windows)
    iq_list, snr_list, power_list = [], [], []

    for i in range(n_windows):
        segment = data[i * window_size : (i + 1) * window_size]
        power = np.mean(np.abs(segment) ** 2)
        power_list.append(10 * np.log10(power + 1e-10))

        try:
            snr_db = estimate_snr(segment, method="m2m4")
        except Exception:
            snr_db = 10.0
        snr_list.append(snr_db)

        max_amp = np.max(np.abs(segment)) + 1e-8
        segment_norm = segment / max_amp
        iq = np.stack([segment_norm.real, segment_norm.imag], axis=0).astype(np.float32)
        iq_list.append(iq)

    return np.array(iq_list), np.array(snr_list, dtype=np.float32), np.array(power_list, dtype=np.float32)


def evaluate_powder(
    model_v1: torch.nn.Module,
    model_v2: torch.nn.Module,
    device: torch.device,
) -> dict:
    """Evaluate both models on POWDER DSSS dataset."""
    print("\n" + "=" * 60)
    print("POWDER DSSS DATASET EVALUATION")
    print("=" * 60)

    if not POWDER_PATH.exists():
        print("POWDER dataset not found, skipping...")
        return {}

    batch_dir = POWDER_PATH / "Batch1_10MHz" / "IQ"
    normal_files = sorted(glob.glob(str(batch_dir / "Only_LTE_frame_*")))[:30]
    anomaly_files = sorted(glob.glob(str(batch_dir / "Combined_LTE_DSSS_frame_*")))[:30]

    print(f"Loading {len(normal_files)} normal + {len(anomaly_files)} anomaly files...")

    # Load data
    normal_iq, normal_snr, normal_power = [], [], []
    for f in normal_files:
        iq, snr, power = load_powder_file(f)
        normal_iq.append(iq)
        normal_snr.append(snr)
        normal_power.append(power)

    anomaly_iq, anomaly_snr, anomaly_power = [], [], []
    for f in anomaly_files:
        iq, snr, power = load_powder_file(f)
        anomaly_iq.append(iq)
        anomaly_snr.append(snr)
        anomaly_power.append(power)

    normal_iq = np.concatenate(normal_iq)
    normal_snr = np.concatenate(normal_snr)
    normal_power = np.concatenate(normal_power)
    anomaly_iq = np.concatenate(anomaly_iq)
    anomaly_snr = np.concatenate(anomaly_snr)
    anomaly_power = np.concatenate(anomaly_power)

    # Normalize
    all_snr_norm = normalize_snr(np.concatenate([normal_snr, anomaly_snr]), snr_range=(-5, 30))
    all_power_norm = normalize_power(np.concatenate([normal_power, anomaly_power]))

    # Create tensors
    n_normal = len(normal_iq)
    train_split = int(n_normal * 0.3)

    train_iq = torch.from_numpy(normal_iq[:train_split])
    train_snr = torch.from_numpy(all_snr_norm[:train_split])
    train_power = torch.from_numpy(all_power_norm[:train_split])

    test_iq = torch.from_numpy(np.concatenate([normal_iq[train_split:], anomaly_iq]))
    test_snr = torch.from_numpy(all_snr_norm[train_split:])
    test_power = torch.from_numpy(all_power_norm[train_split:])
    test_labels = np.concatenate([
        np.zeros(n_normal - train_split),
        np.ones(len(anomaly_iq)),
    ])

    print(f"Train: {len(train_iq)}, Test: {len(test_iq)} (normal={n_normal - train_split}, anomaly={len(anomaly_iq)})")

    results = {"v1": {}, "v2": {}}

    for name, model in [("v1", model_v1), ("v2", model_v2)]:
        print(f"\n--- Model {name} ---")

        # Fit detector
        model.eval()
        with torch.no_grad():
            train_latents = []
            for i in range(0, len(train_iq), 64):
                batch_iq = train_iq[i:i+64].to(device)
                batch_snr = train_snr[i:i+64].to(device)
                batch_power = train_power[i:i+64].to(device)
                mu, _ = model.encode(batch_iq, batch_snr, batch_power)
                train_latents.append(mu.cpu().numpy())

            train_latents = np.concatenate(train_latents)
            latent_mean = torch.from_numpy(train_latents.mean(axis=0)).to(device)
            cov = np.cov(train_latents.T) + np.eye(train_latents.shape[1]) * 1e-6
            latent_cov_inv = torch.from_numpy(np.linalg.inv(cov)).float().to(device)

            # Test
            test_scores = []
            for i in range(0, len(test_iq), 64):
                batch_iq = test_iq[i:i+64].to(device)
                batch_snr = test_snr[i:i+64].to(device)
                batch_power = test_power[i:i+64].to(device)
                mu, _ = model.encode(batch_iq, batch_snr, batch_power)
                diff = mu - latent_mean
                mahal = torch.sqrt(torch.sum(diff @ latent_cov_inv * diff, dim=1))
                test_scores.extend(mahal.cpu().numpy())

        test_scores = np.array(test_scores)
        latent_auroc = roc_auc_score(test_labels, test_scores)

        # Hybrid with chirp
        chirp_det = ChirpDetector()
        chirp_det.fit(train_iq.numpy())
        chirp_scores = chirp_det.score(test_iq.numpy())

        def normalize(s):
            return (s - s.min()) / (s.max() - s.min() + 1e-8)

        hybrid_scores = 0.4 * normalize(test_scores) + 0.6 * normalize(chirp_scores)
        hybrid_auroc = roc_auc_score(test_labels, hybrid_scores)

        results[name] = {
            "latent_auroc": latent_auroc,
            "hybrid_auroc": hybrid_auroc,
            "latent_scores": test_scores,
            "hybrid_scores": hybrid_scores,
            "labels": test_labels,
        }

        print(f"  Latent AUROC: {latent_auroc:.4f}")
        print(f"  Hybrid AUROC: {hybrid_auroc:.4f}")

    return results


# ============ HackRF Dataset Evaluation ============

def evaluate_hackrf(
    model_v1: torch.nn.Module,
    model_v2: torch.nn.Module,
    device: torch.device,
) -> dict:
    """Evaluate both models on HackRF live data."""
    print("\n" + "=" * 60)
    print("HACKRF LIVE DATA EVALUATION")
    print("=" * 60)

    if not HACKRF_PATH.exists():
        print("HackRF dataset not found, skipping...")
        return {}

    with h5py.File(HACKRF_PATH, "r") as f:
        # HackRF stores complex64 signals that need conversion to I/Q [N, 2, seq_len]
        if "signals" in f:
            complex_data = f["signals"][:]  # [N, seq_len] complex64
            # Convert complex to I/Q: stack real and imaginary
            iq_data = np.stack([complex_data.real, complex_data.imag], axis=1).astype(np.float32)
        elif "iq" in f:
            iq_data = f["iq"][:]
        else:
            print(f"Unknown HackRF format. Keys: {list(f.keys())}")
            return {}
        labels = f["labels"][:]
        if "snr" in f:
            snr_data = f["snr"][:].astype(np.float32)
        else:
            snr_data = np.ones(len(iq_data), dtype=np.float32) * 0.5

    print(f"Loaded {len(iq_data)} samples ({(labels == 0).sum()} normal, {(labels == 1).sum()} anomaly)")

    # Split
    normal_idx = np.where(labels == 0)[0]
    anomaly_idx = np.where(labels == 1)[0]

    train_idx = normal_idx[:int(len(normal_idx) * 0.3)]
    test_normal_idx = normal_idx[int(len(normal_idx) * 0.3):]
    test_idx = np.concatenate([test_normal_idx, anomaly_idx])

    train_iq = torch.from_numpy(iq_data[train_idx]).float()
    train_snr = torch.from_numpy(snr_data[train_idx]).float()
    train_power = torch.ones(len(train_idx)) * 0.5

    test_iq = torch.from_numpy(iq_data[test_idx]).float()
    test_snr = torch.from_numpy(snr_data[test_idx]).float()
    test_power = torch.ones(len(test_idx)) * 0.5
    test_labels = labels[test_idx]

    results = {"v1": {}, "v2": {}}

    for name, model in [("v1", model_v1), ("v2", model_v2)]:
        print(f"\n--- Model {name} ---")

        model.eval()
        with torch.no_grad():
            # Fit on training data
            train_latents = []
            for i in range(0, len(train_iq), 64):
                batch_iq = train_iq[i:i+64].to(device)
                batch_snr = train_snr[i:i+64].to(device)
                batch_power = train_power[i:i+64].to(device)
                mu, _ = model.encode(batch_iq, batch_snr, batch_power)
                train_latents.append(mu.cpu().numpy())

            train_latents = np.concatenate(train_latents)
            latent_mean = torch.from_numpy(train_latents.mean(axis=0)).to(device)
            cov = np.cov(train_latents.T) + np.eye(train_latents.shape[1]) * 1e-6
            latent_cov_inv = torch.from_numpy(np.linalg.inv(cov)).float().to(device)

            # Test
            test_scores = []
            for i in range(0, len(test_iq), 64):
                batch_iq = test_iq[i:i+64].to(device)
                batch_snr = test_snr[i:i+64].to(device)
                batch_power = test_power[i:i+64].to(device)
                mu, _ = model.encode(batch_iq, batch_snr, batch_power)
                diff = mu - latent_mean
                mahal = torch.sqrt(torch.sum(diff @ latent_cov_inv * diff, dim=1))
                test_scores.extend(mahal.cpu().numpy())

        test_scores = np.array(test_scores)
        latent_auroc = roc_auc_score(test_labels, test_scores)

        results[name] = {
            "latent_auroc": latent_auroc,
            "latent_scores": test_scores,
            "labels": test_labels,
        }

        print(f"  Latent AUROC: {latent_auroc:.4f}")

    return results


# ============ Visualization ============

def create_comparison_plots(
    synthetic_results: dict,
    anomaly_results: dict,
    powder_results: dict,
    hackrf_results: dict,
    output_dir: Path,
) -> None:
    """Create comparison visualizations."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Overall comparison bar chart
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Latent comparison
    ax = axes[0]
    datasets = ["Synthetic", "POWDER", "HackRF"]
    v1_latent = [
        synthetic_results.get("v1", {}).get("latent_auroc", 0),
        powder_results.get("v1", {}).get("latent_auroc", 0),
        hackrf_results.get("v1", {}).get("latent_auroc", 0),
    ]
    v2_latent = [
        synthetic_results.get("v2", {}).get("latent_auroc", 0),
        powder_results.get("v2", {}).get("latent_auroc", 0),
        hackrf_results.get("v2", {}).get("latent_auroc", 0),
    ]

    x = np.arange(len(datasets))
    width = 0.35
    ax.bar(x - width/2, v1_latent, width, label="V1 (Jan 18)", color="steelblue", alpha=0.8)
    ax.bar(x + width/2, v2_latent, width, label="V2 (Jan 25)", color="darkorange", alpha=0.8)
    ax.set_ylabel("AUROC")
    ax.set_title("Latent-Only Detection")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.legend()
    ax.set_ylim(0.5, 1.0)
    ax.axhline(0.92, color="green", linestyle="--", alpha=0.5, label="Target")
    ax.grid(axis="y", alpha=0.3)

    # Hybrid comparison
    ax = axes[1]
    v1_hybrid = [
        synthetic_results.get("v1", {}).get("hybrid_auroc", 0),
        powder_results.get("v1", {}).get("hybrid_auroc", 0),
        0,  # No hybrid for HackRF
    ]
    v2_hybrid = [
        synthetic_results.get("v2", {}).get("hybrid_auroc", 0),
        powder_results.get("v2", {}).get("hybrid_auroc", 0),
        0,
    ]

    ax.bar(x[:2] - width/2, v1_hybrid[:2], width, label="V1 (Jan 18)", color="steelblue", alpha=0.8)
    ax.bar(x[:2] + width/2, v2_hybrid[:2], width, label="V2 (Jan 25)", color="darkorange", alpha=0.8)
    ax.set_ylabel("AUROC")
    ax.set_title("Hybrid Detection (Latent + ChirpDetector)")
    ax.set_xticks(x[:2])
    ax.set_xticklabels(datasets[:2])
    ax.legend()
    ax.set_ylim(0.5, 1.0)
    ax.axhline(0.95, color="green", linestyle="--", alpha=0.5, label="Target")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "overall_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()

    # 2. Per-anomaly type comparison
    if anomaly_results:
        fig, ax = plt.subplots(figsize=(12, 6))

        anomaly_types = list(anomaly_results.keys())
        x = np.arange(len(anomaly_types))

        v1_latent = [anomaly_results[a]["v1"]["latent_auroc"] for a in anomaly_types]
        v1_hybrid = [anomaly_results[a]["v1"]["hybrid_auroc"] for a in anomaly_types]
        v2_latent = [anomaly_results[a]["v2"]["latent_auroc"] for a in anomaly_types]
        v2_hybrid = [anomaly_results[a]["v2"]["hybrid_auroc"] for a in anomaly_types]

        width = 0.2
        ax.bar(x - 1.5*width, v1_latent, width, label="V1 Latent", color="steelblue", alpha=0.6)
        ax.bar(x - 0.5*width, v1_hybrid, width, label="V1 Hybrid", color="steelblue", alpha=1.0)
        ax.bar(x + 0.5*width, v2_latent, width, label="V2 Latent", color="darkorange", alpha=0.6)
        ax.bar(x + 1.5*width, v2_hybrid, width, label="V2 Hybrid", color="darkorange", alpha=1.0)

        ax.set_ylabel("AUROC")
        ax.set_title("Per-Anomaly Type Detection Performance")
        ax.set_xticks(x)
        ax.set_xticklabels([a.replace("_", "\n") for a in anomaly_types])
        ax.legend(loc="lower right")
        ax.set_ylim(0.5, 1.05)
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "per_anomaly_comparison.png", dpi=150, bbox_inches="tight")
        plt.close()

    # 3. ROC curves for synthetic data
    if synthetic_results:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        for idx, (name, label) in enumerate([("v1", "V1 (Jan 18)"), ("v2", "V2 (Jan 25)")]):
            ax = axes[idx]
            res = synthetic_results[name]

            # Latent ROC
            fpr, tpr, _ = roc_curve(res["labels"], res["latent_scores"])
            ax.plot(fpr, tpr, "b-", linewidth=2, label=f"Latent (AUROC={res['latent_auroc']:.4f})")

            # Hybrid ROC
            fpr, tpr, _ = roc_curve(res["labels"], res["hybrid_scores"])
            ax.plot(fpr, tpr, "r-", linewidth=2, label=f"Hybrid (AUROC={res['hybrid_auroc']:.4f})")

            ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
            ax.set_xlabel("False Positive Rate")
            ax.set_ylabel("True Positive Rate")
            ax.set_title(f"Model {label}")
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_dir / "roc_curves_synthetic.png", dpi=150, bbox_inches="tight")
        plt.close()

    # 4. Score distributions for synthetic data
    if synthetic_results:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        for row, (name, label) in enumerate([("v1", "V1"), ("v2", "V2")]):
            res = synthetic_results[name]
            normal_mask = res["labels"] == 0
            anomaly_mask = res["labels"] == 1

            # Latent scores
            ax = axes[row, 0]
            ax.hist(res["latent_scores"][normal_mask], bins=50, alpha=0.6, label="Normal", density=True)
            ax.hist(res["latent_scores"][anomaly_mask], bins=50, alpha=0.6, label="Anomaly", density=True)
            ax.set_xlabel("Latent Score (Mahalanobis)")
            ax.set_ylabel("Density")
            ax.set_title(f"{label} - Latent Score Distribution")
            ax.legend()

            # Hybrid scores
            ax = axes[row, 1]
            ax.hist(res["hybrid_scores"][normal_mask], bins=50, alpha=0.6, label="Normal", density=True)
            ax.hist(res["hybrid_scores"][anomaly_mask], bins=50, alpha=0.6, label="Anomaly", density=True)
            ax.set_xlabel("Hybrid Score")
            ax.set_ylabel("Density")
            ax.set_title(f"{label} - Hybrid Score Distribution")
            ax.legend()

        plt.tight_layout()
        plt.savefig(output_dir / "score_distributions.png", dpi=150, bbox_inches="tight")
        plt.close()

    print(f"\nPlots saved to {output_dir}/")


def main():
    device = get_device()
    print(f"Using device: {device}")

    # Load config
    config = load_config(str(CONFIG_PATH))

    # Load models
    print("\n" + "=" * 60)
    print("LOADING MODELS")
    print("=" * 60)

    print(f"Loading V1 from: {CHECKPOINT_V1}")
    model_v1 = load_model(CHECKPOINT_V1, config, device)

    print(f"Loading V2 from: {CHECKPOINT_V2}")
    model_v2 = load_model(CHECKPOINT_V2, config, device)

    # Run evaluations
    synthetic_results = evaluate_synthetic(model_v1, model_v2, config, device)
    anomaly_results = evaluate_per_anomaly_type(model_v1, model_v2, config, device)
    powder_results = evaluate_powder(model_v1, model_v2, device)
    hackrf_results = evaluate_hackrf(model_v1, model_v2, device)

    # Create visualizations
    print("\n" + "=" * 60)
    print("CREATING VISUALIZATIONS")
    print("=" * 60)
    create_comparison_plots(synthetic_results, anomaly_results, powder_results, hackrf_results, OUTPUT_DIR)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print("\n| Dataset | Model | Latent AUROC | Hybrid AUROC |")
    print("|---------|-------|--------------|--------------|")
    for dataset, results in [("Synthetic", synthetic_results), ("POWDER", powder_results), ("HackRF", hackrf_results)]:
        if results:
            for ver in ["v1", "v2"]:
                if ver in results:
                    lat = results[ver].get("latent_auroc", "N/A")
                    hyb = results[ver].get("hybrid_auroc", "N/A")
                    lat_str = f"{lat:.4f}" if isinstance(lat, float) else lat
                    hyb_str = f"{hyb:.4f}" if isinstance(hyb, float) else hyb
                    print(f"| {dataset:<7} | {ver.upper():<5} | {lat_str:<12} | {hyb_str:<12} |")

    # Save results
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "synthetic": {k: {kk: vv for kk, vv in v.items() if kk not in ["latent_scores", "hybrid_scores", "labels"]} for k, v in synthetic_results.items()} if synthetic_results else {},
        "powder": {k: {kk: vv for kk, vv in v.items() if kk not in ["latent_scores", "hybrid_scores", "labels"]} for k, v in powder_results.items()} if powder_results else {},
        "hackrf": {k: {kk: vv for kk, vv in v.items() if kk not in ["latent_scores", "labels"]} for k, v in hackrf_results.items()} if hackrf_results else {},
        "per_anomaly": anomaly_results if anomaly_results else {},
    }

    results_path = OUTPUT_DIR / "comparison_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()
