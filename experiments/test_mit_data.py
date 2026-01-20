#!/usr/bin/env python3
"""Test MIT 5G dataset with our trained model."""

import sys
sys.path.insert(0, '/home/babynicky/Work/CLP_Project')

import h5py
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, classification_report

from src.models.snr_encoder import create_model
from src.detection.detector import AnomalyDetector
from src.data.snr_estimation import normalize_snr
from src.utils.config import load_config


def normalize_power(power_db: np.ndarray, power_range=(-20, 10)) -> np.ndarray:
    """Normalize power to [0, 1] range."""
    return np.clip((power_db - power_range[0]) / (power_range[1] - power_range[0]), 0, 1).astype(np.float32)


def load_mit_data(clean_path: str, jammed_path: str, split: str = 'test', max_samples: int = None):
    """Load MIT 5G dataset and convert to our format.

    Args:
        clean_path: Path to clean_5g_dataset.h5
        jammed_path: Path to jammed_5g_dataset.h5
        split: 'train', 'val', or 'test'
        max_samples: Maximum samples per class (None for all)

    Returns:
        iq: [N, 2, 1024] tensor
        snr: [N] tensor (normalized 0-1)
        snr_db: [N] tensor (raw dB)
        power: [N] tensor (normalized 0-1)
        labels: [N] tensor (0=normal, 1=anomaly)
        jamming_types: list of jamming type strings
    """
    # Load clean (normal) signals
    with h5py.File(clean_path, 'r') as f:
        clean_signals = f[split]['signals'][:]
        clean_snr = f[split]['snr'][:]
        n_clean = len(clean_signals) if max_samples is None else min(max_samples, len(clean_signals))
        clean_signals = clean_signals[:n_clean]
        clean_snr = clean_snr[:n_clean]

    # Load jammed (anomalous) signals - only those marked as jammed
    with h5py.File(jammed_path, 'r') as f:
        jammed_mask = f[split]['jammed'][:]
        all_signals = f[split]['signals'][:]
        all_snr = f[split]['snr'][:]
        all_jam_types = f[split]['jamming_type'][:]

        # Filter to only jammed samples
        jammed_signals = all_signals[jammed_mask]
        jammed_snr = all_snr[jammed_mask]
        jammed_types = all_jam_types[jammed_mask]

        n_jammed = len(jammed_signals) if max_samples is None else min(max_samples, len(jammed_signals))
        jammed_signals = jammed_signals[:n_jammed]
        jammed_snr = jammed_snr[:n_jammed]
        jammed_types = jammed_types[:n_jammed]

    # Convert complex64 to [2, 1024] format (I, Q channels)
    def complex_to_iq(signals):
        """Convert complex signals to I/Q format with normalization.

        Returns:
            iq: [N, 2, 1024] normalized I/Q signals
            power_db: [N] power in dB before normalization
        """
        iq_list = []
        power_db_list = []
        for sig in signals:
            # Compute power BEFORE normalization
            power = np.mean(np.abs(sig) ** 2)
            power_db = 10 * np.log10(power + 1e-10)
            power_db_list.append(power_db)

            # Normalize amplitude to max 1.0
            max_amp = np.max(np.abs(sig)) + 1e-8
            sig_norm = sig / max_amp
            # Stack I and Q
            iq = np.stack([sig_norm.real, sig_norm.imag], axis=0).astype(np.float32)
            iq_list.append(iq)
        return np.array(iq_list), np.array(power_db_list, dtype=np.float32)

    clean_iq, clean_power_db = complex_to_iq(clean_signals)
    jammed_iq, jammed_power_db = complex_to_iq(jammed_signals)

    # Combine datasets
    iq = np.concatenate([clean_iq, jammed_iq], axis=0)
    snr_db = np.concatenate([clean_snr, jammed_snr], axis=0)
    power_db = np.concatenate([clean_power_db, jammed_power_db], axis=0)
    labels = np.concatenate([
        np.zeros(len(clean_iq), dtype=np.int64),
        np.ones(len(jammed_iq), dtype=np.int64)
    ])

    # Decode jamming types
    jamming_types = ['none'] * len(clean_iq) + [jt.decode() for jt in jammed_types]

    # Normalize SNR and power to [0, 1]
    snr_normalized = normalize_snr(snr_db, snr_range=(-5, 30))
    power_normalized = normalize_power(power_db, power_range=(-20, 10))

    # Shuffle
    indices = np.random.permutation(len(iq))
    iq = iq[indices]
    snr_db = snr_db[indices]
    snr_normalized = snr_normalized[indices]
    power_normalized = power_normalized[indices]
    labels = labels[indices]
    jamming_types = [jamming_types[i] for i in indices]

    return (
        torch.from_numpy(iq),
        torch.from_numpy(snr_normalized),
        torch.from_numpy(snr_db),
        torch.from_numpy(power_normalized),
        torch.from_numpy(labels),
        jamming_types
    )


def main():
    # Paths
    clean_path = '/home/babynicky/Work/CLP_Project/MIT_DATA/clean_5g_dataset.h5'
    jammed_path = '/home/babynicky/Work/CLP_Project/MIT_DATA/jammed_5g_dataset.h5'
    checkpoint_path = '/home/babynicky/Work/CLP_Project/snr_conditioned_vae_hybrid_v1.pt'
    config_path = '/home/babynicky/Work/CLP_Project/configs/default.yaml'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load config and create model
    print("\nLoading model...")
    config = load_config(config_path)
    model = create_model(config)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    # Load with strict=False to handle minor architecture differences
    missing, unexpected = model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    if missing:
        print(f"Warning: Missing keys: {missing}")
    if unexpected:
        print(f"Warning: Unexpected keys: {unexpected}")
    model = model.to(device).eval()
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    # Initialize lazy layers with dummy forward pass (need both SNR and power)
    dummy_iq = torch.randn(1, 2, 1024, device=device)
    dummy_snr = torch.tensor([0.5], device=device)
    dummy_power = torch.tensor([0.5], device=device)
    with torch.no_grad():
        _ = model(dummy_iq, dummy_snr, dummy_power)

    # Load MIT test data (now includes power)
    print("\nLoading MIT 5G test data...")
    iq, snr, snr_db, power, labels, jam_types = load_mit_data(
        clean_path, jammed_path, split='test', max_samples=5000
    )
    print(f"Loaded {len(iq)} samples: {(labels == 0).sum()} normal, {(labels == 1).sum()} anomalous")

    # Load training data to fit detector (normal signals only)
    print("\nLoading training data to fit detector...")
    train_iq, train_snr, train_snr_db, train_power, train_labels, _ = load_mit_data(
        clean_path, jammed_path, split='train', max_samples=5000
    )
    # Use only normal samples for fitting
    normal_mask = train_labels == 0
    train_iq_normal = train_iq[normal_mask]
    train_snr_normal = train_snr[normal_mask]
    train_power_normal = train_power[normal_mask]

    # Create detector
    print("\nFitting anomaly detector...")
    detector = AnomalyDetector(
        model=model,
        method='latent',  # Use latent-only (best method per CLAUDE.md)
        threshold_method='percentile',
        threshold_percentile=95.0,
        snr_adaptive=True,
        snr_bins=7,
        device=device
    )

    # Fit on normal training data
    from torch.utils.data import TensorDataset, DataLoader
    train_dataset = TensorDataset(train_iq_normal, train_snr_normal, train_power_normal)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

    # Custom fit that handles our data format
    detector.model.eval()
    all_latents = []

    with torch.no_grad():
        for batch_iq, batch_snr, batch_power in train_loader:
            batch_iq = batch_iq.to(device)
            batch_snr = batch_snr.to(device)
            batch_power = batch_power.to(device)

            # Get latent representation (with power conditioning)
            mu, logvar = model.encode(batch_iq, batch_snr, batch_power)
            all_latents.append(mu.cpu().numpy())

    # Compute latent statistics for Mahalanobis distance
    latents = np.concatenate(all_latents, axis=0)
    detector._latent_mean = torch.from_numpy(latents.mean(axis=0)).to(device)

    # Regularized covariance inverse
    cov = np.cov(latents.T)
    cov += np.eye(cov.shape[0]) * 1e-6  # Regularization
    detector._latent_cov_inv = torch.from_numpy(np.linalg.inv(cov)).float().to(device)

    # Compute threshold from training scores
    train_scores = []
    with torch.no_grad():
        for batch_iq, batch_snr, batch_power in train_loader:
            batch_iq = batch_iq.to(device)
            batch_snr = batch_snr.to(device)
            batch_power = batch_power.to(device)
            mu, _ = model.encode(batch_iq, batch_snr, batch_power)

            # Mahalanobis distance
            diff = mu - detector._latent_mean
            mahal = torch.sqrt(torch.sum(diff @ detector._latent_cov_inv * diff, dim=1))
            train_scores.extend(mahal.cpu().numpy())

    detector._threshold = np.percentile(train_scores, 95)
    print(f"Detection threshold (95th percentile): {detector._threshold:.4f}")

    # Test on MIT data
    print("\n" + "="*60)
    print("TESTING ON MIT 5G DATA")
    print("="*60)

    test_dataset = TensorDataset(iq, snr, power, labels)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    all_scores = []
    all_labels = []

    with torch.no_grad():
        for batch_iq, batch_snr, batch_power, batch_labels in test_loader:
            batch_iq = batch_iq.to(device)
            batch_snr = batch_snr.to(device)
            batch_power = batch_power.to(device)

            mu, _ = model.encode(batch_iq, batch_snr, batch_power)
            diff = mu - detector._latent_mean
            mahal = torch.sqrt(torch.sum(diff @ detector._latent_cov_inv * diff, dim=1))

            all_scores.extend(mahal.cpu().numpy())
            all_labels.extend(batch_labels.numpy())

    scores = np.array(all_scores)
    labels_np = np.array(all_labels)
    predictions = scores > detector._threshold

    # Compute metrics
    auroc = roc_auc_score(labels_np, scores)

    print(f"\nOverall AUROC: {auroc:.4f}")
    print(f"\nClassification Report (threshold={detector._threshold:.4f}):")
    print(classification_report(labels_np, predictions, target_names=['Normal', 'Anomaly']))

    # Per-jamming-type analysis
    print("\n" + "="*60)
    print("PER-JAMMING-TYPE AUROC")
    print("="*60)

    # Recompute with fixed seed for reproducibility
    np.random.seed(42)
    iq3, snr3, snr_db3, power3, labels3, jam_types3 = load_mit_data(
        clean_path, jammed_path, split='test', max_samples=5000
    )

    test_dataset3 = TensorDataset(iq3, snr3, power3, labels3)
    test_loader3 = DataLoader(test_dataset3, batch_size=64, shuffle=False)

    all_scores3 = []
    with torch.no_grad():
        for batch_iq, batch_snr, batch_power, _ in test_loader3:
            batch_iq = batch_iq.to(device)
            batch_snr = batch_snr.to(device)
            batch_power = batch_power.to(device)
            mu, _ = model.encode(batch_iq, batch_snr, batch_power)
            diff = mu - detector._latent_mean
            mahal = torch.sqrt(torch.sum(diff @ detector._latent_cov_inv * diff, dim=1))
            all_scores3.extend(mahal.cpu().numpy())

    scores3 = np.array(all_scores3)
    labels3_np = labels3.numpy()

    # Compute per-type AUROC
    unique_types = sorted(set(jam_types3))
    for jtype in unique_types:
        if jtype == 'none':
            continue
        mask = np.array([jt == jtype or jt == 'none' for jt in jam_types3])
        type_scores = scores3[mask]
        type_labels = labels3_np[mask]
        if len(np.unique(type_labels)) < 2:
            continue
        type_auroc = roc_auc_score(type_labels, type_scores)
        count = (np.array(jam_types3) == jtype).sum()
        print(f"  {jtype:12s}: AUROC = {type_auroc:.4f} (n={count})")


if __name__ == '__main__':
    main()
