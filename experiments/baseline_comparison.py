#!/usr/bin/env python3
"""Compare our latent-only VAE detection with standard baseline methods.

Baselines:
1. One-Class SVM on latent space
2. Isolation Forest on latent space
3. PCA-based anomaly detection (reconstruction error)
4. One-Class SVM on raw features
5. Isolation Forest on raw features

This validates that our approach is genuinely better, not just that the task is easy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.data.synthetic import SyntheticRFGenerator
from src.data.datasets import RFDataset
from src.models.snr_encoder import create_model
from src.detection.detector import AnomalyDetector
from src.detection.metrics import compute_metrics


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(checkpoint_path: str, config, device: torch.device):
    """Load model with lazy layer initialization."""
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
    return model


def extract_latent_features(model, loader, device):
    """Extract latent space representations from VAE."""
    model.eval()
    all_latents = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            iq = batch["iq"].to(device)
            snr = batch.get("snr")
            if snr is not None:
                snr = snr.to(device)
            power = batch.get("power")
            if power is not None:
                power = power.to(device)

            # Get latent representation (mu)
            mu, _ = model.encoder(iq, snr, power)
            all_latents.append(mu.cpu().numpy())
            all_labels.append(batch["label"].numpy())

    return np.concatenate(all_latents), np.concatenate(all_labels)


def extract_raw_features(loader):
    """Extract raw I/Q features (flattened)."""
    all_features = []
    all_labels = []

    for batch in loader:
        iq = batch["iq"].numpy()
        # Flatten to [batch, 2*seq_len]
        features = iq.reshape(iq.shape[0], -1)
        all_features.append(features)
        all_labels.append(batch["label"].numpy())

    return np.concatenate(all_features), np.concatenate(all_labels)


def evaluate_baseline(name: str, scores: np.ndarray, labels: np.ndarray) -> dict:
    """Evaluate a baseline method and return metrics."""
    metrics = compute_metrics(scores, labels)
    return {
        "name": name,
        "auroc": metrics.auroc,
        "auprc": metrics.auprc,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
    }


def run_ocsvm_latent(train_latents, test_latents, test_labels):
    """One-Class SVM on latent space."""
    # Train only on normal samples
    normal_mask = np.zeros(len(train_latents), dtype=bool)
    normal_mask[:] = True  # All training data is normal

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_latents)
    test_scaled = scaler.transform(test_latents)

    # One-Class SVM (nu is the expected outlier fraction)
    ocsvm = OneClassSVM(kernel="rbf", nu=0.1, gamma="auto")
    ocsvm.fit(train_scaled)

    # Decision function: negative = anomaly, positive = normal
    # We want higher scores for anomalies, so negate
    scores = -ocsvm.decision_function(test_scaled)

    return evaluate_baseline("One-Class SVM (latent)", scores, test_labels)


def run_isolation_forest_latent(train_latents, test_latents, test_labels):
    """Isolation Forest on latent space."""
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_latents)
    test_scaled = scaler.transform(test_latents)

    iforest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    iforest.fit(train_scaled)

    # Score: negative = anomaly, positive = normal
    scores = -iforest.decision_function(test_scaled)

    return evaluate_baseline("Isolation Forest (latent)", scores, test_labels)


def run_pca_reconstruction(train_features, test_features, test_labels, n_components=32):
    """PCA-based anomaly detection using reconstruction error."""
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_features)
    test_scaled = scaler.transform(test_features)

    pca = PCA(n_components=n_components)
    pca.fit(train_scaled)

    # Reconstruct and compute error
    test_reconstructed = pca.inverse_transform(pca.transform(test_scaled))
    reconstruction_error = np.mean((test_scaled - test_reconstructed) ** 2, axis=1)

    return evaluate_baseline(f"PCA Reconstruction (n={n_components})", reconstruction_error, test_labels)


def run_ocsvm_raw(train_features, test_features, test_labels):
    """One-Class SVM on raw features (downsampled for efficiency)."""
    # Downsample features for computational efficiency
    n_features = min(256, train_features.shape[1])
    indices = np.linspace(0, train_features.shape[1]-1, n_features, dtype=int)
    train_sub = train_features[:, indices]
    test_sub = test_features[:, indices]

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_sub)
    test_scaled = scaler.transform(test_sub)

    ocsvm = OneClassSVM(kernel="rbf", nu=0.1, gamma="auto")
    ocsvm.fit(train_scaled)

    scores = -ocsvm.decision_function(test_scaled)

    return evaluate_baseline("One-Class SVM (raw)", scores, test_labels)


def run_isolation_forest_raw(train_features, test_features, test_labels):
    """Isolation Forest on raw features."""
    # Downsample features for efficiency
    n_features = min(256, train_features.shape[1])
    indices = np.linspace(0, train_features.shape[1]-1, n_features, dtype=int)
    train_sub = train_features[:, indices]
    test_sub = test_features[:, indices]

    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train_sub)
    test_scaled = scaler.transform(test_sub)

    iforest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    iforest.fit(train_scaled)

    scores = -iforest.decision_function(test_scaled)

    return evaluate_baseline("Isolation Forest (raw)", scores, test_labels)


def run_our_method(model, train_loader, test_loader, device):
    """Our VAE latent-only detection method."""
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

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)

    return evaluate_baseline("VAE Latent-Only (Ours)", scores, labels)


def statistical_significance_test(scores1, scores2, labels, name1, name2):
    """Perform paired statistical test between two methods."""
    from sklearn.metrics import roc_auc_score

    # Bootstrap test for AUROC difference
    n_bootstrap = 1000
    auroc_diffs = []

    n_samples = len(labels)
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        auroc1 = roc_auc_score(labels[indices], scores1[indices])
        auroc2 = roc_auc_score(labels[indices], scores2[indices])
        auroc_diffs.append(auroc1 - auroc2)

    auroc_diffs = np.array(auroc_diffs)
    mean_diff = np.mean(auroc_diffs)
    ci_lower = np.percentile(auroc_diffs, 2.5)
    ci_upper = np.percentile(auroc_diffs, 97.5)

    # p-value: proportion of times the difference is <= 0
    p_value = np.mean(auroc_diffs <= 0)

    return {
        "comparison": f"{name1} vs {name2}",
        "mean_diff": mean_diff,
        "ci_95": (ci_lower, ci_upper),
        "p_value": p_value,
        "significant": p_value < 0.05,
    }


def main():
    config = load_config("configs/default.yaml")
    device = get_device()
    print(f"Using device: {device}")

    checkpoint_path = "checkpoints/20260118_184144/best_model.pt"
    model = load_model(checkpoint_path, config, device)
    model.eval()

    print("\n" + "="*70)
    print("BASELINE COMPARISON EXPERIMENT")
    print("="*70)

    # Generate data
    generator = SyntheticRFGenerator(
        sequence_length=config.data.sequence_length,
        sample_rate=config.data.sample_rate,
        seed=42,
    )

    # Training data (normal only)
    train_dataset = RFDataset.from_generator(
        generator=generator,
        num_samples=3000,
        anomaly_ratio=0.0,
        snr_range=tuple(config.data.snr_range),
    )
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False)

    # Test data (with anomalies)
    test_dataset = RFDataset.from_generator(
        generator=generator,
        num_samples=3000,
        anomaly_ratio=0.1,
        snr_range=tuple(config.data.snr_range),
        anomaly_severity=config.data.anomaly_severity,
    )
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    print("\nExtracting features...")

    # Extract latent features using VAE
    train_latents, _ = extract_latent_features(model, train_loader, device)
    test_latents, test_labels = extract_latent_features(model, test_loader, device)

    # Extract raw features
    train_raw, _ = extract_raw_features(train_loader)
    test_raw, _ = extract_raw_features(test_loader)

    print(f"  Latent features shape: {train_latents.shape}")
    print(f"  Raw features shape: {train_raw.shape}")

    # Run all methods
    print("\n" + "-"*70)
    print("Running baseline methods...")
    print("-"*70)

    results = []
    all_scores = {}

    # Our method
    print("\n  Running: VAE Latent-Only (Ours)...")
    our_result = run_our_method(model, train_loader, test_loader, device)
    results.append(our_result)

    # For statistical tests, we need to re-extract our scores
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
    our_scores = []
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
            our_scores.append(result.scores)
    all_scores["ours"] = np.concatenate(our_scores)

    # Latent space baselines
    print("  Running: One-Class SVM (latent)...")
    ocsvm_latent = run_ocsvm_latent(train_latents, test_latents, test_labels)
    results.append(ocsvm_latent)

    print("  Running: Isolation Forest (latent)...")
    iforest_latent = run_isolation_forest_latent(train_latents, test_latents, test_labels)
    results.append(iforest_latent)

    # Raw feature baselines
    print("  Running: PCA Reconstruction...")
    pca_result = run_pca_reconstruction(train_raw, test_raw, test_labels, n_components=32)
    results.append(pca_result)

    print("  Running: One-Class SVM (raw)...")
    ocsvm_raw = run_ocsvm_raw(train_raw, test_raw, test_labels)
    results.append(ocsvm_raw)

    print("  Running: Isolation Forest (raw)...")
    iforest_raw = run_isolation_forest_raw(train_raw, test_raw, test_labels)
    results.append(iforest_raw)

    # Print results table
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"\n{'Method':<35} {'AUROC':>8} {'AUPRC':>8} {'F1':>8}")
    print("-"*70)

    # Sort by AUROC
    results_sorted = sorted(results, key=lambda x: x["auroc"], reverse=True)

    for r in results_sorted:
        marker = " <-- BEST" if r["name"] == results_sorted[0]["name"] else ""
        if "Ours" in r["name"]:
            marker = " <-- OURS" + (" (BEST)" if r["name"] == results_sorted[0]["name"] else "")
        print(f"{r['name']:<35} {r['auroc']:>8.4f} {r['auprc']:>8.4f} {r['f1']:>8.4f}{marker}")

    # Statistical significance tests
    print("\n" + "="*70)
    print("STATISTICAL SIGNIFICANCE (Bootstrap, 1000 iterations)")
    print("="*70)

    # Compare our method vs each baseline
    # Re-compute baseline scores for statistical tests
    scaler = StandardScaler()
    train_latents_scaled = scaler.fit_transform(train_latents)
    test_latents_scaled = scaler.transform(test_latents)

    ocsvm = OneClassSVM(kernel="rbf", nu=0.1, gamma="auto")
    ocsvm.fit(train_latents_scaled)
    ocsvm_scores = -ocsvm.decision_function(test_latents_scaled)

    iforest = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
    iforest.fit(train_latents_scaled)
    iforest_scores = -iforest.decision_function(test_latents_scaled)

    comparisons = [
        ("Ours", all_scores["ours"], "OCSVM-Latent", ocsvm_scores),
        ("Ours", all_scores["ours"], "IForest-Latent", iforest_scores),
    ]

    print(f"\n{'Comparison':<40} {'Δ AUROC':>10} {'95% CI':>20} {'p-value':>10} {'Sig?':>6}")
    print("-"*90)

    for name1, scores1, name2, scores2 in comparisons:
        stat_result = statistical_significance_test(scores1, scores2, test_labels, name1, name2)
        ci_str = f"[{stat_result['ci_95'][0]:.4f}, {stat_result['ci_95'][1]:.4f}]"
        sig_str = "Yes" if stat_result["significant"] else "No"
        print(f"{stat_result['comparison']:<40} {stat_result['mean_diff']:>+10.4f} {ci_str:>20} {stat_result['p_value']:>10.4f} {sig_str:>6}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    our_auroc = our_result["auroc"]
    best_baseline_auroc = max(r["auroc"] for r in results if "Ours" not in r["name"])
    best_baseline_name = [r["name"] for r in results if r["auroc"] == best_baseline_auroc and "Ours" not in r["name"]][0]

    improvement = (our_auroc - best_baseline_auroc) / best_baseline_auroc * 100

    print(f"\n  Our Method AUROC:      {our_auroc:.4f}")
    print(f"  Best Baseline AUROC:   {best_baseline_auroc:.4f} ({best_baseline_name})")
    print(f"  Relative Improvement:  {improvement:+.1f}%")

    if our_auroc > best_baseline_auroc:
        print("\n  ✓ Our method outperforms all baselines")
    else:
        print(f"\n  ✗ Baseline '{best_baseline_name}' performs better")

    return results


if __name__ == "__main__":
    main()
