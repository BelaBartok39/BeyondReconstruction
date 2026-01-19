#!/usr/bin/env python3
"""Visualize VAE latent space using t-SNE and UMAP.

Creates publication-ready figures showing:
1. Normal vs anomaly separation
2. Per-anomaly type clustering
3. Why latent-only detection works
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Try to import UMAP, fall back gracefully if not available
try:
    from umap import UMAP
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("Note: UMAP not installed. Using t-SNE only. Install with: pip install umap-learn")

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.config import load_config
from src.data.synthetic import SyntheticRFGenerator
from src.data.datasets import RFDataset
from src.models.snr_encoder import create_model


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(checkpoint_path: str, config, device: torch.device):
    """Load model with lazy layer initialization."""
    model = create_model(config)
    model = model.to(device)

    dummy_iq = torch.randn(1, 2, config.data.sequence_length, device=device)
    dummy_snr = torch.rand(1, device=device)
    dummy_power = torch.rand(1, device=device)
    with torch.no_grad():
        _ = model(dummy_iq, dummy_snr, dummy_power)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


def extract_latents_with_labels(model, loader, device):
    """Extract latent representations with labels and anomaly types."""
    model.eval()
    all_latents = []
    all_labels = []
    all_types = []

    with torch.no_grad():
        for batch in loader:
            iq = batch["iq"].to(device)
            snr = batch.get("snr")
            if snr is not None:
                snr = snr.to(device)
            power = batch.get("power")
            if power is not None:
                power = power.to(device)

            mu, _ = model.encoder(iq, snr, power)
            all_latents.append(mu.cpu().numpy())
            all_labels.append(batch["label"].numpy())

            # Get anomaly types if available
            if "anomaly_type" in batch:
                all_types.extend(batch["anomaly_type"])
            else:
                # Infer from label
                types = ["normal" if l == 0 else "anomaly" for l in batch["label"].numpy()]
                all_types.extend(types)

    return np.concatenate(all_latents), np.concatenate(all_labels), all_types


def create_dataset_with_types(generator, config, anomaly_type, num_samples=200):
    """Create dataset for specific anomaly type."""
    dataset = RFDataset.from_generator(
        generator=generator,
        num_samples=num_samples,
        anomaly_ratio=1.0,  # All anomalies
        snr_range=tuple(config.data.snr_range),
        anomaly_types=[anomaly_type],
        anomaly_severity=config.data.anomaly_severity,
    )
    return dataset


def plot_tsne_by_class(latents, labels, title, save_path=None):
    """Create t-SNE plot colored by normal/anomaly."""
    print(f"  Computing t-SNE for {title}...")

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    latents_2d = tsne.fit_transform(latents)

    plt.figure(figsize=(10, 8))

    # Plot normal points
    normal_mask = labels == 0
    plt.scatter(
        latents_2d[normal_mask, 0],
        latents_2d[normal_mask, 1],
        c='blue',
        label='Normal',
        alpha=0.6,
        s=30,
    )

    # Plot anomaly points
    anomaly_mask = labels == 1
    plt.scatter(
        latents_2d[anomaly_mask, 0],
        latents_2d[anomaly_mask, 1],
        c='red',
        label='Anomaly',
        alpha=0.6,
        s=30,
    )

    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=11)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved to {save_path}")

    plt.close()
    return latents_2d


def plot_tsne_by_anomaly_type(latents_dict, save_path=None):
    """Create t-SNE plot colored by anomaly type."""
    print("  Computing t-SNE for all anomaly types...")

    # Combine all latents
    all_latents = []
    all_types = []
    for atype, latents in latents_dict.items():
        all_latents.append(latents)
        all_types.extend([atype] * len(latents))

    all_latents = np.concatenate(all_latents)

    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    latents_2d = tsne.fit_transform(all_latents)

    plt.figure(figsize=(12, 10))

    # Color map for anomaly types
    colors = {
        'normal': 'blue',
        'interference': 'orange',
        'frequency_drift': 'green',
        'amplitude_spike': 'red',
        'phase_noise': 'purple',
        'burst_noise': 'brown',
    }

    # Plot each type
    start_idx = 0
    for atype, latents in latents_dict.items():
        end_idx = start_idx + len(latents)
        color = colors.get(atype, 'gray')

        plt.scatter(
            latents_2d[start_idx:end_idx, 0],
            latents_2d[start_idx:end_idx, 1],
            c=color,
            label=atype,
            alpha=0.6,
            s=40,
        )
        start_idx = end_idx

    plt.xlabel('t-SNE Dimension 1', fontsize=12)
    plt.ylabel('t-SNE Dimension 2', fontsize=12)
    plt.title('Latent Space Visualization by Anomaly Type', fontsize=14)
    plt.legend(fontsize=10, loc='best')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved to {save_path}")

    plt.close()
    return latents_2d


def plot_umap_by_anomaly_type(latents_dict, save_path=None):
    """Create UMAP plot colored by anomaly type."""
    if not HAS_UMAP:
        print("  Skipping UMAP (not installed)")
        return None

    print("  Computing UMAP for all anomaly types...")

    # Combine all latents
    all_latents = []
    all_types = []
    for atype, latents in latents_dict.items():
        all_latents.append(latents)
        all_types.extend([atype] * len(latents))

    all_latents = np.concatenate(all_latents)

    umap = UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
    latents_2d = umap.fit_transform(all_latents)

    plt.figure(figsize=(12, 10))

    colors = {
        'normal': 'blue',
        'interference': 'orange',
        'frequency_drift': 'green',
        'amplitude_spike': 'red',
        'phase_noise': 'purple',
        'burst_noise': 'brown',
    }

    start_idx = 0
    for atype, latents in latents_dict.items():
        end_idx = start_idx + len(latents)
        color = colors.get(atype, 'gray')

        plt.scatter(
            latents_2d[start_idx:end_idx, 0],
            latents_2d[start_idx:end_idx, 1],
            c=color,
            label=atype,
            alpha=0.6,
            s=40,
        )
        start_idx = end_idx

    plt.xlabel('UMAP Dimension 1', fontsize=12)
    plt.ylabel('UMAP Dimension 2', fontsize=12)
    plt.title('Latent Space Visualization by Anomaly Type (UMAP)', fontsize=14)
    plt.legend(fontsize=10, loc='best')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved to {save_path}")

    plt.close()
    return latents_2d


def plot_mahalanobis_distribution(latents_dict, save_path=None):
    """Plot Mahalanobis distance distributions by anomaly type."""
    print("  Computing Mahalanobis distance distributions...")

    # Use normal latents to compute mean and covariance
    normal_latents = latents_dict['normal']
    mean = np.mean(normal_latents, axis=0)
    cov = np.cov(normal_latents.T)
    cov += np.eye(cov.shape[0]) * 1e-6  # Regularize
    cov_inv = np.linalg.inv(cov)

    def mahalanobis_distance(latents):
        diff = latents - mean
        left = np.dot(diff, cov_inv)
        return np.sqrt(np.sum(left * diff, axis=1))

    plt.figure(figsize=(12, 6))

    colors = {
        'normal': 'blue',
        'interference': 'orange',
        'frequency_drift': 'green',
        'amplitude_spike': 'red',
        'phase_noise': 'purple',
        'burst_noise': 'brown',
    }

    for atype, latents in latents_dict.items():
        distances = mahalanobis_distance(latents)
        color = colors.get(atype, 'gray')

        plt.hist(
            distances,
            bins=50,
            alpha=0.5,
            label=f'{atype} (mean={np.mean(distances):.1f})',
            color=color,
            density=True,
        )

    plt.xlabel('Mahalanobis Distance', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Mahalanobis Distance Distribution by Anomaly Type', fontsize=14)
    plt.legend(fontsize=9)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved to {save_path}")

    plt.close()


def plot_latent_dimension_analysis(latents_dict, save_path=None):
    """Analyze which latent dimensions are most discriminative."""
    print("  Analyzing discriminative latent dimensions...")

    normal_latents = latents_dict['normal']
    n_dims = normal_latents.shape[1]

    # Compute separation score for each dimension
    separation_scores = {}

    for atype, latents in latents_dict.items():
        if atype == 'normal':
            continue

        scores = []
        for dim in range(n_dims):
            normal_vals = normal_latents[:, dim]
            anomaly_vals = latents[:, dim]

            # Cohen's d effect size
            pooled_std = np.sqrt((np.var(normal_vals) + np.var(anomaly_vals)) / 2)
            cohens_d = abs(np.mean(anomaly_vals) - np.mean(normal_vals)) / (pooled_std + 1e-8)
            scores.append(cohens_d)

        separation_scores[atype] = scores

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(14, 6))

    anomaly_types = [k for k in separation_scores.keys()]
    scores_matrix = np.array([separation_scores[k] for k in anomaly_types])

    im = ax.imshow(scores_matrix, aspect='auto', cmap='YlOrRd')

    ax.set_xticks(range(n_dims))
    ax.set_xticklabels([f'd{i}' for i in range(n_dims)], fontsize=8)
    ax.set_yticks(range(len(anomaly_types)))
    ax.set_yticklabels(anomaly_types, fontsize=10)

    ax.set_xlabel('Latent Dimension', fontsize=12)
    ax.set_ylabel('Anomaly Type', fontsize=12)
    ax.set_title("Cohen's d Effect Size per Latent Dimension", fontsize=14)

    plt.colorbar(im, ax=ax, label="Cohen's d")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved to {save_path}")

    plt.close()

    # Print most discriminative dimensions per anomaly type
    print("\n  Most discriminative dimensions:")
    for atype, scores in separation_scores.items():
        top_dims = np.argsort(scores)[-3:][::-1]
        top_scores = [f"{scores[d]:.2f}" for d in top_dims]
        print(f"    {atype}: dims {list(top_dims)} (d={top_scores})")


def main():
    config = load_config("configs/default.yaml")
    device = get_device()
    print(f"Using device: {device}")

    checkpoint_path = "checkpoints/20260118_184144/best_model.pt"
    model = load_model(checkpoint_path, config, device)
    model.eval()

    generator = SyntheticRFGenerator(
        sequence_length=config.data.sequence_length,
        sample_rate=config.data.sample_rate,
        seed=42,
    )

    # Create output directory
    output_dir = Path("figures")
    output_dir.mkdir(exist_ok=True)

    print("\n" + "="*70)
    print("LATENT SPACE VISUALIZATION")
    print("="*70)

    # Extract latents for each type
    print("\nExtracting latent representations...")

    latents_dict = {}

    # Normal samples
    normal_dataset = RFDataset.from_generator(
        generator=generator,
        num_samples=500,
        anomaly_ratio=0.0,
        snr_range=tuple(config.data.snr_range),
    )
    normal_loader = DataLoader(normal_dataset, batch_size=64, shuffle=False)
    normal_latents, _, _ = extract_latents_with_labels(model, normal_loader, device)
    latents_dict['normal'] = normal_latents
    print(f"  normal: {normal_latents.shape}")

    # Each anomaly type
    anomaly_types = ["interference", "frequency_drift", "amplitude_spike", "phase_noise", "burst_noise"]

    for atype in anomaly_types:
        dataset = create_dataset_with_types(generator, config, atype, num_samples=300)
        loader = DataLoader(dataset, batch_size=64, shuffle=False)
        latents, _, _ = extract_latents_with_labels(model, loader, device)
        latents_dict[atype] = latents
        print(f"  {atype}: {latents.shape}")

    # Generate visualizations
    print("\nGenerating visualizations...")

    # 1. t-SNE by anomaly type
    plot_tsne_by_anomaly_type(
        latents_dict,
        save_path=output_dir / "latent_tsne_by_type.png"
    )

    # 2. UMAP by anomaly type
    plot_umap_by_anomaly_type(
        latents_dict,
        save_path=output_dir / "latent_umap_by_type.png"
    )

    # 3. Mahalanobis distance distribution
    plot_mahalanobis_distribution(
        latents_dict,
        save_path=output_dir / "mahalanobis_distribution.png"
    )

    # 4. Latent dimension analysis
    plot_latent_dimension_analysis(
        latents_dict,
        save_path=output_dir / "latent_dimension_analysis.png"
    )

    # 5. Simple normal vs anomaly t-SNE
    all_latents = np.concatenate([latents_dict['normal']] + [latents_dict[t] for t in anomaly_types])
    all_labels = np.concatenate([
        np.zeros(len(latents_dict['normal'])),
        *[np.ones(len(latents_dict[t])) for t in anomaly_types]
    ])
    plot_tsne_by_class(
        all_latents,
        all_labels,
        "VAE Latent Space: Normal vs Anomaly",
        save_path=output_dir / "latent_tsne_normal_vs_anomaly.png"
    )

    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)
    print(f"\nFigures saved to: {output_dir.absolute()}")
    print("\nGenerated files:")
    for f in sorted(output_dir.glob("*.png")):
        print(f"  - {f.name}")


if __name__ == "__main__":
    main()
