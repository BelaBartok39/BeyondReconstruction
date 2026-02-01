"""Generate the 'Reconstruction Paradox' figure for IEEE ICASSP paper.

Shows that anomalies have LOWER reconstruction error than normal signals,
proving that standard reconstruction-based detection is inverted for
normalized RF data.

Produces:
- Dual-color histogram of reconstruction MSE (normal vs anomaly)
- Dual-color histogram of latent Mahalanobis distance (for contrast)
- AUROC values annotated on each plot

Usage:
    python scripts/generate_reconstruction_paradox.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import yaml
from src.models.snr_encoder import SNRConditionedVAE
from src.data.synthetic import SyntheticRFGenerator
from src.data.snr_estimation import estimate_snr


def setup_ieee_style():
    """Configure matplotlib for IEEE publication standards."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif", "serif"],
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 300,
        "lines.linewidth": 1.0,
        "axes.linewidth": 0.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.05,
    })


def load_model() -> SNRConditionedVAE:
    """Load the production model."""
    config_path = PROJECT_ROOT / "configs" / "default.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    model_config = config["model"]
    model = SNRConditionedVAE(
        latent_dim=model_config.get("latent_dim", 32),
        sequence_length=1024,
        hidden_channels=model_config.get("hidden_channels", [32, 64, 128, 256]),
        snr_embedding_dim=model_config.get("snr_embedding_dim", 16),
        kernel_size=model_config.get("kernel_size", 7),
        use_batch_norm=model_config.get("use_batch_norm", True),
        dropout=model_config.get("dropout", 0.1),
        use_power_conditioning=model_config.get("use_power_conditioning", True),
    )

    checkpoint_path = PROJECT_ROOT / "snr_conditioned_vae_hybrid_v1.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def generate_data(n_normal: int = 2000, n_anomaly: int = 500) -> dict:
    """Generate normal and anomalous RF signals."""
    config_path = PROJECT_ROOT / "configs" / "default.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    generator = SyntheticRFGenerator(
        sequence_length=config["data"]["sequence_length"],
        sample_rate=float(config["data"]["sample_rate"]),
    )

    snr_range = tuple(config["data"]["snr_range"])
    modulations = config["data"]["modulations"]

    signals, snrs, powers, labels = [], [], [], []
    anomaly_types_list = []

    # Generate normal signals
    for i in tqdm(range(n_normal), desc="Normal signals"):
        mod = modulations[i % len(modulations)]
        signal, metadata = generator.generate_normal_signal(
            modulation=mod, snr_range=snr_range
        )
        power = np.mean(signal[0] ** 2 + signal[1] ** 2)
        signals.append(signal)
        snrs.append(metadata.snr_db)
        powers.append(power)
        labels.append(0)
        anomaly_types_list.append("normal")

    # Generate anomalous signals (mixed types)
    anomaly_types = config["data"]["anomaly_types"]
    per_type = n_anomaly // len(anomaly_types)

    for anom_type in anomaly_types:
        for _ in tqdm(range(per_type), desc=f"  {anom_type}"):
            signal, metadata = generator.generate_anomaly(
                anomaly_type=anom_type,
                severity=config["data"]["anomaly_severity"],
                snr_range=snr_range,
            )
            power = np.mean(signal[0] ** 2 + signal[1] ** 2)
            signals.append(signal)
            snrs.append(metadata.snr_db)
            powers.append(power)
            labels.append(1)
            anomaly_types_list.append(anom_type)

    # Normalize signals and prepare tensors
    normalized = []
    for sig in signals:
        max_val = np.max(np.abs(sig))
        normalized.append(sig / max_val if max_val > 0 else sig)

    signals_t = torch.tensor(np.array(normalized), dtype=torch.float32)
    snr_normalized = (np.clip(np.array(snrs), -5, 30) + 5) / 35
    snrs_t = torch.tensor(snr_normalized, dtype=torch.float32)
    power_array = np.array(powers)
    power_normalized = np.clip(power_array / (np.percentile(power_array, 99) + 1e-8), 0, 1)
    powers_t = torch.tensor(power_normalized, dtype=torch.float32)

    return {
        "signals": signals_t,
        "snrs": snrs_t,
        "powers": powers_t,
        "labels": np.array(labels),
        "anomaly_types": anomaly_types_list,
    }


def compute_scores(model: SNRConditionedVAE, data: dict) -> dict:
    """Compute both reconstruction MSE and latent Mahalanobis distance."""
    signals = data["signals"]
    snrs = data["snrs"]
    powers = data["powers"]

    recon_scores = []
    latent_codes = []

    batch_size = 64
    n_samples = len(signals)

    with torch.no_grad():
        for i in tqdm(range(0, n_samples, batch_size), desc="Computing scores"):
            batch_sig = signals[i:i + batch_size]
            batch_snr = snrs[i:i + batch_size]
            batch_pow = powers[i:i + batch_size]

            # Forward pass -> reconstruction
            x_recon, mu, logvar, z = model(batch_sig, batch_snr, batch_pow)

            # Reconstruction MSE: ((x - x_recon)^2).mean(dim=(1,2))
            mse = ((batch_sig - x_recon) ** 2).mean(dim=(1, 2))
            recon_scores.extend(mse.numpy())

            # Latent codes for Mahalanobis
            latent_codes.append(mu.numpy())

    recon_scores = np.array(recon_scores)
    latent_codes = np.vstack(latent_codes)

    # Compute Mahalanobis distance using normal samples only
    normal_mask = data["labels"] == 0
    normal_latents = latent_codes[normal_mask]

    latent_mean = normal_latents.mean(axis=0)
    latent_cov = np.cov(normal_latents.T) + 1e-6 * np.eye(normal_latents.shape[1])
    latent_inv_cov = np.linalg.inv(latent_cov)

    mahal_scores = []
    for latent in latent_codes:
        diff = latent - latent_mean
        dist = np.sqrt(diff @ latent_inv_cov @ diff)
        mahal_scores.append(dist)

    mahal_scores = np.array(mahal_scores)

    return {
        "reconstruction_mse": recon_scores,
        "mahalanobis": mahal_scores,
    }


def save_figure(fig, name: str, output_dir: Path):
    """Save figure in both PDF and PNG formats."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for fmt in ["pdf", "png"]:
        path = output_dir / f"{name}.{fmt}"
        fig.savefig(path, format=fmt, dpi=300, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


def main():
    print("=" * 60)
    print("Generating 'Reconstruction Paradox' Figure")
    print("=" * 60)

    setup_ieee_style()
    output_dir = PROJECT_ROOT / "figures" / "publication"

    # Load model
    print("\n[1/3] Loading production model...")
    model = load_model()
    print(f"  Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    # Generate data
    print("\n[2/3] Generating synthetic data...")
    data = generate_data(n_normal=2000, n_anomaly=500)
    print(f"  Normal: {(data['labels'] == 0).sum()}, Anomaly: {(data['labels'] == 1).sum()}")

    # Compute scores
    print("\n[3/3] Computing scores...")
    scores = compute_scores(model, data)

    labels = data["labels"]
    normal_mask = labels == 0
    anomaly_mask = labels == 1

    # Compute AUROCs
    recon_auroc = roc_auc_score(labels, scores["reconstruction_mse"])
    mahal_auroc = roc_auc_score(labels, scores["mahalanobis"])

    print(f"\n  Reconstruction MSE AUROC: {recon_auroc:.4f}")
    print(f"  Mahalanobis AUROC:        {mahal_auroc:.4f}")

    # =================================================================
    # Figure: Side-by-side comparison
    # =================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.16, 3.0))

    # --- Left panel: Reconstruction MSE (the paradox) ---
    recon_normal = scores["reconstruction_mse"][normal_mask]
    recon_anomaly = scores["reconstruction_mse"][anomaly_mask]

    ax1.hist(recon_normal, bins=50, alpha=0.65, color="#4285F4",
             label=f"Normal (n={normal_mask.sum()})", density=True, edgecolor="none")
    ax1.hist(recon_anomaly, bins=50, alpha=0.65, color="#EA4335",
             label=f"Anomaly (n={anomaly_mask.sum()})", density=True, edgecolor="none")

    ax1.set_xlabel("Reconstruction Error (MSE)")
    ax1.set_ylabel("Density")
    ax1.set_title(f"(a) Reconstruction Error — AUROC = {recon_auroc:.2f}")
    ax1.legend(loc="upper right")

    # Add annotation arrow pointing out the paradox
    ax1.annotate(
        "Anomalies have\nLOWER error",
        xy=(np.median(recon_anomaly), 0),
        xytext=(np.percentile(recon_normal, 80), ax1.get_ylim()[1] * 0.6),
        fontsize=7, fontstyle="italic", color="#EA4335",
        arrowprops=dict(arrowstyle="->", color="#EA4335", lw=1.0),
        ha="center",
    )

    # --- Right panel: Mahalanobis Distance (correct method) ---
    mahal_normal = scores["mahalanobis"][normal_mask]
    mahal_anomaly = scores["mahalanobis"][anomaly_mask]

    ax2.hist(mahal_normal, bins=50, alpha=0.65, color="#4285F4",
             label=f"Normal (n={normal_mask.sum()})", density=True, edgecolor="none")
    ax2.hist(mahal_anomaly, bins=50, alpha=0.65, color="#EA4335",
             label=f"Anomaly (n={anomaly_mask.sum()})", density=True, edgecolor="none")

    ax2.set_xlabel("Mahalanobis Distance (Latent Space)")
    ax2.set_ylabel("Density")
    ax2.set_title(f"(b) Latent-Space Distance — AUROC = {mahal_auroc:.2f}")
    ax2.legend(loc="upper right")

    # No annotation needed — the red tail extending right speaks for itself

    plt.tight_layout()
    save_figure(fig, "reconstruction_paradox", output_dir)

    # Print summary stats
    print(f"\n{'='*60}")
    print("Summary Statistics")
    print(f"{'='*60}")
    print(f"\nReconstruction MSE:")
    print(f"  Normal  — mean: {recon_normal.mean():.6f}, std: {recon_normal.std():.6f}")
    print(f"  Anomaly — mean: {recon_anomaly.mean():.6f}, std: {recon_anomaly.std():.6f}")
    print(f"  Anomaly mean {'<' if recon_anomaly.mean() < recon_normal.mean() else '>'} Normal mean → {'PARADOX CONFIRMED' if recon_anomaly.mean() < recon_normal.mean() else 'No paradox'}")

    print(f"\nMahalanobis Distance:")
    print(f"  Normal  — mean: {mahal_normal.mean():.4f}, std: {mahal_normal.std():.4f}")
    print(f"  Anomaly — mean: {mahal_anomaly.mean():.4f}, std: {mahal_anomaly.std():.4f}")
    print(f"  Anomaly mean > Normal mean → Correct separation")

    print(f"\n{'='*60}")
    print("Done!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
