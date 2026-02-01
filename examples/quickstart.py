"""Quickstart example for RF Anomaly Detection.

This script demonstrates:
1. Loading the production model
2. Generating synthetic RF signals
3. Running anomaly detection
4. Visualizing results

Usage:
    python examples/quickstart.py
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
import matplotlib.pyplot as plt

from src.models.snr_encoder import SNRConditionedVAE, create_model
from src.data.synthetic import SyntheticRFGenerator
from src.detection.detector import AnomalyDetector
from src.utils.config import ConfigLoader


def main():
    # -------------------------------------------------------------------------
    # 1. Setup
    # -------------------------------------------------------------------------
    print("RF Anomaly Detection - Quickstart Example")
    print("=" * 50)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load config
    config_path = project_root / "configs" / "default.yaml"
    config = ConfigLoader.load(config_path)

    # -------------------------------------------------------------------------
    # 2. Load Production Model
    # -------------------------------------------------------------------------
    print("\n[1/4] Loading production model...")

    # Find production model checkpoint
    checkpoint_dirs = [
        project_root / "checkpoints" / "snr_vae_hybrid_v1_20260118",
        project_root / "checkpoints" / "snr_vae_hybrid_v2_20260125",
    ]

    checkpoint_path = None
    for ckpt_dir in checkpoint_dirs:
        best_model = ckpt_dir / "best_model.pt"
        if best_model.exists():
            checkpoint_path = best_model
            break

    if checkpoint_path is None:
        print("No production checkpoint found. Training new model...")
        # Create and train a quick model for demo
        model = create_model(config["model"])
        print("(For full performance, train with: python experiments/train_baseline.py)")
    else:
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model = create_model(config["model"])
        model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)
    model.eval()
    print(f"Model: {type(model).__name__} with {sum(p.numel() for p in model.parameters()):,} parameters")

    # -------------------------------------------------------------------------
    # 3. Generate Synthetic Data
    # -------------------------------------------------------------------------
    print("\n[2/4] Generating synthetic RF signals...")

    generator = SyntheticRFGenerator(
        seq_length=config["data"]["seq_length"],
        sample_rate=config["data"]["sample_rate"],
        snr_range=tuple(config["data"]["snr_range"]),
        modulation_types=config["data"]["modulation_types"],
    )

    # Generate normal signals for fitting detector
    n_train = 200
    train_signals = []
    train_snrs = []
    train_powers = []

    for _ in range(n_train):
        signal, snr = generator.generate_signal()
        power = np.mean(signal[0] ** 2 + signal[1] ** 2)
        train_signals.append(signal)
        train_snrs.append(snr)
        train_powers.append(power)

    # Generate test signals (mix of normal and anomalous)
    n_test = 100
    anomaly_ratio = 0.3
    test_signals = []
    test_snrs = []
    test_powers = []
    test_labels = []

    anomaly_types = ["narrowband_interference", "frequency_drift", "amplitude_spike", "phase_noise", "burst_noise"]

    for i in range(n_test):
        if i < int(n_test * (1 - anomaly_ratio)):
            # Normal signal
            signal, snr = generator.generate_signal()
            test_labels.append(0)
        else:
            # Anomalous signal
            anomaly_type = anomaly_types[i % len(anomaly_types)]
            signal, snr = generator.generate_anomaly(anomaly_type=anomaly_type, severity=4.0)
            test_labels.append(1)

        power = np.mean(signal[0] ** 2 + signal[1] ** 2)
        test_signals.append(signal)
        test_snrs.append(snr)
        test_powers.append(power)

    print(f"Generated {n_train} training samples, {n_test} test samples ({int(anomaly_ratio * 100)}% anomalies)")

    # -------------------------------------------------------------------------
    # 4. Run Anomaly Detection
    # -------------------------------------------------------------------------
    print("\n[3/4] Running anomaly detection...")

    # Create detector
    detector = AnomalyDetector(
        model=model,
        method="latent",  # Use latent-space Mahalanobis distance
        threshold_percentile=95.0,
        snr_adaptive=True,
        device=device,
    )

    # Prepare training data for fitting
    train_batch = {
        "signal": torch.tensor(np.array(train_signals), dtype=torch.float32),
        "snr": torch.tensor(train_snrs, dtype=torch.float32),
        "power": torch.tensor(train_powers, dtype=torch.float32),
    }

    # Fit detector on normal training data
    detector.fit_from_dict(train_batch)
    print("Detector fitted on normal training data")

    # Prepare test data
    test_batch = {
        "signal": torch.tensor(np.array(test_signals), dtype=torch.float32),
        "snr": torch.tensor(test_snrs, dtype=torch.float32),
        "power": torch.tensor(test_powers, dtype=torch.float32),
    }

    # Run detection
    result = detector.detect_from_dict(test_batch)
    test_labels = np.array(test_labels)

    # -------------------------------------------------------------------------
    # 5. Evaluate Results
    # -------------------------------------------------------------------------
    print("\n[4/4] Evaluating results...")

    from sklearn.metrics import roc_auc_score, precision_recall_fscore_support, confusion_matrix

    auroc = roc_auc_score(test_labels, result.scores)
    precision, recall, f1, _ = precision_recall_fscore_support(test_labels, result.predictions, average="binary")
    tn, fp, fn, tp = confusion_matrix(test_labels, result.predictions).ravel()

    print(f"\nResults:")
    print(f"  AUROC:     {auroc:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")
    print(f"  Threshold: {result.threshold:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {tn}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  True Positives:  {tp}")

    # -------------------------------------------------------------------------
    # 6. Visualize
    # -------------------------------------------------------------------------
    print("\nGenerating visualization...")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Score distribution
    ax = axes[0, 0]
    normal_scores = result.scores[test_labels == 0]
    anomaly_scores = result.scores[test_labels == 1]
    ax.hist(normal_scores, bins=30, alpha=0.7, label="Normal", color="blue")
    ax.hist(anomaly_scores, bins=30, alpha=0.7, label="Anomaly", color="red")
    ax.axvline(result.threshold, color="black", linestyle="--", label=f"Threshold ({result.threshold:.2f})")
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Count")
    ax.set_title("Score Distribution")
    ax.legend()

    # ROC curve
    ax = axes[0, 1]
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(test_labels, result.scores)
    ax.plot(fpr, tpr, "b-", linewidth=2, label=f"AUROC = {auroc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Example signals
    ax = axes[1, 0]
    normal_idx = np.where(test_labels == 0)[0][0]
    ax.plot(test_signals[normal_idx][0], label="I", alpha=0.8)
    ax.plot(test_signals[normal_idx][1], label="Q", alpha=0.8)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Normal Signal (score={result.scores[normal_idx]:.2f})")
    ax.legend()

    ax = axes[1, 1]
    anomaly_idx = np.where(test_labels == 1)[0][0]
    ax.plot(test_signals[anomaly_idx][0], label="I", alpha=0.8)
    ax.plot(test_signals[anomaly_idx][1], label="Q", alpha=0.8)
    ax.set_xlabel("Sample")
    ax.set_ylabel("Amplitude")
    ax.set_title(f"Anomaly Signal (score={result.scores[anomaly_idx]:.2f})")
    ax.legend()

    plt.tight_layout()

    output_path = project_root / "figures" / "quickstart_results.png"
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f"Saved visualization to: {output_path}")

    plt.show()

    print("\n" + "=" * 50)
    print("Quickstart complete!")
    print("=" * 50)
    print("\nNext steps:")
    print("  - Train full model: python experiments/train_baseline.py")
    print("  - Evaluate model:   python experiments/evaluate.py --checkpoint <path>")
    print("  - Test on POWDER:   python experiments/test_powder_data.py")


if __name__ == "__main__":
    main()
