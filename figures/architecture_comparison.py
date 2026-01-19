#!/usr/bin/env python3
"""Generate architecture comparison visualizations."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

# Set up figure style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.size'] = 10
plt.rcParams['figure.facecolor'] = 'white'


def draw_block(ax, x, y, width, height, label, color='lightblue', fontsize=9):
    """Draw a rounded rectangle block."""
    box = FancyBboxPatch(
        (x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        facecolor=color, edgecolor='black', linewidth=1.5
    )
    ax.add_patch(box)
    ax.text(x, y, label, ha='center', va='center', fontsize=fontsize, fontweight='bold')


def draw_arrow(ax, start, end, color='black'):
    """Draw an arrow between two points."""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5))


def create_architecture_diagram():
    """Create side-by-side architecture comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 8))

    # Colors
    input_color = '#FFE4B5'  # Moccasin
    encoder_color = '#87CEEB'  # Sky blue
    latent_color = '#98FB98'  # Pale green
    decoder_color = '#DDA0DD'  # Plum
    output_color = '#F0E68C'  # Khaki
    cond_color = '#FFA07A'  # Light salmon

    # ============ Architecture 1: Standard VAE ============
    ax = axes[0]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Standard VAE\nAUROC: 0.42 (reconstruction)', fontsize=12, fontweight='bold')

    # Input
    draw_block(ax, 5, 11, 3, 1, 'I/Q Signal\n[B, 2, 1024]', input_color)
    draw_arrow(ax, (5, 10.5), (5, 9.5))

    # Encoder
    draw_block(ax, 5, 8.5, 3.5, 2, 'Conv Encoder\n32→64→128→256', encoder_color)
    draw_arrow(ax, (5, 7.5), (5, 6.5))

    # Latent
    draw_block(ax, 5, 5.5, 2.5, 2, 'Latent z\nμ, σ² → z\n[B, 32]', latent_color)
    draw_arrow(ax, (5, 4.5), (5, 3.5))

    # Decoder
    draw_block(ax, 5, 2.5, 3.5, 2, 'Conv Decoder\n256→128→64→32', decoder_color)
    draw_arrow(ax, (5, 1.5), (5, 0.5))

    # Output
    draw_block(ax, 5, -0.5, 3, 1, 'Reconstructed\n[B, 2, 1024]', output_color)

    # ============ Architecture 2: SNR-Conditioned VAE ============
    ax = axes[1]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('SNR-Conditioned VAE\nAUROC: 0.94 (latent)', fontsize=12, fontweight='bold')

    # Input
    draw_block(ax, 5, 11, 3, 1, 'I/Q Signal\n[B, 2, 1024]', input_color)
    draw_arrow(ax, (5, 10.5), (5, 9.5))

    # SNR/Power conditioning
    draw_block(ax, 1.5, 8.5, 2, 1.5, 'SNR\nPower', cond_color)
    draw_block(ax, 1.5, 5.5, 2, 1.5, 'Cond\nEmbed', cond_color)
    draw_arrow(ax, (1.5, 7.75), (1.5, 6.25))

    # Encoder
    draw_block(ax, 5, 8.5, 3.5, 2, 'Conv Encoder\n+ Cond Embed', encoder_color)
    draw_arrow(ax, (2.5, 8.5), (3.25, 8.5))
    draw_arrow(ax, (5, 7.5), (5, 6.5))

    # Latent
    draw_block(ax, 5, 5.5, 2.5, 2, 'Latent z\nμ, σ² → z\n[B, 32]', latent_color)
    draw_arrow(ax, (2.5, 5.5), (3.75, 5.5))
    draw_arrow(ax, (5, 4.5), (5, 3.5))

    # Decoder
    draw_block(ax, 5, 2.5, 3.5, 2, 'Conv Decoder\n+ Cond Embed', decoder_color)
    draw_arrow(ax, (5, 1.5), (5, 0.5))

    # Detection callout
    ax.annotate('Mahalanobis\nDistance', xy=(6.25, 5.5), xytext=(8.5, 5.5),
                fontsize=9, ha='center', va='center',
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))

    # Output
    draw_block(ax, 5, -0.5, 3, 1, 'Reconstructed\n[B, 2, 1024]', output_color)

    # ============ Architecture 3: Complex VAE ============
    ax = axes[2]
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title('Complex-Valued VAE\nAUROC: TBD', fontsize=12, fontweight='bold')

    # Input
    draw_block(ax, 5, 11, 3, 1, 'I/Q Signal\n[B, 2, 1024]', input_color)
    draw_arrow(ax, (5, 10.5), (5, 9.5))

    # Split to complex
    ax.text(5, 9.8, 'I + jQ', fontsize=8, ha='center', style='italic')

    # Complex Encoder
    draw_block(ax, 5, 8.5, 3.5, 2, 'Complex Conv\n(Wr+jWi)*(Xr+jXi)', '#ADD8E6')
    draw_arrow(ax, (5, 7.5), (5, 6.5))

    # Magnitude/Phase
    draw_block(ax, 5, 5.5, 2.5, 2, '|z|, ∠z\n→ Latent\n[B, 32]', latent_color)
    draw_arrow(ax, (5, 4.5), (5, 3.5))

    # SNR/Power conditioning (side)
    draw_block(ax, 1.5, 5.5, 2, 1.5, 'SNR\nPower', cond_color)
    draw_arrow(ax, (2.5, 5.5), (3.75, 5.5))

    # Decoder (real)
    draw_block(ax, 5, 2.5, 3.5, 2, 'Real Decoder\n(standard)', decoder_color)
    draw_arrow(ax, (5, 1.5), (5, 0.5))

    # Output
    draw_block(ax, 5, -0.5, 3, 1, 'Reconstructed\n[B, 2, 1024]', output_color)

    # Phase preservation note
    ax.text(8.5, 8.5, 'Phase\nPreserved!', fontsize=9, ha='center', va='center',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    plt.tight_layout()
    plt.savefig('figures/architecture_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: figures/architecture_comparison.png")


def create_results_comparison():
    """Create bar chart comparing all approaches."""
    fig, ax = plt.subplots(figsize=(12, 6))

    # Data
    methods = [
        'Reconstruction\n(baseline)',
        'Recon + Invert',
        'Latent-only',
        'Latent +\nPower Cond',
        'Latent +\nPhase Hybrid',
        'Phase Loss\nTraining*',
    ]
    aurocs = [0.42, 0.56, 0.91, 0.94, 0.96, 0.79]
    colors = ['#ff6b6b', '#ffa502', '#2ed573', '#1e90ff', '#9b59b6', '#f39c12']

    bars = ax.bar(methods, aurocs, color=colors, edgecolor='black', linewidth=1.5)

    # Add value labels
    for bar, auroc in zip(bars, aurocs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{auroc:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('AUROC', fontsize=12)
    ax.set_title('RF Anomaly Detection: Method Comparison', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
    ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='Target (0.9)')

    # Annotation for phase loss
    ax.annotate('*Freq drift only\n(others degraded)',
                xy=(5, 0.79), xytext=(5, 0.55),
                fontsize=9, ha='center',
                arrowprops=dict(arrowstyle='->', color='gray'))

    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.savefig('figures/results_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: figures/results_comparison.png")


def create_per_anomaly_comparison():
    """Create per-anomaly type comparison."""
    fig, ax = plt.subplots(figsize=(12, 6))

    anomaly_types = ['Interference', 'Freq Drift', 'Amp Spike', 'Phase Noise', 'Burst Noise']
    x = np.arange(len(anomaly_types))
    width = 0.25

    # Data from memory file
    latent_only = [0.9061, 0.8004, 1.0000, 0.9488, 0.9999]
    phase_hybrid = [0.9784, 0.8718, 1.0000, 0.9635, 0.9994]
    phase_loss = [0.7019, 0.7909, 0.3723, 0.8666, 0.4331]

    bars1 = ax.bar(x - width, latent_only, width, label='Latent-only', color='#2ed573', edgecolor='black')
    bars2 = ax.bar(x, phase_hybrid, width, label='+ Phase Hybrid', color='#9b59b6', edgecolor='black')
    bars3 = ax.bar(x + width, phase_loss, width, label='Phase Loss Training', color='#f39c12', edgecolor='black')

    ax.set_ylabel('AUROC', fontsize=12)
    ax.set_title('Per-Anomaly Detection Performance', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(anomaly_types, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='Target (0.9)')
    ax.legend(loc='upper right')

    # Highlight freq drift improvement
    ax.annotate('Target\nAnomaly', xy=(1, 0.87), xytext=(1, 1.05),
                fontsize=9, ha='center', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='red', lw=2))

    plt.tight_layout()
    plt.savefig('figures/per_anomaly_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved: figures/per_anomaly_comparison.png")


if __name__ == '__main__':
    create_architecture_diagram()
    create_results_comparison()
    create_per_anomaly_comparison()
    print("\nAll visualizations created!")
