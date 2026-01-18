"""Analyze why VAE reconstructs anomalies better than normal signals."""

import sys
sys.path.insert(0, '/home/babynicky/Work/CLP_Project')

import numpy as np
import matplotlib.pyplot as plt
from src.data.synthetic import SyntheticRFGenerator, AnomalyType

def compute_signal_complexity(iq: np.ndarray) -> dict:
    """Compute various measures of signal complexity."""
    # Convert to complex
    signal = iq[0] + 1j * iq[1]

    # 1. Spectral flatness (closer to 1 = more noise-like, closer to 0 = more tonal)
    fft = np.fft.fft(signal)
    power_spectrum = np.abs(fft) ** 2
    geometric_mean = np.exp(np.mean(np.log(power_spectrum + 1e-10)))
    arithmetic_mean = np.mean(power_spectrum)
    spectral_flatness = geometric_mean / (arithmetic_mean + 1e-10)

    # 2. Amplitude variance (how much does amplitude vary?)
    amplitude = np.abs(signal)
    amp_variance = np.var(amplitude)
    amp_range = np.max(amplitude) - np.min(amplitude)

    # 3. Phase variance (how chaotic is the phase?)
    phase = np.angle(signal)
    # Unwrap to handle wrap-around
    phase_unwrapped = np.unwrap(phase)
    phase_diff = np.diff(phase_unwrapped)
    phase_variance = np.var(phase_diff)

    # 4. Autocorrelation at lag 1 (higher = more predictable)
    autocorr = np.corrcoef(signal[:-1].real, signal[1:].real)[0, 1]

    # 5. Entropy of amplitude histogram
    hist, _ = np.histogram(amplitude, bins=50, density=True)
    hist = hist[hist > 0]
    entropy = -np.sum(hist * np.log(hist + 1e-10))

    return {
        'spectral_flatness': spectral_flatness,
        'amp_variance': amp_variance,
        'amp_range': amp_range,
        'phase_variance': phase_variance,
        'autocorrelation': autocorr,
        'entropy': entropy,
    }


def analyze_signal_types():
    """Compare complexity of normal vs anomaly signals."""
    generator = SyntheticRFGenerator(sequence_length=1024, sample_rate=1e6, seed=42)

    # Generate many samples
    n_samples = 200
    snr_range = (-5, 30)

    normal_metrics = []
    anomaly_metrics = {atype.value: [] for atype in AnomalyType}

    # Generate normal signals
    for _ in range(n_samples):
        iq, _ = generator.generate_normal_signal(snr_range=snr_range)
        normal_metrics.append(compute_signal_complexity(iq))

    # Generate anomaly signals by type
    for atype in AnomalyType:
        for _ in range(n_samples):
            iq, _ = generator.generate_anomaly(anomaly_type=atype, snr_range=snr_range)
            anomaly_metrics[atype.value].append(compute_signal_complexity(iq))

    # Aggregate stats
    def aggregate(metrics_list):
        if not metrics_list:
            return {}
        keys = metrics_list[0].keys()
        return {k: (np.mean([m[k] for m in metrics_list]),
                   np.std([m[k] for m in metrics_list])) for k in keys}

    print("=" * 70)
    print("SIGNAL COMPLEXITY ANALYSIS")
    print("=" * 70)
    print(f"\nSamples per type: {n_samples}, SNR range: {snr_range}")

    # Normal stats
    normal_agg = aggregate(normal_metrics)
    print("\n--- NORMAL SIGNALS ---")
    for k, (mean, std) in normal_agg.items():
        print(f"  {k:20s}: {mean:.4f} +/- {std:.4f}")

    # Anomaly stats
    for atype in AnomalyType:
        agg = aggregate(anomaly_metrics[atype.value])
        print(f"\n--- {atype.value.upper()} ---")
        for k, (mean, std) in agg.items():
            normal_mean = normal_agg[k][0]
            diff = mean - normal_mean
            diff_pct = (diff / normal_mean) * 100 if normal_mean != 0 else 0
            direction = "+" if diff > 0 else ""
            print(f"  {k:20s}: {mean:.4f} +/- {std:.4f}  ({direction}{diff_pct:.1f}% vs normal)")

    return normal_metrics, anomaly_metrics


def visualize_normalization_effect():
    """Show how normalization affects anomalies."""
    generator = SyntheticRFGenerator(sequence_length=1024, sample_rate=1e6, seed=42)

    # Generate a normal signal
    iq_normal, _ = generator.generate_normal_signal(modulation='qpsk', snr_db=20)

    # Generate each anomaly type
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    axes = axes.flatten()

    # Plot normal signal
    axes[0].plot(iq_normal[0, :200], label='I', alpha=0.7)
    axes[0].plot(iq_normal[1, :200], label='Q', alpha=0.7)
    axes[0].set_title('Normal QPSK Signal (normalized)')
    axes[0].legend()
    axes[0].set_ylim(-1.2, 1.2)

    # Plot anomalies
    for i, atype in enumerate(AnomalyType):
        ax = axes[i + 1]
        iq, meta = generator.generate_anomaly(anomaly_type=atype, snr_db=20)
        ax.plot(iq[0, :200], label='I', alpha=0.7)
        ax.plot(iq[1, :200], label='Q', alpha=0.7)
        ax.set_title(f'{atype.value} (normalized)')
        ax.legend()
        ax.set_ylim(-1.2, 1.2)

    plt.tight_layout()
    plt.savefig('/home/babynicky/Work/CLP_Project/experiments/signal_comparison.png', dpi=150)
    print("\nSaved: experiments/signal_comparison.png")
    plt.close()


def analyze_pre_vs_post_normalization():
    """Critical: Show what happens before vs after normalization."""

    # We need to modify the generator temporarily to get pre-normalized signals
    # Let's recreate the signal generation manually

    import numpy as np

    def generate_signal_no_normalize(generator, add_anomaly=None):
        """Generate signal without final normalization."""
        num_symbols = generator.sequence_length // generator.samples_per_symbol + 10
        constellation = generator._constellations[generator.rng.choice(list(generator._constellations.keys()))]
        indices = generator.rng.integers(0, len(constellation), num_symbols)
        symbols = constellation[indices]
        signal = generator._pulse_shape(symbols)
        signal = generator._add_carrier(signal)

        if add_anomaly:
            signal, _ = generator._anomaly_generators[add_anomaly](signal, snr_db=20)

        signal = generator._add_awgn(signal, snr_db=20)
        return signal

    generator = SyntheticRFGenerator(sequence_length=1024, sample_rate=1e6, seed=123)

    print("\n" + "=" * 70)
    print("PRE-NORMALIZATION SIGNAL POWER ANALYSIS")
    print("=" * 70)

    # Compute average power before normalization
    n_samples = 100

    normal_powers = []
    for _ in range(n_samples):
        sig = generate_signal_no_normalize(generator)
        normal_powers.append(np.mean(np.abs(sig) ** 2))

    print(f"\nNormal signal power: {np.mean(normal_powers):.4f} +/- {np.std(normal_powers):.4f}")
    print(f"Normal max amplitude: {np.sqrt(np.mean(normal_powers)) * 2:.4f} (approx)")

    for atype in AnomalyType:
        powers = []
        max_amps = []
        for _ in range(n_samples):
            sig = generate_signal_no_normalize(generator, add_anomaly=atype)
            powers.append(np.mean(np.abs(sig) ** 2))
            max_amps.append(np.max(np.abs(sig)))

        power_ratio = np.mean(powers) / np.mean(normal_powers)
        print(f"\n{atype.value}:")
        print(f"  Power: {np.mean(powers):.4f} +/- {np.std(powers):.4f} ({power_ratio:.2f}x normal)")
        print(f"  Max amplitude: {np.mean(max_amps):.4f} +/- {np.std(max_amps):.4f}")

        # After normalization, all max amplitudes become 1.0
        # So anomalies with high max amplitude get MORE compressed
        compression_factor = np.mean(max_amps) / np.sqrt(np.mean(normal_powers))
        print(f"  Compression factor: {compression_factor:.2f}x more than normal")


def analyze_reconstruction_difficulty():
    """Analyze what makes a signal easy or hard to reconstruct."""
    print("\n" + "=" * 70)
    print("RECONSTRUCTION DIFFICULTY ANALYSIS")
    print("=" * 70)

    print("""
HYPOTHESIS: VAE reconstructs anomalies better because:

1. AMPLITUDE SPIKE anomalies:
   - Before norm: signal has large spike (3-10x amplitude)
   - After norm: spike becomes 1.0, rest of signal becomes TINY (0.1-0.3)
   - Result: Most of signal is near zero = EASY to reconstruct!

2. INTERFERENCE anomalies:
   - Adds a pure sinusoid (extremely structured)
   - Sinusoids are EASIER to reconstruct than modulated signals
   - The interference dominates after normalization

3. BURST NOISE anomalies:
   - Adds Gaussian noise in bursts
   - Gaussian noise is what the VAE sees during training (AWGN)
   - The model is trained to reconstruct through noise = familiar pattern

4. PHASE NOISE anomalies:
   - Random walk phase rotation
   - Preserves amplitude, just rotates phase
   - May be easy because it doesn't change the "shape" much

5. FREQUENCY DRIFT anomalies:
   - Systematic phase rotation (chirp-like)
   - Very structured transformation
   - Model might find this predictable

CORE PROBLEM:
- The normalization step (divide by max amplitude) DESTROYS anomaly signatures
- Anomalies with high amplitude spikes become "empty" signals
- The VAE sees "mostly zeros" which is trivially easy to reconstruct
""")


if __name__ == '__main__':
    analyze_reconstruction_difficulty()
    analyze_pre_vs_post_normalization()
    normal_metrics, anomaly_metrics = analyze_signal_types()
    visualize_normalization_effect()
