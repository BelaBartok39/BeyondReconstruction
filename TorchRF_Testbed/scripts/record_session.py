#!/usr/bin/env python3
"""Record labeled RF dataset with automatic anomaly injection.

Captures RF signals and automatically injects anomalies at specified ratio
to create a labeled dataset for training/testing.

Usage:
    python scripts/record_session.py --output dataset.h5 --samples 1000 --anomaly-ratio 0.2
    python scripts/record_session.py --output dataset.h5 --samples 500 --simulate
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

# Add CLP_Project root first, then testbed
_CLP_ROOT = Path(__file__).parent.parent.parent
_TESTBED_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_CLP_ROOT))
sys.path.insert(0, str(_TESTBED_ROOT))

from TorchRF_Testbed.src.capture import create_capture, is_gnuradio_available
from TorchRF_Testbed.src.injection import inject_anomaly, get_anomaly_types
from TorchRF_Testbed.src.recorder import SessionRecorder
from TorchRF_Testbed.src.utils import estimate_snr, estimate_power, normalize_signal


def record_dataset(
    output_path: str,
    num_samples: int = 1000,
    anomaly_ratio: float = 0.2,
    center_freq: float = 915e6,
    sample_rate: float = 2e6,
    gain: float = 40,
    use_simulation: bool = False,
    anomaly_types: list[str] | None = None,
    severity: float = 1.0,
    verbose: bool = True,
) -> dict:
    """Record a labeled dataset.

    Args:
        output_path: Path to output HDF5 file.
        num_samples: Total number of samples to record.
        anomaly_ratio: Fraction of samples that are anomalies (0-1).
        center_freq: Center frequency in Hz.
        sample_rate: Sample rate in Hz.
        gain: RF gain in dB.
        use_simulation: Use simulated capture.
        anomaly_types: List of anomaly types to inject. If None, uses all.
        severity: Anomaly severity multiplier.
        verbose: Print progress.

    Returns:
        Dict with recording statistics.
    """
    rng = np.random.default_rng()

    # Calculate anomaly schedule
    num_anomalies = int(num_samples * anomaly_ratio)
    num_normal = num_samples - num_anomalies

    # Create random indices for anomaly injection
    anomaly_indices = set(rng.choice(num_samples, num_anomalies, replace=False))

    # Initialize capture and recorder
    capture = create_capture(
        use_simulation=use_simulation,
        center_freq=center_freq,
        sample_rate=sample_rate,
        gain=gain,
    )

    recorder = SessionRecorder(
        output_path,
        sample_rate=sample_rate,
        center_freq=center_freq,
        overwrite=True,
    )

    # Get available anomaly types
    if anomaly_types is None:
        anomaly_types = get_anomaly_types()

    if verbose:
        print(f"\nRecording dataset to: {output_path}")
        print(f"  Total samples: {num_samples}")
        print(f"  Normal: {num_normal} ({100*(1-anomaly_ratio):.0f}%)")
        print(f"  Anomalies: {num_anomalies} ({100*anomaly_ratio:.0f}%)")
        print(f"  Anomaly types: {', '.join(anomaly_types)}")
        print(f"  Severity: {severity}")
        print()

    stats = {
        "normal": 0,
        "anomaly": 0,
        "by_type": {t: 0 for t in anomaly_types},
    }

    try:
        capture.start()

        for i in range(num_samples):
            # Read sample
            try:
                signal = capture.read_samples(timeout=1.0)
            except TimeoutError:
                if verbose:
                    print(f"Warning: Timeout reading sample {i}")
                continue

            # Determine if this should be anomaly
            is_anomaly = i in anomaly_indices
            anomaly_type = None

            if is_anomaly:
                # Select random anomaly type
                anomaly_type = rng.choice(anomaly_types)

                # Inject anomaly
                signal, metadata = inject_anomaly(
                    signal,
                    anomaly_type=anomaly_type,
                    severity=severity,
                    sample_rate=sample_rate,
                    rng=rng,
                )

                stats["anomaly"] += 1
                stats["by_type"][anomaly_type] += 1
            else:
                stats["normal"] += 1

            # Estimate SNR and power
            snr_db = estimate_snr(signal)
            power_db = estimate_power(signal)

            # Record
            recorder.add_sample(
                signal,
                label=is_anomaly,
                anomaly_type=anomaly_type,
                snr_db=snr_db,
                power_db=power_db,
            )

            # Progress
            if verbose and (i + 1) % max(1, num_samples // 20) == 0:
                progress = (i + 1) / num_samples * 100
                print(f"  Progress: {progress:5.1f}% ({i + 1}/{num_samples})")

    except KeyboardInterrupt:
        if verbose:
            print("\nInterrupted by user.")
    finally:
        capture.stop()
        recorder.close()

    if verbose:
        print(f"\nRecording complete!")
        print(f"  File: {output_path}")
        print(f"  Total samples: {recorder.sample_count}")
        print(f"  Normal: {stats['normal']}")
        print(f"  Anomalies: {stats['anomaly']}")
        if stats["anomaly"] > 0:
            print("  Anomaly breakdown:")
            for atype, count in stats["by_type"].items():
                if count > 0:
                    print(f"    {atype}: {count}")

    return stats


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Record labeled RF dataset",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Output settings
    parser.add_argument(
        "--output", "-o",
        type=str,
        required=True,
        help="Output HDF5 file path",
    )
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=1000,
        help="Number of samples to record",
    )
    parser.add_argument(
        "--anomaly-ratio", "-r",
        type=float,
        default=0.2,
        help="Fraction of samples that are anomalies (0-1)",
    )

    # Capture settings
    parser.add_argument(
        "--freq", "-f",
        type=float,
        default=915e6,
        help="Center frequency in Hz",
    )
    parser.add_argument(
        "--sample-rate", "-s",
        type=float,
        default=2e6,
        help="Sample rate in Hz",
    )
    parser.add_argument(
        "--gain", "-g",
        type=float,
        default=40,
        help="RF gain in dB",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Use simulated capture",
    )

    # Anomaly settings
    parser.add_argument(
        "--anomaly-types",
        type=str,
        nargs="+",
        default=None,
        help="Specific anomaly types to inject",
    )
    parser.add_argument(
        "--severity",
        type=float,
        default=1.0,
        help="Anomaly severity multiplier",
    )

    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output",
    )

    args = parser.parse_args()

    # Validate
    if args.anomaly_ratio < 0 or args.anomaly_ratio > 1:
        parser.error("Anomaly ratio must be between 0 and 1")

    if not args.simulate and not is_gnuradio_available():
        print("Warning: GNURadio not available. Using simulated capture.")
        args.simulate = True

    # Record
    record_dataset(
        output_path=args.output,
        num_samples=args.samples,
        anomaly_ratio=args.anomaly_ratio,
        center_freq=args.freq,
        sample_rate=args.sample_rate,
        gain=args.gain,
        use_simulation=args.simulate,
        anomaly_types=args.anomaly_types,
        severity=args.severity,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
