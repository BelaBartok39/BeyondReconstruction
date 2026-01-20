#!/usr/bin/env python3
"""Standalone launcher for live detection CLI."""

import sys
from pathlib import Path

# Setup paths correctly - CLP_Project first
CLP_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(CLP_ROOT))

# Now import and run
from TorchRF_Testbed.src.capture import HackRFCapture
from TorchRF_Testbed.src.detector import LiveDetector
from TorchRF_Testbed.src.injection import inject_anomaly, get_anomaly_types
from TorchRF_Testbed.src.recorder import SessionRecorder

import numpy as np
import time
import select
import sys
import termios
import tty


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Live RF anomaly detection")
    parser.add_argument("--freq", type=float, default=2.437e9, help="Center frequency")
    parser.add_argument("--gain", type=float, default=40, help="RF gain")
    parser.add_argument("--record", type=str, default=None, help="Record to HDF5 file")
    args = parser.parse_args()

    print("=" * 70)
    print("TorchRF Testbed - Live Detection")
    print("=" * 70)
    print(f"Frequency: {args.freq/1e9:.3f} GHz | Gain: {args.gain} dB")
    print("\nKeyboard Controls:")
    print("  [SPACE] Random  [T] Tone  [C] Chirp  [B] Barrage")
    print("  [P] Pulse  [A] Amp spike  [N] Burst noise  [Q] Quit")
    print()

    # Load detector
    print("Loading model...")
    detector = LiveDetector(
        model_path=str(CLP_ROOT / "snr_conditioned_vae_hybrid_v1.pt"),
        config_path=str(CLP_ROOT / "configs" / "default.yaml"),
        device="cpu",
    )

    # Create capture
    print("Initializing HackRF...")
    capture = HackRFCapture(
        center_freq=args.freq,
        sample_rate=2e6,
        gain=args.gain,
        if_gain=32,
        bb_gain=32,
        buffer_size=1024,
    )

    # Optional recorder
    recorder = None
    if args.record:
        recorder = SessionRecorder(args.record, sample_rate=2e6, center_freq=args.freq)

    capture.start()
    time.sleep(0.3)

    # Calibrate
    print("Calibrating...")
    normal_signals = [capture.read_samples(1024, timeout=1.0) for _ in range(20)]
    detector.fit(normal_signals)
    print(f"Threshold: {detector.threshold:.4f}")

    print("\n" + "-" * 70)
    print(f"{'Time':<10} | {'SNR':>6} | {'Power':>8} | {'Score':>8} | Status")
    print("-" * 70)

    # Setup terminal for non-blocking input
    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    key_map = {
        " ": None,  # Random
        "t": "tone",
        "c": "chirp",
        "b": "barrage",
        "p": "pulse",
        "a": "amplitude_spike",
        "n": "burst_noise",
        "f": "frequency_drift",
    }

    running = True
    pending_injection = None
    rng = np.random.default_rng()

    try:
        while running:
            # Check for key press
            if select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.read(1).lower()
                if key == "q":
                    running = False
                    continue
                elif key in key_map:
                    if key == " ":
                        pending_injection = rng.choice(get_anomaly_types())
                    else:
                        pending_injection = key_map[key]

            # Capture and detect
            samples = capture.read_samples(1024, timeout=1.0)

            inject_label = ""
            if pending_injection:
                samples, _ = inject_anomaly(samples, anomaly_type=pending_injection, severity=5.0)
                inject_label = f" [{pending_injection}]"
                pending_injection = None

            result = detector.detect(samples)

            if result.is_anomaly:
                status = "\033[91mANOMALY\033[0m"
            else:
                status = "\033[92mNORMAL\033[0m"

            timestamp = time.strftime("%H:%M:%S")
            print(f"{timestamp:<10} | {result.snr_db:>6.1f} | {result.power_db:>8.1f} | {result.score:>8.4f} | {status}{inject_label}")

            # Record if enabled
            if recorder:
                recorder.add_sample(
                    samples,
                    label=result.is_anomaly,
                    anomaly_type=inject_label.strip(" []") if inject_label else None,
                    snr_db=result.snr_db,
                    power_db=result.power_db,
                    score=result.score,
                )

            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n\nInterrupted.")
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        capture.stop()
        if recorder:
            recorder.close()
            print(f"Recording saved: {args.record}")
        print("\nSession ended.")


if __name__ == "__main__":
    main()
