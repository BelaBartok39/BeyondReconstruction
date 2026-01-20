#!/usr/bin/env python3
"""Live RF anomaly detection with HackRF.

Main CLI for capturing RF signals and performing real-time anomaly detection
with optional software-based anomaly injection.

Usage:
    python scripts/live_detect.py --freq 915e6 --gain 40
    python scripts/live_detect.py --freq 915e6 --gain 40 --inject
    python scripts/live_detect.py --freq 915e6 --gain 40 --record session.h5
    python scripts/live_detect.py --simulate  # Use simulated capture for testing
"""

from __future__ import annotations

import argparse
import sys
import time
import select
import termios
import tty
from datetime import datetime
from pathlib import Path

import numpy as np
import yaml

# Add CLP_Project root first, then testbed
_CLP_ROOT = Path(__file__).parent.parent.parent
_TESTBED_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_CLP_ROOT))
sys.path.insert(0, str(_TESTBED_ROOT))

from TorchRF_Testbed.src.capture import create_capture, is_gnuradio_available
from TorchRF_Testbed.src.detector import LiveDetector, load_detector
from TorchRF_Testbed.src.injection import inject_anomaly, get_anomaly_types, AnomalyType
from TorchRF_Testbed.src.recorder import SessionRecorder
from TorchRF_Testbed.src.utils import estimate_snr, estimate_power


class LiveDetectionCLI:
    """Live detection command-line interface."""

    def __init__(
        self,
        center_freq: float = 915e6,
        sample_rate: float = 2e6,
        gain: float = 40,
        model_path: str | None = None,
        config_path: str | None = None,
        device: str = "cpu",
        use_simulation: bool = False,
        inject_enabled: bool = False,
        record_path: str | None = None,
    ):
        """Initialize live detection CLI.

        Args:
            center_freq: Center frequency in Hz.
            sample_rate: Sample rate in Hz.
            gain: RF gain in dB.
            model_path: Path to model checkpoint.
            config_path: Path to model config.
            device: Device for inference.
            use_simulation: Use simulated capture.
            inject_enabled: Enable keyboard-triggered injection.
            record_path: Path to record session (optional).
        """
        self.center_freq = center_freq
        self.sample_rate = sample_rate
        self.gain = gain
        self.inject_enabled = inject_enabled
        self.record_path = record_path

        # Initialize capture
        self.capture = create_capture(
            use_simulation=use_simulation,
            center_freq=center_freq,
            sample_rate=sample_rate,
            gain=gain,
        )

        # Initialize detector
        self.detector = load_detector(model_path, config_path, device)

        # Initialize recorder if specified
        self.recorder = None
        if record_path:
            self.recorder = SessionRecorder(
                record_path,
                sample_rate=sample_rate,
                center_freq=center_freq,
            )

        # State
        self._running = False
        self._pending_injection: str | None = None
        self._injection_active = False
        self._rng = np.random.default_rng()

        # Display settings
        self._update_interval = 0.1  # seconds
        self._history: list[dict] = []
        self._history_max = 50

    def _print_header(self) -> None:
        """Print CLI header."""
        print("\n" + "=" * 70)
        print("TorchRF Testbed - Live Detection")
        print("=" * 70)
        print(f"Frequency: {self.center_freq / 1e6:.1f} MHz | "
              f"Sample Rate: {self.sample_rate / 1e6:.1f} MHz | "
              f"Gain: {self.gain} dB")
        print(f"Model: {type(self.detector.model).__name__} | "
              f"Threshold: {self.detector.threshold:.2f}")

        if is_gnuradio_available():
            print("Capture: HackRF (live)")
        else:
            print("Capture: Simulated (GNURadio not available)")

        if self.inject_enabled:
            print("\nKeyboard Controls:")
            print("  [SPACE] Random anomaly  [T] Tone  [C] Chirp  [B] Barrage")
            print("  [P] Pulse  [M] Multi-tone  [F] Freq drift  [A] Amplitude spike")
            print("  [R] Toggle recording  [Q] Quit")

        print("\n" + "-" * 70)
        print(f"{'Time':<12} | {'SNR (dB)':>8} | {'Power (dB)':>10} | "
              f"{'Score':>8} | {'Status':<20}")
        print("-" * 70)

    def _format_status(self, result: dict, injected: str | None = None) -> str:
        """Format detection result for display.

        Args:
            result: Detection result dict.
            injected: Injected anomaly type (if any).

        Returns:
            Formatted status line.
        """
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-4]

        status = "ANOMALY" if result["is_anomaly"] else "NORMAL"
        if injected:
            status += f" [{injected}]"

        # Color codes (ANSI)
        if result["is_anomaly"]:
            color = "\033[91m"  # Red
        else:
            color = "\033[92m"  # Green
        reset = "\033[0m"

        return (f"{timestamp:<12} | {result['snr_db']:>8.1f} | "
                f"{result['power_db']:>10.1f} | {result['score']:>8.2f} | "
                f"{color}{status:<20}{reset}")

    def _check_keyboard(self) -> str | None:
        """Check for keyboard input (non-blocking).

        Returns:
            Key pressed or None.
        """
        if not self.inject_enabled:
            return None

        try:
            # Check if input available
            if select.select([sys.stdin], [], [], 0)[0]:
                return sys.stdin.read(1).lower()
        except:
            pass
        return None

    def _handle_key(self, key: str) -> bool:
        """Handle keyboard input.

        Args:
            key: Key pressed.

        Returns:
            True to continue, False to quit.
        """
        key_mapping = {
            " ": None,  # Random
            "t": "tone",
            "c": "chirp",
            "b": "barrage",
            "p": "pulse",
            "m": "multi_tone",
            "f": "frequency_drift",
            "a": "amplitude_spike",
            "s": "phase_noise",
            "n": "burst_noise",
        }

        if key == "q":
            return False
        elif key == "r":
            if self.recorder:
                print("\n[Recording active]")
            else:
                print("\n[Recording not enabled. Use --record flag]")
        elif key in key_mapping:
            if key == " ":
                self._pending_injection = self._rng.choice(get_anomaly_types())
            else:
                self._pending_injection = key_mapping[key]

        return True

    def _process_sample(self, signal: np.ndarray) -> tuple[dict, str | None]:
        """Process a captured signal.

        Args:
            signal: Complex signal array.

        Returns:
            Tuple of (detection result dict, injected anomaly type or None).
        """
        injected_type = None

        # Apply injection if pending
        if self._pending_injection:
            signal, metadata = inject_anomaly(
                signal,
                anomaly_type=self._pending_injection,
                sample_rate=self.sample_rate,
            )
            injected_type = self._pending_injection
            self._pending_injection = None

        # Run detection
        result = self.detector.detect(signal)

        # Record if enabled
        if self.recorder:
            self.recorder.add_sample(
                signal,
                label=result.is_anomaly,
                anomaly_type=injected_type,
                snr_db=result.snr_db,
                power_db=result.power_db,
                score=result.score,
            )

        return {
            "score": result.score,
            "is_anomaly": result.is_anomaly,
            "snr_db": result.snr_db,
            "power_db": result.power_db,
        }, injected_type

    def run(self) -> None:
        """Run the live detection loop."""
        self._running = True

        # Setup terminal for non-blocking input
        old_settings = None
        if self.inject_enabled:
            try:
                old_settings = termios.tcgetattr(sys.stdin)
                tty.setcbreak(sys.stdin.fileno())
            except:
                pass

        try:
            self._print_header()
            self.capture.start()

            last_update = 0
            while self._running:
                # Check keyboard
                key = self._check_keyboard()
                if key and not self._handle_key(key):
                    break

                # Read samples
                try:
                    signal = self.capture.read_samples(timeout=0.5)
                except TimeoutError:
                    continue

                # Process and display
                result, injected = self._process_sample(signal)
                print(self._format_status(result, injected))

                # Add to history
                self._history.append(result)
                if len(self._history) > self._history_max:
                    self._history.pop(0)

                # Rate limit
                time.sleep(max(0, self._update_interval - (time.time() - last_update)))
                last_update = time.time()

        except KeyboardInterrupt:
            print("\n\nInterrupted by user.")
        finally:
            # Cleanup
            self.capture.stop()
            if self.recorder:
                self.recorder.close()
                print(f"\nRecording saved: {self.recorder.output_path}")
                print(f"  Samples: {self.recorder.sample_count}")

            # Restore terminal
            if old_settings:
                try:
                    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
                except:
                    pass

            print("\nSession ended.")
            self._print_summary()

    def _print_summary(self) -> None:
        """Print session summary."""
        if not self._history:
            return

        scores = [h["score"] for h in self._history]
        snrs = [h["snr_db"] for h in self._history]
        anomalies = sum(1 for h in self._history if h["is_anomaly"])

        print("\n" + "=" * 70)
        print("Session Summary")
        print("=" * 70)
        print(f"Samples processed: {len(self._history)}")
        print(f"Anomalies detected: {anomalies} ({100*anomalies/len(self._history):.1f}%)")
        print(f"Score range: {min(scores):.2f} - {max(scores):.2f}")
        print(f"SNR range: {min(snrs):.1f} - {max(snrs):.1f} dB")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Live RF anomaly detection with HackRF",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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

    # Model settings
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to model config",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for inference",
    )

    # Operation modes
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Use simulated capture (no HackRF needed)",
    )
    parser.add_argument(
        "--inject",
        action="store_true",
        help="Enable keyboard-triggered anomaly injection",
    )
    parser.add_argument(
        "--record",
        type=str,
        default=None,
        help="Path to record session HDF5 file",
    )

    args = parser.parse_args()

    # Check GNURadio availability
    if not args.simulate and not is_gnuradio_available():
        print("Warning: GNURadio not available. Using simulated capture.")
        args.simulate = True

    # Run CLI
    cli = LiveDetectionCLI(
        center_freq=args.freq,
        sample_rate=args.sample_rate,
        gain=args.gain,
        model_path=args.model,
        config_path=args.config,
        device=args.device,
        use_simulation=args.simulate,
        inject_enabled=args.inject,
        record_path=args.record,
    )

    cli.run()


if __name__ == "__main__":
    main()
