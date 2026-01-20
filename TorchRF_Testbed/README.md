# TorchRF Testbed

A standalone CLI-based testbed for live RF anomaly detection using HackRF + GNURadio, with software-based anomaly injection and HDF5 recording.

## Features

- **Live RF Capture**: Real-time I/Q signal capture via HackRF using GNURadio/osmosdr
- **Anomaly Detection**: Integration with CLP_Project's SNR-conditioned VAE model
- **Software Injection**: Keyboard-triggered anomaly injection for testing
- **Dataset Recording**: HDF5 recording compatible with MIT RF dataset format
- **Offline Testing**: Replay recorded datasets with full metrics computation

## Quick Start

### Simulated Mode (No Hardware Required)

```bash
# Live detection with simulated capture
python scripts/live_detect.py --simulate --inject

# Record a test dataset
python scripts/record_session.py --output test.h5 --samples 100 --simulate

# Evaluate on recorded data
python scripts/replay_test.py --input test.h5
```

### With HackRF Hardware

```bash
# Basic live detection at 915 MHz
python scripts/live_detect.py --freq 915e6 --gain 40

# With anomaly injection enabled
python scripts/live_detect.py --freq 915e6 --gain 40 --inject

# Record and detect simultaneously
python scripts/live_detect.py --freq 915e6 --inject --record session.h5
```

## Installation

### System Dependencies

GNURadio and gr-osmosdr must be installed via your system package manager:

```bash
# Ubuntu/Debian
sudo apt install gnuradio gr-osmosdr hackrf

# Arch Linux
sudo pacman -S gnuradio gnuradio-osmosdr hackrf

# Fedora
sudo dnf install gnuradio gr-osmosdr hackrf
```

### Python Dependencies

```bash
cd TorchRF_Testbed
pip install -r requirements.txt
```

Note: The testbed requires the parent CLP_Project model files. Ensure the model checkpoint exists at `../snr_conditioned_vae_hybrid_v1.pt`.

## Usage

### Live Detection (`scripts/live_detect.py`)

Primary interface for real-time RF anomaly detection.

```bash
# Basic usage
python scripts/live_detect.py --freq 915e6 --gain 40

# With anomaly injection (keyboard controls)
python scripts/live_detect.py --freq 915e6 --gain 40 --inject

# Record session while detecting
python scripts/live_detect.py --freq 915e6 --gain 40 --record output.h5

# Use simulated capture for testing
python scripts/live_detect.py --simulate --inject
```

**Keyboard Controls** (when `--inject` enabled):
- `SPACE` - Inject random anomaly
- `T` - Inject tone interference
- `C` - Inject chirp/sweep
- `B` - Inject barrage noise
- `P` - Inject pulsed jamming
- `M` - Inject multi-tone
- `F` - Inject frequency drift
- `A` - Inject amplitude spike
- `Q` - Quit

**CLI Output:**
```
TorchRF Testbed - Live Detection
================================
Frequency: 915.0 MHz | Sample Rate: 2.0 MHz | Gain: 40 dB
Model: SNRConditionedVAE | Threshold: 4.24

Time       | SNR (dB) | Power (dB) | Score  | Status
-----------|----------|------------|--------|--------
10:23:45.1 |    15.2  |      -5.3  |   2.31 | NORMAL
10:23:45.2 |    14.8  |      -5.1  |   2.45 | NORMAL
10:23:45.3 |    12.1  |       2.8  |  18.72 | ANOMALY [tone]
10:23:45.4 |    15.0  |      -5.0  |   2.28 | NORMAL
```

### Dataset Recording (`scripts/record_session.py`)

Record labeled datasets with automatic anomaly injection.

```bash
# Record 1000 samples with 20% anomalies
python scripts/record_session.py --output dataset.h5 --samples 1000 --anomaly-ratio 0.2

# Record with specific anomaly types
python scripts/record_session.py --output dataset.h5 --samples 500 \
    --anomaly-types tone chirp barrage

# Record with higher severity anomalies
python scripts/record_session.py --output dataset.h5 --samples 1000 --severity 2.0
```

### Offline Evaluation (`scripts/replay_test.py`)

Test model performance on recorded datasets.

```bash
# Basic evaluation
python scripts/replay_test.py --input dataset.h5

# With custom model
python scripts/replay_test.py --input dataset.h5 --model path/to/model.pt

# Save ROC/PR curves
python scripts/replay_test.py --input dataset.h5 --save-plots --output-dir results/
```

**Output:**
```
Evaluation Results
============================================================

Overall Metrics:
  AUROC: 0.9342
  AUPRC: 0.8856
  F1 Score: 0.8721
  Precision: 0.8900
  Recall: 0.8550

Per Anomaly Type:
  tone:
    AUROC: 0.9821, F1: 0.9512
  chirp:
    AUROC: 0.8923, F1: 0.8234
```

## Configuration

Edit `config.yaml` to customize default settings:

```yaml
# HackRF settings
capture:
  center_freq: 915000000    # Hz
  sample_rate: 2000000      # Hz
  gain: 40                  # dB

# Model settings
model:
  path: "../snr_conditioned_vae_hybrid_v1.pt"
  config: "../configs/default.yaml"
  device: "cpu"

# Detection settings
detection:
  method: "latent"
  threshold_percentile: 95

# Injection settings
injection:
  default_severity: 1.0
  jsr_range: [0, 20]
```

## HDF5 Dataset Format

Recorded datasets follow a schema compatible with MIT RF data:

```
/signals        - complex64 [N, 1024]    # I/Q samples
/labels         - bool [N]               # True = anomaly
/anomaly_types  - string [N]             # Type of anomaly
/snr            - float32 [N]            # Estimated SNR (dB)
/power          - float32 [N]            # Estimated power (dB)
/timestamps     - float64 [N]            # Unix timestamps
/scores         - float32 [N]            # Detection scores
/metadata       - group                  # Capture settings
```

## Supported Anomaly Types

| Type | Description |
|------|-------------|
| `tone` | Single frequency interferer |
| `multi_tone` | Multiple frequency interferers |
| `chirp` | Frequency sweep jamming |
| `barrage` | Wideband noise jamming |
| `pulse` | Pulsed interference |
| `interference` | Narrowband interference (CLP_Project) |
| `frequency_drift` | Frequency drift (CLP_Project) |
| `amplitude_spike` | Amplitude spike/burst (CLP_Project) |
| `phase_noise` | Random phase noise (CLP_Project) |
| `burst_noise` | Impulsive burst noise (CLP_Project) |

## Project Structure

```
TorchRF_Testbed/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── config.yaml                  # Default configuration
├── src/
│   ├── __init__.py
│   ├── capture.py               # HackRF capture via GNURadio
│   ├── injection.py             # Software anomaly injection
│   ├── detector.py              # Model loading + inference
│   ├── recorder.py              # HDF5 recording
│   └── utils.py                 # Signal processing utilities
├── scripts/
│   ├── live_detect.py           # Main CLI: live capture + detection
│   ├── record_session.py        # Record labeled dataset
│   └── replay_test.py           # Test on recorded HDF5 files
└── flowgraphs/
    └── hackrf_source.grc        # GNURadio Companion flowgraph (optional)
```

## Troubleshooting

### GNURadio Not Found

If you get `GNURadio not available` warnings:

1. Ensure gnuradio and gr-osmosdr are installed via system package manager
2. Check Python can import gnuradio: `python -c "from gnuradio import gr; print('OK')"`
3. You may need to add GNURadio's Python path to your environment

### HackRF Not Detected

1. Check USB connection: `hackrf_info`
2. Ensure you have permissions: add user to `plugdev` group
3. Try unplugging and replugging the device

### Model Not Found

The testbed expects the CLP_Project model at `../snr_conditioned_vae_hybrid_v1.pt`. Either:
- Place the model in the expected location
- Use `--model` flag to specify an alternate path

## License

Part of the CLP_Project research codebase.
