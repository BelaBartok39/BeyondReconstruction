# Public RF Datasets for Anomaly Detection

**Last Updated:** 2026-01-29

This document catalogs public RF datasets suitable for testing our anomaly detection model.

---

## Priority Datasets for Our Project

### 1. WASD: Wireless Anomaly Signal Dataset ⭐ (Highly Recommended)

**Why it's perfect:** Specifically designed for anomaly detection in wireless signals with labeled anomalies (tone, chirp, pulse).

| Attribute | Value |
|-----------|-------|
| Source | [IEEE DataPort](https://ieee-dataport.org/open-access/wasd-wireless-anomaly-signal-dataset) |
| Format | Raw I/Q (.bin), Spectrograms (.npy), Labels (.csv) |
| Size | ~140,000 anomaly instances |
| Bands | 19 LTE and 5G bands |
| Anomaly Types | Tone, Chirp, Pulse |
| License | Open Access (IEEE account required) |

**Key Features:**
- Real-world signals from urban environments
- Simulated anomalous signals with labeled bounding boxes
- Interference-to-Signal Ratio (ISR) annotations
- Perfect for evaluating our hybrid detection approach

**Download:**
```bash
# Requires IEEE DataPort account (free)
# Download from: https://ieee-dataport.org/open-access/wasd-wireless-anomaly-signal-dataset
```

---

### 2. RadioML 2018.01A (DeepSig)

**Why it's useful:** Standard benchmark for RF ML, can test OOD modulation detection.

| Attribute | Value |
|-----------|-------|
| Source | [Kaggle](https://www.kaggle.com/datasets/pinxau1000/radioml2018) / [DeepSig](https://www.deepsig.ai/datasets/) |
| Format | HDF5 (complex I/Q) |
| Size | 21.45 GB |
| Samples | 2 million, each 1024 samples |
| Modulations | 24 types (digital + analog) |
| SNR Range | 26 levels |
| License | CC BY-NC-SA 4.0 |

**Key Features:**
- 24 modulation types (BPSK, QPSK, 8PSK, 16QAM, etc.)
- Synthetic + over-the-air captured signals
- Well-documented and widely cited

**Use Case:** Train on subset of modulations, test detection on "unseen" modulations as anomalies.

**Download:**
```bash
# Option 1: Kaggle CLI
kaggle datasets download -d pinxau1000/radioml2018

# Option 2: Direct from DeepSig
wget https://www.deepsig.ai/datasets/2018.01.OSC.0001_1024x2M.h5.tar.gz
```

---

### 3. MIT RF Challenge Dataset

**Why it's useful:** Real 5G signals with interference scenarios.

| Attribute | Value |
|-----------|-------|
| Source | [MIT RF Challenge](https://rfchallenge.mit.edu/datasets/) |
| Format | Various (see site) |
| Content | CommSignal5G1 (5G waveform, 61.44 MHz BW) |
| Signals | 2.4 GHz ISM band + 5G |
| Focus | Blind signal separation |

**Key Features:**
- Real over-the-air recordings
- 5G-compliant waveforms
- Multiple difficulty levels (1-9)
- Academic research benchmark

**Download:**
```bash
# Available at https://rfchallenge.mit.edu/datasets/
# Files partitioned by difficulty level
```

---

### 4. RF Jamming Dataset (Kaggle)

**Why it's useful:** Labeled jamming attacks for detection.

| Attribute | Value |
|-----------|-------|
| Source | [Kaggle](https://www.kaggle.com/datasets/daniaherzalla/radio-frequency-jamming) |
| Format | Various |
| Focus | Jamming detection |
| License | Open |

**Use Case:** Test detection of intentional interference/jamming.

---

### 5. DroneDetect Dataset

**Why it's useful:** Real UAS RF signals with interference variations.

| Attribute | Value |
|-----------|-------|
| Source | [IEEE DataPort](https://ieee-dataport.org/open-access/dronedetect-dataset-radio-frequency-dataset-unmanned-aerial-system-uas-signals-machine) |
| Format | I/Q recordings |
| Drones | 7 models (DJI Mavic, Phantom, Inspire, Parrot) |
| Conditions | Bluetooth interference, WiFi, combined, clean |
| Equipment | Nuand BladeRF SDR + GNURadio |

**Use Case:** Test generalization to completely different signal types (drone control signals vs our WiFi focus).

---

### 6. RF Jamming Dataset for Vehicular Networks

**Why it's useful:** Multiple jamming scenarios with ground truth.

| Attribute | Value |
|-----------|-------|
| Source | [IEEE DataPort](https://ieee-dataport.org/documents/rf-jamming-dataset-vehicular-wireless-networks) |
| Scenarios | No attack, Reactive jammer, Constant jammer |
| Environment | Vehicular Ad-hoc Networks (VANETs) |

**Use Case:** Distinguish intentional jamming from unintentional interference.

---

## DARPA RFML Resources (Code + Data Access)

While DARPA doesn't publish datasets directly, these repositories provide data access:

| Repository | URL | Features |
|------------|-----|----------|
| brysef/rfml | [GitHub](https://github.com/brysef/rfml) | PyTorch RFML library with dataset wrappers |
| neu-spiral/RFMLS-NEU | [GitHub](https://github.com/neu-spiral/RFMLS-NEU) | WiFi/ADS-B fingerprinting code |
| IntelLabs/RFML-Framework | [GitHub](https://github.com/IntelLabs/RFML-Framework) | SigMF format, dataset generators |

---

## Download Priority

For ICASSP paper validation, prioritize in this order:

1. **WASD** - Direct anomaly detection benchmark (most relevant)
2. **RadioML 2018** - Standard ML benchmark (credibility)
3. **RF Jamming (Kaggle)** - Quick to download, jamming focus
4. **MIT RF Challenge** - 5G validation
5. **DroneDetect** - Generalization test

---

## Dataset Comparison

| Dataset | I/Q Format | Anomaly Labels | Size | Difficulty |
|---------|-----------|----------------|------|------------|
| WASD | Yes (.bin) | Yes (CSV) | Large | Medium |
| RadioML | Yes (HDF5) | No (modulation only) | 21 GB | Easy |
| MIT RF | Varies | Partial | Medium | Medium |
| RF Jamming | Varies | Yes | Small | Easy |
| DroneDetect | Yes | Yes (interference type) | Medium | Medium |
| POWDER (ours) | Yes | Yes | Medium | Done ✓ |
| HackRF (ours) | Yes | Yes | Small | Done ✓ |

---

## Notes

- Most IEEE DataPort datasets require a free IEEE account
- RadioML is CC BY-NC-SA (non-commercial only)
- Check licenses before including in public repo
- Large datasets should go on HuggingFace, not GitHub
