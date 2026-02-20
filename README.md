# Spectral-CSI: Bayesian Occupancy & Latency Minimization

![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)
![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange.svg)
![Status: IEEE Format](https://img.shields.io/badge/Status-IEEE%20Format-green)

Spectral-CSI is a privacy-preserving indoor sensing framework that estimates **occupancy (crowd density)** using **Wi‑Fi Channel State Information (CSI)** instead of cameras. The output is not only an occupancy estimate, but also **model uncertainty** (confidence) so building/network controllers can take safe actions when RF conditions are noisy.

This repository currently contains the project documentation + reports. Code modules can be added following the suggested layout in the “Repository Layout” section.

---

## 📋 Abstract
Smart buildings often need occupancy awareness to optimize **HVAC energy** and **network QoS** (e.g., bandwidth throttling, AP steering). Camera-based solutions raise privacy concerns and require line-of-sight.

Spectral-CSI leverages CSI as a fine-grained RF measurement. Human motion introduces stochastic perturbations (micro-Doppler / time-varying multipath) that appear as structured energy changes in the time–frequency domain.

We combine:
- **Spectral feature extraction** (STFT/PSD + denoising)
- A **Bayesian Deep Learning** model (Monte Carlo Dropout over a CNN/ResNet backbone)

to produce:
- Occupancy estimate (mean)
- Uncertainty score (variance) for safe control decisions

---

## 🎯 What This Project Aims to Achieve

**Primary objective**: Estimate indoor occupancy accurately **without identifying individuals**.

**Control objective**: Reduce infrastructure latency/overreaction by driving control logic with a *probabilistic* estimate.

**Operational goals**:
1. **Privacy preservation**: no images, no microphones, no biometric collection.
2. **Real-time inference**: support low-latency streaming windows (e.g., 0.5–2.0 s windows).
3. **Uncertainty-aware automation**: when uncertainty is high, the system flags “low confidence” and avoids aggressive HVAC/network changes.

---

## 🧱 Tech Stack (Detailed)

### Core
- **Python**: 3.9+ (recommended 3.10/3.11 if supported by your PyTorch build)
- **PyTorch**: neural networks, training loops, GPU acceleration
- **Torchvision**: ResNet backbones if using image-like spectrogram inputs

### Signal Processing
- **NumPy**: numerical arrays, windowing
- **SciPy**: STFT/FFT, filters (Butterworth), statistics

### ML Utilities
- **scikit-learn**: metrics (RMSE), preprocessing, baselines
- **tqdm**: progress bars

### Data Handling
- **pandas**: time-series alignment, aggregation, CSV/Parquet I/O

### Visualization
- **matplotlib / seaborn**: spectral heatmaps, uncertainty plots

### Optional Hardware / Data Capture
- **ESP32 CSI Tool** (community firmware/tooling) or
- **Intel 5300 NIC CSI Tool** (research hardware)

### Datasets (Common Choices)
- **Widar3.0** (Wi‑Fi sensing dataset)
- **StanWiFi** (Wi‑Fi sensing dataset)

---

## 🧠 Theoretical Framework (Syllabus Mapping)

This project maps to **Stochastic Processes & Probability Theory** concepts.

### 1) Spectral Feature Extraction (Unit 3)
Over short windows, raw CSI sequences can be treated as approximately **wide-sense stationary (WSS)**. We extract spectral signatures using STFT/PSD to highlight time-varying Doppler components caused by human motion.

Typical operations:
- Windowing + STFT
- PSD estimation
- Autocorrelation-based filtering to reduce static multipath components

### 2) Stochastic Arrival Modeling (Unit 3)
Occupant arrivals can be modeled (at a coarse scale) using a Poisson process:

$$P(N(t)=k) = \frac{(\lambda t)^k e^{-\lambda t}}{k!}$$

This supports forecasting peak occupancy windows, which can be used for proactive network routing / HVAC scheduling.

### 3) Bayesian Inference & Uncertainty (Unit 1 & 2)
Deterministic neural networks often produce overconfident predictions. We use **Monte Carlo Dropout** as an approximation to Bayesian posterior inference.

Instead of outputting only a single count, the model estimates a predictive distribution, commonly summarized by $(\mu, \sigma^2)$.

Decision rule example:
- Low variance: allow automatic HVAC/bandwidth adaptation
- High variance: mark *Low Confidence* and fall back to conservative policies

---

## 🏗 System Architecture

```mermaid
flowchart LR
		A[Raw Wi‑Fi CSI Stream] --> B[STFT / Spectrogram]
		B --> C[Signal Denoising]
		C --> D[Bayesian CNN / ResNet Backbone]
		D --> E[Monte Carlo Sampling (Dropout at Inference)]
		E --> F[Occupancy Mean]
		E --> G[Uncertainty (Variance)]
		F --> H[Optimization API]
		G --> H
		H --> I[HVAC / Network Bandwidth Control]
```

---

## 📊 Key Results (From Project Summary)

| Metric | Spectral-CSI (Ours) | Standard CNN | Improvement |
|---|---:|---:|---:|
| Accuracy | 94.8% | 89.2% | +5.6% |
| RMSE | 0.65 | 1.12 | -42% |
| Latency | 12ms | 45ms | 3.7× faster |

Notes:
- Results depend on dataset, window size, sampling rate, and hardware.
- Latency should be reported with the same compute target (CPU/GPU) and batch size.

---

## ✅ Installation

### 1) Create a virtual environment (recommended)

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

### 2) Install dependencies

If `requirements.txt` exists:

```bash
pip install -r requirements.txt
```

If PyTorch install fails, install PyTorch from the official selector for your CUDA/CPU setup, then reinstall the remaining dependencies.

---

## 🗂️ Repository Layout (Suggested)

Because this repo is currently documentation-first, here is a clean layout to add when you start implementing:

```
spectral-csi/
	data/                   # ignored: raw and processed datasets
	notebooks/              # exploration, plots
	src/
		spectral_csi/
			dsp/                # STFT/PSD, filters
			models/             # Bayesian CNN/ResNet + MC Dropout
			training/           # training loops, loss, metrics
			inference/          # streaming inference + uncertainty
			api/                # control API (optional)
	reports/
		INTERIM_REPORT.md
	requirements.txt
	README.md
```

---

## 🔬 Method Overview (Implementation Notes)

1. **Acquire CSI**: collect complex CSI matrices per subcarrier and antenna.
2. **Preprocess**: calibration, amplitude/phase sanitization (dataset/hardware dependent).
3. **Time–frequency transform**: STFT to create a spectrogram per window.
4. **Denoise**: remove static components, smooth spectral bins, normalize.
5. **Model**: CNN/ResNet processes spectrogram “images”.
6. **Bayesian inference**: enable dropout at inference; run $T$ stochastic passes.
7. **Outputs**: predictive mean occupancy and variance.

---

## 🧾 Citation

If you use this project structure or write-up, cite:

`[Your Name], "Spectral Occupancy Estimation: Minimizing Infrastructure Latency using Bayesian Deep Learning," 2024.`

---

## 📜 License

MIT License (see `LICENSE` if added).
