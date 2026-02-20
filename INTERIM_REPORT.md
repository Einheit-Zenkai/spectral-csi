# INTERIM REPORT — Spectral-CSI

**Project Title:** Spectral-CSI: Bayesian Occupancy & Latency Minimization using Wi‑Fi CSI

**Date:** 20 Feb 2026

**Team / Author(s):** [Your Name / Team Name]

**Guide / Supervisor (if applicable):** [Name]

---

## 1. Executive Summary
Spectral-CSI is a privacy-preserving occupancy estimation framework for smart buildings using Wi‑Fi Channel State Information (CSI). The core deliverable is an uncertainty-aware occupancy estimate that can be used to optimize HVAC energy usage and network latency/QoS decisions without cameras.

This interim report documents the finalized problem statement, theoretical basis, system design, dependency stack, and evaluation targets. Implementation work can proceed by adding a signal-processing pipeline (STFT/PSD + denoising) and a Bayesian deep model (Monte Carlo Dropout) over a CNN/ResNet backbone.

---

## 2. Problem Statement
Smart building control systems (HVAC + enterprise Wi‑Fi) benefit from accurate occupancy awareness. Existing camera solutions create privacy concerns and can fail in poor lighting or occlusion.

**Goal:** Estimate occupancy using CSI-derived spectral features and provide predictive uncertainty to reduce false actions during RF interference.

---

## 3. Objectives

### 3.1 Primary Objectives
1. **Non-intrusive occupancy estimation** using CSI, avoiding any visual identification.
2. **Probabilistic output**: predict occupancy mean and uncertainty (variance).
3. **Low inference latency** suitable for near real-time control.

### 3.2 Secondary Objectives
1. Provide a pathway for **arrival-rate modeling** (Poisson process) to anticipate peak loads.
2. Enable a control interface that can throttle/adjust HVAC and bandwidth based on confidence.

---

## 4. Background & Theory (Syllabus Mapping)

### 4.1 Spectral Processing (Stochastic Processes — Unit 3)
CSI streams over short windows can be treated as approximately WSS. Human motion perturbs multipath components, producing time-varying spectral energy. STFT/PSD-based features are used to isolate these signatures.

### 4.2 Occupant Arrivals (Unit 3)
Occupancy accumulation can be described at a coarse scale with a Poisson process:

$$P(N(t)=k) = \frac{(\lambda t)^k e^{-\lambda t}}{k!}$$

This model supports forecasting and proactive policy planning.

### 4.3 Bayesian Deep Learning (Unit 1 & 2)
Deep models can be overconfident under distribution shift (different rooms, AP placement, interference). Monte Carlo Dropout approximates Bayesian inference by sampling multiple stochastic forward passes, yielding a predictive distribution summarized by $(\mu, \sigma^2)$.

---

## 5. Proposed Methodology

### 5.1 Data Flow
1. Capture CSI frames (per antenna/subcarrier).
2. Preprocess (hardware/dataset dependent): sanitize amplitude/phase, remove outliers.
3. Window the stream and compute STFT / spectrogram representations.
4. Denoise: reduce static components (e.g., autocorrelation filtering), normalize.
5. Model inference:
   - CNN/ResNet feature extraction on spectrogram “images”.
   - Dropout enabled at inference.
   - Run $T$ forward passes.
6. Output:
   - Mean occupancy estimate
   - Variance as uncertainty score
7. Control logic:
   - If variance > threshold → flag low confidence and apply conservative policy.

### 5.2 Evaluation Plan
- **Accuracy** (classification or regression depending on target formulation)
- **RMSE** (for regression count estimates)
- **Latency** (windowing + feature extraction + model inference)
- **Calibration** of uncertainty (e.g., reliability plots; optional)

---

## 6. Tools & Dependencies
The project’s Python stack is captured in `requirements.txt`:
- NumPy, SciPy, pandas
- PyTorch, torchvision
- scikit-learn, tqdm
- matplotlib, seaborn

**Datasets (planned):** Widar3.0 and/or StanWiFi.

**Hardware (optional):** ESP32 CSI Tool or Intel 5300 NIC CSI research tool.

---

## 7. System Architecture

```mermaid
flowchart LR
    A[Raw Wi‑Fi CSI Stream] --> B[STFT / Spectrogram]
    B --> C[Signal Denoising]
    C --> D[Bayesian CNN / ResNet Backbone]
    D --> E[Monte Carlo Sampling]
    E --> F[Occupancy Mean]
    E --> G[Uncertainty (Variance)]
    F --> H[Optimization API]
    G --> H
    H --> I[HVAC / Network Bandwidth Control]
```

---

## 8. Current Progress (as of 20 Feb 2026)
- Documentation expanded with:
  - clear abstract and aims
  - detailed tech stack
  - theoretical mapping (stochastic processes + Bayesian inference)
  - architecture diagram and metrics targets
- Dependency list created in `requirements.txt`.

**Implementation status:** Pipeline and training code are not yet present in this repository (documentation-first stage).

---

## 9. Preliminary Results (from Summary Targets)
If you already have prior experimental runs (e.g., in an external notebook), these are the target/reference metrics:

| Metric | Spectral-CSI (Target) | Standard CNN (Target) |
|---|---:|---:|
| Accuracy | 94.8% | 89.2% |
| RMSE | 0.65 | 1.12 |
| Latency | 12ms | 45ms |

---

## 10. Risks & Mitigations
- **Domain shift** (new rooms/AP placements): use uncertainty thresholding, consider domain adaptation.
- **RF interference / multipath complexity**: denoising + robust normalization; rely on uncertainty output.
- **Label noise** in datasets: add smoothing/aggregation over windows; validate labeling pipeline.

---

## 11. Next Steps
1. Add `src/` scaffold and implement CSI preprocessing + STFT feature extractor.
2. Implement Bayesian CNN/ResNet with MC Dropout and training pipeline.
3. Add evaluation scripts for accuracy/RMSE/latency.
4. (Optional) Create a lightweight API endpoint for infrastructure control integration.

---

## 12. References
- Gal, Y. & Ghahramani, Z. “Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning.”
- Widar3.0 / StanWiFi dataset documentation (as applicable).
