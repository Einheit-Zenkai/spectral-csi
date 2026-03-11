// ==========================================================================
// Spectral-CSI — IEEE Conference Paper Template (Typst)
// ==========================================================================
// Compile: typst compile paper.typ
// Watch:   typst watch paper.typ
// ==========================================================================

#set document(
  title: "Spectral-CSI: Device-Free Occupancy Detection for Granular Energy Optimization using Bayesian Deep Learning on Wi-Fi Channel State Information",
  author: ("Author One", "Author Two"),
  date: auto,
)

// --- IEEE-style page layout ---
#set page(
  paper: "us-letter",
  margin: (top: 0.75in, bottom: 1in, left: 0.625in, right: 0.625in),
  columns: 2,
  numbering: "1",
)

// --- Typography ---
#set text(font: "Times New Roman", size: 10pt)
#set par(justify: true, leading: 0.55em, first-line-indent: 1em)
#set heading(numbering: "I.A.1.")

// Heading styles
#show heading.where(level: 1): it => {
  set text(size: 10pt, weight: "bold")
  set align(center)
  upper(it)
}
#show heading.where(level: 2): it => {
  set text(size: 10pt, weight: "bold", style: "italic")
  it
}

// --- Title block (spans both columns) ---
#place(
  top + center,
  scope: "parent",
  float: true,
  {
    set align(center)
    set par(first-line-indent: 0pt)

    text(size: 24pt, weight: "bold")[
      Spectral-CSI: Device-Free Occupancy Detection\ for Granular Energy Optimization using\ Bayesian Deep Learning on Wi-Fi CSI
    ]
    v(0.5em)
    text(size: 12pt)[
      Author One#super[1], Author Two#super[1] \
      #text(size: 10pt, style: "italic")[
        #super[1]Department of Electronics / Computer Science, University Name, City, Country \
        \{author1, author2\}\@university.edu
      ]
    ]
    v(1em)
    line(length: 100%, stroke: 0.5pt)
    v(0.5em)
  }
)


// ══════════════════════════════════════════════════════════════════════════
//  ABSTRACT
// ══════════════════════════════════════════════════════════════════════════

#heading(level: 1, numbering: none)[Abstract]

#par(first-line-indent: 0pt)[
  _Commercial buildings waste up to 30% of lighting and HVAC energy due to inaccurate occupancy sensing. Passive Infrared (PIR) motion sensors—the industry standard—fail to detect stationary occupants, leading to premature shutoffs ("phantom load") or overly conservative timeouts. We present *Spectral-CSI*, a device-free, privacy-preserving occupancy detection framework that exploits Wi-Fi Channel State Information (CSI) and Bayesian deep learning. Our approach extracts spectral features—including respiration-band (0.2–0.5 Hz) power spectral density—from commodity Wi-Fi signals, and feeds them to a ResNet-18 backbone with Monte Carlo Dropout for probabilistic inference. A statistical hypothesis-testing module provides a 99.9% confidence empty-room decision, enabling "zero-false-negative" automation. On synthetic and benchmark CSI datasets, Spectral-CSI achieves [XX]% static-user detection accuracy with < 1% false-negative rate, projecting [XX]% energy savings over PIR baselines. The system requires no additional hardware, cameras, or wearable devices._
]

#par(first-line-indent: 0pt)[
  *Keywords*—_WiFi CSI, occupancy detection, Bayesian deep learning, Monte Carlo Dropout, spectral analysis, smart buildings, energy optimization_
]

// ══════════════════════════════════════════════════════════════════════════
= Introduction

// WHY — the problem
Modern smart buildings deploy occupancy-driven control for lighting and HVAC systems to reduce energy consumption. The predominant sensing modality—Passive Infrared (PIR) motion detection—suffers from a fundamental limitation: it cannot detect *stationary* occupants. A person reading, typing, or sitting in a meeting generates insufficient motion to trigger a PIR sensor, causing the system to conclude the room is empty and shut off lighting or climate control. This "static user problem" forces building managers to set conservative timeout periods (15–30 minutes), substantially reducing potential energy savings.

// WHAT — alternatives and their gaps
Camera-based solutions achieve high accuracy but raise privacy concerns. Wearable or badge-based systems require user compliance and infrastructure investment. WiFi-based sensing has emerged as an attractive alternative: it leverages *existing* wireless infrastructure, preserves privacy (no images or biometrics), and can detect the subtle signal perturbations caused by human respiration and body presence.

// HOW — our contribution
In this paper, we present *Spectral-CSI*, a complete framework that:
+ Extracts spectral features from WiFi Channel State Information (CSI), isolating the 0.2–0.5 Hz respiration band using STFT and Welch PSD estimation.
+ Employs a Bayesian ResNet-18 with MC Dropout to produce occupancy probability with calibrated uncertainty.
+ Implements a statistical hypothesis-testing module with 99.9% confidence threshold for safe empty-room decisions.
+ Demonstrates projected energy savings of [XX]% over conventional PIR systems.

// ══════════════════════════════════════════════════════════════════════════
= Related Work

== WiFi-Based Activity Recognition
// TODO: Cite Widar3.0 (Zheng et al., MobiSys 2019), WiGest, etc.
// Discuss amplitude/phase-based approaches and their limitations.

== CSI-Based Occupancy Detection
// TODO: Cite Wang et al. (MobiCom 2015), FreeCount, etc.
// Distinguish activity recognition from static presence detection.

== Bayesian Deep Learning for Sensing
// TODO: Cite Gal & Ghahramani (ICML 2016), MC Dropout theory.
// Discuss why uncertainty quantification matters for safety-critical decisions.

== Smart Building Energy Optimization
// TODO: Cite ASHRAE standards, phantom load statistics, PIR limitations.

// ══════════════════════════════════════════════════════════════════════════
= System Model

== WiFi Channel State Information
// Describe H(f,t) = H_static(f) + H_dynamic(f,t) + N(f,t)
// 52 OFDM subcarriers, complex-valued, amplitude extraction
The received CSI at subcarrier $f$ and time $t$ is modelled as:

$ H(f, t) = H_"static" (f) + H_"human" (f, t) + N(f, t) $

where $H_"static"$ represents the time-invariant multipath channel, $H_"human"$ captures perturbations from human presence (breathing, micro-movement), and $N$ is additive noise.

== Signal Model for Human Presence
// Respiration: 0.2–0.5 Hz sinusoidal modulation
// Heartbeat: ~1.2 Hz (weaker)
// Micro-motion: slow stochastic drift
Human respiration modulates the CSI amplitude at $f_"resp" in [0.2, 0.5]$ Hz:

$ H_"human" (f, t) = A_"resp" sin(2 pi f_"resp" t + phi_f) + epsilon(t) $

where $A_"resp"$ is the modulation depth, $phi_f$ is a subcarrier-dependent phase, and $epsilon(t)$ captures micro-motion.

// ══════════════════════════════════════════════════════════════════════════
= Methodology

== Preprocessing Pipeline

=== Wavelet Denoising
// DWT with db4 wavelet, universal threshold σ√(2 ln N), soft thresholding
We apply the Discrete Wavelet Transform (DWT) with a Daubechies-4 wavelet. The universal threshold is computed as $sigma sqrt(2 ln N)$, where $sigma$ is estimated from the finest detail coefficients using the MAD estimator: $hat(sigma) = "median"(|c_d|) / 0.6745$.

=== Outlier Removal
// Chebyshev's inequality: P(|X - μ| ≥ kσ) ≤ 1/k²
Using Chebyshev's inequality with $k = 3$, we clip values beyond $mu plus.minus 3 sigma$, providing distribution-free outlier bounds with at most 11.1% false removal probability.

== Spectral Feature Extraction

=== STFT Spectrogram
// Short-Time Fourier Transform → time-frequency representation
We compute the STFT of each subcarrier's time series using a Hann window of length $L = 256$ samples with 50% overlap:

$ S(tau, omega) = sum_(n=0)^(L-1) x[n + tau] w[n] e^(-j omega n) $

=== Power Spectral Density
// Welch's method → smoothed PSD estimate
Welch's method provides a consistent PSD estimate by averaging periodograms of overlapping segments.

=== Respiration Band Energy
// Integrate PSD over [0.2, 0.5] Hz → primary detection feature
The respiration band energy is computed as:

$ E_"resp" = integral_(0.2)^(0.5) hat(S)(f) d f $

This scalar feature shows significant separation between occupied and empty rooms.

== Statistical Hypothesis Testing

We formulate occupancy detection as:
- $H_0$: Room is empty — signal variance $approx$ noise floor
- $H_1$: Room is occupied — signal contains human signature

Lights are turned OFF *only* if:

$ P("Empty" | "observations") > 99.9% $

// Describe variance Z-test and respiration energy Z-test
// Combined conservative decision (minimum p-value)

== Bayesian Deep Learning

=== ResNet-18 Backbone
// Modified for single-channel spectrogram input
We adapt a pretrained ResNet-18 by replacing the first convolutional layer (3 → 1 channel) and the final classifier (1000 → 2 classes).

=== Monte Carlo Dropout
// T stochastic forward passes → predictive distribution
At inference, we perform $T = 30$ stochastic forward passes with dropout enabled:

$ hat(p)_"occ" = 1/T sum_(t=1)^(T) p_t ("occupied" | x) $

$ hat(sigma)^2 = 1/T sum_(t=1)^(T) (p_t - hat(p)_"occ")^2 $

=== Fail-Safe Decision Logic
// High uncertainty → default to "occupied" (never turn off lights unsafely)
If $hat(sigma)^2 > sigma_"limit"$, the system defaults to "occupied" regardless of $hat(p)$, implementing a fail-safe mechanism.

// ══════════════════════════════════════════════════════════════════════════
= Experimental Setup

== Dataset
// Describe Widar3.0 / synthetic data generation
// Training/validation/test split

== Implementation Details
// Python, PyTorch, SciPy
// Hardware: ESP32 CSI Tool (optional) / Intel 5300
// Training: AdamW, cosine annealing, 30 epochs

== Evaluation Metrics
// Static user accuracy, empty room accuracy, FNR, response latency, energy savings

// ══════════════════════════════════════════════════════════════════════════
= Results

== Spectral Feature Analysis
// Show PSD plots: occupied vs. empty → clear 0.2–0.5 Hz peak
// TODO: Add figure

== Hypothesis Test Performance
// Confusion matrix, P(empty) distributions for both classes
// TODO: Add table

== Bayesian Model Performance
// Accuracy, FNR, uncertainty calibration
// TODO: Add table with comparison to standard CNN

== Energy Savings Projection
// kWh comparison: PIR vs. Spectral-CSI
// TODO: Add table from interim report

// ══════════════════════════════════════════════════════════════════════════
= Discussion

// Strengths: zero-cost deployment, privacy, uncertainty quantification
// Limitations: synthetic data, single-room testing, ESP32 timing constraints
// Future work: real-world deployment, multi-room, transfer learning

// ══════════════════════════════════════════════════════════════════════════
= Conclusion

We presented Spectral-CSI, a WiFi CSI-based occupancy detection system that combines spectral signal processing with Bayesian deep learning for safe, uncertainty-aware decisions. The system addresses the static-user problem that plagues PIR sensors, enabling aggressive energy savings without false shutoffs. Our statistical hypothesis testing framework with 99.9% confidence thresholds and MC Dropout uncertainty quantification provides the safety guarantees needed for real-world building automation.

// Future work includes live deployment with ESP32 CSI hardware, multi-room generalisation, and integration with building management systems.

// ══════════════════════════════════════════════════════════════════════════
// REFERENCES — IEEE style
// ══════════════════════════════════════════════════════════════════════════

// NOTE: Typst does not yet have built-in IEEE bibliography support.
// For now, manually formatted. Consider using Hayagriva (.yml) for bib management.

#heading(level: 1, numbering: none)[References]

#set par(first-line-indent: 0pt)
#set text(size: 8pt)

+ Y. Gal and Z. Ghahramani, "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep Learning," in _Proc. ICML_, 2016.
+ Y. Zheng _et al._, "Zero-Effort Cross-Domain Gesture Recognition with Wi-Fi," in _Proc. ACM MobiSys_, 2019.
+ W. Wang, A. X. Liu, M. Shahzad, K. Ling, and S. Lu, "Understanding and Modeling of WiFi Signal Based Human Activity Recognition," in _Proc. ACM MobiCom_, 2015.
+ Widar 3.0 Dataset, IEEE DataPort. [Online]. Available: https://ieee-dataport.org/open-access/widar-30-wifi-based-activity-recognition-dataset
+ ESP32 CSI Tool, Espressif Systems. [Online]. Available: https://github.com/espressif/esp-csi
+ K. He, X. Zhang, S. Ren, and J. Sun, "Deep Residual Learning for Image Recognition," in _Proc. IEEE CVPR_, 2016.
+ ASHRAE Standard 90.1-2019, "Energy Standard for Buildings Except Low-Rise Residential Buildings."
