# Spectral-CSI

**Smart building occupancy detection using WiFi Channel State Information (CSI).**

---

## What is this?

Spectral-CSI is a proof-of-concept system that analyzes WiFi signals to detect human presence in rooms—even when people are sitting still. Unlike traditional motion sensors that turn off lights when you're working quietly, this system uses signal processing and machine learning to detect stationary occupants.

The goal: Enable smarter building automation that saves energy without frustrating users.

---

## The Problem with Current Systems

Traditional smart lighting uses PIR (Passive Infrared) motion sensors. They have major flaws:

| Problem | Description | Impact |
|---------|-------------|---------|
| False negatives | You sit still to read/work → lights turn off | User frustration, productivity loss |
| Slow timeouts | 15-30 minute delays before turning off | Energy wasted in empty rooms |
| No confidence | Binary on/off, no nuance | Can't make intelligent decisions |

**The idea:** WiFi signals are disrupted by human bodies (water content, breathing, micro-movements). We can detect these disruptions to verify presence—no cameras, no wearables, no privacy invasion.

---

## What it COULD do (theoretically)

| Scenario | Detection Method | Feasibility |
|----------|------------------|-------------|
| Person sitting still | CSI amplitude/phase variations from breathing | Demonstrated in research papers |
| Empty room detection | Signal stability returns to baseline | Demonstrated in research papers |
| Multiple occupants | Distinct signal patterns | More complex, requires good data |
| Through-wall sensing | Signal penetration analysis | Requires specialized hardware setup |

**Important caveat:** Most published research uses controlled lab environments with specialized hardware (Intel 5300 NICs, modified firmware). Real-world performance is likely lower.

---

## What it CANNOT do

Let's be clear about limitations:

- **No real-time accuracy claims yet** - This is a student project, not a validated commercial system
- **Hardware dependent** - Needs WiFi cards that expose CSI data (not all do)
- **Environment specific** - Each room/setup requires calibration
- **No miracle detection** - Can't reliably distinguish between a person and a large pet, or detect people hidden behind metal
- **Limited range** - Works best in small-to-medium rooms (not large halls)

---

## Technical Approach

The planned system architecture:

### Signal Processing Pipeline

1. **Capture:** Extract Channel State Information from WiFi packets
2. **Preprocess:** Denoise using wavelet transforms, remove outliers
3. **Feature Extraction:** Convert to spectrograms, calculate power spectral density
4. **Classification:** Use neural network to classify occupancy state
5. **Decision:** Output probability of occupancy with confidence bounds

### Why This Matters (Academic Context)

This project applies concepts from:
- **Stochastic signal processing** - treating WiFi CSI as random processes
- **Hypothesis testing** - statistically deciding if a room is empty
- **Bayesian inference** - incorporating prior knowledge to prevent false triggers
- **Time series analysis** - detecting patterns like breathing cycles (0.2-0.5 Hz)

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.9+ |
| Signal Processing | NumPy, SciPy (FFT, filtering, wavelets) |
| Machine Learning | PyTorch (planned) |
| Statistics | SciPy.stats (hypothesis testing) |
| Data | Pandas, Matplotlib |
| Hardware | ESP32 CSI Tool / Intel 5300 NIC (optional) |

---

## Project Status

🚧 **Early Development**

- [x] Project planning and documentation
- [x] Dependencies identified
- [ ] CSI data acquisition
- [ ] Preprocessing pipeline
- [ ] Feature extraction
- [ ] Model training
- [ ] Testing and validation

See [INTERIM_REPORT.md](INTERIM_REPORT.md) for detailed progress.

---

## Who is this for?

- Students learning about signal processing and ML
- Researchers exploring WiFi sensing applications
- Anyone interested in non-intrusive occupancy detection
- Offices, malls, and grocery stores wanting automated lighting that turns off when people leave rooms (no WiFi signal detected)

---

## Who is this NOT for?

- Anyone needing a production-ready system right now
- Anyone expecting plug-and-play installation
- Anyone wanting guaranteed accuracy metrics (we're still building it)

---

## Getting Started

### Installation

```bash
git clone https://github.com/yourusername/spectral-csi.git
cd spectral-csi
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows
# source .venv/bin/activate    # Linux/Mac
pip install -r requirements.txt
```

### Next Steps

The codebase is in early stages. Check back for:
- Data collection scripts
- Preprocessing utilities
- Model training notebooks
- Example datasets

---

## Expected Outcomes (Realistic)

Based on published research, systems like this achieve:
- **Empty room detection:** 85-95% accuracy
- **Stationary person detection:** 75-90% accuracy  
- **Response time:** 5-30 seconds (depends on signal window size)

These are research-grade results, not commercial-system guarantees.

---

## Disclaimer

This is an educational project exploring WiFi sensing techniques. It is not intended for deployment in production environments without extensive testing and validation. Users are responsible for ensuring compliance with local regulations regarding wireless signal monitoring.

---

## License

MIT License
