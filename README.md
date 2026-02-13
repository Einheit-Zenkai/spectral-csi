# Spectral-CSI: Privacy-Preserving Smart Building Occupancy Estimation

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A research-grade Bayesian Deep Learning framework for **non-intrusive indoor occupancy estimation** using Wi-Fi Channel State Information (CSI) instead of cameras. This system enables **privacy-preserving smart building** applications that reduce energy consumption and network latency while protecting occupant privacy.

## 🎯 Overview

Spectral-CSI leverages Wi-Fi signal propagation characteristics to estimate indoor occupancy without visual surveillance. The framework combines:

- **Advanced Signal Processing**: CSI preprocessing with amplitude/phase extraction, filtering, and Power Spectral Density (PSD) analysis
- **Bayesian Deep Learning**: Neural networks with uncertainty quantification (both epistemic and aleatoric)
- **Uncertainty-Aware Optimization**: Training strategies that leverage uncertainty estimates for improved performance
- **Privacy-First Design**: No cameras or personal data collection required

### Key Features

✅ **Privacy-Preserving**: Uses Wi-Fi CSI instead of cameras  
✅ **Uncertainty Quantification**: Provides confidence estimates with predictions  
✅ **Modular Architecture**: Clean, extensible codebase for research  
✅ **Academic-Grade**: Comprehensive signal processing and Bayesian inference  
✅ **Easy to Use**: Simple APIs and example scripts  

## 🏗️ Architecture

```
spectral-csi/
├── src/
│   └── spectral_csi/
│       ├── models/              # Bayesian neural network models
│       ├── preprocessing/        # Signal processing for CSI data
│       ├── optimization/         # Uncertainty-aware optimization
│       └── utils/                # Utilities and data loaders
├── configs/                      # Configuration files
├── notebooks/                    # Jupyter notebooks for demos
├── tests/                        # Unit tests
└── data/                         # Data directory (user-provided)
```

## 📦 Installation

### Requirements

- Python 3.10 or higher
- PyTorch 2.0+
- NumPy, SciPy

### Install from source

```bash
# Clone the repository
git clone https://github.com/Einheit-Zenkai/spectral-csi.git
cd spectral-csi

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Quick Install

```bash
pip install torch numpy scipy tqdm pyyaml
pip install -e .
```

## 🚀 Quick Start

### Training a Model

```python
from spectral_csi.models import BayesianOccupancyEstimator
from spectral_csi.preprocessing import CSIPreprocessor
from spectral_csi.utils.data_utils import generate_synthetic_csi_data

# Generate synthetic CSI data for demonstration
csi_data, occupancy = generate_synthetic_csi_data(
    n_samples=1000,
    n_subcarriers=30,
    sequence_length=100
)

# Preprocess CSI data
preprocessor = CSIPreprocessor(n_subcarriers=30)
processed_data = preprocessor.preprocess(csi_data[0])

# Initialize Bayesian model
model = BayesianOccupancyEstimator(
    input_dim=480,
    hidden_dims=[256, 128, 64],
    dropout_rate=0.15
)

# Train the model (see full example in src/train.py)
```

### Making Predictions with Uncertainty

```python
import torch

# Prepare input data
x = torch.FloatTensor(processed_features)

# Get predictions with uncertainty quantification
mean, aleatoric_unc, epistemic_unc = model.predict_with_uncertainty(x)

print(f"Predicted occupancy: {mean.item():.2f}")
print(f"Aleatoric (data) uncertainty: {aleatoric_unc.item():.4f}")
print(f"Epistemic (model) uncertainty: {epistemic_unc.item():.4f}")
```

### Command-Line Training

```bash
# Train with default settings
python src/train.py --gpu --epochs 50

# Train with custom configuration
python src/train.py \
    --n-samples 2000 \
    --hidden-dims 512 256 128 \
    --dropout-rate 0.2 \
    --batch-size 64 \
    --learning-rate 0.001 \
    --gpu
```

### Inference

```bash
# Run inference with trained model
python src/inference.py \
    --checkpoint checkpoints/best_model.pt \
    --n-samples 100 \
    --gpu \
    --save-results
```

## 📊 Model Components

### 1. Signal Preprocessing (`preprocessing/`)

The `CSIPreprocessor` handles raw CSI data processing:

- **Amplitude & Phase Extraction**: Decomposes complex CSI signals
- **Bandpass Filtering**: Removes noise with Butterworth filters
- **Power Spectral Density**: Computes PSD using Welch's method
- **Statistical Features**: Extracts mean, variance, skewness, kurtosis, etc.

```python
from spectral_csi.preprocessing import CSIPreprocessor

preprocessor = CSIPreprocessor(
    sampling_rate=1000.0,
    n_subcarriers=30,
    lowpass_cutoff=50.0,
    highpass_cutoff=0.5
)

result = preprocessor.preprocess(csi_complex_data)
# Returns: amplitude, phase, PSD, and statistical features
```

### 2. Bayesian Models (`models/`)

Two model architectures with uncertainty quantification:

#### `BayesianOccupancyEstimator`
- Fully connected Bayesian neural network
- Monte Carlo Dropout for uncertainty estimation
- Heteroscedastic loss for aleatoric uncertainty

#### `ConvBayesianOccupancyEstimator`
- 1D convolutional architecture for temporal patterns
- Processes CSI sequences directly
- Combined with Bayesian inference

Both models provide:
- **Mean prediction**: Expected occupancy count
- **Aleatoric uncertainty**: Data/observation noise
- **Epistemic uncertainty**: Model uncertainty

### 3. Optimization (`optimization/`)

Advanced training strategies:

- **`UncertaintyAwareOptimizer`**: Weights loss by uncertainty
- **`BayesianOptimizer`**: Hyperparameter tuning with Gaussian Processes
- **`ActiveLearner`**: Selects informative samples for efficient training
- **`UncertaintyWeightedLoss`**: Custom loss functions

## 🧪 Research Applications

### Smart Building Energy Management

```python
# Estimate occupancy to control HVAC systems
predictions, _, epistemic_unc = model.predict_with_uncertainty(csi_stream)

if epistemic_unc < threshold:
    adjust_hvac(predicted_occupancy=predictions.item())
else:
    use_conservative_settings()  # High uncertainty
```

### Privacy-Preserving Surveillance

Replace camera-based systems with CSI-based occupancy detection to:
- Protect occupant privacy
- Reduce data storage requirements
- Enable smart building features without visual surveillance

### Network Resource Allocation

Optimize network resources based on real-time occupancy:
```python
occupancy_estimate = model.predict_with_uncertainty(current_csi)[0]
allocate_bandwidth(num_occupants=occupancy_estimate)
```

## 📈 Performance Metrics

The framework tracks multiple metrics:

- **Regression Metrics**: MSE, RMSE, MAE, R²
- **Uncertainty Metrics**: Mean epistemic/aleatoric uncertainty
- **Calibration**: Prediction intervals and coverage

Example output:
```
Validation Results:
  RMSE: 0.8542
  MAE: 0.6123
  R²: 0.9234
  Mean Epistemic Uncertainty: 0.0451
  Mean Aleatoric Uncertainty: 0.1203
```

## 🔬 Technical Details

### Signal Processing Pipeline

1. **Complex CSI Input**: Shape `(n_samples, n_subcarriers)`
2. **Amplitude/Phase Extraction**: Separate into real-valued components
3. **Bandpass Filtering**: 0.5-50 Hz (configurable)
4. **PSD Computation**: Welch's method with 256-point segments
5. **Feature Extraction**: 8 statistical features per subcarrier
6. **Output**: Feature vector for neural network

### Bayesian Inference

Uses **Monte Carlo Dropout** as a Bayesian approximation:
- Training: Dropout enabled (variational inference)
- Inference: Multiple forward passes (N=50 default)
- Uncertainty: Variance across predictions

### Loss Function

Heteroscedastic loss (negative log-likelihood):
```
L = 0.5 * exp(-log_var) * (y - y_pred)² + 0.5 * log_var
```
This learns both mean and variance, capturing aleatoric uncertainty.

## 📁 Project Structure

```
src/spectral_csi/
├── __init__.py                    # Package initialization
├── models/
│   └── __init__.py                # Bayesian neural networks
├── preprocessing/
│   └── __init__.py                # CSI signal processing
├── optimization/
│   └── __init__.py                # Uncertainty-aware optimization
└── utils/
    ├── __init__.py                # General utilities
    └── data_utils.py              # Data loading and generation

configs/                           # Configuration files
├── default_config.yaml
└── example_config.json

src/
├── train.py                       # Training script
└── inference.py                   # Inference script
```

## 🔧 Configuration

Use YAML or JSON configuration files:

```yaml
# configs/default_config.yaml
model:
  hidden_dims: [256, 128, 64]
  dropout_rate: 0.15
  n_monte_carlo: 50

training:
  epochs: 100
  learning_rate: 0.001
  batch_size: 32
```

## 🤝 Contributing

We welcome contributions! Areas of interest:
- Additional model architectures
- Improved uncertainty quantification methods
- Real CSI data collection tools
- Performance optimizations

## 📝 Citation

If you use Spectral-CSI in your research, please cite:

```bibtex
@software{spectral_csi,
  title = {Spectral-CSI: Privacy-Preserving Smart Building Occupancy Estimation},
  author = {Spectral-CSI Research Team},
  year = {2024},
  url = {https://github.com/Einheit-Zenkai/spectral-csi}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built with PyTorch, NumPy, and SciPy
- Inspired by research in CSI-based sensing and Bayesian deep learning
- Designed for privacy-preserving smart building applications

## 📧 Contact

For questions, issues, or collaborations:
- Open an issue on GitHub
- Visit: https://github.com/Einheit-Zenkai/spectral-csi

---

**Note**: This framework uses synthetic CSI data for demonstration. For real-world applications, you'll need to collect actual CSI data from Wi-Fi devices using tools like [Linux 802.11n CSI Tool](https://dhalperi.github.io/linux-80211n-csitool/) or similar hardware.
