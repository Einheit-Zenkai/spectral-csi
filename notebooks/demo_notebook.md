# Spectral-CSI Demo: Occupancy Estimation with Uncertainty Quantification

This notebook demonstrates the key features of the Spectral-CSI framework for privacy-preserving occupancy estimation using Wi-Fi CSI data.

## Setup

```python
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from spectral_csi.preprocessing import CSIPreprocessor
from spectral_csi.models import BayesianOccupancyEstimator
from spectral_csi.utils import set_seed
from spectral_csi.utils.data_utils import generate_synthetic_csi_data

# Set style and seed
sns.set_style('whitegrid')
set_seed(42)
```

## 1. Generate Synthetic CSI Data

```python
# Generate synthetic CSI data for demonstration
print("Generating synthetic CSI data...")
csi_data, true_occupancy = generate_synthetic_csi_data(
    n_samples=500,
    n_subcarriers=30,
    sequence_length=100,
    max_occupancy=10,
    noise_level=0.1,
    seed=42
)

print(f"CSI data shape: {csi_data.shape}")
print(f"Occupancy range: {true_occupancy.min():.0f} to {true_occupancy.max():.0f}")
```

## 2. Signal Preprocessing

```python
# Initialize preprocessor
preprocessor = CSIPreprocessor(
    sampling_rate=1000.0,
    n_subcarriers=30,
    lowpass_cutoff=50.0,
    highpass_cutoff=0.5
)

# Preprocess a single CSI sample
sample_csi = csi_data[0]
processed = preprocessor.preprocess(sample_csi, extract_features=True)

print("\nProcessed data contains:")
for key in processed.keys():
    if isinstance(processed[key], dict):
        print(f"  {key}: {list(processed[key].keys())}")
    else:
        print(f"  {key}: shape {processed[key].shape}")
```

## 3. Visualize CSI Data

```python
# Visualize amplitude and phase
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Raw amplitude
im1 = axes[0, 0].imshow(np.abs(sample_csi).T, aspect='auto', cmap='viridis')
axes[0, 0].set_title('CSI Amplitude (Raw)')
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('Subcarrier')
plt.colorbar(im1, ax=axes[0, 0])

# Filtered amplitude
im2 = axes[0, 1].imshow(processed['amplitude'].T, aspect='auto', cmap='viridis')
axes[0, 1].set_title('CSI Amplitude (Filtered)')
axes[0, 1].set_xlabel('Time')
axes[0, 1].set_ylabel('Subcarrier')
plt.colorbar(im2, ax=axes[0, 1])

# Power Spectral Density
axes[1, 0].plot(processed['frequencies'], processed['psd_amplitude'][:, 0])
axes[1, 0].set_title('Power Spectral Density (Subcarrier 0)')
axes[1, 0].set_xlabel('Frequency (Hz)')
axes[1, 0].set_ylabel('PSD')
axes[1, 0].grid(True)

# Statistical features
features = processed['amplitude_features']
feature_names = list(features.keys())
feature_values = [np.mean(features[f]) for f in feature_names]
axes[1, 1].bar(range(len(feature_names)), feature_values)
axes[1, 1].set_xticks(range(len(feature_names)))
axes[1, 1].set_xticklabels(feature_names, rotation=45)
axes[1, 1].set_title('Statistical Features')
axes[1, 1].set_ylabel('Mean Value')

plt.tight_layout()
plt.show()
```

## 4. Prepare Training Data

```python
# Preprocess all samples
print("Preprocessing all CSI samples...")
processed_features = []

for i in range(len(csi_data)):
    result = preprocessor.preprocess(csi_data[i], extract_features=True)
    
    # Flatten features
    amp_feats = np.concatenate([v.flatten() for v in result['amplitude_features'].values()])
    phase_feats = np.concatenate([v.flatten() for v in result['phase_features'].values()])
    features = np.concatenate([amp_feats, phase_feats])
    processed_features.append(features)

processed_features = np.array(processed_features)

print(f"Feature matrix shape: {processed_features.shape}")
print(f"Number of features: {processed_features.shape[1]}")
```

## 5. Train Bayesian Model

```python
# Split data
split_idx = int(0.8 * len(processed_features))
train_data = torch.FloatTensor(processed_features[:split_idx])
train_labels = torch.FloatTensor(true_occupancy[:split_idx]).unsqueeze(1)
test_data = torch.FloatTensor(processed_features[split_idx:])
test_labels = torch.FloatTensor(true_occupancy[split_idx:]).unsqueeze(1)

# Initialize model
model = BayesianOccupancyEstimator(
    input_dim=processed_features.shape[1],
    hidden_dims=[256, 128, 64],
    dropout_rate=0.15,
    n_monte_carlo=50
)

# Training loop (simplified)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
epochs = 20

print("Training model...")
losses = []

for epoch in range(epochs):
    model.train()
    
    # Forward pass
    mean, log_var = model(train_data)
    
    # Compute loss
    loss = model.compute_loss((mean, log_var), train_labels)
    
    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    losses.append(loss.item())
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

# Plot training loss
plt.figure(figsize=(10, 5))
plt.plot(losses)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.show()
```

## 6. Make Predictions with Uncertainty

```python
# Make predictions on test set
model.eval()
print("Making predictions with uncertainty quantification...")

mean_pred, aleatoric_unc, epistemic_unc = model.predict_with_uncertainty(test_data)

mean_pred = mean_pred.detach().numpy().squeeze()
aleatoric_unc = aleatoric_unc.detach().numpy().squeeze()
epistemic_unc = epistemic_unc.detach().numpy().squeeze()
total_unc = aleatoric_unc + epistemic_unc
test_labels_np = test_labels.numpy().squeeze()

# Calculate errors
errors = np.abs(mean_pred - test_labels_np)
rmse = np.sqrt(np.mean(errors**2))
mae = np.mean(errors)

print(f"\nTest Set Performance:")
print(f"  RMSE: {rmse:.4f}")
print(f"  MAE: {mae:.4f}")
print(f"  Mean Epistemic Uncertainty: {np.mean(epistemic_unc):.4f}")
print(f"  Mean Aleatoric Uncertainty: {np.mean(aleatoric_unc):.4f}")
```

## 7. Visualize Results

```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Predictions vs True Values
axes[0, 0].scatter(test_labels_np, mean_pred, alpha=0.6)
axes[0, 0].plot([0, 10], [0, 10], 'r--', label='Perfect prediction')
axes[0, 0].set_xlabel('True Occupancy')
axes[0, 0].set_ylabel('Predicted Occupancy')
axes[0, 0].set_title('Predictions vs True Values')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Prediction errors with uncertainty
sort_idx = np.argsort(test_labels_np)
axes[0, 1].errorbar(
    range(len(sort_idx)),
    mean_pred[sort_idx],
    yerr=total_unc[sort_idx],
    fmt='o',
    alpha=0.6,
    capsize=3,
    label='Predictions ± Total Uncertainty'
)
axes[0, 1].plot(test_labels_np[sort_idx], 'r-', label='True Occupancy', linewidth=2)
axes[0, 1].set_xlabel('Sample Index (sorted by true occupancy)')
axes[0, 1].set_ylabel('Occupancy')
axes[0, 1].set_title('Predictions with Uncertainty Bands')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Uncertainty decomposition
x_pos = np.arange(len(mean_pred))
axes[1, 0].bar(x_pos, aleatoric_unc, label='Aleatoric', alpha=0.7)
axes[1, 0].bar(x_pos, epistemic_unc, bottom=aleatoric_unc, label='Epistemic', alpha=0.7)
axes[1, 0].set_xlabel('Sample Index')
axes[1, 0].set_ylabel('Uncertainty')
axes[1, 0].set_title('Uncertainty Decomposition')
axes[1, 0].legend()

# Error vs Uncertainty correlation
axes[1, 1].scatter(total_unc, errors, alpha=0.6)
axes[1, 1].set_xlabel('Total Uncertainty')
axes[1, 1].set_ylabel('Prediction Error')
axes[1, 1].set_title('Error vs Uncertainty Correlation')
axes[1, 1].grid(True)

# Add correlation coefficient
correlation = np.corrcoef(total_unc, errors)[0, 1]
axes[1, 1].text(0.05, 0.95, f'Correlation: {correlation:.3f}',
                transform=axes[1, 1].transAxes,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()
```

## 8. Sample-by-Sample Analysis

```python
print("\n" + "="*70)
print("SAMPLE-BY-SAMPLE PREDICTIONS")
print("="*70)

for i in range(min(10, len(mean_pred))):
    print(f"\nSample {i+1}:")
    print(f"  True Occupancy:        {test_labels_np[i]:.1f}")
    print(f"  Predicted Occupancy:   {mean_pred[i]:.2f}")
    print(f"  Error:                 {errors[i]:.2f}")
    print(f"  Epistemic Uncertainty: {epistemic_unc[i]:.4f}")
    print(f"  Aleatoric Uncertainty: {aleatoric_unc[i]:.4f}")
    print(f"  Total Uncertainty:     {total_unc[i]:.4f}")
```

## Summary

This notebook demonstrated:

1. **CSI Data Generation**: Synthetic Wi-Fi CSI data for testing
2. **Signal Processing**: Amplitude/phase extraction, filtering, PSD computation
3. **Feature Extraction**: Statistical features for occupancy estimation
4. **Bayesian Model**: Neural network with uncertainty quantification
5. **Training**: Simple training loop with loss monitoring
6. **Uncertainty Quantification**: Aleatoric and epistemic uncertainty estimation
7. **Visualization**: Comprehensive analysis of predictions and uncertainties

The framework provides both accurate predictions and reliable uncertainty estimates, enabling safer deployment in real-world smart building applications.
