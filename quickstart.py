#!/usr/bin/env python3
"""
Quick start demo for Spectral-CSI framework.

This script demonstrates the basic workflow:
1. Generate synthetic CSI data
2. Preprocess the data
3. Train a Bayesian model
4. Make predictions with uncertainty quantification
"""

import sys
sys.path.insert(0, 'src')

import numpy as np
import torch
from spectral_csi.preprocessing import CSIPreprocessor
from spectral_csi.models import BayesianOccupancyEstimator
from spectral_csi.utils import set_seed, compute_metrics
from spectral_csi.utils.data_utils import generate_synthetic_csi_data, create_dataloaders


def main():
    print("="*70)
    print("Spectral-CSI Quick Start Demo")
    print("="*70)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # 1. Generate synthetic CSI data
    print("\n[1/5] Generating synthetic CSI data...")
    csi_data, occupancy = generate_synthetic_csi_data(
        n_samples=500,
        n_subcarriers=30,
        sequence_length=100,
        max_occupancy=10,
        seed=42
    )
    print(f"  ✓ Generated {len(csi_data)} samples")
    print(f"  ✓ CSI shape: {csi_data.shape}")
    print(f"  ✓ Occupancy range: {occupancy.min():.0f} to {occupancy.max():.0f}")
    
    # 2. Preprocess CSI data
    print("\n[2/5] Preprocessing CSI data...")
    preprocessor = CSIPreprocessor(
        n_subcarriers=30,
        sampling_rate=1000.0
    )
    
    processed_features = []
    for i in range(len(csi_data)):
        result = preprocessor.preprocess(csi_data[i], extract_features=True)
        
        # Flatten features
        amp_feats = np.concatenate([v.flatten() for v in result['amplitude_features'].values()])
        phase_feats = np.concatenate([v.flatten() for v in result['phase_features'].values()])
        features = np.concatenate([amp_feats, phase_feats])
        processed_features.append(features)
    
    processed_features = np.array(processed_features)
    print(f"  ✓ Feature dimension: {processed_features.shape[1]}")
    
    # 3. Split data and create loaders
    print("\n[3/5] Preparing data loaders...")
    split_idx = int(0.8 * len(processed_features))
    train_data = processed_features[:split_idx]
    train_labels = occupancy[:split_idx]
    test_data = processed_features[split_idx:]
    test_labels = occupancy[split_idx:]
    
    dataloaders = create_dataloaders(
        train_data, train_labels,
        test_data, test_labels,
        batch_size=32,
        shuffle=True
    )
    print(f"  ✓ Training samples: {len(train_data)}")
    print(f"  ✓ Test samples: {len(test_data)}")
    
    # 4. Initialize and train model
    print("\n[4/5] Training Bayesian model...")
    model = BayesianOccupancyEstimator(
        input_dim=processed_features.shape[1],
        hidden_dims=[256, 128, 64],
        dropout_rate=0.15,
        n_monte_carlo=50
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Simple training loop
    epochs = 10
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_data, batch_labels in dataloaders['train']:
            batch_labels = batch_labels.unsqueeze(1)
            
            # Forward pass
            mean, log_var = model(batch_data)
            loss = model.compute_loss((mean, log_var), batch_labels)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        if (epoch + 1) % 2 == 0:
            print(f"  Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(dataloaders['train']):.4f}")
    
    print("  ✓ Training completed!")
    
    # 5. Make predictions with uncertainty
    print("\n[5/5] Making predictions with uncertainty quantification...")
    model.eval()
    
    test_data_tensor = torch.FloatTensor(test_data)
    mean_pred, aleatoric_unc, epistemic_unc = model.predict_with_uncertainty(
        test_data_tensor, n_samples=50
    )
    
    # Convert to numpy
    mean_pred = mean_pred.detach().numpy().squeeze()
    aleatoric_unc = aleatoric_unc.detach().numpy().squeeze()
    epistemic_unc = epistemic_unc.detach().numpy().squeeze()
    
    # Compute metrics
    metrics = compute_metrics(mean_pred, test_labels)
    
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Mean Absolute Error (MAE):    {metrics['mae']:.4f}")
    print(f"Root Mean Squared Error (RMSE): {metrics['rmse']:.4f}")
    print(f"R² Score:                     {metrics['r2']:.4f}")
    print(f"Mean Epistemic Uncertainty:    {np.mean(epistemic_unc):.4f}")
    print(f"Mean Aleatoric Uncertainty:    {np.mean(aleatoric_unc):.4f}")
    
    # Show some example predictions
    print("\n" + "="*70)
    print("SAMPLE PREDICTIONS")
    print("="*70)
    for i in range(min(5, len(mean_pred))):
        print(f"\nSample {i+1}:")
        print(f"  True:       {test_labels[i]:.1f}")
        print(f"  Predicted:  {mean_pred[i]:.2f}")
        print(f"  Error:      {abs(mean_pred[i] - test_labels[i]):.2f}")
        print(f"  Epistemic Unc: {epistemic_unc[i]:.4f}")
        print(f"  Aleatoric Unc: {aleatoric_unc[i]:.4f}")
    
    print("\n" + "="*70)
    print("✅ Demo completed successfully!")
    print("="*70)
    print("\nNext steps:")
    print("1. Replace synthetic data with real CSI measurements")
    print("2. Experiment with different model architectures")
    print("3. Tune hyperparameters for your specific use case")
    print("4. See src/train.py and src/inference.py for full examples")
    print("\nFor more information, check the README.md")


if __name__ == '__main__':
    main()
