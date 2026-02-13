"""
Training script for Spectral-CSI occupancy estimation model.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import numpy as np
import json
from tqdm import tqdm

from spectral_csi.models import BayesianOccupancyEstimator, ConvBayesianOccupancyEstimator
from spectral_csi.preprocessing import CSIPreprocessor
from spectral_csi.optimization import UncertaintyAwareOptimizer
from spectral_csi.utils import set_seed, get_device, compute_metrics, EarlyStopping
from spectral_csi.utils.data_utils import generate_synthetic_csi_data, create_dataloaders


def train_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        target = target.unsqueeze(1)
        
        # Forward pass
        mean, log_var = model(data)
        
        # Compute loss
        loss = model.compute_loss((mean, log_var), target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(dataloader)


def validate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device
) -> dict:
    """Validate model."""
    model.eval()
    predictions = []
    targets = []
    epistemic_uncertainties = []
    aleatoric_uncertainties = []
    
    with torch.no_grad():
        for data, target in tqdm(dataloader, desc='Validation'):
            data = data.to(device)
            
            # Get predictions with uncertainty
            mean_pred, aleatoric_unc, epistemic_unc = model.predict_with_uncertainty(data)
            
            predictions.append(mean_pred.cpu().numpy())
            targets.append(target.numpy())
            epistemic_uncertainties.append(epistemic_unc.cpu().numpy())
            aleatoric_uncertainties.append(aleatoric_unc.cpu().numpy())
    
    predictions = np.concatenate(predictions, axis=0)
    targets = np.concatenate(targets, axis=0)
    epistemic_uncertainties = np.concatenate(epistemic_uncertainties, axis=0)
    aleatoric_uncertainties = np.concatenate(aleatoric_uncertainties, axis=0)
    
    # Compute metrics
    metrics = compute_metrics(predictions.squeeze(), targets)
    metrics['mean_epistemic_uncertainty'] = float(np.mean(epistemic_uncertainties))
    metrics['mean_aleatoric_uncertainty'] = float(np.mean(aleatoric_uncertainties))
    
    return metrics


def main(args):
    """Main training function."""
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Get device
    device = get_device(args.gpu)
    print(f"Using device: {device}")
    
    # Generate synthetic data (in practice, load real CSI data)
    print("Generating synthetic CSI data...")
    csi_data, occupancy = generate_synthetic_csi_data(
        n_samples=args.n_samples,
        n_subcarriers=args.n_subcarriers,
        sequence_length=args.sequence_length,
        seed=args.seed
    )
    
    # Preprocess data
    print("Preprocessing CSI data...")
    preprocessor = CSIPreprocessor(
        n_subcarriers=args.n_subcarriers,
        sampling_rate=args.sampling_rate
    )
    
    processed_features = []
    for i in range(len(csi_data)):
        result = preprocessor.preprocess(csi_data[i], extract_features=True)
        
        # Flatten features for simple model
        amp_feats = np.concatenate([v.flatten() for v in result['amplitude_features'].values()])
        phase_feats = np.concatenate([v.flatten() for v in result['phase_features'].values()])
        features = np.concatenate([amp_feats, phase_feats])
        processed_features.append(features)
    
    processed_features = np.array(processed_features)
    
    # Split into train/val
    split_idx = int(0.8 * len(processed_features))
    train_data = processed_features[:split_idx]
    train_labels = occupancy[:split_idx]
    val_data = processed_features[split_idx:]
    val_labels = occupancy[split_idx:]
    
    print(f"Training samples: {len(train_data)}, Validation samples: {len(val_data)}")
    print(f"Feature dimension: {train_data.shape[1]}")
    
    # Create data loaders
    dataloaders = create_dataloaders(
        train_data, train_labels,
        val_data, val_labels,
        batch_size=args.batch_size,
        shuffle=True
    )
    
    # Initialize model
    print("Initializing Bayesian model...")
    model = BayesianOccupancyEstimator(
        input_dim=train_data.shape[1],
        hidden_dims=args.hidden_dims,
        dropout_rate=args.dropout_rate,
        n_monte_carlo=args.n_monte_carlo
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, mode='min')
    
    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_metrics': []}
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_epoch(model, dataloaders['train'], optimizer, device, epoch)
        history['train_loss'].append(train_loss)
        
        # Validate
        val_metrics = validate(model, dataloaders['val'], device)
        history['val_metrics'].append(val_metrics)
        
        # Update learning rate
        scheduler.step(val_metrics['rmse'])
        
        print(f"\nEpoch {epoch}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val RMSE: {val_metrics['rmse']:.4f}")
        print(f"  Val R²: {val_metrics['r2']:.4f}")
        print(f"  Epistemic Unc: {val_metrics['mean_epistemic_uncertainty']:.4f}")
        print(f"  Aleatoric Unc: {val_metrics['mean_aleatoric_uncertainty']:.4f}")
        
        # Save best model
        if val_metrics['rmse'] < best_val_loss:
            best_val_loss = val_metrics['rmse']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_metrics': val_metrics,
            }, args.save_dir / 'best_model.pt')
            print(f"  --> Saved best model (RMSE: {best_val_loss:.4f})")
        
        # Early stopping
        if early_stopping(val_metrics['rmse']):
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break
    
    # Save training history
    with open(args.save_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print("\nTraining completed!")
    print(f"Best validation RMSE: {best_val_loss:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Spectral-CSI occupancy estimation model')
    
    # Data parameters
    parser.add_argument('--n-samples', type=int, default=1000, help='Number of samples')
    parser.add_argument('--n-subcarriers', type=int, default=30, help='Number of CSI subcarriers')
    parser.add_argument('--sequence-length', type=int, default=100, help='Sequence length')
    parser.add_argument('--sampling-rate', type=float, default=1000.0, help='Sampling rate (Hz)')
    
    # Model parameters
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[256, 128, 64], 
                        help='Hidden layer dimensions')
    parser.add_argument('--dropout-rate', type=float, default=0.15, help='Dropout rate')
    parser.add_argument('--n-monte-carlo', type=int, default=50, help='Monte Carlo samples')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-5, help='Weight decay')
    parser.add_argument('--patience', type=int, default=10, help='Early stopping patience')
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    parser.add_argument('--save-dir', type=Path, default=Path('checkpoints'), 
                        help='Directory to save models')
    
    args = parser.parse_args()
    
    # Create save directory
    args.save_dir.mkdir(exist_ok=True, parents=True)
    
    main(args)
