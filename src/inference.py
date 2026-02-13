"""
Inference script for trained Spectral-CSI model.
"""

import argparse
import torch
import numpy as np
from pathlib import Path
import json

from spectral_csi.models import BayesianOccupancyEstimator
from spectral_csi.preprocessing import CSIPreprocessor
from spectral_csi.utils import set_seed, get_device
from spectral_csi.utils.data_utils import generate_synthetic_csi_data


def predict_with_uncertainty(
    model: torch.nn.Module,
    data: torch.Tensor,
    device: torch.device,
    n_samples: int = 100
) -> dict:
    """
    Make predictions with uncertainty quantification.
    
    Args:
        model: Trained model
        data: Input data
        device: Device to run on
        n_samples: Number of Monte Carlo samples
        
    Returns:
        Dictionary with predictions and uncertainties
    """
    model.eval()
    data = data.to(device)
    
    mean_pred, aleatoric_unc, epistemic_unc = model.predict_with_uncertainty(
        data, n_samples=n_samples
    )
    
    # Total uncertainty
    total_unc = aleatoric_unc + epistemic_unc
    
    return {
        'predictions': mean_pred.cpu().numpy(),
        'aleatoric_uncertainty': aleatoric_unc.cpu().numpy(),
        'epistemic_uncertainty': epistemic_unc.cpu().numpy(),
        'total_uncertainty': total_unc.cpu().numpy()
    }


def main(args):
    """Main inference function."""
    set_seed(args.seed)
    device = get_device(args.gpu)
    
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"Loading model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # Initialize model (need to know architecture)
    # In practice, save model config with checkpoint
    model = BayesianOccupancyEstimator(
        input_dim=args.input_dim,
        hidden_dims=args.hidden_dims,
        dropout_rate=args.dropout_rate,
        n_monte_carlo=args.n_monte_carlo
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']}")
    
    # Generate test data
    print("Generating test data...")
    csi_data, true_occupancy = generate_synthetic_csi_data(
        n_samples=args.n_samples,
        n_subcarriers=args.n_subcarriers,
        sequence_length=args.sequence_length,
        seed=args.seed
    )
    
    # Preprocess
    print("Preprocessing...")
    preprocessor = CSIPreprocessor(
        n_subcarriers=args.n_subcarriers,
        sampling_rate=args.sampling_rate
    )
    
    processed_features = []
    for i in range(len(csi_data)):
        result = preprocessor.preprocess(csi_data[i], extract_features=True)
        amp_feats = np.concatenate([v.flatten() for v in result['amplitude_features'].values()])
        phase_feats = np.concatenate([v.flatten() for v in result['phase_features'].values()])
        features = np.concatenate([amp_feats, phase_feats])
        processed_features.append(features)
    
    processed_features = np.array(processed_features)
    test_data = torch.FloatTensor(processed_features)
    
    # Make predictions
    print("Making predictions with uncertainty quantification...")
    results = predict_with_uncertainty(model, test_data, device, n_samples=args.n_monte_carlo)
    
    # Display results
    print("\n" + "="*60)
    print("INFERENCE RESULTS")
    print("="*60)
    
    for i in range(min(10, len(results['predictions']))):
        pred = results['predictions'][i, 0]
        epistemic = results['epistemic_uncertainty'][i, 0]
        aleatoric = results['aleatoric_uncertainty'][i, 0]
        total = results['total_uncertainty'][i, 0]
        true = true_occupancy[i]
        
        print(f"\nSample {i+1}:")
        print(f"  True Occupancy:        {true:.1f}")
        print(f"  Predicted Occupancy:   {pred:.2f}")
        print(f"  Epistemic Uncertainty: {epistemic:.4f}")
        print(f"  Aleatoric Uncertainty: {aleatoric:.4f}")
        print(f"  Total Uncertainty:     {total:.4f}")
        print(f"  Error:                 {abs(pred - true):.2f}")
    
    # Statistics
    predictions = results['predictions'].squeeze()
    errors = np.abs(predictions - true_occupancy)
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"Mean Absolute Error:       {np.mean(errors):.4f}")
    print(f"RMSE:                      {np.sqrt(np.mean(errors**2)):.4f}")
    print(f"Mean Epistemic Uncertainty: {np.mean(results['epistemic_uncertainty']):.4f}")
    print(f"Mean Aleatoric Uncertainty: {np.mean(results['aleatoric_uncertainty']):.4f}")
    print(f"Mean Total Uncertainty:     {np.mean(results['total_uncertainty']):.4f}")
    
    # Save results
    if args.save_results:
        output = {
            'predictions': predictions.tolist(),
            'true_occupancy': true_occupancy.tolist(),
            'epistemic_uncertainty': results['epistemic_uncertainty'].squeeze().tolist(),
            'aleatoric_uncertainty': results['aleatoric_uncertainty'].squeeze().tolist(),
            'errors': errors.tolist()
        }
        
        output_path = Path(args.output_dir) / 'inference_results.json'
        output_path.parent.mkdir(exist_ok=True, parents=True)
        
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        
        print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Inference with Spectral-CSI model')
    
    # Model parameters
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--input-dim', type=int, default=480, help='Input dimension')
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[256, 128, 64])
    parser.add_argument('--dropout-rate', type=float, default=0.15)
    parser.add_argument('--n-monte-carlo', type=int, default=100, help='Monte Carlo samples')
    
    # Data parameters
    parser.add_argument('--n-samples', type=int, default=100, help='Number of test samples')
    parser.add_argument('--n-subcarriers', type=int, default=30)
    parser.add_argument('--sequence-length', type=int, default=100)
    parser.add_argument('--sampling-rate', type=float, default=1000.0)
    
    # Other parameters
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--save-results', action='store_true', help='Save results to file')
    parser.add_argument('--output-dir', type=str, default='results', help='Output directory')
    
    args = parser.parse_args()
    
    main(args)
