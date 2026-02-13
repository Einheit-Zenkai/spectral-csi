"""
Basic tests for Spectral-CSI framework.
"""

import pytest
import numpy as np
import torch

from spectral_csi.preprocessing import CSIPreprocessor
from spectral_csi.models import BayesianOccupancyEstimator
from spectral_csi.utils.data_utils import generate_synthetic_csi_data, CSIDataset


class TestCSIPreprocessor:
    """Test CSI preprocessing functionality."""
    
    def test_initialization(self):
        """Test preprocessor initialization."""
        preprocessor = CSIPreprocessor(
            sampling_rate=1000.0,
            n_subcarriers=30
        )
        assert preprocessor.sampling_rate == 1000.0
        assert preprocessor.n_subcarriers == 30
    
    def test_amplitude_phase_extraction(self):
        """Test amplitude and phase extraction."""
        preprocessor = CSIPreprocessor()
        
        # Create synthetic complex CSI data
        csi_complex = np.random.randn(100, 30) + 1j * np.random.randn(100, 30)
        
        amplitude, phase = preprocessor.extract_amplitude_phase(csi_complex)
        
        assert amplitude.shape == (100, 30)
        assert phase.shape == (100, 30)
        assert np.all(amplitude >= 0)
    
    def test_bandpass_filter(self):
        """Test bandpass filtering."""
        preprocessor = CSIPreprocessor()
        
        data = np.random.randn(100, 30)
        filtered = preprocessor.apply_bandpass_filter(data)
        
        assert filtered.shape == data.shape
    
    def test_psd_computation(self):
        """Test PSD computation."""
        preprocessor = CSIPreprocessor(nperseg=64)
        
        data = np.random.randn(256, 30)
        frequencies, psd = preprocessor.compute_psd(data)
        
        assert len(frequencies) > 0
        assert psd.shape[1] == 30
        assert np.all(psd >= 0)
    
    def test_full_preprocessing(self):
        """Test full preprocessing pipeline."""
        preprocessor = CSIPreprocessor()
        
        csi_complex = np.random.randn(100, 30) + 1j * np.random.randn(100, 30)
        result = preprocessor.preprocess(csi_complex, extract_features=True)
        
        assert 'amplitude' in result
        assert 'phase' in result
        assert 'psd_amplitude' in result
        assert 'amplitude_features' in result
        assert 'phase_features' in result


class TestBayesianOccupancyEstimator:
    """Test Bayesian occupancy estimation model."""
    
    def test_model_initialization(self):
        """Test model initialization."""
        model = BayesianOccupancyEstimator(
            input_dim=100,
            hidden_dims=[64, 32],
            dropout_rate=0.1
        )
        
        assert model.input_dim == 100
        assert model.dropout_rate == 0.1
    
    def test_forward_pass(self):
        """Test forward pass."""
        model = BayesianOccupancyEstimator(input_dim=50)
        
        x = torch.randn(10, 50)
        mean, log_var = model(x)
        
        assert mean.shape == (10, 1)
        assert log_var.shape == (10, 1)
    
    def test_predict_with_uncertainty(self):
        """Test prediction with uncertainty."""
        model = BayesianOccupancyEstimator(
            input_dim=50,
            n_monte_carlo=10
        )
        
        x = torch.randn(5, 50)
        mean_pred, aleatoric, epistemic = model.predict_with_uncertainty(x)
        
        assert mean_pred.shape == (5, 1)
        assert aleatoric.shape == (5, 1)
        assert epistemic.shape == (5, 1)
        assert torch.all(aleatoric >= 0)
        assert torch.all(epistemic >= 0)
    
    def test_loss_computation(self):
        """Test loss computation."""
        model = BayesianOccupancyEstimator(input_dim=50)
        
        x = torch.randn(10, 50)
        targets = torch.randn(10, 1)
        
        predictions = model(x)
        loss = model.compute_loss(predictions, targets)
        
        assert loss.item() >= 0
        assert not torch.isnan(loss)


class TestDataUtils:
    """Test data utility functions."""
    
    def test_synthetic_data_generation(self):
        """Test synthetic CSI data generation."""
        csi_data, occupancy = generate_synthetic_csi_data(
            n_samples=100,
            n_subcarriers=30,
            sequence_length=50,
            seed=42
        )
        
        assert csi_data.shape == (100, 30, 50)
        assert occupancy.shape == (100,)
        assert np.all(occupancy >= 0)
        assert np.iscomplexobj(csi_data)
    
    def test_csi_dataset(self):
        """Test CSI dataset class."""
        data = np.random.randn(100, 50)
        labels = np.random.rand(100)
        
        dataset = CSIDataset(data, labels)
        
        assert len(dataset) == 100
        
        sample_data, sample_label = dataset[0]
        assert isinstance(sample_data, torch.Tensor)
        assert isinstance(sample_label, torch.Tensor)
        assert sample_data.shape == (50,)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
