"""
Dataset and data loading utilities for CSI data.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Optional, Tuple, Dict
from pathlib import Path


class CSIDataset(Dataset):
    """
    PyTorch Dataset for CSI-based occupancy data.
    """
    
    def __init__(
        self,
        csi_data: np.ndarray,
        occupancy_labels: np.ndarray,
        transform: Optional[callable] = None
    ):
        """
        Initialize CSI dataset.
        
        Args:
            csi_data: CSI data array of shape (n_samples, n_features)
            occupancy_labels: Occupancy labels of shape (n_samples,)
            transform: Optional transform to apply to data
        """
        self.csi_data = torch.FloatTensor(csi_data)
        self.occupancy_labels = torch.FloatTensor(occupancy_labels)
        self.transform = transform
        
        assert len(self.csi_data) == len(self.occupancy_labels), \
            "Data and labels must have same length"
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.csi_data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get data sample.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (csi_data, occupancy_label)
        """
        data = self.csi_data[idx]
        label = self.occupancy_labels[idx]
        
        if self.transform:
            data = self.transform(data)
        
        return data, label


class CSISequenceDataset(Dataset):
    """
    PyTorch Dataset for sequential CSI data (for convolutional models).
    """
    
    def __init__(
        self,
        csi_sequences: np.ndarray,
        occupancy_labels: np.ndarray,
        transform: Optional[callable] = None
    ):
        """
        Initialize CSI sequence dataset.
        
        Args:
            csi_sequences: CSI sequences of shape (n_samples, n_subcarriers, seq_len)
            occupancy_labels: Occupancy labels of shape (n_samples,)
            transform: Optional transform to apply
        """
        self.csi_sequences = torch.FloatTensor(csi_sequences)
        self.occupancy_labels = torch.FloatTensor(occupancy_labels)
        self.transform = transform
        
        assert len(self.csi_sequences) == len(self.occupancy_labels)
    
    def __len__(self) -> int:
        return len(self.csi_sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        sequence = self.csi_sequences[idx]
        label = self.occupancy_labels[idx]
        
        if self.transform:
            sequence = self.transform(sequence)
        
        return sequence, label


def create_dataloaders(
    train_data: np.ndarray,
    train_labels: np.ndarray,
    val_data: Optional[np.ndarray] = None,
    val_labels: Optional[np.ndarray] = None,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    sequence_mode: bool = False
) -> Dict[str, DataLoader]:
    """
    Create training and validation data loaders.
    
    Args:
        train_data: Training data
        train_labels: Training labels
        val_data: Validation data (optional)
        val_labels: Validation labels (optional)
        batch_size: Batch size for training
        shuffle: Whether to shuffle training data
        num_workers: Number of worker processes
        sequence_mode: Whether to use sequence dataset
        
    Returns:
        Dictionary of DataLoaders
    """
    dataset_class = CSISequenceDataset if sequence_mode else CSIDataset
    
    train_dataset = dataset_class(train_data, train_labels)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )
    
    dataloaders = {'train': train_loader}
    
    if val_data is not None and val_labels is not None:
        val_dataset = dataset_class(val_data, val_labels)
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers
        )
        dataloaders['val'] = val_loader
    
    return dataloaders


def generate_synthetic_csi_data(
    n_samples: int = 1000,
    n_subcarriers: int = 30,
    sequence_length: int = 100,
    max_occupancy: int = 10,
    noise_level: float = 0.1,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic CSI data for testing.
    
    Args:
        n_samples: Number of samples
        n_subcarriers: Number of CSI subcarriers
        sequence_length: Length of sequences
        max_occupancy: Maximum occupancy count
        noise_level: Noise level to add
        seed: Random seed
        
    Returns:
        Tuple of (csi_data, occupancy_labels)
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Generate occupancy labels
    occupancy = np.random.randint(0, max_occupancy + 1, size=n_samples)
    
    # Generate CSI data (complex values)
    # Amplitude affected by occupancy
    base_amplitude = 1.0
    amplitude = base_amplitude - 0.05 * occupancy[:, np.newaxis, np.newaxis]
    amplitude = amplitude + noise_level * np.random.randn(n_samples, n_subcarriers, sequence_length)
    
    # Random phase
    phase = 2 * np.pi * np.random.rand(n_samples, n_subcarriers, sequence_length)
    
    # Create complex CSI
    csi_complex = amplitude * np.exp(1j * phase)
    
    return csi_complex, occupancy.astype(np.float32)
