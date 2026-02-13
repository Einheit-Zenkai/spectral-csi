"""
Signal preprocessing module for CSI data.

This module handles raw CSI data processing, including:
- Amplitude and phase extraction
- Noise reduction and filtering
- Power Spectral Density (PSD) computation
- Feature extraction for occupancy estimation
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq
from typing import Tuple, Optional, Dict
import warnings


class CSIPreprocessor:
    """
    Preprocessor for Wi-Fi Channel State Information (CSI) data.
    
    Performs signal processing operations including filtering, PSD computation,
    and feature extraction for occupancy estimation.
    """
    
    def __init__(
        self,
        sampling_rate: float = 1000.0,
        n_subcarriers: int = 30,
        lowpass_cutoff: float = 50.0,
        highpass_cutoff: float = 0.5,
        nperseg: int = 256
    ):
        """
        Initialize CSI preprocessor.
        
        Args:
            sampling_rate: Sampling rate in Hz
            n_subcarriers: Number of CSI subcarriers
            lowpass_cutoff: Low-pass filter cutoff frequency (Hz)
            highpass_cutoff: High-pass filter cutoff frequency (Hz)
            nperseg: Length of each segment for PSD computation
        """
        self.sampling_rate = sampling_rate
        self.n_subcarriers = n_subcarriers
        self.lowpass_cutoff = lowpass_cutoff
        self.highpass_cutoff = highpass_cutoff
        self.nperseg = nperseg
        
        # Design filters
        self._design_filters()
    
    def _design_filters(self):
        """Design Butterworth filters for signal preprocessing."""
        nyquist = self.sampling_rate / 2.0
        
        # Low-pass filter
        self.sos_low = signal.butter(
            4, self.lowpass_cutoff / nyquist, 
            btype='low', output='sos'
        )
        
        # High-pass filter  
        self.sos_high = signal.butter(
            4, self.highpass_cutoff / nyquist,
            btype='high', output='sos'
        )
    
    def extract_amplitude_phase(self, csi_complex: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract amplitude and phase from complex CSI data.
        
        Args:
            csi_complex: Complex CSI data of shape (n_samples, n_subcarriers)
            
        Returns:
            Tuple of (amplitude, phase) arrays
        """
        amplitude = np.abs(csi_complex)
        phase = np.angle(csi_complex)
        
        # Unwrap phase to avoid discontinuities
        phase = np.unwrap(phase, axis=0)
        
        return amplitude, phase
    
    def apply_bandpass_filter(self, data: np.ndarray) -> np.ndarray:
        """
        Apply bandpass filtering to remove noise.
        
        Args:
            data: Input signal of shape (n_samples, n_subcarriers)
            
        Returns:
            Filtered signal
        """
        # Apply high-pass then low-pass filter
        filtered = signal.sosfiltfilt(self.sos_high, data, axis=0)
        filtered = signal.sosfiltfilt(self.sos_low, filtered, axis=0)
        
        return filtered
    
    def compute_psd(self, data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Power Spectral Density using Welch's method.
        
        Args:
            data: Input signal of shape (n_samples, n_subcarriers)
            
        Returns:
            Tuple of (frequencies, psd) arrays
        """
        frequencies, psd = signal.welch(
            data,
            fs=self.sampling_rate,
            nperseg=self.nperseg,
            axis=0
        )
        
        return frequencies, psd
    
    def extract_statistical_features(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Extract statistical features from CSI data.
        
        Args:
            data: Input signal of shape (n_samples, n_subcarriers)
            
        Returns:
            Dictionary of features
        """
        features = {
            'mean': np.mean(data, axis=0),
            'std': np.std(data, axis=0),
            'variance': np.var(data, axis=0),
            'max': np.max(data, axis=0),
            'min': np.min(data, axis=0),
            'median': np.median(data, axis=0),
            'skewness': self._compute_skewness(data),
            'kurtosis': self._compute_kurtosis(data)
        }
        
        return features
    
    def _compute_skewness(self, data: np.ndarray) -> np.ndarray:
        """Compute skewness along time axis."""
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True)
        skewness = np.mean(((data - mean) / (std + 1e-8)) ** 3, axis=0)
        return skewness
    
    def _compute_kurtosis(self, data: np.ndarray) -> np.ndarray:
        """Compute kurtosis along time axis."""
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True)
        kurtosis = np.mean(((data - mean) / (std + 1e-8)) ** 4, axis=0)
        return kurtosis
    
    def preprocess(
        self, 
        csi_complex: np.ndarray,
        extract_features: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Full preprocessing pipeline for CSI data.
        
        Args:
            csi_complex: Complex CSI data of shape (n_samples, n_subcarriers)
            extract_features: Whether to extract statistical features
            
        Returns:
            Dictionary containing preprocessed data and features
        """
        # Extract amplitude and phase
        amplitude, phase = self.extract_amplitude_phase(csi_complex)
        
        # Apply filtering
        amplitude_filtered = self.apply_bandpass_filter(amplitude)
        phase_filtered = self.apply_bandpass_filter(phase)
        
        # Compute PSD
        freq_amp, psd_amp = self.compute_psd(amplitude_filtered)
        freq_phase, psd_phase = self.compute_psd(phase_filtered)
        
        result = {
            'amplitude': amplitude_filtered,
            'phase': phase_filtered,
            'psd_amplitude': psd_amp,
            'psd_phase': psd_phase,
            'frequencies': freq_amp
        }
        
        # Extract features if requested
        if extract_features:
            amp_features = self.extract_statistical_features(amplitude_filtered)
            phase_features = self.extract_statistical_features(phase_filtered)
            
            result['amplitude_features'] = amp_features
            result['phase_features'] = phase_features
        
        return result
