"""
Spectral-CSI: Privacy-Preserving Smart Building Occupancy Estimation

A Bayesian Deep Learning framework for non-intrusive indoor occupancy estimation 
using Wi-Fi Channel State Information (CSI) and Power Spectral Density (PSD) analysis.
"""

__version__ = "0.1.0"
__author__ = "Spectral-CSI Research Team"

from .models import BayesianOccupancyEstimator
from .preprocessing import CSIPreprocessor
from .optimization import UncertaintyAwareOptimizer

__all__ = [
    "BayesianOccupancyEstimator",
    "CSIPreprocessor", 
    "UncertaintyAwareOptimizer",
]
