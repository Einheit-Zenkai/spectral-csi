# Spectral-CSI Core Module
# Signal processing, statistical testing, and Bayesian inference for WiFi CSI occupancy detection.

# Lazy imports — avoids requiring torch/torchvision just to use the signal-processing modules.


def __getattr__(name: str):
    if name == "SpectrumAnalyzer":
        from .spectrum_analyzer import SpectrumAnalyzer
        return SpectrumAnalyzer
    if name == "OccupancyHypothesisTest":
        from .hypothesis_test import OccupancyHypothesisTest
        return OccupancyHypothesisTest
    if name == "BayesianOccupancyNet":
        from .bayesian_model import BayesianOccupancyNet
        return BayesianOccupancyNet
    raise AttributeError(f"module 'core' has no attribute {name!r}")


__all__ = [
    "SpectrumAnalyzer",
    "OccupancyHypothesisTest",
    "BayesianOccupancyNet",
]
