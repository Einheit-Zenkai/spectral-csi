"""
Synthetic CSI Data Generator
=============================
Generates realistic WiFi Channel State Information (CSI) streams for
development and testing *without* requiring ESP32 / Intel 5300 hardware.

The model:
    H(f, t) = H_static(f) + H_human(f, t) + N(f, t)

where:
    H_static  — fixed multipath channel (Rayleigh fading baseline)
    H_human   — perturbation from human presence (breathing, micro-motion)
    N         — additive white Gaussian noise

Each "subcarrier" is one frequency bin of the OFDM channel (like real WiFi).
We simulate `n_subcarriers` bins over `duration` seconds at `sample_rate` Hz.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class CSIConfig:
    """Parameters for synthetic CSI generation."""

    n_subcarriers: int = 52          # 802.11n uses 52 data subcarriers (20 MHz)
    sample_rate: float = 100.0       # Packets per second (Hz)
    duration: float = 30.0           # Seconds of data to generate
    noise_power: float = 0.02        # Variance of AWGN per subcarrier

    # Human presence parameters
    breathing_rate: float = 0.3      # Hz  (≈18 breaths/min, normal adult)
    breathing_amplitude: float = 0.15 # Amplitude modulation depth
    heartbeat_rate: float = 1.2      # Hz  (≈72 bpm)
    heartbeat_amplitude: float = 0.03 # Much weaker than breathing
    micro_motion_std: float = 0.05   # Slow random drift from fidgeting

    # Static channel
    rayleigh_scale: float = 1.0      # Scale of Rayleigh fading envelope

    seed: int | None = 42            # Reproducibility


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------
class SyntheticCSIGenerator:
    """
    Produces amplitude-only CSI matrices of shape (n_samples, n_subcarriers).

    Usage
    -----
    >>> gen = SyntheticCSIGenerator()
    >>> occupied = gen.generate(occupied=True)
    >>> empty    = gen.generate(occupied=False)
    >>> print(occupied.shape)   # (3000, 52)
    """

    def __init__(self, config: CSIConfig | None = None):
        self.cfg = config or CSIConfig()
        self.rng = np.random.default_rng(self.cfg.seed)

    # -- public API ---------------------------------------------------------
    def generate(self, occupied: bool = True) -> np.ndarray:
        """Return CSI amplitude matrix (n_samples, n_subcarriers)."""
        cfg = self.cfg
        n_samples = int(cfg.duration * cfg.sample_rate)
        t = np.arange(n_samples) / cfg.sample_rate  # time vector

        # 1) Static multipath channel (Rayleigh envelope per subcarrier)
        h_static = self.rng.rayleigh(
            cfg.rayleigh_scale, size=(1, cfg.n_subcarriers)
        )
        csi = np.tile(h_static, (n_samples, 1))      # broadcast to all time steps

        # 2) Human perturbation (only if occupied)
        if occupied:
            csi += self._breathing_signal(t, cfg)
            csi += self._heartbeat_signal(t, cfg)
            csi += self._micro_motion(n_samples, cfg)

        # 3) Additive noise
        csi += self.rng.normal(0, np.sqrt(cfg.noise_power),
                               size=(n_samples, cfg.n_subcarriers))

        # Amplitude is non-negative
        csi = np.abs(csi)
        return csi

    def generate_labelled(
        self,
        n_occupied: int = 20,
        n_empty: int = 20,
        duration: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate a labelled dataset of CSI windows.

        Returns
        -------
        X : ndarray of shape (n_occupied + n_empty, n_samples, n_subcarriers)
        y : ndarray of shape (n_occupied + n_empty,)   — 1=occupied, 0=empty
        """
        original_dur = self.cfg.duration
        if duration is not None:
            self.cfg.duration = duration

        windows, labels = [], []
        for _ in range(n_occupied):
            # Slightly randomise breathing rate each window for variety
            self.cfg.breathing_rate = self.rng.uniform(0.2, 0.5)
            windows.append(self.generate(occupied=True))
            labels.append(1)

        for _ in range(n_empty):
            windows.append(self.generate(occupied=False))
            labels.append(0)

        self.cfg.duration = original_dur
        return np.array(windows), np.array(labels)

    # -- internal signals ---------------------------------------------------
    def _breathing_signal(self, t: np.ndarray, cfg: CSIConfig) -> np.ndarray:
        """Sinusoidal amplitude modulation at breathing frequency."""
        phase = self.rng.uniform(0, 2 * np.pi, size=(1, cfg.n_subcarriers))
        # Each subcarrier gets a slightly different phase (realistic multipath)
        signal = cfg.breathing_amplitude * np.sin(
            2 * np.pi * cfg.breathing_rate * t[:, None] + phase
        )
        return signal

    def _heartbeat_signal(self, t: np.ndarray, cfg: CSIConfig) -> np.ndarray:
        """Weak heartbeat modulation — mainly for completeness."""
        phase = self.rng.uniform(0, 2 * np.pi, size=(1, cfg.n_subcarriers))
        return cfg.heartbeat_amplitude * np.sin(
            2 * np.pi * cfg.heartbeat_rate * t[:, None] + phase
        )

    def _micro_motion(self, n_samples: int, cfg: CSIConfig) -> np.ndarray:
        """Slow random walk simulating small body shifts / fidgeting."""
        drift = self.rng.normal(0, cfg.micro_motion_std,
                                size=(n_samples, cfg.n_subcarriers))
        # Low-pass via cumulative sum + scaling
        drift = np.cumsum(drift, axis=0) / np.sqrt(n_samples)
        return drift


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    gen = SyntheticCSIGenerator()
    occ = gen.generate(occupied=True)
    emp = gen.generate(occupied=False)
    print(f"Occupied CSI shape : {occ.shape}  mean={occ.mean():.4f}  std={occ.std():.4f}")
    print(f"Empty    CSI shape : {emp.shape}  mean={emp.mean():.4f}  std={emp.std():.4f}")
