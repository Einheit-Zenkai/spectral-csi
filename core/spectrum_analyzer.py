"""
Spectrum Analyzer — CSI Spectral Feature Extraction
=====================================================
Converts raw CSI amplitude matrices into spectral features used by
downstream classification (hypothesis test + Bayesian model).

Key outputs
-----------
- **STFT spectrogram**  — time × frequency × amplitude image
- **Power Spectral Density (PSD)** — per-subcarrier power spectrum
- **Respiration band energy** — integrated power in 0.2–0.5 Hz

Theory
------
We treat each subcarrier's time series as a realisation of a wide-sense
stationary stochastic process.  The Wiener–Khinchin theorem lets us
estimate PSD from the STFT magnitude squared.

Human breathing at 0.2–0.5 Hz introduces a spectral peak absent in
empty-room signals — that peak is the primary detection feature.
"""

from __future__ import annotations

import numpy as np
from scipy import signal as sig
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class SpectrumConfig:
    """Parameters for spectral analysis."""

    # STFT settings
    window: str = "hann"
    nperseg: int = 256          # FFT window length (samples)
    noverlap: int | None = None # Defaults to nperseg // 2
    nfft: int | None = None     # Zero-pad FFT to this length
    psd_nperseg: int | None = None  # Separate setting for Welch PSD (auto-sized)

    # Respiration band (Hz)
    resp_low: float = 0.2
    resp_high: float = 0.5

    # Wavelet denoising
    denoise: bool = True
    wavelet: str = "db4"
    wavelet_level: int = 4


# ---------------------------------------------------------------------------
# Spectrum Analyzer
# ---------------------------------------------------------------------------
class SpectrumAnalyzer:
    """
    Extracts spectral features from a CSI amplitude matrix.

    Parameters
    ----------
    sample_rate : float
        Packet rate in Hz (must match the CSI capture rate).
    config : SpectrumConfig, optional
        Tuning knobs for STFT / denoising.

    Usage
    -----
    >>> from core.spectrum_analyzer import SpectrumAnalyzer
    >>> sa = SpectrumAnalyzer(sample_rate=100.0)
    >>> features = sa.extract_features(csi_matrix)
    """

    def __init__(self, sample_rate: float = 100.0,
                 config: SpectrumConfig | None = None):
        self.fs = sample_rate
        self.cfg = config or SpectrumConfig()

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def extract_features(self, csi: np.ndarray) -> dict:
        """
        Full feature extraction pipeline.

        Parameters
        ----------
        csi : ndarray of shape (n_samples, n_subcarriers)
            Raw CSI amplitude time series.

        Returns
        -------
        dict with keys:
            spectrogram       — (n_freqs, n_times, n_subcarriers) STFT magnitudes
            frequencies       — 1-D frequency axis (Hz)
            times             — 1-D time axis (s)
            psd               — (n_freqs_psd, n_subcarriers) Welch PSD estimates
            psd_freqs         — 1-D frequency axis for PSD
            resp_band_energy  — (n_subcarriers,) integrated power in 0.2–0.5 Hz
            mean_resp_energy  — scalar, average respiration energy across subcarriers
            variance_profile  — (n_subcarriers,) temporal variance per subcarrier
        """
        if self.cfg.denoise:
            csi = self._wavelet_denoise(csi)

        csi = self._remove_outliers(csi)

        spec_f, spec_t, spec_mag = self.compute_stft(csi)
        psd_f, psd = self.compute_psd(csi)
        resp_energy = self.respiration_band_energy(psd_f, psd)

        return {
            "spectrogram": spec_mag,
            "frequencies": spec_f,
            "times": spec_t,
            "psd": psd,
            "psd_freqs": psd_f,
            "resp_band_energy": resp_energy,
            "mean_resp_energy": float(np.mean(resp_energy)),
            "variance_profile": np.var(csi, axis=0),
        }

    # ------------------------------------------------------------------ #
    #  STFT Spectrogram                                                    #
    # ------------------------------------------------------------------ #

    def compute_stft(
        self, csi: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute Short-Time Fourier Transform for every subcarrier.

        Returns
        -------
        frequencies : (n_freqs,)
        times       : (n_times,)
        magnitude   : (n_freqs, n_times, n_subcarriers)
        """
        cfg = self.cfg
        n_samples, n_sub = csi.shape
        noverlap = cfg.noverlap if cfg.noverlap is not None else cfg.nperseg // 2

        # Compute on first subcarrier to get output shape
        f, t, Zxx0 = sig.stft(
            csi[:, 0], fs=self.fs,
            window=cfg.window, nperseg=cfg.nperseg,
            noverlap=noverlap, nfft=cfg.nfft,
        )

        mag = np.empty((len(f), len(t), n_sub), dtype=np.float64)
        mag[:, :, 0] = np.abs(Zxx0)

        for i in range(1, n_sub):
            _, _, Zxx = sig.stft(
                csi[:, i], fs=self.fs,
                window=cfg.window, nperseg=cfg.nperseg,
                noverlap=noverlap, nfft=cfg.nfft,
            )
            mag[:, :, i] = np.abs(Zxx)

        return f, t, mag

    # ------------------------------------------------------------------ #
    #  Power Spectral Density (Welch)                                      #
    # ------------------------------------------------------------------ #

    def compute_psd(self, csi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Estimate Power Spectral Density per subcarrier using Welch's method.

        Returns
        -------
        freqs : (n_freqs,)
        psd   : (n_freqs, n_subcarriers)
        """
        cfg = self.cfg
        n_samples, n_sub = csi.shape
        noverlap = cfg.noverlap if cfg.noverlap is not None else cfg.nperseg // 2

        # For PSD we want enough frequency resolution to see 0.2-0.5 Hz
        # At 100 Hz sample rate, nperseg=1024 gives df ≈ 0.1 Hz — good enough
        psd_nperseg = cfg.psd_nperseg or min(csi.shape[0], max(cfg.nperseg, 1024))
        psd_noverlap = psd_nperseg // 2

        f, Pxx0 = sig.welch(
            csi[:, 0], fs=self.fs,
            window=cfg.window, nperseg=psd_nperseg,
            noverlap=psd_noverlap, nfft=cfg.nfft,
        )

        psd = np.empty((len(f), n_sub), dtype=np.float64)
        psd[:, 0] = Pxx0

        for i in range(1, n_sub):
            _, Pxx = sig.welch(
                csi[:, i], fs=self.fs,
                window=cfg.window, nperseg=psd_nperseg,
                noverlap=psd_noverlap, nfft=cfg.nfft,
            )
            psd[:, i] = Pxx

        return f, psd

    # ------------------------------------------------------------------ #
    #  Respiration Band Energy                                             #
    # ------------------------------------------------------------------ #

    def respiration_band_energy(
        self, freqs: np.ndarray, psd: np.ndarray
    ) -> np.ndarray:
        """
        Integrate PSD over the respiration band [0.2, 0.5] Hz.

        Returns
        -------
        energy : (n_subcarriers,) — total power in respiration band
        """
        cfg = self.cfg
        mask = (freqs >= cfg.resp_low) & (freqs <= cfg.resp_high)
        df = freqs[1] - freqs[0] if len(freqs) > 1 else 1.0
        # Trapezoidal integration over the band
        return np.trapezoid(psd[mask, :], dx=df, axis=0)

    # ------------------------------------------------------------------ #
    #  Preprocessing helpers                                               #
    # ------------------------------------------------------------------ #

    def _wavelet_denoise(self, csi: np.ndarray) -> np.ndarray:
        """
        Denoise each subcarrier using Discrete Wavelet Transform (DWT).

        Uses soft thresholding at each decomposition level with the
        universal threshold  σ√(2 ln N).
        """
        import pywt

        denoised = np.empty_like(csi)
        for i in range(csi.shape[1]):
            coeffs = pywt.wavedec(csi[:, i], self.cfg.wavelet,
                                  level=self.cfg.wavelet_level)
            # Estimate noise σ from finest detail coefficients
            sigma = np.median(np.abs(coeffs[-1])) / 0.6745
            threshold = sigma * np.sqrt(2 * np.log(len(csi[:, i])))

            # Soft-threshold all detail levels
            new_coeffs = [coeffs[0]]  # keep approximation
            for c in coeffs[1:]:
                new_coeffs.append(pywt.threshold(c, value=threshold, mode="soft"))

            denoised[:, i] = pywt.waverec(new_coeffs, self.cfg.wavelet)[: csi.shape[0]]

        return denoised

    def _remove_outliers(self, csi: np.ndarray, k: float = 3.0) -> np.ndarray:
        """
        Remove outliers using Chebyshev's inequality:
        P(|X - μ| ≥ kσ) ≤ 1/k², so k=3 → ≤ 11.1% false removal.

        Values beyond μ ± kσ are clipped.
        """
        mu = np.mean(csi, axis=0, keepdims=True)
        sigma = np.std(csi, axis=0, keepdims=True)
        return np.clip(csi, mu - k * sigma, mu + k * sigma)

    # ------------------------------------------------------------------ #
    #  Convenience — spectrogram image for CNN input                       #
    # ------------------------------------------------------------------ #

    def to_spectrogram_image(
        self, csi: np.ndarray, target_shape: tuple[int, int] = (224, 224)
    ) -> np.ndarray:
        """
        Convert CSI to a single-channel spectrogram image suitable for
        a ResNet-style CNN.

        Averages the STFT across subcarriers, resizes to `target_shape`,
        and normalises to [0, 1].

        Returns
        -------
        image : (1, H, W)  — single-channel float32 image
        """
        from scipy.ndimage import zoom

        _, _, mag = self.compute_stft(csi)
        # Average over subcarriers → (n_freqs, n_times)
        avg = mag.mean(axis=2)

        # Log-scale for better dynamic range
        avg = np.log1p(avg)

        # Resize to target
        zoom_factors = (
            target_shape[0] / avg.shape[0],
            target_shape[1] / avg.shape[1],
        )
        resized = zoom(avg, zoom_factors, order=1)

        # Normalise to [0, 1]
        rmin, rmax = resized.min(), resized.max()
        if rmax - rmin > 1e-8:
            resized = (resized - rmin) / (rmax - rmin)

        return resized[np.newaxis].astype(np.float32)  # (1, H, W)


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from data.synthetic_csi import SyntheticCSIGenerator

    gen = SyntheticCSIGenerator()
    sa = SpectrumAnalyzer(sample_rate=100.0)

    print("=== Occupied Room ===")
    occ = gen.generate(occupied=True)
    feat_occ = sa.extract_features(occ)
    print(f"  Respiration band energy (mean): {feat_occ['mean_resp_energy']:.6f}")
    print(f"  Variance profile (mean):        {feat_occ['variance_profile'].mean():.6f}")

    print("\n=== Empty Room ===")
    emp = gen.generate(occupied=False)
    feat_emp = sa.extract_features(emp)
    print(f"  Respiration band energy (mean): {feat_emp['mean_resp_energy']:.6f}")
    print(f"  Variance profile (mean):        {feat_emp['variance_profile'].mean():.6f}")

    ratio = feat_occ["mean_resp_energy"] / max(feat_emp["mean_resp_energy"], 1e-12)
    print(f"\n  Resp energy ratio (occ/emp): {ratio:.2f}x")
