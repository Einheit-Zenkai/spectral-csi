"""
Hypothesis Test — Statistical Empty-Room Decision
===================================================
Implements the "Zero-False-Negative" decision logic from the interim report.

Decision framework
------------------
    H0 (Null):       Room is EMPTY  — signal variance ≈ noise floor
    H1 (Alternate):  Room is OCCUPIED — signal contains human signature

We require:
    P(Empty | observations) > 99.9%   before we turn lights OFF.

This module provides:
1. **Variance-ratio test** — compares observed variance to a noise-floor baseline.
2. **Respiration energy test** — checks if PSD energy in 0.2–0.5 Hz is significant.
3. **Combined decision** — both tests must agree for "empty" verdict.
4. **Chebyshev confidence bounds** — distribution-free intervals.
5. **Normal Z-test** — parametric alternative when Gaussian assumption holds.

All methods output a probability + decision, enabling downstream Bayesian
smoothing and fail-safe control.
"""

from __future__ import annotations

import numpy as np
from scipy import stats
from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
@dataclass
class HypothesisConfig:
    """Tunable parameters for the statistical decision module."""

    # Confidence threshold — lights go OFF only if P(empty) exceeds this
    confidence_threshold: float = 0.999   # 99.9%

    # Significance level for individual tests (alpha)
    alpha: float = 0.001                  # corresponds to 99.9%

    # Noise floor (estimated from calibration or empty-room reference)
    # Set via calibrate() or manually.
    noise_variance: float | None = None
    noise_resp_energy: float | None = None

    # Chebyshev k — number of standard deviations for outlier bounds
    chebyshev_k: float = 3.0

    # Minimum calibration samples
    min_calibration_windows: int = 5


# ---------------------------------------------------------------------------
# Hypothesis Test Engine
# ---------------------------------------------------------------------------
class OccupancyHypothesisTest:
    """
    Statistical decision module for occupancy detection.

    Typical workflow
    ----------------
    1. **Calibrate** on known-empty-room data to learn the noise floor.
    2. **Test** incoming CSI feature dicts (from SpectrumAnalyzer).
    3. Use `decision` dict to drive lights / HVAC.

    >>> from core.hypothesis_test import OccupancyHypothesisTest
    >>> ht = OccupancyHypothesisTest()
    >>> ht.calibrate(list_of_empty_features)
    >>> result = ht.test(features)
    >>> print(result["decision"])  # "empty" or "occupied"
    """

    def __init__(self, config: HypothesisConfig | None = None):
        self.cfg = config or HypothesisConfig()
        self._calibrated = False

    # ------------------------------------------------------------------ #
    #  Calibration (learn the noise floor from empty-room recordings)      #
    # ------------------------------------------------------------------ #

    def calibrate(self, empty_features: list[dict]) -> dict:
        """
        Learn noise-floor statistics from a set of *known-empty* feature dicts
        (each produced by SpectrumAnalyzer.extract_features).

        Parameters
        ----------
        empty_features : list of feature dicts (at least 5 recommended)

        Returns
        -------
        dict with calibrated noise_variance and noise_resp_energy.
        """
        if len(empty_features) < self.cfg.min_calibration_windows:
            raise ValueError(
                f"Need at least {self.cfg.min_calibration_windows} empty-room "
                f"windows for calibration, got {len(empty_features)}."
            )

        variances = [np.mean(f["variance_profile"]) for f in empty_features]
        resp_energies = [f["mean_resp_energy"] for f in empty_features]

        # Store mean + std so we can build a distribution
        self._noise_var_mu = float(np.mean(variances))
        self._noise_var_std = float(np.std(variances, ddof=1))
        self._noise_resp_mu = float(np.mean(resp_energies))
        self._noise_resp_std = float(np.std(resp_energies, ddof=1))

        # Convenience — set point estimates in config too
        self.cfg.noise_variance = self._noise_var_mu
        self.cfg.noise_resp_energy = self._noise_resp_mu

        self._calibrated = True

        return {
            "noise_variance_mean": self._noise_var_mu,
            "noise_variance_std": self._noise_var_std,
            "noise_resp_energy_mean": self._noise_resp_mu,
            "noise_resp_energy_std": self._noise_resp_std,
        }

    # ------------------------------------------------------------------ #
    #  Main test entry point                                               #
    # ------------------------------------------------------------------ #

    def test(self, features: dict) -> dict:
        """
        Run the combined hypothesis test on a single feature dict.

        Returns
        -------
        dict with:
            variance_p         — p-value from variance ratio test
            resp_energy_p      — p-value from respiration energy test
            p_empty            — combined probability of empty room
            p_occupied         — 1 - p_empty
            decision           — "empty" or "occupied"
            confidence         — the p_empty value used for the decision
            chebyshev_bound    — Chebyshev upper-bound on false-alarm rate
        """
        self._ensure_calibrated()

        var_result = self._variance_test(features)
        resp_result = self._respiration_test(features)

        # Combined probability: both tests must indicate empty.
        # We use Fisher's method (product of p-values → chi-squared).
        # For safety, we take the *minimum* p-value as the combined
        # confidence — conservative approach (both must agree).
        p_empty = min(var_result["p_empty"], resp_result["p_empty"])
        p_occupied = 1.0 - p_empty

        decision = "empty" if p_empty > self.cfg.confidence_threshold else "occupied"

        # Chebyshev bound: P(|X-μ| ≥ kσ) ≤ 1/k²
        k = self.cfg.chebyshev_k
        chebyshev_false_alarm = 1.0 / (k ** 2)

        return {
            "variance_p": var_result["p_value"],
            "resp_energy_p": resp_result["p_value"],
            "p_empty": p_empty,
            "p_occupied": p_occupied,
            "decision": decision,
            "confidence": p_empty,
            "chebyshev_bound": chebyshev_false_alarm,
            "details": {
                "variance_test": var_result,
                "respiration_test": resp_result,
            },
        }

    # ------------------------------------------------------------------ #
    #  Individual tests                                                    #
    # ------------------------------------------------------------------ #

    def _variance_test(self, features: dict) -> dict:
        """
        One-sided Z-test: is the observed variance significantly above baseline?

        H0: μ_var = noise_var (empty)
        H1: μ_var > noise_var (occupied — human presence adds variance)

        If observed variance is NOT significantly above baseline → room likely empty.
        """
        observed = float(np.mean(features["variance_profile"]))

        # Z-score relative to noise distribution
        if self._noise_var_std < 1e-12:
            # Degenerate case — no variation in calibration
            z = (observed - self._noise_var_mu) / 1e-6
        else:
            z = (observed - self._noise_var_mu) / self._noise_var_std

        # One-sided: probability that the noise floor is ≥ observed
        # If z ≈ 0: observed ≈ noise → probably empty
        # If z >> 0: observed >> noise → probably occupied
        p_value = 1.0 - stats.norm.cdf(z)   # P(X ≥ observed | H0)

        # p_value is high when observed ≈ noise (empty)
        # p_value is low when observed >> noise (occupied)
        p_empty = p_value   # high p → consistent with empty

        return {
            "observed_variance": observed,
            "noise_baseline": self._noise_var_mu,
            "z_score": z,
            "p_value": p_value,
            "p_empty": p_empty,
        }

    def _respiration_test(self, features: dict) -> dict:
        """
        One-sided Z-test on respiration band energy.

        H0: resp_energy = noise_resp (empty — no breathing signature)
        H1: resp_energy > noise_resp (occupied — breathing peak present)
        """
        observed = features["mean_resp_energy"]

        if self._noise_resp_std < 1e-12:
            z = (observed - self._noise_resp_mu) / 1e-6
        else:
            z = (observed - self._noise_resp_mu) / self._noise_resp_std

        p_value = 1.0 - stats.norm.cdf(z)
        p_empty = p_value

        return {
            "observed_resp_energy": observed,
            "noise_baseline": self._noise_resp_mu,
            "z_score": z,
            "p_value": p_value,
            "p_empty": p_empty,
        }

    # ------------------------------------------------------------------ #
    #  Utility: Chebyshev Confidence Interval                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def chebyshev_interval(
        data: np.ndarray, k: float = 3.0
    ) -> tuple[float, float, float]:
        """
        Distribution-free confidence interval using Chebyshev's inequality.

        P(|X - μ| < kσ) ≥ 1 - 1/k²

        Returns (lower, upper, coverage_probability).
        """
        mu = float(np.mean(data))
        sigma = float(np.std(data, ddof=1))
        coverage = 1.0 - 1.0 / (k ** 2)
        return mu - k * sigma, mu + k * sigma, coverage

    # ------------------------------------------------------------------ #
    #  Utility: Normal Confidence Interval                                 #
    # ------------------------------------------------------------------ #

    @staticmethod
    def normal_ci(
        data: np.ndarray, confidence: float = 0.999
    ) -> tuple[float, float]:
        """
        Parametric confidence interval assuming Gaussian distribution.

        Returns (lower, upper).
        """
        n = len(data)
        mu = float(np.mean(data))
        se = float(stats.sem(data))
        h = se * stats.t.ppf((1 + confidence) / 2, n - 1)
        return mu - h, mu + h

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _ensure_calibrated(self):
        if not self._calibrated:
            raise RuntimeError(
                "Hypothesis test is not calibrated. "
                "Call calibrate(empty_features) first."
            )


# ---------------------------------------------------------------------------
# Quick demo
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from data.synthetic_csi import SyntheticCSIGenerator
    from core.spectrum_analyzer import SpectrumAnalyzer

    gen = SyntheticCSIGenerator()
    sa = SpectrumAnalyzer(sample_rate=100.0)
    ht = OccupancyHypothesisTest()

    # --- Calibration: generate several empty-room windows ---
    print("Calibrating on 10 empty-room windows...")
    empty_features = []
    for _ in range(10):
        emp = gen.generate(occupied=False)
        empty_features.append(sa.extract_features(emp))
    cal = ht.calibrate(empty_features)
    print(f"  Noise variance baseline : {cal['noise_variance_mean']:.6f}")
    print(f"  Noise resp energy baseline: {cal['noise_resp_energy_mean']:.6f}")

    # --- Test: occupied room ---
    print("\n--- Testing OCCUPIED room ---")
    occ = gen.generate(occupied=True)
    occ_feat = sa.extract_features(occ)
    result = ht.test(occ_feat)
    print(f"  Decision   : {result['decision']}")
    print(f"  P(empty)   : {result['p_empty']:.6f}")
    print(f"  P(occupied): {result['p_occupied']:.6f}")

    # --- Test: empty room ---
    print("\n--- Testing EMPTY room ---")
    emp = gen.generate(occupied=False)
    emp_feat = sa.extract_features(emp)
    result = ht.test(emp_feat)
    print(f"  Decision   : {result['decision']}")
    print(f"  P(empty)   : {result['p_empty']:.6f}")
    print(f"  P(occupied): {result['p_occupied']:.6f}")

    # --- Chebyshev interval demo ---
    sample = np.array([f["mean_resp_energy"] for f in empty_features])
    lo, hi, cov = ht.chebyshev_interval(sample, k=3.0)
    print(f"\nChebyshev 3σ interval for empty resp energy: [{lo:.6f}, {hi:.6f}]  (coverage ≥ {cov:.2%})")
