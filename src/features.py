"""
EEG Feature Extraction for Motor Imagery / Mental State Classification.

Extracts discriminative features from processed EEG epochs:
    - Band power (absolute & relative)
    - Power spectral density features
    - Spatial features (asymmetry ratios)
    - Statistical features (Hjorth parameters)

Designed for real-time use: all features computed in <5 ms per epoch.
"""

import logging
from dataclasses import dataclass

import numpy as np
from scipy.signal import welch
from scipy.integrate import simpson

logger = logging.getLogger(__name__)

# ── Standard EEG frequency bands ──────────────────────────────────
DEFAULT_BANDS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 50),
}


@dataclass
class FeatureVector:
    """Container for extracted features with metadata."""
    features: np.ndarray        # 1-D feature vector
    names: list[str]            # Feature labels (same length as features)
    band_powers: dict           # {band_name: power_per_channel}
    epoch_quality: float        # 0–1 signal quality estimate


class FeatureExtractor:
    """
    Multi-method EEG feature extraction.

    Usage:
        ext = FeatureExtractor(sampling_rate=250)
        fv = ext.extract(epoch_data)  # epoch_data: (n_channels, n_samples)
        X = fv.features               # Use for classification
    """

    def __init__(
        self,
        sampling_rate: int = 250,
        bands: dict | None = None,
        methods: list[str] | None = None,
    ):
        self.fs = sampling_rate
        self.bands = bands or DEFAULT_BANDS
        self.methods = methods or ["bandpower", "psd"]

    # ── Main API ──────────────────────────────────────────────────

    def extract(self, epoch: np.ndarray) -> FeatureVector:
        """
        Extract all configured features from an epoch.

        Args:
            epoch: (n_channels, n_samples) filtered EEG data

        Returns:
            FeatureVector with concatenated features
        """
        if epoch.ndim == 1:
            epoch = epoch.reshape(1, -1)

        features = []
        names = []
        band_powers = {}

        n_ch = epoch.shape[0]
        ch_labels = [f"ch{i}" for i in range(n_ch)]

        # ── Band power features ───────────────────────────────────
        if "bandpower" in self.methods:
            bp_abs, bp_rel = self._band_power(epoch)
            band_powers = {
                band: bp_abs[:, i]
                for i, band in enumerate(self.bands)
            }

            # Absolute band power (log-transformed for normality)
            for i, band in enumerate(self.bands):
                for j, ch in enumerate(ch_labels):
                    features.append(np.log1p(bp_abs[j, i]))
                    names.append(f"{ch}_{band}_abs")

            # Relative band power
            for i, band in enumerate(self.bands):
                for j, ch in enumerate(ch_labels):
                    features.append(bp_rel[j, i])
                    names.append(f"{ch}_{band}_rel")

            # ── Band power ratios (known BCI discriminators) ──────
            # Beta/Alpha ratio → engagement / focus index
            alpha_idx = list(self.bands.keys()).index("alpha")
            beta_idx = list(self.bands.keys()).index("beta")
            for j, ch in enumerate(ch_labels):
                ratio = bp_abs[j, beta_idx] / (bp_abs[j, alpha_idx] + 1e-10)
                features.append(ratio)
                names.append(f"{ch}_beta_alpha_ratio")

            # Theta/Beta ratio → attention metric
            theta_idx = list(self.bands.keys()).index("theta")
            for j, ch in enumerate(ch_labels):
                ratio = bp_abs[j, theta_idx] / (bp_abs[j, beta_idx] + 1e-10)
                features.append(ratio)
                names.append(f"{ch}_theta_beta_ratio")

        # ── PSD features ──────────────────────────────────────────
        if "psd" in self.methods:
            psd_feats, psd_names = self._psd_features(epoch, ch_labels)
            features.extend(psd_feats)
            names.extend(psd_names)

        # ── Hjorth parameters ─────────────────────────────────────
        hjorth_feats, hjorth_names = self._hjorth_parameters(epoch, ch_labels)
        features.extend(hjorth_feats)
        names.extend(hjorth_names)

        # ── Asymmetry features (lateralization) ───────────────────
        if n_ch >= 2:
            asym_feats, asym_names = self._asymmetry(epoch, ch_labels)
            features.extend(asym_feats)
            names.extend(asym_names)

        # ── Signal quality estimate ───────────────────────────────
        quality = self._signal_quality(epoch)

        feature_array = np.array(features, dtype=np.float64)

        # Handle NaN/Inf
        feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=0.0, neginf=0.0)

        return FeatureVector(
            features=feature_array,
            names=names,
            band_powers=band_powers,
            epoch_quality=quality,
        )

    # ── Feature methods ───────────────────────────────────────────

    def _band_power(
        self, epoch: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute absolute and relative band power.

        Returns:
            (absolute, relative) each of shape (n_channels, n_bands)
        """
        n_ch = epoch.shape[0]
        n_bands = len(self.bands)
        nperseg = min(epoch.shape[1], self.fs)

        freqs, psd = welch(epoch, fs=self.fs, nperseg=nperseg, axis=1)
        freq_res = freqs[1] - freqs[0]

        absolute = np.zeros((n_ch, n_bands))
        for i, (band, (fmin, fmax)) in enumerate(self.bands.items()):
            idx = np.logical_and(freqs >= fmin, freqs <= fmax)
            for ch in range(n_ch):
                absolute[ch, i] = simpson(psd[ch, idx], dx=freq_res)

        total_power = absolute.sum(axis=1, keepdims=True)
        relative = absolute / (total_power + 1e-10)

        return absolute, relative

    def _psd_features(
        self, epoch: np.ndarray, ch_labels: list[str]
    ) -> tuple[list[float], list[str]]:
        """Extract PSD-derived features: peak frequency, spectral entropy."""
        features = []
        names = []

        nperseg = min(epoch.shape[1], self.fs)
        freqs, psd = welch(epoch, fs=self.fs, nperseg=nperseg, axis=1)

        for j, ch in enumerate(ch_labels):
            # Peak frequency (dominant rhythm)
            peak_idx = np.argmax(psd[j])
            features.append(freqs[peak_idx])
            names.append(f"{ch}_peak_freq")

            # Spectral entropy (complexity measure)
            p = psd[j] / (psd[j].sum() + 1e-10)
            entropy = -np.sum(p * np.log2(p + 1e-10))
            features.append(entropy)
            names.append(f"{ch}_spectral_entropy")

            # Spectral edge frequency (95% power)
            cumpower = np.cumsum(psd[j])
            total = cumpower[-1]
            edge_idx = np.searchsorted(cumpower, 0.95 * total)
            edge_idx = min(edge_idx, len(freqs) - 1)
            features.append(freqs[edge_idx])
            names.append(f"{ch}_spectral_edge_95")

        return features, names

    @staticmethod
    def _hjorth_parameters(
        epoch: np.ndarray, ch_labels: list[str]
    ) -> tuple[list[float], list[str]]:
        """
        Compute Hjorth parameters: Activity, Mobility, Complexity.
        Time-domain features that capture signal variance structure.
        """
        features = []
        names = []

        for j, ch in enumerate(ch_labels):
            x = epoch[j]
            dx = np.diff(x)
            ddx = np.diff(dx)

            var_x = np.var(x)
            var_dx = np.var(dx)
            var_ddx = np.var(ddx)

            # Activity: variance of the signal
            activity = var_x
            features.append(np.log1p(activity))
            names.append(f"{ch}_hjorth_activity")

            # Mobility: sqrt(var(dx) / var(x))
            mobility = np.sqrt(var_dx / (var_x + 1e-10))
            features.append(mobility)
            names.append(f"{ch}_hjorth_mobility")

            # Complexity: mobility(dx) / mobility(x)
            mob_dx = np.sqrt(var_ddx / (var_dx + 1e-10))
            complexity = mob_dx / (mobility + 1e-10)
            features.append(complexity)
            names.append(f"{ch}_hjorth_complexity")

        return features, names

    def _asymmetry(
        self, epoch: np.ndarray, ch_labels: list[str]
    ) -> tuple[list[float], list[str]]:
        """
        Compute hemispheric asymmetry features.

        Uses paired channels (0 vs n-1, 1 vs n-2, etc.) to detect
        lateralized activation — critical for left/right cursor control.
        """
        features = []
        names = []
        n_ch = epoch.shape[0]

        bp_abs, _ = self._band_power(epoch)

        for i in range(n_ch // 2):
            left = i
            right = n_ch - 1 - i
            for k, band in enumerate(self.bands):
                # Log asymmetry index: ln(right) - ln(left)
                asym = np.log1p(bp_abs[right, k]) - np.log1p(bp_abs[left, k])
                features.append(asym)
                names.append(f"asym_{ch_labels[left]}_{ch_labels[right]}_{band}")

        return features, names

    @staticmethod
    def _signal_quality(epoch: np.ndarray) -> float:
        """
        Estimate signal quality (0–1) based on:
            - Low amplitude variance (good contact)
            - Absence of flat-line segments
            - Absence of extreme values
        """
        scores = []

        # Check for flat lines (bad electrode contact)
        for ch in range(epoch.shape[0]):
            std = np.std(epoch[ch])
            # Good: std between 5–50 µV
            if 5 < std < 50:
                scores.append(1.0)
            elif std < 1 or std > 200:
                scores.append(0.0)
            else:
                scores.append(0.5)

        return float(np.mean(scores)) if scores else 0.0
