"""
Real-time EEG signal processing pipeline.

Handles filtering, artifact rejection, and windowing for downstream
feature extraction. Designed for <10 ms latency per window.
"""

import logging
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, welch

logger = logging.getLogger(__name__)


@dataclass
class ProcessingConfig:
    """Signal processing parameters."""
    sampling_rate: int = 250
    notch_freq: float = 60.0
    bandpass_low: float = 1.0
    bandpass_high: float = 50.0
    artifact_threshold_uv: float = 100.0
    artifact_rejection_enabled: bool = True


@dataclass
class EpochResult:
    """Container for a processed EEG epoch."""
    data: np.ndarray                     # (n_channels, n_samples), filtered
    is_clean: bool = True                # False if artifact detected
    timestamp: float = 0.0
    artifact_channels: list[int] = field(default_factory=list)


class SignalProcessor:
    """
    Real-time EEG signal processing.

    Pipeline:
        1. DC offset removal (detrend)
        2. Notch filter (power line noise)
        3. Bandpass filter (1–50 Hz)
        4. Artifact rejection (amplitude threshold)
        5. Re-referencing (common average reference)

    Usage:
        proc = SignalProcessor(ProcessingConfig(sampling_rate=250))
        epoch = proc.process(raw_data)  # raw_data: (n_channels, n_samples)
    """

    def __init__(self, config: Optional[ProcessingConfig] = None):
        self.config = config or ProcessingConfig()
        self._build_filters()
        logger.info(
            "SignalProcessor ready: fs=%d, notch=%.1f, bp=[%.1f, %.1f]",
            self.config.sampling_rate,
            self.config.notch_freq,
            self.config.bandpass_low,
            self.config.bandpass_high,
        )

    # ── Public API ────────────────────────────────────────────────

    def process(self, raw: np.ndarray, timestamp: float = 0.0) -> EpochResult:
        """
        Full processing pipeline on a raw EEG epoch.

        Args:
            raw: np.ndarray of shape (n_channels, n_samples)
            timestamp: epoch timestamp for logging

        Returns:
            EpochResult with filtered data and artifact info
        """
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)

        data = raw.astype(np.float64).copy()

        # 1. DC removal
        data = self._detrend(data)

        # 2. Notch filter
        data = self._apply_notch(data)

        # 3. Bandpass filter
        data = self._apply_bandpass(data)

        # 4. Common average reference
        data = self._rereference(data)

        # 5. Artifact rejection
        is_clean, bad_channels = self._check_artifacts(data)

        return EpochResult(
            data=data,
            is_clean=is_clean,
            timestamp=timestamp,
            artifact_channels=bad_channels,
        )

    def compute_psd(
        self, data: np.ndarray, nperseg: int = 256
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute power spectral density via Welch's method.

        Args:
            data: (n_channels, n_samples)
            nperseg: FFT segment length

        Returns:
            (frequencies, psd) where psd shape is (n_channels, n_freqs)
        """
        nperseg = min(nperseg, data.shape[1])
        freqs, psd = welch(
            data,
            fs=self.config.sampling_rate,
            nperseg=nperseg,
            noverlap=nperseg // 2,
            axis=1,
        )
        return freqs, psd

    # ── Filter construction ───────────────────────────────────────

    def _build_filters(self) -> None:
        fs = self.config.sampling_rate
        nyq = fs / 2.0

        # Notch (Q=30 is standard for EEG)
        self._notch_b, self._notch_a = iirnotch(
            self.config.notch_freq, Q=30.0, fs=fs
        )

        # Bandpass (4th-order Butterworth)
        low = self.config.bandpass_low / nyq
        high = self.config.bandpass_high / nyq
        # Clamp to valid range
        low = max(low, 0.001)
        high = min(high, 0.999)
        self._bp_b, self._bp_a = butter(4, [low, high], btype="band")

    # ── Pipeline steps ────────────────────────────────────────────

    @staticmethod
    def _detrend(data: np.ndarray) -> np.ndarray:
        """Remove DC offset (mean) per channel."""
        return data - data.mean(axis=1, keepdims=True)

    def _apply_notch(self, data: np.ndarray) -> np.ndarray:
        """Apply notch filter to remove power-line interference."""
        # Minimum samples for filtfilt with this filter
        min_len = 3 * max(len(self._notch_b), len(self._notch_a))
        if data.shape[1] < min_len:
            return data
        return filtfilt(self._notch_b, self._notch_a, data, axis=1)

    def _apply_bandpass(self, data: np.ndarray) -> np.ndarray:
        """Apply bandpass filter."""
        min_len = 3 * max(len(self._bp_b), len(self._bp_a))
        if data.shape[1] < min_len:
            return data
        return filtfilt(self._bp_b, self._bp_a, data, axis=1)

    @staticmethod
    def _rereference(data: np.ndarray) -> np.ndarray:
        """Common average reference: subtract mean across channels."""
        if data.shape[0] > 1:
            return data - data.mean(axis=0, keepdims=True)
        return data

    def _check_artifacts(
        self, data: np.ndarray
    ) -> tuple[bool, list[int]]:
        """
        Reject epochs with peak amplitude exceeding threshold.

        Returns:
            (is_clean, list_of_bad_channel_indices)
        """
        if not self.config.artifact_rejection_enabled:
            return True, []

        threshold = self.config.artifact_threshold_uv
        peak_amplitudes = np.max(np.abs(data), axis=1)
        bad = np.where(peak_amplitudes > threshold)[0].tolist()

        is_clean = len(bad) == 0
        if not is_clean:
            logger.debug(
                "Artifact detected on channels %s (peaks: %s µV)",
                bad,
                peak_amplitudes[bad].round(1),
            )
        return is_clean, bad


# ── Utility: Sliding window generator ─────────────────────────────

def sliding_windows(
    data: np.ndarray,
    window_samples: int,
    step_samples: int,
) -> list[np.ndarray]:
    """
    Generate overlapping windows from continuous data.

    Args:
        data: (n_channels, n_total_samples)
        window_samples: Window length in samples
        step_samples: Step size in samples

    Returns:
        List of (n_channels, window_samples) arrays
    """
    n_total = data.shape[1]
    windows = []
    start = 0
    while start + window_samples <= n_total:
        windows.append(data[:, start : start + window_samples])
        start += step_samples
    return windows

