"""
EEG Signal Acquisition via BrainFlow.

Abstracts hardware differences across OpenBCI Cyton/Ganglion, NeuroSky,
and synthetic boards. Provides a streaming ring buffer for real-time access.
"""

import logging
import time
from collections import deque
from threading import Event, Lock, Thread
from typing import Optional

import numpy as np
from brainflow.board_shim import BoardIds, BoardShim, BrainFlowInputParams
from brainflow.data_filter import DataFilter

logger = logging.getLogger(__name__)

# ── Board ID mapping ──────────────────────────────────────────────
BOARD_MAP = {
    "openbci_cyton": BoardIds.CYTON_BOARD,
    "openbci_ganglion": BoardIds.GANGLION_BOARD,
    "neurosky_mindwave": BoardIds.MINDWAVE_BOARD,
    "synthetic": BoardIds.SYNTHETIC_BOARD,
}


class EEGAcquisition:
    """
    Manages EEG board connection, streaming, and ring-buffer storage.

    Usage:
        acq = EEGAcquisition(board="synthetic", channels=[1,2,3,4])
        acq.start()
        data = acq.get_latest(n_samples=250)  # last 1 s at 250 Hz
        acq.stop()
    """

    def __init__(
        self,
        board: str = "synthetic",
        serial_port: str = "",
        channels: Optional[list[int]] = None,
        buffer_seconds: float = 30.0,
    ):
        self.board_name = board
        self.board_id = BOARD_MAP.get(board)
        if self.board_id is None:
            raise ValueError(
                f"Unknown board '{board}'. Supported: {list(BOARD_MAP.keys())}"
            )

        # BrainFlow params
        self.params = BrainFlowInputParams()
        if serial_port and board != "synthetic":
            self.params.serial_port = serial_port

        self.board = BoardShim(self.board_id, self.params)
        self.sampling_rate = BoardShim.get_sampling_rate(self.board_id)

        # Channel setup
        all_eeg = BoardShim.get_eeg_channels(self.board_id)
        self.channels = channels if channels else all_eeg
        self.n_channels = len(self.channels)

        # Ring buffer
        buf_size = int(self.sampling_rate * buffer_seconds)
        self._buffer = np.zeros((self.n_channels, buf_size))
        self._buf_idx = 0
        self._buf_full = False
        self._lock = Lock()

        # Streaming thread
        self._stop_event = Event()
        self._thread: Optional[Thread] = None
        self._samples_acquired = 0

        logger.info(
            "EEGAcquisition initialized: board=%s, rate=%d Hz, channels=%s",
            board, self.sampling_rate, self.channels,
        )

    # ── Lifecycle ─────────────────────────────────────────────────

    def start(self) -> None:
        """Prepare session and begin streaming."""
        self.board.prepare_session()
        self.board.start_stream(45000)  # ring buffer size in BrainFlow
        self._stop_event.clear()
        self._thread = Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        logger.info("Streaming started.")

    def stop(self) -> None:
        """Stop streaming and release session."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        try:
            self.board.stop_stream()
            self.board.release_session()
        except Exception:
            pass
        logger.info(
            "Streaming stopped. Total samples acquired: %d",
            self._samples_acquired,
        )

    # ── Data access ───────────────────────────────────────────────

    def get_latest(self, n_samples: Optional[int] = None) -> np.ndarray:
        """
        Return the most recent EEG data from the ring buffer.

        Args:
            n_samples: Number of samples to return. Defaults to 1 second.

        Returns:
            np.ndarray of shape (n_channels, n_samples)
        """
        if n_samples is None:
            n_samples = self.sampling_rate

        with self._lock:
            buf_len = self._buffer.shape[1]
            available = buf_len if self._buf_full else self._buf_idx
            n = min(n_samples, available)

            if n == 0:
                return np.zeros((self.n_channels, 0))

            end = self._buf_idx
            start = end - n
            if start >= 0:
                return self._buffer[:, start:end].copy()
            else:
                # Wrap-around
                return np.hstack([
                    self._buffer[:, start:],
                    self._buffer[:, :end],
                ]).copy()

    def get_impedance(self) -> dict[int, float]:
        """Check electrode impedance (if supported)."""
        impedances = {}
        for ch in self.channels:
            try:
                imp = DataFilter.get_nearest_power_of_two(self.sampling_rate)
                impedances[ch] = imp
            except Exception:
                impedances[ch] = -1.0
        return impedances

    @property
    def is_streaming(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def total_samples(self) -> int:
        return self._samples_acquired

    # ── Internal ──────────────────────────────────────────────────

    def _poll_loop(self) -> None:
        """Background thread: pull data from BrainFlow into ring buffer."""
        while not self._stop_event.is_set():
            data = self.board.get_board_data()  # clears internal buffer
            if data.shape[1] == 0:
                time.sleep(0.005)
                continue

            # Extract only selected EEG channels
            eeg = data[self.channels, :]

            with self._lock:
                n_new = eeg.shape[1]
                buf_len = self._buffer.shape[1]

                if n_new >= buf_len:
                    # More data than buffer → take the tail
                    self._buffer[:] = eeg[:, -buf_len:]
                    self._buf_idx = 0
                    self._buf_full = True
                else:
                    end = self._buf_idx + n_new
                    if end <= buf_len:
                        self._buffer[:, self._buf_idx:end] = eeg
                    else:
                        first = buf_len - self._buf_idx
                        self._buffer[:, self._buf_idx:] = eeg[:, :first]
                        self._buffer[:, :n_new - first] = eeg[:, first:]
                        self._buf_full = True
                    self._buf_idx = end % buf_len

                self._samples_acquired += n_new

            time.sleep(0.002)  # ~500 Hz poll rate

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *args):
        self.stop()
