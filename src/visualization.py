"""
Real-time EEG Visualization Dashboard.

Displays:
    - Raw EEG traces (scrolling)
    - Power spectrum (live FFT)
    - Band power bars
    - Classification state + confidence
    - Cursor position overlay

Uses matplotlib with blitting for ~30 FPS on most systems.
"""

import logging
import time
from collections import deque
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("TkAgg")  # Interactive backend
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Circle

logger = logging.getLogger(__name__)

# ── Color scheme ──────────────────────────────────────────────────
COLORS = {
    "bg": "#0D1117",
    "text": "#C9D1D9",
    "grid": "#21262D",
    "rest": "#6E7681",
    "focus": "#F85149",
    "relax": "#58A6FF",
    "left_think": "#D29922",
    "right_think": "#3FB950",
    "eeg_lines": ["#58A6FF", "#3FB950", "#D29922", "#F85149",
                   "#BC8CFF", "#79C0FF", "#56D364", "#E3B341"],
    "bands": {
        "delta": "#6E7681",
        "theta": "#D29922",
        "alpha": "#58A6FF",
        "beta": "#F85149",
        "gamma": "#BC8CFF",
    },
}

BAND_NAMES = ["delta", "theta", "alpha", "beta", "gamma"]


class LiveDashboard:
    """
    Real-time EEG visualization dashboard.

    Usage:
        dash = LiveDashboard(n_channels=4, sampling_rate=250)
        dash.start()
        # In your main loop:
        dash.update(raw_data, band_powers, state, confidence, cursor_pos)
        # When done:
        dash.stop()
    """

    def __init__(
        self,
        n_channels: int = 4,
        sampling_rate: int = 250,
        display_seconds: float = 5.0,
        update_interval: float = 0.033,  # ~30 FPS
    ):
        self.n_channels = n_channels
        self.fs = sampling_rate
        self.display_samples = int(display_seconds * sampling_rate)
        self.update_interval = update_interval

        # Scrolling data buffer
        self._eeg_buffer = np.zeros((n_channels, self.display_samples))

        # State tracking
        self._last_update = 0.0
        self._frame_count = 0
        self._fps = 0.0
        self._fps_timer = time.time()

    def start(self) -> None:
        """Initialize the matplotlib figure and axes."""
        plt.style.use("dark_background")
        self.fig = plt.figure(figsize=(16, 10), facecolor=COLORS["bg"])
        self.fig.canvas.manager.set_window_title("🧠 EEG Cursor Control — Live Dashboard")

        gs = GridSpec(3, 3, figure=self.fig, hspace=0.35, wspace=0.3)

        # ── Raw EEG traces (top, spans full width) ────────────────
        self.ax_eeg = self.fig.add_subplot(gs[0, :])
        self.ax_eeg.set_facecolor(COLORS["bg"])
        self.ax_eeg.set_title("Raw EEG Channels", color=COLORS["text"], fontsize=12)
        self.ax_eeg.set_xlabel("Time (s)", color=COLORS["text"])
        self.ax_eeg.set_ylabel("Amplitude (µV)", color=COLORS["text"])
        self.ax_eeg.tick_params(colors=COLORS["text"])

        t = np.arange(self.display_samples) / self.fs
        self.eeg_lines = []
        for i in range(self.n_channels):
            color = COLORS["eeg_lines"][i % len(COLORS["eeg_lines"])]
            line, = self.ax_eeg.plot(t, np.zeros(self.display_samples),
                                      color=color, linewidth=0.8, alpha=0.9,
                                      label=f"Ch {i+1}")
            self.eeg_lines.append(line)
        self.ax_eeg.legend(loc="upper right", fontsize=8)
        self.ax_eeg.set_xlim(0, self.display_samples / self.fs)
        self.ax_eeg.set_ylim(-80, 80)

        # ── Band Power Bars (middle-left) ─────────────────────────
        self.ax_bands = self.fig.add_subplot(gs[1, 0])
        self.ax_bands.set_facecolor(COLORS["bg"])
        self.ax_bands.set_title("Band Power", color=COLORS["text"], fontsize=12)
        self.band_bars = self.ax_bands.bar(
            BAND_NAMES,
            [0] * 5,
            color=[COLORS["bands"][b] for b in BAND_NAMES],
            edgecolor="none",
        )
        self.ax_bands.set_ylim(0, 1)
        self.ax_bands.tick_params(colors=COLORS["text"])

        # ── Classification State (middle-center) ──────────────────
        self.ax_state = self.fig.add_subplot(gs[1, 1])
        self.ax_state.set_facecolor(COLORS["bg"])
        self.ax_state.set_title("Decoded State", color=COLORS["text"], fontsize=12)
        self.ax_state.set_xlim(-1, 1)
        self.ax_state.set_ylim(-1, 1)
        self.ax_state.set_aspect("equal")
        self.ax_state.axis("off")

        self.state_text = self.ax_state.text(
            0, 0.2, "REST", fontsize=28, fontweight="bold",
            ha="center", va="center", color=COLORS["rest"],
        )
        self.confidence_text = self.ax_state.text(
            0, -0.3, "0%", fontsize=16,
            ha="center", va="center", color=COLORS["text"],
        )
        self.state_circle = Circle((0, 0.2), 0.6, fill=False,
                                    linewidth=3, color=COLORS["rest"])
        self.ax_state.add_patch(self.state_circle)

        # ── Cursor Position (middle-right) ────────────────────────
        self.ax_cursor = self.fig.add_subplot(gs[1, 2])
        self.ax_cursor.set_facecolor(COLORS["bg"])
        self.ax_cursor.set_title("Cursor Position", color=COLORS["text"], fontsize=12)
        self.ax_cursor.set_xlim(0, 1920)
        self.ax_cursor.set_ylim(1080, 0)  # Y inverted
        self.ax_cursor.set_aspect("equal")
        self.cursor_dot, = self.ax_cursor.plot(
            960, 540, "o", color="#58A6FF", markersize=12
        )
        self.cursor_trail, = self.ax_cursor.plot(
            [], [], "-", color="#58A6FF", alpha=0.3, linewidth=1
        )
        self._cursor_history = deque(maxlen=200)

        # ── Power Spectrum (bottom, full width) ───────────────────
        self.ax_psd = self.fig.add_subplot(gs[2, :])
        self.ax_psd.set_facecolor(COLORS["bg"])
        self.ax_psd.set_title("Power Spectrum", color=COLORS["text"], fontsize=12)
        self.ax_psd.set_xlabel("Frequency (Hz)", color=COLORS["text"])
        self.ax_psd.set_ylabel("Power (dB)", color=COLORS["text"])
        self.ax_psd.set_xlim(0, 50)
        self.ax_psd.tick_params(colors=COLORS["text"])
        self.psd_line, = self.ax_psd.plot([], [], color="#58A6FF", linewidth=1.5)

        # ── FPS counter ───────────────────────────────────────────
        self.fps_text = self.fig.text(
            0.98, 0.98, "FPS: 0", fontsize=8,
            color=COLORS["text"], ha="right", va="top",
        )

        plt.ion()
        plt.show(block=False)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update(
        self,
        raw_eeg: Optional[np.ndarray] = None,
        band_powers: Optional[dict] = None,
        state: str = "rest",
        confidence: float = 0.0,
        cursor_pos: tuple[int, int] = (960, 540),
        psd_freqs: Optional[np.ndarray] = None,
        psd_power: Optional[np.ndarray] = None,
    ) -> None:
        """
        Update all dashboard panels.

        Call this at ~30 Hz from the main loop.
        """
        now = time.time()
        if now - self._last_update < self.update_interval:
            return
        self._last_update = now

        # ── EEG traces ────────────────────────────────────────────
        if raw_eeg is not None:
            n_new = raw_eeg.shape[1]
            self._eeg_buffer = np.roll(self._eeg_buffer, -n_new, axis=1)
            self._eeg_buffer[:, -n_new:] = raw_eeg[:self.n_channels]
            for i, line in enumerate(self.eeg_lines):
                if i < self._eeg_buffer.shape[0]:
                    line.set_ydata(self._eeg_buffer[i])

        # ── Band power bars ───────────────────────────────────────
        if band_powers:
            max_power = 1e-10
            for band in BAND_NAMES:
                if band in band_powers:
                    val = np.mean(band_powers[band])
                    max_power = max(max_power, val)
            for i, band in enumerate(BAND_NAMES):
                if band in band_powers:
                    val = np.mean(band_powers[band]) / max_power
                    self.band_bars[i].set_height(val)

        # ── State display ─────────────────────────────────────────
        color = COLORS.get(state, COLORS["rest"])
        self.state_text.set_text(state.upper().replace("_", " "))
        self.state_text.set_color(color)
        self.confidence_text.set_text(f"{confidence:.0%}")
        self.state_circle.set_edgecolor(color)
        self.state_circle.set_linewidth(2 + confidence * 4)

        # ── Cursor position ───────────────────────────────────────
        self.cursor_dot.set_data([cursor_pos[0]], [cursor_pos[1]])
        self._cursor_history.append(cursor_pos)
        if len(self._cursor_history) > 1:
            xs, ys = zip(*self._cursor_history)
            self.cursor_trail.set_data(xs, ys)

        # ── PSD ───────────────────────────────────────────────────
        if psd_freqs is not None and psd_power is not None:
            # Average across channels, convert to dB
            avg_psd = np.mean(psd_power, axis=0) if psd_power.ndim > 1 else psd_power
            psd_db = 10 * np.log10(avg_psd + 1e-10)
            self.psd_line.set_data(psd_freqs, psd_db)
            self.ax_psd.set_ylim(psd_db.min() - 5, psd_db.max() + 5)

        # ── FPS ───────────────────────────────────────────────────
        self._frame_count += 1
        if now - self._fps_timer >= 1.0:
            self._fps = self._frame_count / (now - self._fps_timer)
            self._frame_count = 0
            self._fps_timer = now
        self.fps_text.set_text(f"FPS: {self._fps:.0f}")

        # Redraw
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def stop(self) -> None:
        """Close the dashboard."""
        plt.close(self.fig)

    def screenshot(self, path: str) -> None:
        """Save current dashboard state as image."""
        self.fig.savefig(path, dpi=150, facecolor=COLORS["bg"],
                          bbox_inches="tight")
        logger.info("Screenshot saved: %s", path)
