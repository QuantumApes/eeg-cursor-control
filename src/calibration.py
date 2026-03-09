"""
Guided Calibration Session for EEG Cursor Control.

Walks the user through recording labeled EEG epochs for each mental state.
Provides visual/audio cues, collects data, trains the classifier, and
saves the model.

Design inspired by Neuralink's N1 calibration protocol: short, gamified
trials with immediate feedback on decode quality.
"""

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import numpy as np

from .acquisition import EEGAcquisition
from .classifier import MentalStateClassifier, TrainingMetrics
from .features import FeatureExtractor
from .processing import ProcessingConfig, SignalProcessor

logger = logging.getLogger(__name__)

# ── Visual cues for terminal-based calibration ────────────────────
STATE_PROMPTS = {
    "rest": {
        "instruction": "RELAX — Clear your mind. Breathe normally.",
        "emoji": "😌",
        "color": "\033[90m",  # Gray
    },
    "focus": {
        "instruction": "FOCUS — Concentrate intensely. Mental math: count back from 100 by 7s.",
        "emoji": "🎯",
        "color": "\033[91m",  # Red
    },
    "relax": {
        "instruction": "RELAX DEEPLY — Close your eyes. Visualize a calm beach.",
        "emoji": "🌊",
        "color": "\033[94m",  # Blue
    },
    "left_think": {
        "instruction": "LEFT HAND — Imagine squeezing your LEFT hand into a fist.",
        "emoji": "👈",
        "color": "\033[93m",  # Yellow
    },
    "right_think": {
        "instruction": "RIGHT HAND — Imagine squeezing your RIGHT hand into a fist.",
        "emoji": "👉",
        "color": "\033[92m",  # Green
    },
}

RESET_COLOR = "\033[0m"


class CalibrationSession:
    """
    Guided calibration to collect training data and build a personalized model.

    Usage:
        session = CalibrationSession(
            acquisition=acq,
            classes=["rest", "focus", "relax", "left_think", "right_think"],
        )
        metrics = session.run()       # Interactive calibration
        session.save("models/my_model.pkl")
    """

    def __init__(
        self,
        acquisition: EEGAcquisition,
        classes: list[str],
        epochs_per_class: int = 40,
        epoch_duration: float = 4.0,
        rest_between: float = 2.0,
        classifier_type: str = "lda",
        output_dir: str = "data/calibration",
    ):
        self.acq = acquisition
        self.classes = classes
        self.epochs_per_class = epochs_per_class
        self.epoch_duration = epoch_duration
        self.rest_between = rest_between
        self.classifier_type = classifier_type
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Processing pipeline
        self.processor = SignalProcessor(ProcessingConfig(
            sampling_rate=self.acq.sampling_rate
        ))
        self.extractor = FeatureExtractor(sampling_rate=self.acq.sampling_rate)
        self.classifier = MentalStateClassifier(
            classes=classes,
            classifier_type=classifier_type,
        )

        # Data storage
        self.X_train: list[np.ndarray] = []
        self.y_train: list[str] = []
        self.raw_epochs: list[np.ndarray] = []

    # ── Main calibration loop ─────────────────────────────────────

    def run(self) -> TrainingMetrics:
        """
        Execute the full calibration procedure.

        Returns:
            TrainingMetrics from classifier training
        """
        total = len(self.classes) * self.epochs_per_class
        trial = 0

        self._print_header()

        # Randomized trial order (balanced)
        trial_order = self.classes * self.epochs_per_class
        rng = np.random.default_rng(42)
        rng.shuffle(trial_order)

        for label in trial_order:
            trial += 1
            self._run_trial(label, trial, total)

        # Train classifier
        print("\n\033[1m⚡ Training classifier...\033[0m\n")
        X = np.array(self.X_train)
        y = np.array(self.y_train)

        metrics = self.classifier.train(X, y)

        # Save raw data
        self._save_data(X, y)

        return metrics

    def _run_trial(self, label: str, trial_num: int, total: int) -> None:
        """Execute a single calibration trial."""
        prompt = STATE_PROMPTS.get(label, STATE_PROMPTS["rest"])

        # Rest period
        print(f"\n  [{trial_num}/{total}] Prepare for next trial...")
        self._countdown(self.rest_between)

        # Cue
        print(
            f"\n  {prompt['color']}{prompt['emoji']}  {prompt['instruction']}"
            f"{RESET_COLOR}"
        )

        # Record epoch
        n_samples = int(self.epoch_duration * self.acq.sampling_rate)
        time.sleep(0.1)  # Brief settling time

        # Wait for data to accumulate
        start_time = time.time()
        while time.time() - start_time < self.epoch_duration:
            elapsed = time.time() - start_time
            remaining = self.epoch_duration - elapsed
            bar_len = 30
            filled = int(bar_len * elapsed / self.epoch_duration)
            bar = "█" * filled + "░" * (bar_len - filled)
            print(f"\r  Recording: [{bar}] {remaining:.1f}s ", end="", flush=True)
            time.sleep(0.1)

        print()  # Newline after progress bar

        # Grab recorded data
        raw = self.acq.get_latest(n_samples)
        if raw.shape[1] < n_samples // 2:
            logger.warning("Insufficient data for trial %d, skipping", trial_num)
            return

        # Process
        epoch_result = self.processor.process(raw)
        if not epoch_result.is_clean:
            logger.debug("Artifact in trial %d (channels: %s)", trial_num, epoch_result.artifact_channels)
            # Still use it — artifact rejection is soft for calibration

        # Extract features
        fv = self.extractor.extract(epoch_result.data)

        # Store
        self.X_train.append(fv.features)
        self.y_train.append(label)
        self.raw_epochs.append(raw)

        # Quality feedback
        quality_bar = "●" * int(fv.epoch_quality * 5) + "○" * (5 - int(fv.epoch_quality * 5))
        print(f"  Signal quality: [{quality_bar}] {fv.epoch_quality:.0%}")

    # ── Persistence ───────────────────────────────────────────────

    def save(self, model_path: str) -> None:
        """Save trained classifier."""
        self.classifier.save(model_path)

    def _save_data(self, X: np.ndarray, y: np.ndarray) -> None:
        """Save calibration data for offline analysis."""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        np.savez(
            self.output_dir / f"calibration_{timestamp}.npz",
            X=X,
            y=y,
            classes=self.classes,
            sampling_rate=self.acq.sampling_rate,
        )
        logger.info("Calibration data saved to %s", self.output_dir)

    # ── UI helpers ────────────────────────────────────────────────

    @staticmethod
    def _print_header() -> None:
        print("\n" + "=" * 60)
        print("  🧠  EEG CURSOR CONTROL — CALIBRATION SESSION")
        print("=" * 60)
        print("  Follow the prompts. Stay still. Focus on the task.")
        print("  Each trial lasts ~4 seconds with rest breaks.")
        print("=" * 60)

    @staticmethod
    def _countdown(seconds: float) -> None:
        for i in range(int(seconds), 0, -1):
            print(f"\r  Starting in {i}... ", end="", flush=True)
            time.sleep(1.0)
        remaining = seconds - int(seconds)
        if remaining > 0:
            time.sleep(remaining)
