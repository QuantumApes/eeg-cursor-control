"""
Test suite for DIY EEG Cursor Control.

Run: pytest tests/ -v
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.processing import ProcessingConfig, SignalProcessor, sliding_windows
from src.features import FeatureExtractor, FeatureVector
from src.classifier import MentalStateClassifier, ClassificationResult
from src.cursor import CursorConfig, CursorController


# ══════════════════════════════════════════════════════════════════
#  Signal Processing Tests
# ══════════════════════════════════════════════════════════════════

class TestSignalProcessor:
    """Test the EEG signal processing pipeline."""

    @pytest.fixture
    def processor(self):
        return SignalProcessor(ProcessingConfig(sampling_rate=250))

    @pytest.fixture
    def synthetic_eeg(self):
        """Generate synthetic EEG: alpha + noise."""
        rng = np.random.default_rng(42)
        fs = 250
        t = np.arange(fs) / fs  # 1 second
        n_ch = 4
        data = np.zeros((n_ch, fs))
        for ch in range(n_ch):
            # 10 Hz alpha + 60 Hz noise + random
            data[ch] = (
                20 * np.sin(2 * np.pi * 10 * t)       # Alpha
                + 5 * np.sin(2 * np.pi * 60 * t)       # Line noise
                + rng.standard_normal(fs) * 3           # Background
            )
        return data

    def test_process_shape(self, processor, synthetic_eeg):
        result = processor.process(synthetic_eeg)
        assert result.data.shape == synthetic_eeg.shape

    def test_dc_removal(self, processor, synthetic_eeg):
        # Add DC offset
        data = synthetic_eeg + 1000
        result = processor.process(data)
        # Mean should be near zero after processing
        assert abs(result.data.mean()) < 1.0

    def test_artifact_detection(self, processor):
        # Create data with artifacts (huge amplitude)
        data = np.zeros((4, 250))
        data[2, 100:110] = 500  # Spike on channel 2
        result = processor.process(data)
        assert not result.is_clean
        assert 2 in result.artifact_channels

    def test_clean_signal_passes(self, processor, synthetic_eeg):
        result = processor.process(synthetic_eeg)
        assert result.is_clean

    def test_psd_output(self, processor, synthetic_eeg):
        result = processor.process(synthetic_eeg)
        freqs, psd = processor.compute_psd(result.data)
        assert len(freqs) == psd.shape[1]
        assert psd.shape[0] == synthetic_eeg.shape[0]


class TestSlidingWindows:
    def test_correct_count(self):
        data = np.zeros((4, 1000))
        windows = sliding_windows(data, window_samples=250, step_samples=125)
        # (1000 - 250) / 125 + 1 = 7
        assert len(windows) == 7

    def test_window_shape(self):
        data = np.zeros((4, 500))
        windows = sliding_windows(data, 250, 125)
        for w in windows:
            assert w.shape == (4, 250)


# ══════════════════════════════════════════════════════════════════
#  Feature Extraction Tests
# ══════════════════════════════════════════════════════════════════

class TestFeatureExtractor:
    @pytest.fixture
    def extractor(self):
        return FeatureExtractor(sampling_rate=250)

    @pytest.fixture
    def processed_epoch(self):
        rng = np.random.default_rng(42)
        return rng.standard_normal((4, 250)) * 20

    def test_feature_vector_shape(self, extractor, processed_epoch):
        fv = extractor.extract(processed_epoch)
        assert isinstance(fv, FeatureVector)
        assert len(fv.features) == len(fv.names)
        assert len(fv.features) > 0

    def test_no_nans(self, extractor, processed_epoch):
        fv = extractor.extract(processed_epoch)
        assert not np.any(np.isnan(fv.features))
        assert not np.any(np.isinf(fv.features))

    def test_band_powers_present(self, extractor, processed_epoch):
        fv = extractor.extract(processed_epoch)
        assert "alpha" in fv.band_powers
        assert "beta" in fv.band_powers
        assert all(len(v) == 4 for v in fv.band_powers.values())

    def test_signal_quality_range(self, extractor, processed_epoch):
        fv = extractor.extract(processed_epoch)
        assert 0.0 <= fv.epoch_quality <= 1.0

    def test_single_channel(self, extractor):
        data = np.random.default_rng(0).standard_normal((1, 250)) * 20
        fv = extractor.extract(data)
        assert len(fv.features) > 0


# ══════════════════════════════════════════════════════════════════
#  Classifier Tests
# ══════════════════════════════════════════════════════════════════

class TestClassifier:
    @pytest.fixture
    def trained_classifier(self):
        rng = np.random.default_rng(42)
        classes = ["rest", "focus", "relax"]
        clf = MentalStateClassifier(classes=classes, classifier_type="lda")

        # Separable synthetic data
        n_per_class = 50
        n_features = 20
        X = np.vstack([
            rng.standard_normal((n_per_class, n_features)) + i * 2
            for i, _ in enumerate(classes)
        ])
        y = np.repeat(classes, n_per_class)
        clf.train(X, y, verbose=False)
        return clf, X, y

    def test_training_accuracy(self, trained_classifier):
        clf, X, y = trained_classifier
        assert clf.is_trained
        result = clf.predict(X[0])
        assert result.predicted_class in ["rest", "focus", "relax"]

    def test_prediction_confidence(self, trained_classifier):
        clf, X, _ = trained_classifier
        result = clf.predict(X[0])
        assert 0.0 <= result.confidence <= 1.0

    def test_all_probabilities(self, trained_classifier):
        clf, X, _ = trained_classifier
        result = clf.predict(X[0])
        assert len(result.all_probabilities) == 3
        assert abs(sum(result.all_probabilities.values()) - 1.0) < 0.01

    def test_latency(self, trained_classifier):
        clf, X, _ = trained_classifier
        result = clf.predict(X[0])
        assert result.latency_ms < 10  # Should be < 1ms typically

    def test_save_load(self, trained_classifier, tmp_path):
        clf, X, _ = trained_classifier
        path = str(tmp_path / "model.pkl")
        clf.save(path)

        loaded = MentalStateClassifier.load(path)
        assert loaded.is_trained
        r1 = clf.predict(X[0])
        r2 = loaded.predict(X[0])
        assert r1.predicted_class == r2.predicted_class

    def test_all_classifier_types(self):
        rng = np.random.default_rng(42)
        classes = ["a", "b"]
        X = np.vstack([
            rng.standard_normal((30, 10)),
            rng.standard_normal((30, 10)) + 3,
        ])
        y = np.repeat(classes, 30)

        for clf_type in ["lda", "svm", "random_forest", "neural_net"]:
            clf = MentalStateClassifier(classes=classes, classifier_type=clf_type)
            metrics = clf.train(X, y, verbose=False)
            assert metrics.cv_accuracy > 0.5


# ══════════════════════════════════════════════════════════════════
#  Cursor Controller Tests
# ══════════════════════════════════════════════════════════════════

class TestCursorController:
    @pytest.fixture
    def controller(self):
        ctrl = CursorController(CursorConfig(speed=10, confidence_threshold=0.5))
        ctrl.set_headless(True)
        ctrl.center_cursor()
        return ctrl

    def test_rest_no_movement(self, controller):
        result = ClassificationResult("rest", 0.9, {}, 0.1)
        status = controller.update(result)
        assert status["dx"] == 0 and status["dy"] == 0

    def test_focus_moves_up(self, controller):
        # Multiple updates to overcome smoothing
        for _ in range(20):
            result = ClassificationResult("focus", 0.9, {}, 0.1)
            status = controller.update(result)
        assert status["dy"] < 0  # Negative = up

    def test_low_confidence_gating(self, controller):
        result = ClassificationResult("focus", 0.3, {}, 0.1)
        status = controller.update(result)
        # Below threshold → treated as rest
        assert status["state"] == "rest"

    def test_stats_tracking(self, controller):
        result = ClassificationResult("rest", 0.9, {}, 0.1)
        controller.update(result)
        assert controller.stats["total_updates"] == 1
