"""
Mental State Classifier for EEG-based cursor control.

Supports multiple classification algorithms optimized for real-time BCI:
    - LDA (Linear Discriminant Analysis) — fastest, solid baseline
    - SVM (Support Vector Machine) — best for small datasets
    - Random Forest — handles nonlinear patterns
    - Neural Network — highest potential accuracy

Includes calibration, online adaptation, and confidence estimation.
"""

import json
import logging
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Output of a single classification."""
    predicted_class: str
    confidence: float               # 0–1 probability of predicted class
    all_probabilities: dict         # {class_name: probability}
    latency_ms: float               # Inference time


@dataclass
class TrainingMetrics:
    """Metrics from model training/evaluation."""
    accuracy: float
    cv_accuracy: float
    cv_std: float
    per_class: dict                 # {class: {precision, recall, f1}}
    confusion_matrix: np.ndarray
    n_samples: int
    n_features: int
    training_time: float


class MentalStateClassifier:
    """
    Real-time mental state classifier for BCI cursor control.

    Usage:
        clf = MentalStateClassifier(
            classes=["rest", "focus", "relax", "left_think", "right_think"],
            classifier_type="lda"
        )
        metrics = clf.train(X_train, y_train)
        result = clf.predict(feature_vector)
    """

    CLASSIFIER_MAP = {
        "lda": lambda: LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        "svm": lambda: SVC(kernel="rbf", C=1.0, gamma="scale", probability=True),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
        ),
        "neural_net": lambda: MLPClassifier(
            hidden_layer_sizes=(128, 64, 32),
            activation="relu",
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.15,
            random_state=42,
        ),
    }

    def __init__(
        self,
        classes: list[str],
        classifier_type: str = "lda",
        cv_folds: int = 5,
    ):
        self.classes = classes
        self.classifier_type = classifier_type
        self.cv_folds = cv_folds

        if classifier_type not in self.CLASSIFIER_MAP:
            raise ValueError(
                f"Unknown classifier '{classifier_type}'. "
                f"Supported: {list(self.CLASSIFIER_MAP.keys())}"
            )

        # Build sklearn pipeline: StandardScaler → Classifier
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", self.CLASSIFIER_MAP[classifier_type]()),
        ])

        self._is_trained = False
        self._adaptation_buffer: list[tuple[np.ndarray, str]] = []

        logger.info(
            "Classifier initialized: type=%s, classes=%s",
            classifier_type, classes,
        )

    # ── Training ──────────────────────────────────────────────────

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        verbose: bool = True,
    ) -> TrainingMetrics:
        """
        Train the classifier with cross-validation.

        Args:
            X: Feature matrix (n_samples, n_features)
            y: Labels array (n_samples,)
            verbose: Print training report

        Returns:
            TrainingMetrics with performance summary
        """
        t_start = time.time()

        # Cross-validation
        cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(self.pipeline, X, y, cv=cv, scoring="accuracy")

        # Final fit on all data
        self.pipeline.fit(X, y)
        self._is_trained = True

        y_pred = self.pipeline.predict(X)
        train_acc = accuracy_score(y, y_pred)

        # Per-class metrics
        report = classification_report(y, y_pred, output_dict=True, zero_division=0)
        per_class = {
            cls: {
                "precision": report[cls]["precision"],
                "recall": report[cls]["recall"],
                "f1": report[cls]["f1-score"],
            }
            for cls in self.classes
            if cls in report
        }

        cm = confusion_matrix(y, y_pred, labels=self.classes)
        training_time = time.time() - t_start

        metrics = TrainingMetrics(
            accuracy=train_acc,
            cv_accuracy=float(cv_scores.mean()),
            cv_std=float(cv_scores.std()),
            per_class=per_class,
            confusion_matrix=cm,
            n_samples=len(y),
            n_features=X.shape[1],
            training_time=training_time,
        )

        if verbose:
            self._print_report(metrics)

        logger.info(
            "Training complete: cv_acc=%.1f%% ± %.1f%%, time=%.2fs",
            metrics.cv_accuracy * 100,
            metrics.cv_std * 100,
            training_time,
        )

        return metrics

    # ── Prediction ────────────────────────────────────────────────

    def predict(self, features: np.ndarray) -> ClassificationResult:
        """
        Classify a single feature vector.

        Args:
            features: 1-D feature array

        Returns:
            ClassificationResult with class, confidence, and latency
        """
        if not self._is_trained:
            raise RuntimeError("Classifier not trained. Call train() first.")

        t_start = time.perf_counter()

        X = features.reshape(1, -1)
        pred = self.pipeline.predict(X)[0]

        # Get probabilities
        if hasattr(self.pipeline.named_steps["classifier"], "predict_proba"):
            probs = self.pipeline.predict_proba(X)[0]
        else:
            # Decision function fallback (softmax approximation)
            dec = self.pipeline.decision_function(X)[0]
            exp_dec = np.exp(dec - np.max(dec))
            probs = exp_dec / exp_dec.sum()

        latency_ms = (time.perf_counter() - t_start) * 1000

        all_probs = {
            cls: float(probs[i])
            for i, cls in enumerate(self.pipeline.classes_)
        }

        return ClassificationResult(
            predicted_class=pred,
            confidence=float(max(probs)),
            all_probabilities=all_probs,
            latency_ms=latency_ms,
        )

    # ── Online Adaptation ─────────────────────────────────────────

    def adapt(
        self, features: np.ndarray, label: str, buffer_size: int = 50
    ) -> None:
        """
        Buffer labeled samples for periodic online re-training.

        Implements a simple sliding-window adaptation strategy
        to handle non-stationarity in EEG signals.
        """
        self._adaptation_buffer.append((features, label))
        if len(self._adaptation_buffer) > buffer_size:
            self._adaptation_buffer.pop(0)

    def retrain_with_adaptation(
        self, X_original: np.ndarray, y_original: np.ndarray
    ) -> TrainingMetrics:
        """Re-train combining original data with adaptation buffer."""
        if not self._adaptation_buffer:
            return self.train(X_original, y_original, verbose=False)

        X_adapt = np.array([x for x, _ in self._adaptation_buffer])
        y_adapt = np.array([y for _, y in self._adaptation_buffer])

        X_combined = np.vstack([X_original, X_adapt])
        y_combined = np.concatenate([y_original, y_adapt])

        return self.train(X_combined, y_combined, verbose=False)

    # ── Persistence ───────────────────────────────────────────────

    def save(self, path: str) -> None:
        """Save trained model to disk."""
        data = {
            "pipeline": self.pipeline,
            "classes": self.classes,
            "classifier_type": self.classifier_type,
            "is_trained": self._is_trained,
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)
        logger.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: str) -> "MentalStateClassifier":
        """Load a trained model from disk."""
        with open(path, "rb") as f:
            data = pickle.load(f)

        obj = cls(
            classes=data["classes"],
            classifier_type=data["classifier_type"],
        )
        obj.pipeline = data["pipeline"]
        obj._is_trained = data["is_trained"]
        logger.info("Model loaded from %s", path)
        return obj

    # ── Internals ─────────────────────────────────────────────────

    @staticmethod
    def _print_report(metrics: TrainingMetrics) -> None:
        print("\n" + "=" * 60)
        print("  EEG CLASSIFIER TRAINING REPORT")
        print("=" * 60)
        print(f"  Samples:          {metrics.n_samples}")
        print(f"  Features:         {metrics.n_features}")
        print(f"  Training acc:     {metrics.accuracy:.1%}")
        print(f"  CV accuracy:      {metrics.cv_accuracy:.1%} ± {metrics.cv_std:.1%}")
        print(f"  Training time:    {metrics.training_time:.2f}s")
        print("-" * 60)
        print("  Per-class metrics:")
        for cls, m in metrics.per_class.items():
            print(
                f"    {cls:15s}  P={m['precision']:.2f}  "
                f"R={m['recall']:.2f}  F1={m['f1']:.2f}"
            )
        print("=" * 60 + "\n")

    @property
    def is_trained(self) -> bool:
        return self._is_trained
