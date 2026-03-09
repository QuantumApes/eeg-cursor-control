#!/usr/bin/env python3
"""
DIY EEG Cursor Control — Main Entry Point

Modes:
    calibrate   Record training data and build your personalized model
    run         Live cursor control using a trained model
    demo        Synthetic data demo (no hardware required)
    benchmark   Test pipeline latency and throughput

Usage:
    python main.py calibrate --board synthetic --classifier lda
    python main.py run --model models/my_model.pkl
    python main.py demo
    python main.py benchmark
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

# ── Setup paths ───────────────────────────────────────────────────
ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

from src.acquisition import EEGAcquisition
from src.calibration import CalibrationSession
from src.classifier import MentalStateClassifier
from src.cursor import CursorConfig, CursorController
from src.features import FeatureExtractor
from src.processing import ProcessingConfig, SignalProcessor


def load_config(config_path: str = "config/default.yaml") -> dict:
    """Load YAML configuration."""
    with open(ROOT / config_path) as f:
        return yaml.safe_load(f)


def setup_logging(level: str = "INFO") -> None:
    """Configure logging."""
    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


# ══════════════════════════════════════════════════════════════════
#  MODE: CALIBRATE
# ══════════════════════════════════════════════════════════════════

def cmd_calibrate(args, config):
    """Run guided calibration session."""
    classes = config["classifier"]["classes"]
    training = config["classifier"]["training"]

    print("🧠 Initializing EEG acquisition...")
    acq = EEGAcquisition(
        board=args.board or config["hardware"]["board"],
        serial_port=config["hardware"].get("serial_port", ""),
        channels=config["hardware"]["channels"],
    )

    with acq:
        session = CalibrationSession(
            acquisition=acq,
            classes=classes,
            epochs_per_class=training["epochs_per_class"],
            epoch_duration=training["epoch_duration"],
            rest_between=training["rest_between"],
            classifier_type=args.classifier or config["classifier"]["type"],
        )
        metrics = session.run()

        # Save model
        model_dir = ROOT / "models"
        model_dir.mkdir(exist_ok=True)
        model_path = model_dir / f"model_{time.strftime('%Y%m%d_%H%M%S')}.pkl"
        session.save(str(model_path))

        print(f"\n✅ Model saved: {model_path}")
        print(f"   CV Accuracy: {metrics.cv_accuracy:.1%} ± {metrics.cv_std:.1%}")


# ══════════════════════════════════════════════════════════════════
#  MODE: RUN (Live Cursor Control)
# ══════════════════════════════════════════════════════════════════

def cmd_run(args, config):
    """Live cursor control with a trained model."""
    if not args.model:
        # Find most recent model
        model_dir = ROOT / "models"
        models = sorted(model_dir.glob("model_*.pkl"))
        if not models:
            print("❌ No trained model found. Run 'calibrate' first.")
            return
        args.model = str(models[-1])

    print(f"🧠 Loading model: {args.model}")
    classifier = MentalStateClassifier.load(args.model)

    processor = SignalProcessor(ProcessingConfig(
        sampling_rate=config["hardware"]["sampling_rate"],
        notch_freq=config["processing"]["notch_freq"],
        bandpass_low=config["processing"]["bandpass"]["low"],
        bandpass_high=config["processing"]["bandpass"]["high"],
    ))
    extractor = FeatureExtractor(sampling_rate=config["hardware"]["sampling_rate"])

    cursor_config = CursorConfig(
        mode=config["cursor"]["mode"],
        speed=config["cursor"]["speed"],
        smoothing=config["cursor"]["smoothing"],
        confidence_threshold=config["cursor"]["confidence_threshold"],
    )
    cursor = CursorController(cursor_config)

    # Optional: live visualization
    dashboard = None
    if config["debug"]["live_plot"]:
        try:
            from src.visualization import LiveDashboard
            dashboard = LiveDashboard(
                n_channels=len(config["hardware"]["channels"]),
                sampling_rate=config["hardware"]["sampling_rate"],
            )
            dashboard.start()
        except ImportError:
            print("⚠️  Visualization requires matplotlib. Continuing without dashboard.")

    print("🧠 Starting EEG acquisition...")
    acq = EEGAcquisition(
        board=args.board or config["hardware"]["board"],
        serial_port=config["hardware"].get("serial_port", ""),
        channels=config["hardware"]["channels"],
    )

    window_samples = int(
        config["processing"]["buffer_seconds"] * config["hardware"]["sampling_rate"]
    )

    print("\n" + "=" * 50)
    print("  🎮  CURSOR CONTROL ACTIVE")
    print("  Press Ctrl+C to stop")
    print("=" * 50 + "\n")

    try:
        with acq:
            cursor.center_cursor()
            while True:
                # 1. Get latest EEG window
                raw = acq.get_latest(window_samples)
                if raw.shape[1] < window_samples // 2:
                    time.sleep(0.01)
                    continue

                # 2. Process
                epoch = processor.process(raw)
                if not epoch.is_clean:
                    continue

                # 3. Extract features
                fv = extractor.extract(epoch.data)

                # 4. Classify
                result = classifier.predict(fv.features)

                # 5. Move cursor
                status = cursor.update(result)

                # 6. Visualization
                if dashboard:
                    freqs, psd = processor.compute_psd(epoch.data)
                    dashboard.update(
                        raw_eeg=raw,
                        band_powers=fv.band_powers,
                        state=result.predicted_class,
                        confidence=result.confidence,
                        cursor_pos=status["position"],
                        psd_freqs=freqs,
                        psd_power=psd,
                    )

                # 7. Console output
                print(
                    f"\r  State: {result.predicted_class:12s} | "
                    f"Conf: {result.confidence:.0%} | "
                    f"Pos: {status['position']} | "
                    f"Latency: {result.latency_ms:.1f}ms  ",
                    end="", flush=True,
                )

                time.sleep(0.033)  # ~30 Hz

    except KeyboardInterrupt:
        print("\n\n🛑 Cursor control stopped.")
        print(f"   Stats: {cursor.stats}")
    finally:
        if dashboard:
            dashboard.stop()


# ══════════════════════════════════════════════════════════════════
#  MODE: DEMO (Synthetic — No Hardware Needed)
# ══════════════════════════════════════════════════════════════════

def cmd_demo(args, config):
    """
    Run a full demo with synthetic EEG data.
    Perfect for testing the pipeline without hardware.
    """
    print("🧠 DIY EEG CURSOR CONTROL — DEMO MODE")
    print("=" * 50)
    print("  Using synthetic EEG data (no hardware required)")
    print("  This demonstrates the full pipeline end-to-end.\n")

    # Override config for synthetic board
    config["hardware"]["board"] = "synthetic"

    # Quick calibration with fewer epochs
    config["classifier"]["training"]["epochs_per_class"] = 10
    config["classifier"]["training"]["epoch_duration"] = 2.0
    config["classifier"]["training"]["rest_between"] = 0.5

    classes = config["classifier"]["classes"]

    acq = EEGAcquisition(board="synthetic", channels=[1, 2, 3, 4])

    with acq:
        # Generate synthetic training data
        print("📊 Generating synthetic training data...")
        processor = SignalProcessor(ProcessingConfig(sampling_rate=acq.sampling_rate))
        extractor = FeatureExtractor(sampling_rate=acq.sampling_rate)

        X_train = []
        y_train = []
        n_samples = int(2.0 * acq.sampling_rate)

        for label in classes:
            for _ in range(15):
                time.sleep(0.05)
                raw = acq.get_latest(n_samples)
                if raw.shape[1] < n_samples // 2:
                    continue
                epoch = processor.process(raw)
                fv = extractor.extract(epoch.data)
                X_train.append(fv.features)
                y_train.append(label)

        X = np.array(X_train)
        y = np.array(y_train)

        print(f"   Collected {len(y)} training samples ({X.shape[1]} features)\n")

        # Train classifier
        classifier = MentalStateClassifier(classes=classes, classifier_type="lda")
        metrics = classifier.train(X, y)

        # Live demo
        print("\n🎮 Running live demo (10 seconds)...")
        cursor = CursorController(CursorConfig(speed=10, confidence_threshold=0.3))
        cursor.set_headless(True)
        cursor.center_cursor()

        start = time.time()
        while time.time() - start < 10:
            raw = acq.get_latest(n_samples)
            if raw.shape[1] < n_samples // 2:
                time.sleep(0.01)
                continue

            epoch = processor.process(raw)
            fv = extractor.extract(epoch.data)
            result = classifier.predict(fv.features)
            status = cursor.update(result)

            print(
                f"\r  [{time.time()-start:5.1f}s] "
                f"State: {result.predicted_class:12s} | "
                f"Conf: {result.confidence:.0%} | "
                f"Pos: {status['position']} | "
                f"{result.latency_ms:.1f}ms  ",
                end="", flush=True,
            )
            time.sleep(0.033)

        print(f"\n\n✅ Demo complete! Stats: {cursor.stats}")


# ══════════════════════════════════════════════════════════════════
#  MODE: BENCHMARK
# ══════════════════════════════════════════════════════════════════

def cmd_benchmark(args, config):
    """Benchmark pipeline latency and throughput."""
    print("⚡ PIPELINE BENCHMARK")
    print("=" * 50)

    n_ch = 4
    fs = 250
    n_samples = fs  # 1 second window
    n_iterations = 1000

    processor = SignalProcessor(ProcessingConfig(sampling_rate=fs))
    extractor = FeatureExtractor(sampling_rate=fs)
    classifier = MentalStateClassifier(
        classes=["rest", "focus", "relax"],
        classifier_type="lda",
    )

    # Generate dummy training data
    rng = np.random.default_rng(42)
    X_dummy = rng.standard_normal((90, 50))
    y_dummy = np.repeat(["rest", "focus", "relax"], 30)
    classifier.train(X_dummy, y_dummy, verbose=False)

    # Benchmark processing
    raw = rng.standard_normal((n_ch, n_samples)) * 20
    times = {"process": [], "features": [], "classify": [], "total": []}

    for _ in range(n_iterations):
        t0 = time.perf_counter()

        t1 = time.perf_counter()
        epoch = processor.process(raw)
        t2 = time.perf_counter()

        fv = extractor.extract(epoch.data)
        t3 = time.perf_counter()

        result = classifier.predict(fv.features)
        t4 = time.perf_counter()

        times["process"].append((t2 - t1) * 1000)
        times["features"].append((t3 - t2) * 1000)
        times["classify"].append((t4 - t3) * 1000)
        times["total"].append((t4 - t1) * 1000)

    print(f"\n  Iterations: {n_iterations}")
    print(f"  Window:     {n_samples/fs:.1f}s ({n_samples} samples, {n_ch} channels)")
    print(f"\n  {'Stage':<15s} {'Mean':>8s} {'P50':>8s} {'P95':>8s} {'P99':>8s}")
    print("  " + "-" * 47)
    for stage, vals in times.items():
        arr = np.array(vals)
        print(
            f"  {stage:<15s} "
            f"{arr.mean():7.2f}ms "
            f"{np.percentile(arr, 50):7.2f}ms "
            f"{np.percentile(arr, 95):7.2f}ms "
            f"{np.percentile(arr, 99):7.2f}ms"
        )

    total_mean = np.mean(times["total"])
    max_fps = 1000 / total_mean
    print(f"\n  Max throughput: {max_fps:.0f} Hz ({total_mean:.2f}ms per frame)")
    print(f"  ✅ {'PASSES' if total_mean < 33 else 'FAILS'} real-time requirement (<33ms for 30 Hz)")


# ══════════════════════════════════════════════════════════════════
#  CLI
# ══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="🧠 DIY EEG Cursor Control — Brain-Computer Interface",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # calibrate
    p_cal = sub.add_parser("calibrate", help="Record training data & build model")
    p_cal.add_argument("--board", type=str, help="Board type override")
    p_cal.add_argument("--classifier", type=str, help="Classifier type override")
    p_cal.add_argument("--config", type=str, default="config/default.yaml")

    # run
    p_run = sub.add_parser("run", help="Live cursor control")
    p_run.add_argument("--model", type=str, help="Path to trained model")
    p_run.add_argument("--board", type=str, help="Board type override")
    p_run.add_argument("--config", type=str, default="config/default.yaml")

    # demo
    p_demo = sub.add_parser("demo", help="Synthetic demo (no hardware)")
    p_demo.add_argument("--config", type=str, default="config/default.yaml")

    # benchmark
    p_bench = sub.add_parser("benchmark", help="Pipeline latency benchmark")
    p_bench.add_argument("--config", type=str, default="config/default.yaml")

    args = parser.parse_args()
    config = load_config(args.config)
    setup_logging(config.get("debug", {}).get("log_level", "INFO"))

    commands = {
        "calibrate": cmd_calibrate,
        "run": cmd_run,
        "demo": cmd_demo,
        "benchmark": cmd_benchmark,
    }
    commands[args.command](args, config)


if __name__ == "__main__":
    main()
