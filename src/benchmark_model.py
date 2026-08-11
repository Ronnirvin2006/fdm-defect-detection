import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("KERAS_HOME", str(Path(__file__).resolve().parents[1] / ".keras"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import tensorflow as tf

from config import IMAGE_SIZE, MODELS_DIR, REPORTS_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark single-image CPU inference.")
    parser.add_argument("--model", type=Path, default=MODELS_DIR / "best_model.keras")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=30)
    args = parser.parse_args()

    model = tf.keras.models.load_model(args.model, compile=False)
    sample = np.zeros((1, *IMAGE_SIZE, 3), dtype=np.float32)
    for _ in range(args.warmup):
        model.predict(sample, verbose=0)

    timings = []
    for _ in range(args.runs):
        started = time.perf_counter()
        model.predict(sample, verbose=0)
        timings.append(time.perf_counter() - started)

    mean_seconds = sum(timings) / len(timings)
    result = {
        "device": "CPU",
        "tensorflow_version": tf.__version__,
        "model": str(args.model),
        "parameters": int(model.count_params()),
        "model_size_bytes": int(args.model.stat().st_size),
        "input_shape": [1, *IMAGE_SIZE, 3],
        "warmup_runs": args.warmup,
        "timed_runs": args.runs,
        "mean_ms_per_image": mean_seconds * 1000,
        "median_ms_per_image": float(np.median(timings) * 1000),
        "estimated_serial_fps": 1.0 / mean_seconds,
        "note": "Timing includes the Keras predict call on one synthetic image and varies by host load.",
    }
    output = REPORTS_DIR / "inference_benchmark.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
