import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("KERAS_HOME", str(Path(__file__).resolve().parents[1] / ".keras"))

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize

from config import BATCH_SIZE, DATA_RAW, FIGURES_DIR, IMAGE_SIZE, MODELS_DIR, REPORTS_DIR
from train import collect_files, make_dataset, stratified_split
from inspect_dataset import find_class_root


def configure_tensorflow() -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
    print(f"TensorFlow: {tf.__version__}")
    print(f"GPUs: {tf.config.list_physical_devices('GPU')}")


def plot_multiclass_roc(y_true, probabilities, class_names, path: Path) -> dict:
    y_bin = label_binarize(y_true, classes=np.arange(len(class_names)))
    auc_values = {}

    fig, ax = plt.subplots(figsize=(8, 6))
    for idx, class_name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_bin[:, idx], probabilities[:, idx])
        class_auc = auc(fpr, tpr)
        auc_values[class_name] = float(class_auc)
        ax.plot(fpr, tpr, label=f"{class_name} AUC={class_auc:.3f}")

    macro_auc = roc_auc_score(y_bin, probabilities, average="macro", multi_class="ovr")
    weighted_auc = roc_auc_score(y_bin, probabilities, average="weighted", multi_class="ovr")
    auc_values["macro_ovr"] = float(macro_auc)
    auc_values["weighted_ovr"] = float(weighted_auc)

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_title("One-vs-Rest ROC Curves")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return auc_values


def measure_inference_time(model, test_ds, warmup_batches: int = 2, timed_batches: int = 10) -> dict:
    image_count = 0
    elapsed = 0.0

    for batch_idx, (images, _) in enumerate(test_ds.take(warmup_batches + timed_batches)):
        if batch_idx < warmup_batches:
            model.predict(images, verbose=0)
            continue
        start = time.perf_counter()
        model.predict(images, verbose=0)
        elapsed += time.perf_counter() - start
        image_count += int(images.shape[0])

    avg_ms = (elapsed / image_count) * 1000 if image_count else 0.0
    return {
        "timed_images": image_count,
        "total_seconds": float(elapsed),
        "average_ms_per_image": float(avg_ms),
        "estimated_fps": float(1000.0 / avg_ms) if avg_ms > 0 else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate trained FDM defect model.")
    parser.add_argument("--model", type=Path, default=MODELS_DIR / "best_model.keras")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    configure_tensorflow()

    class_root = find_class_root(DATA_RAW)
    paths, labels, class_names = collect_files(class_root)
    _, _, (test_paths, test_labels) = stratified_split(paths, labels)
    test_ds = make_dataset(test_paths, test_labels, False, args.batch_size)

    model = tf.keras.models.load_model(args.model)
    test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
    probabilities = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)
    y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    auc_values = plot_multiclass_roc(y_true, probabilities, class_names, FIGURES_DIR / "roc_curves.png")
    timing = measure_inference_time(model, test_ds)

    metrics = {
        "class_names": class_names,
        "test_images": int(len(test_paths)),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_accuracy),
        "image_size": list(IMAGE_SIZE),
        "batch_size": args.batch_size,
        "roc_auc": auc_values,
        "inference_time": timing,
    }

    (REPORTS_DIR / "evaluation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (REPORTS_DIR / "evaluation_classification_report.txt").write_text(report, encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(report)
    print(f"Saved ROC curve: {FIGURES_DIR / 'roc_curves.png'}")


if __name__ == "__main__":
    main()
