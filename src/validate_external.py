import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("KERAS_HOME", str(Path(__file__).resolve().parents[1] / ".keras"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from config import IMAGE_SIZE, MODELS_DIR, REPORTS_DIR, FIGURES_DIR
from inspect_dataset import IMAGE_EXTS, find_class_root


def collect_external_images(data_root: Path, class_names: list[str]):
    class_root = find_class_root(data_root)
    paths = []
    labels = []
    missing = []
    for class_index, class_name in enumerate(class_names):
        class_dir = class_root / class_name
        if not class_dir.exists():
            missing.append(class_name)
            continue
        for path in sorted(class_dir.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                paths.append(str(path))
                labels.append(class_index)
    if not paths:
        raise FileNotFoundError(f"No labeled class-folder images found under {data_root}")
    return np.asarray(paths), np.asarray(labels, dtype=np.int32), missing


def decode(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMAGE_SIZE)
    return tf.cast(image, tf.float32), label


def save_confusion(cm: np.ndarray, class_names: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 8))
    image = ax.imshow(cm, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        xlabel="Predicted label",
        ylabel="True label",
        title="External Dataset Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right")
    for row in range(cm.shape[0]):
        for column in range(cm.shape[1]):
            ax.text(column, row, cm[row, column], ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the active model on a separate labeled dataset.")
    parser.add_argument("data_root", type=Path, help="External folder containing one subfolder per class.")
    parser.add_argument("--model", type=Path, default=MODELS_DIR / "best_model.keras")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    class_names = json.loads((MODELS_DIR / "class_names.json").read_text(encoding="utf-8"))
    paths, labels, missing = collect_external_images(args.data_root, class_names)
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.map(decode, num_parallel_calls=tf.data.AUTOTUNE).batch(args.batch_size).prefetch(tf.data.AUTOTUNE)

    model = tf.keras.models.load_model(args.model, compile=False)
    probabilities = model.predict(dataset, verbose=1)
    predictions = np.argmax(probabilities, axis=1)
    present_indices = sorted(set(labels.tolist()))
    present_names = [class_names[index] for index in present_indices]
    report = classification_report(
        labels,
        predictions,
        labels=present_indices,
        target_names=present_names,
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    report_text = classification_report(
        labels,
        predictions,
        labels=present_indices,
        target_names=present_names,
        digits=4,
        zero_division=0,
    )
    accuracy = float(np.mean(predictions == labels))
    result = {
        "data_root": str(args.data_root.resolve()),
        "images": int(len(paths)),
        "accuracy": accuracy,
        "present_classes": present_names,
        "missing_classes": missing,
        "classification_report": report,
        "note": "External validity requires images not used in training or model selection.",
    }
    (REPORTS_DIR / "external_validation_metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (REPORTS_DIR / "external_validation_report.txt").write_text(report_text, encoding="utf-8")
    save_confusion(
        confusion_matrix(labels, predictions, labels=np.arange(len(class_names))),
        class_names,
        FIGURES_DIR / "external_validation_confusion_matrix.png",
    )
    print(json.dumps({key: value for key, value in result.items() if key != "classification_report"}, indent=2))
    print(report_text)


if __name__ == "__main__":
    main()
