import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("KERAS_HOME", str(Path(__file__).resolve().parents[1] / ".keras"))

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from config import (
    BATCH_SIZE,
    DATA_RAW,
    FIGURES_DIR,
    IMAGE_SIZE,
    LOGS_DIR,
    MODELS_DIR,
    REPORTS_DIR,
    SEED,
    TEST_SPLIT,
    VALIDATION_SPLIT,
)
from inspect_dataset import IMAGE_EXTS, find_class_root


AUTOTUNE = tf.data.AUTOTUNE


def configure_tensorflow(mixed_precision: bool) -> None:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if mixed_precision and gpus:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    print(f"TensorFlow: {tf.__version__}")
    print(f"GPUs: {gpus}")


def collect_files(class_root: Path):
    class_names = sorted([p.name for p in class_root.iterdir() if p.is_dir()])
    class_to_id = {name: idx for idx, name in enumerate(class_names)}
    file_paths = []
    labels = []
    for class_name in class_names:
        for path in sorted((class_root / class_name).rglob("*")):
            if path.suffix.lower() in IMAGE_EXTS:
                file_paths.append(str(path))
                labels.append(class_to_id[class_name])
    return np.array(file_paths), np.array(labels, dtype=np.int32), class_names


def stratified_split(paths, labels):
    from sklearn.model_selection import train_test_split

    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        paths,
        labels,
        test_size=VALIDATION_SPLIT + TEST_SPLIT,
        random_state=SEED,
        stratify=labels,
    )
    relative_test = TEST_SPLIT / (VALIDATION_SPLIT + TEST_SPLIT)
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths,
        temp_labels,
        test_size=relative_test,
        random_state=SEED,
        stratify=temp_labels,
    )
    return (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels)


def decode_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32)
    return image, label


def make_dataset(paths, labels, training: bool, batch_size: int):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(decode_image, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds


def build_model(num_classes: int, dropout: float, architecture: str):
    inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
    augmenter = tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal"),
            tf.keras.layers.RandomRotation(0.05),
            tf.keras.layers.RandomZoom(0.08),
            tf.keras.layers.RandomContrast(0.08),
        ],
        name="augmentation",
    )
    x = augmenter(inputs)

    if architecture == "efficientnet_b0":
        x = tf.keras.applications.efficientnet.preprocess_input(x)
        base = tf.keras.applications.EfficientNetB0(
            input_shape=(*IMAGE_SIZE, 3),
            include_top=False,
            weights="imagenet",
        )
        fine_tune_start = -30
    elif architecture == "mobilenet_v2":
        x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
        base = tf.keras.applications.MobileNetV2(
            input_shape=(*IMAGE_SIZE, 3),
            include_top=False,
            weights="imagenet",
        )
        fine_tune_start = -35
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")

    base.trainable = False
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(dropout)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
    model = tf.keras.Model(inputs, outputs)
    return model, base, fine_tune_start


def compile_model(model, learning_rate: float):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def plot_history(history, path: Path) -> None:
    metrics = history.history
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(metrics["accuracy"], label="train")
    axes[0].plot(metrics["val_accuracy"], label="val")
    axes[0].set_title("Accuracy")
    axes[0].legend()
    axes[1].plot(metrics["loss"], label="train")
    axes[1].plot(metrics["val_loss"], label="val")
    axes[1].set_title("Loss")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def save_confusion_matrix(cm, class_names, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(7, len(class_names)), max(6, len(class_names))))
    im = ax.imshow(cm, cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(class_names)),
        yticks=np.arange(len(class_names)),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", rotation_mode="anchor")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train FDM defect classifier.")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--fine-tune-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--dropout", type=float, default=0.25)
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument(
        "--architecture",
        choices=["efficientnet_b0", "mobilenet_v2"],
        default="efficientnet_b0",
        help="EfficientNet-B0 is stronger for Colab/Kaggle; MobileNetV2 is lighter for weak laptops.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate dataset/model setup without training.")
    args = parser.parse_args()

    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    configure_tensorflow(args.mixed_precision)

    class_root = find_class_root(DATA_RAW)
    paths, labels, class_names = collect_files(class_root)
    (train_paths, train_labels), (val_paths, val_labels), (test_paths, test_labels) = stratified_split(paths, labels)

    train_ds = make_dataset(train_paths, train_labels, True, args.batch_size)
    val_ds = make_dataset(val_paths, val_labels, False, args.batch_size)
    test_ds = make_dataset(test_paths, test_labels, False, args.batch_size)

    model, base, fine_tune_start = build_model(len(class_names), args.dropout, args.architecture)
    compile_model(model, 1e-3)

    class_weights_array = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(class_names)),
        y=train_labels,
    )
    class_weight = {idx: float(weight) for idx, weight in enumerate(class_weights_array)}

    if args.dry_run:
        print("Dry run complete.")
        print(f"Class root: {class_root}")
        print(f"Classes: {class_names}")
        print(f"Train/val/test: {len(train_paths)}/{len(val_paths)}/{len(test_paths)}")
        print(f"Class weights: {class_weight}")
        model.summary()
        return

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            MODELS_DIR / "best_model.keras",
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
        ),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        tf.keras.callbacks.CSVLogger(LOGS_DIR / "training_log.csv"),
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=callbacks,
        class_weight=class_weight,
    )

    if args.fine_tune_epochs > 0:
        base.trainable = True
        for layer in base.layers[:fine_tune_start]:
            layer.trainable = False
        compile_model(model, 1e-5)
        fine_history = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=args.fine_tune_epochs,
            callbacks=callbacks,
            class_weight=class_weight,
        )
        for key, values in fine_history.history.items():
            history.history.setdefault(key, []).extend(values)

    model = tf.keras.models.load_model(MODELS_DIR / "best_model.keras")
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    probabilities = model.predict(test_ds, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)
    y_true = np.concatenate([y.numpy() for _, y in test_ds], axis=0)

    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    cm = confusion_matrix(y_true, y_pred)

    plot_history(history, FIGURES_DIR / "training_curves.png")
    save_confusion_matrix(cm, class_names, FIGURES_DIR / "confusion_matrix.png")

    metrics = {
        "class_names": class_names,
        "train_images": int(len(train_paths)),
        "validation_images": int(len(val_paths)),
        "test_images": int(len(test_paths)),
        "test_loss": float(test_loss),
        "test_accuracy": float(test_acc),
        "image_size": IMAGE_SIZE,
        "batch_size": args.batch_size,
        "architecture": args.architecture,
        "class_weight": class_weight,
    }
    (MODELS_DIR / "class_names.json").write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (REPORTS_DIR / "classification_report.txt").write_text(report, encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print(report)
    print(f"Saved model: {MODELS_DIR / 'best_model.keras'}")


if __name__ == "__main__":
    main()
