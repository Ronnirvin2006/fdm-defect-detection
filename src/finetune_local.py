"""Fine-tune the already downloaded project model without network access."""

import argparse
import json
import csv
import time
import hashlib
import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

from config import DATA_RAW, IMAGE_SIZE, MODELS_DIR, REPORTS_DIR, SEED
from inspect_dataset import IMAGE_EXTS, find_class_root


AUTOTUNE = tf.data.AUTOTUNE


def load_fixed_split(path, available_paths, class_names):
    saved = json.loads(path.read_text())
    result = []
    known = {str(Path(p).resolve(strict=True)) for p in available_paths}
    seen = set()
    for part in ('train', 'validation'):
        entries = saved[part]
        resolved = [str(Path(p).resolve(strict=True)) for p in entries]
        if not entries or len(resolved) != len(set(resolved)) or seen.intersection(resolved):
            raise ValueError('Split is empty or contains duplicate/overlapping source files')
        if not set(resolved).issubset(known):
            raise ValueError('Split contains files outside the current dataset')
        labels = np.array([class_names.index(Path(p).parent.name) for p in entries], dtype=np.int32)
        if len(set(labels)) != len(class_names):
            raise ValueError(f'{part} must contain all classes')
        result.append((np.array(entries), labels))
        seen.update(resolved)
    # Confirm the supplied manifest preserves the established recording/day split.
    expected_train, expected_val = split_files(np.array(available_paths), np.array([class_names.index(Path(p).parent.name) for p in available_paths]))
    if not set(result[0][0]).issubset(set(expected_train[0])) or set(result[1][0]) != set(expected_val[0]):
        raise ValueError('Manifest does not match the established grouped split')
    return tuple(result)


def training_weights(labels, mode):
    weights = compute_class_weight('balanced', classes=np.arange(len(set(labels))), y=labels)
    if mode == 'none':
        weights = np.ones_like(weights)
    elif mode == 'sqrt':
        weights = np.sqrt(weights)
        weights /= np.mean(weights[labels])
    return {i: float(w) for i, w in enumerate(weights)}


def boost_class_weights(weights, class_names, selected_names, factor):
    selected = {name.strip() for name in selected_names.split(',') if name.strip()}
    unknown = selected.difference(class_names)
    if unknown:
        raise ValueError(f'Unknown class in --boost-classes: {sorted(unknown)}')
    if factor < 1.0:
        raise ValueError('--boost-factor must be at least 1.0')
    return {
        index: float(weight * factor if class_names[index] in selected else weight)
        for index, weight in weights.items()
    }


def atomic_save(model, destination):
    temporary = destination.with_name(destination.stem + '.saving.keras')
    model.save(temporary)
    os.replace(temporary, destination)


class SaveLatest(tf.keras.callbacks.Callback):
    def __init__(self, path):
        super().__init__()
        self.path = path

    def on_epoch_end(self, epoch, logs=None):
        atomic_save(self.model, self.path)


def configure_device(mixed_precision):
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    if mixed_precision and gpus:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
    print(f"TensorFlow: {tf.__version__}")
    print(f"GPUs: {gpus}")
    if not gpus:
        print("WARNING: No TensorFlow GPU detected; training will use the CPU.")


def collect_files(class_root: Path):
    class_names = sorted(path.name for path in class_root.iterdir() if path.is_dir())
    class_to_id = {name: index for index, name in enumerate(class_names)}
    paths, labels = [], []
    for class_name in class_names:
        for path in sorted((class_root / class_name).rglob("*")):
            if path.suffix.lower() in IMAGE_EXTS:
                paths.append(str(path))
                labels.append(class_to_id[class_name])
    return np.asarray(paths), np.asarray(labels, dtype=np.int32), class_names


def split_files(paths, labels):
    from sklearn.model_selection import GroupShuffleSplit
    source = Path(__file__).resolve().parents[1] / 'data/sources/Printing_Errors/general_data/all_images_no_filter.csv'
    with source.open() as handle:
        recordings = {r['image']: r['recording'] for r in csv.DictReader(handle, delimiter=';')}
    groups = []
    for path in paths:
        name = Path(path).resolve(strict=True).name
        if name in recordings:
            groups.append('nimbus:' + recordings[name])
        elif name.startswith('Image_'):
            groups.append('wengmhu:' + name.split('_')[1][:8])
        else:
            raise ValueError(f'No source group for {path}')
    groups = np.array(groups)
    candidates = []
    for train, val in GroupShuffleSplit(n_splits=500, test_size=.2, random_state=SEED).split(paths, labels, groups):
        if len(set(labels[train])) != len(set(labels)) or len(set(labels[val])) != len(set(labels)):
            continue
        score = np.mean(abs(np.bincount(labels[val]) / np.bincount(labels) - .2))
        candidates.append((score, train, val))
    if not candidates:
        raise ValueError('Cannot form grouped validation containing all classes')
    _, train, val = min(candidates, key=lambda item: item[0])
    assert not set(groups[train]) & set(groups[val])
    return (paths[train], labels[train]), (paths[val], labels[val])


def decode_image(path, label):
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMAGE_SIZE)
    return tf.cast(image, tf.float32), label


def make_dataset(paths, labels, training, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    return dataset.map(decode_image, num_parallel_calls=2).batch(batch_size).prefetch(1)


def find_backbone(model):
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model) and layer.name != "augmentation":
            return layer
    return None


def compile_model(model, learning_rate):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        jit_compile=False,
    )


class TimeLimit(tf.keras.callbacks.Callback):
    def __init__(self, hours):
        super().__init__()
        self.deadline = time.monotonic() + hours * 3600
        self.expired = False

    def on_train_batch_end(self, batch, logs=None):
        if time.monotonic() >= self.deadline:
            self.expired = True
            self.model.stop_training = True


def main():
    parser = argparse.ArgumentParser(description="Offline fine-tuning from models/best_model.keras")
    parser.add_argument("--model", type=Path, default=MODELS_DIR / "best_model.keras")
    parser.add_argument("--output-model", type=Path, default=MODELS_DIR / "local_finetuned_model.keras")
    parser.add_argument("--epochs", type=int, default=5, help="Head fine-tuning epochs")
    parser.add_argument("--fine-tune-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--fine-tune-lr", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--mixed-precision", action="store_true", help="Sets policy for new layers; saved layers retain their stored policy.")
    parser.add_argument('--smoke-test', action='store_true')
    parser.add_argument('--hours', type=float, default=2.5)
    parser.add_argument('--max-per-class', type=int, default=0, help='0 uses all training images')
    parser.add_argument('--split', type=Path, help='Reuse a saved grouped split.json')
    parser.add_argument('--class-weight-mode', choices=['balanced', 'sqrt', 'none'], default='balanced')
    parser.add_argument('--boost-classes', default='', help='Comma-separated classes needing extra recall/precision focus')
    parser.add_argument('--boost-factor', type=float, default=1.0, help='Multiplier applied to --boost-classes weights')
    parser.add_argument('--minutes', type=float, help='Training budget; baseline and final evaluation take extra time')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.minutes is not None:
        args.hours = args.minutes / 60
    if args.hours <= 0 or args.batch_size <= 0 or args.max_per_class < 0 or min(args.epochs, args.fine_tune_epochs) < 0 or args.epochs + args.fine_tune_epochs == 0:
        parser.error('Invalid time, batch size, image cap or epoch count')

    configure_device(args.mixed_precision)
    if not tf.config.list_physical_devices('GPU'):
        raise RuntimeError('GPU required for this local run')
    tf.keras.utils.set_random_seed(SEED)
    if args.output_model.resolve() == args.model.resolve() or args.output_model.exists():
        raise ValueError('Output must be a new file, separate from the source model')

    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")

    class_root = find_class_root(DATA_RAW)
    paths, labels, class_names = collect_files(class_root)
    if len(class_names) < 2 or len(paths) == 0:
        raise RuntimeError(f"No usable class-folder dataset found under {class_root}")

    model = tf.keras.models.load_model(args.model, compile=False)
    mapping = args.model.with_suffix('.classes.json')
    if not mapping.exists():
        mapping = args.model.parent / 'class_names.json'
    saved_names = json.loads(mapping.read_text())
    if class_names != saved_names:
        raise ValueError('Dataset label order does not match the saved model')
    if model.output_shape[-1] != len(class_names):
        raise ValueError(
            f"Dataset has {len(class_names)} classes {class_names}, but model outputs "
            f"{model.output_shape[-1]}. Use the matching dataset/model pair."
        )

    if args.split:
        (train_paths, train_labels), (val_paths, val_labels) = load_fixed_split(args.split, paths, class_names)
    else:
        (train_paths, train_labels), (val_paths, val_labels) = split_files(paths, labels)
    if args.max_per_class:
        rng = np.random.default_rng(SEED)
        selected = np.concatenate([rng.permutation(np.flatnonzero(train_labels == i))[:args.max_per_class] for i in range(len(class_names))])
        train_paths, train_labels = train_paths[selected], train_labels[selected]
    print('Training / grouped validation:', len(train_paths), len(val_paths), flush=True)
    train_ds = make_dataset(train_paths, train_labels, True, args.batch_size)
    val_ds = make_dataset(val_paths, val_labels, False, args.batch_size)
    class_weight = training_weights(train_labels, args.class_weight_mode)
    if args.boost_classes:
        class_weight = boost_class_weights(class_weight, class_names, args.boost_classes, args.boost_factor)
    print('Class weights:', class_weight, flush=True)
    if args.dry_run:
        print('Dry run passed: model, classes, GPU and fixed split validated. No training performed.')
        return

    backbone = find_backbone(model)
    if backbone is None:
        raise RuntimeError('Saved backbone missing')
    backbone.trainable = False
    compile_model(model, args.learning_rate)
    if args.smoke_test:
        model.fit(train_ds.take(8), epochs=1, verbose=2)
        backbone.trainable = True
        for layer in backbone.layers:
            layer.trainable = False
        for layer in backbone.layers[-30:]:
            layer.trainable = not isinstance(layer, tf.keras.layers.BatchNormalization)
        compile_model(model, args.fine_tune_lr)
        model.fit(train_ds.take(8), epochs=1, verbose=2)
        print('GPU head and backbone training passed.', flush=True)
        return
    run_dir = REPORTS_DIR / ('local_' + time.strftime('%Y%m%d_%H%M%S'))
    run_dir.mkdir(exist_ok=False)
    print('Run reports:', run_dir, flush=True)
    settings = {k:str(v) if isinstance(v, Path) else v for k,v in vars(args).items()}
    settings['source_model_sha256'] = hashlib.sha256(args.model.read_bytes()).hexdigest()
    settings['seed'] = SEED
    settings['continuation'] = 'Loads saved weights with a new optimiser; not exact interrupted-optimiser resume.'
    (run_dir / 'settings.json').write_text(json.dumps(settings, indent=2))
    (run_dir / 'split.json').write_text(json.dumps({'train':train_paths.tolist(), 'validation':val_paths.tolist()}, indent=2))
    baseline = model.evaluate(val_ds, verbose=1)
    (run_dir / 'baseline.json').write_text(json.dumps({'validation_loss':baseline[0], 'validation_accuracy':baseline[1]}))
    args.output_model.parent.mkdir(parents=True, exist_ok=True)
    atomic_save(model, args.output_model)
    latest_path = args.output_model.with_name(args.output_model.stem + '_latest.keras')
    if latest_path.exists():
        raise ValueError(f'Latest checkpoint already exists: {latest_path}')
    args.output_model.with_suffix('.classes.json').write_text(json.dumps(class_names, indent=2))
    limit = TimeLimit(args.hours)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(args.output_model, monitor="val_loss", save_best_only=True, mode="min", initial_value_threshold=baseline[0]),
        limit,
        SaveLatest(latest_path),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=1, min_lr=1e-7, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=args.patience, restore_best_weights=True),
        tf.keras.callbacks.CSVLogger(run_dir / 'epochs.csv', append=True),
    ]

    backbone = find_backbone(model)
    if backbone is None:
        raise RuntimeError("Could not find the saved feature-extraction backbone in the model")

    backbone.trainable = False
    compile_model(model, args.learning_rate)
    interrupted = False
    try:
        if args.epochs:
            model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=args.epochs,
                callbacks=callbacks,
                class_weight=class_weight,
                verbose=1,
            )
    except KeyboardInterrupt:
        atomic_save(model, latest_path)
        interrupted = True
        print(f'Interrupted weights saved: {latest_path}', flush=True)

    if args.fine_tune_epochs > 0 and not limit.expired and not interrupted:
        model.load_weights(args.output_model)
        backbone.trainable = True
        for layer in backbone.layers[:-30]:
            layer.trainable = False
        for layer in backbone.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
        compile_model(model, args.fine_tune_lr)
        try:
            model.fit(
                train_ds,
                validation_data=val_ds,
                initial_epoch=args.epochs,
                epochs=args.epochs + args.fine_tune_epochs,
                callbacks=callbacks,
                class_weight=class_weight,
                verbose=1,
            )
        except KeyboardInterrupt:
            atomic_save(model, latest_path)
            interrupted = True
            print(f'Interrupted weights saved: {latest_path}', flush=True)

    (run_dir / 'status.json').write_text(json.dumps({'interrupted':interrupted, 'time_limit_reached':limit.expired, 'best_model':str(args.output_model), 'latest_model':str(latest_path)}, indent=2))
    if interrupted:
        print('Stopped by user; best checkpoint retained. Final evaluation skipped.', flush=True)
        return

    model = tf.keras.models.load_model(args.output_model, compile=False)
    model.compile(loss="sparse_categorical_crossentropy", metrics=["accuracy"], jit_compile=False)
    test_loss, test_accuracy = model.evaluate(val_ds, verbose=0)
    probabilities = model.predict(val_ds, verbose=0)
    y_pred = np.argmax(probabilities, axis=1)
    y_true = val_labels
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4, zero_division=0)
    np.savez_compressed(run_dir / 'validation_predictions.npz', paths=val_paths, labels=val_labels, probabilities=probabilities)
    np.savetxt(run_dir / 'confusion_matrix.csv', confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names))), fmt='%d', delimiter=',', header=','.join(class_names))
    from validate_external import save_confusion
    save_confusion(confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names))), class_names, run_dir / 'confusion_matrix.png', title='Grouped validation (not independent test)')
    metrics = {
        "model": str(args.output_model),
        "class_names": class_names,
        "train_images": int(len(train_paths)),
        "validation_images": int(len(val_paths)),
        "evaluation_note": "Grouped validation may overlap source model training history; not an independent test result.",
        "validation_loss": float(test_loss),
        "validation_accuracy": float(test_accuracy),
        "batch_size": args.batch_size,
        "offline": True,
        "class_weight": class_weight,
    }
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run_dir / "classification_report.txt").write_text(report, encoding="utf-8")
    (args.output_model.parent / "local_finetuned_class_names.json").write_text(
        json.dumps(class_names, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))
    print(report)
    print(f"Saved candidate model: {args.output_model}")
    print("The active models/best_model.keras was not replaced.")


if __name__ == "__main__":
    main()
