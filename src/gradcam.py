import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("KERAS_HOME", str(Path(__file__).resolve().parents[1] / ".keras"))

import matplotlib
import numpy as np
from PIL import Image
import tensorflow as tf

from config import FIGURES_DIR, IMAGE_SIZE, MODELS_DIR


def load_image_array(path: Path) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize(IMAGE_SIZE)
    return np.asarray(image, dtype=np.float32)


def find_last_feature_layer(model: tf.keras.Model) -> tf.keras.layers.Layer:
    for layer in reversed(model.layers):
        output_shape = getattr(layer, "output_shape", None)
        if output_shape is None and hasattr(layer, "output"):
            output_shape = layer.output.shape
        if output_shape is not None and len(output_shape) == 4:
            return layer
    raise ValueError("Could not find a 4D feature layer for Grad-CAM.")


def make_gradcam_heatmap(model: tf.keras.Model, image_batch: tf.Tensor, class_index: int | None = None) -> np.ndarray:
    target_layer = find_last_feature_layer(model)

    with tf.GradientTape() as tape:
        x = image_batch
        feature_maps = None
        for layer in model.layers[1:]:
            x = layer(x, training=False)
            if layer is target_layer:
                feature_maps = x
        predictions = x
        if feature_maps is None:
            raise ValueError("Could not capture feature maps for Grad-CAM.")
        if class_index is None:
            class_index = int(tf.argmax(predictions[0]))
        class_score = predictions[:, class_index]

    grads = tape.gradient(class_score, feature_maps)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = feature_maps[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def save_overlay(original: np.ndarray, heatmap: np.ndarray, output_path: Path, alpha: float) -> None:
    heatmap = Image.fromarray(np.uint8(255 * heatmap)).resize((original.shape[1], original.shape[0]))
    heatmap = np.asarray(heatmap)
    color_map = matplotlib.colormaps["jet"]
    colored_heatmap = np.uint8(color_map(heatmap)[:, :, :3] * 255)
    overlay = np.uint8((1 - alpha) * original + alpha * colored_heatmap)
    Image.fromarray(overlay).save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM explanation for one image.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=MODELS_DIR / "best_model.keras")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--alpha", type=float, default=0.4)
    args = parser.parse_args()

    class_names = json.loads((MODELS_DIR / "class_names.json").read_text(encoding="utf-8"))
    model = tf.keras.models.load_model(args.model)

    original = load_image_array(args.image)
    image_batch = tf.expand_dims(tf.convert_to_tensor(original), axis=0)
    probabilities = model.predict(image_batch, verbose=0)[0]
    class_index = int(np.argmax(probabilities))
    prediction = class_names[class_index]

    heatmap = make_gradcam_heatmap(model, image_batch, class_index)
    output_path = args.output or FIGURES_DIR / f"gradcam_{args.image.stem}.png"
    save_overlay(original, heatmap, output_path, args.alpha)

    print(f"Image: {args.image}")
    print(f"Prediction: {prediction} ({probabilities[class_index] * 100:.2f}%)")
    print(f"Saved Grad-CAM: {output_path}")


if __name__ == "__main__":
    main()
