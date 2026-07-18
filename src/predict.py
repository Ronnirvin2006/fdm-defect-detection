import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf

from config import IMAGE_SIZE, MODELS_DIR


def load_image(path: Path):
    image = tf.io.read_file(str(path))
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image = tf.image.resize(image, IMAGE_SIZE)
    return tf.expand_dims(tf.cast(image, tf.float32), axis=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict FDM defect class for one image.")
    parser.add_argument("image", type=Path)
    parser.add_argument("--model", type=Path, default=MODELS_DIR / "best_model.keras")
    args = parser.parse_args()

    class_names = json.loads((MODELS_DIR / "class_names.json").read_text(encoding="utf-8"))
    model = tf.keras.models.load_model(args.model)
    probs = model.predict(load_image(args.image), verbose=0)[0]
    order = np.argsort(probs)[::-1]

    print(f"Image: {args.image}")
    print(f"Prediction: {class_names[order[0]]} ({probs[order[0]] * 100:.2f}%)")
    print("Top classes:")
    for idx in order[:5]:
        print(f"  {class_names[idx]}: {probs[idx] * 100:.2f}%")


if __name__ == "__main__":
    main()
