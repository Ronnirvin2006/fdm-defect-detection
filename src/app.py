import json
import os
from pathlib import Path

os.environ.setdefault("KERAS_HOME", str(Path(__file__).resolve().parents[1] / ".keras"))

import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

from config import IMAGE_SIZE, MODELS_DIR
from defect_knowledge import recommendation_for


st.set_page_config(page_title="FDM Defect Detection", layout="wide")


@st.cache_resource
def load_model_and_classes():
    model = tf.keras.models.load_model(MODELS_DIR / "best_model.keras")
    class_names = json.loads((MODELS_DIR / "class_names.json").read_text(encoding="utf-8"))
    return model, class_names


def preprocess(image: Image.Image) -> tf.Tensor:
    image = image.convert("RGB").resize(IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32)
    return tf.expand_dims(tf.convert_to_tensor(array), axis=0)


def predict(image: Image.Image):
    model, class_names = load_model_and_classes()
    probabilities = model.predict(preprocess(image), verbose=0)[0]
    order = np.argsort(probabilities)[::-1]
    return class_names, probabilities, order


def show_prediction(image: Image.Image) -> None:
    class_names, probabilities, order = predict(image)
    top_idx = int(order[0])
    predicted_class = class_names[top_idx]
    confidence = float(probabilities[top_idx] * 100)
    knowledge = recommendation_for(predicted_class)

    left, right = st.columns([1, 1])
    with left:
        st.image(image, caption="Input image", use_container_width=True)
    with right:
        st.metric("Prediction", predicted_class)
        st.metric("Confidence", f"{confidence:.2f}%")
        st.write("Class confidence")
        st.bar_chart({class_names[idx]: float(probabilities[idx] * 100) for idx in order})

    st.subheader("Defect Interpretation")
    st.write(knowledge["description"])

    col1, col2 = st.columns(2)
    with col1:
        st.write("Possible causes")
        for cause in knowledge["possible_causes"]:
            st.write(f"- {cause}")
    with col2:
        st.write("Corrective actions")
        for action in knowledge["corrective_actions"]:
            st.write(f"- {action}")


def main() -> None:
    st.title("AI-Based FDM 3D Printing Defect Detection")
    st.caption("Current trained model detects: Cracking, Layer_shifting, Off_platform, Stringing, and Warping.")

    tab_upload, tab_camera = st.tabs(["Upload Image", "Camera Snapshot"])

    with tab_upload:
        uploaded = st.file_uploader("Upload a 3D print image", type=["jpg", "jpeg", "png", "bmp", "webp"])
        if uploaded:
            show_prediction(Image.open(uploaded))

    with tab_camera:
        camera_image = st.camera_input("Take a live camera snapshot")
        if camera_image:
            show_prediction(Image.open(camera_image))

    st.divider()
    st.write(
        "More defect classes such as under-extrusion, over-extrusion, nozzle clog, blobs/zits, "
        "and no-defect require additional labeled training data before the model can reliably predict them."
    )


if __name__ == "__main__":
    main()
