import json
import os
import threading
from pathlib import Path
from tempfile import TemporaryDirectory

os.environ.setdefault("KERAS_HOME", str(Path(__file__).resolve().parents[1] / ".keras"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st

ENABLE_WEBRTC = os.environ.get("FDM_ENABLE_WEBRTC", "1") == "1"
WEBRTC_IMPORT_ERROR = None
if ENABLE_WEBRTC:
    try:
        import av
        from streamlit_webrtc import RTCConfiguration, VideoProcessorBase, WebRtcMode, webrtc_streamer
    except Exception as exc:
        ENABLE_WEBRTC = False
        WEBRTC_IMPORT_ERROR = str(exc)
        av = None
        RTCConfiguration = None
        VideoProcessorBase = object
        WebRtcMode = None
        webrtc_streamer = None
else:
    av = None
    RTCConfiguration = None
    VideoProcessorBase = object
    WebRtcMode = None
    webrtc_streamer = None

import tensorflow as tf

from config import IMAGE_SIZE, MODELS_DIR
from defect_knowledge import recommendation_for


st.set_page_config(page_title="FDM Defect Detection", layout="wide")
RTC_CONFIGURATION = (
    RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    if ENABLE_WEBRTC
    else None
)


@st.cache_resource
def load_model_and_classes():
    model = tf.keras.models.load_model(MODELS_DIR / "best_model.keras", compile=False)
    class_names = json.loads((MODELS_DIR / "class_names.json").read_text(encoding="utf-8"))
    return model, class_names


def safe_image(image_file) -> Image.Image:
    image = Image.open(image_file).convert("RGB")
    image.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
    return image.copy()


def preprocess(image: Image.Image) -> tf.Tensor:
    image = image.convert("RGB").resize(IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32)
    return tf.expand_dims(tf.convert_to_tensor(array), axis=0)


def predict(image: Image.Image):
    model, class_names = load_model_and_classes()
    probabilities = model.predict(preprocess(image), verbose=0)[0]
    order = np.argsort(probabilities)[::-1]
    return class_names, probabilities, order


def prediction_payload(class_names, probabilities, order) -> dict:
    top_idx = int(order[0])
    predicted_class = class_names[top_idx]
    confidence = float(probabilities[top_idx] * 100)
    knowledge = recommendation_for(predicted_class)
    return {
        "class_names": class_names,
        "probabilities": probabilities,
        "order": order,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "knowledge": knowledge,
    }


def risk_level(predicted_class: str, confidence: float) -> str:
    if predicted_class == "No_defect":
        return "Normal" if confidence >= 70 else "Uncertain normal"
    if predicted_class in {"Off_platform", "Spaghetti"}:
        return "Critical"
    if predicted_class in {"Under_extrusion", "Warping", "Layer_shifting"}:
        return "High" if confidence >= 70 else "Medium"
    return "Medium" if confidence >= 70 else "Low-confidence"


def ai_feedback_text(payload: dict) -> list[str]:
    predicted_class = payload["predicted_class"]
    confidence = payload["confidence"]
    level = risk_level(predicted_class, confidence)
    if predicted_class == "No_defect":
        headline = "The print currently looks healthy. Continue monitoring for changes."
    elif level == "Critical":
        headline = "This defect can quickly ruin the print. Pause or stop the printer and inspect it."
    elif confidence < 60:
        headline = "The model is uncertain. Use this as a warning and visually inspect the print."
    else:
        headline = "The model sees a likely defect. Check the suggested cause and correction."
    return [
        f"AI feedback: {headline}",
        f"Risk level: {level}",
        f"Detected class: {predicted_class} with {confidence:.2f}% confidence",
    ]


def confidence_chart(payload: dict) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Class": [payload["class_names"][idx] for idx in payload["order"]],
            "Confidence": [float(payload["probabilities"][idx] * 100) for idx in payload["order"]],
        }
    ).set_index("Class")


def show_feedback(payload: dict) -> None:
    knowledge = payload["knowledge"]
    for line in ai_feedback_text(payload):
        st.write(line)

    col1, col2 = st.columns(2)
    with col1:
        st.write("Possible causes")
        for cause in knowledge["possible_causes"]:
            st.write(f"- {cause}")
    with col2:
        st.write("Corrective actions")
        for action in knowledge["corrective_actions"]:
            st.write(f"- {action}")


def show_prediction(image: Image.Image) -> None:
    try:
        payload = prediction_payload(*predict(image))
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return

    left, right = st.columns([1, 1])
    with left:
        st.image(image, caption="Input image", use_container_width=True)
    with right:
        st.metric("Prediction", payload["predicted_class"])
        st.metric("Confidence", f"{payload['confidence']:.2f}%")
        st.write("Class confidence")
        st.bar_chart(confidence_chart(payload))

    st.subheader("AI Feedback")
    st.write(payload["knowledge"]["description"])
    show_feedback(payload)


def read_video_samples(video_file, max_frames: int = 36):
    """Read a bounded set of evenly spaced frames without retaining the upload."""
    video_bytes = video_file.getvalue()
    suffix = Path(getattr(video_file, "name", "video.mp4")).suffix or ".mp4"
    prediction_dir = MODELS_DIR.parent / "outputs" / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)

    with TemporaryDirectory(dir=prediction_dir) as temporary_dir:
        video_path = Path(temporary_dir) / f"input{suffix}"
        video_path.write_bytes(video_bytes)
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError("The uploaded video could not be opened. Use MP4, MOV, AVI, or WEBM.")

        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        fps = fps if np.isfinite(fps) and fps > 0 else 0.0
        if total_frames <= 0:
            capture.release()
            raise ValueError("The video does not contain readable frames.")

        selected_indices = np.linspace(
            0, total_frames - 1, num=min(max_frames, total_frames), dtype=int
        )
        selected_indices = set(int(index) for index in selected_indices)
        frames = []
        timestamps = []
        frame_index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index in selected_indices:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(rgb))
                timestamps.append(frame_index / fps if fps else float(frame_index))
            frame_index += 1
        capture.release()

    if not frames:
        raise ValueError("No readable video frames were found.")
    duration = (total_frames / fps) if fps else None
    return frames, timestamps, fps, total_frames, duration


def analyze_video(video_file) -> dict:
    frames, timestamps, fps, total_frames, duration = read_video_samples(video_file)
    model, class_names = load_model_and_classes()
    batch = np.stack(
        [np.asarray(frame.resize(IMAGE_SIZE), dtype=np.float32) for frame in frames], axis=0
    )
    probabilities = model.predict(batch, verbose=0)
    mean_probabilities = probabilities.mean(axis=0)
    mean_order = np.argsort(mean_probabilities)[::-1]
    payload = prediction_payload(class_names, mean_probabilities, mean_order)

    timeline = []
    for timestamp, frame_probabilities in zip(timestamps, probabilities):
        order = np.argsort(frame_probabilities)[::-1]
        top_index = int(order[0])
        timeline.append(
            {
                "Time (s)": round(timestamp, 2),
                "Prediction": class_names[top_index],
                "Confidence (%)": round(float(frame_probabilities[top_index] * 100), 2),
            }
        )

    class_counts = pd.Series(row["Prediction"] for row in timeline).value_counts()
    observed_problems = [name for name in class_counts.index if name != "No_defect"]
    return {
        "payload": payload,
        "frames": frames,
        "timeline": timeline,
        "fps": fps,
        "total_frames": total_frames,
        "duration": duration,
        "sampled_frames": len(frames),
        "class_counts": class_counts,
        "observed_problems": observed_problems,
    }


def show_video_analysis(video_file) -> None:
    video_bytes = video_file.getvalue()
    st.video(video_bytes)
    try:
        with st.spinner("Sampling video frames and checking for print defects..."):
            result = analyze_video(video_file)
    except Exception as exc:
        st.error(f"Video analysis failed: {exc}")
        return

    payload = result["payload"]
    duration_text = f"{result['duration']:.1f} s" if result["duration"] else "unknown duration"
    st.caption(
        f"Analysed {result['sampled_frames']} evenly spaced frames from {duration_text}. "
        "The uploaded video is removed after analysis."
    )
    st.subheader("Video result")
    left, right = st.columns([1, 1])
    with left:
        st.metric("Most likely overall condition", payload["predicted_class"])
        st.metric("Average confidence", f"{payload['confidence']:.2f}%")
    with right:
        if result["observed_problems"]:
            st.write("Problems observed in sampled frames")
            for problem in result["observed_problems"]:
                st.write(f"- {problem} ({int(result['class_counts'][problem])} frame(s))")
        else:
            st.write("No trained defect was the top prediction in the sampled frames.")

    st.subheader("AI Feedback")
    st.write(payload["knowledge"]["description"])
    show_feedback(payload)
    st.write("Frame-by-frame sampled predictions")
    st.dataframe(pd.DataFrame(result["timeline"]), hide_index=True, use_container_width=True)
    st.write("Sampled frames")
    st.image(
        result["frames"],
        caption=[f"{row['Time (s)']} s: {row['Prediction']}" for row in result["timeline"]],
        width=180,
    )


class LiveDefectProcessor(VideoProcessorBase):
    def __init__(self):
        self.model, self.class_names = load_model_and_classes()
        self.lock = threading.Lock()
        self.frame_count = 0
        self.last_prediction = "Waiting..."
        self.last_confidence = 0.0
        self.last_feedback = "Point the camera at the print."

    def recv(self, frame):
        image = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        if self.frame_count % 10 == 1:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb)
            probabilities = self.model.predict(preprocess(pil_image), verbose=0)[0]
            order = np.argsort(probabilities)[::-1]
            payload = prediction_payload(self.class_names, probabilities, order)
            feedback = ai_feedback_text(payload)[0].replace("AI feedback: ", "")
            with self.lock:
                self.last_prediction = payload["predicted_class"]
                self.last_confidence = payload["confidence"]
                self.last_feedback = feedback

        with self.lock:
            label = f"{self.last_prediction} ({self.last_confidence:.1f}%)"
            feedback = self.last_feedback

        color = (0, 220, 0) if self.last_prediction == "No_defect" else (0, 165, 255)
        cv2.rectangle(image, (12, 12), (620, 92), (0, 0, 0), thickness=-1)
        cv2.putText(image, label, (24, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(image, feedback[:68], (24, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        return av.VideoFrame.from_ndarray(image, format="bgr24")


def main() -> None:
    st.title("AI-Based FDM 3D Printing Defect Detection")
    _, class_names = load_model_and_classes()
    st.caption(f"Current trained model detects: {', '.join(class_names)}.")

    tab_upload, tab_video, tab_live, tab_camera = st.tabs(
        ["Upload Image", "Upload Video", "Live Webcam", "Camera Snapshot"]
    )

    with tab_upload:
        uploaded = st.file_uploader("Upload a 3D print image", type=["jpg", "jpeg", "png", "bmp", "webp"])
        if uploaded:
            show_prediction(safe_image(uploaded))

    with tab_video:
        st.write("Upload a short printer-monitoring video. The app samples frames across the clip and reports the conditions it sees.")
        uploaded_video = st.file_uploader(
            "Upload a defect-monitoring video",
            type=["mp4", "mov", "avi", "webm"],
            key="video-upload",
            help="Short clips work best. Up to 36 evenly spaced frames are analysed.",
        )
        if uploaded_video and st.button("Analyse video", type="primary"):
            show_video_analysis(uploaded_video)

    with tab_live:
        if ENABLE_WEBRTC:
            st.write("Start the camera and point it at the print. Prediction is refreshed every few frames.")
            webrtc_streamer(
                key="fdm-live-detection",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                video_processor_factory=LiveDefectProcessor,
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,
            )
        else:
            st.warning(
                "Continuous live webcam streaming is unavailable in this deployment. "
                "Use Camera Snapshot or Upload Video instead."
            )
            if WEBRTC_IMPORT_ERROR:
                st.caption(f"WebRTC setup message: {WEBRTC_IMPORT_ERROR}")

    with tab_camera:
        camera_image = st.camera_input("Take a live camera snapshot")
        if camera_image:
            show_prediction(safe_image(camera_image))

    st.divider()
    st.write(
        "The live camera is a prototype monitor. For final quality decisions, use the confidence score "
        "and visually confirm the print condition."
    )


if __name__ == "__main__":
    main()
