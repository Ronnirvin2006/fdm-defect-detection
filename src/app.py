import json
import os
import threading
import urllib.request
from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory

os.environ.setdefault("KERAS_HOME", str(Path(__file__).resolve().parents[1] / ".keras"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import hashlib
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
from scene_guard import assess_print_scene, blocked_prediction


st.set_page_config(page_title="FDM Quality Desk", page_icon="F", layout="wide", initial_sidebar_state="expanded")


def deployment_secret(name: str) -> str:
    """Read a deployment secret without requiring a local secrets file."""
    value = os.environ.get(name, "")
    if value:
        return value
    try:
        return str(st.secrets.get(name, ""))
    except Exception:
        return ""


def cloudflare_turn_servers() -> list[dict]:
    """Fetch short-lived TURN credentials when Cloudflare secrets are configured."""
    key_id = deployment_secret("CLOUDFLARE_TURN_KEY_ID")
    api_token = deployment_secret("CLOUDFLARE_TURN_API_TOKEN")
    if not key_id or not api_token:
        return []
    url = f"https://rtc.live.cloudflare.com/v1/turn/keys/{key_id}/credentials/generate-ice-servers"
    request = urllib.request.Request(
        url,
        headers={"Authorization": f"Bearer {api_token}", "User-Agent": "fdm-defect-detection"},
    )
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload.get("iceServers", [])
    except Exception:
        return []


turn_urls = [item.strip() for item in deployment_secret("FDM_TURN_URLS").split(",") if item.strip()]
if not turn_urls and deployment_secret("FDM_TURN_URL"):
    turn_urls = [deployment_secret("FDM_TURN_URL")]
turn_username = deployment_secret("FDM_TURN_USERNAME")
turn_credential = deployment_secret("FDM_TURN_CREDENTIAL")
ice_servers = [{"urls": ["stun:stun.l.google.com:19302"]}]
ice_servers.extend(cloudflare_turn_servers())
if turn_urls and turn_username and turn_credential:
    ice_servers.append(
        {
            "urls": turn_urls,
            "username": turn_username,
            "credential": turn_credential,
        }
    )
RTC_CONFIGURATION = (
    RTCConfiguration({"iceServers": ice_servers})
    if ENABLE_WEBRTC
    else None
)


def inject_dashboard_style() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stAppViewContainer"] { background: #101419; }
        [data-testid="stHeader"] { background: #101419; border-bottom: 1px solid #283039; }
        [data-testid="stSidebar"] { background: #151a20; border-right: 1px solid #283039; }
        [data-testid="stSidebar"] > div:first-child { padding-top: 1.1rem; }
        .block-container { max-width: 1440px; padding-top: 1.6rem; padding-bottom: 2.5rem; }
        .fdm-hero { padding: 1.3rem 1.45rem; border: 1px solid #303a44; border-radius: 8px;
            background: #171d24; margin-bottom: 1rem; }
        .fdm-kicker { color: #57c7c4; font-size: 0.72rem; letter-spacing: 0.12em; text-transform: uppercase; font-weight: 700; }
        .fdm-hero h1 { margin: 0.35rem 0 0.25rem; color: #f2f5f7; font-size: clamp(1.65rem, 3vw, 2.45rem); letter-spacing: 0; }
        .fdm-hero p { margin: 0; color: #aeb8c0; max-width: 760px; }
        .fdm-badge { display: inline-block; margin-top: 0.9rem; padding: 0.28rem 0.62rem;
            border: 1px solid #2d6869; border-radius: 5px; background: #173033; color: #8fe2dd; font-size: 0.76rem; }
        .fdm-section { color: #eef2f4; font-size: 1rem; font-weight: 700; letter-spacing: 0; margin: 0.7rem 0 0.45rem; }
        .fdm-muted { color: #9eafb5; font-size: 0.83rem; }
        div[data-testid="stMetric"] { background: #171d24; border: 1px solid #303a44; padding: 0.8rem;
            border-radius: 6px; min-height: 6.3rem; }
        div[data-testid="stMetricLabel"] { color: #9eafb5; }
        div[data-testid="stMetricValue"] { color: #f4f7f8; }
        .fdm-panel { background: #171d24; border: 1px solid #303a44; border-radius: 6px; padding: 1rem; }
        .fdm-statusline { display:flex; align-items:center; gap:.5rem; color:#b7c2c9; font-size:.84rem; }
        .fdm-statusdot { display:inline-block; width:8px; height:8px; border-radius:50%; background:#57c7c4; }
        .fdm-footnote { color: #93a6ad; border-top: 1px solid #283039; padding-top: 0.9rem; font-size: 0.82rem; }
        div[data-baseweb="tab-list"] { gap: 1.3rem; border-bottom: 1px solid #303a44; }
        button[data-baseweb="tab"] { color: #9eafb5; padding: .7rem .15rem; }
        button[data-baseweb="tab"][aria-selected="true"] { color: #72d8d3; }
        button[kind="primary"] { background: #1f8f91; border-color: #2aa9a6; }
        button[kind="primary"]:hover { background: #2aa9a6; }
        [data-testid="stFileUploader"] { background: #171d24; border: 1px dashed #49616b; border-radius: 6px; padding: .25rem; }
        @media (max-width: 800px) { .block-container { padding-left: 1rem; padding-right: 1rem; } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_session() -> None:
    st.session_state.setdefault("scan_history", [])
    st.session_state.setdefault("last_inspection", None)


def inspection_token(source: str, label: str, payload: dict, content: bytes | None = None) -> str:
    digest = hashlib.sha1()
    digest.update(source.encode("utf-8"))
    digest.update(label.encode("utf-8"))
    digest.update(f"{payload['predicted_class']}:{payload['confidence']:.5f}".encode("utf-8"))
    if content:
        digest.update(content)
    return digest.hexdigest()


def remember_inspection(source: str, label: str, payload: dict, content: bytes | None = None, detail: str = "") -> None:
    token = inspection_token(source, label, payload, content)
    history = st.session_state["scan_history"]
    if any(item["token"] == token for item in history):
        st.session_state["last_inspection"] = history[next(i for i, item in enumerate(history) if item["token"] == token)]
        return
    record = {
        "token": token,
        "time": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,
        "input": label,
        "prediction": payload["predicted_class"],
        "confidence (%)": round(payload["confidence"], 2),
        "risk": risk_level(payload["predicted_class"], payload["confidence"]),
        "detail": detail,
    }
    st.session_state["scan_history"] = [record, *history[:24]]
    st.session_state["last_inspection"] = record


def show_workspace_summary(class_names: list[str]) -> None:
    history = st.session_state["scan_history"]
    latest = st.session_state.get("last_inspection")
    st.markdown(
        "<div class='fdm-hero'><div class='fdm-kicker'>FDM quality desk / inspection workspace</div>"
        "<h1>Know when a print starts to drift.</h1>"
        "<p>Review a print image, analyse a short clip, or use a live camera view with one consistent defect-classification workflow.</p>"
        "<span class='fdm-badge'>Model ready · EfficientNetB0 · scene gate enabled · 8 classes</span></div>",
        unsafe_allow_html=True,
    )
    cols = st.columns(4)
    with cols[0]:
        st.metric("Latest prediction", latest["prediction"] if latest else "No scan yet")
    with cols[1]:
        st.metric("Confidence", f"{latest['confidence (%)']:.2f}%" if latest else "--")
    with cols[2]:
        st.metric("Risk", latest["risk"] if latest else "--")
    with cols[3]:
        st.metric("Session scans", len(history))
    st.caption(f"Coverage: {', '.join(class_names)}")


def show_sidebar(class_names: list[str]) -> None:
    with st.sidebar:
        st.markdown("# FDM Quality Desk")
        st.caption("Operator-focused print inspection")
        st.divider()
        st.markdown("### System status")
        st.markdown("<div class='fdm-statusline'><span class='fdm-statusdot'></span><span>Model loaded and ready</span></div>", unsafe_allow_html=True)
        st.write("**Architecture:** EfficientNetB0")
        st.write("**Input:** 224 × 224 RGB")
        st.write(f"**Coverage:** {len(class_names)} classes")
        st.divider()
        st.markdown("### Inspection sources")
        st.write("Image upload  ·  Video review  ·  Live webcam  ·  Snapshot")
        st.caption("Browser camera access requires HTTPS and permission.")
        st.divider()
        st.markdown("### Session tools")
        history = st.session_state["scan_history"]
        if history:
            report = pd.DataFrame(history).drop(columns=["token"], errors="ignore")
            st.download_button(
                "Download inspection log",
                report.to_csv(index=False).encode("utf-8"),
                "fdm_inspection_log.csv",
                "text/csv",
                use_container_width=True,
            )
            if st.button("Clear session history", use_container_width=True):
                st.session_state["scan_history"] = []
                st.session_state["last_inspection"] = None
                st.rerun()
        else:
            st.caption("Your inspection history will appear here during this browser session.")
        st.divider()
        st.caption("Decision support only. Confirm important printer actions visually before changing hardware or settings.")


def show_recent_inspections() -> None:
    history = st.session_state["scan_history"]
    st.markdown("<div class='fdm-section'>Recent inspections</div>", unsafe_allow_html=True)
    if not history:
        st.info("No inspections in this session yet. Start with an image or video upload.")
        return
    table = pd.DataFrame(history).drop(columns=["token", "detail"], errors="ignore")
    st.dataframe(table, hide_index=True, use_container_width=True)


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
        "scene": {"valid": True, "status": "Print scene accepted", "reason": "Printable structure is visible enough to inspect."},
    }


def scene_payload(image: Image.Image, class_names, probabilities, order) -> dict:
    frame = cv2.cvtColor(np.asarray(image.convert("RGB")), cv2.COLOR_RGB2BGR)
    scene = assess_print_scene(frame)
    if scene["valid"]:
        payload = prediction_payload(class_names, probabilities, order)
        payload["scene"] = scene
        return payload
    blocked = blocked_prediction(scene)
    blocked["predicted_class"] = blocked["class_name"]
    blocked["class_names"] = class_names
    blocked["probabilities"] = np.zeros(len(class_names), dtype=np.float32)
    blocked["order"] = np.arange(len(class_names))
    blocked["confidence"] = 0.0
    blocked["knowledge"] = {
        "description": blocked["description"],
        "possible_causes": blocked["possible_causes"],
        "corrective_actions": blocked["corrective_actions"],
    }
    return blocked


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


def image_quality(image: Image.Image) -> dict:
    grayscale = np.asarray(image.convert("L"), dtype=np.float32)
    return {
        "brightness": float(grayscale.mean()),
        "sharpness": float(cv2.Laplacian(grayscale, cv2.CV_32F).var()),
    }


def show_prediction(image: Image.Image) -> dict | None:
    try:
        class_names, probabilities, order = predict(image)
        payload = scene_payload(image, class_names, probabilities, order)
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
        return None

    left, right = st.columns([1, 1])
    with left:
        st.image(image, caption="Input image", use_container_width=True)
    with right:
        st.metric("Decision", payload["predicted_class"])
        st.metric("Confidence", f"{payload['confidence']:.2f}%")
        st.write("Class confidence")
        st.bar_chart(confidence_chart(payload))

    quality = image_quality(image)
    if quality["brightness"] < 35:
        st.warning("Image quality check: the frame is very dark, so confidence may be less reliable.")
    elif quality["brightness"] > 225:
        st.warning("Image quality check: the frame is overexposed, so defect details may be lost.")
    if quality["sharpness"] < 35:
        st.warning("Image quality check: the frame appears blurred. Hold the camera steady and rescan.")

    if not payload["scene"]["valid"]:
        st.warning(f"Scene rejected: {payload['scene']['reason']}")
    st.subheader("AI feedback")
    st.write(payload["knowledge"]["description"])
    show_feedback(payload)
    return payload


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
    raw_probabilities = model.predict(batch, verbose=0)
    valid_indices = []
    scenes = []
    for index, frame in enumerate(frames):
        scene = assess_print_scene(cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))
        scenes.append(scene)
        if scene["valid"]:
            valid_indices.append(index)
    if not valid_indices:
        payload = blocked_prediction(scenes[0])
        payload["predicted_class"] = payload["class_name"]
        payload["class_names"] = class_names
        payload["probabilities"] = np.zeros(len(class_names), dtype=np.float32)
        payload["order"] = np.arange(len(class_names))
        payload["knowledge"] = {"description": payload["description"], "possible_causes": payload["possible_causes"], "corrective_actions": payload["corrective_actions"]}
        probabilities = raw_probabilities
        mean_probabilities = np.zeros(len(class_names), dtype=np.float32)
    else:
        probabilities = raw_probabilities[valid_indices]
        mean_probabilities = probabilities.mean(axis=0)
        mean_order = np.argsort(mean_probabilities)[::-1]
        payload = prediction_payload(class_names, mean_probabilities, mean_order)
        payload["scene"] = {"valid": True, "status": "Print scene accepted", "reason": "At least one printable frame was accepted."}

    timeline = []
    for frame_index, (timestamp, frame_probabilities) in enumerate(zip(timestamps, raw_probabilities)):
        order = np.argsort(frame_probabilities)[::-1]
        top_index = int(order[0])
        timeline.append(
            {
                "Time (s)": round(timestamp, 2),
                "Prediction": class_names[top_index] if scenes[frame_index]["valid"] else "Scene rejected",
                "Confidence (%)": round(float(frame_probabilities[top_index] * 100), 2) if scenes[frame_index]["valid"] else 0.0,
                "Scene": scenes[frame_index]["status"],
            }
        )

    class_counts = pd.Series(row["Prediction"] for row in timeline).value_counts()
    observed_problems = [name for name in class_counts.index if name not in {"No_defect", "Scene rejected"}]
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
        "valid_frames": len(valid_indices),
    }


def show_video_analysis(video_file) -> dict | None:
    video_bytes = video_file.getvalue()
    st.video(video_bytes)
    try:
        with st.spinner("Sampling video frames and checking for print defects..."):
            result = analyze_video(video_file)
    except Exception as exc:
        st.error(f"Video analysis failed: {exc}")
        return None

    payload = result["payload"]
    duration_text = f"{result['duration']:.1f} s" if result["duration"] else "unknown duration"
    st.caption(f"Analysed {result['sampled_frames']} evenly spaced frames from {duration_text}; {result['valid_frames']} frames passed the scene gate. The uploaded video is removed after analysis.")
    st.subheader("Video result")
    left, right = st.columns([1, 1])
    with left:
        st.metric("Overall decision", payload["predicted_class"])
        st.metric("Average confidence", f"{payload['confidence']:.2f}%")
    with right:
        if result["observed_problems"]:
            st.write("Problems observed in sampled frames")
            for problem in result["observed_problems"]:
                st.write(f"- {problem} ({int(result['class_counts'][problem])} frame(s))")
        else:
            st.write("No accepted frame produced a trained defect decision.")
    if not payload["scene"]["valid"]:
        st.warning(f"Scene rejected: {payload['scene']['reason']}")

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
    return result


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
            payload = scene_payload(pil_image, self.class_names, probabilities, order)
            feedback = ai_feedback_text(payload)[0].replace("AI feedback: ", "")
            with self.lock:
                self.last_prediction = payload["predicted_class"]
                self.last_confidence = payload["confidence"]
                self.last_feedback = feedback

        with self.lock:
            label = f"{self.last_prediction} ({self.last_confidence:.1f}%)"
            feedback = self.last_feedback

        color = (0, 220, 0) if self.last_prediction == "No_defect" else ((0, 165, 255) if self.last_prediction != "Out_of_scope" else (0, 180, 220))
        cv2.rectangle(image, (12, 12), (620, 92), (0, 0, 0), thickness=-1)
        cv2.putText(image, label, (24, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        cv2.putText(image, feedback[:68], (24, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        return av.VideoFrame.from_ndarray(image, format="bgr24")


def main() -> None:
    inject_dashboard_style()
    init_session()
    _, class_names = load_model_and_classes()
    show_sidebar(class_names)
    show_workspace_summary(class_names)

    tab_upload, tab_video, tab_live, tab_camera = st.tabs(
        ["Inspect image", "Review video", "Live camera", "Snapshot"]
    )

    with tab_upload:
        st.markdown("<div class='fdm-section'>Inspect a single print frame</div>", unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "Choose a print image",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            help="Use a clear, well-lit view of the printed part.",
        )
        if uploaded:
            image = safe_image(uploaded)
            payload = show_prediction(image)
            if payload:
                remember_inspection("Image upload", uploaded.name, payload, uploaded.getvalue())

    with tab_video:
        st.markdown("<div class='fdm-section'>Review a printer-monitoring clip</div>", unsafe_allow_html=True)
        st.caption("The app samples frames across the clip and reports the conditions it sees. Short, steady clips work best.")
        uploaded_video = st.file_uploader(
            "Choose a video",
            type=["mp4", "mov", "avi", "webm"],
            key="video-upload",
            help="Short clips work best. Up to 36 evenly spaced frames are analysed.",
        )
        if uploaded_video and st.button("Analyse video", type="primary"):
            result = show_video_analysis(uploaded_video)
            if result:
                remember_inspection(
                    "Video upload",
                    uploaded_video.name,
                    result["payload"],
                    uploaded_video.getvalue(),
                    f"{result['sampled_frames']} sampled frames",
                )

    with tab_live:
        if ENABLE_WEBRTC:
            st.markdown("<div class='fdm-section'>Continuous browser-camera inspection</div>", unsafe_allow_html=True)
            st.caption("On phones, the app requests the rear/environment camera first. Allow camera permission and point it at the print; prediction refreshes every few frames.")
            webrtc_streamer(
                key="fdm-live-detection",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                video_processor_factory=LiveDefectProcessor,
                media_stream_constraints={
                    "video": {"facingMode": {"ideal": "environment"}},
                    "audio": False,
                },
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
        st.markdown("<div class='fdm-section'>Capture one frame for inspection</div>", unsafe_allow_html=True)
        camera_image = st.camera_input("Open the camera")
        if camera_image:
            image = safe_image(camera_image)
            payload = show_prediction(image)
            if payload:
                remember_inspection("Camera snapshot", "browser camera", payload, camera_image.getvalue())

    show_recent_inspections()
    st.divider()
    st.markdown(
        "<div class='fdm-footnote'>The system is decision support, not an automatic printer controller. "
        "Confirm important defects visually before changing temperature, speed, axes, or hardware.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
