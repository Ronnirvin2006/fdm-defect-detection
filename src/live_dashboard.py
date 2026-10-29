import json
import os
import signal
import threading
import time
import csv
from io import BytesIO, StringIO
from collections import deque
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

os.environ.setdefault("KERAS_HOME", str(Path(__file__).resolve().parents[1] / ".keras"))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

from config import IMAGE_SIZE, MODELS_DIR
from defect_knowledge import recommendation_for
from scene_guard import assess_print_scene, blocked_prediction


PROJECT_ROOT = Path(__file__).resolve().parents[1]
HOST = os.getenv("FDM_DASHBOARD_HOST", "127.0.0.1")
PORT = int(os.getenv("FDM_DASHBOARD_PORT", "8765"))
CAMERA_INDEX = int(os.getenv("FDM_CAMERA_INDEX", "0"))
PREDICTION_INTERVAL = float(os.getenv("FDM_PREDICTION_INTERVAL", "0.8"))


def risk_level(predicted_class: str, confidence: float) -> str:
    if confidence < 60:
        return "Uncertain"
    if predicted_class == "No_defect":
        return "Normal"
    if predicted_class in {"Off_platform", "Spaghetti"}:
        return "Critical"
    if predicted_class in {"Under_extrusion", "Warping", "Layer_shifting"}:
        return "High" if confidence >= 70 else "Medium"
    return "Medium" if confidence >= 70 else "Uncertain"


def machine_guidance(predicted_class: str) -> dict:
    guidance = {
        "nozzle_temperature": "Maintain current setting; verify against filament specification.",
        "print_speed": "Maintain current setting while monitoring.",
        "x_axis": "No automatic adjustment recommended.",
        "y_axis": "No automatic adjustment recommended.",
        "z_axis": "No automatic adjustment recommended.",
        "nozzle_cleaning": "Not required by the current visual result.",
    }
    overrides = {
        "Cracking": {
            "nozzle_temperature": "Consider a small increase within the filament limit.",
            "print_speed": "Consider reducing speed to improve layer bonding.",
        },
        "Layer_shifting": {
            "print_speed": "Reduce speed, acceleration, and jerk before the next print.",
            "x_axis": "Inspect X belt tension, pulley screws, and free movement.",
            "y_axis": "Inspect Y belt tension, pulley screws, and free movement.",
            "z_axis": "Inspect for binding; do not apply an offset automatically.",
        },
        "Off_platform": {
            "print_speed": "Reduce first-layer speed.",
            "x_axis": "Verify X homing and that the sliced model is inside the bed area.",
            "y_axis": "Verify Y homing and that the sliced model is inside the bed area.",
            "z_axis": "Recheck bed level and Z-offset manually.",
        },
        "Spaghetti": {
            "print_speed": "Stop the failed print before changing speed.",
            "x_axis": "Inspect after stopping the printer.",
            "y_axis": "Inspect after stopping the printer.",
            "z_axis": "Inspect Z movement and first-layer offset before restarting.",
            "nozzle_cleaning": "Recommended after stopping and cooling safely.",
        },
        "Stringing": {
            "nozzle_temperature": "Consider a small decrease and tune retraction.",
            "print_speed": "Increase travel speed only within printer limits.",
        },
        "Under_extrusion": {
            "nozzle_temperature": "Verify temperature; a small increase may improve flow.",
            "print_speed": "Reduce print speed until extrusion is stable.",
            "nozzle_cleaning": "Recommended: inspect for a partial clog or perform a cold pull.",
        },
        "Warping": {
            "print_speed": "Reduce first-layer speed.",
            "z_axis": "Recheck first-layer Z-offset and bed leveling manually.",
        },
    }
    guidance.update(overrides.get(predicted_class, {}))
    return guidance


class SharedState:
    def __init__(self, predictor):
        self.lock = threading.Lock()
        self.predictor = predictor
        self.frame = None
        self.running = True
        self.video_clients = 0
        self.camera_requested = threading.Event()
        self.nozzle_cleaned = False
        self.nozzle_cleaned_at = None
        self.history = deque(maxlen=120)
        self.video_result = None
        self.model = {
            "architecture": "EfficientNetB0",
            "input_size": "224 x 224",
            "classes": list(predictor.class_names),
            "status": "Loaded",
        }
        self.camera = {
            "connected": False,
            "message": "Camera off - open the dashboard to start",
            "fps": 0.0,
        }
        self.prediction = {
            "class_name": "Waiting",
            "confidence": 0.0,
            "risk": "Unknown",
            "description": "Waiting for the first camera frame.",
            "possible_causes": [],
            "corrective_actions": [],
            "probabilities": [],
            "machine_guidance": machine_guidance("No_defect"),
            "scene": {"valid": False, "status": "Waiting", "reason": "Waiting for a camera frame."},
            "updated_at": None,
        }
        self.telemetry = disconnected_telemetry("OctoPrint is not configured")

    def set_prediction(self, prediction: dict, source: str = "Live camera") -> None:
        with self.lock:
            stored_prediction = dict(prediction)
            stored_prediction["source"] = source
            self.prediction = stored_prediction
            self.history.append(
                {
                    "time": prediction.get("updated_at"),
                    "source": source,
                    "class_name": prediction.get("class_name"),
                    "confidence": round(float(prediction.get("confidence", 0.0)), 2),
                    "risk": prediction.get("risk"),
                }
            )

    def clear_history(self) -> None:
        with self.lock:
            self.history.clear()

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "camera": dict(self.camera),
                "prediction": dict(self.prediction),
                "telemetry": dict(self.telemetry),
                "model": dict(self.model),
                "history": list(self.history)[-24:],
                "video_result": self.video_result,
                "nozzle_cleaning": {
                    "confirmed_by_operator": self.nozzle_cleaned,
                    "confirmed_at": self.nozzle_cleaned_at,
                },
            }

    def add_video_client(self) -> None:
        with self.lock:
            self.video_clients += 1
        self.camera_requested.set()

    def remove_video_client(self) -> None:
        with self.lock:
            self.video_clients = max(0, self.video_clients - 1)

    def has_video_clients(self) -> bool:
        with self.lock:
            return self.video_clients > 0


def disconnected_telemetry(message: str) -> dict:
    return {
        "connected": False,
        "source": "OctoPrint",
        "message": message,
        "printer_state": "Disconnected",
        "nozzle_actual": None,
        "nozzle_target": None,
        "bed_actual": None,
        "bed_target": None,
        "progress": None,
        "time_left_seconds": None,
        "file_name": None,
        "print_speed": None,
        "x_position": None,
        "y_position": None,
        "z_position": None,
        "updated_at": None,
    }


class OctoPrintMonitor(threading.Thread):
    def __init__(self, state: SharedState):
        super().__init__(daemon=True)
        self.state = state
        self.base_url = os.getenv("FDM_OCTOPRINT_URL", "").rstrip("/")
        self.api_key = os.getenv("FDM_OCTOPRINT_API_KEY", "")

    def request_json(self, endpoint: str) -> dict:
        request = Request(
            f"{self.base_url}{endpoint}",
            headers={"X-Api-Key": self.api_key, "Accept": "application/json"},
        )
        with urlopen(request, timeout=4) as response:
            return json.load(response)

    def run(self):
        if not self.base_url or not self.api_key:
            return
        while self.state.running:
            try:
                printer = self.request_json("/api/printer")
                job = self.request_json("/api/job")
                temperatures = printer.get("temperature", {})
                tool = temperatures.get("tool0", {})
                bed = temperatures.get("bed", {})
                progress = job.get("progress", {})
                telemetry = {
                    "connected": True,
                    "source": "OctoPrint",
                    "message": "Live printer values",
                    "printer_state": printer.get("state", {}).get("text", "Unknown"),
                    "nozzle_actual": tool.get("actual"),
                    "nozzle_target": tool.get("target"),
                    "bed_actual": bed.get("actual"),
                    "bed_target": bed.get("target"),
                    "progress": progress.get("completion"),
                    "time_left_seconds": progress.get("printTimeLeft"),
                    "file_name": job.get("job", {}).get("file", {}).get("display"),
                    # Standard OctoPrint status does not expose current XYZ or motion speed.
                    "print_speed": None,
                    "x_position": None,
                    "y_position": None,
                    "z_position": None,
                    "updated_at": datetime.now().isoformat(timespec="seconds"),
                }
            except (HTTPError, URLError, TimeoutError, ValueError) as exc:
                telemetry = disconnected_telemetry(f"OctoPrint connection failed: {exc}")
            with self.state.lock:
                self.state.telemetry = telemetry
            time.sleep(2)


class DefectPredictor:
    def __init__(self):
        self.model = tf.keras.models.load_model(MODELS_DIR / "best_model.keras", compile=False)
        self.class_names = json.loads((MODELS_DIR / "class_names.json").read_text(encoding="utf-8"))
        self.probability_history = deque(maxlen=5)
        self.lock = threading.Lock()

    def predict(self, frame: np.ndarray, use_history: bool = True) -> dict:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb).resize(IMAGE_SIZE)
        batch = np.expand_dims(np.asarray(image, dtype=np.float32), axis=0)
        with self.lock:
            current_probabilities = self.model.predict(batch, verbose=0)[0]
            if use_history:
                self.probability_history.append(current_probabilities)
                probabilities = np.mean(self.probability_history, axis=0)
            else:
                probabilities = current_probabilities
        return self.from_probabilities(probabilities)

    def from_probabilities(self, probabilities: np.ndarray) -> dict:
        probabilities = np.asarray(probabilities, dtype=np.float32)
        order = np.argsort(probabilities)[::-1]
        top_index = int(order[0])
        class_name = self.class_names[top_index]
        confidence = float(probabilities[top_index] * 100)
        knowledge = recommendation_for(class_name)
        description = knowledge["description"]
        if confidence < 60:
            description = (
                "Low-confidence result: visually inspect the print and wait for a stable reading before "
                f"changing printer settings. Closest class: {class_name}. {description}"
            )
        return {
            "class_name": class_name,
            "confidence": confidence,
            "risk": risk_level(class_name, confidence),
            "description": description,
            "possible_causes": knowledge["possible_causes"],
            "corrective_actions": knowledge["corrective_actions"],
            "probabilities": [
                {"class_name": self.class_names[int(index)], "confidence": float(probabilities[index] * 100)}
                for index in order
            ],
            "raw_probabilities": probabilities.tolist(),
            "machine_guidance": machine_guidance(class_name),
            "scene": {"valid": True, "status": "Print scene accepted", "reason": "Printable structure is visible enough to inspect."},
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }


def analyse_video(predictor: DefectPredictor, video_bytes: bytes, suffix: str = ".mp4") -> dict:
    """Sample an uploaded clip and return a guarded multi-frame classification."""
    predictions = []
    scene_checks = []
    predictions_dir = PROJECT_ROOT / "outputs" / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=predictions_dir) as temporary_dir:
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
        selected = set(np.linspace(0, total_frames - 1, min(36, total_frames), dtype=int).tolist())
        index = 0
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if index in selected:
                scene = assess_print_scene(frame)
                scene_checks.append({"time": round(index / fps, 2) if fps else float(index), "scene": scene})
                if scene["valid"]:
                    predictions.append((index, predictor.predict(frame, use_history=False)))
            index += 1
        capture.release()

    if not predictions:
        scene = scene_checks[0]["scene"] if scene_checks else {"valid": False, "reason": "No readable frame."}
        blocked = blocked_prediction(scene)
        return {
            "prediction": blocked,
            "sampled_frames": len(scene_checks),
            "valid_frames": 0,
            "duration": (total_frames / fps) if fps else None,
            "observed_problems": [],
            "timeline": [],
            "scene_checks": scene_checks,
        }

    raw = np.mean([prediction["raw_probabilities"] for _, prediction in predictions], axis=0)
    overall = predictor.from_probabilities(raw)
    timeline = []
    for index, prediction in predictions:
        timeline.append(
            {
                "time": round(index / fps, 2) if fps else float(index),
                "class_name": prediction["class_name"],
                "confidence": round(prediction["confidence"], 2),
            }
        )
    counts = {}
    for row in timeline:
        counts[row["class_name"]] = counts.get(row["class_name"], 0) + 1
    observed = [name for name, count in sorted(counts.items(), key=lambda item: -item[1]) if name != "No_defect"]
    overall["scene"] = {
        "valid": True,
        "status": "Print scene accepted",
        "reason": f"{len(predictions)} of {len(scene_checks)} sampled frames contained printable structure.",
    }
    return {
        "prediction": overall,
        "sampled_frames": len(scene_checks),
        "valid_frames": len(predictions),
        "duration": (total_frames / fps) if fps else None,
        "observed_problems": observed,
        "timeline": timeline,
        "scene_checks": scene_checks,
    }


class CameraMonitor(threading.Thread):
    def __init__(self, state: SharedState):
        super().__init__(daemon=True)
        self.state = state
        self.predictor = state.predictor

    @staticmethod
    def placeholder(message: str) -> bytes:
        image = np.full((720, 1280, 3), (22, 25, 29), dtype=np.uint8)
        cv2.putText(image, "FDM LIVE MONITOR", (55, 320), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (230, 235, 240), 3)
        cv2.putText(image, message[:70], (55, 380), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (90, 180, 255), 2)
        return cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 82])[1].tobytes()

    def run(self):
        while self.state.running:
            if not self.state.has_video_clients():
                with self.state.lock:
                    self.state.camera = {
                        "connected": False,
                        "message": "Camera off - open the dashboard to start",
                        "fps": 0.0,
                    }
                    self.state.frame = self.placeholder("Camera is off")
                self.state.camera_requested.clear()
                self.state.camera_requested.wait(timeout=0.5)
                continue

            capture = cv2.VideoCapture(CAMERA_INDEX)
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            if not capture.isOpened():
                message = f"Camera {CAMERA_INDEX} could not be opened"
                with self.state.lock:
                    self.state.camera = {"connected": False, "message": message, "fps": 0.0}
                    self.state.frame = self.placeholder(message)
                capture.release()
                time.sleep(1)
                continue

            last_prediction = 0.0
            frame_counter = 0
            fps_started = time.monotonic()
            measured_fps = 0.0
            with self.state.lock:
                self.state.camera = {"connected": True, "message": "Live OpenCV camera", "fps": 0.0}

            while self.state.running and self.state.has_video_clients():
                ok, frame = capture.read()
                if not ok:
                    with self.state.lock:
                        self.state.camera["connected"] = False
                        self.state.camera["message"] = "Camera frame read failed"
                    time.sleep(0.2)
                    continue

                now = time.monotonic()
                if now - last_prediction >= PREDICTION_INTERVAL:
                    try:
                        scene = assess_print_scene(frame)
                        prediction = (
                            self.predictor.predict(frame)
                            if scene["valid"]
                            else blocked_prediction(scene)
                        )
                        self.state.set_prediction(prediction, "Live camera")
                    except Exception as exc:
                        with self.state.lock:
                            self.state.prediction["description"] = f"Prediction failed: {exc}"
                    last_prediction = now

                frame_counter += 1
                elapsed = now - fps_started
                if elapsed >= 1:
                    measured_fps = frame_counter / elapsed
                    frame_counter = 0
                    fps_started = now

                with self.state.lock:
                    prediction = dict(self.state.prediction)
                    self.state.camera["fps"] = round(measured_fps, 1)

                label = f"{prediction['class_name']}  {prediction['confidence']:.1f}%"
                color = (62, 207, 142) if prediction["class_name"] == "No_defect" else (78, 153, 255)
                cv2.rectangle(frame, (20, 20), (620, 118), (18, 21, 25), -1)
                cv2.putText(frame, label, (42, 64), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                cv2.putText(frame, f"Risk: {prediction['risk']}", (42, 99), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (235, 238, 242), 2)
                encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 82])[1].tobytes()
                with self.state.lock:
                    self.state.frame = encoded

            capture.release()
            with self.state.lock:
                self.state.camera = {
                    "connected": False,
                    "message": "Camera off - live page closed",
                    "fps": 0.0,
                }
                self.state.frame = self.placeholder("Camera is off")


INDEX_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FDM Live Defect Monitor</title>
  <style>
    :root { color-scheme: dark; --bg:#0f1215; --panel:#171b20; --line:#30363d; --text:#edf1f5; --muted:#9ba5af; --cyan:#31b7c7; --green:#3ecf8e; --amber:#ffb454; --red:#ff6868; }
    * { box-sizing:border-box; }
    body { margin:0; background:var(--bg); color:var(--text); font:14px/1.45 system-ui,sans-serif; }
    header { height:64px; padding:0 24px; border-bottom:1px solid var(--line); display:flex; align-items:center; justify-content:space-between; }
    h1 { font-size:20px; margin:0; letter-spacing:0; }
    .status { display:flex; gap:8px; align-items:center; color:var(--muted); }
    .dot { width:9px; height:9px; border-radius:50%; background:var(--red); }
    .dot.on { background:var(--green); }
    main { display:grid; grid-template-columns:minmax(0, 1fr) 360px; min-height:calc(100vh - 64px); }
    .vision { padding:20px; min-width:0; }
    .video-wrap { position:relative; background:#07090b; border:1px solid var(--line); aspect-ratio:16/9; overflow:hidden; }
    .video-wrap img { width:100%; height:100%; object-fit:contain; display:block; }
    .tools { display:flex; gap:12px; align-items:center; flex-wrap:wrap; padding:14px 0 0; color:var(--muted); }
    input[type=file] { max-width:320px; }
    .upload-block { display:grid; gap:7px; padding:12px 0 0; }
    .upload-block + .upload-block { border-top:1px solid var(--line); margin-top:12px; }
    .video-result { margin-top:12px; padding:12px; border:1px solid var(--line); background:var(--panel); }
    .video-result strong { color:var(--cyan); }
    .prediction { display:grid; grid-template-columns:minmax(170px,1fr) 130px 110px minmax(130px,1fr); gap:16px; padding:18px 0; border-bottom:1px solid var(--line); }
    .label { color:var(--muted); font-size:12px; text-transform:uppercase; }
    .value { font-size:24px; font-weight:650; margin-top:3px; }
    .feedback { padding-top:18px; display:grid; grid-template-columns:1fr 1fr; gap:24px; }
    .feedback h2, aside h2 { font-size:14px; margin:0 0 10px; }
    .feedback ul { margin:0; padding-left:18px; color:#cbd2d9; }
    aside { border-left:1px solid var(--line); background:var(--panel); padding:20px; overflow:auto; }
    .section { padding:0 0 20px; margin:0 0 20px; border-bottom:1px solid var(--line); }
    .grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
    .metric { border:1px solid var(--line); padding:10px; min-height:70px; }
    .metric strong { display:block; font-size:18px; margin-top:5px; }
    .unavailable { color:var(--muted); }
    .guide { display:grid; gap:9px; }
    .guide div { border-left:3px solid var(--cyan); padding-left:10px; }
    .history { margin-top:22px; border-top:1px solid var(--line); padding-top:18px; }
    .history-head { display:flex; align-items:center; justify-content:space-between; gap:12px; margin-bottom:10px; }
    .history-head h2 { margin:0; font-size:14px; }
    .history-actions { display:flex; gap:8px; flex-wrap:wrap; }
    .history-table { width:100%; border-collapse:collapse; font-size:12px; }
    .history-table th, .history-table td { text-align:left; padding:9px 8px; border-bottom:1px solid var(--line); }
    .history-table th { color:var(--muted); font-weight:500; }
    .history-table td:nth-child(4) { color:var(--cyan); font-weight:650; }
    .model-classes { color:#cbd2d9; line-height:1.8; }
    button { border:1px solid #4b5864; background:#232a31; color:var(--text); padding:9px 12px; cursor:pointer; }
    button:hover { background:#2c353e; }
    .note { color:var(--muted); font-size:12px; margin-top:9px; }
    @media (max-width:900px) { main { grid-template-columns:1fr; } aside { border-left:0; border-top:1px solid var(--line); } .prediction { grid-template-columns:1fr 1fr; } .feedback { grid-template-columns:1fr; } }
    @media (max-width:560px) { .prediction { grid-template-columns:1fr; } .history-head { align-items:flex-start; flex-direction:column; } .history-table { min-width:560px; } .history { overflow:auto; } }
  </style>
</head>
<body>
  <header><div><h1>FDM Quality Desk</h1><span class="note">AI-assisted fused deposition modelling inspection</span></div><div class="status"><span id="camera-dot" class="dot"></span><span id="camera-status">Starting camera</span></div></header>
  <main>
    <section class="vision">
      <div class="video-wrap"><img src="/video" alt="Live OpenCV camera feed"></div>
      <div class="tools">
        <label for="image-upload">Upload photo:</label>
        <input id="image-upload" type="file" accept="image/png,image/jpeg,image/webp,image/bmp">
        <span id="upload-status">Live camera is active. Uploaded photos use the same model.</span>
      </div>
      <div class="upload-block">
        <label for="video-upload">Upload video:</label>
        <input id="video-upload" type="file" accept="video/mp4,video/quicktime,video/x-msvideo,video/webm">
        <span id="video-status" class="note">The clip is sampled frame by frame and temporary data is deleted after analysis.</span>
        <div id="video-result" class="video-result" hidden><strong id="video-summary"></strong><div id="video-problems"></div></div>
      </div>
      <div class="prediction">
        <div><div class="label">Detected condition</div><div id="class-name" class="value">Waiting</div></div>
        <div><div class="label">Confidence</div><div id="confidence" class="value">0.0%</div></div>
        <div><div class="label">Risk</div><div id="risk" class="value">Unknown</div></div>
        <div><div class="label">Source</div><div id="source" class="value">--</div></div>
      </div>
      <div class="feedback">
        <div><h2>AI feedback</h2><p id="description">Waiting for the first frame.</p><h2>Possible causes</h2><ul id="causes"></ul></div>
        <div><h2>Corrective actions</h2><ul id="actions"></ul></div>
      </div>
      <div class="history">
        <div class="history-head"><h2>Recent inspections</h2><div class="history-actions"><button id="download-history" type="button">Download CSV</button><button id="clear-history" type="button">Clear</button></div></div>
        <div id="history-empty" class="note">No predictions recorded yet.</div>
        <table id="history-table" class="history-table" hidden><thead><tr><th>Time</th><th>Source</th><th>Condition</th><th>Confidence</th><th>Risk</th></tr></thead><tbody></tbody></table>
      </div>
    </section>
    <aside>
      <div class="section">
        <h2>Printer connection</h2>
        <div class="status"><span id="printer-dot" class="dot"></span><span id="printer-state">Disconnected</span></div>
        <div id="printer-message" class="note">OctoPrint is not configured</div>
      </div>
      <div class="section">
        <h2>AI model</h2>
        <div class="grid">
          <div class="metric"><span class="label">Status</span><strong id="model-status">Loading</strong></div>
          <div class="metric"><span class="label">Input</span><strong id="model-input">224 × 224</strong></div>
          <div class="metric"><span class="label">Scene gate</span><strong id="scene-status">Waiting</strong></div>
        </div>
        <p id="model-classes" class="model-classes"></p>
      </div>
      <div class="section">
        <h2>Live telemetry</h2>
        <div class="grid">
          <div class="metric"><span class="label">Nozzle</span><strong id="nozzle">Unavailable</strong></div>
          <div class="metric"><span class="label">Bed</span><strong id="bed">Unavailable</strong></div>
          <div class="metric"><span class="label">Progress</span><strong id="progress">Unavailable</strong></div>
          <div class="metric"><span class="label">Speed</span><strong id="speed">Unavailable</strong></div>
          <div class="metric"><span class="label">X position</span><strong id="x-position">Unavailable</strong></div>
          <div class="metric"><span class="label">Y position</span><strong id="y-position">Unavailable</strong></div>
          <div class="metric"><span class="label">Z position</span><strong id="z-position">Unavailable</strong></div>
          <div class="metric"><span class="label">Time left</span><strong id="time-left">Unavailable</strong></div>
        </div>
        <p class="note">Temperature and job progress can come from OctoPrint. Standard OctoPrint status does not report live XYZ position or actual motion speed.</p>
      </div>
      <div class="section">
        <h2>Suggested checks</h2>
        <div class="guide">
          <div><span class="label">Temperature</span><br><span id="guide-temp"></span></div>
          <div><span class="label">Speed</span><br><span id="guide-speed"></span></div>
          <div><span class="label">X axis</span><br><span id="guide-x"></span></div>
          <div><span class="label">Y axis</span><br><span id="guide-y"></span></div>
          <div><span class="label">Z axis</span><br><span id="guide-z"></span></div>
        </div>
      </div>
      <div class="section">
        <h2>Nozzle cleaning</h2>
        <p id="cleaning-recommendation"></p>
        <p id="cleaning-status" class="unavailable">Not verified</p>
        <button id="clean-button" type="button">Mark cleaned by operator</button>
        <p class="note">This is an operator checklist, not a camera or printer sensor reading.</p>
      </div>
    </aside>
  </main>
  <script>
    const text = (id, value) => document.getElementById(id).textContent = value;
    const list = (id, values) => document.getElementById(id).innerHTML = values.map(v => `<li>${v}</li>`).join('');
    const escapeHtml = value => String(value ?? '').replace(/[&<>"']/g, char => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#039;'}[char]));
    const metric = value => value === null || value === undefined ? 'Unavailable' : value;
    const temperature = (actual, target) => actual === null || actual === undefined ? 'Unavailable' : `${actual.toFixed(1)} / ${target?.toFixed(0) ?? '-'} C`;
    function renderHistory(rows) {
      const table = document.getElementById('history-table');
      const empty = document.getElementById('history-empty');
      if (!rows || !rows.length) { table.hidden = true; empty.hidden = false; return; }
      empty.hidden = true; table.hidden = false;
      table.querySelector('tbody').innerHTML = rows.slice().reverse().map(row => `<tr><td>${escapeHtml(row.time)}</td><td>${escapeHtml(row.source)}</td><td>${escapeHtml(row.class_name)}</td><td>${Number(row.confidence).toFixed(2)}%</td><td>${escapeHtml(row.risk)}</td></tr>`).join('');
    }
    async function refresh() {
      try {
        const response = await fetch('/api/status', {cache:'no-store'});
        const data = await response.json();
        document.getElementById('camera-dot').classList.toggle('on', data.camera.connected);
        text('camera-status', `${data.camera.message} | ${data.camera.fps} FPS`);
        text('class-name', data.prediction.class_name);
        text('confidence', `${data.prediction.confidence.toFixed(1)}%`);
        text('risk', data.prediction.risk);
        text('source', data.prediction.source || '--');
        text('scene-status', data.prediction.scene?.status || 'Waiting');
        text('description', data.prediction.description);
        list('causes', data.prediction.possible_causes);
        list('actions', data.prediction.corrective_actions);
        text('model-status', data.model.status);
        text('model-input', data.model.input_size);
        text('model-classes', data.model.classes.join(' · '));
        renderHistory(data.history);
        document.getElementById('printer-dot').classList.toggle('on', data.telemetry.connected);
        text('printer-state', data.telemetry.printer_state);
        text('printer-message', data.telemetry.message);
        text('nozzle', temperature(data.telemetry.nozzle_actual, data.telemetry.nozzle_target));
        text('bed', temperature(data.telemetry.bed_actual, data.telemetry.bed_target));
        text('progress', data.telemetry.progress == null ? 'Unavailable' : `${data.telemetry.progress.toFixed(1)}%`);
        text('speed', metric(data.telemetry.print_speed));
        text('x-position', metric(data.telemetry.x_position));
        text('y-position', metric(data.telemetry.y_position));
        text('z-position', metric(data.telemetry.z_position));
        text('time-left', data.telemetry.time_left_seconds == null ? 'Unavailable' : `${Math.round(data.telemetry.time_left_seconds / 60)} min`);
        const guide = data.prediction.machine_guidance;
        text('guide-temp', guide.nozzle_temperature);
        text('guide-speed', guide.print_speed);
        text('guide-x', guide.x_axis);
        text('guide-y', guide.y_axis);
        text('guide-z', guide.z_axis);
        text('cleaning-recommendation', guide.nozzle_cleaning);
        text('cleaning-status', data.nozzle_cleaning.confirmed_by_operator ? `Yes, confirmed at ${data.nozzle_cleaning.confirmed_at}` : 'No, not verified');
        text('clean-button', data.nozzle_cleaning.confirmed_by_operator ? 'Clear cleaning confirmation' : 'Mark cleaned by operator');
      } catch (error) { text('camera-status', `Dashboard connection error: ${error}`); }
    }
    document.getElementById('clean-button').addEventListener('click', async () => {
      await fetch('/api/nozzle-cleaning', {method:'POST'});
      refresh();
    });
    document.getElementById('download-history').addEventListener('click', () => { window.location.href = '/api/history.csv'; });
    document.getElementById('clear-history').addEventListener('click', async () => {
      await fetch('/api/clear-history', {method:'POST'});
      refresh();
    });
    document.getElementById('image-upload').addEventListener('change', async (event) => {
      const file = event.target.files[0];
      if (!file) return;
      text('upload-status', `Checking ${file.name}...`);
      try {
        const response = await fetch('/api/upload-image', {
          method: 'POST',
          headers: {'Content-Type': file.type || 'application/octet-stream'},
          body: file
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || 'Upload failed');
        text('upload-status', `Prediction updated from ${file.name}`);
        refresh();
      } catch (error) {
        text('upload-status', `Upload prediction failed: ${error.message}`);
      }
    });
    document.getElementById('video-upload').addEventListener('change', async (event) => {
      const file = event.target.files[0];
      if (!file) return;
      text('video-status', `Analysing ${file.name}...`);
      try {
        const response = await fetch('/api/upload-video', {
          method: 'POST',
          headers: {'Content-Type': file.type || 'video/mp4'},
          body: file
        });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || 'Video analysis failed');
        const result = payload.result;
        const prediction = result.prediction;
        const videoResult = document.getElementById('video-result');
        videoResult.hidden = false;
        text('video-summary', `${prediction.class_name} · ${Number(prediction.confidence).toFixed(1)}% · ${result.valid_frames}/${result.sampled_frames} valid frames`);
        document.getElementById('video-problems').textContent = result.observed_problems.length ? `Observed: ${result.observed_problems.join(', ')}` : prediction.scene.reason;
        text('video-status', `Analysis complete for ${file.name}.`);
        refresh();
      } catch (error) {
        text('video-status', `Video analysis failed: ${error.message}`);
      }
    });
    refresh(); setInterval(refresh, 1000);
  </script>
</body>
</html>"""


class DashboardHandler(BaseHTTPRequestHandler):
    state: SharedState = None

    def send_bytes(self, content: bytes, content_type: str, status=HTTPStatus.OK):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def do_GET(self):
        if self.path == "/":
            self.send_bytes(INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if self.path == "/api/status":
            payload = json.dumps(self.state.snapshot()).encode("utf-8")
            self.send_bytes(payload, "application/json")
            return
        if self.path == "/api/history.csv":
            rows = self.state.snapshot()["history"]
            if rows:
                columns = ["time", "source", "class_name", "confidence", "risk"]
                text_buffer = StringIO()
                writer = csv.DictWriter(text_buffer, fieldnames=columns)
                writer.writeheader()
                writer.writerows(rows)
                text = text_buffer.getvalue()
            else:
                text = "time,source,class_name,confidence,risk\n"
            self.send_bytes(text.encode("utf-8"), "text/csv; charset=utf-8")
            return
        if self.path == "/video":
            self.state.add_video_client()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            try:
                while self.state.running:
                    with self.state.lock:
                        frame = self.state.frame
                    if frame:
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
                    time.sleep(1 / 15)
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                self.state.remove_video_client()
            return
        self.send_bytes(b"Not found", "text/plain", HTTPStatus.NOT_FOUND)

    def do_POST(self):
        if self.path == "/api/clear-history":
            self.state.clear_history()
            self.send_bytes(b'{"ok":true}', "application/json")
            return
        if self.path == "/api/nozzle-cleaning":
            with self.state.lock:
                self.state.nozzle_cleaned = not self.state.nozzle_cleaned
                self.state.nozzle_cleaned_at = (
                    datetime.now().isoformat(timespec="seconds") if self.state.nozzle_cleaned else None
                )
            self.send_bytes(b'{"ok":true}', "application/json")
            return
        if self.path == "/api/upload-image":
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                if content_length <= 0:
                    raise ValueError("No image data received")
                image_bytes = self.rfile.read(content_length)
                image = Image.open(BytesIO(image_bytes)).convert("RGB")
                rgb = np.asarray(image, dtype=np.uint8)
                frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                scene = assess_print_scene(frame)
                prediction = (
                    self.state.predictor.predict(frame, use_history=False)
                    if scene["valid"]
                    else blocked_prediction(scene)
                )
                self.state.set_prediction(prediction, "Uploaded image")
                self.send_bytes(json.dumps({"ok": True, "prediction": prediction}).encode("utf-8"), "application/json")
            except Exception as exc:
                payload = json.dumps({"ok": False, "error": str(exc)}).encode("utf-8")
                self.send_bytes(payload, "application/json", HTTPStatus.BAD_REQUEST)
            return
        if self.path == "/api/upload-video":
            try:
                content_length = int(self.headers.get("Content-Length", "0"))
                if content_length <= 0 or content_length > 200 * 1024 * 1024:
                    raise ValueError("Video must be between 1 byte and 200 MB.")
                video_bytes = self.rfile.read(content_length)
                result = analyse_video(self.state.predictor, video_bytes)
                self.state.set_prediction(result["prediction"], "Uploaded video")
                self.state.video_result = result
                self.send_bytes(json.dumps({"ok": True, "result": result}).encode("utf-8"), "application/json")
            except Exception as exc:
                payload = json.dumps({"ok": False, "error": str(exc)}).encode("utf-8")
                self.send_bytes(payload, "application/json", HTTPStatus.BAD_REQUEST)
            return
        self.send_bytes(b"Not found", "text/plain", HTTPStatus.NOT_FOUND)

    def log_message(self, message_format, *args):
        if self.path != "/api/status":
            super().log_message(message_format, *args)


def main():
    predictor = DefectPredictor()
    state = SharedState(predictor)
    DashboardHandler.state = state
    camera = CameraMonitor(state)
    telemetry = OctoPrintMonitor(state)
    camera.start()
    telemetry.start()
    server = ThreadingHTTPServer((HOST, PORT), DashboardHandler)

    def stop_server(_signum, _frame):
        state.running = False
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, stop_server)
    signal.signal(signal.SIGTERM, stop_server)
    print(f"FDM live dashboard: http://{HOST}:{PORT}")
    if not os.getenv("FDM_OCTOPRINT_URL"):
        print("Printer telemetry: disconnected (set FDM_OCTOPRINT_URL and FDM_OCTOPRINT_API_KEY)")
    try:
        server.serve_forever()
    finally:
        state.running = False
        server.server_close()


if __name__ == "__main__":
    main()
