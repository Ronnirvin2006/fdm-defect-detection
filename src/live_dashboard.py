import json
import os
import signal
import threading
import time
from collections import deque
from datetime import datetime
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
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
    def __init__(self):
        self.lock = threading.Lock()
        self.frame = None
        self.running = True
        self.video_clients = 0
        self.camera_requested = threading.Event()
        self.nozzle_cleaned = False
        self.nozzle_cleaned_at = None
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
            "updated_at": None,
        }
        self.telemetry = disconnected_telemetry("OctoPrint is not configured")

    def snapshot(self) -> dict:
        with self.lock:
            return {
                "camera": dict(self.camera),
                "prediction": dict(self.prediction),
                "telemetry": dict(self.telemetry),
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


class CameraMonitor(threading.Thread):
    def __init__(self, state: SharedState):
        super().__init__(daemon=True)
        self.state = state
        self.model = tf.keras.models.load_model(MODELS_DIR / "best_model.keras", compile=False)
        self.class_names = json.loads((MODELS_DIR / "class_names.json").read_text(encoding="utf-8"))
        self.probability_history = deque(maxlen=5)

    def predict(self, frame: np.ndarray) -> dict:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb).resize(IMAGE_SIZE)
        batch = np.expand_dims(np.asarray(image, dtype=np.float32), axis=0)
        current_probabilities = self.model.predict(batch, verbose=0)[0]
        self.probability_history.append(current_probabilities)
        probabilities = np.mean(self.probability_history, axis=0)
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
            "machine_guidance": machine_guidance(class_name),
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }

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
                        prediction = self.predict(frame)
                        with self.state.lock:
                            self.state.prediction = prediction
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
    .prediction { display:grid; grid-template-columns:180px 130px 1fr; gap:16px; padding:18px 0; border-bottom:1px solid var(--line); }
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
    button { border:1px solid #4b5864; background:#232a31; color:var(--text); padding:9px 12px; cursor:pointer; }
    button:hover { background:#2c353e; }
    .note { color:var(--muted); font-size:12px; margin-top:9px; }
    @media (max-width:900px) { main { grid-template-columns:1fr; } aside { border-left:0; border-top:1px solid var(--line); } .prediction { grid-template-columns:1fr 1fr; } .feedback { grid-template-columns:1fr; } }
  </style>
</head>
<body>
  <header><h1>FDM Live Defect Monitor</h1><div class="status"><span id="camera-dot" class="dot"></span><span id="camera-status">Starting camera</span></div></header>
  <main>
    <section class="vision">
      <div class="video-wrap"><img src="/video" alt="Live OpenCV camera feed"></div>
      <div class="prediction">
        <div><div class="label">Detected condition</div><div id="class-name" class="value">Waiting</div></div>
        <div><div class="label">Confidence</div><div id="confidence" class="value">0.0%</div></div>
        <div><div class="label">Risk</div><div id="risk" class="value">Unknown</div></div>
      </div>
      <div class="feedback">
        <div><h2>AI feedback</h2><p id="description">Waiting for the first frame.</p><h2>Possible causes</h2><ul id="causes"></ul></div>
        <div><h2>Corrective actions</h2><ul id="actions"></ul></div>
      </div>
    </section>
    <aside>
      <div class="section">
        <h2>Printer connection</h2>
        <div class="status"><span id="printer-dot" class="dot"></span><span id="printer-state">Disconnected</span></div>
        <div id="printer-message" class="note">OctoPrint is not configured</div>
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
    const metric = value => value === null || value === undefined ? 'Unavailable' : value;
    const temperature = (actual, target) => actual === null || actual === undefined ? 'Unavailable' : `${actual.toFixed(1)} / ${target?.toFixed(0) ?? '-'} C`;
    async function refresh() {
      try {
        const response = await fetch('/api/status', {cache:'no-store'});
        const data = await response.json();
        document.getElementById('camera-dot').classList.toggle('on', data.camera.connected);
        text('camera-status', `${data.camera.message} | ${data.camera.fps} FPS`);
        text('class-name', data.prediction.class_name);
        text('confidence', `${data.prediction.confidence.toFixed(1)}%`);
        text('risk', data.prediction.risk);
        text('description', data.prediction.description);
        list('causes', data.prediction.possible_causes);
        list('actions', data.prediction.corrective_actions);
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
        if self.path == "/api/nozzle-cleaning":
            with self.state.lock:
                self.state.nozzle_cleaned = not self.state.nozzle_cleaned
                self.state.nozzle_cleaned_at = (
                    datetime.now().isoformat(timespec="seconds") if self.state.nozzle_cleaned else None
                )
            self.send_bytes(b'{"ok":true}', "application/json")
            return
        self.send_bytes(b"Not found", "text/plain", HTTPStatus.NOT_FOUND)

    def log_message(self, message_format, *args):
        if self.path != "/api/status":
            super().log_message(message_format, *args)


def main():
    state = SharedState()
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
