"""Direct browser-camera inference for remote printer checks, without UI reruns."""
import argparse
import io
import json
import os
import signal
import threading
import time
import hashlib
import logging
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import cv2
import numpy as np
from PIL import Image
import tensorflow as tf

from config import IMAGE_SIZE, MODELS_DIR
from defect_knowledge import recommendation_for
from scene_guard import assess_print_scene

ASSETS = Path(__file__).parent / "camera_component"
WEB = Path(__file__).parent / "web"


class LivePredictor:
    def __init__(self):
        for gpu in tf.config.list_physical_devices("GPU"):
            tf.config.experimental.set_memory_growth(gpu, True)
        self.model = tf.keras.models.load_model(MODELS_DIR / "best_model.keras", compile=False)
        self.names = json.loads((MODELS_DIR / "class_names.json").read_text())
        self.model_id = hashlib.sha256((MODELS_DIR / "best_model.keras").read_bytes()).hexdigest()[:12]
        self.lock = threading.Lock()
        self.infer = tf.function(
            lambda batch: self.model(batch, training=False),
            input_signature=[tf.TensorSpec([1, *IMAGE_SIZE, 3], tf.float32)],
        )
        self.infer(tf.zeros([1, *IMAGE_SIZE, 3], tf.float32))

    def analyse(self, raw, explain=False):
        started = time.perf_counter()
        with Image.open(io.BytesIO(raw)) as opened:
            if opened.format not in {"JPEG", "PNG", "WEBP"} or max(opened.size) > 4096:
                raise ValueError("Expected a supported image at most 4096 pixels wide/high")
            image = opened.convert("RGB")
        scene = assess_print_scene(cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR))
        if not scene["valid"]:
            return {"label":"Scene rejected", "status":"rejected", "feedback":scene["reason"], "scene":scene, "class_name":None, "confidence":0, "processing_ms":round((time.perf_counter()-started)*1000, 1)}
        batch = np.asarray(image.resize(IMAGE_SIZE), dtype=np.float32)[None]
        # Do not let requests from multiple browsers queue up old camera frames.
        if not self.lock.acquire(blocking=False):
            return {"label":"Analysis busy", "feedback":"Waiting for the next live update", "busy":True}
        try:
            probs = self.infer(tf.convert_to_tensor(batch)).numpy()[0]
            overlay = None
            if explain:
                from gradcam import make_gradcam_heatmap
                with tf.device("/CPU:0"):
                    heatmap = make_gradcam_heatmap(self.model, tf.convert_to_tensor(batch), int(np.argmax(probs)))
                base = np.asarray(image.resize((448, 448)), dtype=np.float32)
                heat = cv2.resize(np.asarray(heatmap, dtype=np.float32), (448, 448))
                colour = cv2.cvtColor(cv2.applyColorMap(np.uint8(255 * heat), cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
                rendered = Image.fromarray(np.uint8(base * .65 + colour * .35))
                output = io.BytesIO()
                rendered.save(output, format="JPEG", quality=85)
                import base64
                overlay = "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode()
        finally:
            self.lock.release()
        index = int(np.argmax(probs))
        name, confidence = self.names[index], float(probs[index] * 100)
        knowledge = recommendation_for(name)
        description = knowledge["description"]
        feedback = knowledge["corrective_actions"][0]
        if confidence < 70:
            feedback = "Uncertain result. Visually inspect the print before taking action."
        return {
            "label":f"{name.replace('_', ' ')} ({confidence:.1f}%)",
            "feedback":feedback, "description":description,
            "class_name":name, "confidence":confidence,
            "status":"uncertain" if confidence < 70 else ("normal" if name == "No_defect" else "defect"),
            "scene":scene, "causes":knowledge["possible_causes"], "actions":knowledge["corrective_actions"],
            "probabilities":[{"name":self.names[i], "value":round(float(value)*100, 3)} for i,value in enumerate(probs)],
            "explanation":overlay,
            "processing_ms":round((time.perf_counter()-started)*1000, 1),
        }


class CameraHandler(BaseHTTPRequestHandler):
    predictor = None
    protocol_version = "HTTP/1.1"

    def respond(self, body, content_type="application/json", status=200):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Permissions-Policy", "camera=(self), microphone=()")
        self.send_header("Referrer-Policy", "same-origin")
        self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' blob: data:; media-src 'self' blob:; connect-src 'self'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self.respond(b'{"ready":true}')
        elif self.path == "/favicon.ico":
            self.respond(b"", "image/x-icon", status=204)
        elif self.path == "/":
            self.respond((WEB / "index.html").read_bytes(), "text/html; charset=utf-8")
        elif self.path in {"/dashboard.css", "/dashboard.js", "/vendor/lucide.min.js"}:
            mime = "text/css" if self.path.endswith(".css") else "text/javascript"
            self.respond((WEB / self.path.lstrip("/")).read_bytes(), mime)
        elif self.path == "/api/info":
            self.respond(json.dumps({"classes":self.predictor.names, "model_id":self.predictor.model_id, "ready":True}).encode())
        elif self.path == "/camera":
            html = (ASSETS / "index.html").read_text()
            html = html.replace('<meta charset="utf-8">', '<meta charset="utf-8"><meta name="fdm-transport" content="http"><title>FDM Live Inspection</title>')
            html = html.replace('body { margin:0;', 'body { max-width:1000px; margin:20px auto; padding:0 16px;')
            html = html.replace('<div class="toolbar">', '<h1 style="font-size:24px">FDM Live Inspection</h1><p>Live print monitoring</p><div class="toolbar">', 1)
            self.respond(html.encode(), "text/html; charset=utf-8")
        elif self.path == "/camera.js":
            self.respond((ASSETS / "camera.js").read_bytes(), "text/javascript")
        else:
            self.respond(b'{"error":"Not found"}', status=404)

    def do_POST(self):
        self.connection.settimeout(15)
        if self.path not in {"/api/frame", "/api/image", "/api/explain"}:
            self.close_connection = True
            self.respond(b'{"error":"Not found"}', status=404)
            return
        try:
            if self.headers.get("Sec-Fetch-Site") == "cross-site":
                raise ValueError("Cross-site requests are not allowed")
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= (600_000 if self.path == "/api/frame" else 4_000_000):
                raise ValueError("Frame exceeds request size limit")
            raw = self.rfile.read(length)
            if len(raw) != length:
                raise ValueError("Incomplete camera frame")
            result = self.predictor.analyse(raw, explain=self.path == "/api/explain")
            self.respond(json.dumps(result).encode(), status=503 if result.get("busy") else 200)
        except (BrokenPipeError, ConnectionResetError):
            self.close_connection = True
        except (ValueError, OSError) as exc:
            self.close_connection = True
            self.respond(json.dumps({"error":True, "label":"Frame rejected", "feedback":str(exc)}).encode(), status=400)
        except Exception:
            logging.exception("Camera frame analysis failed")
            self.close_connection = True
            self.respond(b'{"error":true,"label":"Analysis unavailable","feedback":"Try again in a moment."}', status=500)

    def log_message(self, message_format, *args):
        if len(args) < 2 or str(args[1]) not in {"200", "204"}:
            super().log_message(message_format, *args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8502)
    args = parser.parse_args()
    CameraHandler.predictor = LivePredictor()
    server = ThreadingHTTPServer(("127.0.0.1", args.port), CameraHandler)
    server.daemon_threads = True
    def stop(*_):
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    print(f"FDM remote camera ready: http://127.0.0.1:{args.port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
