import io
import json
from pathlib import Path
import sys
import threading
import unittest
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from http.server import ThreadingHTTPServer

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from remote_camera_server import CameraHandler, LivePredictor
from scene_guard import assess_print_scene


class FakePredictor:
    names = ["No_defect", "Stringing"]
    model_id = "test-model"

    def analyse(self, raw, explain=False):
        if raw != b"frame":
            raise ValueError("Invalid frame")
        return {"status":"normal", "class_name":"No_defect", "confidence":90, "explanation":explain}


class RemoteCameraTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        class Handler(CameraHandler):
            predictor = FakePredictor()
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown(); cls.server.server_close(); cls.thread.join(timeout=5)

    def test_dashboard_and_assets(self):
        for path in ["/", "/dashboard.js", "/dashboard.css", "/vendor/lucide.min.js", "/health", "/api/info"]:
            with self.subTest(path=path), urlopen(self.url+path) as response:
                self.assertEqual(response.status, 200)
                self.assertTrue(response.read())

    def test_predictions_and_explanation(self):
        for path, expected in [("/api/frame",False),("/api/image",False),("/api/explain",True)]:
            with urlopen(Request(self.url+path,data=b"frame")) as response:
                self.assertEqual(json.load(response)["explanation"],expected)

    def test_invalid_input(self):
        for payload in [b"", b"invalid", b"x"*600001]:
            with self.assertRaises(HTTPError) as error:
                urlopen(Request(self.url+"/api/frame",data=payload))
            self.assertEqual(error.exception.code,400)

    def test_cross_site_denied(self):
        request=Request(self.url+"/api/frame",data=b"frame",headers={"Sec-Fetch-Site":"cross-site"})
        with self.assertRaises(HTTPError) as error:
            urlopen(request)
        self.assertEqual(error.exception.code,400)

    def test_wall_does_not_reach_classifier(self):
        predictor=LivePredictor.__new__(LivePredictor)
        image=io.BytesIO();Image.new("RGB",(480,360),(130,130,130)).save(image,format="JPEG")
        result=predictor.analyse(image.getvalue())
        self.assertEqual(result["status"],"rejected")
        self.assertIsNone(result["class_name"])

    def test_dark_view_rejected(self):
        self.assertFalse(assess_print_scene(np.zeros((360,480,3),dtype=np.uint8))["valid"])


if __name__=="__main__":unittest.main()
