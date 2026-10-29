import base64
import io
import sys
import unittest
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from browser_camera import decode_camera_frame


class CameraFrameTests(unittest.TestCase):
    def frame(self, size=(640, 480)):
        output = io.BytesIO()
        Image.new("RGB", size).save(output, format="JPEG")
        return "data:image/jpeg;base64," + base64.b64encode(output.getvalue()).decode()

    def test_valid_jpeg(self):
        image = decode_camera_frame(self.frame())
        self.assertEqual(image.size, (640, 480))
        self.assertEqual(image.mode, "RGB")

    def test_invalid_values(self):
        for value in [None, {}, "", "data:image/png;base64,abc", "data:image/jpeg;base64,!!!"]:
            with self.subTest(value=value), self.assertRaises(ValueError):
                decode_camera_frame(value)

    def test_oversized_payload(self):
        with self.assertRaises(ValueError):
            decode_camera_frame("data:image/jpeg;base64," + "A" * 1_000_001)

    def test_oversized_dimensions(self):
        with self.assertRaises(ValueError):
            decode_camera_frame(self.frame((1281, 10)))


if __name__ == "__main__":
    unittest.main()
