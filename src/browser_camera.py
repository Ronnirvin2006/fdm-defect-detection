"""Browser camera frames over Streamlit's connection, without a WebRTC peer."""

import base64
import io
from pathlib import Path

from PIL import Image
import streamlit.components.v1 as components


camera_component = components.declare_component(
    "fdm_browser_camera", path=str(Path(__file__).parent / "camera_component")
)


def decode_camera_frame(value):
    if not isinstance(value, str) or not value.startswith("data:image/jpeg;base64,"):
        raise ValueError("Camera frame must be a JPEG data URL")
    if len(value) > 1_000_000:
        raise ValueError("Camera frame exceeds upload limit")
    raw = base64.b64decode(value.split(",", 1)[1], validate=True)
    with Image.open(io.BytesIO(raw)) as image:
        if image.width > 1280 or image.height > 1280:
            raise ValueError("Camera frame dimensions exceed limit")
        return image.convert("RGB").copy()
