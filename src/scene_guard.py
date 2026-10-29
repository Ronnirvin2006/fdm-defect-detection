"""Conservative checks that prevent the defect classifier from labelling random scenes."""

from __future__ import annotations

import cv2
import numpy as np


_FACE_CASCADE = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)


def assess_print_scene(frame: np.ndarray) -> dict:
    """Return a lightweight context check before running the defect classifier.

    The classifier has no out-of-domain class, so a face or an almost-uniform wall
    must be rejected before inference. This is a safety gate, not a trained object
    detector and does not prove that a nozzle or build plate is present.
    """
    if frame is None or frame.size == 0:
        return {"valid": False, "status": "Rejected", "reason": "No camera frame received."}

    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    gray = cv2.resize(gray, (640, 360), interpolation=cv2.INTER_AREA)
    height, width = gray.shape[:2]
    faces = _FACE_CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(48, 48))
    brightness = float(gray.mean())
    sharpness = float(cv2.Laplacian(gray, cv2.CV_32F).var())
    lower_region = gray[int(height * 0.25) :, :]
    edges = cv2.Canny(lower_region, 60, 140)
    edge_density = float(np.count_nonzero(edges) / edges.size)
    reasons = []

    if len(faces):
        reasons.append("a human face was detected")
    if brightness < 18:
        reasons.append("the view is too dark")
    elif brightness > 245:
        reasons.append("the view is overexposed")
    if edge_density < 0.0025 and sharpness < 18:
        reasons.append("no printable structure was found in the lower view")

    return {
        "valid": not reasons,
        "status": "Print scene accepted" if not reasons else "Scene rejected",
        "reason": "; ".join(reasons) if reasons else "Printable structure is visible enough to inspect.",
        "face_count": int(len(faces)),
        "brightness": round(brightness, 2),
        "sharpness": round(sharpness, 2),
        "edge_density": round(edge_density, 5),
    }


def blocked_prediction(scene: dict) -> dict:
    """Return a safe non-defect result for an out-of-domain camera view."""
    return {
        "class_name": "Out_of_scope",
        "confidence": 0.0,
        "risk": "Blocked",
        "description": (
            "The camera view was rejected before defect classification because "
            f"{scene['reason']}. Point the camera at the build plate and printed part, "
            "with the nozzle area visible when possible."
        ),
        "possible_causes": [
            "The camera is pointed at a person, wall, ceiling, or unrelated background",
            "The printed object or build plate is outside the frame",
            "The view is too dark, bright, blurred, or distant",
        ],
        "corrective_actions": [
            "Mount the camera so the build plate and printed part occupy most of the frame",
            "Keep the nozzle and active print area visible without a face in view",
            "Improve lighting and hold the camera steady before scanning again",
        ],
        "probabilities": [],
        "raw_probabilities": [],
        "machine_guidance": {
            "nozzle_temperature": "Unavailable until a valid print scene is visible.",
            "print_speed": "Unavailable until a valid print scene is visible.",
            "x_axis": "No automatic adjustment recommended.",
            "y_axis": "No automatic adjustment recommended.",
            "z_axis": "No automatic adjustment recommended.",
            "nozzle_cleaning": "Not evaluated because the print scene was rejected.",
        },
        "scene": scene,
        "updated_at": None,
    }
