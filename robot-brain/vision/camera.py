"""
vision/camera.py — Camera Capture
====================================
OpenCV wrapper for USB webcams and Pi Camera (via V4L2).

Pi Camera v3 note:
  On Pi 5, enable the camera in raspi-config, then install:
    sudo apt install python3-opencv libcamera-tools
  The Pi Camera shows up as /dev/video0 just like a USB webcam.
  If cv2.VideoCapture(0) fails, try index 1 or 2.

USB webcam: just plug in — index 0 usually works immediately.
"""

from __future__ import annotations

import base64
import logging
from typing import Optional

import cv2
import numpy as np

log = logging.getLogger(__name__)


class Camera:
    """
    Wraps cv2.VideoCapture. Thread-unsafe — use from one thread only.

    Usage:
        with Camera(index=0) as cam:
            b64 = cam.capture_jpeg_b64()
            meta = cam.analyze()
    """

    def __init__(self, index: int = 0, width: int = 640, height: int = 480) -> None:
        self._cap = cv2.VideoCapture(index)
        if not self._cap.isOpened():
            raise RuntimeError(
                f"Cannot open camera at index {index}. "
                "Check that your webcam is plugged in, or try a different index."
            )
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._prev_gray: Optional[np.ndarray] = None
        self.index = index
        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.info("Camera %d opened at %dx%d", index, actual_w, actual_h)

    # ------------------------------------------------------------------
    # Core capture
    # ------------------------------------------------------------------

    def capture(self) -> np.ndarray:
        """Return a BGR frame as a numpy array. Raises RuntimeError on failure."""
        ret, frame = self._cap.read()
        if not ret or frame is None:
            raise RuntimeError("Failed to read frame from camera. Is it still connected?")
        return frame

    def capture_jpeg_b64(self, quality: int = 75) -> str:
        """
        Capture one frame and return it as a base64-encoded JPEG string.
        quality: 0-100 (lower = smaller, faster to send to LLM)
        """
        frame = self.capture()
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            raise RuntimeError("Failed to JPEG-encode frame.")
        return base64.b64encode(buf.tobytes()).decode("utf-8")

    # ------------------------------------------------------------------
    # Local CV analysis (fast — no LLM call)
    # ------------------------------------------------------------------

    def analyze(self) -> dict:
        """
        Quick local computer vision — runs in milliseconds, no API call.
        Returns dict with brightness, motion, and basic scene stats.
        """
        frame = self.capture()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        brightness = float(np.mean(gray))

        # Motion: mean absolute diff vs previous frame
        motion_detected = False
        motion_level = 0.0
        if self._prev_gray is not None and self._prev_gray.shape == gray.shape:
            diff = cv2.absdiff(self._prev_gray, gray)
            motion_level = float(np.mean(diff))
            motion_detected = motion_level > 8.0  # tunable threshold

        self._prev_gray = gray.copy()

        # Edge density as a rough scene complexity measure
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(np.mean(edges > 0))

        h, w = frame.shape[:2]
        return {
            "resolution": f"{w}x{h}",
            "brightness": round(brightness, 1),          # 0 (black) – 255 (white)
            "motion_detected": motion_detected,
            "motion_level": round(motion_level, 2),
            "scene_complexity": round(edge_density, 3),  # 0.0 – 1.0
        }

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def release(self) -> None:
        self._cap.release()
        log.info("Camera %d released.", self.index)

    def __enter__(self) -> "Camera":
        return self

    def __exit__(self, *_) -> None:
        self.release()


# ---------------------------------------------------------------------------
# Factory: auto-detect first working camera
# ---------------------------------------------------------------------------

def auto_detect_camera(
    max_index: int = 4,
    width: int = 640,
    height: int = 480,
) -> Camera:
    """
    Try camera indices 0 … max_index-1 and return the first one that opens.
    Raises RuntimeError if none are found.
    """
    for i in range(max_index):
        try:
            cam = Camera(index=i, width=width, height=height)
            log.info("Auto-detected camera at index %d", i)
            return cam
        except RuntimeError:
            log.debug("Camera index %d not available.", i)

    raise RuntimeError(
        "No camera detected (tried indices 0–%d). "
        "Plug in your webcam and try again. "
        "On Raspberry Pi, ensure the camera is enabled with: sudo raspi-config" % (max_index - 1)
    )

