"""
Vision capture — screen grab or webcam frame.
Works on macOS (dev) and Raspberry Pi 5 (production).
"""
from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass

log = logging.getLogger("x1.vision")


@dataclass
class Frame:
    frame_b64: str   # JPEG, base64-encoded
    source: str      # 'screen' | 'webcam'
    width: int
    height: int


def capture_screen(max_size: tuple[int, int] = (640, 480)) -> Frame:
    """Grab the primary display using Pillow ImageGrab."""
    try:
        from PIL import ImageGrab  # type: ignore[import]
    except ImportError:
        raise RuntimeError("Pillow not installed — run: pip install Pillow")

    try:
        img = ImageGrab.grab()
        if img is None:
            raise RuntimeError("ImageGrab returned None — Screen Recording permission may be denied")
        w, h = img.size
        if w == 0 or h == 0:
            raise RuntimeError("ImageGrab returned empty image — Screen Recording permission likely denied")
        img = img.convert("RGB")
        img.thumbnail(max_size)
        tw, th = img.size

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=70)
        b64 = base64.b64encode(buf.getvalue()).decode()
        log.info("Screen captured: %dx%d → %dx%d (%d B b64)", w, h, tw, th, len(b64))
        return Frame(frame_b64=b64, source="screen", width=tw, height=th)
    except Exception as e:
        raise RuntimeError(
            f"Screen capture failed ({e}). "
            "On macOS, grant Screen Recording permission: "
            "System Settings → Privacy & Security → Screen Recording → enable Terminal"
        ) from e


def capture_webcam(device: int = 0, max_size: tuple[int, int] = (640, 480)) -> Frame | None:
    """Capture one frame from a webcam via OpenCV. Returns None if unavailable."""
    try:
        import cv2  # type: ignore[import]
    except ImportError:
        log.debug("opencv-python not installed — webcam unavailable")
        return None

    cap = cv2.VideoCapture(device)
    if not cap.isOpened():
        log.warning("Webcam device %d not available", device)
        return None

    ret, frame = cap.read()
    cap.release()
    if not ret:
        log.warning("Webcam read failed on device %d", device)
        return None

    h, w = frame.shape[:2]
    if w > max_size[0] or h > max_size[1]:
        scale = min(max_size[0] / w, max_size[1] / h)
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
        h, w = frame.shape[:2]

    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    b64 = base64.b64encode(buf.tobytes()).decode()
    log.info("Webcam captured: %dx%d (%d B b64)", w, h, len(b64))
    return Frame(frame_b64=b64, source="webcam", width=w, height=h)


def capture(source: str = "screen", device: int = 0) -> Frame:
    """Capture from source ('screen' or 'webcam'), falling back to screen."""
    if source == "webcam":
        frame = capture_webcam(device=device)
        if frame is not None:
            return frame
        log.warning("Webcam unavailable — falling back to screen capture")
    return capture_screen()


def describe_frame(frame: Frame, api_key: str) -> str:
    """Send a captured frame to Anthropic and return a one-sentence description."""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        resp = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=80,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": frame.frame_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": (
                            "Describe what you see in one short sentence. "
                            "Be specific: objects, people, environment. No filler."
                        ),
                    },
                ],
            }],
        )
        return resp.content[0].text.strip()
    except Exception as e:
        log.warning("Vision describe failed: %s", e)
        return f"[vision error] {e}"

