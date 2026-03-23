"""
Material Tagger
================
Runs in a background thread. Grabs a webcam frame every second,
classifies visible materials into ARIA's scrap list.

Classification is local — no cloud calls. Uses simple heuristics on
HSV color ranges + edge density. Good enough for the scrap pile phase.
ARIA decides what's worth keeping. No human approval needed.

Scrap list persists to memory/scrap_list.json.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

log = logging.getLogger("x1.material_tagger")

SCRAP_PATH = Path(__file__).parent.parent / "memory" / "scrap_list.json"

# ── Material categories ARIA can recognise ──────────────────────────────────
MATERIAL_LABELS = [
    "shiny_metal",
    "aluminum",
    "bent_wire",
    "usable_plastic_shell",
    "crackable_plastic",
    "dark_metal",
    "fabric",
    "cardboard",
    "trash",
    "unknown",
]


@dataclass
class ScrapEntry:
    label: str
    confidence: float          # 0.0–1.0 (heuristic estimate)
    first_seen: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    last_seen: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    sightings: int = 1
    notes: str = ""


class MaterialTagger:
    """
    Background material scanner. Runs independently once start() is called.
    Observers can subscribe via on_detection(label, confidence).
    """

    def __init__(
        self,
        interval_s: float = 1.0,
        scrap_path: Path = SCRAP_PATH,
        on_detection: Optional[Callable[[str, float], None]] = None,
    ) -> None:
        self.interval_s = interval_s
        self.scrap_path = scrap_path
        self.on_detection = on_detection
        self._scrap: dict[str, ScrapEntry] = {}
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._load()

    # ── Persistence ────────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self.scrap_path.exists():
            try:
                raw = json.loads(self.scrap_path.read_text())
                self._scrap = {k: ScrapEntry(**v) for k, v in raw.items()}
                log.info("Scrap list: %d item(s) loaded", len(self._scrap))
            except Exception as exc:
                log.warning("Scrap list load failed: %s", exc)

    def _save(self) -> None:
        self.scrap_path.parent.mkdir(parents=True, exist_ok=True)
        self.scrap_path.write_text(
            json.dumps({k: asdict(v) for k, v in self._scrap.items()}, indent=2)
        )

    # ── Classification ─────────────────────────────────────────────────────────

    def _classify_frame(self, frame) -> list[tuple[str, float]]:
        """
        HSV + edge heuristic classifier. Returns list of (label, confidence).
        Falls back to ["unknown", 0.5] if cv2 is unavailable.
        """
        try:
            import cv2
            import numpy as np

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_density = edges.mean() / 255.0

            results: list[tuple[str, float]] = []

            # Shiny metal: high saturation+value, moderate edges
            mask_shiny = cv2.inRange(hsv, (0, 0, 200), (180, 50, 255))
            if mask_shiny.mean() > 15:
                results.append(("shiny_metal", min(0.9, mask_shiny.mean() / 50)))

            # Aluminum: low sat, high value, high edge density
            mask_al = cv2.inRange(hsv, (0, 0, 170), (180, 40, 240))
            if mask_al.mean() > 10 and edge_density > 0.08:
                results.append(("aluminum", min(0.85, mask_al.mean() / 40)))

            # Bent wire: very high edge density, thin structures
            if edge_density > 0.18:
                results.append(("bent_wire", min(0.8, edge_density)))

            # Plastic shell: moderate sat, smooth (low edges)
            mask_pl = cv2.inRange(hsv, (0, 30, 80), (180, 200, 220))
            if mask_pl.mean() > 20 and edge_density < 0.1:
                results.append(("usable_plastic_shell", min(0.75, mask_pl.mean() / 60)))

            # Dark metal: low value, moderate edges
            mask_dark = cv2.inRange(hsv, (0, 0, 0), (180, 60, 80))
            if mask_dark.mean() > 20 and edge_density > 0.05:
                results.append(("dark_metal", min(0.7, mask_dark.mean() / 60)))

            return results or [("unknown", 0.4)]

        except ImportError:
            return [("unknown", 0.5)]
        except Exception as exc:
            log.debug("Frame classify error: %s", exc)
            return [("unknown", 0.3)]

    # ── Record ─────────────────────────────────────────────────────────────────

    def _record(self, label: str, confidence: float) -> None:
        now = datetime.now(timezone.utc).isoformat()
        if label in self._scrap:
            e = self._scrap[label]
            e.sightings += 1
            e.last_seen = now
            e.confidence = max(e.confidence, confidence)
        else:
            self._scrap[label] = ScrapEntry(label=label, confidence=confidence)
            log.info("New scrap item tagged: %s (%.0f%%)", label, confidence * 100)
            if self.on_detection:
                try:
                    self.on_detection(label, confidence)
                except Exception:
                    pass
        self._save()

    # ── Background loop ────────────────────────────────────────────────────────

    def _loop(self) -> None:
        try:
            import cv2
            cap = cv2.VideoCapture(0)
        except ImportError:
            log.warning("cv2 not available — material tagger in stub mode")
            cap = None

        while self._running:
            if cap and cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    for label, conf in self._classify_frame(frame):
                        self._record(label, conf)
            time.sleep(self.interval_s)

        if cap:
            cap.release()

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="material-tagger")
        self._thread.start()
        log.info("Material tagger started (interval=%.1fs)", self.interval_s)

    def stop(self) -> None:
        self._running = False

    # ── Public API ─────────────────────────────────────────────────────────────

    def get_scrap_list(self) -> dict[str, ScrapEntry]:
        return dict(self._scrap)

    def report(self) -> str:
        if not self._scrap:
            return "Scrap list empty — nothing tagged yet."
        lines = ["=== Scrap List ==="]
        for label, e in sorted(self._scrap.items(), key=lambda x: -x[1].sightings):
            lines.append(f"  {label}: seen {e.sightings}x, conf={e.confidence:.0%}")
        return "\n".join(lines)

