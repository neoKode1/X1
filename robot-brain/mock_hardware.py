"""
mock_hardware.py — Hardware Abstraction Layer
===============================================
Two implementations of the same HardwareInterface:

  MockHardware      — prints actions to terminal, no real pins touched.
                      Use this for Phase 1 testing on any machine.

  RaspberryPi5Hardware — real GPIO via gpiozero + lgpio backend.
                         Compatible with Pi 5's RP1 chip.
                         Install: pip install gpiozero lgpio

Switch by setting HARDWARE_PLATFORM=mock|raspberry_pi in your env,
or call create_hardware(config) which picks automatically.
"""

from __future__ import annotations

import logging
import subprocess
import time
from typing import TYPE_CHECKING, Optional, Protocol

from action_schema import (
    AnyAction, LookAction, MoveAction, QueryAction,
    SpeakAction, ThinkAction, WaitAction,
)
from config import HardwareConfig, RobotConfig, config as default_config

if TYPE_CHECKING:
    from vision.camera import Camera
    from vision.vision_adapter import VisionAdapter

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Hardware protocol
# ---------------------------------------------------------------------------

class HardwareInterface(Protocol):
    def execute(self, action: AnyAction) -> str:
        """Execute one action. Returns a short status string."""
        ...

    def shutdown(self) -> None:
        """Clean up GPIO / release resources."""
        ...


# ---------------------------------------------------------------------------
# Mock hardware (Phase 1 — runs on any machine)
# ---------------------------------------------------------------------------

class MockHardware:
    """Simulates all robot hardware via terminal output."""

    def __init__(
        self,
        cfg: HardwareConfig,
        camera: Optional["Camera"] = None,
        vision: Optional["VisionAdapter"] = None,
    ) -> None:
        self._cfg = cfg
        self._camera = camera
        self._vision = vision
        log.info("MockHardware initialized (no real GPIO).")

    def execute(self, action: AnyAction) -> str:
        if isinstance(action, MoveAction):
            return self._move(action)
        elif isinstance(action, SpeakAction):
            return self._speak(action)
        elif isinstance(action, WaitAction):
            return self._wait(action)
        elif isinstance(action, LookAction):
            return self._look(action)
        elif isinstance(action, ThinkAction):
            log.debug("THINK: %s", action.note)
            return f"[THINK] {action.note}"
        elif isinstance(action, QueryAction):
            return f"[QUERY PENDING] {action.question}"
        return f"[UNKNOWN ACTION] {action}"

    def _move(self, action: MoveAction) -> str:
        duration = min(action.duration, self._cfg.max_move_duration_sec)
        speed_pct = int(action.speed * 100)
        msg = f"[MOTOR] {action.direction.upper()} @ {speed_pct}% for {duration:.1f}s"
        print(f"  🤖 {msg}")
        time.sleep(duration)
        print(f"  🤖 [MOTOR] STOP")
        return msg

    def _speak(self, action: SpeakAction) -> str:
        print(f"  🔊 {action.text}")
        if self._cfg.speaker_enabled:
            try:
                subprocess.run(
                    [self._cfg.tts_command, action.text],
                    timeout=15, check=False,
                    capture_output=True,
                )
            except (FileNotFoundError, subprocess.TimeoutExpired):
                pass   # TTS binary not installed — just print
        return f"[SPEAK] {action.text}"

    def _wait(self, action: WaitAction) -> str:
        print(f"  ⏳ Waiting {action.duration:.1f}s...")
        time.sleep(action.duration)
        return f"[WAIT] {action.duration:.1f}s"

    def _look(self, action: LookAction) -> str:
        if self._camera is not None and self._vision is not None:
            try:
                frame_b64 = self._camera.capture_jpeg_b64(quality=self._cfg.jpeg_quality
                                                           if hasattr(self._cfg, "jpeg_quality") else 75)
                description = self._vision.describe(frame_b64)
                print(f"  👁️  [VISION] {description}")
                return f"[VISION] {description}"
            except Exception as exc:
                log.error("Vision capture failed: %s", exc)
                print(f"  👁️  [VISION ERROR] {exc}")
        # Fallback: no camera attached
        msg = f"[CAMERA] Looking {action.direction} — no vision pipeline active"
        print(f"  👁️  {msg}")
        return msg

    def shutdown(self) -> None:
        if self._camera is not None:
            try:
                self._camera.release()
            except Exception:
                pass
        log.info("MockHardware shutdown.")


# ---------------------------------------------------------------------------
# Real Raspberry Pi 5 hardware
# ---------------------------------------------------------------------------

class RaspberryPi5Hardware:
    """
    Real GPIO control for Raspberry Pi 5.
    Uses gpiozero with the lgpio backend (required for Pi 5's RP1 chip).
    RPi.GPIO is NOT compatible with Pi 5 — do not use it.

    Wiring assumed (BCM pin numbers from config.py):
      Left motor:  forward=17, backward=27, enable(PWM)=18
      Right motor: forward=22, backward=23, enable(PWM)=19
    """

    def __init__(
        self,
        cfg: HardwareConfig,
        camera: Optional["Camera"] = None,
        vision: Optional["VisionAdapter"] = None,
    ) -> None:
        self._cfg = cfg
        self._camera = camera
        self._vision = vision
        self._setup_gpio()
        log.info("RaspberryPi5Hardware initialized (lgpio backend).")

    def _setup_gpio(self) -> None:
        try:
            import os
            os.environ.setdefault("GPIOZERO_PIN_FACTORY", "lgpio")
            from gpiozero import Motor, OutputDevice
            self._left_motor = Motor(
                forward=self._cfg.motor_left_forward_pin,
                backward=self._cfg.motor_left_backward_pin,
                enable=self._cfg.motor_enable_left_pin,
                pin_factory=None,  # uses GPIOZERO_PIN_FACTORY env
            )
            self._right_motor = Motor(
                forward=self._cfg.motor_right_forward_pin,
                backward=self._cfg.motor_right_backward_pin,
                enable=self._cfg.motor_enable_right_pin,
                pin_factory=None,
            )
        except ImportError as exc:
            raise ImportError(
                "gpiozero and lgpio required for Pi 5. "
                "Run: pip install gpiozero lgpio"
            ) from exc

    def execute(self, action: AnyAction) -> str:
        if isinstance(action, MoveAction):
            return self._move(action)
        elif isinstance(action, SpeakAction):
            return self._speak(action)
        elif isinstance(action, WaitAction):
            return self._wait(action)
        elif isinstance(action, LookAction):
            if self._camera is not None and self._vision is not None:
                try:
                    frame_b64 = self._camera.capture_jpeg_b64()
                    description = self._vision.describe(frame_b64)
                    log.info("VISION: %s", description)
                    return f"[VISION] {description}"
                except Exception as exc:
                    log.error("Vision capture failed: %s", exc)
            return f"[LOOK] {action.direction} — no vision pipeline"
        elif isinstance(action, ThinkAction):
            return f"[THINK] {action.note}"
        elif isinstance(action, QueryAction):
            return f"[QUERY PENDING] {action.question}"
        return f"[UNKNOWN] {action}"

    def _move(self, action: MoveAction) -> str:
        duration = min(action.duration, self._cfg.max_move_duration_sec)
        speed = min(action.speed, self._cfg.max_speed)
        direction = action.direction

        if direction == "forward":
            self._left_motor.forward(speed)
            self._right_motor.forward(speed)
        elif direction == "backward":
            self._left_motor.backward(speed)
            self._right_motor.backward(speed)
        elif direction == "left":
            self._left_motor.backward(speed * 0.5)
            self._right_motor.forward(speed)
        elif direction == "right":
            self._left_motor.forward(speed)
            self._right_motor.backward(speed * 0.5)
        elif direction == "stop":
            self._left_motor.stop()
            self._right_motor.stop()
            return "[MOTOR] STOP"

        time.sleep(duration)
        self._left_motor.stop()
        self._right_motor.stop()
        return f"[MOTOR] {direction} @ {int(speed*100)}% for {duration:.1f}s → STOP"

    def _speak(self, action: SpeakAction) -> str:
        try:
            subprocess.run(
                [self._cfg.tts_command, action.text],
                timeout=15, check=False,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            log.warning("TTS command '%s' failed.", self._cfg.tts_command)
        return f"[SPEAK] {action.text}"

    def _wait(self, action: WaitAction) -> str:
        time.sleep(action.duration)
        return f"[WAIT] {action.duration:.1f}s"

    def shutdown(self) -> None:
        try:
            self._left_motor.close()
            self._right_motor.close()
        except Exception:
            pass
        if self._camera is not None:
            try:
                self._camera.release()
            except Exception:
                pass
        log.info("Pi5 GPIO + camera released.")


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_hardware(
    cfg: RobotConfig = default_config,
    camera: Optional["Camera"] = None,
    vision: Optional["VisionAdapter"] = None,
) -> HardwareInterface:
    if cfg.hardware.platform == "raspberry_pi":
        return RaspberryPi5Hardware(cfg.hardware, camera=camera, vision=vision)
    return MockHardware(cfg.hardware, camera=camera, vision=vision)

