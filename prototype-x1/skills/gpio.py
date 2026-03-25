"""
GPIO skills for Raspberry Pi 5 (X1 robot).

Auto-registered as: move, speak, look_servo, led

Falls back to mock output on non-Pi hardware so the brain can run
on macOS/dev without errors.
"""
from __future__ import annotations

import logging
import time

log = logging.getLogger("x1.gpio")

# ── Hardware detection ────────────────────────────────────────────────────────

try:
    import RPi.GPIO as _GPIO  # type: ignore[import]
    _HAS_GPIO = True
    _GPIO.setmode(_GPIO.BCM)
    _GPIO.setwarnings(False)
    log.info("RPi.GPIO available — running on Pi hardware")
except ImportError:
    _HAS_GPIO = False
    log.info("RPi.GPIO not found — GPIO skills in mock mode")

try:
    from gpiozero import Motor, Servo, LED as GpioLED  # type: ignore[import]
    _HAS_GPIOZERO = True
except ImportError:
    _HAS_GPIOZERO = False

# ── Pin map (BCM numbering) ───────────────────────────────────────────────────
# Adjust to match your wiring
_MOTOR_LEFT_FWD  = 17
_MOTOR_LEFT_BWD  = 18
_MOTOR_RIGHT_FWD = 27
_MOTOR_RIGHT_BWD = 22
_SERVO_PAN_PIN   = 12   # PWM-capable pin
_SERVO_TILT_PIN  = 13
_LED_PIN         = 26


# ── Motor helpers ─────────────────────────────────────────────────────────────

def _gpio_move(direction: str, duration_s: float) -> None:
    if not _HAS_GPIOZERO:
        return
    left  = Motor(_MOTOR_LEFT_FWD,  _MOTOR_LEFT_BWD)
    right = Motor(_MOTOR_RIGHT_FWD, _MOTOR_RIGHT_BWD)
    try:
        if direction == "forward":
            left.forward(); right.forward()
        elif direction == "backward":
            left.backward(); right.backward()
        elif direction == "left":
            left.backward(); right.forward()
        elif direction == "right":
            left.forward(); right.backward()
        time.sleep(duration_s)
    finally:
        left.stop(); right.stop()


# ── Skill functions ───────────────────────────────────────────────────────────

def skill_move(direction: str = "forward", duration_ms: int = 500) -> str:
    """Move the robot chassis. direction: forward|backward|left|right."""
    direction = direction.lower().strip()
    if direction not in ("forward", "backward", "left", "right"):
        return f"[MOVE ERROR] Unknown direction: {direction!r}"
    duration_s = max(0.05, min(duration_ms / 1000, 10.0))

    if _HAS_GPIOZERO:
        try:
            _gpio_move(direction, duration_s)
            return f"Moved {direction} for {duration_ms}ms"
        except Exception as e:
            return f"[MOVE ERROR] {e}"
    else:
        log.info("MOCK MOVE %s %.2fs", direction, duration_s)
        time.sleep(min(duration_s, 0.1))  # don't block long in mock
        return f"[MOCK] Moved {direction} for {duration_ms}ms"


def skill_look_servo(pan: int = 0, tilt: int = 0) -> str:
    """Aim the camera servo. pan/tilt in degrees (-90 to +90)."""
    pan  = max(-90, min(90, int(pan)))
    tilt = max(-90, min(90, int(tilt)))

    if _HAS_GPIOZERO:
        try:
            pan_servo  = Servo(_SERVO_PAN_PIN)
            tilt_servo = Servo(_SERVO_TILT_PIN)
            pan_servo.value  = pan  / 90.0
            tilt_servo.value = tilt / 90.0
            time.sleep(0.5)
            return f"Servo → pan={pan}° tilt={tilt}°"
        except Exception as e:
            return f"[SERVO ERROR] {e}"
    else:
        log.info("MOCK SERVO pan=%d tilt=%d", pan, tilt)
        return f"[MOCK] Servo → pan={pan}° tilt={tilt}°"


def skill_speak(text: str) -> str:
    """Speak text aloud using espeak (Pi) or say (macOS)."""
    import subprocess, shlex
    for cmd in (["espeak", text], ["say", "-v", "Flo", text]):
        try:
            subprocess.run(cmd, timeout=30, check=True,
                           capture_output=True)
            return f"Spoken: {text[:80]}"
        except FileNotFoundError:
            continue
        except subprocess.CalledProcessError as e:
            return f"[SPEAK ERROR] {e}"
    log.info("MOCK SPEAK: %s", text)
    return f"[MOCK] Would speak: {text[:80]}"


def skill_led(state: str = "on") -> str:
    """Toggle the status LED. state: on|off|blink."""
    state = state.lower().strip()
    if _HAS_GPIOZERO:
        try:
            led = GpioLED(_LED_PIN)
            if state == "on":
                led.on()
            elif state == "off":
                led.off()
            elif state == "blink":
                led.blink(on_time=0.2, off_time=0.2, n=5, background=True)
            return f"LED {state}"
        except Exception as e:
            return f"[LED ERROR] {e}"
    log.info("MOCK LED %s", state)
    return f"[MOCK] LED {state}"

