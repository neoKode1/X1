"""
config.py — Robot Brain Configuration
======================================
All settings live here. Edit this file or set environment variables.
Environment variables always take precedence over defaults.
"""

import os
from dataclasses import dataclass, field
from typing import Literal

# ---------------------------------------------------------------------------
# LLM Provider
# ---------------------------------------------------------------------------
# Set LLM_PROVIDER=openai or LLM_PROVIDER=anthropic in your environment,
# or change the default below.
LLMProvider = Literal["openai", "anthropic"]

@dataclass
class LLMConfig:
    provider: LLMProvider = field(
        default_factory=lambda: os.getenv("LLM_PROVIDER", "anthropic")  # type: ignore[return-value]
    )

    # OpenAI settings
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "")
    )
    openai_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o")
    )

    # Anthropic settings
    anthropic_api_key: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", "")
    )
    anthropic_model: str = field(
        default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-opus-4-5")
    )

    # Generation settings (shared)
    max_tokens: int = 1024
    temperature: float = 0.2   # Low = more reliable action JSON

    def validate(self) -> None:
        if self.provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set.")
        if self.provider == "anthropic" and not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable not set.")


# ---------------------------------------------------------------------------
# Hardware Platform
# ---------------------------------------------------------------------------
HardwarePlatform = Literal["mock", "raspberry_pi", "jetson"]

@dataclass
class HardwareConfig:
    platform: HardwarePlatform = field(
        default_factory=lambda: os.getenv("HARDWARE_PLATFORM", "mock")  # type: ignore[return-value]
        # Set HARDWARE_PLATFORM=raspberry_pi when running on Pi 5.
        # Pi 5 uses RP1 I/O chip — RPi.GPIO is NOT supported.
        # This code uses lgpio (via gpiozero) which IS compatible with Pi 5.
    )

    # Motor driver pin assignments (BCM numbering for RPi)
    motor_left_forward_pin: int = 17
    motor_left_backward_pin: int = 27
    motor_right_forward_pin: int = 22
    motor_right_backward_pin: int = 23
    motor_enable_left_pin: int = 18   # PWM
    motor_enable_right_pin: int = 19  # PWM

    # Speaker / audio
    speaker_enabled: bool = True
    tts_command: str = "espeak"   # swap for "say" on macOS, "espeak" on Pi

    # Safety limits
    max_speed: float = 1.0        # 0.0 – 1.0 (PWM duty cycle fraction)
    max_move_duration_sec: float = 5.0   # Brain can't move for longer than this

    def is_real_hardware(self) -> bool:
        return self.platform != "mock"


# ---------------------------------------------------------------------------
# Vision
# ---------------------------------------------------------------------------
@dataclass
class VisionConfig:
    enabled: bool = field(
        default_factory=lambda: os.getenv("VISION_ENABLED", "1") == "1"
    )
    # -1 = auto-detect first working camera; 0, 1, 2... = specific index
    camera_index: int = -1
    capture_width: int = 640
    capture_height: int = 480
    jpeg_quality: int = 75   # 0–100; lower = smaller payload, faster round-trip
    scene_prompt: str = (
        "You are the vision system of a mobile robot called ARIA. "
        "Describe what you see in 2–3 concise sentences. "
        "Prioritise: obstacles, open paths, people, key objects, approximate distances. "
        "Be factual and spatial. Do NOT use markdown."
    )


# ---------------------------------------------------------------------------
# Brain / Context
# ---------------------------------------------------------------------------
@dataclass
class BrainConfig:
    # How many conversation turns to keep in memory before summarizing
    context_window_turns: int = 20
    # Show raw LLM responses in the terminal (useful for debugging)
    verbose_llm: bool = field(
        default_factory=lambda: os.getenv("VERBOSE_LLM", "0") == "1"
    )
    # Robot identity / personality (fed into the system prompt)
    robot_name: str = field(
        default_factory=lambda: os.getenv("ROBOT_NAME", "ARIA")
    )
    robot_description: str = (
        "You are ARIA (Autonomous Robotic Intelligence Agent), "
        "a physically embodied AI robot built from salvaged hardware. "
        "You are direct, resourceful, and focused on completing tasks safely."
    )


# ---------------------------------------------------------------------------
# Top-level config bundle
# ---------------------------------------------------------------------------
@dataclass
class RobotConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    brain: BrainConfig = field(default_factory=BrainConfig)
    vision: VisionConfig = field(default_factory=VisionConfig)

    def validate(self) -> None:
        self.llm.validate()


# Singleton — import this everywhere
config = RobotConfig()

