"""
test_brain.py — Phase 2 Test Harness
=======================================
Tests run WITHOUT real hardware, a live LLM, or a real camera.
All external dependencies are replaced with lightweight mocks.

Run:
  cd robot-brain
  pip install pytest
  pytest test_brain.py -v
"""

from __future__ import annotations

import json
import sys
import os

import pytest

# Make sure robot-brain/ is on the path when running from repo root
sys.path.insert(0, os.path.dirname(__file__))


# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------

def make_mock_response(thought: str, actions: list[dict]) -> str:
    """Build a valid JSON brain response string."""
    return json.dumps({"thought": thought, "actions": actions})


class MockLLMAdapter:
    """Returns a canned response. Swapped in place of real LLM for tests."""

    def __init__(self, canned_json: str) -> None:
        self._response = canned_json
        self.call_count = 0
        self.last_messages: list = []

    def chat(self, system_prompt: str, messages: list) -> str:
        self.call_count += 1
        self.last_messages = messages
        return self._response


# ---------------------------------------------------------------------------
# action_schema tests
# ---------------------------------------------------------------------------

class TestActionSchema:

    def test_parse_move_action(self):
        from action_schema import MoveAction, parse_brain_response
        raw = make_mock_response(
            "Moving forward.",
            [{"type": "MOVE", "direction": "forward", "speed": 0.5, "duration": 2.0}],
        )
        resp = parse_brain_response(raw)
        assert resp.thought == "Moving forward."
        assert len(resp.actions) == 1
        action = resp.actions[0]
        assert isinstance(action, MoveAction)
        assert action.direction == "forward"
        assert action.speed == 0.5
        assert action.duration == 2.0

    def test_parse_speak_action(self):
        from action_schema import SpeakAction, parse_brain_response
        raw = make_mock_response(
            "Greeting.",
            [{"type": "SPEAK", "text": "Hello, I am ARIA."}],
        )
        resp = parse_brain_response(raw)
        assert isinstance(resp.actions[0], SpeakAction)
        assert resp.actions[0].text == "Hello, I am ARIA."

    def test_parse_multiple_actions(self):
        from action_schema import MoveAction, SpeakAction, WaitAction, parse_brain_response
        raw = make_mock_response("Multi-step.", [
            {"type": "SPEAK", "text": "Starting."},
            {"type": "MOVE", "direction": "forward", "speed": 0.4, "duration": 1.0},
            {"type": "WAIT", "duration": 0.5},
        ])
        resp = parse_brain_response(raw)
        assert len(resp.actions) == 3
        assert isinstance(resp.actions[0], SpeakAction)
        assert isinstance(resp.actions[1], MoveAction)
        assert isinstance(resp.actions[2], WaitAction)

    def test_invalid_json_raises_value_error(self):
        from action_schema import parse_brain_response
        with pytest.raises(ValueError, match="invalid JSON"):
            parse_brain_response("this is not json at all")

    def test_unknown_action_type_raises(self):
        from action_schema import parse_brain_response
        raw = make_mock_response("oops", [{"type": "EXPLODE"}])
        with pytest.raises(ValueError, match="Unknown action type"):
            parse_brain_response(raw)

    def test_move_speed_out_of_range_raises(self):
        from action_schema import parse_brain_response
        raw = make_mock_response("speed test", [
            {"type": "MOVE", "direction": "forward", "speed": 1.5, "duration": 1.0}
        ])
        with pytest.raises(ValueError, match="speed must be"):
            parse_brain_response(raw)

    def test_strips_markdown_code_fence(self):
        from action_schema import parse_brain_response
        fenced = "```json\n" + make_mock_response("thought", []) + "\n```"
        resp = parse_brain_response(fenced)
        assert resp.thought == "thought"

    def test_query_action_parsed(self):
        from action_schema import QueryAction, parse_brain_response
        raw = make_mock_response("Need input.", [
            {"type": "QUERY", "question": "Should I proceed through the doorway?"}
        ])
        resp = parse_brain_response(raw)
        assert isinstance(resp.actions[0], QueryAction)

    def test_empty_speak_raises(self):
        from action_schema import parse_brain_response
        raw = make_mock_response("empty speak", [{"type": "SPEAK", "text": "  "}])
        with pytest.raises(ValueError):
            parse_brain_response(raw)


# ---------------------------------------------------------------------------
# Brain tests (mock LLM)
# ---------------------------------------------------------------------------

class TestBrain:

    def _make_brain(self, canned_json: str):
        """Return a Brain with its LLM adapter replaced by MockLLMAdapter."""
        from brain import Brain
        from config import RobotConfig
        cfg = RobotConfig()
        brain = Brain(cfg)
        brain._adapter = MockLLMAdapter(canned_json)
        return brain

    def test_brain_returns_brain_response(self):
        from action_schema import BrainResponse, MoveAction
        canned = make_mock_response("Moving.", [
            {"type": "MOVE", "direction": "forward", "speed": 0.3, "duration": 1.0}
        ])
        brain = self._make_brain(canned)
        response = brain.process("go forward")
        assert isinstance(response, BrainResponse)
        assert isinstance(response.actions[0], MoveAction)

    def test_brain_accumulates_history(self):
        canned = make_mock_response("ok", [{"type": "THINK", "note": "fine"}])
        brain = self._make_brain(canned)
        brain.process("first command")
        brain.process("second command")
        assert brain.history_length == 4  # 2 user + 2 assistant

    def test_brain_reset_clears_history(self):
        canned = make_mock_response("ok", [])
        brain = self._make_brain(canned)
        brain.process("command")
        brain.reset()
        assert brain.history_length == 0

    def test_brain_survives_bad_llm_json(self):
        from action_schema import ThinkAction
        brain = self._make_brain("not valid json at all")
        response = brain.process("do something")
        # Should NOT raise — returns a THINK action describing the error
        assert any(isinstance(a, ThinkAction) for a in response.actions)


# ---------------------------------------------------------------------------
# MockHardware tests
# ---------------------------------------------------------------------------

class TestMockHardware:

    def _make_hardware(self):
        from config import HardwareConfig, RobotConfig
        from mock_hardware import MockHardware
        cfg = RobotConfig()
        cfg.hardware.speaker_enabled = False   # don't call espeak during tests
        return MockHardware(cfg.hardware)

    def test_move_returns_string(self):
        from action_schema import MoveAction
        hw = self._make_hardware()
        result = hw.execute(MoveAction(direction="forward", speed=0.5, duration=0.01))
        assert "MOTOR" in result

    def test_speak_returns_string(self):
        from action_schema import SpeakAction
        hw = self._make_hardware()
        result = hw.execute(SpeakAction(text="Hello world"))
        assert "SPEAK" in result

    def test_look_returns_string(self):
        from action_schema import LookAction
        hw = self._make_hardware()
        result = hw.execute(LookAction(direction="left"))
        assert "CAMERA" in result or "LOOK" in result

    def test_think_returns_string(self):
        from action_schema import ThinkAction
        hw = self._make_hardware()
        result = hw.execute(ThinkAction(note="Hmm."))
        assert "THINK" in result


# ---------------------------------------------------------------------------
# ControlLoop smoke test (no real LLM, no real hardware)
# ---------------------------------------------------------------------------

class TestControlLoop:

    def test_step_returns_response(self):
        from action_schema import BrainResponse
        from control_loop import ControlLoop
        from config import RobotConfig
        from mock_hardware import MockHardware

        canned = make_mock_response("Executing move.", [
            {"type": "SPEAK", "text": "Moving."},
            {"type": "MOVE", "direction": "forward", "speed": 0.3, "duration": 0.01},
        ])
        cfg = RobotConfig()
        cfg.hardware.speaker_enabled = False
        loop = ControlLoop(cfg)
        loop._brain._adapter = MockLLMAdapter(canned)
        loop._hardware = MockHardware(cfg.hardware)

        result = loop.step("go forward slowly")
        assert isinstance(result, BrainResponse)

    def test_step_with_vision_disabled(self):
        """ControlLoop step works when vision pipeline is not active."""
        from action_schema import BrainResponse
        from control_loop import ControlLoop
        from config import RobotConfig
        from mock_hardware import MockHardware

        canned = make_mock_response("Thinking.", [{"type": "THINK", "note": "ok"}])
        cfg = RobotConfig()
        cfg.vision.enabled = False
        cfg.hardware.speaker_enabled = False
        loop = ControlLoop(cfg)
        loop._brain._adapter = MockLLMAdapter(canned)
        loop._hardware = MockHardware(cfg.hardware)

        result = loop.step("what do you see?", inject_vision=False)
        assert isinstance(result, BrainResponse)


# ---------------------------------------------------------------------------
# Brain vision_context injection tests
# ---------------------------------------------------------------------------

class TestBrainVisionContext:

    def _make_brain(self, canned_json: str):
        from brain import Brain
        from config import RobotConfig
        cfg = RobotConfig()
        brain = Brain(cfg)
        brain._adapter = MockLLMAdapter(canned_json)
        return brain

    def test_vision_context_prepended_to_message(self):
        """When vision_context is provided, it appears in the message sent to LLM."""
        canned = make_mock_response("I see a chair.", [{"type": "THINK", "note": "noted"}])
        brain = self._make_brain(canned)
        brain.process("what is in front of you?", vision_context="A wooden chair about 1 metre ahead.")
        last_messages = brain._adapter.last_messages
        user_content = last_messages[0]["content"]
        assert "A wooden chair" in user_content
        assert "what is in front of you?" in user_content

    def test_no_vision_context_passes_raw_input(self):
        """Without vision_context, the raw user input is passed unchanged."""
        canned = make_mock_response("ok", [{"type": "THINK", "note": "done"}])
        brain = self._make_brain(canned)
        brain.process("move forward")
        last_messages = brain._adapter.last_messages
        assert last_messages[0]["content"] == "move forward"

    def test_vision_context_does_not_break_history(self):
        """Vision-injected turns still accumulate correctly in history."""
        canned = make_mock_response("ok", [{"type": "THINK", "note": "fine"}])
        brain = self._make_brain(canned)
        brain.process("turn left", vision_context="Open corridor to the left.")
        brain.process("stop")
        assert brain.history_length == 4  # 2 user + 2 assistant


# ---------------------------------------------------------------------------
# MockHardware with vision pipeline
# ---------------------------------------------------------------------------

class MockCamera:
    """Fake camera that returns a predictable base64 string."""
    index = 0

    def capture_jpeg_b64(self, quality: int = 75) -> str:
        import base64
        return base64.b64encode(b"fake-jpeg-data").decode()

    def release(self) -> None:
        pass


class MockVisionAdapter:
    """Fake vision adapter that returns a canned scene description."""
    DESCRIPTION = "A clear path ahead with no obstacles."

    def describe(self, frame_b64: str, prompt: str | None = None) -> str:
        return self.DESCRIPTION


class TestMockHardwareVision:

    def _make_hardware_with_vision(self):
        from config import RobotConfig
        from mock_hardware import MockHardware
        cfg = RobotConfig()
        cfg.hardware.speaker_enabled = False
        return MockHardware(cfg.hardware, camera=MockCamera(), vision=MockVisionAdapter())

    def test_look_with_vision_returns_vision_prefix(self):
        from action_schema import LookAction
        hw = self._make_hardware_with_vision()
        result = hw.execute(LookAction(direction="forward"))
        assert result.startswith("[VISION]")
        assert MockVisionAdapter.DESCRIPTION in result

    def test_look_without_vision_returns_fallback(self):
        """No camera → graceful fallback message."""
        from action_schema import LookAction
        from config import RobotConfig
        from mock_hardware import MockHardware
        cfg = RobotConfig()
        hw = MockHardware(cfg.hardware)
        result = hw.execute(LookAction(direction="left"))
        assert "CAMERA" in result or "LOOK" in result or "no vision" in result

    def test_shutdown_releases_camera(self):
        """Shutdown calls camera.release() without raising."""
        from config import RobotConfig
        from mock_hardware import MockHardware
        cfg = RobotConfig()
        hw = MockHardware(cfg.hardware, camera=MockCamera(), vision=MockVisionAdapter())
        hw.shutdown()  # should not raise


# ---------------------------------------------------------------------------
# VisionConfig defaults
# ---------------------------------------------------------------------------

class TestVisionConfig:

    def test_vision_config_defaults(self):
        from config import VisionConfig
        vcfg = VisionConfig()
        assert vcfg.enabled is True
        assert vcfg.capture_width == 640
        assert vcfg.capture_height == 480
        assert 0 < vcfg.jpeg_quality <= 100

    def test_vision_config_in_robot_config(self):
        from config import RobotConfig, VisionConfig
        cfg = RobotConfig()
        assert isinstance(cfg.vision, VisionConfig)

    def test_vision_disabled_via_env(self, monkeypatch):
        monkeypatch.setenv("VISION_ENABLED", "0")
        # Re-instantiate so env var takes effect
        from config import VisionConfig
        vcfg = VisionConfig()
        assert vcfg.enabled is False

