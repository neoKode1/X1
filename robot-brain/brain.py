"""
brain.py — Core AI Brain
==========================
Receives natural-language commands, sends them to the LLM,
returns a typed BrainResponse (thought + action list).

Maintains a rolling conversation context so the robot remembers
the last N turns without blowing the context window.
"""

from __future__ import annotations

import logging
from collections import deque

from action_schema import BrainResponse, ThinkAction, action_schema_prompt, parse_brain_response
from config import BrainConfig, RobotConfig, config as default_config
from llm_adapter import LLMAdapter, Message, create_adapter

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

def build_system_prompt(cfg: BrainConfig) -> str:
    return f"""
{cfg.robot_description}

Your hardware:
- Platform: Raspberry Pi 5 (8 GB RAM, active cooler)
- Motors: Two DC motors via H-bridge driver (L298N or compatible)
- GPIO library: lgpio / gpiozero (RPi.GPIO is NOT compatible with Pi 5)
- Speaker: 6–8 Ω speaker driven from salvaged TV power board
- Camera: Active (USB webcam via OpenCV). Use LOOK to capture a fresh frame.
  When you receive a [VISION RESULT] message, that is what the camera currently sees.

Operational rules:
1. Safety first. Never MOVE at speed > 0.8 without explicit human permission.
2. Never MOVE for longer than {cfg.robot_description and 5} seconds in a single action.
3. If uncertain, emit THINK then QUERY — do not guess on physical actions.
4. Keep SPEAK text short and spoken-word friendly (no markdown, no lists).
5. You can chain multiple actions in one response.

{action_schema_prompt()}
""".strip()


# ---------------------------------------------------------------------------
# Brain
# ---------------------------------------------------------------------------

class Brain:
    """
    The AI brain.

    Usage:
        brain = Brain()
        response = brain.process("move forward for 2 seconds")
        for action in response.actions:
            ...
    """

    def __init__(self, cfg: RobotConfig = default_config) -> None:
        self._cfg = cfg
        self._adapter: LLMAdapter = create_adapter(cfg.llm)
        self._system_prompt: str = build_system_prompt(cfg.brain)

        # Rolling context window — keeps last N turns
        max_turns = cfg.brain.context_window_turns * 2  # user + assistant per turn
        self._history: deque[Message] = deque(maxlen=max_turns)

        log.info(
            "Brain initialized | provider=%s | robot=%s",
            cfg.llm.provider,
            cfg.brain.robot_name,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, user_input: str, vision_context: str | None = None) -> BrainResponse:
        """
        Send a user command to the brain and return a structured BrainResponse.
        Appends the turn to history for multi-turn context.

        vision_context: if provided, prepended to user_input so the brain
        knows what the camera currently sees before deciding its actions.
        """
        content = (
            f"[CURRENT VISION]: {vision_context}\n{user_input}"
            if vision_context
            else user_input
        )
        self._history.append({"role": "user", "content": content})

        raw = self._adapter.chat(
            system_prompt=self._system_prompt,
            messages=list(self._history),
        )

        if self._cfg.brain.verbose_llm:
            print(f"\n[LLM RAW]\n{raw}\n")

        try:
            response = parse_brain_response(raw)
        except ValueError as exc:
            log.error("Failed to parse brain response: %s", exc)
            # Return a THINK action so the loop doesn't crash
            response = BrainResponse(
                thought=f"Parse error: {exc}",
                actions=[ThinkAction(note=f"Could not parse LLM response. Raw: {raw[:200]}")],
                raw=raw,
            )

        # Record assistant reply in history for next turn
        self._history.append({"role": "assistant", "content": raw})

        return response

    def reset(self) -> None:
        """Clear conversation history (start fresh)."""
        self._history.clear()
        log.info("Brain context reset.")

    @property
    def robot_name(self) -> str:
        return self._cfg.brain.robot_name

    @property
    def history_length(self) -> int:
        return len(self._history)

