"""
action_schema.py — Robot Action Types
=======================================
The brain outputs a list of these actions as JSON.
Every action is typed and validated here so bad LLM output fails loudly.

Action JSON format (the brain produces this):
{
  "thought": "...",         # Brain's internal reasoning (always present)
  "actions": [
    {"type": "MOVE",  "direction": "forward", "speed": 0.5, "duration": 1.0},
    {"type": "SPEAK", "text": "Moving forward."},
    {"type": "WAIT",  "duration": 0.5},
    {"type": "LOOK",  "direction": "left"},
    {"type": "THINK", "note": "Obstacle detected, reconsidering route."},
    {"type": "QUERY", "question": "Should I proceed past the door?"}
  ]
}
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal


# ---------------------------------------------------------------------------
# Individual action types
# ---------------------------------------------------------------------------

MoveDirection = Literal["forward", "backward", "left", "right", "stop"]
LookDirection = Literal["left", "right", "up", "down", "center"]


@dataclass
class MoveAction:
    """Drive the robot in a direction."""
    type: Literal["MOVE"] = "MOVE"
    direction: MoveDirection = "stop"
    speed: float = 0.5        # 0.0 – 1.0
    duration: float = 1.0     # seconds

    def validate(self) -> None:
        if not 0.0 <= self.speed <= 1.0:
            raise ValueError(f"MOVE speed must be 0.0–1.0, got {self.speed}")
        if self.duration <= 0:
            raise ValueError(f"MOVE duration must be positive, got {self.duration}")


@dataclass
class SpeakAction:
    """Play audio / text-to-speech."""
    type: Literal["SPEAK"] = "SPEAK"
    text: str = ""

    def validate(self) -> None:
        if not self.text.strip():
            raise ValueError("SPEAK action requires non-empty text.")


@dataclass
class WaitAction:
    """Pause execution."""
    type: Literal["WAIT"] = "WAIT"
    duration: float = 1.0     # seconds

    def validate(self) -> None:
        if self.duration <= 0:
            raise ValueError(f"WAIT duration must be positive, got {self.duration}")


@dataclass
class LookAction:
    """Pan/tilt camera head."""
    type: Literal["LOOK"] = "LOOK"
    direction: LookDirection = "center"

    def validate(self) -> None:
        pass  # direction is already a Literal


@dataclass
class ThinkAction:
    """Internal reasoning only — no hardware effect."""
    type: Literal["THINK"] = "THINK"
    note: str = ""

    def validate(self) -> None:
        pass


@dataclass
class QueryAction:
    """Ask the human operator a question before proceeding."""
    type: Literal["QUERY"] = "QUERY"
    question: str = ""

    def validate(self) -> None:
        if not self.question.strip():
            raise ValueError("QUERY action requires a non-empty question.")


# Union of all action types
AnyAction = MoveAction | SpeakAction | WaitAction | LookAction | ThinkAction | QueryAction

ACTION_TYPE_MAP: dict[str, type] = {
    "MOVE":  MoveAction,
    "SPEAK": SpeakAction,
    "WAIT":  WaitAction,
    "LOOK":  LookAction,
    "THINK": ThinkAction,
    "QUERY": QueryAction,
}


# ---------------------------------------------------------------------------
# Brain response (thought + action list)
# ---------------------------------------------------------------------------

@dataclass
class BrainResponse:
    thought: str
    actions: list[AnyAction] = field(default_factory=list)
    raw: str = ""   # original LLM text, kept for debugging


def parse_brain_response(raw_text: str) -> BrainResponse:
    """
    Parse the LLM's JSON output into a typed BrainResponse.
    Raises ValueError if the JSON is malformed or actions are invalid.
    """
    # Strip markdown code fences if the model wraps in ```json ... ```
    text = raw_text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])

    try:
        data: dict[str, Any] = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Brain returned invalid JSON:\n{raw_text}\n\nError: {exc}") from exc

    thought = data.get("thought", "")
    raw_actions: list[dict[str, Any]] = data.get("actions", [])

    parsed_actions: list[AnyAction] = []
    for raw_action in raw_actions:
        action_type = raw_action.get("type", "").upper()
        cls = ACTION_TYPE_MAP.get(action_type)
        if cls is None:
            raise ValueError(f"Unknown action type: {action_type!r}")
        # Build dataclass from dict, ignoring the 'type' key
        kwargs = {k: v for k, v in raw_action.items() if k != "type"}
        action = cls(**kwargs)  # type: ignore[call-arg]
        action.validate()
        parsed_actions.append(action)  # type: ignore[arg-type]

    return BrainResponse(thought=thought, actions=parsed_actions, raw=raw_text)


def action_schema_prompt() -> str:
    """Return a compact schema description to embed in the system prompt."""
    return """
You MUST respond with valid JSON matching this exact schema (no markdown, no prose):
{
  "thought": "<your internal reasoning>",
  "actions": [
    // One or more of:
    {"type": "MOVE",  "direction": "forward|backward|left|right|stop", "speed": 0.0-1.0, "duration": 1.0},
    {"type": "SPEAK", "text": "<what to say aloud>"},
    {"type": "WAIT",  "duration": 1.0},
    {"type": "LOOK",  "direction": "left|right|up|down|center"},
    {"type": "THINK", "note": "<internal note, no hardware effect>"},
    {"type": "QUERY", "question": "<ask the human operator>"}
  ]
}
Rules:
- Always include "thought" even if actions list is empty.
- MOVE speed is 0.0 to 1.0. duration is seconds (max 5.0 for safety).
- Use QUERY when human input is required before proceeding.
- Prefer THINK before complex action sequences.
"""

