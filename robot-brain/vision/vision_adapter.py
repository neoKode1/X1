"""
vision/vision_adapter.py — Vision LLM Interface
=================================================
Sends a JPEG frame (base64) to GPT-4o vision or Claude vision.
Returns a natural-language scene description for the brain to reason over.

Supported:
  OpenAI  — gpt-4o          (image_url with base64 data URI)
  Anthropic — claude-3-5-sonnet / claude-opus-4-5  (image content block)
"""

from __future__ import annotations

import logging
from typing import Protocol

from config import LLMConfig, VisionConfig

log = logging.getLogger(__name__)

# Default robot-vision prompt (overridable via VisionConfig)
DEFAULT_SCENE_PROMPT = (
    "You are the vision system of a mobile robot called ARIA. "
    "Describe what you see in 2–3 concise sentences. "
    "Prioritise: obstacles, open paths, people, key objects, approximate distances. "
    "Be factual and spatial. Do NOT use markdown."
)


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

class VisionAdapter(Protocol):
    def describe(self, frame_b64: str, prompt: str | None = None) -> str:
        """Send a JPEG base64 frame to the vision model. Return scene description."""
        ...


# ---------------------------------------------------------------------------
# OpenAI vision adapter
# ---------------------------------------------------------------------------

class OpenAIVisionAdapter:
    def __init__(self, cfg: LLMConfig, vision_cfg: VisionConfig) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("pip install openai") from exc
        self._client = OpenAI(api_key=cfg.openai_api_key)
        self._model = cfg.openai_model          # gpt-4o supports vision
        self._max_tokens = 300                  # descriptions are short
        self._scene_prompt = vision_cfg.scene_prompt

    def describe(self, frame_b64: str, prompt: str | None = None) -> str:
        text_prompt = prompt or self._scene_prompt
        data_uri = f"data:image/jpeg;base64,{frame_b64}"
        log.debug("OpenAI vision request | model=%s", self._model)
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": text_prompt},
                        {"type": "image_url", "image_url": {"url": data_uri, "detail": "low"}},
                    ],
                }],
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            raise RuntimeError(f"OpenAI vision error: {exc}") from exc


# ---------------------------------------------------------------------------
# Anthropic vision adapter
# ---------------------------------------------------------------------------

class AnthropicVisionAdapter:
    def __init__(self, cfg: LLMConfig, vision_cfg: VisionConfig) -> None:
        try:
            import anthropic  # noqa: F401
        except ImportError as exc:
            raise ImportError("pip install anthropic") from exc
        import anthropic as _anthropic
        self._client = _anthropic.Anthropic(api_key=cfg.anthropic_api_key)
        self._model = cfg.anthropic_model
        self._max_tokens = 300
        self._scene_prompt = vision_cfg.scene_prompt

    def describe(self, frame_b64: str, prompt: str | None = None) -> str:
        text_prompt = prompt or self._scene_prompt
        log.debug("Anthropic vision request | model=%s", self._model)
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": frame_b64,
                            },
                        },
                        {"type": "text", "text": text_prompt},
                    ],
                }],
            )
            return response.content[0].text if response.content else ""
        except Exception as exc:
            raise RuntimeError(f"Anthropic vision error: {exc}") from exc


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_vision_adapter(llm_cfg: LLMConfig, vision_cfg: VisionConfig) -> VisionAdapter:
    if llm_cfg.provider == "openai":
        log.info("Vision adapter: OpenAI (%s)", llm_cfg.openai_model)
        return OpenAIVisionAdapter(llm_cfg, vision_cfg)
    elif llm_cfg.provider == "anthropic":
        log.info("Vision adapter: Anthropic (%s)", llm_cfg.anthropic_model)
        return AnthropicVisionAdapter(llm_cfg, vision_cfg)
    else:
        raise ValueError(f"Unknown LLM provider for vision: {llm_cfg.provider!r}")

