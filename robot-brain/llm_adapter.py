"""
llm_adapter.py — Unified LLM Interface
========================================
Provides a single call() interface that works with either:
  - OpenAI  (gpt-4o, o1-preview, etc.)
  - Anthropic (claude-opus-4-5, claude-3-5-sonnet, etc.)

Switch providers by setting LLM_PROVIDER=openai|anthropic in your env.
"""

from __future__ import annotations

import logging
from typing import Protocol

from config import LLMConfig

log = logging.getLogger(__name__)

Message = dict[str, str]   # {"role": "user"|"assistant"|"system", "content": "..."}


# ---------------------------------------------------------------------------
# Provider protocol — both adapters implement this
# ---------------------------------------------------------------------------

class LLMAdapter(Protocol):
    def chat(
        self,
        system_prompt: str,
        messages: list[Message],
    ) -> str:
        """
        Send a conversation to the LLM and return the assistant's reply text.
        Raises RuntimeError on API errors.
        """
        ...


# ---------------------------------------------------------------------------
# OpenAI adapter
# ---------------------------------------------------------------------------

class OpenAIAdapter:
    def __init__(self, cfg: LLMConfig) -> None:
        try:
            from openai import OpenAI  # lazy import so anthropic-only setups don't break
        except ImportError as exc:
            raise ImportError("Install openai: pip install openai") from exc

        self._client = OpenAI(api_key=cfg.openai_api_key)
        self._model = cfg.openai_model
        self._max_tokens = cfg.max_tokens
        self._temperature = cfg.temperature

    def chat(self, system_prompt: str, messages: list[Message]) -> str:
        full_messages: list[Message] = [
            {"role": "system", "content": system_prompt},
            *messages,
        ]
        log.debug("OpenAI request | model=%s | turns=%d", self._model, len(messages))
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=full_messages,  # type: ignore[arg-type]
                max_tokens=self._max_tokens,
                temperature=self._temperature,
            )
            text = response.choices[0].message.content or ""
            log.debug("OpenAI response | tokens=%s", response.usage)
            return text
        except Exception as exc:
            raise RuntimeError(f"OpenAI API error: {exc}") from exc


# ---------------------------------------------------------------------------
# Anthropic adapter
# ---------------------------------------------------------------------------

class AnthropicAdapter:
    def __init__(self, cfg: LLMConfig) -> None:
        try:
            import anthropic  # lazy import
        except ImportError as exc:
            raise ImportError("Install anthropic: pip install anthropic") from exc

        self._client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)
        self._model = cfg.anthropic_model
        self._max_tokens = cfg.max_tokens
        self._temperature = cfg.temperature

    def chat(self, system_prompt: str, messages: list[Message]) -> str:
        # Anthropic uses system separately from messages
        anthropic_messages = [
            {"role": m["role"], "content": m["content"]}
            for m in messages
            if m["role"] in ("user", "assistant")
        ]
        log.debug("Anthropic request | model=%s | turns=%d", self._model, len(anthropic_messages))
        try:
            import anthropic
            response = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system_prompt,
                messages=anthropic_messages,  # type: ignore[arg-type]
            )
            text = response.content[0].text if response.content else ""
            log.debug("Anthropic response | stop=%s", response.stop_reason)
            return text
        except Exception as exc:
            raise RuntimeError(f"Anthropic API error: {exc}") from exc


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_adapter(cfg: LLMConfig) -> LLMAdapter:
    """Return the correct adapter based on config."""
    if cfg.provider == "openai":
        log.info("LLM provider: OpenAI (%s)", cfg.openai_model)
        return OpenAIAdapter(cfg)
    elif cfg.provider == "anthropic":
        log.info("LLM provider: Anthropic (%s)", cfg.anthropic_model)
        return AnthropicAdapter(cfg)
    else:
        raise ValueError(f"Unknown LLM provider: {cfg.provider!r}. Use 'openai' or 'anthropic'.")

