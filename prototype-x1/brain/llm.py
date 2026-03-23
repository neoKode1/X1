"""
LLM dispatch: Ollama first, cloud fallback.
Returns a plain string (the assistant reply).
"""
from __future__ import annotations
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import LLMConfig

log = logging.getLogger("x1.llm")

Message = dict  # {"role": "user"|"assistant"|"system", "content": str}


def _try_ollama(cfg: "LLMConfig", messages: list[Message]) -> str | None:
    try:
        import ollama
        resp = ollama.chat(
            model=cfg.ollama_model,
            messages=messages,
            options={"temperature": cfg.temperature, "num_predict": cfg.max_tokens},
        )
        return resp["message"]["content"]
    except Exception as e:
        log.warning("Ollama unavailable (%s), falling back to cloud", e)
        return None


def _try_anthropic(cfg: "LLMConfig", messages: list[Message]) -> str:
    import anthropic
    system = next((m["content"] for m in messages if m["role"] == "system"), "")
    conv = [m for m in messages if m["role"] != "system"]
    client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)
    resp = client.messages.create(
        model=cfg.anthropic_model,
        max_tokens=cfg.max_tokens,
        system=system,
        messages=conv,
    )
    return resp.content[0].text


def _try_openai(cfg: "LLMConfig", messages: list[Message]) -> str:
    from openai import OpenAI
    client = OpenAI(api_key=cfg.openai_api_key)
    resp = client.chat.completions.create(
        model=cfg.openai_model,
        messages=messages,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
    )
    return resp.choices[0].message.content or ""


def stream_llm(cfg: "LLMConfig", messages: list[Message]):
    """Yield (token_text, provider) tuples as they arrive from Ollama.

    Falls back to collecting the full cloud reply and yielding it as one chunk
    if Ollama is unavailable (cloud providers don't stream here yet).
    """
    try:
        import ollama
        stream = ollama.chat(
            model=cfg.ollama_model,
            messages=messages,
            stream=True,
            options={"temperature": cfg.temperature, "num_predict": cfg.max_tokens},
        )
        for chunk in stream:
            token = chunk.get("message", {}).get("content", "")
            if token:
                yield token, "ollama"
        return
    except Exception as e:
        log.warning("Ollama streaming unavailable (%s), falling back to cloud", e)

    # Cloud fallback — yield the full reply as a single token
    reply, provider = call_llm(cfg, messages)
    yield reply, provider


def call_llm(cfg: "LLMConfig", messages: list[Message]) -> tuple[str, str]:
    """Returns (reply_text, provider_used)."""
    # 1. Try Ollama
    reply = _try_ollama(cfg, messages)
    if reply is not None:
        return reply, "ollama"

    # 2. Cloud fallback
    if cfg.cloud_fallback == "anthropic" and cfg.anthropic_api_key:
        return _try_anthropic(cfg, messages), "anthropic"
    if cfg.cloud_fallback == "openai" and cfg.openai_api_key:
        return _try_openai(cfg, messages), "openai"

    # 3. Try whichever cloud key is present
    if cfg.anthropic_api_key:
        return _try_anthropic(cfg, messages), "anthropic"
    if cfg.openai_api_key:
        return _try_openai(cfg, messages), "openai"

    raise RuntimeError("No LLM available: Ollama offline and no cloud API key set.")
