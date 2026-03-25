"""
LLM dispatch: Ollama first, cloud fallback.
Returns a plain string (the assistant reply).

Supports multimodal (image) messages for vision-capable models like Qwen3-VL.
Messages may include an "images" key with a list of base64-encoded image strings.
"""
from __future__ import annotations
import logging
from typing import TYPE_CHECKING

from .filters import strip_think_blocks

if TYPE_CHECKING:
    from .config import LLMConfig

log = logging.getLogger("x1.llm")

Message = dict  # {"role": ..., "content": str, "images"?: list[str]}


def _try_ollama(cfg: "LLMConfig", messages: list[Message]) -> str | None:
    """Non-streaming Ollama call via raw HTTP (bypasses broken ollama library think=False)."""
    try:
        import requests as _requests
        import json as _json

        base_url = cfg.ollama_host.rstrip("/")
        url = f"{base_url}/api/chat"
        resp = _requests.post(url, json={
            "model": cfg.ollama_model,
            "messages": messages,
            "stream": False,
            "think": False,
            "options": {
                "temperature": cfg.temperature,
                "num_predict": cfg.max_tokens,
                "num_ctx": cfg.num_ctx,
            },
        }, timeout=120)
        resp.raise_for_status()
        data = resp.json()
        content = data.get("message", {}).get("content", "") or ""
        return strip_think_blocks(content)
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





def _ollama_stream_attempt(cfg: "LLMConfig", messages: list[Message],
                           timeout_sec: float = 120,
                           think_timeout_sec: float = 15,
                           temperature: float | None = None):
    """Single raw-HTTP streaming attempt against Ollama.

    Yields (token_text, "ollama") tuples.
    Raises _ThinkingTimeout if no content token arrives within think_timeout_sec.
    Raises on connection / HTTP errors.
    """
    import time as _time
    import requests as _requests
    import json as _json

    base_url = cfg.ollama_host.rstrip("/")
    url = f"{base_url}/api/chat"
    t0 = _time.monotonic()

    resp = _requests.post(url, json={
        "model": cfg.ollama_model,
        "messages": messages,
        "stream": True,
        "think": False,
        "options": {
            "temperature": temperature if temperature is not None else cfg.temperature,
            "num_predict": cfg.max_tokens,
            "num_ctx": cfg.num_ctx,
        },
    }, stream=True, timeout=timeout_sec)
    resp.raise_for_status()

    first_content_time = None
    thinking_chars = 0
    content_chars = 0

    for line in resp.iter_lines():
        if not line:
            continue
        data = _json.loads(line)
        msg = data.get("message", {})

        thinking = msg.get("thinking", "")
        if thinking:
            thinking_chars += len(thinking)

        # Check thinking timeout — abort early if model is stuck
        elapsed = _time.monotonic() - t0
        if first_content_time is None and elapsed > think_timeout_sec:
            resp.close()
            raise _ThinkingTimeout(thinking_chars, elapsed)

        token = msg.get("content", "")
        if not token:
            if data.get("done"):
                break
            continue

        if first_content_time is None:
            first_content_time = _time.monotonic()
            log.info("⏱ First content token at %.1fs (after %d chars thinking)",
                     first_content_time - t0, thinking_chars)

        clean = strip_think_blocks(token)
        if clean:
            content_chars += len(clean)
            yield clean, "ollama"

        if data.get("done"):
            break

    elapsed = _time.monotonic() - t0
    log.info("⏱ Ollama done in %.1fs — thinking=%d chars, content=%d chars",
             elapsed, thinking_chars, content_chars)


class _ThinkingTimeout(Exception):
    def __init__(self, thinking_chars: int, elapsed: float):
        self.thinking_chars = thinking_chars
        self.elapsed = elapsed
        super().__init__(f"Thinking timeout: {thinking_chars} chars in {elapsed:.1f}s")


_MINIMAL_SYSTEM = (
    "You are ARIA, a cyberpunk robot. Neokode is your founder. "
    "Answer in 1-2 sentences. Be direct."
)


def _strip_to_minimal(messages: list[Message]) -> list[Message]:
    """Return an ultra-minimal prompt: tiny system + last user message only."""
    user = [m for m in reversed(messages) if m.get("role") == "user"][:1]
    return [{"role": "system", "content": _MINIMAL_SYSTEM}] + user


def stream_llm(cfg: "LLMConfig", messages: list[Message]):
    """Yield (token_text, provider) tuples as they arrive from Ollama.

    Strategy: attempt streaming with full context. If the model gets stuck
    thinking for >25s, abort and retry with a minimal prompt (system + user
    only, no history/memories). If that also fails, fall back to cloud.
    """
    # Attempt 1: full context, moderate thinking budget (20s)
    # If the model can answer with full context, it usually does within 20s.
    # If not, the minimal prompt retry is faster than waiting longer.
    try:
        yield from _ollama_stream_attempt(cfg, messages, think_timeout_sec=20)
        return
    except _ThinkingTimeout as e:
        log.warning("⏱ Thinking timeout (attempt 1): %d chars in %.1fs — retrying minimal",
                    e.thinking_chars, e.elapsed)
    except Exception as e:
        log.warning("Ollama streaming failed (attempt 1): %s", e)
        # Fall through to retry below

    # Attempt 2: minimal prompt + low temp + generous thinking budget (45s)
    try:
        minimal = _strip_to_minimal(messages)
        log.info("⏱ Retry with minimal prompt (%d messages), temp=0.15", len(minimal))
        yield from _ollama_stream_attempt(cfg, minimal, timeout_sec=90,
                                          think_timeout_sec=45, temperature=0.15)
        return
    except _ThinkingTimeout as e:
        log.warning("⏱ Thinking timeout (attempt 2): %d chars in %.1fs — cloud fallback",
                    e.thinking_chars, e.elapsed)
    except Exception as e:
        log.warning("Ollama streaming failed (attempt 2): %s — cloud fallback", e)

    # Cloud fallback — skip Ollama entirely (it's likely still busy)
    reply, provider = _cloud_only(cfg, messages)
    if reply:
        yield reply, provider
    else:
        log.error("All LLM backends failed — no response")


def _cloud_only(cfg: "LLMConfig", messages: list[Message]) -> tuple[str, str]:
    """Try cloud providers only — skips Ollama entirely."""
    try:
        if cfg.cloud_fallback == "anthropic" and cfg.anthropic_api_key:
            return _try_anthropic(cfg, messages), "anthropic"
        if cfg.cloud_fallback == "openai" and cfg.openai_api_key:
            return _try_openai(cfg, messages), "openai"
        if cfg.anthropic_api_key:
            return _try_anthropic(cfg, messages), "anthropic"
        if cfg.openai_api_key:
            return _try_openai(cfg, messages), "openai"
    except Exception as e:
        log.error("Cloud fallback failed: %s", e)
    return "", "none"


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
