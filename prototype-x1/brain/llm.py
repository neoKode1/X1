"""
LLM dispatch: Claude primary, Ollama fallback.
Claude handles conversation + vision. Ollama is the local safety net.
"""
from __future__ import annotations
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import LLMConfig

log = logging.getLogger("x1.llm")

Message = dict  # {"role": ..., "content": str}

# ── Anthropic (primary) ──────────────────────────────────────────────────────
_anthropic_client = None

def _get_anthropic_client(cfg: "LLMConfig"):
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        _anthropic_client = anthropic.Anthropic(api_key=cfg.anthropic_api_key)
    return _anthropic_client


def _prep_anthropic(messages: list[Message]) -> tuple[str, list[Message]]:
    """Split system prompt from conversation messages."""
    system_parts = [m["content"] for m in messages if m["role"] == "system"]
    system = "\n\n".join(system_parts) if system_parts else ""
    conv = [m for m in messages if m["role"] != "system"]
    return system, conv


def _try_anthropic(cfg: "LLMConfig", messages: list[Message]) -> str | None:
    """Non-streaming Anthropic call."""
    try:
        client = _get_anthropic_client(cfg)
        system, conv = _prep_anthropic(messages)
        resp = client.messages.create(
            model=cfg.anthropic_model,
            max_tokens=cfg.max_tokens,
            system=system,
            messages=conv,
        )
        return resp.content[0].text
    except Exception as e:
        log.warning("Anthropic failed (%s), falling back", e)
        return None


def _stream_anthropic(cfg: "LLMConfig", messages: list[Message]):
    """Yield (token, provider) from Anthropic streaming API."""
    client = _get_anthropic_client(cfg)
    system, conv = _prep_anthropic(messages)
    with client.messages.stream(
        model=cfg.anthropic_model,
        max_tokens=cfg.max_tokens,
        system=system,
        messages=conv,
    ) as stream:
        for text in stream.text_stream:
            yield text, "anthropic"


# ── Ollama (fallback) ────────────────────────────────────────────────────────
_ollama_session = None

def _get_ollama_session():
    global _ollama_session
    if _ollama_session is None:
        import requests as _requests
        _ollama_session = _requests.Session()
    return _ollama_session


def _try_ollama(cfg: "LLMConfig", messages: list[Message]) -> str | None:
    """Non-streaming Ollama call."""
    try:
        session = _get_ollama_session()
        url = f"{cfg.ollama_host.rstrip('/')}/api/chat"
        resp = session.post(url, json={
            "model": cfg.ollama_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": cfg.temperature,
                "num_predict": cfg.max_tokens,
                "num_ctx": cfg.num_ctx,
            },
        }, timeout=60)
        resp.raise_for_status()
        return resp.json().get("message", {}).get("content", "") or ""
    except Exception as e:
        log.warning("Ollama fallback also failed: %s", e)
        return None


def _stream_ollama(cfg: "LLMConfig", messages: list[Message]):
    """Yield (token, provider) from Ollama streaming API."""
    import json as _json
    session = _get_ollama_session()
    url = f"{cfg.ollama_host.rstrip('/')}/api/chat"
    resp = session.post(url, json={
        "model": cfg.ollama_model,
        "messages": messages,
        "stream": True,
        "options": {
            "temperature": cfg.temperature,
            "num_predict": cfg.max_tokens,
            "num_ctx": cfg.num_ctx,
        },
    }, stream=True, timeout=60)
    resp.raise_for_status()
    for line in resp.iter_lines():
        if not line:
            continue
        data = _json.loads(line)
        token = data.get("message", {}).get("content", "")
        if token:
            yield token, "ollama"
        if data.get("done"):
            break


# ── Dispatch: primary → fallback ─────────────────────────────────────────────

def call_llm(cfg: "LLMConfig", messages: list[Message]) -> tuple[str, str]:
    """Non-streaming. Returns (reply_text, provider)."""
    # Primary: Anthropic
    if cfg.anthropic_api_key:
        reply = _try_anthropic(cfg, messages)
        if reply is not None:
            return reply, "anthropic"
    # Fallback: Ollama
    reply = _try_ollama(cfg, messages)
    if reply is not None:
        return reply, "ollama"
    return "", "none"


def stream_llm(cfg: "LLMConfig", messages: list[Message]):
    """Yield (token, provider). Claude primary, Ollama fallback."""
    # Primary: stream from Anthropic
    if cfg.anthropic_api_key:
        try:
            yield from _stream_anthropic(cfg, messages)
            return
        except Exception as e:
            log.warning("Anthropic stream failed (%s), falling back to Ollama", e)

    # Fallback: stream from Ollama
    try:
        yield from _stream_ollama(cfg, messages)
    except Exception as e:
        log.error("All LLM backends failed: %s", e)
