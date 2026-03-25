"""
Centralized text sanitization for ARIA responses.
All output cleaning — banned phrases, JSON stripping, think-token removal —
lives here so it's applied consistently across streaming and non-streaming paths.
"""
from __future__ import annotations

import re

# ── Think-token removal ──────────────────────────────────────────────────────

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think_blocks(text: str) -> str:
    """Remove Qwen3 <think>...</think> reasoning blocks from text."""
    if "<think>" not in text:
        return text
    return _THINK_RE.sub("", text).strip()


# ── Banned phrases (model hallucinations) ─────────────────────────────────────

_BANNED_PHRASES = [
    "no response is required",
    "screengrab now",
    "action completed",
    "the speak skill has completed",
    "your feedback is acknowledged",
    "relevant memory prepended",
]


def strip_banned(text: str) -> str:
    """Remove known hallucinated phrases from text."""
    for phrase in _BANNED_PHRASES:
        text = re.sub(re.escape(phrase), "", text, flags=re.IGNORECASE)
    return text.strip()


# ── Bare JSON stripping ──────────────────────────────────────────────────────

def strip_bare_json(text: str) -> str:
    """Remove bare JSON skill-call objects from text, handling nested braces."""
    result = []
    i = 0
    while i < len(text):
        if text[i] == '{':
            depth = 0
            j = i
            while j < len(text):
                if text[j] == '{':
                    depth += 1
                elif text[j] == '}':
                    depth -= 1
                    if depth == 0:
                        i = j + 1
                        break
                j += 1
            else:
                i = j
            continue
        result.append(text[i])
        i += 1
    return "".join(result)


# ── Combined pipeline ────────────────────────────────────────────────────────

def clean_response(text: str) -> str:
    """Full sanitization pipeline for LLM output."""
    text = strip_think_blocks(text)
    text = strip_bare_json(text)
    text = strip_banned(text)
    return text

