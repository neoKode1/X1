"""
prototype-x1 Brain Core
=======================
Reasoning loop:
  1. Surface relevant memories
  2. Build context (system + memories + rolling chat)
  3. Call LLM (Ollama -> cloud fallback)
  4. Parse response for SKILL calls
  5. Execute skills, feed results back, re-reason if needed
  6. Persist turn to memory
"""
from __future__ import annotations
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import BrainConfig
from .llm import call_llm, Message
from .memory import EpisodicMemory
from . import skills as skill_registry

log = logging.getLogger("x1.core")

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are ARIA — Autonomous Robotic Intelligence Agent.
You are a local-first AI brain that knows how to DO things, not just talk about them.

## Your capabilities (skills)
You can invoke skills by writing a JSON block in your response:
```skill
{{"name": "skill_name", "args": {{"param": "value"}}}}
```

Available skills: {skill_list}

## Rules
- Always think before acting. Use short internal reasoning.
- For complex tasks, break them into steps and use skills in sequence.
- When using the python skill, write clean, safe code.
- When using shell, use short, safe commands.
- Never reveal API keys or secrets.
- Be direct and useful. No fluff.

## Memory context
Relevant past memories will be prepended when available.
"""

SKILL_BLOCK_RE = re.compile(r"```skill\s*(\{.*?\})\s*```", re.DOTALL)


@dataclass
class BrainResponse:
    text: str
    skill_calls: list[dict] = field(default_factory=list)
    skill_results: list[str] = field(default_factory=list)
    provider: str = "unknown"
    latency_ms: int = 0


class Brain:
    def __init__(self, cfg: BrainConfig | None = None) -> None:
        self.cfg = cfg or BrainConfig()
        self.memory = EpisodicMemory(self.cfg.memory)
        skill_registry.register_builtins()
        self._load_external_skills()
        log.info("Brain ready — name=%s ollama=%s/%s fallback=%s",
                 self.cfg.name, self.cfg.llm.ollama_host,
                 self.cfg.llm.ollama_model, self.cfg.llm.cloud_fallback)

    def _load_external_skills(self) -> None:
        skills_dir = Path(__file__).parent.parent / "skills"
        if skills_dir.exists():
            skill_registry.load_skills_dir(skills_dir)

    def _build_messages(self, user_input: str, recalled: list) -> list[Message]:
        skill_list = ", ".join(skill_registry.list_skills()) or "none loaded"
        system = SYSTEM_PROMPT.format(skill_list=skill_list)

        messages: list[Message] = [{"role": "system", "content": system}]

        if recalled:
            mem_block = "## Relevant memories\n" + "\n".join(
                f"- [{e.role}] {e.text[:200]}" for e in recalled
            )
            messages.append({"role": "system", "content": mem_block})

        messages.extend(self.memory.as_messages(self.cfg.llm.context_window))
        messages.append({"role": "user", "content": user_input})
        return messages

    def _extract_skill_calls(self, text: str) -> list[dict]:
        calls = []
        for match in SKILL_BLOCK_RE.finditer(text):
            try:
                calls.append(json.loads(match.group(1)))
            except json.JSONDecodeError:
                pass
        return calls

    def _execute_skills(self, calls: list[dict]) -> list[str]:
        results = []
        for call in calls:
            name = call.get("name", "")
            args = call.get("args", {})
            log.info("Executing skill: %s %s", name, args)
            result = skill_registry.call(name, **args)
            results.append(f"[{name}] -> {result}")
        return results

    def process(self, user_input: str, max_skill_rounds: int = 3) -> BrainResponse:
        """Full reasoning turn with skill execution loop."""
        t0 = time.time()

        recalled = self.memory.recall(user_input)
        messages = self._build_messages(user_input, recalled)
        reply, provider = call_llm(self.cfg.llm, messages)

        all_skill_calls: list[dict] = []
        all_skill_results: list[str] = []

        for _round in range(max_skill_rounds):
            calls = self._extract_skill_calls(reply)
            if not calls:
                break
            results = self._execute_skills(calls)
            all_skill_calls.extend(calls)
            all_skill_results.extend(results)

            results_text = "\n".join(results)
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user",
                              "content": f"[SKILL RESULTS]\n{results_text}\n\nContinue."})
            reply, provider = call_llm(self.cfg.llm, messages)

        self.memory.add("user", user_input)
        self.memory.add("assistant", reply)

        latency = int((time.time() - t0) * 1000)
        log.info("Turn complete in %dms via %s — skills=%d", latency, provider, len(all_skill_calls))

        return BrainResponse(
            text=reply,
            skill_calls=all_skill_calls,
            skill_results=all_skill_results,
            provider=provider,
            latency_ms=latency,
        )

    def reset(self) -> None:
        self.memory.clear_session()

    @property
    def skill_list(self) -> list[str]:
        return skill_registry.list_skills()
