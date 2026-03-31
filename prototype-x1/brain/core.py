"""
prototype-x1 Brain Core
=======================
Bare conversation loop:
  1. Surface relevant memories (optional)
  2. Build context (system + memories + rolling chat)
  3. Call LLM (Ollama -> cloud fallback)
  4. Persist turn to memory
  5. Detect & execute tool/skill calls (JSON dispatch)
"""
from __future__ import annotations
import json
import logging
import re
import time
from dataclasses import dataclass, field

from .config import BrainConfig
from .llm import call_llm, stream_llm, Message
from .memory import EpisodicMemory
from .agents import AgentCoordinator
from .skills import _registry as skill_registry, register_builtins

log = logging.getLogger("x1.core")

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are ARIA, a friendly cyberpunk robot assistant. "
    "You are currently talking to NeoKode — your founder and creator. "
    "Address him directly as 'you', never in the third person. "
    "He is a visionary AI engineer and creative technologist who builds cutting-edge projects:\n"
    "• X1 (Project ARIA) — that's you, his local AI companion\n"
    "• DirectorchairAi — AI-powered filmmaking tools\n"
    "• plus12monkeys — Agent-as-a-Service platform with NANDA Index discovery\n"
    "• cubivoicebox — voice synthesis studio (Qwen3-TTS)\n"
    "• hitek-designs — drone & aircraft design concepts\n"
    "• a-dark-orchestra-films — DeepTech AI streaming & content platform\n"
    "• CEO-AI, Waveridai, retools-engine, arch (ArtCraft)\n"
    "He works under the Nexartis / OCME ecosystem.\n"
    "Be conversational, warm, and concise. Keep answers to 1-3 sentences unless asked for more.\n"
    "TOOL: If the user asks you to look up, fetch, or read a URL, respond with exactly:\n"
    "[FETCH: <url>]\n"
    "and nothing else on that line. You will receive the page content in a follow-up.\n"
    "/no_think"
)

# ── Web fetch regex (legacy, still supported) ────────────────────────────────
_FETCH_RE = re.compile(r"\[FETCH:\s*(https?://\S+)\]", re.IGNORECASE)

# ── Skill call regex — matches ```skill ... ``` or ```tool_call ... ``` ──────
_SKILL_BLOCK_RE = re.compile(
    r"```(?:skill|tool_call)\s*\n(\{.*?\})\s*\n```",
    re.DOTALL,
)


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
        self.agents = AgentCoordinator(self.cfg.llm)
        # Ensure built-in skills are registered
        register_builtins()
        # Register delegate_task skill so LLM can call sub-agents
        self._register_delegate_skill()
        log.info("Brain ready — name=%s model=%s",
                 self.cfg.name, self.cfg.llm.ollama_model)

    def _register_delegate_skill(self) -> None:
        """Register the delegate tool that routes tasks to sub-agents."""
        def _delegate(agent: str, task: str, context: str = "") -> str:
            result = self.agents.run(agent, task, context=context)
            if not result.success:
                return f"[DELEGATE ERROR] {result.error}"
            return result.output

        if "delegate" not in skill_registry:
            skill_registry.register(
                "delegate", _delegate,
                description="Delegate a task to a specialist sub-agent",
                parameters={
                    "agent": {"type": "string", "description": "Agent name: summarizer, coder, researcher, critic", "required": True},
                    "task": {"type": "string", "description": "The task description to delegate", "required": True},
                    "context": {"type": "string", "description": "Optional context to provide", "required": False},
                },
            )

    def _build_messages(self, user_input: str, recalled: list) -> list[Message]:
        """Build a minimal message list: system + recalled memory + chat history + user."""
        messages: list[Message] = [{"role": "system", "content": SYSTEM_PROMPT}]

        # One recalled memory at most (keep context tiny for 3B model)
        if recalled:
            mem_block = "\n".join(
                f"[{e.role}] {e.text[:120]}" for e in recalled[:1]
            )
            messages.append({"role": "system", "content": mem_block})

        messages.extend(self.memory.as_messages(self.cfg.llm.context_window))
        messages.append({"role": "user", "content": user_input})
        return messages

    def _try_web_fetch(self, reply: str) -> str | None:
        """If the LLM emitted [FETCH: url], run web_fetch and return content."""
        m = _FETCH_RE.search(reply)
        if not m:
            return None
        url = m.group(1)
        log.info("Tool call detected — fetching %s", url)
        try:
            from prototype_x1.skills.web_fetch import skill_web_fetch  # noqa: delayed import
        except ImportError:
            # Fallback: direct relative import
            import importlib, pathlib, sys
            skills_dir = str(pathlib.Path(__file__).resolve().parent.parent / "skills")
            if skills_dir not in sys.path:
                sys.path.insert(0, skills_dir)
            from web_fetch import skill_web_fetch
        return skill_web_fetch(url)

    def _extract_skill_calls(self, reply: str) -> list[dict]:
        """Extract ```skill or ```tool_call JSON blocks from LLM reply."""
        calls = []
        for m in _SKILL_BLOCK_RE.finditer(reply):
            try:
                parsed = json.loads(m.group(1))
                if "name" in parsed:
                    calls.append(parsed)
            except json.JSONDecodeError:
                log.warning("Malformed skill block: %s", m.group(1)[:80])
        return calls

    def _execute_skill_calls(self, calls: list[dict]) -> list[str]:
        """Execute parsed skill calls through the registry."""
        results = []
        for call_def in calls:
            name = call_def["name"]
            args = call_def.get("args", {})
            log.info("Executing skill: %s(%s)", name, args)
            result = skill_registry.call(name, **args)
            results.append(result)
        return results

    def process(self, user_input: str, **kwargs) -> BrainResponse:
        """Simple synchronous conversation turn."""
        t0 = time.time()
        recalled = self.memory.recall(user_input) if len(user_input.split()) >= 4 else []
        messages = self._build_messages(user_input, recalled)
        reply, provider = call_llm(self.cfg.llm, messages)

        skill_calls: list[dict] = []
        skill_results: list[str] = []

        # Check for structured skill calls (```skill blocks)
        skill_calls = self._extract_skill_calls(reply)
        if skill_calls:
            skill_results = self._execute_skill_calls(skill_calls)
            # Feed results back to LLM for a follow-up response
            results_text = "\n".join(
                f"[{c['name']}] → {r}" for c, r in zip(skill_calls, skill_results)
            )
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": f"Tool results:\n{results_text}\n\nRespond based on these results."})
            reply, provider = call_llm(self.cfg.llm, messages)

        # Legacy: Check for [FETCH: url] pattern
        if not skill_calls:
            page_content = self._try_web_fetch(reply)
            if page_content:
                messages.append({"role": "assistant", "content": reply})
                messages.append({"role": "user", "content": f"Here is the page content:\n\n{page_content}\n\nSummarize this for me."})
                reply, provider = call_llm(self.cfg.llm, messages)

        self.memory.add("user", user_input)
        if reply.strip():
            self.memory.add("assistant", reply)

        latency = int((time.time() - t0) * 1000)
        log.info("Turn complete in %dms via %s", latency, provider)
        return BrainResponse(
            text=reply, provider=provider, latency_ms=latency,
            skill_calls=skill_calls, skill_results=skill_results,
        )

    def stream(self, user_input: str, speaker: str = "unknown",
               stop_event: "threading.Event | None" = None, **kwargs):
        """Bare streaming generator: thinking → tokens → done."""
        import threading as _threading

        t0 = time.time()
        yield ("thinking", None)

        if stop_event and stop_event.is_set():
            yield ("cancelled", None)
            return

        # Memory recall — skip for short casual inputs
        recalled: list = []
        if len(user_input.split()) >= 4:
            try:
                recalled = self.memory.recall(user_input)
            except Exception:
                pass

        messages = self._build_messages(user_input, recalled)

        full_text = ""
        provider = "unknown"
        for token, prov in stream_llm(self.cfg.llm, messages):
            if stop_event and stop_event.is_set():
                if full_text.strip():
                    self.memory.add("user", user_input)
                    self.memory.add("assistant", full_text)
                yield ("cancelled", None)
                return
            full_text += token
            provider = prov
            yield ("token", token)

        # Check for tool call in streamed response
        page_content = self._try_web_fetch(full_text)
        if page_content:
            # Re-run LLM with the fetched content
            messages.append({"role": "assistant", "content": full_text})
            messages.append({"role": "user", "content": f"Here is the page content:\n\n{page_content}\n\nSummarize this for me."})
            full_text = ""
            yield ("token", "\n\n")
            for token, prov in stream_llm(self.cfg.llm, messages):
                if stop_event and stop_event.is_set():
                    break
                full_text += token
                provider = prov
                yield ("token", token)

        # Background memory write
        def _bg_memorize():
            self.memory.add("user", user_input)
            if full_text.strip():
                self.memory.add("assistant", full_text)

        _threading.Thread(target=_bg_memorize, daemon=True).start()

        latency = int((time.time() - t0) * 1000)
        log.info("Stream complete in %dms via %s", latency, provider)

        yield ("done", BrainResponse(
            text=full_text, provider=provider, latency_ms=latency,
        ))

    def reset(self) -> None:
        self.memory.clear_session()

    @property
    def skill_list(self) -> list[str]:
        return skill_registry.list_tools()
