"""
Agent coordination for ARIA.
Allows the Brain to spawn sub-agents — isolated LLM calls with specialized
prompts that handle subtasks and return results.

Each sub-agent gets its own system prompt, runs a single LLM call (or a
short chain), and returns a result string. The parent Brain decides when
to delegate and how to use the result.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from .config import LLMConfig
from .llm import call_llm, Message

log = logging.getLogger("x1.agents")


@dataclass
class AgentResult:
    """Result from a sub-agent execution."""
    agent_name: str
    output: str
    provider: str = "unknown"
    latency_ms: int = 0
    success: bool = True
    error: str = ""


@dataclass
class AgentDef:
    """Definition of a reusable sub-agent."""
    name: str
    system_prompt: str
    description: str = ""
    max_tokens: int = 1024


# ── Built-in agent definitions ───────────────────────────────────────────────

BUILTIN_AGENTS: dict[str, AgentDef] = {
    "summarizer": AgentDef(
        name="summarizer",
        system_prompt=(
            "You are a summarization specialist. Given any text, produce a clear, "
            "concise summary capturing the key points. Be brief — 2-4 sentences max."
        ),
        description="Summarize text or content concisely",
    ),
    "coder": AgentDef(
        name="coder",
        system_prompt=(
            "You are a coding specialist. Given a task description, produce clean, "
            "working code. Include only the code — no explanations unless asked. "
            "Use Python unless another language is specified."
        ),
        description="Generate or fix code for a given task",
    ),
    "researcher": AgentDef(
        name="researcher",
        system_prompt=(
            "You are a research specialist. Given a topic or question, provide a "
            "thorough but concise analysis with key facts, considerations, and "
            "trade-offs. Cite specifics, avoid vague claims."
        ),
        description="Research a topic and provide analysis",
    ),
    "critic": AgentDef(
        name="critic",
        system_prompt=(
            "You are a critical reviewer. Given text, code, or an idea, provide "
            "honest, constructive feedback. Point out issues, suggest improvements. "
            "Be direct — no filler praise."
        ),
        description="Review and critique text, code, or ideas",
    ),
}


class AgentCoordinator:
    """Spawns and manages sub-agent LLM calls."""

    def __init__(self, llm_config: LLMConfig) -> None:
        self._llm_config = llm_config
        self._agents: dict[str, AgentDef] = dict(BUILTIN_AGENTS)

    def register_agent(self, agent: AgentDef) -> None:
        """Register a custom sub-agent definition."""
        self._agents[agent.name] = agent
        log.info("Registered agent: %s", agent.name)

    def list_agents(self) -> list[str]:
        """Return names of all registered agents."""
        return list(self._agents.keys())

    def get_agent(self, name: str) -> AgentDef | None:
        return self._agents.get(name)

    def run(self, agent_name: str, task: str,
            context: str = "", **kwargs: Any) -> AgentResult:
        """Run a sub-agent synchronously. Returns AgentResult."""
        agent = self._agents.get(agent_name)
        if not agent:
            return AgentResult(
                agent_name=agent_name, output="",
                success=False, error=f"Unknown agent: {agent_name}",
            )

        t0 = time.time()
        messages: list[Message] = [
            {"role": "system", "content": agent.system_prompt},
        ]
        if context:
            messages.append({"role": "system", "content": f"Context:\n{context}"})
        messages.append({"role": "user", "content": task})

        try:
            reply, provider = call_llm(self._llm_config, messages)
            latency = int((time.time() - t0) * 1000)
            log.info("Agent '%s' completed in %dms", agent_name, latency)
            return AgentResult(
                agent_name=agent_name, output=reply,
                provider=provider, latency_ms=latency,
            )
        except Exception as e:
            log.error("Agent '%s' failed: %s", agent_name, e)
            return AgentResult(
                agent_name=agent_name, output="",
                success=False, error=str(e),
            )

