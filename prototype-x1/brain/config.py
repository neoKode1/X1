"""prototype-x1 brain configuration. Env vars override defaults."""
from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent.parent / ".env")
except ImportError:
    pass


@dataclass
class LLMConfig:
    ollama_host: str = field(default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.2"))
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    anthropic_model: str = field(default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-3-5-haiku-20241022"))
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    cloud_fallback: str = field(default_factory=lambda: os.getenv("CLOUD_FALLBACK", "anthropic"))
    temperature: float = 0.7
    max_tokens: int = 300
    context_window: int = 20


@dataclass
class MemoryConfig:
    enabled: bool = field(default_factory=lambda: os.getenv("MEMORY_ENABLED", "true").lower() != "false")
    persist_dir: str = field(default_factory=lambda: os.getenv(
        "MEMORY_DIR", str(Path(__file__).parent.parent / "memory" / "chroma")))
    collection: str = "x1_episodic"
    top_k: int = 5


@dataclass
class BrainConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")
    name: str = "ARIA"
