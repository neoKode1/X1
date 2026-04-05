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
    # Primary: Anthropic Claude (vision + reasoning)
    primary: str = field(default_factory=lambda: os.getenv("LLM_PRIMARY", "anthropic"))
    anthropic_api_key: str = field(default_factory=lambda: os.getenv("ANTHROPIC_API_KEY", ""))
    anthropic_model: str = field(default_factory=lambda: os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"))
    # Fallback: Ollama (local)
    ollama_host: str = field(default_factory=lambda: os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "llama3.2"))
    # Optional: OpenAI
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_model: str = field(default_factory=lambda: os.getenv("OPENAI_MODEL", "gpt-4o-mini"))
    cloud_fallback: str = field(default_factory=lambda: os.getenv("CLOUD_FALLBACK", "ollama"))
    temperature: float = 0.4
    max_tokens: int = 1024
    num_ctx: int = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
    context_window: int = 6


@dataclass
class MemoryConfig:
    enabled: bool = field(default_factory=lambda: os.getenv("MEMORY_ENABLED", "true").lower() != "false")
    persist_dir: str = field(default_factory=lambda: os.getenv(
        "MEMORY_DIR", str(Path(__file__).parent.parent / "memory" / "chroma")))
    db_path: str = field(default_factory=lambda: os.getenv(
        "CONVERSATION_DB", str(Path(__file__).parent.parent / "memory" / "conversations.db")))
    collection: str = "x1_episodic"
    top_k: int = 5


@dataclass
class BrainConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    debug: bool = field(default_factory=lambda: os.getenv("DEBUG", "false").lower() == "true")
    name: str = "ARIA"
