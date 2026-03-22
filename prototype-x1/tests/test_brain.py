"""
Tests for prototype-x1 brain: config, skills, memory, core (mock LLM).
Run: cd ~/Desktop/prototypes/prototype-x1 && pytest tests/ -v
"""
from __future__ import annotations
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Make sure the package root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

# ── Config ─────────────────────────────────────────────────────────────────────

def test_brain_config_defaults():
    from brain.config import BrainConfig
    cfg = BrainConfig()
    assert cfg.name == "ARIA"
    assert cfg.llm.ollama_model  # non-empty
    assert cfg.llm.context_window > 0
    assert cfg.memory.top_k > 0


def test_llm_config_env_override(monkeypatch):
    monkeypatch.setenv("OLLAMA_MODEL", "my-custom-model")
    # Re-import to pick up new env
    import importlib
    import brain.config as cfg_mod
    importlib.reload(cfg_mod)
    from brain.config import LLMConfig
    llm = LLMConfig()
    assert llm.ollama_model == "my-custom-model"


# ── Skills ─────────────────────────────────────────────────────────────────────

def test_register_and_call():
    from brain import skills
    skills._registry.clear()
    skills.register("greet", lambda name="world": f"Hello, {name}!")
    assert "greet" in skills.list_skills()
    assert skills.call("greet", name="X1") == "Hello, X1!"


def test_unknown_skill_returns_error():
    from brain import skills
    result = skills.call("nonexistent_skill")
    assert "[SKILL ERROR]" in result


def test_builtin_python_skill():
    from brain import skills
    skills.register_builtins()
    result = skills.call("python", code="print(6 * 7)")
    assert result == "42"


def test_builtin_python_skill_error():
    from brain import skills
    skills.register_builtins()
    result = skills.call("python", code="raise ValueError('boom')")
    assert "[PYTHON ERROR]" in result


def test_builtin_list_dir(tmp_path):
    from brain import skills
    skills.register_builtins()
    (tmp_path / "foo.txt").touch()
    result = skills.call("list_dir", path=str(tmp_path))
    assert "foo.txt" in result


def test_builtin_read_write_file(tmp_path):
    from brain import skills
    skills.register_builtins()
    target = str(tmp_path / "test.txt")
    skills.call("write_file", path=target, content="hello x1")
    result = skills.call("read_file", path=target)
    assert result == "hello x1"


def test_builtin_read_file_missing():
    from brain import skills
    skills.register_builtins()
    result = skills.call("read_file", path="/does/not/exist/ever.txt")
    assert "[FILE ERROR]" in result


def test_builtin_shell():
    from brain import skills
    skills.register_builtins()
    result = skills.call("shell", command="echo hello_from_x1")
    assert "hello_from_x1" in result


# ── Memory ─────────────────────────────────────────────────────────────────────

def test_memory_short_term():
    from brain.config import MemoryConfig
    from brain.memory import EpisodicMemory
    cfg = MemoryConfig(enabled=False)  # no ChromaDB for unit tests
    mem = EpisodicMemory(cfg)
    mem.add("user", "what time is it")
    mem.add("assistant", "I don't have a clock skill yet")
    assert mem.session_length == 2
    msgs = mem.as_messages()
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"


def test_memory_clear():
    from brain.config import MemoryConfig
    from brain.memory import EpisodicMemory
    cfg = MemoryConfig(enabled=False)
    mem = EpisodicMemory(cfg)
    mem.add("user", "hello")
    mem.clear_session()
    assert mem.session_length == 0


def test_memory_recall_no_chroma():
    from brain.config import MemoryConfig
    from brain.memory import EpisodicMemory
    cfg = MemoryConfig(enabled=False)
    mem = EpisodicMemory(cfg)
    # recall should return empty list gracefully when ChromaDB is off
    result = mem.recall("anything")
    assert result == []


def test_memory_recent_cap():
    from brain.config import MemoryConfig
    from brain.memory import EpisodicMemory
    cfg = MemoryConfig(enabled=False)
    mem = EpisodicMemory(cfg)
    for i in range(30):
        mem.add("user", f"message {i}")
    assert len(mem.recent(10)) == 10
    assert len(mem.recent(50)) == 30


# ── Core (mocked LLM) ──────────────────────────────────────────────────────────

def _make_brain(mock_reply: str = "Hello from mocked LLM"):
    """Create a Brain with all LLM calls mocked out."""
    from brain.config import BrainConfig, MemoryConfig
    cfg = BrainConfig()
    cfg.memory = MemoryConfig(enabled=False)  # no ChromaDB
    with patch("brain.core.call_llm", return_value=(mock_reply, "mock")):
        from brain.core import Brain
        brain = Brain(cfg)
    return brain


def test_brain_process_basic():
    from brain.config import BrainConfig, MemoryConfig
    from brain.core import Brain, BrainResponse

    cfg = BrainConfig()
    cfg.memory = MemoryConfig(enabled=False)

    with patch("brain.core.call_llm", return_value=("I am ARIA", "mock")):
        brain = Brain(cfg)
        response = brain.process("who are you")

    assert isinstance(response, BrainResponse)
    assert response.text == "I am ARIA"
    assert response.provider == "mock"
    assert response.latency_ms >= 0
    assert response.skill_calls == []


def test_brain_persists_to_memory():
    from brain.config import BrainConfig, MemoryConfig
    from brain.core import Brain

    cfg = BrainConfig()
    cfg.memory = MemoryConfig(enabled=False)

    with patch("brain.core.call_llm", return_value=("reply text", "mock")):
        brain = Brain(cfg)
        brain.process("remember this")

    msgs = brain.memory.as_messages()
    assert any(m["role"] == "user" and "remember this" in m["content"] for m in msgs)
    assert any(m["role"] == "assistant" and "reply text" in m["content"] for m in msgs)


def test_brain_skill_extraction():
    from brain.config import BrainConfig, MemoryConfig
    from brain.core import Brain

    cfg = BrainConfig()
    cfg.memory = MemoryConfig(enabled=False)

    skill_reply = '''Sure, let me run that.
```skill
{"name": "python", "args": {"code": "print(1 + 1)"}}
```
Done.'''

    call_returns = [(skill_reply, "mock"), ("The answer is 2", "mock")]

    with patch("brain.core.call_llm", side_effect=call_returns):
        brain = Brain(cfg)
        response = brain.process("what is 1+1")

    assert len(response.skill_calls) == 1
    assert response.skill_calls[0]["name"] == "python"
    assert any("2" in r for r in response.skill_results)


def test_brain_reset():
    from brain.config import BrainConfig, MemoryConfig
    from brain.core import Brain

    cfg = BrainConfig()
    cfg.memory = MemoryConfig(enabled=False)

    with patch("brain.core.call_llm", return_value=("ok", "mock")):
        brain = Brain(cfg)
        brain.process("hello")
        assert brain.memory.session_length > 0
        brain.reset()
        assert brain.memory.session_length == 0
