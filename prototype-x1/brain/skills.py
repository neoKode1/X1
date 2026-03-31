"""
Skill registry for ARIA's brain.
Provides register/call/list interface + built-in skills (python, shell, file ops).

This module wraps brain.tools.ToolRegistry with a simpler "skills" API
that the test suite and brain core both use.
"""
from __future__ import annotations

import io
import logging
import os
import subprocess
import sys
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

from .tools import ToolRegistry

log = logging.getLogger("x1.skills")

# ── Module-level registry ────────────────────────────────────────────────────
_registry = ToolRegistry()


def register(name: str, fn, description: str | None = None, **kw) -> None:
    """Register a skill (tool) by name."""
    _registry.register(name, fn, description=description, **kw)


def call(skill_name: str, **kwargs) -> str:
    """Call a registered skill. Returns error string if not found."""
    result = _registry.call(skill_name, **kwargs)
    # Normalise error prefix for backward compat
    if result.startswith("[TOOL ERROR]"):
        result = result.replace("[TOOL ERROR]", "[SKILL ERROR]", 1)
    return result


def list_skills() -> list[str]:
    """Return names of all registered skills."""
    return _registry.list_tools()


# ── Built-in skills ──────────────────────────────────────────────────────────

def _builtin_python(code: str) -> str:
    """Execute Python code and return stdout."""
    buf = io.StringIO()
    try:
        with redirect_stdout(buf), redirect_stderr(buf):
            exec(code, {"__builtins__": __builtins__})  # noqa: S102
        output = buf.getvalue().strip()
        return output if output else "(no output)"
    except Exception as e:
        return f"[PYTHON ERROR] {e}"


def _builtin_shell(command: str) -> str:
    """Run a shell command and return combined output."""
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=30,
        )
        out = (result.stdout + result.stderr).strip()
        return out if out else "(no output)"
    except subprocess.TimeoutExpired:
        return "[SHELL ERROR] Command timed out (30s)"
    except Exception as e:
        return f"[SHELL ERROR] {e}"


def _builtin_list_dir(path: str = ".") -> str:
    """List files in a directory."""
    try:
        entries = sorted(os.listdir(path))
        return "\n".join(entries) if entries else "(empty directory)"
    except Exception as e:
        return f"[DIR ERROR] {e}"


def _builtin_read_file(path: str) -> str:
    """Read a file and return its content."""
    try:
        return Path(path).read_text()
    except Exception as e:
        return f"[FILE ERROR] {e}"


def _builtin_write_file(path: str, content: str) -> str:
    """Write content to a file."""
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(content)
        return f"Wrote {len(content)} chars to {path}"
    except Exception as e:
        return f"[FILE ERROR] {e}"


def register_builtins() -> None:
    """Register all built-in skills (python, shell, file ops)."""
    builtins = {
        "python": _builtin_python,
        "shell": _builtin_shell,
        "list_dir": _builtin_list_dir,
        "read_file": _builtin_read_file,
        "write_file": _builtin_write_file,
    }
    for name, fn in builtins.items():
        if name not in _registry:
            _registry.register(name, fn)

