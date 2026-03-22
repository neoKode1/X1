"""
Skill registry with hot-reload.
Skills are plain Python callables registered by name.
The brain can discover and invoke them at runtime.
"""
from __future__ import annotations
import importlib
import importlib.util
import logging
import traceback
from pathlib import Path
from typing import Callable, Any

log = logging.getLogger("x1.skills")

SkillFn = Callable[..., str]  # skills return a string result

_registry: dict[str, SkillFn] = {}
_skill_modules: dict[str, Any] = {}   # name -> module, for hot-reload


def register(name: str, fn: SkillFn, *, override: bool = False) -> None:
    if name in _registry and not override:
        log.debug("Skill %r already registered, skipping", name)
        return
    _registry[name] = fn
    log.info("Skill registered: %s", name)


def call(skill_name: str, **kwargs: Any) -> str:
    if skill_name not in _registry:
        return f"[SKILL ERROR] Unknown skill: {skill_name!r}. Available: {list_skills()}"
    try:
        return _registry[skill_name](**kwargs)
    except Exception:
        err = traceback.format_exc()
        log.error("Skill %r raised:\n%s", skill_name, err)
        return f"[SKILL ERROR] {skill_name} failed:\n{err}"


def list_skills() -> list[str]:
    return sorted(_registry.keys())


def load_skills_dir(skills_dir: Path) -> None:
    """Dynamically load all .py files in skills_dir as skill modules."""
    for path in sorted(skills_dir.glob("*.py")):
        if path.name.startswith("_"):
            continue
        _load_skill_file(path)


def _load_skill_file(path: Path) -> None:
    name = path.stem
    spec = importlib.util.spec_from_file_location(f"x1.skills.{name}", path)
    if spec is None or spec.loader is None:
        return
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        _skill_modules[name] = mod
        log.info("Loaded skill module: %s", name)
    except Exception:
        log.error("Failed to load skill %s:\n%s", name, traceback.format_exc())


def hot_reload(skills_dir: Path) -> list[str]:
    """Re-execute changed skill modules. Returns list of reloaded names."""
    reloaded = []
    for name, mod in list(_skill_modules.items()):
        path = Path(mod.__file__)  # type: ignore[arg-type]
        if path.exists():
            _load_skill_file(path)
            reloaded.append(name)
    return reloaded


# ── Built-in skills ────────────────────────────────────────────────────────────

def _skill_read_file(path: str) -> str:
    p = Path(path).expanduser()
    if not p.exists():
        return f"[FILE ERROR] Not found: {path}"
    return p.read_text(errors="replace")[:8000]


def _skill_write_file(path: str, content: str) -> str:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return f"Written {len(content)} chars to {path}"


def _skill_list_dir(path: str = ".") -> str:
    p = Path(path).expanduser()
    if not p.is_dir():
        return f"[DIR ERROR] Not a directory: {path}"
    entries = sorted(p.iterdir(), key=lambda x: (x.is_file(), x.name))
    return "\n".join(("[DIR]  " if e.is_dir() else "[FILE] ") + e.name for e in entries)


def _skill_shell(command: str) -> str:
    import subprocess
    import shlex
    try:
        result = subprocess.run(
            shlex.split(command), capture_output=True, text=True, timeout=15)
        out = (result.stdout + result.stderr).strip()
        return out[:4000] or "(no output)"
    except Exception as e:
        return f"[SHELL ERROR] {e}"


def _skill_screengrab() -> str:
    try:
        import PIL.ImageGrab
        import base64
        import io
        img = PIL.ImageGrab.grab()
        img.thumbnail((640, 480))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=70)
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"[SCREEN CAPTURED] base64 JPEG ({len(b64)} chars)"
    except Exception as e:
        return f"[SCREENGRAB ERROR] {e}"


def _skill_python(code: str) -> str:
    import io
    import contextlib
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            exec(code, {"__name__": "__x1_exec__"})  # noqa: S102
        return buf.getvalue().strip() or "(executed, no output)"
    except Exception as e:
        return f"[PYTHON ERROR] {e}\n{buf.getvalue()}"


def register_builtins() -> None:
    register("read_file",  _skill_read_file)
    register("write_file", _skill_write_file)
    register("list_dir",   _skill_list_dir)
    register("shell",      _skill_shell)
    register("screengrab", _skill_screengrab)
    register("python",     _skill_python)
