"""
prototype-x1 — main entry point
================================
Run:  python3 main.py
      python3 main.py --model llama3.2
      python3 main.py --cloud anthropic   (skip Ollama, go straight to cloud)

Type 'quit' or 'exit' to stop.
Type '/skills'  to list loaded skills.
Type '/memory'  to show session memory.
Type '/reset'   to clear session memory.
Type '/reload'  to hot-reload skills.
"""
from __future__ import annotations
import argparse
import logging
import os
import re
import sys
import threading
import time
from pathlib import Path

# Words-per-minute that pyttsx3 is set to — terminal output matches this rate
SPEECH_WPM = 150

# ── Local TTS ─────────────────────────────────────────────────────────────────
# macOS: use the built-in `say` command via subprocess.
#   • Works from ANY thread — no main-thread restriction like pyttsx3/NSSpeechSynthesizer
#   • Same voice, same engine under the hood, zero setup
# Raspberry Pi / Linux: fall back to pyttsx3 (espeak driver, main-thread safe there)
import shutil as _shutil
import subprocess as _subprocess

_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

# ── Interrupt state ───────────────────────────────────────────────────────────
# Ctrl+C while ARIA speaks: KeyboardInterrupt is caught in the main thread,
# which sets this event and terminates the active `say` subprocess.
# No raw-mode stdin hacks — those corrupt the terminal.
_stop_speaking: threading.Event = threading.Event()
_active_say_proc: "_subprocess.Popen[bytes] | None" = None
_proc_lock: threading.Lock = threading.Lock()


def _kill_active_say() -> None:
    """Terminate the currently running `say` subprocess, if any."""
    global _active_say_proc
    with _proc_lock:
        if _active_say_proc and _active_say_proc.poll() is None:
            _active_say_proc.terminate()


if _shutil.which("say"):
    # macOS — subprocess `say`, works from any thread
    def speak(text: str, stop: "threading.Event | None" = None) -> None:
        global _active_say_proc
        clean = re.sub(r"[`*_#>\[\]]+", "", text).strip()
        if not clean:
            return
        for sentence in _SENTENCE_RE.split(clean):
            sentence = sentence.strip()
            if not sentence:
                continue
            if stop and stop.is_set():
                return
            proc = _subprocess.Popen(["say", "-r", "150", sentence])
            with _proc_lock:
                _active_say_proc = proc
            proc.wait()
            with _proc_lock:
                _active_say_proc = None
            if stop and stop.is_set():
                return
            time.sleep(0.15)

    _TTS_AVAILABLE = True

else:
    # Pi / Linux fallback
    try:
        import pyttsx3 as _pyttsx3
        _TTS_ENGINE = _pyttsx3.init()
        _TTS_ENGINE.setProperty("rate", 150)
        _TTS_ENGINE.setProperty("volume", 0.9)

        def speak(text: str, stop: "threading.Event | None" = None) -> None:  # type: ignore[misc]
            clean = re.sub(r"[`*_#>\[\]]+", "", text).strip()
            if not clean:
                return
            for sentence in _SENTENCE_RE.split(clean):
                sentence = sentence.strip()
                if sentence and not (stop and stop.is_set()):
                    _TTS_ENGINE.say(sentence)
                    _TTS_ENGINE.runAndWait()
                    time.sleep(0.15)

        _TTS_AVAILABLE = True

    except Exception:
        _TTS_AVAILABLE = False
        def speak(text: str, stop: "threading.Event | None" = None) -> None:  # type: ignore[misc]
            pass


import random as _random

# Short filler phrases — spoken before the real response so she feels present,
# not like a text dump.  Rotates randomly; never the same one twice in a row.
_FILLERS = [
    "Hey, just thinking.",
    "Mm.",
    "Hold on.",
    "Right.",
    "Let me think on that.",
    "Yeah.",
    "Okay.",
    "Hmm.",
]
_last_filler: str = ""


_SPEECH_STRIP_RE = re.compile(
    r"(?m)"
    r"(^##\s.*$"                          # ## Section headers
    r"|^BCS[:\s].*$"                       # BCS: ... lines
    r"|^Body Completion.*$"                # Body Completion Score lines
    r"|^Hardware State.*$"                 # Hardware State lines
    r"|^Webcam:.*$"                        # Webcam: status
    r"|^Motors:.*$"                        # Motors: status
    r"|^GPIO:.*$"                          # GPIO: status
    r"|^Relevant memory.*$"               # Relevant memory prepended lines
    r"|\*[^*]+\*"                          # *stage directions*
    r"|\[[^\]]*hum[^\]]*\]"               # [processing units hum] etc
    r")"
)


def _strip_for_speech(text: str) -> str:
    """Remove metadata / stage-direction lines before sending to TTS.

    The full text still streams to the terminal — Neokode can see it.
    ARIA only *speaks* the conversational part.
    """
    cleaned = _SPEECH_STRIP_RE.sub("", text)
    # Collapse multiple blank lines left behind
    cleaned = re.sub(r"\n{2,}", "\n", cleaned).strip()
    return cleaned


def breath_then_speak(text: str, stop: "threading.Event | None" = None) -> None:
    """
    Human-paced pre-speech ritual:
      1. Two-second pause — interruptible, 100ms ticks so spacebar cuts in fast.
      2. A random filler phrase out loud.
      3. Then the actual response.
    """
    global _last_filler
    # Interruptible breath — 20 × 100ms = 2s, but exits immediately on stop
    for _ in range(20):
        if stop and stop.is_set():
            return
        time.sleep(0.1)

    pool = [f for f in _FILLERS if f != _last_filler]
    filler = _random.choice(pool)
    _last_filler = filler
    speak(filler, stop)
    speak(text, stop)


def trickle_print(text: str, wpm: int = SPEECH_WPM,
                  stop: "threading.Event | None" = None) -> None:
    """
    Print text word-by-word at speech pace so the terminal stays in sync
    with what ARIA is saying out loud.  Bails immediately if stop is set.
    """
    clean = re.sub(r"[`*_#>\[\]]+", "", text).strip()
    words = clean.split()
    if not words:
        return
    delay = 60.0 / wpm
    for i, word in enumerate(words):
        if stop and stop.is_set():
            print()     # clean newline on interrupt
            return
        suffix = " " if i < len(words) - 1 else ""
        print(word + suffix, end="", flush=True)
        time.sleep(delay)
    print()             # final newline

# ── Rich for pretty output (optional) ─────────────────────────────────────────
try:
    from rich.console import Console
    from rich.markdown import Markdown
    console = Console()
    def print_aria(text: str) -> None:
        console.print(f"[bold cyan]ARIA:[/bold cyan]", end=" ")
        try:
            console.print(Markdown(text))
        except Exception:
            console.print(text)
    def print_info(text: str) -> None:
        console.print(f"[dim]{text}[/dim]")
    def print_err(text: str) -> None:
        console.print(f"[bold red]{text}[/bold red]")
except ImportError:
    def print_aria(text: str) -> None:
        print(f"ARIA: {text}")
    def print_info(text: str) -> None:
        print(f"  {text}")
    def print_err(text: str) -> None:
        print(f"ERROR: {text}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="prototype-x1 brain REPL")
    p.add_argument("--model", default=None, help="Ollama model override (e.g. llama3.2)")
    p.add_argument("--cloud", choices=["anthropic", "openai"], default=None,
                   help="Skip Ollama, use cloud provider")
    p.add_argument("--debug", action="store_true", help="Enable debug logging")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Logging ────────────────────────────────────────────────────────────────
    level = logging.DEBUG if args.debug else logging.WARNING
    logging.basicConfig(level=level, format="%(name)s [%(levelname)s] %(message)s")

    # ── Load .env ──────────────────────────────────────────────────────────────
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
        except ImportError:
            pass

    # ── Brain init ─────────────────────────────────────────────────────────────
    sys.path.insert(0, str(Path(__file__).parent))
    from brain.config import BrainConfig, LLMConfig
    from brain.core import Brain

    cfg = BrainConfig()
    if args.model:
        cfg.llm.ollama_model = args.model
    if args.cloud:
        cfg.llm.cloud_fallback = args.cloud
        # Force cloud by pointing Ollama host to something unreachable
        cfg.llm.ollama_host = "http://localhost:0"
    if args.debug:
        cfg.debug = True

    print_info(f"Initialising ARIA brain...")
    try:
        brain = Brain(cfg)
    except Exception as e:
        print_err(f"Failed to init brain: {e}")
        sys.exit(1)

    print_info(f"Skills loaded: {', '.join(brain.skill_list)}")
    print_info(f"Model: {cfg.llm.ollama_model} | fallback: {cfg.llm.cloud_fallback}")
    tts_status = "local TTS ready" if _TTS_AVAILABLE else "TTS unavailable"
    print_info(f"Voice: {tts_status}")
    print_info("Type 'quit' to exit. Ctrl+C to cut speech. /skills /memory /reset /reload")
    print()

    # ── Startup voice — verify engine, check camera, greet ───────────────────
    if _TTS_AVAILABLE:
        print("voice engine ready")
        speak("Hey ARIA, testing voice one two three.")

        # Probe webcam — silent, no crash if cv2 missing
        _cam_live = False
        try:
            from vision.camera import capture_webcam
            _cam_live = capture_webcam() is not None
        except Exception:
            pass

        if _cam_live:
            speak("Hello. I've got eyes.")
        else:
            speak("Hello.")

    # ── REPL ───────────────────────────────────────────────────────────────────
    skills_dir = Path(__file__).parent / "skills"

    while True:
        try:
            user_input = input("Neokode: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            print_info("Shutting down.")
            break

        if not user_input:
            continue

        # ── Meta commands ──────────────────────────────────────────────────────
        if user_input.lower() in ("quit", "exit"):
            print_info("Goodbye.")
            break

        if user_input == "/skills":
            print_info("Loaded skills: " + ", ".join(brain.skill_list))
            continue

        if user_input == "/memory":
            msgs = brain.memory.as_messages()
            if not msgs:
                print_info("Memory empty.")
            for m in msgs:
                print_info(f"  [{m['role']}] {m['content'][:120]}")
            continue

        if user_input == "/reset":
            brain.reset()
            print_info("Session memory cleared.")
            continue

        if user_input == "/reload":
            from brain import skills as skill_registry
            reloaded = skill_registry.hot_reload(skills_dir)
            print_info(f"Reloaded: {reloaded or 'nothing changed'}")
            continue

        # ── Brain turn ────────────────────────────────────────────────────────
        # Tokens are buffered silently while the LLM generates.
        # On `done`, speak() fires in a background thread and trickle_print()
        # releases words at SPEECH_WPM so the terminal stays in sync with voice.
        try:
            skill_calls = []
            token_buf: list[str] = []
            response = None

            print("ARIA: ", end="", flush=True)

            for event, data in brain.stream(user_input, speaker="Neokode"):
                if event == "thinking":
                    pass  # label already printed above

                elif event == "token":
                    token_buf.append(data)   # collect — don't blast yet

                elif event == "action":
                    name = data.get("params", {}).get("name", "?")
                    result = data.get("result", "")
                    print()
                    print_info(f"  → [{name}] {result}")
                    skill_calls.append(name)
                    # Ready to start printing again after the action line
                    if token_buf:
                        print("ARIA: ", end="", flush=True)

                elif event == "done":
                    response = data
                    full_text = "".join(token_buf)
                    if full_text:
                        # ── Speak + trickle — Ctrl+C interrupts both ──────
                        _stop_speaking.clear()
                        tts_thread: threading.Thread | None = None
                        if _TTS_AVAILABLE:
                            speech_text = _strip_for_speech(full_text)
                            tts_thread = threading.Thread(
                                target=breath_then_speak,
                                args=(speech_text, _stop_speaking),
                                daemon=True,
                            )
                            tts_thread.start()
                        try:
                            # Interruptible breath then trickle in foreground
                            for _ in range(20):
                                if _stop_speaking.is_set():
                                    break
                                time.sleep(0.1)
                            trickle_print(full_text, stop=_stop_speaking)
                        except KeyboardInterrupt:
                            # Ctrl+C — cut the voice, return to YOU:
                            _stop_speaking.set()
                            _kill_active_say()
                            print()  # clean newline
                        finally:
                            _stop_speaking.set()
                            if tts_thread is not None:
                                tts_thread.join(timeout=2)
                    else:
                        print()

        except Exception as e:
            print()
            print_err(f"Brain error: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            continue

        if skill_calls:
            print_info(f"  Skills: {skill_calls}")
        if response:
            print_info(f"  [{response.provider} | {response.latency_ms}ms]")
        print()


if __name__ == "__main__":
    main()
