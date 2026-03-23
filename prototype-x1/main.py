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

if _shutil.which("say"):
    # macOS path — `say -r 150` matches our SPEECH_WPM target
    def speak(text: str) -> None:
        clean = re.sub(r"[`*_#>\[\]]+", "", text).strip()
        if not clean:
            return
        for sentence in _SENTENCE_RE.split(clean):
            sentence = sentence.strip()
            if sentence:
                _subprocess.run(["say", "-r", "150", sentence], check=False)
                time.sleep(0.15)

    _TTS_AVAILABLE = True

else:
    # Pi / Linux fallback — pyttsx3 is main-thread safe on those platforms
    try:
        import pyttsx3 as _pyttsx3
        _TTS_ENGINE = _pyttsx3.init()
        _TTS_ENGINE.setProperty("rate", 150)
        _TTS_ENGINE.setProperty("volume", 0.9)

        def speak(text: str) -> None:
            clean = re.sub(r"[`*_#>\[\]]+", "", text).strip()
            if not clean:
                return
            for sentence in _SENTENCE_RE.split(clean):
                sentence = sentence.strip()
                if sentence:
                    _TTS_ENGINE.say(sentence)
                    _TTS_ENGINE.runAndWait()
                    time.sleep(0.15)

        _TTS_AVAILABLE = True

    except Exception:
        _TTS_AVAILABLE = False
        def speak(text: str) -> None:  # type: ignore[misc]
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


def breath_then_speak(text: str) -> None:
    """
    Human-paced pre-speech ritual:
      1. Two-second pause — she's processing, not blasting.
      2. A random filler phrase out loud — she's here, not a text dump.
      3. Then the actual response.
    """
    global _last_filler
    time.sleep(2)                                   # the breath

    # Pick a filler that isn't the same as last time
    pool = [f for f in _FILLERS if f != _last_filler]
    filler = _random.choice(pool)
    _last_filler = filler
    speak(filler)

    speak(text)                                     # the real thing


def trickle_print(text: str, wpm: int = SPEECH_WPM) -> None:
    """
    Print text word-by-word at speech pace so the terminal stays in sync
    with what ARIA is saying out loud.  Humans read ~150 WPM; so does she.
    """
    clean = re.sub(r"[`*_#>\[\]]+", "", text).strip()
    words = clean.split()
    if not words:
        return
    delay = 60.0 / wpm          # seconds per word at target WPM
    for i, word in enumerate(words):
        suffix = " " if i < len(words) - 1 else ""
        print(word + suffix, end="", flush=True)
        time.sleep(delay)
    print()                     # final newline

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
    print_info("Type 'quit' to exit. /skills /memory /reset /reload for controls.")
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
            user_input = input("YOU: ").strip()
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
                        # breath_then_speak: 2s pause → filler → response, in background
                        # trickle_print releases words at SPEECH_WPM in foreground
                        tts_thread: threading.Thread | None = None
                        if _TTS_AVAILABLE:
                            tts_thread = threading.Thread(
                                target=breath_then_speak, args=(full_text,), daemon=True
                            )
                            tts_thread.start()
                        time.sleep(2)           # foreground waits the same breath
                        trickle_print(full_text)
                        if tts_thread is not None:
                            tts_thread.join()   # wait for voice before showing YOU:
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
