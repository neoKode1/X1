"""
prototype-x1 — main entry point
================================
Run:  python3 main.py
      python3 main.py --listen            (always-on mic — 3-way conversation)
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
import queue
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
    _SAY_VOICE = "Flo"
    _SAY_RATE = "210"  # WPM — 150 was sluggish, 210 feels natural-fast

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
            proc = _subprocess.Popen(["say", "-v", _SAY_VOICE, "-r", _SAY_RATE, sentence])
            with _proc_lock:
                _active_say_proc = proc
            proc.wait()
            with _proc_lock:
                _active_say_proc = None
            if stop and stop.is_set():
                return
            time.sleep(0.08)  # tighter gap between sentences

    def speak_sentence(sentence: str, stop: "threading.Event | None" = None) -> None:
        """Speak a single sentence immediately. Used for streaming TTS."""
        global _active_say_proc
        sentence = re.sub(r"[`*_#>\[\]]+", "", sentence).strip()
        if not sentence or (stop and stop.is_set()):
            return
        proc = _subprocess.Popen(["say", "-v", _SAY_VOICE, "-r", _SAY_RATE, sentence])
        with _proc_lock:
            _active_say_proc = proc
        proc.wait()
        with _proc_lock:
            _active_say_proc = None

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

        def speak_sentence(sentence: str, stop: "threading.Event | None" = None) -> None:  # type: ignore[misc]
            sentence = re.sub(r"[`*_#>\[\]]+", "", sentence).strip()
            if sentence and not (stop and stop.is_set()):
                _TTS_ENGINE.say(sentence)
                _TTS_ENGINE.runAndWait()

        _TTS_AVAILABLE = True

    except Exception:
        _TTS_AVAILABLE = False
        def speak(text: str, stop: "threading.Event | None" = None) -> None:  # type: ignore[misc]
            pass
        def speak_sentence(sentence: str, stop: "threading.Event | None" = None) -> None:  # type: ignore[misc]
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


# Phrases llama3.2 outputs despite being told not to — strip at output level
_BANNED_PHRASES = [
    "no response is required",
    "screengrab now",
    "action completed",
    "the speak skill has completed",
    "your feedback is acknowledged",
    "relevant memory prepended",
]

def _strip_banned(text: str) -> str:
    """Remove banned narration phrases (case-insensitive)."""
    for phrase in _BANNED_PHRASES:
        text = re.sub(re.escape(phrase), "", text, flags=re.IGNORECASE)
    return text.strip()

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
    r"|\[[^\]]+\]"                         # [any bracket content] emotes
    r"|\([^)]{3,80}\)"                     # (parenthetical asides up to 80 chars)
    r"|(?:^|\s)(?:silence|neural hum|hums?|chuckles? quietly|chuckles?|sighs?|pauses?|whispers?|exhales?|inhales?|laughs?|smiles?|whirs?|beeps?|clicks?|buzzes?|hisses?|crackles?)\s*[,.]?"  # bare action words
    r")"
)


def _strip_for_speech(text: str) -> str:
    """Remove metadata / stage-direction lines before sending to TTS.

    The full text still streams to the terminal — Neokode can see it.
    ARIA only *speaks* the conversational part.
    """
    cleaned = _SPEECH_STRIP_RE.sub("", text)
    cleaned = _EMOJI_STRIP_RE.sub("", cleaned)
    # Collapse multiple blank lines left behind
    cleaned = re.sub(r"\n{2,}", "\n", cleaned).strip()
    return cleaned


def breath_then_speak(text: str, stop: "threading.Event | None" = None) -> None:
    """Speak immediately — no delays, no filler."""
    speak(text, stop)


_SKILL_BLOCK_STRIP_RE = re.compile(r"```skill.*?```", re.DOTALL)
# Emoji / pictograph ranges — terminal can't render them; UI will handle later
_EMOJI_STRIP_RE = re.compile(
    u"[\U0001F300-\U0001FAFF"   # misc symbols, pictographs, emoticons, transport
    u"\U00002600-\U000027BF"    # misc symbols, dingbats
    u"\U0000FE00-\U0000FE0F"    # variation selectors
    u"\U00002300-\U000023FF"    # misc technical
    u"]+", re.UNICODE
)


def _strip_bare_json_blocks(text: str) -> str:
    """Remove bare JSON skill-call objects from text, handling nested braces."""
    result = []
    i = 0
    while i < len(text):
        if text[i] == '{':
            # Peek to see if this looks like a skill call
            snippet = text[i:i+40]
            if re.match(r'\{\s*"(?:skill|name|action)"\s*:', snippet):
                # Walk forward counting braces to find the balanced close
                depth = 0
                j = i
                while j < len(text):
                    if text[j] == '{':
                        depth += 1
                    elif text[j] == '}':
                        depth -= 1
                        if depth == 0:
                            i = j + 1  # skip the whole block
                            break
                    j += 1
                else:
                    i = j  # ran off the end — drop the rest
                continue
        result.append(text[i])
        i += 1
    return "".join(result)


def trickle_print(text: str, wpm: int = SPEECH_WPM,
                  stop: "threading.Event | None" = None) -> None:
    """
    Print text word-by-word at speech pace so the terminal stays in sync
    with what ARIA is saying out loud.  Bails immediately if stop is set.

    Skill call blocks (```skill ... ``` or bare JSON) are stripped — they
    already appear via the action event handler above.
    """
    # Strip skill JSON blocks — shown via action events, not trickle
    text = _SKILL_BLOCK_STRIP_RE.sub("", text)
    text = _strip_bare_json_blocks(text)
    # Strip emojis — terminal can't render them; UI will handle them later
    text = _EMOJI_STRIP_RE.sub("", text)
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
    p.add_argument("--listen", action="store_true",
                   help="Always-on mic mode — ARIA listens instead of waiting for keyboard input")
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

    # ── Input source: mic or keyboard ─────────────────────────────────────────
    _mic_listener = None

    def _input_stream():
        """Yield (speaker, text) tuples from keyboard or mic."""
        nonlocal _mic_listener
        if args.listen:
            from voice.listener import MicListener
            print_info("Mic mode active — listening. Ctrl+C to quit.")
            _mic_listener = MicListener(model_size="tiny")
            try:
                for text in _mic_listener.listen():
                    print(f"\n[mic] {text}")
                    yield "Neokode", text
            except KeyboardInterrupt:
                _mic_listener.stop()
        else:
            while True:
                try:
                    text = input("Neokode: ").strip()
                    if text:
                        yield "Neokode", text
                except (KeyboardInterrupt, EOFError):
                    return

    for speaker, user_input in _input_stream():
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
        # Tokens stream to terminal live. Sentences queue for TTS as they
        # complete mid-stream — she starts talking before the LLM finishes.
        # Ctrl+C interrupts speech immediately.
        try:
            skill_calls = []
            token_buf: list[str] = []
            response = None
            _brace_depth = 0       # tracks { } nesting to suppress skill JSON
            _in_fence = False      # tracks ```skill fences
            _sentence_buf: list[str] = []  # accumulates clean chars for TTS
            _tts_queue: "queue.Queue[str | None]" = queue.Queue()

            # TTS worker — reads sentences from queue, speaks them in order
            def _tts_worker():
                if _mic_listener is not None:
                    _mic_listener.muted.set()
                while True:
                    sentence = _tts_queue.get()
                    if sentence is None:  # poison pill
                        break
                    if _stop_speaking.is_set():
                        continue  # drain queue without speaking
                    speak_sentence(sentence, _stop_speaking)
                if _mic_listener is not None:
                    time.sleep(0.3)
                    _mic_listener.muted.clear()

            _stop_speaking.clear()
            tts_thread: threading.Thread | None = None
            if _TTS_AVAILABLE:
                tts_thread = threading.Thread(target=_tts_worker, daemon=True)
                tts_thread.start()

            print("ARIA: ", end="", flush=True)

            for event, data in brain.stream(user_input, speaker="Neokode"):
                if event == "thinking":
                    pass

                elif event == "token":
                    token_buf.append(data)
                    # ── Stream tokens live — suppress skill JSON ──
                    for ch in data:
                        # Detect ```skill fence openings
                        if ch == '`':
                            tail = "".join(token_buf)
                            if tail.rstrip().endswith("```skill") or tail.rstrip().endswith("```"):
                                _in_fence = not _in_fence
                            continue
                        if _in_fence:
                            continue
                        # Track bare JSON braces (skill calls without fences)
                        if ch == '{':
                            _brace_depth += 1
                        if _brace_depth > 0:
                            if ch == '}':
                                _brace_depth -= 1
                            continue  # suppress everything inside { }
                        # Print clean character immediately
                        print(ch, end="", flush=True)
                        # Accumulate for TTS — flush on sentence boundaries
                        _sentence_buf.append(ch)
                        if ch in '.!?' and len(_sentence_buf) > 3:
                            sentence = "".join(_sentence_buf).strip()
                            sentence = _strip_banned(sentence)
                            if sentence:
                                _tts_queue.put(sentence)
                            _sentence_buf.clear()

                elif event == "action":
                    name = data.get("params", {}).get("name", "?")
                    result = data.get("result", "")
                    print()
                    print_info(f"  → [{name}] {result}")
                    skill_calls.append(name)
                    print("ARIA: ", end="", flush=True)
                    _sentence_buf.clear()

                elif event == "done":
                    response = data
                    # Flush any remaining sentence fragment to TTS
                    leftover = "".join(_sentence_buf).strip()
                    leftover = _strip_banned(leftover)
                    if leftover:
                        _tts_queue.put(leftover)
                    _tts_queue.put(None)  # signal TTS worker to exit
                    full_text = "".join(token_buf)
                    if args.debug:
                        print(f"\n[DEBUG raw] {repr(full_text)}", flush=True)
                    print(flush=True)  # final newline
                    # Wait for TTS to finish (interruptible)
                    if tts_thread is not None:
                        try:
                            while tts_thread.is_alive():
                                tts_thread.join(timeout=0.2)
                        except KeyboardInterrupt:
                            _stop_speaking.set()
                            _kill_active_say()
                            _tts_queue.put(None)
                            print()

            # Ensure TTS cleanup
            if tts_thread is not None:
                _stop_speaking.set()
                _tts_queue.put(None)
                tts_thread.join(timeout=2)
            if _mic_listener is not None:
                _mic_listener.muted.clear()

        except KeyboardInterrupt:
            _stop_speaking.set()
            _kill_active_say()
            if _mic_listener is not None:
                _mic_listener.muted.clear()
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
