"""
X1 FastAPI Bridge
Connects the SvelteKit dashboard to the prototype-x1 brain via WebSocket.
Handles: brain streaming, mic listener, TTS, pause/resume.
"""
import asyncio
import json
import logging
import os
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Brain path — prototype-x1 ─────────────────────────────────────────────────
BRAIN_PATH = Path(__file__).parent.parent / "prototype-x1"
sys.path.insert(0, str(BRAIN_PATH))

try:
    from brain.core import Brain
    from brain.config import BrainConfig
    BRAIN_AVAILABLE = True
except ImportError as _e:
    BRAIN_AVAILABLE = False
    Brain = None        # type: ignore
    BrainConfig = None  # type: ignore
    _import_err = str(_e)
else:
    _import_err = ""

# Try to load mic listener
try:
    from voice.listener import MicListener
    MIC_AVAILABLE = True
except ImportError:
    MIC_AVAILABLE = False
    MicListener = None  # type: ignore

# ── App setup ─────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("x1.server")

app = FastAPI(title="X1 Brain Bridge", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:4173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Brain singleton ───────────────────────────────────────────────────────────
_brain: Any = None
_start_time = time.time()

def get_brain():
    global _brain
    if _brain is None and BRAIN_AVAILABLE:
        cfg = BrainConfig()
        _brain = Brain(cfg)
        log.info("Brain initialised — ollama=%s fallback=%s",
                 cfg.llm.ollama_model, cfg.llm.cloud_fallback)
    return _brain

# ── TTS engine ────────────────────────────────────────────────────────────────
_SAY_RATE = "210"
_HAS_SAY = shutil.which("say") is not None
_active_say_proc: "subprocess.Popen[bytes] | None" = None
_proc_lock = threading.Lock()

# Phrases llama3.2 outputs despite being told not to
_BANNED_PHRASES = [
    "no response is required", "screengrab now", "action completed",
    "the speak skill has completed", "your feedback is acknowledged",
    "relevant memory prepended",
]

def _strip_banned(text: str) -> str:
    for phrase in _BANNED_PHRASES:
        text = re.sub(re.escape(phrase), "", text, flags=re.IGNORECASE)
    return text.strip()

def _strip_bare_json(text: str) -> str:
    """Remove bare JSON skill-call objects from text, handling nested braces."""
    result = []
    i = 0
    while i < len(text):
        if text[i] == '{':
            depth = 0
            j = i
            while j < len(text):
                if text[j] == '{':
                    depth += 1
                elif text[j] == '}':
                    depth -= 1
                    if depth == 0:
                        i = j + 1
                        break
                j += 1
            else:
                i = j
            continue
        result.append(text[i])
        i += 1
    return "".join(result)

def _kill_active_say() -> None:
    """Kill running `say` process immediately. Safe to call from any thread."""
    global _active_say_proc
    with _proc_lock:
        proc = _active_say_proc
        _active_say_proc = None
    if proc and proc.poll() is None:
        try:
            proc.kill()
        except OSError:
            pass

def _speak_sentence(sentence: str, stop: threading.Event) -> None:
    """Speak a single sentence via macOS `say`. Skips if stop is set."""
    global _active_say_proc
    if not _HAS_SAY:
        return
    sentence = re.sub(r"[`*_#>\[\]]+", "", sentence).strip()
    if not sentence or stop.is_set():
        return
    proc = subprocess.Popen(["say", "-r", _SAY_RATE, sentence])
    with _proc_lock:
        _active_say_proc = proc
    proc.wait()
    with _proc_lock:
        if _active_say_proc is proc:     # only clear if WE still own it
            _active_say_proc = None

# ── Message helpers ───────────────────────────────────────────────────────────
def msg(kind: str, payload: Any) -> str:
    return json.dumps({"kind": kind, "payload": payload, "ts": int(time.time() * 1000)})

def status_msg(state: str, extra: dict | None = None) -> str:
    brain = get_brain()
    if not BRAIN_AVAILABLE:
        model_name = "mock"
    elif brain:
        model_name = f"{brain.cfg.llm.ollama_model} / {brain.cfg.llm.cloud_fallback}"
    else:
        model_name = "—"
    payload = {
        "state": state,
        "model": model_name,
        "uptime_s": int(time.time() - _start_time),
        "skills": brain.skill_list if brain else [],
        "mic": MIC_AVAILABLE,
        "tts": _HAS_SAY,
    }
    if extra:
        payload.update(extra)
    return msg("status", payload)

# ── Streaming bridge: sync generator → async WebSocket + TTS ─────────────────
async def _stream_brain(ws: WebSocket, brain: Any, user_text: str,
                        speaker: str = "unknown",
                        stop_speaking: threading.Event | None = None,
                        mic_listener: Any = None,
                        cancel_event: threading.Event | None = None) -> bool:
    """Run brain.stream() in a thread, relay events to WebSocket, fire TTS.

    Returns True if completed normally, False if cancelled by cancel_event.
    """
    loop = asyncio.get_running_loop()
    aqueue: asyncio.Queue = asyncio.Queue()
    _DONE = object()

    def run_in_thread():
        try:
            for event in brain.stream(user_text, speaker=speaker,
                                       stop_event=cancel_event):
                loop.call_soon_threadsafe(aqueue.put_nowait, event)
        except Exception as exc:
            loop.call_soon_threadsafe(aqueue.put_nowait, ("error", str(exc)))
        finally:
            loop.call_soon_threadsafe(aqueue.put_nowait, _DONE)

    threading.Thread(target=run_in_thread, daemon=True).start()

    # TTS: sentence-level streaming — speak as sentences complete mid-stream
    tts_q: "queue.Queue[str | None]" = queue.Queue()
    stop_ev = stop_speaking or threading.Event()

    def tts_worker():
        if mic_listener is not None:
            mic_listener.muted.set()
        try:
            while True:
                sentence = tts_q.get()
                if sentence is None:
                    break
                if stop_ev.is_set() or (cancel_event and cancel_event.is_set()):
                    continue  # drain queue without speaking
                _speak_sentence(sentence, stop_ev)
                try:
                    loop.call_soon_threadsafe(
                        aqueue.put_nowait, ("_tts", True))
                except Exception:
                    pass
        finally:
            if mic_listener is not None:
                time.sleep(0.3)
                mic_listener.muted.clear()
            try:
                loop.call_soon_threadsafe(
                    aqueue.put_nowait, ("_tts", False))
            except Exception:
                pass

    if _HAS_SAY:
        threading.Thread(target=tts_worker, daemon=True).start()

    final_response: dict | None = None
    brace_depth = 0
    sentence_buf: list[str] = []
    was_cancelled = False

    while True:
        event = await aqueue.get()
        if event is _DONE:
            break

        kind, data = event
        if kind == "cancelled":
            was_cancelled = True
            tts_q.put(None)
            # Don't return yet — drain remaining events until _DONE
            continue
        if was_cancelled:
            continue  # skip events after cancellation
        if kind == "_tts":
            await ws.send_text(msg("tts", {"speaking": data}))
            continue
        if kind == "thinking":
            await ws.send_text(msg("thinking", {}))
        elif kind == "token":
            # Filter JSON skill calls from both UI stream and TTS
            clean_chars: list[str] = []
            for ch in data:
                if ch == '{':
                    brace_depth += 1
                if brace_depth > 0:
                    if ch == '}':
                        brace_depth -= 1
                    continue
                clean_chars.append(ch)
                sentence_buf.append(ch)
                if ch in '.!?' and len(sentence_buf) > 3:
                    sentence = "".join(sentence_buf).strip()
                    sentence = _strip_banned(sentence)
                    if sentence:
                        tts_q.put(sentence)
                    sentence_buf.clear()
            # Only send clean text to the UI (no raw JSON)
            clean_text = "".join(clean_chars)
            if clean_text:
                await ws.send_text(msg("token", {"text": clean_text}))
        elif kind == "action":
            await ws.send_text(msg("action", {"action": data}))
            sentence_buf.clear()
        elif kind == "vision":
            await ws.send_text(msg("vision", data))
        elif kind == "done":
            br = data  # BrainResponse
            # Strip JSON blocks and banned phrases from final text
            clean_final = _strip_bare_json(br.text)
            clean_final = _strip_banned(clean_final)
            final_response = {
                "text": clean_final,
                "provider": br.provider,
                "latency_ms": br.latency_ms,
                "skill_calls": br.skill_calls,
            }
            # Flush remaining sentence fragment
            leftover = "".join(sentence_buf).strip()
            leftover = _strip_banned(leftover)
            if leftover:
                tts_q.put(leftover)
            tts_q.put(None)  # signal TTS worker to exit
        elif kind == "error":
            log.error("Brain stream error: %s", data)
            tts_q.put(None)
            await ws.send_text(msg("error", {"message": data}))
            return True

    if not was_cancelled and final_response is not None:
        await ws.send_text(msg("response", final_response))

    return not was_cancelled


# ── Mic listener background task ─────────────────────────────────────────────
_mic_listener: Any = None
_mic_stop = threading.Event()

def _start_mic_listener():
    """Start the mic listener in a background thread. Returns the listener."""
    global _mic_listener
    if not MIC_AVAILABLE or _mic_listener is not None:
        return _mic_listener
    _mic_listener = MicListener(model_size="base")
    log.info("Mic listener starting…")
    return _mic_listener


# ── WebSocket handler ─────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    log.info("Client connected: %s", ws.client)
    brain = get_brain()
    stop_speaking = threading.Event()

    # Start mic listener
    mic = _start_mic_listener()
    _paused = False  # True = fully paused (no listening, no responding)

    # Send initial status
    await ws.send_text(status_msg("idle"))

    # Background task: relay mic transcriptions to this WebSocket
    # Uses debouncing + cancellation for fluid conversation
    mic_task = None
    _cancel_brain = threading.Event()  # signals brain to stop mid-stream
    _DEBOUNCE_SEC = 0.6  # wait for Whisper fragments to settle

    if mic is not None:
        async def mic_relay():
            """Run mic.listen() in a thread, relay transcriptions to WS + brain.

            Key behaviors:
            - Debounces rapid Whisper fragments into a single input
            - Cancels ongoing brain response when new speech arrives
            - Never blocks on a stale response
            """
            loop = asyncio.get_running_loop()
            mic_q: asyncio.Queue = asyncio.Queue()

            def mic_thread():
                try:
                    for text in mic.listen():
                        loop.call_soon_threadsafe(mic_q.put_nowait, text)
                except Exception as e:
                    log.error("Mic listener error: %s", e)

            threading.Thread(target=mic_thread, daemon=True).start()

            processing = False  # True while brain is streaming a response

            while True:
                try:
                    text = await mic_q.get()
                except asyncio.CancelledError:
                    break

                # ── Debounce: collect rapid fragments ──────────────────
                collected = text
                deadline = asyncio.get_event_loop().time() + _DEBOUNCE_SEC
                while True:
                    remaining = deadline - asyncio.get_event_loop().time()
                    if remaining <= 0:
                        break
                    try:
                        more = await asyncio.wait_for(mic_q.get(), timeout=remaining)
                        collected = more  # use the latest (most complete) fragment
                        deadline = asyncio.get_event_loop().time() + _DEBOUNCE_SEC
                    except asyncio.TimeoutError:
                        break

                text = collected.strip()
                if not text:
                    continue

                # ── Drop everything while paused ─────────────────────
                if _paused:
                    log.debug("Paused — dropping mic text: %r", text)
                    continue

                # ── Cancel any in-progress response ───────────────────
                if processing:
                    log.info("New speech arrived — cancelling current response")
                    _cancel_brain.set()
                    _kill_active_say()
                    # Give brain thread a moment to see the cancel
                    await asyncio.sleep(0.05)

                try:
                    # Send mic transcription to UI
                    await ws.send_text(msg("mic", {"text": text}))

                    # Feed into brain
                    if brain:
                        log.info("Debounced input → brain: %r", text)
                        _cancel_brain.clear()
                        stop_speaking.clear()
                        processing = True
                        await ws.send_text(msg("thinking", {}))
                        completed = await _stream_brain(
                            ws, brain, text, speaker="Neokode",
                            stop_speaking=stop_speaking,
                            mic_listener=mic,
                            cancel_event=_cancel_brain,
                        )
                        processing = False
                        log.info("Brain stream finished — completed=%s", completed)
                        if completed:
                            await ws.send_text(status_msg("idle"))
                        else:
                            log.info("Response cancelled — ready for new input")
                    else:
                        log.warning("No brain available — mic text dropped: %r", text)
                except Exception as exc:
                    log.error("mic_relay brain error: %s", exc, exc_info=True)
                    processing = False

        mic_task = asyncio.create_task(mic_relay())

    try:
        while True:
            raw = await ws.receive_text()
            data = json.loads(raw)
            kind = data.get("kind")

            if kind == "command":
                user_text: str = data["payload"]["text"]
                speaker: str = data["payload"].get("speaker", "Neokode")
                log.info("Command received: %s (speaker=%s)", user_text, speaker)

                stop_speaking.clear()

                if brain:
                    _cancel_brain.clear()
                    await _stream_brain(ws, brain, user_text, speaker=speaker,
                                        stop_speaking=stop_speaking,
                                        mic_listener=mic,
                                        cancel_event=_cancel_brain)
                else:
                    reason = _import_err if not BRAIN_AVAILABLE else "brain init failed"
                    await ws.send_text(msg("thinking", {}))
                    await asyncio.sleep(0.4)
                    await ws.send_text(msg("response", {
                        "text": f"[MOCK] {reason}. You said: {user_text}",
                        "provider": "mock",
                        "latency_ms": 400,
                        "skill_calls": [],
                    }))

                await ws.send_text(status_msg("idle"))

            elif kind == "pause":
                # Full stop: TTS + brain + mic listening
                log.info("Pause received — killing TTS, cancelling brain, stopping listener")
                _paused = True
                stop_speaking.set()
                _cancel_brain.set()
                _kill_active_say()
                if _mic_listener is not None:
                    _mic_listener.muted.set()
                    # Drain any queued audio so it doesn't fire on resume
                    try:
                        while not _mic_listener._q.empty():
                            _mic_listener._q.get_nowait()
                    except Exception:
                        pass
                await ws.send_text(msg("tts", {"speaking": False}))
                await ws.send_text(msg("mic", {"text": ""}))  # clear indicator
                await ws.send_text(status_msg("idle", {"paused": True, "mic": False}))

            elif kind == "resume":
                log.info("Resume received — resuming listener")
                _paused = False
                stop_speaking.clear()
                _cancel_brain.clear()
                if _mic_listener is not None:
                    _mic_listener.muted.clear()
                await ws.send_text(status_msg("idle", {"paused": False, "mic": True}))

    except WebSocketDisconnect:
        log.info("Client disconnected")
    except Exception as e:
        log.exception("WebSocket error: %s", e)
        try:
            await ws.send_text(msg("error", {"message": str(e)}))
        except Exception:
            pass
    finally:
        # Kill any running TTS so `say` doesn't keep talking after disconnect
        stop_speaking.set()
        _kill_active_say()
        if mic_task:
            mic_task.cancel()
        # Stop the mic listener so it can be restarted on next connection
        if _mic_listener is not None:
            _mic_listener.stop()
            _mic_listener._listening = False
            _mic_listener._stop.clear()  # reset for next connection

# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"ok": True, "brain_available": BRAIN_AVAILABLE, "uptime_s": int(time.time() - _start_time)}


# ── Skill Builder REST API ─────────────────────────────────────────────────────
SKILLS_DIR = BRAIN_PATH / "skills"


class SkillSaveRequest(BaseModel):
    code: str


@app.get("/skills")
def list_skills():
    """List all .py skill files in prototype-x1/skills/."""
    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(p.name for p in SKILLS_DIR.glob("*.py") if not p.name.startswith("_"))
    return {"skills": files}


@app.post("/skills/{name}")
def save_skill(name: str, body: SkillSaveRequest):
    """Save (or overwrite) a skill file and hot-reload it into the brain."""
    if not name.replace("_", "").isalnum():
        raise HTTPException(status_code=400, detail="Skill name must be alphanumeric + underscores")
    if name.startswith("_"):
        raise HTTPException(status_code=400, detail="Skill name cannot start with underscore")

    SKILLS_DIR.mkdir(parents=True, exist_ok=True)
    skill_path = SKILLS_DIR / f"{name}.py"
    skill_path.write_text(body.code, encoding="utf-8")
    log.info("Skill saved: %s (%d bytes)", skill_path, len(body.code))

    # Hot-reload into the running brain
    brain = get_brain()
    if brain:
        try:
            from brain.skills import load_skill_file  # type: ignore[import]
            load_skill_file(skill_path)
            log.info("Hot-reloaded skill: %s", name)
        except Exception as exc:
            log.warning("Hot-reload failed for %s: %s", name, exc)

    return {"ok": True, "file": skill_path.name, "bytes": len(body.code)}

