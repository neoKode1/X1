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

def _kill_active_say() -> None:
    global _active_say_proc
    with _proc_lock:
        if _active_say_proc and _active_say_proc.poll() is None:
            _active_say_proc.terminate()

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
                        mic_listener: Any = None) -> None:
    """Run brain.stream() in a thread, relay events to WebSocket, fire TTS."""
    loop = asyncio.get_running_loop()
    aqueue: asyncio.Queue = asyncio.Queue()
    _DONE = object()

    def run_in_thread():
        try:
            for event in brain.stream(user_text, speaker=speaker):
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
                if stop_ev.is_set():
                    continue  # drain queue without speaking
                _speak_sentence(sentence, stop_ev)
                # Notify client TTS is speaking
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

    while True:
        event = await aqueue.get()
        if event is _DONE:
            break

        kind, data = event
        if kind == "_tts":
            await ws.send_text(msg("tts", {"speaking": data}))
            continue
        if kind == "thinking":
            await ws.send_text(msg("thinking", {}))
        elif kind == "token":
            await ws.send_text(msg("token", {"text": data}))
            # Parse for TTS sentence boundaries (suppress JSON skill calls)
            for ch in data:
                if ch == '{':
                    brace_depth += 1
                if brace_depth > 0:
                    if ch == '}':
                        brace_depth -= 1
                    continue
                sentence_buf.append(ch)
                if ch in '.!?' and len(sentence_buf) > 3:
                    sentence = "".join(sentence_buf).strip()
                    sentence = _strip_banned(sentence)
                    if sentence:
                        tts_q.put(sentence)
                    sentence_buf.clear()
        elif kind == "action":
            await ws.send_text(msg("action", {"action": data}))
            sentence_buf.clear()
        elif kind == "vision":
            await ws.send_text(msg("vision", data))
        elif kind == "done":
            br = data  # BrainResponse
            final_response = {
                "text": br.text,
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
            tts_q.put(None)
            await ws.send_text(msg("error", {"message": data}))
            return

    if final_response is not None:
        await ws.send_text(msg("response", final_response))


# ── Mic listener background task ─────────────────────────────────────────────
_mic_listener: Any = None
_mic_stop = threading.Event()

def _start_mic_listener():
    """Start the mic listener in a background thread. Returns the listener."""
    global _mic_listener
    if not MIC_AVAILABLE or _mic_listener is not None:
        return _mic_listener
    _mic_listener = MicListener(model_size="tiny")
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

    # Send initial status
    await ws.send_text(status_msg("idle"))

    # Background task: relay mic transcriptions to this WebSocket
    mic_task = None
    if mic is not None:
        async def mic_relay():
            """Run mic.listen() in a thread, relay transcriptions to WS + brain."""
            loop = asyncio.get_running_loop()
            mic_q: asyncio.Queue = asyncio.Queue()

            def mic_thread():
                try:
                    for text in mic.listen():
                        loop.call_soon_threadsafe(mic_q.put_nowait, text)
                except Exception as e:
                    log.error("Mic listener error: %s", e)

            threading.Thread(target=mic_thread, daemon=True).start()

            while True:
                try:
                    text = await mic_q.get()
                except asyncio.CancelledError:
                    break
                log.info("Mic: %s", text)
                # Send mic transcription to UI
                await ws.send_text(msg("mic", {"text": text}))
                # Feed into brain automatically
                if brain:
                    stop_speaking.clear()
                    await ws.send_text(status_msg("thinking"))
                    await _stream_brain(ws, brain, text, speaker="Neokode",
                                        stop_speaking=stop_speaking,
                                        mic_listener=mic)
                    await ws.send_text(status_msg("idle"))

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
                    await _stream_brain(ws, brain, user_text, speaker=speaker,
                                        stop_speaking=stop_speaking,
                                        mic_listener=mic)
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
                # Stop TTS immediately, keep mic open
                log.info("Pause received — killing TTS")
                stop_speaking.set()
                _kill_active_say()
                await ws.send_text(msg("tts", {"speaking": False}))
                await ws.send_text(status_msg("idle", {"paused": True}))

            elif kind == "resume":
                log.info("Resume received")
                stop_speaking.clear()
                await ws.send_text(status_msg("idle", {"paused": False}))

    except WebSocketDisconnect:
        log.info("Client disconnected")
    except Exception as e:
        log.exception("WebSocket error: %s", e)
        try:
            await ws.send_text(msg("error", {"message": str(e)}))
        except Exception:
            pass
    finally:
        if mic_task:
            mic_task.cancel()

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

