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

from fastapi import FastAPI, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
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
    allow_origins=["*"],  # local dev — file:// and any localhost port
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
        log.info("Brain initialised — primary=%s (%s) fallback=%s (%s)",
                 cfg.llm.primary, cfg.llm.anthropic_model if cfg.llm.anthropic_api_key else "no key",
                 cfg.llm.cloud_fallback, cfg.llm.ollama_model)
    return _brain

# ── TTS engine ────────────────────────────────────────────────────────────────
# Neural TTS via edge-tts (Microsoft neural voices) with afplay fallback
_EDGE_TTS_VOICE = os.getenv("TTS_VOICE", "en-US-AvaMultilingualNeural")
_active_say_proc: "subprocess.Popen[bytes] | None" = None
_proc_lock = threading.Lock()

# Check for edge-tts availability, fall back to macOS say
try:
    import edge_tts  # noqa: F401
    _HAS_EDGE_TTS = True
except ImportError:
    _HAS_EDGE_TTS = False

_HAS_SAY = shutil.which("say") is not None
_HAS_AFPLAY = shutil.which("afplay") is not None
_TTS_AVAILABLE = _HAS_EDGE_TTS or _HAS_SAY

if _HAS_EDGE_TTS:
    log.info("TTS: edge-tts available — using neural voice %s", _EDGE_TTS_VOICE)
elif _HAS_SAY:
    log.info("TTS: edge-tts not found, falling back to macOS say")
else:
    log.warning("TTS: no TTS engine available")

# Centralized text filters — imported from brain.filters
try:
    from brain.filters import strip_banned as _strip_banned, strip_bare_json as _strip_bare_json, clean_response as _clean_response
except ImportError:
    # Minimal fallback if brain module not available
    def _strip_banned(text: str) -> str: return text
    def _strip_bare_json(text: str) -> str: return text
    def _clean_response(text: str) -> str: return text

def _kill_active_say() -> None:
    """Kill running audio playback process immediately. Safe to call from any thread."""
    global _active_say_proc
    with _proc_lock:
        proc = _active_say_proc
        _active_say_proc = None
    if proc and proc.poll() is None:
        try:
            proc.kill()
        except OSError:
            pass

# Persistent event loop for edge-tts (avoids asyncio.run() per sentence)
_tts_loop: asyncio.AbstractEventLoop | None = None
_tts_loop_thread: threading.Thread | None = None

def _get_tts_loop() -> asyncio.AbstractEventLoop:
    """Get or create a persistent event loop running in a background thread."""
    global _tts_loop, _tts_loop_thread
    if _tts_loop is not None and _tts_loop.is_running():
        return _tts_loop
    _tts_loop = asyncio.new_event_loop()
    def _run():
        asyncio.set_event_loop(_tts_loop)
        _tts_loop.run_forever()
    _tts_loop_thread = threading.Thread(target=_run, daemon=True)
    _tts_loop_thread.start()
    return _tts_loop

def _generate_audio_edge(sentence: str) -> str | None:
    """Generate audio file via edge-tts. Returns temp file path or None."""
    import tempfile
    sentence = re.sub(r"[`*_#>\[\]]+", "", sentence).strip()
    if not sentence:
        return None
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tmp_path = tmp.name
    tmp.close()
    try:
        import edge_tts as _edge
        loop = _get_tts_loop()
        future = asyncio.run_coroutine_threadsafe(
            _edge.Communicate(sentence, _EDGE_TTS_VOICE).save(tmp_path), loop
        )
        future.result(timeout=15)
        return tmp_path
    except Exception as e:
        log.warning("edge-tts generation failed: %s", e)
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        return None

def _play_audio_file(path: str, stop: threading.Event) -> None:
    """Play an audio file and clean it up afterwards."""
    global _active_say_proc
    if stop.is_set():
        return
    try:
        player = "afplay" if _HAS_AFPLAY else "aplay"
        proc = subprocess.Popen([player, path])
        with _proc_lock:
            _active_say_proc = proc
        proc.wait()
        with _proc_lock:
            if _active_say_proc is proc:
                _active_say_proc = None
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass

def _speak_sentence_say(sentence: str, stop: threading.Event) -> None:
    """Fallback: speak via macOS `say`."""
    global _active_say_proc
    if not _HAS_SAY:
        return
    sentence = re.sub(r"[`*_#>\[\]]+", "", sentence).strip()
    if not sentence or stop.is_set():
        return
    proc = subprocess.Popen(["say", "-v", "Flo", "-r", "210", sentence])
    with _proc_lock:
        _active_say_proc = proc
    proc.wait()
    with _proc_lock:
        if _active_say_proc is proc:
            _active_say_proc = None

def _speak_sentence(sentence: str, stop: threading.Event) -> None:
    """Speak a single sentence. Uses edge-tts if available, else macOS say."""
    if _HAS_EDGE_TTS:
        audio_path = _generate_audio_edge(sentence)
        if audio_path:
            _play_audio_file(audio_path, stop)
        elif _HAS_SAY:
            _speak_sentence_say(sentence, stop)
    elif _HAS_SAY:
        _speak_sentence_say(sentence, stop)
    # else: no TTS available, silently skip

# ── Message helpers ───────────────────────────────────────────────────────────
def msg(kind: str, payload: Any) -> str:
    return json.dumps({"kind": kind, "payload": payload, "ts": int(time.time() * 1000)})

def status_msg(state: str, extra: dict | None = None) -> str:
    brain = get_brain()
    if not BRAIN_AVAILABLE:
        model_name = "mock"
    elif brain:
        model_name = f"{brain.cfg.llm.anthropic_model} (fallback: {brain.cfg.llm.ollama_model})"
    else:
        model_name = "—"
    payload = {
        "state": state,
        "model": model_name,
        "uptime_s": int(time.time() - _start_time),
        "skills": brain.skill_list if brain else [],
        "mic": MIC_AVAILABLE,
        "tts": _TTS_AVAILABLE,
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

    # TTS: pipelined — generate audio ahead while playing current sentence
    tts_q: "queue.Queue[str | None]" = queue.Queue()
    # audio_q holds pre-generated file paths (or None = done sentinel)
    audio_q: "queue.Queue[str | None]" = queue.Queue(maxsize=3)
    stop_ev = stop_speaking or threading.Event()

    def tts_generator():
        """Thread 1: reads sentences, generates audio files, pushes to audio_q."""
        while True:
            sentence = tts_q.get()
            if sentence is None:
                audio_q.put(None)  # signal player to stop
                break
            if stop_ev.is_set() or (cancel_event and cancel_event.is_set()):
                continue  # drain without generating
            if _HAS_EDGE_TTS:
                path = _generate_audio_edge(sentence)
                if path:
                    audio_q.put(path)
                elif _HAS_SAY:
                    # edge-tts failed for this sentence — fall back inline
                    audio_q.put(("say", sentence))  # type: ignore[arg-type]
            elif _HAS_SAY:
                audio_q.put(("say", sentence))  # type: ignore[arg-type]

    def tts_player():
        """Thread 2: plays pre-generated audio files back-to-back (minimal gaps)."""
        _we_muted = False
        if mic_listener is not None and not mic_listener.muted.is_set():
            mic_listener.muted.set()
            _we_muted = True
        try:
            while True:
                item = audio_q.get()
                if item is None:
                    break
                if stop_ev.is_set() or (cancel_event and cancel_event.is_set()):
                    # Clean up any generated files we're skipping
                    if isinstance(item, str):
                        try:
                            os.unlink(item)
                        except OSError:
                            pass
                    continue
                if isinstance(item, tuple) and item[0] == "say":
                    _speak_sentence_say(item[1], stop_ev)
                elif isinstance(item, str):
                    _play_audio_file(item, stop_ev)
                try:
                    loop.call_soon_threadsafe(
                        aqueue.put_nowait, ("_tts", True))
                except Exception:
                    pass
        finally:
            if mic_listener is not None and _we_muted:
                time.sleep(0.3)
                mic_listener.muted.clear()
            try:
                loop.call_soon_threadsafe(
                    aqueue.put_nowait, ("_tts", False))
            except Exception:
                pass

    if _TTS_AVAILABLE:
        threading.Thread(target=tts_generator, daemon=True).start()
        threading.Thread(target=tts_player, daemon=True).start()

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
    global _mic_listener
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

    # ── Mic volume streamer (sends levels to frontend ~5x/sec) ────────────
    async def volume_streamer():
        """Send mic volume levels to the UI for the live level bar."""
        while True:
            try:
                if mic is not None and hasattr(mic, 'volume'):
                    vol = mic.volume
                    is_speech = vol > 0.03
                    await ws.send_json({"kind": "mic_level", "payload": {
                        "volume": round(vol, 4),
                        "speech": is_speech,
                    }})
                await asyncio.sleep(0.2)  # 5 Hz
            except Exception:
                break

    vol_task = None
    if mic is not None:
        vol_task = asyncio.create_task(volume_streamer())

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
                # Full stop: TTS + brain + mic listening — hard interrupt
                log.info("Pause received — killing TTS, cancelling brain, stopping listener")
                _paused = True
                stop_speaking.set()
                _cancel_brain.set()
                _kill_active_say()
                if _mic_listener is not None:
                    _mic_listener.muted.set()
                    _mic_listener.flush()  # drain queued audio
                await ws.send_text(msg("tts", {"speaking": False}))
                await ws.send_text(msg("mic", {"text": ""}))  # clear indicator
                await ws.send_text(status_msg("idle", {"paused": True, "mic": False}))

            elif kind == "resume":
                log.info("Resume received — resuming listener")
                _paused = False
                stop_speaking.clear()
                _cancel_brain.clear()
                if _mic_listener is not None:
                    _mic_listener.flush()  # clear stale audio before resuming
                    _mic_listener.muted.clear()
                await ws.send_text(status_msg("idle", {"paused": False, "mic": True}))

            elif kind == "vision":
                # Face/expression data from MediaPipe in-browser detection
                payload = data.get("payload", {})
                face_present = payload.get("face", False)
                expressions = payload.get("expressions", {})
                frame_b64 = payload.get("frame")  # base64 JPEG from browser
                log.info("Vision: face=%s expressions=%s frame=%s",
                         face_present,
                         {k: round(v, 2) for k, v in expressions.items()} if expressions else {},
                         f"{len(frame_b64)}chars" if frame_b64 else "none")
                brain = get_brain()
                if brain is not None:
                    # Derive mood from expressions
                    smile = (expressions.get("mouthSmileLeft", 0) + expressions.get("mouthSmileRight", 0)) / 2
                    frown = (expressions.get("mouthFrownLeft", 0) + expressions.get("mouthFrownRight", 0)) / 2
                    brow_up = expressions.get("browInnerUp", 0)
                    jaw_open = expressions.get("jawOpen", 0)
                    if smile > 0.4:
                        mood = "smiling"
                    elif frown > 0.3:
                        mood = "frowning"
                    elif brow_up > 0.4:
                        mood = "surprised"
                    elif jaw_open > 0.5:
                        mood = "mouth open / talking"
                    else:
                        mood = "neutral"
                    brain.update_vision(
                        state={"face_present": face_present, "expressions": expressions, "ts": time.time()},
                        frame_b64=frame_b64,
                        mood=mood,
                    )
                    log.info("Vision mood: %s", mood)

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
        if vol_task:
            vol_task.cancel()
        # Stop the mic listener so it can be restarted on next connection
        if _mic_listener is not None:
            _mic_listener.stop()
            _mic_listener._listening = False
            _mic_listener._stop.clear()  # reset for next connection
            _mic_listener = None  # force fresh instance on next connection

# ── Serve UI ──────────────────────────────────────────────────────────────────
UI_HTML = Path(__file__).parent.parent / "u.html"

@app.get("/")
def serve_ui():
    """Serve the main UI page."""
    if UI_HTML.exists():
        return FileResponse(UI_HTML, media_type="text/html")
    raise HTTPException(status_code=404, detail="u.html not found")

# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"ok": True, "brain_available": BRAIN_AVAILABLE, "uptime_s": int(time.time() - _start_time)}


# ── Webcam snapshot endpoint ──────────────────────────────────────────────────
@app.get("/webcam/snapshot")
def webcam_snapshot():
    """Return a single JPEG frame from the webcam. 204 if unavailable."""
    try:
        _root = str(BRAIN_PATH)
        if _root not in sys.path:
            sys.path.insert(0, _root)
        from vision.camera import capture_webcam  # type: ignore[import]
        frame = capture_webcam(device=0, max_size=(320, 240))
        if frame is None:
            return Response(status_code=204)
        import base64
        jpg_bytes = base64.b64decode(frame.frame_b64)
        return Response(content=jpg_bytes, media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})
    except Exception as exc:
        log.warning("Webcam snapshot failed: %s", exc)
        return Response(status_code=204)


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



# ── Knowledge Base (RAG) REST API ─────────────────────────────────────────────
KNOWLEDGE_DIR = BRAIN_PATH / "knowledge"

def _get_knowledge_store():
    """Lazy-init the knowledge store from the running brain or standalone."""
    brain = get_brain()
    if brain and hasattr(brain, "knowledge"):
        return brain.knowledge
    # Standalone fallback
    from brain.knowledge import KnowledgeStore  # type: ignore[import]
    persist_dir = str(BRAIN_PATH / "memory" / "chroma")
    return KnowledgeStore(persist_dir=persist_dir)


@app.get("/knowledge")
def list_knowledge():
    """List all documents in the knowledge base."""
    store = _get_knowledge_store()
    return {"docs": store.list_docs(), "total_chunks": store.count}


@app.post("/knowledge/upload")
async def upload_knowledge(file: UploadFile):
    """Upload a file and ingest it into the knowledge base."""
    KNOWLEDGE_DIR.mkdir(parents=True, exist_ok=True)
    dest = KNOWLEDGE_DIR / file.filename
    content = await file.read()
    dest.write_bytes(content)
    log.info("Knowledge file saved: %s (%d bytes)", dest.name, len(content))

    store = _get_knowledge_store()
    doc = store.ingest_file(dest, source="upload")
    if not doc:
        raise HTTPException(status_code=400, detail=f"Could not ingest {file.filename}")
    return {"ok": True, "doc": {"filename": doc.filename, "doc_id": doc.doc_id,
                                 "chunks": doc.chunks, "size_bytes": doc.size_bytes}}


@app.post("/knowledge/ingest")
def ingest_knowledge_folder():
    """Ingest all files from the knowledge drop folder."""
    store = _get_knowledge_store()
    docs = store.ingest_folder()
    return {"ok": True, "ingested": len(docs),
            "docs": [{"filename": d.filename, "doc_id": d.doc_id, "chunks": d.chunks}
                     for d in docs]}


@app.delete("/knowledge/{doc_id}")
def delete_knowledge(doc_id: str):
    """Remove a document from the knowledge base by doc_id."""
    store = _get_knowledge_store()
    removed = store.remove_doc(doc_id)
    if removed == 0:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"ok": True, "doc_id": doc_id, "chunks_removed": removed}


@app.post("/knowledge/search")
def search_knowledge(q: str, top_k: int = 3):
    """Search the knowledge base."""
    store = _get_knowledge_store()
    hits = store.search(q, top_k=top_k)
    return {"query": q, "results": hits}
