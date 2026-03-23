"""
X1 FastAPI Bridge
Connects the SvelteKit dashboard to the prototype-x1 brain via WebSocket.
"""
import asyncio
import json
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

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
    }
    if extra:
        payload.update(extra)
    return msg("status", payload)

# ── Streaming bridge: sync generator → async WebSocket ───────────────────────
async def _stream_brain(ws: WebSocket, brain: Any, user_text: str) -> None:
    """Run brain.stream() in a thread and relay events to the WebSocket."""
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    _DONE = object()

    def run_in_thread():
        try:
            for event in brain.stream(user_text):
                loop.call_soon_threadsafe(queue.put_nowait, event)
        except Exception as exc:
            loop.call_soon_threadsafe(queue.put_nowait, ("error", str(exc)))
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, _DONE)

    threading.Thread(target=run_in_thread, daemon=True).start()

    final_response: dict | None = None

    while True:
        event = await queue.get()
        if event is _DONE:
            break

        kind, data = event
        if kind == "thinking":
            await ws.send_text(msg("thinking", {}))
        elif kind == "token":
            await ws.send_text(msg("token", {"text": data}))
        elif kind == "action":
            await ws.send_text(msg("action", {"action": data}))
        elif kind == "done":
            br = data  # BrainResponse
            final_response = {
                "text": br.text,
                "provider": br.provider,
                "latency_ms": br.latency_ms,
                "skill_calls": br.skill_calls,
            }
        elif kind == "error":
            await ws.send_text(msg("error", {"message": data}))
            return

    if final_response is not None:
        await ws.send_text(msg("response", final_response))


# ── WebSocket handler ─────────────────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    log.info("Client connected: %s", ws.client)
    brain = get_brain()

    # Send initial status
    await ws.send_text(status_msg("idle"))

    try:
        while True:
            raw = await ws.receive_text()
            data = json.loads(raw)

            if data.get("kind") == "command":
                user_text: str = data["payload"]["text"]
                log.info("Command received: %s", user_text)

                if brain:
                    await _stream_brain(ws, brain, user_text)
                else:
                    # Mock mode — brain not available
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

    except WebSocketDisconnect:
        log.info("Client disconnected")
    except Exception as e:
        log.exception("WebSocket error: %s", e)
        await ws.send_text(msg("error", {"message": str(e)}))

# ── Health check ──────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"ok": True, "brain_available": BRAIN_AVAILABLE, "uptime_s": int(time.time() - _start_time)}

