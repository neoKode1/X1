"""
X1 FastAPI Bridge
Connects the SvelteKit dashboard to the prototype-x1 brain via WebSocket.
"""
import asyncio
import json
import logging
import sys
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

                await ws.send_text(msg("thinking", {}))

                if brain:
                    # Run brain in thread pool to avoid blocking the event loop
                    loop = asyncio.get_running_loop()
                    response = await loop.run_in_executor(
                        None, brain.process, user_text
                    )

                    # Emit each skill call as an "action" event
                    for call, result in zip(response.skill_calls, response.skill_results):
                        await ws.send_text(msg("action", {
                            "skill": call.get("name", "unknown"),
                            "args": call.get("args", {}),
                            "result": result,
                        }))
                        await asyncio.sleep(0.05)

                    # Emit final response
                    await ws.send_text(msg("response", {
                        "text": response.text,
                        "provider": response.provider,
                        "latency_ms": response.latency_ms,
                        "skill_calls": response.skill_calls,
                    }))
                else:
                    # Mock mode — brain not available
                    reason = _import_err if not BRAIN_AVAILABLE else "brain init failed"
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

