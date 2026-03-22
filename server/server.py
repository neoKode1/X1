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

# ── Brain path ────────────────────────────────────────────────────────────────
BRAIN_PATH = Path(__file__).parent.parent / "robot-brain"
sys.path.insert(0, str(BRAIN_PATH))

try:
    from brain import Brain
    from config import RobotConfig
    BRAIN_AVAILABLE = True
except ImportError:
    BRAIN_AVAILABLE = False
    Brain = None  # type: ignore
    RobotConfig = None  # type: ignore

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
        config = RobotConfig()
        _brain = Brain(config)
        log.info("Brain initialised — model: %s", config.llm.model)
    return _brain

# ── Message helpers ───────────────────────────────────────────────────────────
def msg(kind: str, payload: Any) -> str:
    return json.dumps({"kind": kind, "payload": payload, "ts": int(time.time() * 1000)})

def status_msg(state: str) -> str:
    brain = get_brain()
    model_name = "mock" if not BRAIN_AVAILABLE else (brain.config.llm.model if brain else "—")
    return msg("status", {
        "state": state,
        "model": model_name,
        "uptime_s": int(time.time() - _start_time),
    })

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
                    # Run brain in thread pool to avoid blocking event loop
                    loop = asyncio.get_event_loop()
                    response = await loop.run_in_executor(
                        None, brain.process, user_text
                    )

                    # Emit actions
                    for action in response.actions:
                        await ws.send_text(msg("action", {
                            "action": {
                                "type": action.type,
                                "params": action.params,
                                "result": action.result if hasattr(action, "result") else None,
                            }
                        }))
                        await asyncio.sleep(0.05)

                    # Emit final response
                    await ws.send_text(msg("response", {
                        "text": response.text,
                        "actions": [
                            {"type": a.type, "params": a.params}
                            for a in response.actions
                        ],
                    }))
                else:
                    # Mock mode — brain not available
                    await asyncio.sleep(0.8)
                    await ws.send_text(msg("response", {
                        "text": f"[MOCK] Brain not loaded. You said: {user_text}",
                        "actions": [],
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

