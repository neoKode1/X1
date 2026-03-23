"""
ARIA Peer Node — backup_001
============================
A pre-authorized secondary node that can run alongside or instead of
the main brain. Identified by PEER_ID, trusted at Friend tier without
manual introduction.

Spec:
  ID            : backup_001
  Role          : Friend (Tier 2)
  Voice         : low_latency  (local TTS, no cloud round-trip)
  Memory        : shared scrap_list.json + bcs.json
  BCS version   : v3
  Tone protocol : founder pattern detection active
  Loop          : raw (sync, no streaming, minimal overhead)
  Handshake     : HMAC-SHA256 challenge/response

Handshake flow:
  1. Peer builds a token:  HMAC-SHA256(secret, f"{peer_id}:{ts}")
  2. Peer sends:           {"peer_id", "role", "bcs_version", "ts", "token"}
  3. Brain calls accept_peer_handshake(payload) → validates token, window
  4. Brain returns:        {"ok": True, "bcs": <summary>, "scrap": <report>}
"""
from __future__ import annotations

import hashlib
import hmac
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .core import Brain

log = logging.getLogger("x1.peer")

# ── Peer identity (hardcoded, matches trust._PEERS) ───────────────────────────

PEER_ID      = "backup_001"
VOICE_MODE   = "low_latency"
BCS_VERSION  = "v3"
PEER_ROLE    = "friend"

_SECRET_PATH = Path(__file__).parent.parent / "memory" / "peer_secret.key"
_HS_WINDOW_S = 30      # handshake token valid ± 30 seconds


# ── Secret management ─────────────────────────────────────────────────────────

def _load_or_create_secret() -> bytes:
    """Load shared HMAC secret, or generate and persist a new one."""
    if _SECRET_PATH.exists():
        return _SECRET_PATH.read_bytes()
    _SECRET_PATH.parent.mkdir(parents=True, exist_ok=True)
    import os
    secret = os.urandom(32)
    _SECRET_PATH.write_bytes(secret)
    log.info("Peer secret generated → %s", _SECRET_PATH)
    return secret


# ── Token ─────────────────────────────────────────────────────────────────────

def generate_token(peer_id: str, secret: bytes, ts: float) -> str:
    """HMAC-SHA256 token for the given peer_id + timestamp."""
    msg = f"{peer_id}:{ts:.0f}".encode()
    return hmac.new(secret, msg, hashlib.sha256).hexdigest()


def build_handshake(peer_id: str = PEER_ID, secret: bytes | None = None) -> dict:
    """Build a handshake payload ready to send to the main brain."""
    if secret is None:
        secret = _load_or_create_secret()
    ts = time.time()
    return {
        "peer_id":     peer_id,
        "role":        PEER_ROLE,
        "bcs_version": BCS_VERSION,
        "voice":       VOICE_MODE,
        "ts":          ts,
        "token":       generate_token(peer_id, secret, ts),
    }


def verify_handshake(payload: dict, secret: bytes | None = None) -> tuple[bool, str]:
    """
    Validate a peer handshake payload.
    Returns (ok: bool, reason: str).
    """
    if secret is None:
        secret = _load_or_create_secret()

    peer_id = payload.get("peer_id", "")
    ts      = float(payload.get("ts", 0))
    token   = payload.get("token", "")

    if abs(time.time() - ts) > _HS_WINDOW_S:
        return False, f"token expired (drift={abs(time.time()-ts):.1f}s)"

    expected = generate_token(peer_id, secret, ts)
    if not hmac.compare_digest(token, expected):
        return False, "token mismatch"

    return True, "ok"


# ── Peer config dataclass ─────────────────────────────────────────────────────

@dataclass
class PeerConfig:
    peer_id:     str   = PEER_ID
    role:        str   = PEER_ROLE
    voice:       str   = VOICE_MODE
    bcs_version: str   = BCS_VERSION
    connected_at: float = field(default_factory=time.time)


# ── Raw loop ──────────────────────────────────────────────────────────────────

def raw_process(brain: "Brain", user_input: str, speaker: str = PEER_ID) -> str:
    """
    Stripped-down sync reasoning call for the backup node.
    No streaming, no WebSocket overhead — returns the reply string directly.
    Founder tone protocol is active (speaker is trusted Friend tier).
    """
    response = brain.process(user_input, speaker=speaker)
    return response.text


def raw_loop(brain: "Brain") -> None:
    """
    Interactive CLI raw loop for backup_001.
    Reads stdin, calls raw_process, prints reply.
    Ctrl-C to exit.
    """
    cfg = PeerConfig()
    print(f"[{cfg.peer_id}] raw loop — voice={cfg.voice} bcs={cfg.bcs_version}")
    print(f"[{cfg.peer_id}] BCS: {brain.bcs.state.summary()}")
    print("─" * 60)

    try:
        while True:
            try:
                user_input = input("you> ").strip()
            except EOFError:
                break
            if not user_input:
                continue
            if user_input.lower() in ("exit", "quit", "bye"):
                break
            reply = raw_process(brain, user_input)
            print(f"aria> {reply}\n")
    except KeyboardInterrupt:
        pass

    print(f"\n[{cfg.peer_id}] loop exited — BCS: {brain.bcs.state.summary()}")

