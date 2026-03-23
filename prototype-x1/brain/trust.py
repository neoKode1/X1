"""
ARIA Trust Registry
====================
Tier 1 — Founder  (Neokode): hardcoded, immutable, highest authority.
                               Can introduce friends and revoke trust.
Tier 2 — Friends  : introduced by the Founder in a live session.
                    Full cooperative trust — help, skill access, state access.
Tier 3 — Unknown  : polite conversation only. No skills, no state access.

The Founder list is hardcoded and cannot be changed at runtime.
The Friend list persists across sessions via memory/trust.json.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from pathlib import Path

log = logging.getLogger("x1.trust")

# Hardcoded — cannot be promoted or demoted at runtime
_FOUNDERS: frozenset[str] = frozenset({"neokode", "neocode"})

# Pre-authorized peer nodes — Friend tier, no manual introduction required.
# These are ARIA instances or trusted subsystems, not humans.
_PEERS: frozenset[str] = frozenset({"backup_001"})

REGISTRY_PATH = Path(__file__).parent.parent / "memory" / "trust.json"


class Tier(IntEnum):
    UNKNOWN = 0
    FRIEND  = 5
    FOUNDER = 10


@dataclass
class TrustEntry:
    name: str
    introduced_by: str
    introduced_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    notes: str = ""


class TrustRegistry:
    """Persistent trust registry for ARIA's social model."""

    def __init__(self, path: Path = REGISTRY_PATH) -> None:
        self.path = path
        self._friends: dict[str, TrustEntry] = {}
        self._load()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text())
                self._friends = {k: TrustEntry(**v) for k, v in raw.items()}
                log.info("Trust registry: %d friend(s) loaded", len(self._friends))
            except Exception as exc:
                log.warning("Trust registry load failed: %s", exc)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps({k: asdict(v) for k, v in self._friends.items()}, indent=2)
        )

    # ── Tier resolution ──────────────────────────────────────────────────────

    def get_tier(self, name: str) -> Tier:
        key = name.strip().lower()
        if key in _FOUNDERS:
            return Tier.FOUNDER
        if key in _PEERS or key in self._friends:
            return Tier.FRIEND
        return Tier.UNKNOWN

    def is_founder(self, name: str) -> bool:
        return self.get_tier(name) == Tier.FOUNDER

    def is_trusted(self, name: str) -> bool:
        return self.get_tier(name) >= Tier.FRIEND

    def is_peer(self, name: str) -> bool:
        return name.strip().lower() in _PEERS

    def describe(self, name: str) -> str:
        key = name.strip().lower()
        tier = self.get_tier(name)
        if tier == Tier.FOUNDER:
            return f"{name} is the Founder — full authority."
        if key in _PEERS:
            return f"{name} is a pre-authorized Peer node — Friend tier."
        if tier == Tier.FRIEND:
            e = self._friends[key]
            return f"{name} is a Friend (introduced by {e.introduced_by})."
        return f"{name} is Unknown — restricted to basic conversation."

    # ── Mutation ─────────────────────────────────────────────────────────────

    def introduce(self, name: str, introduced_by: str, notes: str = "") -> bool:
        """Add a friend. Only a Founder can introduce. Returns True on success."""
        if not self.is_founder(introduced_by):
            log.warning("introduce() blocked — %r lacks Founder authority", introduced_by)
            return False
        key = name.strip().lower()
        self._friends[key] = TrustEntry(
            name=name, introduced_by=introduced_by, notes=notes
        )
        self._save()
        log.info("Friend added: %r (by %r)", name, introduced_by)
        return True

    def revoke(self, name: str, by: str) -> bool:
        """Remove a friend. Only a Founder can revoke. Returns True on success."""
        if not self.is_founder(by):
            log.warning("revoke() blocked — %r lacks Founder authority", by)
            return False
        key = name.strip().lower()
        if key in self._friends:
            del self._friends[key]
            self._save()
            log.info("Friend revoked: %r (by %r)", name, by)
            return True
        return False

    def list_friends(self) -> list[str]:
        return [e.name for e in self._friends.values()]

