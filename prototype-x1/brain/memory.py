"""
Episodic memory for prototype-x1.
- Short-term: rolling list of (role, text) tuples (in-process)
- Long-term: ChromaDB vector store (optional, gracefully disabled if unavailable)
  - x1_episodic: raw conversation turns (semantic recall)
  - x1_knowledge: extracted facts, preferences, learned info
- Persistent: SQLite conversation store (survives restarts)
- Knowledge extraction: after each turn, extract key facts via lightweight heuristics
"""
from __future__ import annotations
import json
import logging
import re
import time
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import MemoryConfig

log = logging.getLogger("x1.memory")

# ── Fact extraction patterns ────────────────────────────────────────────────
# These patterns detect statements that contain learnable facts
_FACT_PATTERNS = [
    # "my X is Y" / "I'm X" / "I am X"
    re.compile(r"\bmy\s+(name|website|company|email|project|app|site|domain|business|team)\b.{3,80}", re.I),
    re.compile(r"\bi(?:'m| am)\s+(?:a |an |the )?\w+", re.I),
    # URLs mentioned by user
    re.compile(r"https?://\S{5,}", re.I),
    # "I prefer/like/want/hate/need"
    re.compile(r"\bi\s+(?:prefer|like|want|hate|need|use|love|enjoy)\b.{3,80}", re.I),
    # "call me X" / "remember that"
    re.compile(r"\b(?:call me|remember that|keep in mind|don't forget)\b.{3,80}", re.I),
    # Project/tech mentions
    re.compile(r"\b(?:built with|using|runs on|deployed on|stack is)\b.{3,80}", re.I),
]


@dataclass
class MemoryEntry:
    role: str        # "user" | "assistant" | "system"
    text: str
    ts: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


class EpisodicMemory:
    """Short-term rolling buffer + optional ChromaDB long-term store + SQLite persistence."""

    def __init__(self, cfg: "MemoryConfig") -> None:
        self.cfg = cfg
        self._log: list[MemoryEntry] = []
        self._chroma = None
        self._collection = None
        self._knowledge = None       # x1_knowledge collection
        self._store = None           # ConversationStore
        self._conversation_id = None # active conversation ID
        self._turn_count = 0         # turns this session (for summary trigger)
        if cfg.enabled:
            self._init_chroma()
        # Persistence is always enabled (independent of ChromaDB)
        self._init_persistence()

    def _init_chroma(self) -> None:
        try:
            import chromadb
            Path(self.cfg.persist_dir).mkdir(parents=True, exist_ok=True)
            self._chroma = chromadb.PersistentClient(path=self.cfg.persist_dir)
            self._collection = self._chroma.get_or_create_collection(self.cfg.collection)
            self._knowledge = self._chroma.get_or_create_collection("x1_knowledge")
            log.info("ChromaDB ready at %s (episodic=%d, knowledge=%d)",
                     self.cfg.persist_dir, self._collection.count(),
                     self._knowledge.count())
        except Exception as e:
            log.warning("ChromaDB unavailable (%s) — long-term memory disabled", e)
            self._chroma = None

    def _init_persistence(self) -> None:
        """Initialize SQLite conversation store and resume last session."""
        try:
            from .persistence import ConversationStore
            self._store = ConversationStore(self.cfg.db_path)

            # Resume last conversation if it ended recently (< 30 min ago)
            last_id = self._store.get_last_conversation_id()
            if last_id:
                turns = self._store.get_turns(last_id)
                if turns:
                    last_ts = turns[-1]["ts"]
                    age_min = (time.time() - last_ts) / 60
                    if age_min < 30:
                        # Resume — reload only the last few turns to keep context small
                        # (full history stays in SQLite for browsing, but LLM only sees recent)
                        max_reload = min(len(turns), 4)  # max 4 turns (2 exchanges)
                        for t in turns[-max_reload:]:
                            self._log.append(MemoryEntry(
                                role=t["role"], text=t["content"], ts=t["ts"]
                            ))
                        self._conversation_id = last_id
                        log.info("Resumed conversation %s (%d/%d turns, %.0f min ago)",
                                 last_id[:8], max_reload, len(turns), age_min)
                        return

            # Start fresh conversation
            self._conversation_id = self._store.start_conversation()
        except Exception as e:
            log.warning("SQLite persistence unavailable (%s) — conversations won't persist", e)
            self._store = None

    def add(self, role: str, text: str, **metadata) -> None:
        entry = MemoryEntry(role=role, text=text, metadata=metadata)
        self._log.append(entry)
        self._turn_count += 1

        # Persist to SQLite
        if self._store and self._conversation_id and role != "system":
            try:
                self._store.add_turn(
                    self._conversation_id, role, text, entry.ts
                )
            except Exception as e:
                log.debug("SQLite add_turn failed: %s", e)

        # Persist to ChromaDB (semantic search)
        if self._collection and role != "system":
            try:
                self._collection.add(
                    documents=[text],
                    metadatas=[{"role": role, "ts": entry.ts, **metadata}],
                    ids=[f"{role}_{int(entry.ts * 1000)}"],
                )
            except Exception as e:
                log.debug("ChromaDB add failed: %s", e)

        # Extract knowledge from user messages (background thread)
        if role == "user" and self._knowledge:
            threading.Thread(
                target=self._extract_and_store_facts,
                args=(text, entry.ts),
                daemon=True,
            ).start()

    # ── Episodic recall ────────────────────────────────────────────────────────

    def recall(self, query: str, top_k: int | None = None) -> list[MemoryEntry]:
        """Semantic search over long-term episodic memory."""
        k = top_k or self.cfg.top_k
        if not self._collection or not query.strip():
            return []
        try:
            results = self._collection.query(query_texts=[query], n_results=k)
            entries = []
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                entries.append(MemoryEntry(
                    role=meta.get("role", "?"),
                    text=doc,
                    ts=meta.get("ts", 0.0),
                ))
            return entries
        except Exception as e:
            log.debug("ChromaDB query failed: %s", e)
            return []

    # ── Knowledge recall ───────────────────────────────────────────────────────

    def recall_knowledge(self, query: str, top_k: int = 3) -> list[MemoryEntry]:
        """Semantic search over extracted knowledge (facts, preferences)."""
        if not self._knowledge or not query.strip():
            return []
        try:
            count = self._knowledge.count()
            if count == 0:
                return []
            k = min(top_k, count)
            results = self._knowledge.query(query_texts=[query], n_results=k)
            entries = []
            for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
                entries.append(MemoryEntry(
                    role="knowledge",
                    text=doc,
                    ts=meta.get("ts", 0.0),
                    metadata=meta,
                ))
            return entries
        except Exception as e:
            log.debug("Knowledge recall failed: %s", e)
            return []

    # ── Fact extraction ────────────────────────────────────────────────────────

    def _extract_and_store_facts(self, text: str, ts: float) -> None:
        """Extract factual statements from user text and store in knowledge DB."""
        if len(text.split()) < 3:
            return  # too short to contain meaningful facts

        facts = []
        for pattern in _FACT_PATTERNS:
            matches = pattern.findall(text)
            if matches:
                # Store the full sentence context, not just the match
                facts.append(text.strip())
                break  # one fact per message is enough

        # Also detect explicit "remember" commands
        lower = text.lower()
        if any(kw in lower for kw in ("remember", "keep in mind", "don't forget", "note that")):
            facts.append(text.strip())

        for fact in facts:
            try:
                fact_id = f"fact_{int(ts * 1000)}"
                # Check for duplicate (similar fact already stored)
                existing = self._knowledge.query(
                    query_texts=[fact], n_results=1
                )
                if existing["documents"] and existing["documents"][0]:
                    # If very similar fact exists (same first 50 chars), skip
                    old = existing["documents"][0][0]
                    if old[:50].lower() == fact[:50].lower():
                        return

                self._knowledge.add(
                    documents=[fact],
                    metadatas=[{"ts": ts, "type": "user_fact", "source": "extraction"}],
                    ids=[fact_id],
                )
                log.info("Learned fact: %s", fact[:80])
            except Exception as e:
                log.debug("Knowledge store failed: %s", e)

    def learn_fact(self, fact: str, fact_type: str = "explicit") -> None:
        """Explicitly store a fact in knowledge memory."""
        if not self._knowledge:
            return
        try:
            ts = time.time()
            self._knowledge.add(
                documents=[fact],
                metadatas=[{"ts": ts, "type": fact_type, "source": "explicit"}],
                ids=[f"fact_{int(ts * 1000)}"],
            )
            log.info("Stored explicit fact: %s", fact[:80])
        except Exception as e:
            log.debug("learn_fact failed: %s", e)

    # ── Session summaries ──────────────────────────────────────────────────────

    def summarize_session(self) -> str | None:
        """Generate a summary of the current session for long-term storage.
        Returns the summary text, or None if session too short."""
        if self._turn_count < 4:
            return None  # too short to summarize

        # Build a condensed view of the session
        turns = self._log[-20:]  # last 20 turns max
        lines = []
        for e in turns:
            if e.role == "system":
                continue
            prefix = "User" if e.role == "user" else "ARIA"
            lines.append(f"{prefix}: {e.text[:200]}")

        session_text = "\n".join(lines)
        # Store as a knowledge entry for future recall
        summary = f"Session summary ({len(turns)} turns): " + session_text[:800]

        if self._knowledge:
            try:
                ts = time.time()
                self._knowledge.add(
                    documents=[summary],
                    metadatas=[{
                        "ts": ts, "type": "session_summary",
                        "source": "auto", "turn_count": self._turn_count,
                        "conversation_id": self._conversation_id or "unknown",
                    }],
                    ids=[f"summary_{int(ts * 1000)}"],
                )
                log.info("Session summary stored (%d turns)", self._turn_count)
            except Exception as e:
                log.debug("Session summary store failed: %s", e)

        # Also store in SQLite conversation summary field
        if self._store and self._conversation_id:
            try:
                self._store.update_summary(self._conversation_id, summary)
            except Exception as e:
                log.debug("SQLite summary update failed: %s", e)

        return summary

    # ── Standard accessors ─────────────────────────────────────────────────────

    def recent(self, n: int = 20) -> list[MemoryEntry]:
        return self._log[-n:]

    def as_messages(self, n: int = 20) -> list[dict]:
        return [{"role": e.role, "content": e.text} for e in self.recent(n)]

    def clear_session(self) -> None:
        # Summarize before clearing
        self.summarize_session()
        # End current conversation in DB, start a new one
        if self._store and self._conversation_id:
            try:
                self._store.end_conversation(self._conversation_id)
                self._conversation_id = self._store.start_conversation()
            except Exception as e:
                log.debug("SQLite session reset failed: %s", e)
        self._log.clear()
        self._turn_count = 0

    @property
    def session_length(self) -> int:
        return len(self._log)

    @property
    def knowledge_count(self) -> int:
        """Number of learned facts in knowledge store."""
        if self._knowledge:
            try:
                return self._knowledge.count()
            except Exception:
                pass
        return 0
