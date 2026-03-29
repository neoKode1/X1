"""
Episodic memory for prototype-x1.
- Short-term: rolling list of (role, text) tuples (in-process)
- Long-term: ChromaDB vector store (optional, gracefully disabled if unavailable)
- Persistent: SQLite conversation store (survives restarts)
"""
from __future__ import annotations
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import MemoryConfig

log = logging.getLogger("x1.memory")


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
        self._store = None          # ConversationStore
        self._conversation_id = None  # active conversation ID
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
            log.info("ChromaDB ready at %s (%d entries)",
                     self.cfg.persist_dir, self._collection.count())
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

    def recall(self, query: str, top_k: int | None = None) -> list[MemoryEntry]:
        """Semantic search over long-term memory."""
        k = top_k or self.cfg.top_k
        if not self._collection or not query.strip():
            return []
        try:
            # Query directly — ChromaDB handles empty collections gracefully
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

    def recent(self, n: int = 20) -> list[MemoryEntry]:
        return self._log[-n:]

    def as_messages(self, n: int = 20) -> list[dict]:
        return [{"role": e.role, "content": e.text} for e in self.recent(n)]

    def clear_session(self) -> None:
        # End current conversation in DB, start a new one
        if self._store and self._conversation_id:
            try:
                self._store.end_conversation(self._conversation_id)
                self._conversation_id = self._store.start_conversation()
            except Exception as e:
                log.debug("SQLite session reset failed: %s", e)
        self._log.clear()

    @property
    def session_length(self) -> int:
        return len(self._log)
