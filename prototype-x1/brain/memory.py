"""
Episodic memory for prototype-x1.
- Short-term: rolling list of (role, text) tuples (in-process)
- Long-term: ChromaDB vector store (optional, gracefully disabled if unavailable)
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
    """Short-term rolling buffer + optional ChromaDB long-term store."""

    def __init__(self, cfg: "MemoryConfig") -> None:
        self.cfg = cfg
        self._log: list[MemoryEntry] = []
        self._chroma = None
        self._collection = None
        if cfg.enabled:
            self._init_chroma()

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

    def add(self, role: str, text: str, **metadata) -> None:
        entry = MemoryEntry(role=role, text=text, metadata=metadata)
        self._log.append(entry)
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
            count = self._collection.count()
            if count == 0:
                return []
            results = self._collection.query(query_texts=[query], n_results=min(k, count))
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
        self._log.clear()

    @property
    def session_length(self) -> int:
        return len(self._log)
