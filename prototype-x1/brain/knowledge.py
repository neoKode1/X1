"""
Knowledge Base (RAG) for prototype-x1.
Ingests documents from a drop folder into a dedicated ChromaDB collection.
Chunks text into ~500-char segments with overlap for better retrieval.
"""
from __future__ import annotations
import hashlib
import logging
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

log = logging.getLogger("x1.knowledge")

KNOWLEDGE_DIR = Path(__file__).parent.parent / "knowledge"
_CHUNK_SIZE = 500       # chars per chunk
_CHUNK_OVERLAP = 80     # overlap between chunks
_SUPPORTED_EXT = {".txt", ".md", ".json", ".csv", ".py", ".log"}


@dataclass
class KnowledgeDoc:
    """Metadata for an ingested document."""
    filename: str
    doc_id: str
    chunks: int
    size_bytes: int


class KnowledgeStore:
    """Dedicated ChromaDB collection for static knowledge / reference docs."""

    def __init__(self, persist_dir: str, collection: str = "x1_knowledge") -> None:
        self._chroma = None
        self._collection = None
        self._collection_name = collection
        try:
            import chromadb
            Path(persist_dir).mkdir(parents=True, exist_ok=True)
            self._chroma = chromadb.PersistentClient(path=persist_dir)
            self._collection = self._chroma.get_or_create_collection(collection)
            log.info("Knowledge store ready (%d chunks)", self._collection.count())
        except Exception as e:
            log.warning("Knowledge store unavailable: %s", e)

    # ── Chunking ──────────────────────────────────────────────────────────────

    @staticmethod
    def _chunk_text(text: str, size: int = _CHUNK_SIZE,
                    overlap: int = _CHUNK_OVERLAP) -> list[str]:
        """Split text into overlapping chunks."""
        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = start + size
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - overlap
        return chunks

    @staticmethod
    def _file_hash(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]

    # ── Ingestion ─────────────────────────────────────────────────────────────

    def ingest_file(self, path: Path, source: str = "upload") -> KnowledgeDoc | None:
        """Read a single file, chunk it, and add to ChromaDB."""
        if not self._collection:
            log.warning("Knowledge store not available — skipping %s", path.name)
            return None
        if path.suffix.lower() not in _SUPPORTED_EXT:
            log.info("Skipping unsupported file type: %s", path.name)
            return None

        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            log.error("Failed to read %s: %s", path.name, e)
            return None

        if not text.strip():
            return None

        doc_id = self._file_hash(path)
        chunks = self._chunk_text(text)

        # Remove old version of this doc if re-ingesting
        self._remove_doc_chunks(doc_id)

        ids = [f"kb_{doc_id}_{i}" for i in range(len(chunks))]
        metadatas = [{"filename": path.name, "doc_id": doc_id,
                      "chunk_idx": i, "source": source}
                     for i in range(len(chunks))]

        self._collection.add(documents=chunks, metadatas=metadatas, ids=ids)
        log.info("Ingested '%s' → %d chunks (doc_id=%s)", path.name, len(chunks), doc_id)
        return KnowledgeDoc(filename=path.name, doc_id=doc_id,
                            chunks=len(chunks), size_bytes=len(text))

    def ingest_folder(self, folder: Path | None = None) -> list[KnowledgeDoc]:
        """Ingest all supported files from the knowledge drop folder."""
        folder = folder or KNOWLEDGE_DIR
        folder.mkdir(parents=True, exist_ok=True)
        results: list[KnowledgeDoc] = []
        for f in sorted(folder.iterdir()):
            if f.is_file() and not f.name.startswith("."):
                doc = self.ingest_file(f, source="folder")
                if doc:
                    results.append(doc)
        return results

    def _remove_doc_chunks(self, doc_id: str) -> int:
        """Remove all chunks for a given doc_id. Returns count removed."""
        if not self._collection:
            return 0
        try:
            existing = self._collection.get(where={"doc_id": doc_id})
            if existing["ids"]:
                self._collection.delete(ids=existing["ids"])
                return len(existing["ids"])
        except Exception:
            pass
        return 0

    # ── Search ────────────────────────────────────────────────────────────────

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Semantic search over knowledge base. Returns list of {text, filename, score}."""
        if not self._collection or not query.strip():
            return []
        try:
            count = self._collection.count()
            if count == 0:
                return []
            results = self._collection.query(
                query_texts=[query], n_results=min(top_k, count))
            hits: list[dict] = []
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            dists = results.get("distances", [[]])[0]
            for doc, meta, dist in zip(docs, metas, dists):
                hits.append({
                    "text": doc,
                    "filename": meta.get("filename", "?"),
                    "doc_id": meta.get("doc_id", "?"),
                    "score": round(1.0 - dist, 4) if dist < 1 else 0.0,
                })
            return hits
        except Exception as e:
            log.debug("Knowledge search failed: %s", e)
            return []

    # ── Listing ───────────────────────────────────────────────────────────────

    def list_docs(self) -> list[dict]:
        """List all unique documents in the knowledge base."""
        if not self._collection:
            return []
        try:
            all_meta = self._collection.get()["metadatas"]
            seen: dict[str, dict] = {}
            for m in all_meta:
                did = m.get("doc_id", "?")
                if did not in seen:
                    seen[did] = {"doc_id": did, "filename": m.get("filename", "?"), "chunks": 0}
                seen[did]["chunks"] += 1
            return list(seen.values())
        except Exception:
            return []

    def remove_doc(self, doc_id: str) -> int:
        """Remove a document by doc_id. Returns chunks removed."""
        removed = self._remove_doc_chunks(doc_id)
        log.info("Removed doc %s (%d chunks)", doc_id, removed)
        return removed

    @property
    def count(self) -> int:
        return self._collection.count() if self._collection else 0

