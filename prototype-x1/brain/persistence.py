"""
SQLite conversation persistence for ARIA.
Stores conversation sessions and individual turns so history survives restarts.
Uses SQLAlchemy + SQLite — zero external services required.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from sqlalchemy import (
    Column, DateTime, Float, ForeignKey, Integer, String, Text,
    create_engine,
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session

log = logging.getLogger("x1.persistence")

Base = declarative_base()


# ── Models ────────────────────────────────────────────────────────────────────

class Conversation(Base):
    """A single conversation session (one per ARIA boot or manual reset)."""
    __tablename__ = "conversations"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    started_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    ended_at = Column(DateTime, nullable=True)
    summary = Column(Text, nullable=True)

    turns = relationship("Turn", back_populates="conversation",
                         order_by="Turn.seq", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<Conversation {self.id[:8]} started={self.started_at}>"


class Turn(Base):
    """One turn in a conversation (user or assistant message)."""
    __tablename__ = "turns"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    conversation_id = Column(String, ForeignKey("conversations.id"), nullable=False)
    seq = Column(Integer, nullable=False)  # ordering within conversation
    role = Column(String, nullable=False)  # "user" | "assistant" | "system"
    content = Column(Text, nullable=False)
    timestamp = Column(Float, nullable=False)  # epoch seconds
    metadata_json = Column(Text, nullable=True)  # optional JSON blob

    conversation = relationship("Conversation", back_populates="turns")

    def __repr__(self) -> str:
        return f"<Turn seq={self.seq} role={self.role} len={len(self.content)}>"


# ── Database manager ──────────────────────────────────────────────────────────

class ConversationStore:
    """Manages SQLite-backed conversation persistence."""

    def __init__(self, db_path: str | Path) -> None:
        db_path = Path(db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self._engine = create_engine(
            f"sqlite:///{db_path}",
            connect_args={"check_same_thread": False},
        )
        Base.metadata.create_all(bind=self._engine)
        self._Session = sessionmaker(bind=self._engine)

        log.info("ConversationStore ready at %s", db_path)

    # ── Session helpers ───────────────────────────────────────────────────────

    def _db(self) -> Session:
        return self._Session()

    # ── Conversation lifecycle ────────────────────────────────────────────────

    def start_conversation(self) -> str:
        """Create a new conversation, return its ID."""
        db = self._db()
        try:
            conv = Conversation()
            db.add(conv)
            db.commit()
            cid = conv.id
            log.info("Started conversation %s", cid[:8])
            return cid
        finally:
            db.close()

    def end_conversation(self, conversation_id: str) -> None:
        """Mark a conversation as ended."""
        db = self._db()
        try:
            conv = db.query(Conversation).get(conversation_id)
            if conv:
                conv.ended_at = datetime.now(timezone.utc)
                db.commit()
        finally:
            db.close()

    def update_summary(self, conversation_id: str, summary: str) -> None:
        """Store a session summary on the conversation record."""
        db = self._db()
        try:
            conv = db.query(Conversation).get(conversation_id)
            if conv:
                conv.summary = summary
                db.commit()
        finally:
            db.close()

    # ── Turn management ───────────────────────────────────────────────────────

    def add_turn(self, conversation_id: str, role: str, content: str,
                 timestamp: float, metadata_json: Optional[str] = None) -> None:
        """Append a turn to an existing conversation."""
        db = self._db()
        try:
            # Get next seq number
            max_seq = (
                db.query(Turn.seq)
                .filter(Turn.conversation_id == conversation_id)
                .order_by(Turn.seq.desc())
                .first()
            )
            next_seq = (max_seq[0] + 1) if max_seq else 0

            turn = Turn(
                conversation_id=conversation_id,
                seq=next_seq,
                role=role,
                content=content,
                timestamp=timestamp,
                metadata_json=metadata_json,
            )
            db.add(turn)
            db.commit()
        finally:
            db.close()

    def get_turns(self, conversation_id: str) -> list[dict]:
        """Return all turns for a conversation as dicts."""
        db = self._db()
        try:
            turns = (
                db.query(Turn)
                .filter(Turn.conversation_id == conversation_id)
                .order_by(Turn.seq)
                .all()
            )
            return [
                {"role": t.role, "content": t.content, "ts": t.timestamp}
                for t in turns
            ]
        finally:
            db.close()

    def get_last_conversation_id(self, only_open: bool = True) -> Optional[str]:
        """Return the most recent conversation ID, or None.

        Args:
            only_open: If True, only return conversations that haven't been
                       explicitly ended (ended_at is NULL).
        """
        db = self._db()
        try:
            q = db.query(Conversation)
            if only_open:
                q = q.filter(Conversation.ended_at.is_(None))
            conv = q.order_by(Conversation.started_at.desc()).first()
            return conv.id if conv else None
        finally:
            db.close()

