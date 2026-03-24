"""
Developer notes skill — lets ARIA leave persistent notes for the developer.

Auto-registered as: dev_note, read_dev_notes, clear_dev_notes
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

log = logging.getLogger("x1.dev_notes")

NOTES_FILE = Path(__file__).parent.parent / "memory" / "dev_notes.json"


def _load_notes() -> list[dict]:
    if NOTES_FILE.exists():
        try:
            return json.loads(NOTES_FILE.read_text())
        except Exception:
            return []
    return []


def _save_notes(notes: list[dict]) -> None:
    NOTES_FILE.parent.mkdir(parents=True, exist_ok=True)
    NOTES_FILE.write_text(json.dumps(notes, indent=2))


def skill_dev_note(note: str, category: str = "general") -> str:
    """Write a note for the developer. ARIA uses this to communicate
    what she needs help with — feature requests, bugs, observations."""
    entry = {
        "ts": time.time(),
        "category": category,
        "note": note,
        "read": False,
    }
    notes = _load_notes()
    notes.append(entry)
    _save_notes(notes)
    log.info("Dev note saved: [%s] %s", category, note[:80])
    return f"Note saved for developer: [{category}] {note}"


def skill_read_dev_notes(unread_only: str = "true") -> str:
    """Read all developer notes. Pass unread_only='false' to see all."""
    notes = _load_notes()
    if not notes:
        return "No developer notes yet."
    show_unread = unread_only.lower() != "false"
    filtered = [n for n in notes if not n.get("read")] if show_unread else notes
    if not filtered:
        return "No unread developer notes."
    lines = []
    for i, n in enumerate(filtered):
        ts = time.strftime("%Y-%m-%d %H:%M", time.localtime(n["ts"]))
        lines.append(f"[{i+1}] [{n['category']}] {ts} — {n['note']}")
    # Mark shown notes as read
    for n in filtered:
        n["read"] = True
    _save_notes(notes)
    return "\n".join(lines)


def skill_clear_dev_notes() -> str:
    """Clear all developer notes."""
    _save_notes([])
    return "All developer notes cleared."

