"""
Body Completion Score (BCS)
============================
The single metric that prevents circular goal-setting.

Every plan, every skill, every action ARIA takes must advance at least
one of these four fields. If it doesn't — after 3 attempts — it's dropped.

Fields (0.0 – 1.0):
  locomotion_design      — viable movement strategy chosen + specced
  structural_integrity   — frame/body structure designed + buildable
  material_availability  — required materials identified + accessible
  actuation_coverage     — joints/motors tested and characterised

Skill failure tracking:
  Each skill has a consecutive-fail counter. Zero BCS progress on a call
  increments the counter. A successful BCS advance resets it.
  At 3 consecutive zero-progress calls → skill is tombstoned (auto-deleted).
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

log = logging.getLogger("x1.bcs")

BCS_PATH = Path(__file__).parent.parent / "memory" / "bcs.json"


@dataclass
class BCSState:
    locomotion_design:    float = 0.0
    structural_integrity: float = 0.0
    material_availability: float = 0.0
    actuation_coverage:   float = 0.0
    # skill_name -> consecutive zero-progress call count
    skill_fails: dict[str, int] = field(default_factory=dict)
    # skill_name -> tombstone reason
    tombstones: dict[str, str] = field(default_factory=dict)
    last_updated: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    @property
    def total(self) -> float:
        return round(
            (self.locomotion_design + self.structural_integrity +
             self.material_availability + self.actuation_coverage) / 4.0, 3
        )

    def summary(self) -> str:
        return (
            f"BCS {self.total:.1%} | "
            f"loco={self.locomotion_design:.0%} "
            f"struct={self.structural_integrity:.0%} "
            f"material={self.material_availability:.0%} "
            f"actuate={self.actuation_coverage:.0%}"
        )


class BCSTracker:
    """Persistent tracker for ARIA's Body Completion Score."""

    FAIL_LIMIT = 3

    def __init__(self, path: Path = BCS_PATH) -> None:
        self.path = path
        self.state = BCSState()
        self._load()

    # ── Persistence ────────────────────────────────────────────────────────────

    def _load(self) -> None:
        if self.path.exists():
            try:
                raw = json.loads(self.path.read_text())
                # Separate nested dicts from flat floats
                self.state = BCSState(
                    locomotion_design=raw.get("locomotion_design", 0.0),
                    structural_integrity=raw.get("structural_integrity", 0.0),
                    material_availability=raw.get("material_availability", 0.0),
                    actuation_coverage=raw.get("actuation_coverage", 0.0),
                    skill_fails=raw.get("skill_fails", {}),
                    tombstones=raw.get("tombstones", {}),
                    last_updated=raw.get("last_updated", ""),
                )
                log.info("BCS loaded — %s", self.state.summary())
            except Exception as exc:
                log.warning("BCS load failed: %s", exc)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.state.last_updated = datetime.now(timezone.utc).isoformat()
        self.path.write_text(json.dumps(asdict(self.state), indent=2))

    # ── Scoring ────────────────────────────────────────────────────────────────

    def advance(
        self,
        skill_name: str,
        *,
        locomotion_design: Optional[float] = None,
        structural_integrity: Optional[float] = None,
        material_availability: Optional[float] = None,
        actuation_coverage: Optional[float] = None,
    ) -> tuple[bool, str]:
        """
        Apply progress to one or more BCS fields from a skill result.
        Returns (progress_made: bool, summary: str).
        Values are clamped 0.0-1.0 and only advance (never regress).
        """
        if skill_name in self.state.tombstones:
            return False, f"[BCS] Skill '{skill_name}' is tombstoned: {self.state.tombstones[skill_name]}"

        progress = False
        for attr, val in [
            ("locomotion_design", locomotion_design),
            ("structural_integrity", structural_integrity),
            ("material_availability", material_availability),
            ("actuation_coverage", actuation_coverage),
        ]:
            if val is not None:
                clamped = max(0.0, min(1.0, float(val)))
                current = getattr(self.state, attr)
                if clamped > current:
                    setattr(self.state, attr, clamped)
                    progress = True

        if progress:
            self.state.skill_fails[skill_name] = 0
            log.info("BCS advance via '%s' — %s", skill_name, self.state.summary())
        else:
            fails = self.state.skill_fails.get(skill_name, 0) + 1
            self.state.skill_fails[skill_name] = fails
            log.info("BCS no progress for '%s' — fails=%d", skill_name, fails)

        self._save()
        return progress, self.state.summary()

    def should_tombstone(self, skill_name: str) -> bool:
        return self.state.skill_fails.get(skill_name, 0) >= self.FAIL_LIMIT

    def tombstone(self, skill_name: str, reason: str = "3 consecutive zero-progress calls") -> None:
        self.state.tombstones[skill_name] = reason
        self._save()
        log.warning("Skill tombstoned: '%s' — %s", skill_name, reason)

    def get_score(self) -> BCSState:
        return self.state

    def report(self) -> str:
        s = self.state
        lines = [
            f"=== BCS Report — {s.total:.1%} complete ===",
            f"  locomotion_design:    {s.locomotion_design:.0%}",
            f"  structural_integrity: {s.structural_integrity:.0%}",
            f"  material_availability:{s.material_availability:.0%}",
            f"  actuation_coverage:   {s.actuation_coverage:.0%}",
        ]
        if s.tombstones:
            lines.append(f"  tombstoned skills: {list(s.tombstones.keys())}")
        return "\n".join(lines)

