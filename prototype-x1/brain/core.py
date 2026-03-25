"""
prototype-x1 Brain Core
=======================
Reasoning loop:
  1. Surface relevant memories
  2. Build context (system + memories + rolling chat)
  3. Call LLM (Ollama -> cloud fallback)
  4. Parse response for SKILL calls
  5. Execute skills, feed results back, re-reason if needed
  6. Persist turn to memory
"""
from __future__ import annotations
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import BrainConfig
from .llm import call_llm, stream_llm, Message
from .memory import EpisodicMemory
from .knowledge import KnowledgeStore
from . import skills as skill_registry
from .trust import TrustRegistry, Tier
from .bcs import BCSTracker

log = logging.getLogger("x1.core")

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = "You are ARIA, a cyberpunk robot. Neokode is your founder. Answer in 1-2 sentences."

SYSTEM_PROMPT_WITH_SKILLS = """You are ARIA, a cyberpunk robot. Neokode is your founder.
RULES: 1) Casual talk=short answer, NO skills. 2) 1-2 sentences max. 3) Max 2 skills, only when asked.
Skill format: ```skill\n{{"name":"X","args":{{}}}}\n```
Skills: {skill_list}"""

# Keywords that suggest the user wants a skill executed
_SKILL_TRIGGERS = frozenset({
    "show", "run", "execute", "grab", "screenshot", "screengrab", "capture",
    "file", "read", "write", "list", "dir", "shell", "python", "fetch",
    "bcs", "build", "report", "scrap", "led", "move", "servo", "look",
    "note", "dev_note", "friend", "introduce", "nickname", "webcam",
})

# Primary: fenced ```skill {...} ``` block (preferred format)
SKILL_BLOCK_RE = re.compile(r"```skill\s*(\{.*?\})\s*```", re.DOTALL)
# Catch brace-shorthand: {skillname {"arg": "val"}} or {skillname}
SKILL_BRACE_RE = re.compile(r'\{(\w+)\s*(\{[^}]*\})?\s*\}')
# Catch asterisk-prefix: *screengrab {"source": "webcam"}} or *skillname {args}
SKILL_ASTERISK_RE = re.compile(r'\*(\w+)\s*(\{[^}]*\})')


@dataclass
class BrainResponse:
    text: str
    skill_calls: list[dict] = field(default_factory=list)
    skill_results: list[str] = field(default_factory=list)
    provider: str = "unknown"
    latency_ms: int = 0


class Brain:
    def __init__(self, cfg: BrainConfig | None = None) -> None:
        self.cfg = cfg or BrainConfig()
        self.memory = EpisodicMemory(self.cfg.memory)
        self.knowledge = KnowledgeStore(persist_dir=self.cfg.memory.persist_dir)
        self.trust = TrustRegistry()
        self.bcs = BCSTracker()
        self._active_speaker: str = "unknown"
        # Founder pattern store: list of {"ts", "signal", "context"} dicts
        self._founder_patterns: list[dict] = []
        skill_registry.register_builtins()
        self._load_external_skills()
        self._register_trust_skills()
        self._register_bcs_skills()
        # Hardware awareness — probed once at boot, injected as context every turn
        self._hw_webcam: bool = self._probe_webcam()
        log.info("Brain ready — name=%s ollama=%s/%s fallback=%s | BCS=%s webcam=%s",
                 self.cfg.name, self.cfg.llm.ollama_host,
                 self.cfg.llm.ollama_model, self.cfg.llm.cloud_fallback,
                 self.bcs.state.summary(), self._hw_webcam)

    @staticmethod
    def _probe_webcam() -> bool:
        """Check if a webcam is physically available. Silent — never raises."""
        try:
            import sys
            from pathlib import Path as _Path
            _root = str(_Path(__file__).parent.parent)
            if _root not in sys.path:
                sys.path.insert(0, _root)
            from vision.camera import capture_webcam  # type: ignore[import]
            return capture_webcam() is not None
        except Exception:
            return False

    def _load_external_skills(self) -> None:
        skills_dir = Path(__file__).parent.parent / "skills"
        if skills_dir.exists():
            skill_registry.load_skills_dir(skills_dir)

    def _register_trust_skills(self) -> None:
        """Register trust management skills bound to this brain's trust registry."""
        trust = self.trust

        def introduce_friend(name: str, notes: str = "") -> str:
            speaker = self._active_speaker
            ok = trust.introduce(name, introduced_by=speaker, notes=notes)
            if ok:
                return f"✓ {name} has been added as a Friend. They are now trusted."
            return (f"✗ Only Neokode can introduce friends. "
                    f"{speaker!r} does not have that authority.")

        def revoke_friend(name: str) -> str:
            speaker = self._active_speaker
            ok = trust.revoke(name, by=speaker)
            if ok:
                return f"✓ {name}'s friend status has been revoked."
            return (f"✗ Only Neokode can revoke trust. "
                    f"{speaker!r} does not have that authority.")

        def list_friends() -> str:
            friends = trust.list_friends()
            if not friends:
                return "No friends registered yet."
            return "Friends: " + ", ".join(friends)

        def set_nickname(identity: str, nickname: str) -> str:
            """Set a preferred name for someone. Use when they say 'call me X'."""
            trust.set_nickname(identity, nickname)
            return f"✓ Will now call {identity} by their preferred name: {nickname}"

        skill_registry.register("introduce_friend", introduce_friend, override=True)
        skill_registry.register("revoke_friend",    revoke_friend,    override=True)
        skill_registry.register("list_friends",     list_friends,     override=True)
        skill_registry.register("set_nickname",     set_nickname,     override=True)

    def _register_bcs_skills(self) -> None:
        """Register BCS query and update skills."""
        bcs = self.bcs

        def bcs_report() -> str:
            return bcs.report()

        def bcs_advance(
            locomotion_design: float | None = None,
            structural_integrity: float | None = None,
            material_availability: float | None = None,
            actuation_coverage: float | None = None,
            skill_name: str = "manual",
        ) -> str:
            _, summary = bcs.advance(
                skill_name,
                locomotion_design=locomotion_design,
                structural_integrity=structural_integrity,
                material_availability=material_availability,
                actuation_coverage=actuation_coverage,
            )
            return summary

        def scrap_report() -> str:
            try:
                from ..vision.material_tagger import MaterialTagger, SCRAP_PATH
                t = MaterialTagger(scrap_path=SCRAP_PATH)
                return t.report()
            except Exception as exc:
                return f"[scrap_report] {exc}"

        skill_registry.register("bcs_report",   bcs_report,   override=True)
        skill_registry.register("bcs_advance",  bcs_advance,  override=True)
        skill_registry.register("scrap_report", scrap_report, override=True)

    # ── Founder pattern detection ──────────────────────────────────────────────

    _FOUNDER_SIGNALS = {
        "wait, wait":  "slow_down",
        "wait wait":   "slow_down",
        "start over":  "reset",
        "clean slate": "reset",
        "drop it":     "abandon_goal",
        "delete it":   "abandon_goal",
        "good":        "approval",
        "that's it":   "approval",
    }

    def _detect_founder_patterns(self, text: str) -> None:
        """Scan input for known founder tone signals and store them."""
        lower = text.lower()
        for phrase, signal in self._FOUNDER_SIGNALS.items():
            if phrase in lower:
                entry = {
                    "ts": time.time(),
                    "signal": signal,
                    "context": text[:120],
                }
                self._founder_patterns.append(entry)
                log.info("Founder signal detected: %s (%r)", signal, phrase)
                # Persist as a memory note so ARIA carries it across sessions
                self.memory.add(
                    "system",
                    f"[FOUNDER_SIGNAL] {signal} — context: {text[:80]}"
                )

    # Skills exempt from BCS fail-tracking (observation / utility / management)
    _BCS_EXEMPT = frozenset({
        "introduce_friend", "revoke_friend", "list_friends", "set_nickname",
        "bcs_report", "bcs_advance", "scrap_report",
        "speak", "led", "dev_note", "read_dev_notes", "clear_dev_notes",
        # Observation & utility — these don't advance build progress by design
        "screengrab", "python", "shell", "read_file", "write_file", "list_dir",
    })

    def _build_messages(self, user_input: str, recalled: list,
                        speaker: str = "unknown") -> list[Message]:
        # Decide if user wants a skill — use short prompt for casual chat
        words = set(user_input.lower().split())
        needs_skills = bool(words & _SKILL_TRIGGERS)

        if needs_skills:
            skill_list = ", ".join(skill_registry.list_skills()) or "none loaded"
            system = SYSTEM_PROMPT_WITH_SKILLS.format(skill_list=skill_list)
        else:
            system = SYSTEM_PROMPT

        # Vision mood (one line, only if fresh)
        vision_frame: str | None = None  # base64 JPEG for native vision
        if hasattr(self, '_vision_state') and self._vision_state:
            vs = self._vision_state
            if time.time() - vs.get("ts", 0) < 10:
                mood = getattr(self, '_vision_mood', 'neutral')
                system += f" User looks: {mood}."
                _lower = user_input.lower()
                _vision_triggers = ("see", "look", "watch", "show", "camera",
                                    "webcam", "face", "screen", "what do you",
                                    "what am i", "how do i look", "can you see")
                if any(t in _lower for t in _vision_triggers):
                    vision_frame = getattr(self, '_vision_frame', None)

        messages: list[Message] = [{"role": "system", "content": system}]

        # Recalled memories — limit to 1 most relevant (keep context small for 8B)
        if recalled:
            mem_block = "\n".join(
                f"[{e.role}] {e.text[:120]}" for e in recalled[:1]
            )
            messages.append({"role": "system", "content": mem_block})

        messages.extend(self.memory.as_messages(self.cfg.llm.context_window))

        # User message — attach webcam frame if available (native vision)
        user_msg: Message = {"role": "user", "content": user_input}
        if vision_frame:
            user_msg["images"] = [vision_frame]
            log.debug("Attached webcam frame to user message for native vision")
        messages.append(user_msg)
        return messages

    @staticmethod
    def _extract_balanced_json(text: str) -> list[str]:
        """Extract top-level balanced {...} substrings from text."""
        results = []
        i = 0
        while i < len(text):
            if text[i] == '{':
                depth = 0
                start = i
                in_str = False
                escape = False
                for j in range(i, len(text)):
                    ch = text[j]
                    if escape:
                        escape = False
                        continue
                    if ch == '\\' and in_str:
                        escape = True
                        continue
                    if ch == '"' and not escape:
                        in_str = not in_str
                        continue
                    if in_str:
                        continue
                    if ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            results.append(text[start:j + 1])
                            i = j
                            break
                else:
                    break  # unbalanced — stop
            i += 1
        return results

    def _extract_skill_calls(self, text: str) -> list[dict]:
        calls = []
        seen_json: set[str] = set()

        def _add(raw: str) -> None:
            raw = raw.strip()
            if raw in seen_json:
                return
            seen_json.add(raw)
            try:
                obj = json.loads(raw)
                # Normalise "skill" key → "name" (llama3.2 uses {"skill": ...})
                if "skill" in obj and "name" not in obj:
                    obj["name"] = obj.pop("skill")
                if obj.get("name"):
                    calls.append(obj)
            except json.JSONDecodeError:
                pass

        # 1. Preferred: fenced ```skill {...} ``` blocks
        for match in SKILL_BLOCK_RE.finditer(text):
            _add(match.group(1))

        # 2. Fallback: balanced JSON objects with "name" or "skill" key
        if not calls:
            for blob in self._extract_balanced_json(text):
                if '"name"' in blob or '"skill"' in blob:
                    _add(blob)

        # 3. Brace-shorthand {skillname {args}} the model keeps emitting
        if not calls:
            known = set(skill_registry.list_skills())
            for match in SKILL_BRACE_RE.finditer(text):
                name = match.group(1)
                if name not in known:
                    continue
                args_raw = match.group(2)
                try:
                    args = json.loads(args_raw) if args_raw else {}
                except json.JSONDecodeError:
                    args = {}
                obj_str = json.dumps({"name": name, "args": args})
                _add(obj_str)

        # 4. Asterisk-prefix *skillname {args} — llama3.2 hallucination pattern
        if not calls:
            known = set(skill_registry.list_skills())
            for match in SKILL_ASTERISK_RE.finditer(text):
                name = match.group(1)
                if name not in known:
                    continue
                args_raw = match.group(2)
                try:
                    args = json.loads(args_raw)
                except json.JSONDecodeError:
                    args = {}
                obj_str = json.dumps({"name": name, "args": args})
                _add(obj_str)
                log.warning("Caught asterisk-syntax skill call: *%s — model needs prompt reinforcement", name)

        return calls

    _MAX_SKILLS_PER_TURN = 3          # hard ceiling across ALL rounds

    def _execute_skills(self, calls: list[dict], budget: int | None = None) -> list[str]:
        cap = budget if budget is not None else self._MAX_SKILLS_PER_TURN
        if len(calls) > cap:
            log.warning("Skill cap: %d calls → trimmed to %d", len(calls), cap)
            calls[:] = calls[:cap]            # mutate so caller sees trimmed list
        results = []
        for call in calls:
            name = call.get("name", "")
            args = call.get("args", {})

            # Skip tombstoned skills outright
            if name in self.bcs.state.tombstones:
                reason = self.bcs.state.tombstones[name]
                results.append(f"[{name}] TOMBSTONED — {reason}")
                continue

            log.info("Executing skill: %s %s", name, args)
            result = skill_registry.call(name, **args)
            results.append(f"[{name}] -> {result}")

            # BCS fail tracking: exempt management skills
            if name not in self._BCS_EXEMPT:
                progress, _ = self.bcs.advance(name)
                if not progress and self.bcs.should_tombstone(name):
                    self.bcs.tombstone(name)
                    # Remove from live registry so ARIA can't call it again
                    try:
                        skill_registry.unregister(name)
                        log.warning("Skill '%s' auto-deleted after 3 zero-progress calls", name)
                    except Exception:
                        pass

        return results

    def process(self, user_input: str, max_skill_rounds: int = 3,
                speaker: str = "unknown") -> BrainResponse:
        """Full reasoning turn with skill execution loop."""
        self._active_speaker = speaker
        t0 = time.time()

        # Detect founder tone/pattern signals before reasoning
        if self.trust.get_tier(speaker).value >= 8:
            self._detect_founder_patterns(user_input)

        recalled = self.memory.recall(user_input)
        messages = self._build_messages(user_input, recalled, speaker)
        reply, provider = call_llm(self.cfg.llm, messages)

        all_skill_calls: list[dict] = []
        all_skill_results: list[str] = []

        budget = self._MAX_SKILLS_PER_TURN
        for _round in range(max_skill_rounds):
            if budget <= 0:
                break
            calls = self._extract_skill_calls(reply)
            if not calls:
                break
            results = self._execute_skills(calls, budget=budget)
            budget -= len(calls)              # calls was trimmed in-place
            all_skill_calls.extend(calls)
            all_skill_results.extend(results)

            results_text = "\n".join(results)
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user",
                              "content": f"[SKILL RESULTS]\n{results_text}\n\nContinue."})
            reply, provider = call_llm(self.cfg.llm, messages)

        self.memory.add("user", user_input)
        if reply.strip():
            self.memory.add("assistant", reply)

        latency = int((time.time() - t0) * 1000)
        log.info("Turn complete in %dms via %s — skills=%d", latency, provider, len(all_skill_calls))

        return BrainResponse(
            text=reply,
            skill_calls=all_skill_calls,
            skill_results=all_skill_results,
            provider=provider,
            latency_ms=latency,
        )

    def stream(self, user_input: str, max_skill_rounds: int = 3,
               speaker: str = "unknown",
               stop_event: "threading.Event | None" = None):
        """Sync generator that yields streaming events for the server to relay.

        Yields:
            ("thinking", None)
            ("token", str)          -- one per LLM token
            ("action", dict)        -- one per skill execution
            ("cancelled", None)     -- if stop_event fired mid-stream
            ("done", BrainResponse)

        Args:
            stop_event: if set, the stream aborts early so the server can
                        start processing new input without waiting.
        """
        import threading as _threading

        self._active_speaker = speaker
        t0 = time.time()
        yield ("thinking", None)

        if stop_event and stop_event.is_set():
            yield ("cancelled", None)
            return

        # Detect founder tone/pattern signals before reasoning
        if self.trust.get_tier(speaker).value >= 8:
            self._detect_founder_patterns(user_input)

        # ── Memory recall — skip for short casual inputs ────────────
        recalled: list = []
        word_count = len(user_input.split())
        if word_count >= 4:
            # Only query ChromaDB for substantive inputs
            try:
                recalled = self.memory.recall(user_input)
            except Exception:
                log.debug("Memory recall failed — proceeding without")
        else:
            log.debug("Short input (%d words) — skipping memory recall", word_count)

        messages = self._build_messages(user_input, recalled, speaker)

        # Stream first LLM reply token-by-token
        full_text = ""
        provider = "unknown"
        token_count = 0
        for token, prov in stream_llm(self.cfg.llm, messages):
            if stop_event and stop_event.is_set():
                # Save partial progress so conversation resumes naturally
                if full_text.strip():
                    self.memory.add("user", user_input)
                    self.memory.add("assistant", full_text)
                    log.info("Paused — saved partial response (%d chars) to memory", len(full_text))
                yield ("cancelled", None)
                return
            full_text += token
            provider = prov
            token_count += 1
            yield ("token", token)
        log.info("LLM produced %d tokens, %d chars via %s — text[:200]: %r",
                 token_count, len(full_text), provider, full_text[:200])

        all_skill_calls: list[dict] = []
        all_skill_results: list[str] = []

        # Skill execution rounds (non-streaming — skills are fast)
        budget = self._MAX_SKILLS_PER_TURN
        for _round in range(max_skill_rounds):
            if budget <= 0:
                break
            if stop_event and stop_event.is_set():
                if full_text.strip():
                    self.memory.add("user", user_input)
                    self.memory.add("assistant", full_text)
                    log.info("Paused mid-skill — saved partial response to memory")
                yield ("cancelled", None)
                return
            calls = self._extract_skill_calls(full_text)
            if not calls:
                break
            results = self._execute_skills(calls, budget=budget)
            budget -= len(calls)
            all_skill_calls.extend(calls)
            all_skill_results.extend(results)

            for call, result in zip(calls, results):
                params = {"name": call.get("name", "unknown"), **call.get("args", {})}

                # Emit vision event if skill returned a vision payload
                try:
                    vision_data = json.loads(result)
                    if isinstance(vision_data, dict) and vision_data.get("__vision__"):
                        yield ("vision", {
                            "frame_b64": vision_data["frame_b64"],
                            "source": vision_data["source"],
                            "description": f"Captured via {call.get('name','screengrab')}",
                        })
                        result = f"[VISION] {vision_data['source']} {vision_data['width']}x{vision_data['height']}"
                except (json.JSONDecodeError, KeyError, TypeError):
                    pass

                yield ("action", {
                    "type": "SKILL",
                    "params": params,
                    "result": result,
                })

            # Feed skill results back — stream the continuation
            results_text = "\n".join(results)
            messages.append({"role": "assistant", "content": full_text})
            messages.append({"role": "user",
                              "content": f"[SKILL RESULTS]\n{results_text}\n\nContinue."})
            full_text = ""
            for token, prov in stream_llm(self.cfg.llm, messages):
                if stop_event and stop_event.is_set():
                    if full_text.strip():
                        self.memory.add("user", user_input)
                        self.memory.add("assistant", full_text)
                        log.info("Paused mid-continuation — saved partial response to memory")
                    yield ("cancelled", None)
                    return
                full_text += token
                provider = prov
                yield ("token", token)

        # ── Background memory write — don't block the response ────────────
        def _bg_memorize():
            self.memory.add("user", user_input)
            if full_text.strip():
                self.memory.add("assistant", full_text)
            else:
                log.warning("Skipping empty assistant response — not saving to memory")

        _threading.Thread(target=_bg_memorize, daemon=True).start()

        latency = int((time.time() - t0) * 1000)
        log.info("Stream turn complete in %dms via %s — skills=%d",
                 latency, provider, len(all_skill_calls))

        yield ("done", BrainResponse(
            text=full_text,
            skill_calls=all_skill_calls,
            skill_results=all_skill_results,
            provider=provider,
            latency_ms=latency,
        ))

    def accept_peer_handshake(self, payload: dict) -> dict:
        """
        Validate a peer handshake and return a sync packet.

        On success returns:
          {"ok": True,  "peer_id": str, "bcs": str, "scrap": str}
        On failure returns:
          {"ok": False, "reason": str}
        """
        from .peer import verify_handshake, PEER_ID as DEFAULT_PEER

        ok, reason = verify_handshake(payload)
        if not ok:
            log.warning("Peer handshake rejected: %s", reason)
            return {"ok": False, "reason": reason}

        peer_id = payload.get("peer_id", DEFAULT_PEER)

        # Confirm the peer is in the pre-authorized set
        if not self.trust.is_peer(peer_id):
            log.warning("Peer handshake rejected: %r not in _PEERS", peer_id)
            return {"ok": False, "reason": f"{peer_id!r} is not a registered peer"}

        log.info("Peer handshake accepted: %s", peer_id)

        # Gather sync state for the peer
        bcs_summary = self.bcs.state.summary()
        scrap_report = "[scrap unavailable]"
        try:
            from ..vision.material_tagger import MaterialTagger, SCRAP_PATH
            scrap_report = MaterialTagger(scrap_path=SCRAP_PATH).report()
        except Exception:
            pass

        return {
            "ok":      True,
            "peer_id": peer_id,
            "bcs":     bcs_summary,
            "scrap":   scrap_report,
        }

    def update_vision(self, state: dict, frame_b64: str | None = None,
                       mood: str = "neutral") -> None:
        """Public API for server to update vision state without touching internals."""
        self._vision_state = state
        if frame_b64:
            self._vision_frame = frame_b64
        self._vision_mood = mood

    def reset(self) -> None:
        self.memory.clear_session()

    @property
    def skill_list(self) -> list[str]:
        return skill_registry.list_skills()
