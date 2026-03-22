"""
control_loop.py — Main Robot Control Loop
==========================================
Wires together: Brain → Vision → Action Parser → Hardware Executor

Flow per turn:
  1. Get user command (text input or future sensor event)
  2. (Optional) Capture ambient frame → vision description → inject as context
  3. Brain processes command + vision context → BrainResponse (thought + actions)
  4. Each action executed on hardware in order
  5. LOOK action → capture fresh frame → description fed back to brain
  6. QUERY actions pause and collect human input before continuing

REPL commands: anything natural | look | snap | status | reset | quit

Run this file directly for an interactive REPL session.
"""

from __future__ import annotations

import logging
from typing import Optional

from action_schema import AnyAction, BrainResponse, LookAction, QueryAction
from brain import Brain
from config import RobotConfig, config as default_config
from mock_hardware import HardwareInterface, create_hardware

log = logging.getLogger(__name__)


class ControlLoop:
    """
    Orchestrates Brain ↔ Hardware interaction.

    Usage:
        loop = ControlLoop()
        loop.run()          # interactive REPL
        # or single-shot:
        response = loop.step("move forward 1 second")
    """

    def __init__(self, cfg: RobotConfig = default_config) -> None:
        self._cfg = cfg
        self._brain = Brain(cfg)
        self._camera = None
        self._vision_adapter = None
        self._vision_active = False
        self._init_vision()
        self._hardware: HardwareInterface = create_hardware(
            cfg, camera=self._camera, vision=self._vision_adapter
        )
        self._running = False

    def _init_vision(self) -> None:
        """Initialise camera and vision adapter if vision is enabled in config."""
        if not self._cfg.vision.enabled:
            log.info("Vision disabled (VISION_ENABLED=0).")
            return
        try:
            from vision.camera import Camera, auto_detect_camera
            from vision.vision_adapter import create_vision_adapter

            vcfg = self._cfg.vision
            if vcfg.camera_index == -1:
                self._camera = auto_detect_camera(
                    width=vcfg.capture_width, height=vcfg.capture_height
                )
            else:
                self._camera = Camera(
                    index=vcfg.camera_index,
                    width=vcfg.capture_width,
                    height=vcfg.capture_height,
                )
            self._vision_adapter = create_vision_adapter(self._cfg.llm, vcfg)
            self._vision_active = True
            log.info("Vision pipeline active (camera index %d).", self._camera.index)
        except Exception as exc:
            log.warning("Vision init failed — running without camera: %s", exc)
            print(f"  ⚠️  Vision unavailable: {exc}")

    def _get_vision_context(self, prompt: str | None = None) -> str | None:
        """Capture a frame and return a scene description, or None on failure."""
        if not self._vision_active or self._camera is None or self._vision_adapter is None:
            return None
        try:
            frame_b64 = self._camera.capture_jpeg_b64(self._cfg.vision.jpeg_quality)
            return self._vision_adapter.describe(frame_b64, prompt)
        except Exception as exc:
            log.error("Vision context capture failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # Single step — useful for tests and scripted sequences
    # ------------------------------------------------------------------

    def step(self, user_input: str, inject_vision: bool = True) -> BrainResponse:
        """
        Process one command and execute all resulting actions.
        inject_vision: if True and vision is active, capture ambient frame first.
        Returns the full BrainResponse for inspection.
        """
        log.info("INPUT: %s", user_input)

        # Ambient vision — let the brain see what's in front of it right now
        vision_ctx: str | None = None
        if inject_vision:
            vision_ctx = self._get_vision_context()

        response = self._brain.process(user_input, vision_context=vision_ctx)
        print(f"\n  💭 {self._brain.robot_name}: {response.thought}")

        pending_query: Optional[QueryAction] = None
        look_result: Optional[str] = None

        for action in response.actions:
            if isinstance(action, QueryAction):
                pending_query = action
                break
            result = self._execute_action(action)
            # LOOK action with vision pipeline returns "[VISION] <description>"
            if isinstance(action, LookAction) and result.startswith("[VISION]"):
                look_result = result[len("[VISION] "):]

        # Feed LOOK result back into brain so it can reason about what it saw
        if look_result:
            print(f"\n  👁️  Feeding vision back to brain...")
            vision_response = self._brain.process(f"[VISION RESULT] You can see: {look_result}")
            print(f"\n  💭 {self._brain.robot_name}: {vision_response.thought}")
            for action in vision_response.actions:
                self._execute_action(action)

        if pending_query:
            print(f"\n  ❓ {self._brain.robot_name} asks: {pending_query.question}")
            human_answer = input("  Your answer: ").strip()
            if human_answer:
                followup = self._brain.process(human_answer)
                print(f"\n  💭 {self._brain.robot_name}: {followup.thought}")
                for action in followup.actions:
                    self._execute_action(action)

        return response

    # ------------------------------------------------------------------
    # Interactive REPL
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Start the interactive command loop (blocks until exit)."""
        print(self._banner())
        self._running = True

        try:
            while self._running:
                try:
                    raw = input(f"\n[{self._brain.robot_name}] > ").strip()
                except (EOFError, KeyboardInterrupt):
                    print("\nShutting down...")
                    break

                if not raw:
                    continue

                if raw.lower() in ("exit", "quit", "q"):
                    print("Goodbye.")
                    break

                if raw.lower() == "reset":
                    self._brain.reset()
                    print("Brain context cleared.")
                    continue

                if raw.lower() in ("look", "snap", "what do you see"):
                    # Manual vision snapshot without sending to brain
                    desc = self._get_vision_context()
                    if desc:
                        print(f"\n  👁️  [SNAPSHOT] {desc}")
                    else:
                        print("  ⚠️  No camera available.")
                    continue

                if raw.lower() == "status":
                    vision_status = (
                        f"active (camera {self._camera.index})"
                        if self._vision_active else "inactive"
                    )
                    print(f"  Platform : {self._cfg.hardware.platform}")
                    print(f"  Provider : {self._cfg.llm.provider}")
                    print(f"  Vision   : {vision_status}")
                    print(f"  History  : {self._brain.history_length} messages")
                    continue

                try:
                    self.step(raw)
                except RuntimeError as exc:
                    print(f"\n  ⚠️  Error: {exc}")
                    log.error("Step error: %s", exc)

        finally:
            self._hardware.shutdown()
            log.info("Control loop stopped.")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _execute_action(self, action: AnyAction) -> str:
        result = self._hardware.execute(action)
        log.debug("Action %s → %s", type(action).__name__, result)
        return result

    def _banner(self) -> str:
        platform = self._cfg.hardware.platform.upper()
        provider = self._cfg.llm.provider.upper()
        vision_note = (
            f"  Vision   : active (cam {self._camera.index})\n"
            if self._vision_active else ""
        )
        return (
            f"\n{'='*55}\n"
            f"  {self._brain.robot_name} — Robot Brain  (Phase 2)\n"
            f"  Platform : {platform}   Provider : {provider}\n"
            f"{vision_note}"
            f"  Commands : type anything | look | status | reset | quit\n"
            f"{'='*55}"
        )

