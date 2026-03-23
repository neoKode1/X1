"""
prototype-x1 — main entry point
================================
Run:  python3 main.py
      python3 main.py --model llama3.2
      python3 main.py --cloud anthropic   (skip Ollama, go straight to cloud)

Type 'quit' or 'exit' to stop.
Type '/skills'  to list loaded skills.
Type '/memory'  to show session memory.
Type '/reset'   to clear session memory.
Type '/reload'  to hot-reload skills.
"""
from __future__ import annotations
import argparse
import logging
import os
import sys
from pathlib import Path

# ── Rich for pretty output (optional) ─────────────────────────────────────────
try:
    from rich.console import Console
    from rich.markdown import Markdown
    console = Console()
    def print_aria(text: str) -> None:
        console.print(f"[bold cyan]ARIA:[/bold cyan]", end=" ")
        try:
            console.print(Markdown(text))
        except Exception:
            console.print(text)
    def print_info(text: str) -> None:
        console.print(f"[dim]{text}[/dim]")
    def print_err(text: str) -> None:
        console.print(f"[bold red]{text}[/bold red]")
except ImportError:
    def print_aria(text: str) -> None:
        print(f"ARIA: {text}")
    def print_info(text: str) -> None:
        print(f"  {text}")
    def print_err(text: str) -> None:
        print(f"ERROR: {text}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="prototype-x1 brain REPL")
    p.add_argument("--model", default=None, help="Ollama model override (e.g. llama3.2)")
    p.add_argument("--cloud", choices=["anthropic", "openai"], default=None,
                   help="Skip Ollama, use cloud provider")
    p.add_argument("--debug", action="store_true", help="Enable debug logging")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── Logging ────────────────────────────────────────────────────────────────
    level = logging.DEBUG if args.debug else logging.WARNING
    logging.basicConfig(level=level, format="%(name)s [%(levelname)s] %(message)s")

    # ── Load .env ──────────────────────────────────────────────────────────────
    env_path = Path(__file__).parent / ".env"
    if env_path.exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
        except ImportError:
            pass

    # ── Brain init ─────────────────────────────────────────────────────────────
    sys.path.insert(0, str(Path(__file__).parent))
    from brain.config import BrainConfig, LLMConfig
    from brain.core import Brain

    cfg = BrainConfig()
    if args.model:
        cfg.llm.ollama_model = args.model
    if args.cloud:
        cfg.llm.cloud_fallback = args.cloud
        # Force cloud by pointing Ollama host to something unreachable
        cfg.llm.ollama_host = "http://localhost:0"
    if args.debug:
        cfg.debug = True

    print_info(f"Initialising ARIA brain...")
    try:
        brain = Brain(cfg)
    except Exception as e:
        print_err(f"Failed to init brain: {e}")
        sys.exit(1)

    print_info(f"Skills loaded: {', '.join(brain.skill_list)}")
    print_info(f"Model: {cfg.llm.ollama_model} | fallback: {cfg.llm.cloud_fallback}")
    print_info("Type 'quit' to exit. /skills /memory /reset /reload for controls.")
    print()

    # ── REPL ───────────────────────────────────────────────────────────────────
    skills_dir = Path(__file__).parent / "skills"

    while True:
        try:
            user_input = input("YOU: ").strip()
        except (KeyboardInterrupt, EOFError):
            print()
            print_info("Shutting down.")
            break

        if not user_input:
            continue

        # ── Meta commands ──────────────────────────────────────────────────────
        if user_input.lower() in ("quit", "exit"):
            print_info("Goodbye.")
            break

        if user_input == "/skills":
            print_info("Loaded skills: " + ", ".join(brain.skill_list))
            continue

        if user_input == "/memory":
            msgs = brain.memory.as_messages()
            if not msgs:
                print_info("Memory empty.")
            for m in msgs:
                print_info(f"  [{m['role']}] {m['content'][:120]}")
            continue

        if user_input == "/reset":
            brain.reset()
            print_info("Session memory cleared.")
            continue

        if user_input == "/reload":
            from brain import skills as skill_registry
            reloaded = skill_registry.hot_reload(skills_dir)
            print_info(f"Reloaded: {reloaded or 'nothing changed'}")
            continue

        # ── Brain turn ─────────────────────────────────────────────────────────
        try:
            response = brain.process(user_input, speaker="Neokode")
        except Exception as e:
            print_err(f"Brain error: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
            continue

        print_aria(response.text)

        if response.skill_calls:
            print_info(f"  Skills used: {[c['name'] for c in response.skill_calls]}")
        print_info(f"  [{response.provider} | {response.latency_ms}ms]")
        print()


if __name__ == "__main__":
    main()
