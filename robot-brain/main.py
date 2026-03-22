"""
main.py — Robot Brain Entry Point
===================================
Usage:
  python main.py                      # interactive REPL (mock hardware)
  python main.py --platform raspberry_pi   # real Pi 5 GPIO
  python main.py --provider openai    # use GPT-4o
  python main.py --provider anthropic # use Claude (default)
  python main.py --verbose            # show raw LLM output
  python main.py --command "move forward"  # single command then exit
"""

from __future__ import annotations

import argparse
import logging
import os
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ARIA Robot Brain — Phase 1",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--provider",
        choices=["openai", "anthropic"],
        help="LLM provider (overrides LLM_PROVIDER env var)",
    )
    parser.add_argument(
        "--platform",
        choices=["mock", "raspberry_pi"],
        help="Hardware platform (overrides HARDWARE_PLATFORM env var)",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show raw LLM responses",
    )
    parser.add_argument(
        "--command", "-c",
        type=str,
        default=None,
        help="Run a single command and exit (non-interactive mode)",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return parser.parse_args()


def apply_args_to_env(args: argparse.Namespace) -> None:
    """Push CLI flags into environment so config picks them up before import."""
    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider
    if args.platform:
        os.environ["HARDWARE_PLATFORM"] = args.platform
    if args.verbose:
        os.environ["VERBOSE_LLM"] = "1"


def main() -> int:
    args = parse_args()
    apply_args_to_env(args)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Import after env is set so config reads correct values
    from config import config
    from control_loop import ControlLoop

    try:
        config.validate()
    except ValueError as exc:
        print(f"❌ Configuration error: {exc}")
        print("\nSet your API key:")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        print("  export OPENAI_API_KEY=sk-...")
        return 1

    loop = ControlLoop(config)

    if args.command:
        # Non-interactive: run one command and exit
        try:
            loop.step(args.command)
        except RuntimeError as exc:
            print(f"⚠️  Error: {exc}")
            return 1
        return 0

    # Interactive REPL
    loop.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())

