"""CLI report over an experiment workspace.

Usage:
    python -m apeiron.experiment <experiment_path> [--gc-pins]
"""

from __future__ import annotations

import argparse

from apeiron.experiment.workspace import Experiment


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Experiment workspace report")
    p.add_argument("path", help="experiment workspace directory")
    p.add_argument(
        "--gc-pins",
        action="store_true",
        help="release artifact pins held by finished or vanished runs",
    )
    args = p.parse_args(argv)

    exp = Experiment(args.path)
    print(exp.report())
    if args.gc_pins:
        print(f"\nReleased {exp.gc_pins()} stale pin(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
