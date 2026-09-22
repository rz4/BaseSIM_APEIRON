"""``python -m apeiron.experiment <path>`` -- what is in an experiment."""

from __future__ import annotations

import argparse

from apeiron.experiment.workspace import Experiment


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m apeiron.experiment",
        description="Summarise the runs under an experiment directory.",
    )
    parser.add_argument("path", help="an experiment directory, or one holding several")
    args = parser.parse_args(argv)
    print(Experiment(args.path).report())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
