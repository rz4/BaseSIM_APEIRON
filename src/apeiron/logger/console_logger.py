"""Console logging: a tiered, coloured stdout logger.

The verbosity string accepted here is the one a TOML config carries, so the
parsing is deliberately forgiving: names are upper-cased and stripped, an
``INFO:n`` tier subtracts ``n`` from ``logging.INFO`` (and only from INFO),
and a name the level map does not hold falls back to ``INFO`` rather than
raising -- a config typo must not take a training run down.
"""

from __future__ import annotations

import logging
import sys
from typing import Callable

#: The level names a verbosity string may use.
LEVEL_MAP: dict[str, int] = {
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

#: Default format used by the stream handler.
DEFAULT_FORMAT = "%(asctime)s | %(levelname)-8s | %(message)s"

#: Default ``asctime`` format.
DEFAULT_DATE_FORMAT = "%H:%M:%S"

#: The logger name the console handler is attached to.
LOGGER_NAME = "apeiron"


class ColoredFormatter(logging.Formatter):
    """A ``logging.Formatter`` that wraps the level name in ANSI colour.

    The record is restored after formatting so a second handler sees it
    untouched.
    """

    COLORS: dict[str, str] = {
        "DEBUG": "\033[36m",  # cyan
        "INFO": "\033[32m",  # green
        "WARNING": "\033[33m",  # yellow
        "ERROR": "\033[31m",  # red
        "CRITICAL": "\033[1;31m",  # bold red
    }

    RESET = "\033[0m"

    def format(self, record: logging.LogRecord) -> str:
        original = record.levelname
        color = self.COLORS.get(original)
        if color is not None:
            record.levelname = f"{color}{original}{self.RESET}"
        try:
            return super().format(record)
        finally:
            record.levelname = original


class StepFilter(logging.Filter):
    """Attaches the current training step to every record.

    Parameters
    ----------
    get_step : Callable[[], int] | None
        Returns the step to stamp onto a record. A missing or failing
        callable yields ``0`` rather than losing the record.
    """

    def __init__(self, get_step: Callable[[], int] | None = None) -> None:
        super().__init__()
        self.get_step = get_step

    def filter(self, record: logging.LogRecord) -> bool:
        step = 0
        if self.get_step is not None:
            try:
                step = self.get_step()
            except Exception:  # pragma: no cover - never drop a record
                step = 0
        record.step = step
        return True


class ConsoleLogger:
    """Stdout logging with a tiered ``INFO`` level.

    Parameters
    ----------
    verbosity : str
        Either a level name (``"INFO"``, ``"debug"``, ...) or a tiered
        ``"INFO:n"`` string. Stored upper-cased.
    name : str
        The underlying ``logging.Logger`` name.
    get_step : Callable[[], int] | None
        Optional step source, attached through a :class:`StepFilter`.
    colored : bool
        Whether to use :class:`ColoredFormatter`.
    """

    def __init__(
        self,
        verbosity: str = "INFO",
        name: str = LOGGER_NAME,
        get_step: Callable[[], int] | None = None,
        colored: bool = True,
    ) -> None:
        self._verbosity = str(verbosity).upper()
        self._name = name
        self._get_step = get_step
        self._colored = colored

        self._logger = logging.getLogger(name)
        self._logger.propagate = False
        self._handler = logging.StreamHandler(stream=sys.stdout)
        formatter_cls = ColoredFormatter if colored else logging.Formatter
        self._handler.setFormatter(
            formatter_cls(DEFAULT_FORMAT, datefmt=DEFAULT_DATE_FORMAT)
        )
        self._filter = StepFilter(get_step=get_step)
        self._handler.addFilter(self._filter)

        # A fresh ConsoleLogger owns the handler set: repeated construction
        # (entry points, tests) must not duplicate every line.
        for existing in list(self._logger.handlers):
            self._logger.removeHandler(existing)
        self._logger.addHandler(self._handler)

        self._logger.setLevel(self._get_log_level(self._verbosity))

    # ------------------------------------------------------------------
    # verbosity parsing
    # ------------------------------------------------------------------
    def _parse_verbosity(self, verbosity: str) -> tuple[str, int]:
        """Split a verbosity string into ``(base_level, tier)``.

        ``"INFO:3"`` -> ``("INFO", 3)``; a tier that is absent, negative or
        not a number is ``0``.
        """
        raw = str(verbosity).strip().upper()
        base, sep, tier_text = raw.partition(":")
        tier = 0
        if sep:
            try:
                tier = int(tier_text)
            except ValueError:
                tier = 0
        if tier < 0:
            tier = 0
        return base, tier

    def _get_log_level(self, verbosity: str) -> int:
        """Resolve a verbosity string to a numeric level.

        The tier offset applies to ``INFO`` only. An unrecognised name maps
        to ``INFO`` rather than raising -- a logging setting should not kill
        a run -- but it WARNS: the old silent fallback meant
        ``verbosity = "WARN"`` printed the full info stream, more output
        than asked for, on exactly the runs trying to be quiet
        (claims/LOG.md A52).
        """
        base_level, tier = self._parse_verbosity(verbosity)
        if base_level == "INFO":
            return logging.INFO - tier
        if base_level not in LEVEL_MAP:
            import warnings

            warnings.warn(
                f"unrecognised verbosity {verbosity!r}; falling back to INFO. "
                f"Known names: DEBUG, INFO, INFO:0-9, WARNING, ERROR, "
                f"CRITICAL (did you mean WARNING rather than WARN?)",
                stacklevel=3,
            )
        return LEVEL_MAP.get(base_level, logging.INFO)

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------
    @property
    def verbosity(self) -> str:
        """The verbosity string as stored (upper-cased, not stripped)."""
        return self._verbosity

    @verbosity.setter
    def verbosity(self, value: str) -> None:
        self._verbosity = str(value).upper()
        self._logger.setLevel(self._get_log_level(self._verbosity))

    @property
    def logger(self) -> logging.Logger:
        """The underlying ``logging.Logger``."""
        return self._logger

    @property
    def level(self) -> int:
        """The numeric level currently in effect."""
        return self._logger.level

    # ------------------------------------------------------------------
    # emit
    # ------------------------------------------------------------------
    def info(self, msg: str, level: int = 0) -> None:
        """Log at ``INFO`` minus ``level``.

        A tier deep enough to reach ``DEBUG`` is emitted as ``DEBUG`` so it
        stays inside the documented level range.
        """
        resolved = logging.INFO - int(level)
        if resolved <= logging.DEBUG:
            self._logger.debug(msg)
        else:
            self._logger.log(resolved, msg)

    def debug(self, msg: str) -> None:
        self._logger.debug(msg)

    def warning(self, msg: str) -> None:
        self._logger.warning(msg)

    def error(self, msg: str) -> None:
        self._logger.error(msg)

    def critical(self, msg: str) -> None:
        self._logger.critical(msg)

    def exception(self, msg: str) -> None:
        """Log ``msg`` at ``ERROR`` with the active traceback attached."""
        self._logger.exception(msg)
