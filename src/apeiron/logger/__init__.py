"""Logging for apeiron: a console logger and the metrics backends.

This package's namespace is a contract between layers -- a sibling imports
`Logger`, `configure_backend`, `VALID_STAGES` and the rest from here, so
every name below is part of the surface and not a convenience.

Nothing else may be re-exported: `from __future__ import annotations` in
particular would leak an `annotations` attribute into the namespace.
"""

from apeiron.logger.console_logger import ConsoleLogger
from apeiron.logger.logger import (
    Logger,
    MetricsBackend,
    configure_backend,
    get_logger,
    reset_logger,
)
from apeiron.logger.mlflow_logger import MLFlowLogger
from apeiron.logger.wandb_logger import VALID_STAGES, StageType, WandBLogger

__all__ = [
    "ConsoleLogger",
    "Logger",
    "MLFlowLogger",
    "MetricsBackend",
    "StageType",
    "VALID_STAGES",
    "WandBLogger",
    "configure_backend",
    "get_logger",
    "reset_logger",
]
