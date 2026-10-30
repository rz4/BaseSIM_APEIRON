"""The unified logger every entry point builds.

:class:`Logger` is a thin facade over a console logger and one metrics
backend. The backend is chosen once, at construction:

* ``"wandb"``  -> :class:`~apeiron.logger.wandb_logger.WandBLogger`
* ``"mlflow"`` -> :class:`~apeiron.logger.mlflow_logger.MLFlowLogger`
* anything else (including ``"none"``) -> ``WandBLogger(enabled=False)``

The disabled wandb backend is used as the null object on purpose: the
facade needs no second code path and CSV output keeps working, so
``backend = "none"`` means "no run is opened and nothing is sent", not "no
backend object".
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, cast

from apeiron.logger.console_logger import ConsoleLogger
from apeiron.logger.mlflow_logger import MLFlowLogger
from apeiron.logger.wandb_logger import StageType, WandBLogger

#: The metrics backends a config may ask for.
MetricsBackend = Literal["wandb", "mlflow", "none"]

#: Values :func:`configure_backend` accepts.
VALID_BACKENDS: tuple[str, ...] = ("wandb", "mlflow", "none")

#: The process-wide singleton handed out by :func:`get_logger`.
_LOGGER: "Logger | None" = None


class Logger:
    """Console logging plus one metrics backend, behind one object.

    Parameters
    ----------
    backend : MetricsBackend
        Which metrics backend to build. Defaults to ``"wandb"``, so a
        config with no ``[logging]`` section talks to the service.
    verbosity : str
        Console verbosity, e.g. ``"INFO"`` or ``"INFO:2"``.
    csv_path : str | Path | None
        Where the backend writes its metrics CSV.
    """

    def __init__(
        self,
        backend: MetricsBackend = "wandb",
        verbosity: str = "INFO",
        csv_path: str | Path | None = None,
    ) -> None:
        self._backend = backend

        if backend == "wandb":
            self._metrics: Any = WandBLogger(enabled=True, csv_path=csv_path)
        elif backend == "mlflow":
            self._metrics = MLFlowLogger(enabled=True, csv_path=csv_path)
        else:
            # "none" -- and anything unrecognised -- takes the disabled
            # backend, which records and writes CSVs but opens no run.
            self._metrics = WandBLogger(enabled=False, csv_path=csv_path)

        self._console = ConsoleLogger(
            verbosity=verbosity, get_step=lambda: self._metrics.step
        )

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------
    @property
    def backend(self) -> str:
        """The backend name this logger was built with."""
        return self._backend

    @property
    def verbosity(self) -> str:
        return self._console.verbosity

    @verbosity.setter
    def verbosity(self, value: str) -> None:
        self._console.verbosity = value

    @property
    def step(self) -> int:
        return self._metrics.step

    @property
    def current_stage(self) -> str | None:
        return self._metrics.current_stage

    @property
    def url(self) -> str | None:
        """The run's URL, or ``None`` when no run is open."""
        return self._metrics.url

    @property
    def csv_path(self) -> Path | None:
        return self._metrics.csv_path

    @csv_path.setter
    def csv_path(self, value: str | Path | None) -> None:
        self._metrics.csv_path = Path(value) if value is not None else None

    @property
    def metrics_history(self) -> list[dict[str, Any]]:
        return self._metrics.metrics_history

    # ------------------------------------------------------------------
    # metrics delegation
    # ------------------------------------------------------------------
    def init(self, cfg: Any, **kwargs: Any) -> Any:
        """Open the backend's run."""
        return self._metrics.init(cfg, **kwargs)

    def stage(self, name: StageType) -> None:
        self._metrics.stage(name)

    def get_stage_step(self, stage: StageType) -> int:
        return self._metrics.get_stage_step(stage)

    def log(
        self,
        metrics: dict[str, Any],
        step: int | None = None,
        commit: bool = True,
        prefix: bool = True,
        increment: bool = True,
    ) -> None:
        """Forward one row of metrics to the backend, keywords and all."""
        self._metrics.log(
            metrics, step=step, commit=commit, prefix=prefix, increment=increment
        )

    def save(
        self,
        path: str | Path,
        base_path: str | Path | None = None,
        policy: str = "now",
    ) -> None:
        self._metrics.save(path, base_path=base_path, policy=policy)

    def finish(
        self, exit_code: int | None = None, save_csv: bool = True
    ) -> Path | None:
        return self._metrics.finish(exit_code=exit_code, save_csv=save_csv)

    def to_dataframe(self) -> Any:
        return self._metrics.to_dataframe()

    def to_csv(self, csv_path: str | Path | None = None) -> Path | None:
        return self._metrics.to_csv(csv_path)

    # ------------------------------------------------------------------
    # console delegation
    # ------------------------------------------------------------------
    def info(self, msg: str, level: int = 0) -> None:
        self._console.info(msg, level)

    def debug(self, msg: str) -> None:
        self._console.debug(msg)

    def warning(self, msg: str) -> None:
        self._console.warning(msg)

    def error(self, msg: str) -> None:
        self._console.error(msg)

    def critical(self, msg: str) -> None:
        self._console.critical(msg)

    def exception(self, msg: str) -> None:
        self._console.exception(msg)


def configure_backend(cfg: Any) -> MetricsBackend:
    """Resolve the metrics backend a config asks for.

    A config with no ``[logging]`` section -- and ``None`` itself -- means
    ``"none"``: a run that never asked for telemetry must not reach the
    network (claims/LOG.md A18; previously this defaulted to wandb). An
    MLflow tracking URI is applied here, before any run is started.

    Raises
    ------
    ValueError
        If the configured backend is not one of ``VALID_BACKENDS``.
    """
    if cfg is None:
        return "none"

    logging_cfg = getattr(cfg, "logging", None)
    if logging_cfg is None:
        return "none"

    backend = getattr(logging_cfg, "backend", None) or "none"
    if backend not in VALID_BACKENDS:
        raise ValueError(
            f"Invalid logging backend {backend!r}; expected one of "
            f"{list(VALID_BACKENDS)}"
        )

    if backend == "mlflow":
        tracking_uri = getattr(logging_cfg, "mlflow_tracking_uri", None)
        if tracking_uri:
            import mlflow

            mlflow.set_tracking_uri(tracking_uri)

    return cast(MetricsBackend, backend)


def get_logger(
    backend: MetricsBackend = "wandb",
    verbosity: str = "INFO",
    csv_path: str | Path | None = None,
) -> Logger:
    """The process-wide :class:`Logger`.

    The arguments are used only when no instance exists yet: configure the
    singleton before anything else logs, or the first caller's settings
    win. :func:`reset_logger` clears it.
    """
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = Logger(backend=backend, verbosity=verbosity, csv_path=csv_path)
    return _LOGGER


def reset_logger() -> None:
    """Drop the singleton so the next :func:`get_logger` rebuilds it."""
    global _LOGGER
    _LOGGER = None
