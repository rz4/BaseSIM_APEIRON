"""MLflow metrics backend.

Mirrors :class:`~apeiron.logger.wandb_logger.WandBLogger`'s interface, with
three deliberate differences:

* metric keys are namespaced with **dots** (``eval.acc``), because that is
  how MLflow groups them;
* metrics are logged against the **per-stage** step, so a sparse global
  step cannot downsample away a rare ``drift.detected`` event;
* booleans become ``0``/``1`` ints and non-numeric values are dropped,
  because MLflow only accepts numbers.

``mlflow`` itself is imported lazily into ``_mlflow`` so constructing the
logger needs neither the package nor a tracking store.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

from apeiron.logger.wandb_logger import VALID_STAGES, StageType

try:
    import pandas as pd
except Exception:  # pragma: no cover - pandas is optional
    pd = None  # type: ignore[assignment]

#: MLflow rejects param values longer than this.
MAX_PARAM_LENGTH = 500

#: Default experiment name when a caller names none.
DEFAULT_EXPERIMENT = "apeiron"


class MLFlowLogger:
    """Metrics backend writing to MLflow and/or a CSV.

    Parameters
    ----------
    enabled : bool
        When false no run is started and nothing is sent.
    csv_path : str | Path | None
        Where :meth:`to_csv` writes. History is only retained when a path
        is configured.
    """

    def __init__(
        self,
        enabled: bool = True,
        csv_path: str | Path | None = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.csv_path: Path | None = Path(csv_path) if csv_path is not None else None
        self.run: Any = None

        self._mlflow: Any = None
        self._step = 0
        self._current_stage: str | None = None
        self._stage_steps: dict[str, int] = {stage: 0 for stage in VALID_STAGES}
        self.metrics_history: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # lazy import
    # ------------------------------------------------------------------
    def _get_mlflow(self) -> Any:
        """Import ``mlflow`` on first use and cache the module."""
        if self._mlflow is None:
            import mlflow  # imported here so construction stays cheap

            self._mlflow = mlflow
        return self._mlflow

    # ------------------------------------------------------------------
    # properties
    # ------------------------------------------------------------------
    @property
    def step(self) -> int:
        """The global step, incremented by every committing :meth:`log`."""
        return self._step

    @property
    def current_stage(self) -> str | None:
        """The stage most recently named, or ``None``."""
        return self._current_stage

    @property
    def url(self) -> str | None:
        """The run's web URL, or ``None`` for a local tracking store."""
        if self.run is None:
            return None
        uri = str(self._get_mlflow().get_tracking_uri() or "")
        if not uri.startswith(("http://", "https://")):
            return None
        info = getattr(self.run, "info", None)
        if info is None:
            return None
        return (
            f"{uri.rstrip('/')}/#/experiments/{info.experiment_id}"
            f"/runs/{info.run_id}"
        )

    # ------------------------------------------------------------------
    # stages
    # ------------------------------------------------------------------
    def stage(self, name: StageType) -> None:
        """Select the stage subsequent metrics belong to."""
        if name not in VALID_STAGES:
            raise ValueError(
                f"Invalid stage {name!r}; expected one of {list(VALID_STAGES)}"
            )
        self._current_stage = name

    def get_stage_step(self, stage: StageType) -> int:
        """The step counter for ``stage``; ``0`` for one never logged to."""
        return self._stage_steps.get(stage, 0)

    # ------------------------------------------------------------------
    # config flattening
    # ------------------------------------------------------------------
    def _flatten_config(self, config: dict[str, Any], prefix: str = "") -> dict[str, Any]:
        """Flatten nested dicts into dot-separated keys."""
        flat: dict[str, Any] = {}
        for key, value in config.items():
            name = f"{prefix}{key}"
            if isinstance(value, dict):
                flat.update(self._flatten_config(value, prefix=f"{name}."))
            else:
                flat[name] = value
        return flat

    # ------------------------------------------------------------------
    # run lifecycle
    # ------------------------------------------------------------------
    def init(
        self,
        cfg: Any,
        project: str | None = None,
        name: str | None = None,
        tags: list[str] | None = None,
        notes: str | None = None,
        group: str | None = None,
        **kwargs: Any,
    ) -> Any:
        """Start a run against the configured tracking store, unless disabled."""
        if not self.enabled:
            return None

        mlflow = self._get_mlflow()
        mlflow.set_experiment(project or DEFAULT_EXPERIMENT)
        mlflow.enable_system_metrics_logging()

        run_tags: dict[str, Any] = {}
        if group is not None:
            run_tags["group"] = group
        for index, tag in enumerate(tags or []):
            run_tags[f"user_tag_{index}"] = tag

        self.run = mlflow.start_run(run_name=name, tags=run_tags or None)

        if notes:
            mlflow.set_tag("mlflow.note.content", notes)

        params = self._flatten_config(_as_dict(cfg))
        mlflow.log_params({k: _truncate(v) for k, v in params.items()})
        return self.run

    def finish(self, exit_code: int | None = None, save_csv: bool = True) -> Path | None:
        """End the run, optionally writing the CSV first.

        A non-zero ``exit_code`` marks the run ``FAILED``.
        """
        written = self.to_csv() if save_csv else None
        if self.enabled and self.run is not None:
            mlflow = self._get_mlflow()
            if exit_code:
                mlflow.set_tag("mlflow.runStatus", "FAILED")
            mlflow.end_run()
            self.run = None
        return written

    # ------------------------------------------------------------------
    # logging
    # ------------------------------------------------------------------
    def log(
        self,
        metrics: dict[str, Any],
        step: int | None = None,
        commit: bool = True,
        prefix: bool = True,
        increment: bool = True,
    ) -> None:
        """Record one row of metrics.

        ``commit`` is accepted for interface parity with the wandb backend;
        MLflow has no uncommitted row.
        """
        if increment:
            self._step += 1
            if self._current_stage:
                self._stage_steps[self._current_stage] = (
                    self._stage_steps.get(self._current_stage, 0) + 1
                )

        current_step = self._step if step is None else step

        prefixed_metrics: dict[str, Any] = {}
        for key, value in metrics.items():
            dotted = key.replace("/", ".")
            if prefix and self._current_stage and "." not in dotted:
                prefixed_metrics[f"{self._current_stage}.{dotted}"] = value
            else:
                prefixed_metrics[dotted] = value

        prefixed_metrics["step"] = current_step
        if self._current_stage:
            prefixed_metrics[f"{self._current_stage}.step"] = self._stage_steps.get(
                self._current_stage, 0
            )

        if self.csv_path is not None:
            self.metrics_history.append(prefixed_metrics)

        if not (self.enabled and self.run is not None):
            return

        numeric = {
            key: _as_number(value)
            for key, value in prefixed_metrics.items()
            if key != "step" and _as_number(value) is not None
        }
        if not numeric:
            return

        # the per-stage step, so rare events are not downsampled away by a
        # sparse global step
        if step is not None:
            logged_step = step
        elif self._current_stage:
            logged_step = self._stage_steps.get(self._current_stage, 0)
        else:
            logged_step = self._step
        self._get_mlflow().log_metrics(numeric, step=logged_step)

    def save(
        self,
        path: str | Path,
        base_path: str | Path | None = None,
        policy: str = "now",
    ) -> None:
        """Log a file as a run artifact; a no-op without a run.

        ``base_path`` and ``policy`` exist for interface parity with the
        wandb backend, which needs them.
        """
        if self.enabled and self.run is not None:
            self._get_mlflow().log_artifact(str(path))

    # ------------------------------------------------------------------
    # records
    # ------------------------------------------------------------------
    def to_dataframe(self) -> Any:
        """The recorded history in long format (``step``/``metric``/``value``)."""
        if not self.metrics_history or pd is None:
            return None
        rows = [
            {"step": entry.get("step"), "metric": key, "value": value}
            for entry in self.metrics_history
            for key, value in entry.items()
            if key != "step"
        ]
        if not rows:
            return None
        return pd.DataFrame(rows, columns=["step", "metric", "value"])

    def to_csv(self, csv_path: str | Path | None = None) -> Path | None:
        """Write the history to ``csv_path`` or the configured path."""
        target = csv_path if csv_path is not None else self.csv_path
        if target is None:
            return None
        frame = self.to_dataframe()
        if frame is None:
            return None
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(target, index=False)
        return target


def _as_dict(cfg: Any) -> dict[str, Any]:
    """A config dataclass as a plain nested dict; a dict unchanged."""
    if dataclasses.is_dataclass(cfg) and not isinstance(cfg, type):
        return dataclasses.asdict(cfg)
    return dict(cfg)


def _truncate(value: Any) -> Any:
    """Clip over-long string params to MLflow's limit."""
    if isinstance(value, str) and len(value) > MAX_PARAM_LENGTH:
        return value[:MAX_PARAM_LENGTH]
    return value


def _as_number(value: Any) -> int | float | None:
    """A value MLflow can store, or ``None`` if it cannot.

    Booleans become ints rather than being cast through float, so
    ``detected`` reads back as ``1`` and not ``1.0``.
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int) or isinstance(value, float):
        return value
    return None
