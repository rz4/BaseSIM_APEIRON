"""PyTorch model harness: the seam between a concrete model/data stream and APEIRON.

A harness owns the model, the dataloaders, the optimizer and the criterion.
The monitor and the trainer only ever talk to this surface, so every hook a
subclass is expected to provide refuses loudly when it is not provided: an
empty body would hand ``None`` to a caller that unpacks it into loaders.
"""

from __future__ import annotations

import os
import warnings
from abc import ABC, abstractmethod
from typing import Any, Callable, Sequence

import torch

from apeiron.config.configuration import Config
from apeiron.evaluation.metrics import accuracy

#: Default retention budget for the per-task train-test matrix ``R``.
DEFAULT_MAX_TASK_RECORDS = 50

#: Filename stem of a post-adaptation checkpoint, and the pointer file that
#: always names the newest one.
_CKPT_PREFIX = "drift_adaptation_"
_CKPT_SUFFIX = ".pt"
_LATEST_POINTER = "latest"


class BaseModelHarness(ABC):
    """Abstract base for every model harness.

    Subclasses supply the model-specific hooks; everything the framework
    calls across the layer boundary (``_unpack``, ``_to_scalar``, ``eval``,
    ``register_task``, ``eval_past_tasks``, ``task_diagonals``,
    ``ckpts_enabled``, ``save_ckpt``) is implemented here.
    """

    def __init__(self, cfg: Config, model: torch.nn.Module):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.model = model.to(self.device)

        #: name -> callable(output, target) -> scalar-ish
        self.eval_metrics: dict[str, Callable[..., Any]] = {"accuracy": accuracy}

        #: Retention budget for the task registry. Eviction keeps the MOST
        #: RECENT records so that forgetting stays measurable against what
        #: came before.
        self.max_task_records: int = DEFAULT_MAX_TASK_RECORDS

        #: [(frozen_eval_set, diagonal_metrics), ...], oldest first.
        self._task_records: list[tuple[Any, list[float]]] = []

    # ------------------------------------------------------------------
    # Deprecated-hook bridge
    # ------------------------------------------------------------------
    def __init_subclass__(cls, **kwargs: Any) -> None:
        """Bridge the historically misspelled ``get_optmizer`` hook.

        A subclass that implements only the old spelling still satisfies the
        ABC: the implementation is aliased onto ``get_optimizer`` (which is
        the only name the framework calls) and a ``DeprecationWarning`` is
        raised where the class is defined.
        """
        super().__init_subclass__(**kwargs)
        legacy = cls.__dict__.get("get_optmizer")
        if legacy is None:
            return
        warnings.warn(
            "`get_optmizer` is deprecated and will be removed; rename the hook "
            "to `get_optimizer`.",
            DeprecationWarning,
            stacklevel=3,
        )
        if "get_optimizer" not in cls.__dict__:

            def _bridged_get_optimizer(self, *args: Any, **kwargs: Any):
                return legacy(self, *args, **kwargs)

            _bridged_get_optimizer.__name__ = "get_optimizer"
            _bridged_get_optimizer.__qualname__ = f"{cls.__qualname__}.get_optimizer"
            _bridged_get_optimizer.__doc__ = (
                "Deprecated bridge onto this class's `get_optmizer`."
            )
            cls.get_optimizer = _bridged_get_optimizer  # type: ignore[assignment]

    def get_optmizer(self):
        """Deprecated alias of :meth:`get_optimizer` (note the missing ``i``)."""
        warnings.warn(
            "`get_optmizer` is deprecated and will be removed; call "
            "`get_optimizer` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.get_optimizer()

    # ------------------------------------------------------------------
    # Hooks a subclass must provide
    # ------------------------------------------------------------------
    @abstractmethod
    def get_optimizer(self):
        """Return the optimizer used for continual-learning updates."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement get_optimizer(); the "
            "framework has no default optimizer for a harness."
        )

    @abstractmethod
    def update_data_stream(self):
        """Advance the harness's data stream by one window."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement update_data_stream(); the "
            "monitor advances the stream through this hook."
        )

    @abstractmethod
    def get_stream_dataloader(self):
        """Return the dataloader the continuous monitor iterates."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement get_stream_dataloader(); "
            "the monitor has nothing to iterate without it."
        )

    @abstractmethod
    def get_hist_dataloaders(self):
        """Return ``(hist_train_loader, hist_val_loader)``, or ``(None, None)``."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement get_hist_dataloaders(); "
            "return (None, None) explicitly when the harness keeps no history."
        )

    def get_train_dataloaders(self):
        """Return ``(train_loader, val_loader)`` for the current window.

        Not abstract -- a harness may inherit this refusal -- but a harness
        that is trained or evaluated has to override it.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement get_train_dataloaders(); "
            "training and evaluation both read the current window from it."
        )

    @abstractmethod
    def get_criterion(self):
        """Return the loss function used for training and loss metrics."""
        raise NotImplementedError(
            f"{type(self).__name__} must implement get_criterion(); the "
            "framework has no default loss for a harness."
        )

    # ------------------------------------------------------------------
    # Batch / value seams, called by the monitor
    # ------------------------------------------------------------------
    def _unpack(self, batch) -> tuple[Any, Any]:
        """Split a batch into ``(inputs, targets)``.

        Two things out, or an error: a loader yielding ``(x, y, weight)`` is
        a real shape, and silently dropping the third element would train on
        mislabelled pairs.
        """
        if not isinstance(batch, (tuple, list)):
            raise ValueError(
                f"cannot unpack a batch of type {type(batch).__name__} into "
                "(inputs, targets); override _unpack() for this loader"
            )
        if len(batch) != 2:
            raise ValueError(
                f"expected a batch of 2 elements (inputs, targets), got "
                f"{len(batch)}; override _unpack() for this loader"
            )
        return batch[0], batch[1]

    @staticmethod
    def _to_scalar(value) -> float:
        """Reduce a metric value to one float, averaging non-scalar tensors."""
        if isinstance(value, torch.Tensor):
            if value.ndim == 0:
                return float(value.item())
            return float(value.mean().item())
        return float(value)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def _eval_loader(self, loader) -> list[float]:
        """Score ``self.eval_metrics`` over ``loader``, averaged per SAMPLE.

        Raises ``RuntimeError`` on a loader with no batches: a model that was
        never evaluated must not read as a model that scored zero, because
        that zero travels into the CSV and into the drift detector.
        """
        names = list(self.eval_metrics)
        totals = {name: 0.0 for name in names}
        n_samples = 0
        n_batches = 0

        self.model.eval()
        with torch.no_grad():
            for batch in loader:
                inputs, targets = self._unpack(batch)
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                outputs = self.model(inputs)
                size = int(targets.shape[0]) if targets.ndim > 0 else 1
                for name in names:
                    totals[name] += (
                        self._to_scalar(self.eval_metrics[name](outputs, targets))
                        * size
                    )
                n_samples += size
                n_batches += 1

        if n_batches == 0 or n_samples == 0:
            raise RuntimeError(
                "cannot evaluate an empty dataloader: it yielded no batches, "
                "and reporting 0.0 would be indistinguishable from a model "
                "that genuinely scored zero"
            )
        return [totals[name] / n_samples for name in names]

    def eval(self) -> list[float]:
        """Score the current validation window with ``self.eval_metrics``."""
        _, val_loader = self.get_train_dataloaders()
        return self._eval_loader(val_loader)

    def history_eval(self) -> list[float] | None:
        """Score the historical window, or ``None`` when there is no history."""
        hist_train_loader, hist_val_loader = self.get_hist_dataloaders()
        loader = hist_val_loader if hist_val_loader is not None else hist_train_loader
        if loader is None:
            return None
        return self._eval_loader(loader)

    # ------------------------------------------------------------------
    # Per-task transfer metrics (the train-test matrix R)
    # ------------------------------------------------------------------
    @property
    def task_diagonals(self) -> list[list[float]]:
        """The diagonal of ``R``: one metric row per registered task, oldest first."""
        return [list(diagonal) for _, diagonal in self._task_records]

    def register_task(self, diagonal_metrics: Sequence[float]) -> None:
        """Freeze the current validation window as a past task.

        The eval set is materialised (a copy, not a reference to a window the
        stream will replace) and stored together with the diagonal measured
        on it, so eviction can never misalign a task's eval set from its
        diagonal.
        """
        _, val_loader = self.get_train_dataloaders()
        self._task_records.append(
            (self._freeze_eval_set(val_loader), [float(m) for m in diagonal_metrics])
        )
        self._enforce_task_retention()

    def _freeze_eval_set(self, loader) -> list[tuple[Any, Any]]:
        """Materialise ``loader`` into detached copies of its batches."""
        frozen = []
        for batch in loader:
            inputs, targets = self._unpack(batch)
            frozen.append((inputs.detach().clone(), targets.detach().clone()))
        return frozen

    def _enforce_task_retention(self) -> None:
        """Drop the OLDEST records beyond ``max_task_records``."""
        cap = int(self.max_task_records)
        if cap <= 0:
            self._task_records.clear()
        elif len(self._task_records) > cap:
            del self._task_records[: len(self._task_records) - cap]

    def eval_past_tasks(self) -> list[list[float]]:
        """Score the CURRENT model on every frozen window, oldest first."""
        return [self._eval_loader(frozen) for frozen, _ in self._task_records]

    # ------------------------------------------------------------------
    # FIFO post-adaptation checkpointing
    # ------------------------------------------------------------------
    @property
    def ckpts_enabled(self) -> bool:
        """Whether checkpointing has both a retention budget and a path."""
        model_cfg = self.cfg.model
        return bool(int(model_cfg.max_ckpts) > 0 and model_cfg.ckpts_path)

    def build_checkpoint_payload(self) -> dict[str, Any]:
        """The object ``save_ckpt`` writes; subclasses may extend it."""
        return self.model.state_dict()

    def save_ckpt(self, event) -> str:
        """Write ``drift_adaptation_<event>.pt`` and update the ``latest`` pointer.

        The newest ``max_ckpts`` files are kept; older ones are evicted by
        modification time.
        """
        directory = self.cfg.model.ckpts_path
        os.makedirs(directory, exist_ok=True)

        filename = f"{_CKPT_PREFIX}{event}{_CKPT_SUFFIX}"
        path = os.path.join(directory, filename)
        torch.save(self.build_checkpoint_payload(), path)

        with open(os.path.join(directory, _LATEST_POINTER), "w") as handle:
            handle.write(filename)

        self._evict_old_ckpts(directory)
        return path

    def _evict_old_ckpts(self, directory: str) -> None:
        """Keep only the newest ``max_ckpts`` checkpoints in ``directory``."""
        cap = int(self.cfg.model.max_ckpts)
        if cap <= 0:
            return
        existing = [
            name
            for name in os.listdir(directory)
            if name.startswith(_CKPT_PREFIX) and name.endswith(_CKPT_SUFFIX)
        ]
        # Oldest first; the name breaks ties on filesystems whose timestamps
        # are coarser than a save takes.
        existing.sort(
            key=lambda name: (os.stat(os.path.join(directory, name)).st_mtime_ns, name)
        )
        for name in existing[: max(0, len(existing) - cap)]:
            os.remove(os.path.join(directory, name))
