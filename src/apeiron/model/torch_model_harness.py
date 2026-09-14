from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional, Callable, Tuple, List, Dict

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader, TensorDataset
from torch.optim import Optimizer

from apeiron.config.configuration import Config

if TYPE_CHECKING:
    from apeiron.experiment.residency import DataResolver
    from apeiron.experiment.sources import WindowSpec

MetricFn = Callable[[Tensor, Tensor], Any]
CriterionFn = Callable[[Tensor, Tensor], Tensor]


class BaseModelHarness(ABC):
    """
    Members
    -------
    self.model : nn.Module
    self.cfg   : Dict[str, Any]   # e.g., {"device": "cuda", ...}

    You must implement:
      - get_loader(self)    -> DataLoader | Iterable
      - get_criterion(self) -> CriterionFn
    """

    def __init__(self, cfg: Config, model: nn.Module):
        self.model = model
        self.cfg = cfg
        device = torch.device(self.cfg.device)
        self.model.to(device)

        self.eval_metrics: Dict[str, MetricFn] = {}

        # Injected by the framework in experiment mode when the harness
        # declares its windows (see describe_window); None otherwise.
        self.data_resolver: Optional["DataResolver"] = None

        # One entry per drift event, oldest first, paired with R[i][i] (see
        # register_task). The first element is either a frozen in-memory
        # DataLoader (legacy) or an int window index (declarative -- rebuilt
        # on demand via build_window_eval_loader).
        self._task_records: List[Tuple[Any, List[float]]] = []
        self.max_task_records: int = 50

    @abstractmethod
    def get_optmizer(self) -> Optimizer:
        """
        Returns the optimizer object compatible with the trainable parameters
        supports parameter groups for, e.g., different learning rates
        """
        raise NotImplementedError

    # ----- subclass hooks -----

    @abstractmethod
    def update_data_stream(self) -> None:
        """
        Updates the data stream potentially leading to data drift
        """
        raise NotImplementedError

    @abstractmethod
    def get_stream_dataloader(self) -> DataLoader:
        """
        Returns a training and validation dataloader compatible with the model input
        that will be used for continual learning
        """
        raise NotImplementedError

    @abstractmethod
    def get_hist_dataloaders(
        self,
    ) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
        """
        Returns a training and validation dataloader with historical data (to measure drift) compatible with the model input
        If there is no historical data, return None
        """
        raise NotImplementedError

    @torch.no_grad()
    def get_train_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Returns a training and validation dataloader compatible with the model input
        that will be used to loop over for inference
        """
        raise NotImplementedError

    @abstractmethod
    def get_criterion(self) -> CriterionFn:
        """Return a loss function compatible with model output and dataloader labels"""
        raise NotImplementedError

    # ----- declarative data protocol (optional) -----

    def describe_window(self, window: int) -> Optional["WindowSpec"]:
        """Declare stream window ``window``'s data needs, or None.

        Returning a :class:`~apeiron.experiment.sources.WindowSpec` opts
        into framework-managed residency: before ``update_data_stream()``
        for that window, the monitor materializes the spec's objects into
        the experiment artifact store (pinned for the run) and prefetches
        the next window's spec; the harness reads the local files under
        ``self.data_resolver.store_root``. A window index past the end of
        the stream should return None (stops prefetch).

        Default None = imperative legacy mode: the harness fetches its own
        data inside ``update_data_stream()``.
        """
        return None

    # ----- resilience: harness-owned RNG state (loader shuffle epochs) -----

    def rng_state_dict(self) -> Dict[str, int]:
        """Epoch counters of the harness's stateful shuffling samplers.

        Shuffling loaders that persist across CL events within a window
        carry hidden state (how many epoch permutations they have drawn).
        Harnesses using :class:`~apeiron.experiment.determinism.EpochSeededSampler`
        report ``{role: epochs_started}`` here so resilience snapshots can
        restore exact shuffle positions. Default: no stateful samplers.
        """
        return {}

    def load_rng_state_dict(self, state: Dict[str, int], in_cl: bool) -> None:
        """Restore sampler epoch counters captured by :meth:`rng_state_dict`.

        :param in_cl: True when the snapshot was taken inside a CL loop --
            the training sampler's current epoch was then in progress and
            must be re-drawn (counter - 1); at a monitoring snapshot the
            training loader sits at an epoch boundary (counter unchanged).
        """
        pass

    # ----- batch handling (override for non-tuple batches) -----

    def batch_to_device(self, batch: Any, device: Any) -> Any:
        """Move a loader batch to a device, preserving its structure."""
        if isinstance(batch, dict):
            return {
                k: v.to(device) if hasattr(v, "to") else v for k, v in batch.items()
            }
        return [t.to(device) if hasattr(t, "to") else t for t in batch]

    def batch_size_of(self, batch: Any) -> Optional[int]:
        """Sample count of a loader batch, or None if not inspectable."""
        try:
            probe = next(iter(batch.values())) if isinstance(batch, dict) else batch[1]
        except (IndexError, TypeError, KeyError, StopIteration):
            return None
        shape = getattr(probe, "shape", None)
        return int(shape[0]) if shape is not None and len(shape) > 0 else None

    # ----- helpers -----
    def _unpack(self, batch: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        """
        Returns the input and label from a batch
        Note: override for compatibility with subclassed dataloader

        :param batch: batch of data from dataloader
        :type batch: Tuple[Tensor, Tensor]

        :return: input and label of the batch
        :rtype: Tuple[Tensor, Tensor]
        """
        x, y = batch
        return x, y

    @staticmethod
    def _to_scalar(x: Tensor | float) -> float:
        if isinstance(x, torch.Tensor):
            return float(x.mean().item() if x.ndim > 0 else x.item())
        return float(x)

    @torch.no_grad()
    def _eval_loader(self, loader: DataLoader) -> List[float]:
        """Stream over batches; return mean(metric) over batches (order preserved)."""
        self.model.eval()
        sums = [0.0 for _ in self.eval_metrics]
        counts = [0 for _ in self.eval_metrics]

        for batch in loader:  # assumes iterable
            x, y = self._unpack(batch)
            x, y = x.to(self.cfg.device), y.to(self.cfg.device)

            # TODO: Add cuda amp support later. Needs config entry for amp
            # if self.cfg.amp:

            #     with torch.autocast(
            #         device_type=self.device.type,
            #         dtype=(
            #             torch.float16 if self.device.type == "cuda" else torch.bfloat16
            #         ),
            #     ):
            #         y_hat = self.model(x)
            # else:
            y_hat = self.model(x)

            batch_size = y.shape[0]
            for i, m in enumerate(self.eval_metrics.values()):
                metric_value = self._to_scalar(m(y_hat, y))
                # For metrics that return percentages (like accuracy), we need to
                # convert back to counts for proper averaging across variable batch sizes
                sums[i] += metric_value * batch_size
                counts[i] += batch_size

        if counts[0] == 0:
            raise RuntimeError("Empty loader: nothing to evaluate.")

        return [s / c for s, c in zip(sums, counts)]

    @torch.no_grad()
    def eval(self) -> List[float]:
        """Stream over batches; return mean(metric) over batches (order preserved)."""
        return self._eval_loader(self.get_train_dataloaders()[1])

    @torch.no_grad()
    def history_eval(self) -> Optional[List[float]]:
        """Stream over batches; return mean(metric) over batches (order preserved).

        Returns None if no historical data is available.
        """
        hist_loaders = self.get_hist_dataloaders()
        if hist_loaders is None or hist_loaders[1] is None:
            return None

        return self._eval_loader(hist_loaders[1])

    # ----- per-task evaluation (train-test matrix R) -----

    def build_window_eval_loader(self, window: int) -> Optional[DataLoader]:
        """Rebuild the eval set of a PAST stream window, or None (default).

        Overriding this opts the transfer-metric registry into declarative
        mode: :meth:`register_task` stores only the window index (and the
        deterministic window definition rebuilds the data on demand)
        instead of freezing the split into memory. This keeps registry
        memory constant and makes the registry snapshot-able for resume.
        """
        return None

    @property
    def _declarative_tasks(self) -> bool:
        return (
            type(self).build_window_eval_loader
            is not BaseModelHarness.build_window_eval_loader
        )

    def register_task(self, diagonal_metrics: List[float], window: int = -1) -> None:
        """Record the task just finished so later events can measure forgetting.

        A *task* is one drift event: the window the detector fired on and the CL
        loop adapted to. Its eval set is stored either as a window REFERENCE
        (harnesses overriding :meth:`build_window_eval_loader`; rebuilt on
        demand from the deterministic window definition) or, legacy, by
        freezing the validation split into memory. Either travels with
        ``diagonal_metrics`` -- ``R[i][i]``, the score on the window measured
        right after adapting to it -- so eviction can never misalign a task's
        eval set from its diagonal.

        :param diagonal_metrics: ``eval()`` output for the current window,
            taken after the CL loop finished.
        :param window: the window index being registered (declarative mode);
            ignored in legacy mode.
        """
        if self._declarative_tasks:
            self._task_records.append((window, list(diagonal_metrics)))
        else:
            xs: List[Tensor] = []
            ys: List[Tensor] = []
            for batch in self.get_train_dataloaders()[1]:
                x, y = self._unpack(batch)
                xs.append(x.detach().cpu().clone())
                ys.append(y.detach().cpu().clone())

            frozen = DataLoader(
                TensorDataset(torch.cat(xs), torch.cat(ys)),
                batch_size=self.cfg.train.batch_size,
                shuffle=False,
            )
            self._task_records.append((frozen, list(diagonal_metrics)))

        # Cap retained tasks; BWT then averages over the surviving ones.
        while len(self._task_records) > self.max_task_records:
            self._task_records.pop(0)

    @torch.no_grad()
    def eval_past_tasks(self) -> List[List[float]]:
        """Score the current model on every registered task's eval set.

        Returns row ``T`` of the train-test matrix below the diagonal --
        ``[R[T][i] for i < T]``, oldest task first, index-aligned with
        :attr:`task_diagonals`. Empty until at least one task is registered.
        """
        results = []
        for ref, _ in self._task_records:
            loader = ref if isinstance(ref, DataLoader) else None
            if loader is None:
                loader = self.build_window_eval_loader(int(ref))
            assert loader is not None, f"cannot rebuild eval loader for task {ref}"
            results.append(self._eval_loader(loader))
        return results

    # ----- resilience: registry capture/restore (declarative mode only) -----

    def task_records_refs(self) -> Optional[List[Tuple[int, List[float]]]]:
        """Registry as (window, diagonal) references, or None in legacy mode."""
        if not self._declarative_tasks:
            return None
        return [(int(ref), list(diag)) for ref, diag in self._task_records]

    def restore_task_records(self, refs: List[Tuple[int, List[float]]]) -> None:
        self._task_records = [(int(w), list(d)) for w, d in refs]

    @property
    def task_diagonals(self) -> List[List[float]]:
        """``R[i][i]`` per registered task, oldest first.

        Index-aligned with :meth:`eval_past_tasks`.
        """
        return [diagonal for _, diagonal in self._task_records]

    @property
    def ckpts_enabled(self) -> bool:
        return self.cfg.model.max_ckpts > 0 and bool(self.cfg.model.ckpts_path)

    def build_checkpoint_payload(self) -> Any:
        """Build the checkpoint object to save.

        Subclasses can override this to include additional metadata beyond weights
        (e.g., preprocessing scalers, feature names, architecture parameters)
        so that saved checkpoints match the format expected by the loader.

        Returns
        -------
        By default, returns ``model.state_dict()`` (weights only).
        """
        return self.model.state_dict()

    def save_ckpt(self, event: int) -> str:
        """Persist model state, evict oldest when over budget.

        Analysis checkpoints live under ``<ckpts_path>/analysis/``;
        resilience snapshots (full-state, for restart) live beside them
        under ``resilience/`` and are managed by the monitor.
        """
        d = Path(self.cfg.model.ckpts_path) / "analysis"
        d.mkdir(parents=True, exist_ok=True)

        fname = f"drift_adaptation_{event}.pt"
        payload = self.build_checkpoint_payload()
        torch.save(payload, d / fname)
        (d / "latest").write_text(fname)

        # Guillotine the oldest survivors. Metric-based retention policies
        # ("best_current"/"best_hist") are applied by the monitor via
        # apeiron.experiment.retention instead -- inline FIFO here would
        # delete candidates before they can be scored.
        if self.cfg.model.ckpt_retention == "latest":
            alive = sorted(
                d.glob("drift_adaptation_*.pt"), key=lambda p: p.stat().st_mtime
            )
            while len(alive) > self.cfg.model.max_ckpts:
                alive.pop(0).unlink()

        return str(d / fname)
