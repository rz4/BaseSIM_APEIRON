"""Continual-learning trainer driven by a drift event.

``ContinuousMonitor`` watches a stream, and when its detector fires it hands
control to :meth:`ContinuousTrainer.outer_cl_training_loop`.  One call to that
method is one *round* of adaptation: the window that fired is scored, the
configured updater is driven for ``train.max_iter`` optimizer steps over that
window, and the round is scored again so the transfer numbers a continual
learning paper reports (forward transfer, backward transfer) can be recorded.

Logging layout of one round, which downstream analysis depends on:

``cl`` stage
    one row per inner iteration -- ``jvp_reg_total_loss``,
    ``jvp_reg_forgetting_loss``, ``jvp_reg_generation_loss``,
    ``drift_event_id`` -- then one closing ``cperf_*`` row summarising the
    profiler.
``eval`` stage
    the post-round validation row (which *annotates* the round's final step
    rather than advancing it), then the transfer row that closes the round.
"""

from __future__ import annotations

from typing import Any, Iterator, Optional, Sequence, Tuple


from apeiron.config.configuration import Config
from apeiron.logger.logger import get_logger
from apeiron.model.torch_model_harness import BaseModelHarness
from apeiron.profilers.count_flops import FLOPSProfiler
from apeiron.training.updater.base import BaseUpdater
from apeiron.training.updater.create_updater import create_updater

Batch = Any


class ContinuousTrainer:
    """Adapt a model to one drift event at a time.

    Attributes:
        cfg: Resolved configuration.
        modelHarness: Harness supplying the model, data and criterion.
        logger: Logger handed in by the caller; used for console output.
        profiler: Optional :class:`FLOPSProfiler`; ``None`` disables measuring.
        cl_updater: The update strategy chosen by ``continual_learning.update_mode``.
        optimizer: The optimizer the harness built for the model.
    """

    def __init__(
        self,
        cfg: Config,
        modelHarness: BaseModelHarness,
        logger: Any = None,
        profiler: Optional[FLOPSProfiler] = None,
    ) -> None:
        self.cfg = cfg
        self.modelHarness = modelHarness
        self.logger = logger
        self.profiler = profiler

        self.model = modelHarness.model
        self.optimizer = modelHarness.get_optimizer()
        self.cl_updater: BaseUpdater = create_updater(
            cfg=cfg, modelHarness=modelHarness
        )

    # ------------------------------------------------------------------
    # batching
    # ------------------------------------------------------------------
    def _safe_next(
        self,
        current_iter: Iterator[Batch],
        loader: Any,
        min_batch: Optional[int] = None,
    ) -> Tuple[Iterator[Batch], Batch]:
        """Pull the next batch, restarting ``loader`` once it is exhausted.

        The iterator is returned alongside the batch so the caller can carry on
        from where this left off: a round of learning walks its window rather
        than restarting it on every step.

        ``min_batch`` refuses a batch that is smaller than one full training
        batch.  A window smaller than ``train.batch_size`` can never satisfy
        it, and restarting the loader would retry the same rejected data for
        ever, so the second pass raises instead of spinning.

        A batch whose second element cannot be inspected is accepted: "cannot
        measure this" is not "this is too small".
        """
        device = self.cfg.device
        largest = 0
        restarted = False

        while True:
            try:
                batch = next(current_iter)
            except StopIteration:
                if restarted:
                    raise ValueError(self._short_window_message(min_batch, largest))
                current_iter = iter(loader)
                restarted = True
                try:
                    batch = next(current_iter)
                except StopIteration:
                    raise ValueError(
                        "the loader yielded no batches at all; there is nothing "
                        "in this window to train on"
                    ) from None

            try:
                shape = getattr(batch[1], "shape", None)
            except (IndexError, TypeError):
                # If we cannot inspect batch size, just accept the batch
                return current_iter, [b.to(device) for b in batch]

            if min_batch is None:
                return current_iter, [b.to(device) for b in batch]

            size = int(shape[0]) if shape is not None and len(shape) else 0
            if size >= min_batch:
                return current_iter, [b.to(device) for b in batch]

            largest = max(largest, size)

    @staticmethod
    def _short_window_message(min_batch: Optional[int], largest: int) -> str:
        return (
            f"no batch in this loader holds the {min_batch} samples min_batch "
            f"asks for -- the largest is {largest}. The window is smaller than "
            f"one batch."
        )

    # ------------------------------------------------------------------
    # logging helpers
    # ------------------------------------------------------------------
    def _log_validation(
        self,
        tag: str,
        cur_metrics: Optional[Sequence[float]],
        hist_metrics: Optional[Sequence[float]],
        drift_event_id: int,
    ) -> None:
        """Record one validation row, naming every metric by its harness label.

        ``eval()`` returns a positional list and ``eval_metrics`` holds the
        labels in the same order, so the two are zipped.  An absent domain
        contributes nothing rather than a column of ``None``.
        """
        logger = get_logger(__name__)
        names = list(self.modelHarness.eval_metrics.keys())

        payload: dict[str, Any] = {"drift_event_id": drift_event_id}
        if cur_metrics is not None:
            for name, value in zip(names, cur_metrics):
                payload[f"val_{tag}_cur_{name}"] = value
        if hist_metrics is not None:
            for name, value in zip(names, hist_metrics):
                payload[f"val_{tag}_hist_{name}"] = value

        logger.stage("eval")
        # increment=False: annotate the CL round's step, do not advance it.
        logger.log(payload, commit=False, increment=False)

    # ------------------------------------------------------------------
    # transfer metrics
    # ------------------------------------------------------------------
    def compute_bwt(self, metric_index: int = 0) -> Optional[float]:
        """Backward transfer: mean over retained tasks of ``R[T][i] - R[i][i]``.

        ``None`` before any task is registered -- the sum is empty and ``0.0``
        would read as "no forgetting" rather than "not yet measurable".

        The sign follows the raw metric: with a higher-is-better metric a drop
        reads negative, while a lower-is-better one (MAE) reads POSITIVE when
        it degrades.
        """
        rows = self.modelHarness.eval_past_tasks()
        diagonals = self.modelHarness.task_diagonals

        deltas = [
            float(row[metric_index]) - float(diagonal[metric_index])
            for row, diagonal in zip(rows, diagonals)
        ]
        if not deltas:
            return None
        return sum(deltas) / len(deltas)

    # ------------------------------------------------------------------
    # the loops
    # ------------------------------------------------------------------
    def outer_cl_training_loop(self, drift_event_id: int = 0) -> int:
        """Adapt the model to one drift event; returns 0.

        :param drift_event_id: Identifier of the drift event that triggered
            this round, carried on every row the round logs.
        :type drift_event_id: int
        """
        logger = get_logger(__name__)

        cur_train_loader, _cur_val_loader = self.modelHarness.get_train_dataloaders()
        hist_loaders = self.modelHarness.get_hist_dataloaders()
        hist_train_loader = hist_loaders[0] if hist_loaders is not None else None

        # R[i-1][i]: the window that fired, scored before adapting to it.
        pre_cl_metrics = self.modelHarness.eval()
        _pre_hist_metrics = self.modelHarness.history_eval()

        self.cl_updater.cl_preprocessing()

        train_iter: Iterator[Batch] = iter(cur_train_loader)
        hist_iter: Optional[Iterator[Batch]] = (
            iter(hist_train_loader) if hist_train_loader is not None else None
        )

        for iter_count in range(1, int(self.cfg.train.max_iter) + 1):
            (
                gen_loss,
                reg_loss,
                train_iter,
                hist_iter,
            ) = self.inner_cl_training_loop(
                iter_count=iter_count,
                cur_train_loader=cur_train_loader,
                train_iter=train_iter,
                hist_train_loader=hist_train_loader,
                hist_iter=hist_iter,
            )

            logger.stage("cl")
            logger.log(
                {
                    "jvp_reg_total_loss": gen_loss + reg_loss,
                    "jvp_reg_forgetting_loss": reg_loss,
                    "jvp_reg_generation_loss": gen_loss,
                    "drift_event_id": drift_event_id,
                }
            )

        self.cl_updater.cl_postprocessing()

        # R[i][i]: the same window, scored after adapting to it.
        post_cl_metrics = self.modelHarness.eval()
        post_hist_metrics = self.modelHarness.history_eval()

        self._log_validation("post", post_cl_metrics, post_hist_metrics, drift_event_id)

        # BWT is measured against the tasks registered BEFORE this one, so it
        # is computed before this round joins the registry.
        bwt = self.compute_bwt()
        self.modelHarness.register_task(post_cl_metrics)

        transfer: dict[str, Any] = {
            # The event that produced these numbers. Forward and backward
            # transfer are per-adaptation quantities, and without this key
            # the row could not be joined back to the drift event it scores
            # (claims/LOG.md A31).
            "drift_event_id": drift_event_id,
            "test_pre_cl_acc": pre_cl_metrics[0],
            "test_curr_acc": post_cl_metrics[0],
            # FWT = R[i][i] - R[i-1][i]: what adapting bought on this window.
            "fwt": float(post_cl_metrics[0]) - float(pre_cl_metrics[0]),
        }
        if post_hist_metrics is not None:
            transfer["test_hist_acc"] = post_hist_metrics[0]
        if bwt is not None:
            transfer["bwt"] = bwt

        logger.stage("eval")
        logger.log(transfer, commit=False)

        if self.profiler is not None:
            performance = self.profiler.get_performance()
            if performance:
                logger.stage("cl")
                logger.log({f"cperf_{k}": v for k, v in performance.items()})
                self.profiler.print_performance(logger=self.logger, level=1)

        return 0

    def inner_cl_training_loop(
        self,
        iter_count: int,
        cur_train_loader: Any,
        train_iter: Iterator[Batch],
        hist_train_loader: Optional[Any] = None,
        hist_iter: Optional[Iterator[Batch]] = None,
    ) -> Tuple[float, float, Iterator[Batch], Optional[Iterator[Batch]]]:
        """One optimizer step, over ``train.grad_accumulation_steps`` batches.

        Returns ``(generation_loss, regularization_loss, train_iter,
        hist_iter)``.  The iterators come back so the caller can carry on from
        where this step left off instead of restarting its window.

        Nothing is profiled during warmup, and the measured block is one
        optimizer step rather than one accumulation step: ``get_performance()``
        averages over blocks, so the block has to be the unit it means to
        average.
        """
        batch_size = int(self.cfg.train.batch_size)
        accumulation_steps = max(1, int(self.cfg.train.grad_accumulation_steps))
        measuring = (
            self.profiler is not None and iter_count > self.profiler.warmup_iters
        )

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        self.cl_updater.update_pre_fwd_bwd()

        gen_loss = 0.0
        for step in range(accumulation_steps):
            train_iter, cur_batch = self._safe_next(
                train_iter, cur_train_loader, min_batch=batch_size
            )

            hist_batch = None
            if hist_train_loader is not None:
                if hist_iter is None:
                    hist_iter = iter(hist_train_loader)
                hist_iter, hist_batch = self._safe_next(
                    hist_iter, hist_train_loader, min_batch=batch_size
                )

            if measuring and step == 0:
                with self.profiler.measure_flops(tag="update_fwd_bwd"):
                    gen_loss += float(self.cl_updater.fwd_bwd(cur_batch, hist_batch))
            else:
                gen_loss += float(self.cl_updater.fwd_bwd(cur_batch, hist_batch))

        reg_loss = float(self.cl_updater.update_post_fwd_bwd())

        if measuring:
            with self.profiler.measure_flops_optimizer(
                self.model, self.cfg.device, tag="optimizer"
            ):
                self.optimizer.step()
        else:
            self.optimizer.step()

        self.cl_updater.update_post_optimizer_call()

        return gen_loss, reg_loss, train_iter, hist_iter
