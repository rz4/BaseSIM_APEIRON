from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from apeiron.config.configuration import Config
from apeiron.experiment.determinism import set_rng_state
from apeiron.experiment.run import Run
from apeiron.model.torch_model_harness import BaseModelHarness
from apeiron.profilers import FLOPSProfiler
from apeiron.training.updater.create_updater import create_updater
from apeiron.logger import get_logger


class ContinuousTrainer:
    """Trainer for continuous/continual learning with drift handling."""

    def __init__(
        self,
        cfg: Config,
        modelHarness: BaseModelHarness,
        logger: Any,
        profiler: Optional[FLOPSProfiler],
        run: Optional[Run] = None,
    ) -> None:
        """Initialize the continuous trainer with config, model, logger, and profiler.

        ``run`` is the run directory when there is one; the trainer records
        what each round of learning started from and achieved. None reproduces
        legacy behavior.
        """
        self.modelHarness = modelHarness
        self.cfg = cfg
        self.logger = logger
        self._run = run

        self.profiler = profiler
        self.criterion = modelHarness.get_criterion()
        self.optimizer = modelHarness.get_optmizer()

        self.cl_updater = create_updater(cfg=self.cfg, modelHarness=self.modelHarness)

        # Batches each training loader has handed out during the current
        # round, where its samplers stood when the round began, and the
        # callback the monitor uses to save state inside one.
        self._consumed: dict[str, int] = {}
        # Where the training loaders have got to. The inner loop advances them
        # and writes them back here, because the outer loop has to hand the
        # advanced ones to the next iteration.
        self._train_iter: Optional[Iterator] = None
        self._hist_iter: Optional[Iterator] = None
        self._round_samplers: dict[str, int] = {}
        self._round_pre_cur: Optional[list[float]] = None
        self._round_pre_hist: Optional[list[float]] = None
        self.on_inner_step: Optional[Any] = None

    def _record(self, kind: str, **payload: object) -> None:
        """Append an event to the run's log. No-op without a run directory."""
        if self._run is not None:
            self._run.record(kind, **payload)

    @staticmethod
    def _metric_values(values: Optional[Any]) -> Optional[list[float]]:
        return None if values is None else [float(v) for v in values]

    def _fast_forward(
        self, current_iter: Iterator, loader: DataLoader, count: int
    ) -> Iterator:
        """Take and discard ``count`` batches, wrapping around as the run did."""
        for _ in range(count):
            try:
                next(current_iter)
            except StopIteration:
                current_iter = iter(loader)
                next(current_iter)
        return current_iter

    def round_state(self, drift_event_id: int, iteration: int) -> dict[str, Any]:
        """Enough to re-enter this round at the next iteration.

        The samplers are recorded as they stood when the round *started*, not
        as they stand now: resuming rebuilds the loaders and replays their
        position, so what matters is where the replay begins.
        """
        return {
            "drift_event_id": drift_event_id,
            "next_iteration": iteration + 1,
            "pre_cur": self._round_pre_cur,
            "pre_hist": self._round_pre_hist,
            "consumed": dict(self._consumed),
            "samplers": dict(self._round_samplers),
        }

    def state_dict(self) -> dict[str, Any]:
        """Training state that has to survive a restart.

        A CL round is replayed from its start on resume, so nothing from inside
        a round is here -- only the optimizer and whatever memory the updater
        carries between drift events.
        """
        return {
            "optimizer": self.optimizer.state_dict(),
            "updater": self.cl_updater.state_dict(),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore what :meth:`state_dict` saved."""
        if "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
        if "updater" in state:
            self.cl_updater.load_state_dict(state["updater"])

    def _safe_next(
        self,
        current_iter: Iterator,
        loader: DataLoader,
        min_batch: Optional[int] = None,
        role: str = "",
    ) -> tuple[Iterator, list[torch.Tensor]]:
        """Get next batch from iterator, restarting on exhaustion and enforcing min batch size.

        Counts every batch actually taken, including the undersized ones this
        skips past. A resume replays that count rather than working it out
        from the iteration number: the two differ as soon as a short batch has
        been skipped, and the run then diverges in a way that is hard to see.
        """
        while True:
            try:
                batch = next(current_iter)
            except StopIteration:
                current_iter = iter(loader)
                batch = next(current_iter)
            if role:
                self._consumed[role] = self._consumed.get(role, 0) + 1

            if min_batch is None:
                return current_iter, [b.to(self.cfg.device) for b in batch]

            # Try to enforce batch-size on the second element (x, y)
            try:
                y = batch[1]
                if getattr(y, "shape", None) is not None and y.shape[0] >= min_batch:
                    return current_iter, [b.to(self.cfg.device) for b in batch]
            except (IndexError, TypeError):
                # If we cannot inspect batch size, just accept the batch
                return current_iter, [b.to(self.cfg.device) for b in batch]

    def _log_validation(
        self,
        tag: str,
        cur: Optional[list],
        hist: Optional[list],
        drift_event_id: int,
    ) -> None:
        """Record every eval metric by name, for both domains.

        ``eval()`` returns a positional list; ``eval_metrics`` holds the labels
        in the same order.
        """
        logger = get_logger(__name__)
        names = self.modelHarness.eval_metrics
        payload: dict[str, float] = {"drift_event_id": drift_event_id}
        for domain, values in (("cur", cur), ("hist", hist)):
            for name, value in zip(names, values or ()):
                payload[f"val_{tag}_{domain}_{name}"] = float(value)
        logger.stage("eval")
        # increment=False: annotate the CL round's step, do not advance it.
        logger.log(payload, commit=False, increment=False)

    def compute_bwt(self, metric_index: int = 0) -> Optional[float]:
        """Backward transfer for the task that just finished adapting.

        ``BWT = (1/(T-1)) * sum_{i<T} ( R[T][i] - R[i][i] )``

        where a *task* is one drift event, ``R[T][i]`` is the current model's
        score on task ``i``'s validation split and ``R[i][i]`` is the score on
        that same split recorded right after adapting to it. It therefore
        compares one task across two model states -- the definition of
        forgetting -- rather than comparing different tasks at one state.

        Sign follows the raw metric, so the reading depends on the metric's
        direction: with a higher-is-better metric (accuracy) negative means
        forgetting, while with a lower-is-better one (SLAC-FEL's MAE) *positive*
        means forgetting.

        :param metric_index: which entry of ``eval_metrics`` to use, matching the
            index behind ``test_curr_acc``/``test_hist_acc``.
        :type metric_index: int

        :return: BWT over the retained past tasks, or None before any task has
            been registered (the first drift event, where the sum is empty).
        :rtype: Optional[float]
        """
        past_metrics = self.modelHarness.eval_past_tasks()
        if not past_metrics:
            return None

        diagonals = self.modelHarness.task_diagonals
        deltas = [
            row[metric_index] - diagonal[metric_index]
            for row, diagonal in zip(past_metrics, diagonals)
        ]
        return sum(deltas) / len(deltas)

    def outer_cl_training_loop(
        self,
        drift_event_id: int = 0,
        resume: Optional[dict[str, Any]] = None,
    ) -> int:
        """Run the outer continuous learning training loop for a drift event.

        With ``resume``, the round is re-entered where it was interrupted: the
        evaluation that preceded it already happened and its result is carried
        in, the updater's preparation already ran, and the loaders are wound
        forward to the batch the next iteration would have seen.
        """
        logger = get_logger(__name__)
        cur_train_loader, cur_test_loader = self.modelHarness.get_train_dataloaders()
        hist_train_loader, hist_test_loader = self.modelHarness.get_hist_dataloaders()

        if resume is None:
            self._consumed = {}
            self._round_samplers = self.modelHarness.sampler_state()
        else:
            self._consumed = dict(resume.get("consumed") or {})
            self._round_samplers = dict(resume.get("samplers") or {})
            self.modelHarness.load_sampler_state(self._round_samplers)

        train_iter: Iterator = iter(cur_train_loader)
        hist_train_iter: Optional[Iterator] = (
            iter(hist_train_loader) if hist_train_loader is not None else None
        )
        self._train_iter, self._hist_iter = train_iter, hist_train_iter

        start_iteration = 0
        if resume is None:
            cur_validation_metrics = self.modelHarness.eval()
            hist_validation_metrics = self.modelHarness.history_eval()
        else:
            start_iteration = int(resume["next_iteration"])
            cur_validation_metrics = list(resume["pre_cur"])
            hist_validation_metrics = (
                list(resume["pre_hist"]) if resume.get("pre_hist") is not None else None
            )
            train_iter = self._fast_forward(
                train_iter, cur_train_loader, self._consumed.get("train", 0)
            )
            if hist_train_iter is not None and hist_train_loader is not None:
                hist_train_iter = self._fast_forward(
                    hist_train_iter, hist_train_loader, self._consumed.get("hist", 0)
                )
            self._train_iter, self._hist_iter = train_iter, hist_train_iter
            if resume.get("rng") is not None:
                # After the replay, not before: winding loaders forward is not
                # what the interrupted run was doing at this point.
                set_rng_state(resume["rng"])

        self._round_pre_cur = self._metric_values(cur_validation_metrics)
        self._round_pre_hist = self._metric_values(hist_validation_metrics)

        # R[i-1][i]: this window scored by the model that has not yet adapted to
        # it. Kept for the FWT delta once the post-CL score (R[i][i]) is in.
        pre_cl_validation_metrics = cur_validation_metrics

        if resume is None:
            self._record(
                "cl_started",
                drift_event_id=drift_event_id,
                update_mode=self.cfg.continual_learning.update_mode,
                metrics=list(self.modelHarness.eval_metrics),
                pre_cur=self._round_pre_cur,
                pre_hist=self._round_pre_hist,
            )

        logger.info("==== Continual Learning ====")
        logger.info("\tInitial test acc: {}".format(cur_validation_metrics[0]), level=1)
        if hist_validation_metrics is not None:
            logger.info(
                "\tInitial historical test acc: {}".format(hist_validation_metrics[0]),
                level=1,
            )
        else:
            logger.info("\tNo historical data available for evaluation", level=1)

        self.modelHarness.model.train()
        # 2) run the outer loop
        desc = "CL Updates (drift_event_id={})".format(drift_event_id)
        progress_bar = tqdm(
            range(start_iteration, self.cfg.train.max_iter), desc=desc, leave=True
        )
        if resume is None:
            # On a resume this already ran, and the accumulators it set up
            # came back with the updater's state.
            self.cl_updater.cl_preprocessing()

        iter_count = self.cfg.train.max_iter
        if self.cl_updater is not None:  # default: do nothing
            for iter_count in progress_bar:
                generation_loss, forgetting_loss = self.inner_cl_training_loop(
                    iter_count=iter_count,
                    cur_train_loader=cur_train_loader,
                    # Where the previous iteration left off, not where this
                    # round began: passing the original iterator back means
                    # that once it is exhausted every later step restarts the
                    # loader and takes only the first batch of a new shuffle.
                    train_iter=self._train_iter,
                    hist_train_loader=hist_train_loader,
                    hist_train_iter=self._hist_iter,
                )

                logger.stage("cl")
                logger.log(
                    {
                        "jvp_reg_total_loss": generation_loss + forgetting_loss,
                        "jvp_reg_forgetting_loss": forgetting_loss,
                        "jvp_reg_generation_loss": generation_loss,
                        "drift_event_id": drift_event_id,
                    },
                    # commit=iter_count < (cfg.continuous_learning.max_iter - 1),
                )

                # Explicitly cleanup batch tensors to free GPU memory
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if self.on_inner_step is not None:
                    self.on_inner_step(drift_event_id, iter_count)

        self.cl_updater.cl_postprocessing()

        cur_validation_metrics = self.modelHarness.eval()
        hist_validation_metrics = self.modelHarness.history_eval()
        self._log_validation(
            "post", cur_validation_metrics, hist_validation_metrics, drift_event_id
        )

        logger.info(f"\tTest Accuracy: {cur_validation_metrics[0]:.1f}%", level=1)
        if hist_validation_metrics is not None:
            logger.info(
                f"\tHist Test Accuracy: {hist_validation_metrics[0]:.1f}%",
                level=1,
            )

        else:
            logger.info("\tNo historical data available for evaluation", level=1)

        # FWT = R[i][i] - R[i-1][i]: how much adapting to this window moved the
        # score on it, i.e. the gain CL delivered on the task that triggered.
        # Available at every drift event, including the first.
        fwt = cur_validation_metrics[0] - pre_cl_validation_metrics[0]
        logger.info(f"\tFWT: {fwt:.4g}", level=1)

        bwt = self.compute_bwt()
        if bwt is not None:
            logger.info(f"\tBWT: {bwt:.4g}", level=1)

        self._record(
            "cl_finished",
            drift_event_id=drift_event_id,
            iterations=iter_count + 1,
            post_cur=self._metric_values(cur_validation_metrics),
            post_hist=self._metric_values(hist_validation_metrics),
            fwt=float(fwt),
            bwt=None if bwt is None else float(bwt),
        )

        logger.stage("eval")
        eval_metrics: dict[str, float] = {
            "test_curr_acc": cur_validation_metrics[0],
            "test_pre_cl_acc": pre_cl_validation_metrics[0],
            "fwt": fwt,
        }
        if hist_validation_metrics is not None:
            eval_metrics["test_hist_acc"] = hist_validation_metrics[0]
        if bwt is not None:
            eval_metrics["bwt"] = bwt
        logger.log(eval_metrics, commit=False)

        # Register *after* BWT so this event's window becomes task T only for
        # subsequent events -- R[T][T] belongs on the diagonal, not in the sum.
        self.modelHarness.register_task(cur_validation_metrics)

        if self.profiler:
            flops_perf = self.profiler.get_performance()
            self.profiler.print_performance()
            logger.stage("cl")
            self.logger.log(
                {
                    **{f"cperf_{k}": v for k, v in flops_perf.items()},
                },
            )

        return 0

    def inner_cl_training_loop(
        self,
        iter_count: int,
        cur_train_loader: DataLoader,
        train_iter: Iterator,
        hist_train_loader: Optional[DataLoader] = None,
        hist_train_iter: Optional[Iterator] = None,
    ) -> tuple[float, float]:
        """Run a single inner training iteration with forward/backward and optimizer step."""
        self.optimizer.zero_grad()
        self.cl_updater.update_pre_fwd_bwd()

        # Forward and backward
        loss = 0.0
        for step in range(self.cfg.train.grad_accumulation_steps):
            train_iter, train_batch = self._safe_next(
                train_iter,
                cur_train_loader,
                min_batch=self.cfg.train.batch_size,
                role="train",
            )
            if hist_train_iter is not None and hist_train_loader is not None:
                hist_train_iter, hist_train_batch = self._safe_next(
                    hist_train_iter,
                    hist_train_loader,
                    min_batch=self.cfg.train.batch_size,
                    role="hist",
                )
            else:
                hist_train_batch = None

            # Cast batches to tuple type expected by fwd_bwd
            train_batch_tuple = (train_batch[0], train_batch[1])
            hist_batch_tuple = (
                (hist_train_batch[0], hist_train_batch[1])
                if hist_train_batch is not None
                else None
            )

            # Run profiler for forward and backward after warmup for one of the grad acc steps.
            if self.profiler and iter_count > self.profiler.warmup_iters and step == 0:
                with self.profiler.measure_flops(tag="update_fwd_bwd"):
                    loss += self.cl_updater.fwd_bwd(train_batch_tuple, hist_batch_tuple)
            else:
                loss += self.cl_updater.fwd_bwd(train_batch_tuple, hist_batch_tuple)

        reg_loss = self.cl_updater.update_post_fwd_bwd()

        # 3) Update with optimizer
        if self.profiler and iter_count > self.profiler.warmup_iters:
            with self.profiler.measure_flops_optimizer(
                tag="optimizer", model=self.modelHarness.model, device=self.cfg.device
            ):
                self.optimizer.step()
        else:
            self.optimizer.step()

        self.cl_updater.update_post_optimizer_call()

        # The caller needs where these got to, not where they started.
        self._train_iter, self._hist_iter = train_iter, hist_train_iter

        return loss, reg_loss
