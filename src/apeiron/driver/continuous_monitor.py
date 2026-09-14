"""Continuous drift monitoring system.

This module implements a continuous monitoring architecture that:
- Maintains detector state across data stream updates
- Processes batches continuously until drift is detected
- Pauses for learning when drift is detected
- Resumes monitoring with updated model weights
- Automatically extends the data stream when exhausted
"""

from __future__ import annotations
from typing import TYPE_CHECKING

import torch
import numpy as np

from apeiron.config.configuration import Config
from apeiron.drift_detection.load_drift_detector import load_drift_detector
from apeiron.drift_detection.detectors.base import DriftSignal
from apeiron.profilers import FLOPSProfiler
from apeiron.logger import get_logger
from apeiron.training import ContinuousTrainer
from tqdm import tqdm

if TYPE_CHECKING:
    from apeiron.experiment.journal import Journal
    from apeiron.experiment.sources import WindowSpec
    from apeiron.model.torch_model_harness import BaseModelHarness


class ContinuousMonitor:
    """Continuous drift monitoring system.

    This class manages the continuous monitoring of a data stream, detecting
    drift and dispatching learning modules as needed.

    Attributes:
        cfg: Configuration object
        modelHarness: Model harness containing model and data loaders
        logger: Logger for metrics
        detector: Persistent drift detector instance
        metric_idx: Index of metric to monitor
        detection_interval: Number of batches between drift checks
        max_stream_updates: Maximum number of stream extensions
        stream_update_count: Number of times stream has been extended
        batch_count: Total number of batches processed
        metric_buffer: Buffer for accumulating metrics between checks
    """

    def __init__(
        self,
        cfg: Config,
        modelHarness: BaseModelHarness,
        journal: "Journal | None" = None,
    ):
        """Initialize continuous monitor.

        Args:
            cfg: Configuration object
            modelHarness: Model harness containing model and data loaders
            journal: Optional run event journal (experiment mode); every
                window advance, drift check, drift event, and checkpoint is
                recorded when present.
        """
        self.cfg = cfg
        self.modelHarness = modelHarness
        self.journal = journal
        self.logger = get_logger()

        # Create persistent detector instance
        self.detector = load_drift_detector(cfg)

        # Create performance profiler
        self.flops_profiler = FLOPSProfiler()

        # Create trainer
        self.trainer = ContinuousTrainer(
            cfg=self.cfg,
            modelHarness=self.modelHarness,
            logger=self.logger,
            profiler=self.flops_profiler,
            journal=self.journal,
        )

        # Configuration
        self.metric_idx = cfg.drift_detection.metric_index
        self.detection_interval = cfg.drift_detection.detection_interval
        self.max_stream_updates = cfg.drift_detection.max_stream_updates
        self.aggregation = cfg.drift_detection.aggregation

        # State tracking
        self.stream_update_count = 0
        self.batch_count = 0
        self.drift_event_count = 0

        # Metrics accumulation
        self.metric_buffer: list[list[float]] = []

        # Last declared window spec (harnesses opting into the declarative
        # data protocol); used for journaling and residency.
        self._last_window_spec: "WindowSpec | None" = None

        # Resilience snapshots (experiment mode with snapshot_interval > 0):
        # full-state bundles saved every N updates -- stream batches while
        # monitoring, inner iterations during CL -- plus on interrupt signal.
        self.snapshot_interval = (
            cfg.experiment.snapshot_interval if cfg.experiment is not None else 0
        )
        self.snapshots = None
        if self.snapshot_interval > 0 and cfg.model.ckpts_path:
            from pathlib import Path

            from apeiron.experiment.snapshot import SnapshotManager

            self.snapshots = SnapshotManager(Path(cfg.model.ckpts_path) / "resilience")
        self.batches_into_window = 0
        self.interrupt_requested = False  # set by signal handlers
        self._update_counter = 0
        self._updates_since_snapshot = 0
        self._resume_skip_batches = 0
        self._resume_cl: dict | None = None
        self._cl_ctx: dict | None = None
        self._resumed_from_snapshot = False
        self._pending_harness_rng: tuple | None = None
        self.trainer.on_inner_step = self._on_cl_step

        self.logger.info("==== ContinuousMonitor initialized ====", level=0)
        self.logger.info(f"\tDetector: {cfg.drift_detection.detector_name}", level=1)
        self.logger.info(f"\tMonitoring metric index: {self.metric_idx}", level=1)
        self.logger.info(
            f"\tDetection interval: {self.detection_interval} batches", level=1
        )
        self.logger.info(f"\tAggregation method: {self.aggregation}", level=1)
        self.logger.info(f"\tMax stream updates: {self.max_stream_updates}", level=1)

    def run(self) -> None:
        """Main continuous monitoring loop.

        This loop continues until max_stream_updates is reached. It processes
        batches from the data stream, checks for drift at regular intervals,
        and dispatches learning modules when drift is detected.
        """
        self.logger.info("==== Starting Continuous Monitoring ====", level=0)

        # Initialize the data stream. A continued run has a nonzero
        # stream_update_count restored from its journal; the extra calls
        # fast-forward the harness to the window where monitoring stopped
        # (window sequences are deterministic, so this replays identically).
        # Prefetch only fires for the window we will actually process.
        self.logger.info("\tInitializing first data stream...", level=1)
        resume_skip = self._resume_skip_batches  # window reset clears it below
        for w in range(self.stream_update_count + 1):
            self._advance_stream(w, prefetch_next=(w == self.stream_update_count))
        if not self._resumed_from_snapshot:
            # The replayed window was journaled before the interruption;
            # re-journaling it would break crash-equivalence.
            self._journal_window()
        self._resume_skip_batches = resume_skip
        if self._pending_harness_rng is not None:
            harness_rng, was_in_cl = self._pending_harness_rng
            self._pending_harness_rng = None
            self.modelHarness.load_rng_state_dict(harness_rng, in_cl=was_in_cl)

        # A snapshot taken mid-CL: finish the interrupted CL loop first,
        # then resume the stream at the batch where the drift had fired.
        if self._resume_cl is not None:
            resume = self._resume_cl
            self._resume_cl = None
            self.logger.info(
                f"Resuming interrupted CL loop (event {resume['drift_event_id']}, "
                f"iteration {resume['start_iter']})",
                level=0,
            )
            self.trainer.outer_cl_training_loop(
                drift_event_id=resume["drift_event_id"], resume=resume
            )
            self._post_cl()

        while not self._should_stop():
            try:
                self._process_stream()
            except StopIteration:
                # Stream exhausted, extend it
                self._extend_stream()

        self.logger.info("==== Continuous Monitoring Complete ====", level=0)
        self.logger.info(f"\tTotal batches processed: {self.batch_count}", level=1)
        self.logger.info(f"\tTotal stream updates: {self.stream_update_count}", level=1)

    def _process_stream(self) -> None:
        """Process batches from current data stream.

        Iterates through the current data loader, evaluating batches and
        checking for drift at regular intervals. When drift is detected,
        pauses monitoring and dispatches the learning module.

        Raises:
            StopIteration: When the data loader is exhausted
        """
        val_loader = self.modelHarness.get_stream_dataloader()

        # On resume, replay the loader past the batches processed before the
        # interruption (deterministic shuffles make positions reproducible);
        # they are already reflected in the restored counters and buffer.
        skip = self._resume_skip_batches
        self._resume_skip_batches = 0

        for batch_idx, batch in tqdm(
            enumerate(val_loader),
            desc="Processing batches",
            leave=False,
        ):
            if batch_idx < skip:
                self.batches_into_window = batch_idx + 1
                continue

            # Evaluate batch and compute all metrics
            metrics = self._evaluate_batch(batch)
            self.metric_buffer.append(metrics)
            self.batch_count += 1
            self.batches_into_window += 1

            # Check drift at specified interval
            if (
                self.detection_interval > 0
                and self.batch_count % self.detection_interval == 0
            ):
                drift_signal = self._check_drift()

                if drift_signal.drift_detected:
                    self._handle_drift(drift_signal)

            self._tick_snapshot(phase="monitor")

        raise StopIteration()

    def _evaluate_batch(self, batch: tuple[torch.Tensor, torch.Tensor]) -> list[float]:
        """Evaluate model on a single batch and compute all metrics.

        Args:
            batch: Tuple of (inputs, targets)

        Returns:
            List of metric values (one per eval_metric)
        """
        self.modelHarness.model.eval()

        if self.batch_count > self.flops_profiler.warmup_iters:
            # Profile inference after warmup
            with self.flops_profiler.measure_flops(tag="infer"):
                with torch.no_grad():
                    x, y = self.modelHarness._unpack(batch)
                    x, y = x.to(self.cfg.device), y.to(self.cfg.device)

                    # Forward pass
                    y_hat = self.modelHarness.model(x)

                    # Compute all metrics
                    metrics = []
                    eval_metrics_log = {}
                    for key, metric_fn in self.modelHarness.eval_metrics.items():
                        value = self.modelHarness._to_scalar(metric_fn(y_hat, y))
                        metrics.append(value)
                        eval_metrics_log[key] = value

                    # Log all eval metrics in one call
                    self.logger.stage("eval")
                    self.logger.log(eval_metrics_log)
        else:
            # Skip profiling during warmup
            with torch.no_grad():
                x, y = self.modelHarness._unpack(batch)
                x, y = x.to(self.cfg.device), y.to(self.cfg.device)

                # Forward pass
                y_hat = self.modelHarness.model(x)

                # Compute all metrics
                metrics = []
                for key, metric_fn in self.modelHarness.eval_metrics.items():
                    value = self.modelHarness._to_scalar(metric_fn(y_hat, y))
                    metrics.append(value)

        return metrics

    def _check_drift(self) -> DriftSignal:
        """Aggregate buffered metrics and check for drift.

        Takes the buffered metrics, aggregates them according to the
        configured aggregation method, extracts the monitored metric,
        and updates the detector.

        Returns:
            DriftSignal from the detector
        """
        if not self.metric_buffer:
            # Edge case: no metrics buffered
            raise RuntimeError(
                "Model Harness requires evaluation metrics"
            )  # Todo: This should be checked in model harness

        # Extract the monitored metric from all buffered metrics
        metric_values = [m[self.metric_idx] for m in self.metric_buffer]

        # Aggregate according to configured method
        if self.aggregation == "mean":
            agg_metric = float(np.mean(metric_values))
        elif self.aggregation == "median":
            agg_metric = float(np.median(metric_values))
        elif self.aggregation == "last":
            agg_metric = float(metric_values[-1])
        else:
            # Default to mean
            agg_metric = float(np.mean(metric_values))

        # Clear buffer
        self.metric_buffer = []

        # Update detector with aggregated metric
        # NOTE: Profiler only covers Pytorch operations
        # Even so, we measure runtime to see if it's a potential bottleneck
        if self.batch_count > self.flops_profiler.warmup_iters:
            with self.flops_profiler.measure_flops(tag="detector"):
                drift_signal = self.detector.update(agg_metric)
        else:
            drift_signal = self.detector.update(agg_metric)

        # Log drift metrics
        self._log_metrics(drift_signal, agg_metric)

        # Journal every check unsampled (the CSV subsamples detected=0 rows)
        if self.journal is not None:
            self.journal.record(
                "drift_check",
                batch_count=self.batch_count,
                metric=agg_metric,
                score=drift_signal.drift_score,
                detected=drift_signal.drift_detected,
                regime=drift_signal.regime.value if drift_signal.regime else None,
            )

        return drift_signal

    def _handle_drift(self, drift_signal: DriftSignal) -> None:
        """Handle detected drift by dispatching learning module.

        PAUSES monitoring, dispatches the continual learning loop,
        then RESUMES monitoring with updated model weights.

        Args:
            drift_signal: The drift signal from the detector
        """
        self.drift_event_count += 1
        self.logger.info(
            f"==== DRIFT DETECTED (Event #{self.drift_event_count})! ====", level=0
        )
        # Log data timestamp range if the harness tracks it
        timerange = getattr(self.modelHarness, "current_window_timerange", None)
        if timerange is not None:
            self.logger.info(
                f"\tData time range: {timerange[0]} → {timerange[1]}", level=1
            )
        self.logger.info(
            f"\tRegime: {drift_signal.regime.value if drift_signal.regime else 'N/A'}",
            level=1,
        )
        self.logger.info(f"\tDrift Score: {drift_signal.drift_score:.4f}", level=1)
        self.logger.info(
            f"\tConfidence: {drift_signal.confidence if drift_signal.confidence else 'N/A'}",
            level=1,
        )

        # Log profiler performance summary
        self.flops_profiler.print_performance(logger=self.logger, level=2)

        if self.journal is not None:
            self.journal.record(
                "drift_detected",
                batch_count=self.batch_count,
                drift_event_id=self.drift_event_count,
                score=drift_signal.drift_score,
                regime=drift_signal.regime.value if drift_signal.regime else None,
                confidence=drift_signal.confidence,
                stream_update_count=self.stream_update_count,
            )

        self.logger.info("-> Dispatching continual learning module...", level=0)

        # PAUSE monitoring, dispatch learning module
        self.trainer.outer_cl_training_loop(
            drift_event_id=self.drift_event_count,
        )
        self._post_cl()

    def _post_cl(self) -> None:
        """After a CL loop (fresh or resumed): checkpoint, retention, reset."""
        if self.modelHarness.ckpts_enabled:
            ckptpath = self.modelHarness.save_ckpt(event=self.drift_event_count)
            self.logger.info(f"* Checkpoint saved to: {ckptpath}", level=0)
            if self.journal is not None:
                self.journal.record(
                    "checkpoint_saved",
                    batch_count=self.batch_count,
                    drift_event_id=self.drift_event_count,
                    path=ckptpath,
                )
            if self.cfg.model.ckpt_retention != "latest":
                from apeiron.experiment.retention import apply_retention

                deleted = apply_retention(
                    self.cfg.model.ckpts_path,
                    max_ckpts=self.cfg.model.max_ckpts,
                    policy=self.cfg.model.ckpt_retention,
                    journal=self.journal,
                    higher_is_better=next(
                        iter(
                            getattr(self.modelHarness, "higher_is_better", {}).values()
                        ),
                        True,
                    ),
                )
                if deleted:
                    self.logger.info(
                        f"* Retention ({self.cfg.model.ckpt_retention}) evicted: "
                        f"{', '.join(deleted)}",
                        level=1,
                    )
                    if self.journal is not None:
                        self.journal.record(
                            "checkpoints_evicted",
                            drift_event_id=self.drift_event_count,
                            policy=self.cfg.model.ckpt_retention,
                            deleted=deleted,
                        )

        self.logger.info("<- Continual learning complete.", level=0)

        # Optionally reset detector after learning
        if self.cfg.drift_detection.reset_after_learning:
            self.logger.debug("Resetting detector state...")
            self.detector.reset()

        self.logger.info("==== RESUMING MONITORING! ====", level=0)

    def _extend_stream(self) -> None:
        """Extend the data stream when exhausted.

        Calls update_data_stream() to load the next buffer of data.
        This does NOT necessarily mean drift occurred - drift is detected
        by the statistical detector based on metric changes.
        """
        self.stream_update_count += 1

        self.logger.info(
            f"\tStream exhausted. Loading next data buffer. {self.stream_update_count}/{self.max_stream_updates}",
            level=1,
        )

        # Load next data buffer
        self._advance_stream(self.stream_update_count)
        self._journal_window()

    def _advance_stream(self, window: int, prefetch_next: bool = True) -> None:
        """Advance the harness one window, with managed residency when declared.

        Harnesses that declare their windows (``describe_window`` returning a
        WindowSpec) get the framework treatment: the window's objects are
        materialized (pinned) BEFORE ``update_data_stream()`` builds loaders,
        and the next window's objects are prefetched afterwards. Undeclared
        harnesses just get ``update_data_stream()`` -- legacy behavior.
        """
        spec = self.modelHarness.describe_window(window)
        resolver = self.modelHarness.data_resolver
        if spec is not None and resolver is not None:
            resolver.materialize(spec)
        self._last_window_spec = spec
        self.modelHarness.update_data_stream()
        self.batches_into_window = 0
        if prefetch_next and spec is not None and resolver is not None:
            resolver.prefetch(self.modelHarness.describe_window(window + 1))

    def _journal_window(self) -> None:
        """Record a window advance, including harness window info if exposed."""
        if self.journal is None:
            return
        spec = self._last_window_spec
        timerange = getattr(self.modelHarness, "current_window_timerange", None)
        self.journal.record(
            "window_started",
            batch_count=self.batch_count,
            stream_update_count=self.stream_update_count,
            label=spec.label if spec is not None else None,
            data_fingerprint=(
                spec.fingerprint
                if spec is not None
                else getattr(self.modelHarness, "current_window_fingerprint", None)
            ),
            data_time_start=timerange[0] if timerange else None,
            data_time_end=timerange[1] if timerange else None,
        )

    # ----- resilience -----

    def _on_cl_step(self, drift_event_id: int, iter_count: int, pre_metrics: dict):
        """Trainer callback after each inner CL iteration (snapshot point)."""
        self._cl_ctx = {
            "drift_event_id": drift_event_id,
            "iter_count": iter_count,
            **pre_metrics,
        }
        self._tick_snapshot(phase="cl")
        self._cl_ctx = None

    def _tick_snapshot(self, phase: str) -> None:
        """Count one update; snapshot on the interval or on interrupt.

        On interrupt: save (regardless of the interval), journal
        run_interrupted, and exit cleanly -- the walltime-signal path.
        """
        if self.snapshots is not None:
            self._update_counter += 1
            self._updates_since_snapshot += 1
            if self.interrupt_requested or (
                self._updates_since_snapshot >= self.snapshot_interval
            ):
                self._save_snapshot(phase)
                self._updates_since_snapshot = 0
        if self.interrupt_requested:
            if self.journal is not None:
                self.journal.record(
                    "run_interrupted", batch_count=self.batch_count, phase=phase
                )
                self.journal.close()
            self.logger.info(
                "==== INTERRUPT: snapshot saved, exiting cleanly ====", level=0
            )
            raise SystemExit(0)

    def _save_snapshot(self, phase: str) -> None:
        import pickle

        from apeiron.experiment.run import Run
        from apeiron.experiment.snapshot import rng_states

        assert self.snapshots is not None
        state = {
            "phase": phase,
            "behavior_config_sha": Run._behavior_config_hash(self.cfg),
            "batch_count": self.batch_count,
            "stream_update_count": self.stream_update_count,
            "drift_event_count": self.drift_event_count,
            "batches_into_window": self.batches_into_window,
            "metric_buffer": [list(m) for m in self.metric_buffer],
            "cl": dict(self._cl_ctx) if phase == "cl" and self._cl_ctx else None,
            "model": self.modelHarness.model.state_dict(),
            "optimizer": self.trainer.optimizer.state_dict(),
            "updater": self.trainer.cl_updater.state_dict(),
            "detector": pickle.dumps(self.detector),
            "rng": rng_states(),
            "harness_rng": self.modelHarness.rng_state_dict(),
            "journal_last_id": (
                self.journal.last_id() if self.journal is not None else 0
            ),
            "task_records": self.modelHarness.task_records_refs(),
        }
        path = self.snapshots.save(state, step=self._update_counter)
        if self.journal is not None:
            self.journal.record(
                "snapshot_saved",
                batch_count=self.batch_count,
                step=self._update_counter,
                phase=phase,
                path=str(path),
            )

    def restore_snapshot(self, state: dict) -> None:
        """Restore full run state from a resilience snapshot (before run())."""
        import pickle

        from apeiron.experiment.run import Run
        from apeiron.experiment.snapshot import restore_rng_states

        # Roll the journal back to the snapshot's position: events past it
        # were journaled between the snapshot and the crash and will be
        # re-created identically by the deterministic replay.
        if self.journal is not None and state.get("journal_last_id"):
            dropped = self.journal.truncate_after(int(state["journal_last_id"]))
            if dropped:
                self.logger.info(
                    f"[resume] rolled back {dropped} journal event(s) past the "
                    "snapshot; replay re-creates them",
                    level=0,
                )

        expected = Run._behavior_config_hash(self.cfg)
        if state.get("behavior_config_sha") != expected:
            self.logger.warning(
                "[resume] behavior config differs from the snapshot's -- "
                "the continuation will not be crash-equivalent"
            )

        self.batch_count = int(state["batch_count"])
        self.stream_update_count = int(state["stream_update_count"])
        self.drift_event_count = int(state["drift_event_count"])
        self.metric_buffer = [list(m) for m in state["metric_buffer"]]
        self._resume_skip_batches = int(state["batches_into_window"])
        self._update_counter = int(state["step"])

        self.modelHarness.model.load_state_dict(state["model"])
        self.trainer.optimizer.load_state_dict(state["optimizer"])
        self.trainer.cl_updater.load_state_dict(state["updater"])
        self.detector = pickle.loads(state["detector"])
        restore_rng_states(state["rng"])
        if state.get("task_records") is not None:
            self.modelHarness.restore_task_records(state["task_records"])
        # Harness sampler epochs must be applied AFTER the window fast-forward
        # rebuilds the loaders (run() does this via _pending_harness_rng).
        self._pending_harness_rng = (
            state.get("harness_rng") or {},
            state["phase"] == "cl",
        )
        self._resumed_from_snapshot = True

        if state["phase"] == "cl" and state.get("cl"):
            cl = state["cl"]
            self._resume_cl = {
                "drift_event_id": int(cl["drift_event_id"]),
                "start_iter": int(cl["iter_count"]) + 1,
                "pre_cur_metrics": cl["pre_cur_metrics"],
                "pre_hist_metrics": cl.get("pre_hist_metrics"),
            }
        self.logger.info(
            f"Restored snapshot: step {self._update_counter}, phase "
            f"{state['phase']}, window {self.stream_update_count}, "
            f"batch {self.batch_count}",
            level=0,
        )

    def _should_stop(self) -> bool:
        """Check if monitoring should stop.

        Returns:
            True if max_stream_updates has been reached
        """
        return self.stream_update_count >= self.max_stream_updates

    def _log_metrics(self, drift_signal: DriftSignal, metric_value: float) -> None:
        """Log drift detection metrics.

        Args:
            drift_signal: The drift signal from the detector
            metric_value: The aggregated metric value
        """
        flops_perf = self.flops_profiler.get_performance()

        # Log all drift metrics in a single call.
        # detected=0 is sampled at 10% to reduce log volume; detected=1 is
        # always included so no true drift events are dropped.
        log_detected = drift_signal.drift_detected or (np.random.random() <= 0.1)
        self.logger.stage("drift")
        # Include data timestamp range if available from the harness
        timerange = getattr(self.modelHarness, "current_window_timerange", None)
        ts_fields = {}
        if timerange is not None:
            ts_fields["data_time_start"] = timerange[0]
            ts_fields["data_time_end"] = timerange[1]
        self.logger.log(
            {
                **(
                    {"detected": int(drift_signal.drift_detected)}
                    if log_detected
                    else {}
                ),
                "score": drift_signal.drift_score,
                "regime": (drift_signal.regime.value if drift_signal.regime else "N/A"),
                "confidence": (
                    drift_signal.confidence if drift_signal.confidence else "N/A"
                ),
                f"metric_{self.metric_idx}": metric_value,
                **ts_fields,
                **{f"cperf_{k}": v for k, v in flops_perf.items()},
            },
        )
