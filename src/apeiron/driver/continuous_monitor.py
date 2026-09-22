"""Continuous drift monitoring system.

This module implements a continuous monitoring architecture that:
- Maintains detector state across data stream updates
- Processes batches continuously until drift is detected
- Pauses for learning when drift is detected
- Resumes monitoring with updated model weights
- Automatically extends the data stream when exhausted
"""

from __future__ import annotations
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import numpy as np

from apeiron.config.configuration import Config
from apeiron.drift_detection.load_drift_detector import load_drift_detector
from apeiron.experiment.determinism import rng_state, set_rng_state
from apeiron.experiment.model_info import shapes_hash, tensor_table, unwrap
from apeiron.experiment.restart import SCHEMA_VERSION, RunInterrupted
from apeiron.experiment.run import behavior_hash
from apeiron.drift_detection.detectors.base import DriftSignal
from apeiron.profilers import FLOPSProfiler
from apeiron.logger import get_logger
from apeiron.training import ContinuousTrainer
from tqdm import tqdm

if TYPE_CHECKING:
    from apeiron.experiment import Run
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
        run: Run | None = None,
    ):
        """Initialize continuous monitor.

        Args:
            cfg: Configuration object
            modelHarness: Model harness containing model and data loaders
            run: Optional run directory; when given, the loop records what it
                does to the run's event log. None reproduces legacy behavior.
        """
        self.cfg = cfg
        self.modelHarness = modelHarness
        self.logger = get_logger()
        # `run()` is the monitoring loop, so the collaborator is kept private.
        self._run = run

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
            run=run,
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

        # Restart bookkeeping. All inert without a run directory.
        self.restart_interval = cfg.experiment.restart_interval if cfg.experiment else 0
        self._interrupt = False
        self._resuming = False
        self._batches_in_pass = 0  # batches taken from the current window's loader
        self._window_rng: dict | None = None  # rng as it was when that loader was made
        self._resume_skip = 0
        self._resume_rng: dict | None = None
        self._resume_window_rng: dict | None = None
        self._window_inputs: list[str] = []
        self._pass_samplers: dict[str, int] = {}
        self._resume_samplers: dict[str, int] | None = None
        # Set while a round of learning is running, so a save taken inside one
        # knows where to come back to.
        self._round: tuple[int, int] | None = None
        self._resume_round: dict | None = None
        self.trainer.on_inner_step = self._on_inner_step

        self.logger.info("==== ContinuousMonitor initialized ====", level=0)
        self.logger.info(f"\tDetector: {cfg.drift_detection.detector_name}", level=1)
        self.logger.info(f"\tMonitoring metric index: {self.metric_idx}", level=1)
        self.logger.info(
            f"\tDetection interval: {self.detection_interval} batches", level=1
        )
        self.logger.info(f"\tAggregation method: {self.aggregation}", level=1)
        self.logger.info(f"\tMax stream updates: {self.max_stream_updates}", level=1)

    def request_interrupt(self) -> None:
        """Ask the loop to save and stop at the next batch boundary.

        Called from a signal handler, so it does nothing but set a flag.
        """
        self._interrupt = True

    def capture_state(self) -> dict:
        """Everything needed to carry on from this batch boundary."""
        model = unwrap(self.modelHarness.model)
        return {
            "schema": SCHEMA_VERSION,
            "config_sha256": behavior_hash(self.cfg),
            "shapes_sha256": shapes_hash(tensor_table(model)),
            "journal_last_id": (
                self._run.journal.last_id() if self._run is not None else 0
            ),
            "batch_count": self.batch_count,
            "stream_update_count": self.stream_update_count,
            "drift_event_count": self.drift_event_count,
            "batches_in_pass": self._batches_in_pass,
            "metric_buffer": [list(m) for m in self.metric_buffer],
            "pass_samplers": dict(self._pass_samplers),
            "round": (
                None if self._round is None else self.trainer.round_state(*self._round)
            ),
            "model": model.state_dict(),
            "trainer": self.trainer.state_dict(),
            "detector": self.detector,
            "rng": rng_state(),
            "window_rng": self._window_rng,
        }

    def restore_state(self, state: dict) -> None:
        """Put the loop back where a saved state left it.

        Refuses states from a different config or a differently shaped model,
        both of which would otherwise fail later and less clearly.
        """
        if state.get("schema") != SCHEMA_VERSION:
            raise ValueError(
                f"restart state schema {state.get('schema')} != {SCHEMA_VERSION}"
            )

        model = unwrap(self.modelHarness.model)
        expected = shapes_hash(tensor_table(model))
        if state.get("shapes_sha256") != expected:
            raise ValueError(
                "restart state was written for a differently shaped model; "
                "the harness that built it is not the one running now"
            )
        if state.get("config_sha256") != behavior_hash(self.cfg):
            raise ValueError("restart state was written under a different config")

        model.load_state_dict(state["model"])
        self.trainer.load_state_dict(state["trainer"])
        self.detector = state["detector"]

        self.batch_count = state["batch_count"]
        self.stream_update_count = state["stream_update_count"]
        self.drift_event_count = state["drift_event_count"]
        self.metric_buffer = [list(m) for m in state["metric_buffer"]]

        # The log is committed per event and so runs ahead of a restart file
        # written every N batches. Drop the tail; replaying re-creates it.
        if self._run is not None:
            dropped = self._run.journal.truncate_after(int(state["journal_last_id"]))
            if dropped:
                self.logger.info(
                    f"\tRolled back {dropped} journal event(s) past the restart point",
                    level=1,
                )

        self._resuming = True
        self._resume_skip = int(state["batches_in_pass"])
        self._resume_rng = state["rng"]
        self._resume_window_rng = state["window_rng"]
        self._resume_samplers = dict(state.get("pass_samplers") or {})

        self._resume_round = state.get("round")
        if self._resume_round is not None:
            # The generators have to come back inside the round, after its
            # loaders are wound forward -- not here.
            self._resume_round = {**self._resume_round, "rng": state["rng"]}
            self._resume_rng = None

    def _on_inner_step(self, drift_event_id: int, iteration: int) -> None:
        """Called by the trainer after each step of learning."""
        self._round = (drift_event_id, iteration)
        try:
            self._tick_restart()
        finally:
            self._round = None

    def _save_restart(self, reason: str) -> None:
        if self._run is not None:
            self._run.save_restart(self.capture_state(), reason=reason)

    def _tick_restart(self) -> None:
        """Called at every batch boundary."""
        if self._run is None:
            return
        if self._interrupt:
            self._save_restart("interrupt")
            self._run.record("run_interrupted", batch=self.batch_count)
            self.logger.info(
                f"==== Interrupted at batch {self.batch_count}; state saved ====",
                level=0,
            )
            raise RunInterrupted()
        if self.restart_interval <= 0:
            return
        # Inside a round the batch counter stands still, so the iteration is
        # what advances.
        counter = self._round[1] + 1 if self._round is not None else self.batch_count
        if counter % self.restart_interval == 0:
            self._save_restart("interval")

    def _ensure_window_inputs(self) -> None:
        """Make the coming window's data present before the harness opens it."""
        self._window_inputs = sorted(
            self.modelHarness.window_inputs(self.stream_update_count)
        )
        stats = self.modelHarness.ensure_window_inputs(self.stream_update_count)
        if stats is None:
            return
        self.logger.info(
            f"\tWindow {self.stream_update_count} data: {stats.present} present, "
            f"{stats.fetched} fetched ({stats.bytes_fetched / 1e6:.1f} MB) "
            f"in {stats.seconds:.1f}s",
            level=1,
        )
        self._record("dataset", window=self.stream_update_count, **stats.as_payload())

    def _record(self, kind: str, **payload: object) -> None:
        """Append an event to the run's log. No-op without a run directory."""
        if self._run is not None:
            self._run.record(kind, **payload)

    def run(self) -> None:
        """Main continuous monitoring loop.

        This loop continues until max_stream_updates is reached. It processes
        batches from the data stream, checks for drift at regular intervals,
        and dispatches learning modules when drift is detected.
        """
        self.logger.info("==== Starting Continuous Monitoring ====", level=0)

        # Initialize first data stream
        self.logger.info("\tInitializing first data stream...", level=1)
        self._ensure_window_inputs()
        self.modelHarness.update_data_stream()
        if self._resuming:
            # Windows are produced in order, so getting back to window N means
            # asking for N more. Their events are already in the log.
            for _ in range(self.stream_update_count):
                self._ensure_window_inputs()
                self.modelHarness.update_data_stream()
            self.logger.info(
                f"\tResumed at window {self.stream_update_count}, "
                f"batch {self.batch_count}",
                level=1,
            )
            self._finish_interrupted_round()
        else:
            self._record(
                "window",
                index=self.stream_update_count,
                inputs=self._window_inputs,
            )

        while not self._should_stop():
            try:
                self._process_stream()
            except StopIteration:
                # Stream exhausted, extend it
                self._extend_stream()

        self.logger.info("==== Continuous Monitoring Complete ====", level=0)
        self.logger.info(f"\tTotal batches processed: {self.batch_count}", level=1)
        self.logger.info(f"\tTotal stream updates: {self.stream_update_count}", level=1)

    def _finish_interrupted_round(self) -> None:
        """Re-enter a round of learning that a save caught in the middle.

        The stream carries on from the batch where drift fired, which is what
        an uninterrupted run would have done.
        """
        if self._resume_round is None:
            return
        resume, self._resume_round = self._resume_round, None
        self.logger.info(
            f"\tFinishing interrupted learning: event "
            f"{resume['drift_event_id']}, from iteration "
            f"{resume['next_iteration']}",
            level=0,
        )
        self.trainer.outer_cl_training_loop(
            drift_event_id=int(resume["drift_event_id"]), resume=resume
        )
        self._save_checkpoint()
        if self.cfg.drift_detection.reset_after_learning:
            self.detector.reset()

    def _process_stream(self) -> None:
        """Process batches from current data stream.

        Iterates through the current data loader, evaluating batches and
        checking for drift at regular intervals. When drift is detected,
        pauses monitoring and dispatches the learning module.

        Raises:
            StopIteration: When the data loader is exhausted
        """
        # A shuffling loader seeds itself from the global generator when its
        # iterator is made. Recording that generator state, and putting it back
        # before rebuilding the loader on resume, is what makes the resumed
        # pass see the same batches in the same order.
        if self._resume_window_rng is not None:
            set_rng_state(self._resume_window_rng)
            self._window_rng = self._resume_window_rng
            self._resume_window_rng = None
        else:
            self._window_rng = rng_state()

        if self._resume_samplers is not None:
            self.modelHarness.load_sampler_state(self._resume_samplers)
            self._resume_samplers = None
        self._pass_samplers = self.modelHarness.sampler_state()

        val_loader = self.modelHarness.get_stream_dataloader()
        self._batches_in_pass = 0

        batches = enumerate(val_loader)
        if self._resume_skip:
            skip, self._resume_skip = self._resume_skip, 0
            for _ in range(skip):
                if next(batches, None) is None:
                    break
            self._batches_in_pass = skip
            if self._resume_rng is not None:
                # Forward progress continues from the generator state the run
                # had when it was saved, not from where replaying left it.
                set_rng_state(self._resume_rng)
                self._resume_rng = None

        for batch_idx, batch in tqdm(
            batches,
            desc="Processing batches",
            leave=False,
        ):
            # Evaluate batch and compute all metrics
            metrics = self._evaluate_batch(batch)
            self.metric_buffer.append(metrics)
            self.batch_count += 1
            self._batches_in_pass += 1

            # Check drift at specified interval
            if (
                self.detection_interval > 0
                and self.batch_count % self.detection_interval == 0
            ):
                drift_signal = self._check_drift()

                if drift_signal.drift_detected:
                    self._handle_drift(drift_signal)

            self._tick_restart()

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

        # Every check, including the ones that found nothing: the log is a
        # record of decisions, not a sample of them.
        self._record(
            "drift_check",
            window=self.stream_update_count,
            batch=self.batch_count,
            value=agg_metric,
            detected=bool(drift_signal.drift_detected),
            score=drift_signal.drift_score,
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
        self._record(
            "drift",
            event=self.drift_event_count,
            window=self.stream_update_count,
            batch=self.batch_count,
            score=drift_signal.drift_score,
            regime=(drift_signal.regime.value if drift_signal.regime else None),
        )
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

        self.logger.info("-> Dispatching continual learning module...", level=0)

        # PAUSE monitoring, dispatch learning module
        self.trainer.outer_cl_training_loop(
            drift_event_id=self.drift_event_count,
        )

        self._save_checkpoint()

        self.logger.info("<- Continual learning complete.", level=0)

        # Optionally reset detector after learning
        if self.cfg.drift_detection.reset_after_learning:
            self.logger.debug("Resetting detector state...")
            self.detector.reset()

        self.logger.info("==== RESUMING MONITORING! ====", level=0)

    def _save_checkpoint(self) -> None:
        """Write the post-learning checkpoint, if checkpointing is on."""
        if not self.modelHarness.ckpts_enabled:
            return
        ckptpath = self.modelHarness.save_ckpt(event=self.drift_event_count)
        self.logger.info(f"* Checkpoint saved to: {ckptpath}", level=0)
        self._record(
            "checkpoint",
            event=self.drift_event_count,
            name=Path(ckptpath).name,
            path=ckptpath,
        )

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
        self._ensure_window_inputs()
        self.modelHarness.update_data_stream()
        self._record(
            "window", index=self.stream_update_count, inputs=self._window_inputs
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
