"""The monitoring loop: evaluate the stream, detect drift, adapt to it.

:class:`ContinuousMonitor` is the framework's default driver. It walks the
harness's stream window by window, evaluates every batch it sees, aggregates
those measurements into the single number the detector consumes, and hands a
detection to the continual-learning trainer.

Two side runners subclass it -- a drift-only monitor that never adapts and a
scheduled runner that adapts on a timetable rather than on a detection -- so
the seams they use (:meth:`_aggregate_buffer`, :meth:`_log_metrics`,
:meth:`_adapt`, and a :meth:`_process_stream` that may end by raising
``StopIteration``) are part of this class's contract rather than private
details.
"""

from __future__ import annotations

from functools import cached_property
from typing import Any, List, Optional

import numpy as np
import torch

from apeiron.config.configuration import Config
from apeiron.drift_detection.detectors.base import BaseDriftDetector, DriftSignal
from apeiron.drift_detection.load_drift_detector import load_drift_detector
from apeiron.logger import configure_backend, get_logger
from apeiron.model.torch_model_harness import BaseModelHarness
from apeiron.profilers import FLOPSProfiler
from apeiron.training.continuous_trainer import ContinuousTrainer


class ContinuousMonitor:
    """Monitor a data stream, detect drift, and adapt the model to it."""

    def __init__(self, cfg: Config, modelHarness: BaseModelHarness) -> None:
        """Initialize the monitor from a config and a model harness.

        Neither the detector nor the trainer is built here: a drift-only run
        never adapts and a scheduled run never consults a detector, so both are
        :class:`functools.cached_property` and are constructed on first use.
        """
        self.cfg = cfg
        self.modelHarness = modelHarness

        self.logger = get_logger(
            verbosity=cfg.verbosity,
            backend=configure_backend(cfg),
            csv_path=cfg.logging.metrics_output_path if cfg.logging else None,
        )
        self.flops_profiler = FLOPSProfiler()

        # Announced when the run starts; both side runners overwrite it.
        self.label: str = "Continuous Monitoring"

        # Drift-detection settings, read once so a subclass can see them.
        self.detection_interval: int = cfg.drift_detection.detection_interval
        self.max_stream_updates: int = cfg.drift_detection.max_stream_updates
        self.aggregation: str = cfg.drift_detection.aggregation
        self.metric_idx: int = cfg.drift_detection.metric_index

        # Bookkeeping. These are state on the object, not measurements, so
        # none of them is written into the metrics record.
        self.stream_update_count: int = 0
        self.batch_count: int = 0
        self.check_count: int = 0
        self.drift_event_count: int = 0

        # One row of eval metrics per batch since the last drift check.
        self.metric_buffer: List[List[float]] = []
        # The value the detector was last updated with; the side runners log it.
        self.last_agg_metric: Optional[float] = None

    # ----- lazily built collaborators -----

    @cached_property
    def detector(self) -> BaseDriftDetector:
        """The drift detector, built on first consultation."""
        return load_drift_detector(self.cfg)

    @cached_property
    def trainer(self) -> ContinuousTrainer:
        """The continual-learning trainer, built on first adaptation."""
        return ContinuousTrainer(
            cfg=self.cfg,
            modelHarness=self.modelHarness,
            logger=self.logger,
            profiler=self.flops_profiler,
        )

    # ----- the loop -----

    def run(self) -> None:
        """Walk the stream until the update budget is spent.

        One ``_process_stream()`` call is one window. A window ends by raising
        ``StopIteration`` -- the base class's own does, and so does the
        scheduled runner's override -- which is caught here and answered by
        extending the stream.
        """
        self.logger.info(f"==== {self.label} ====")

        # Prime the first window. This one does not count against the budget.
        self.modelHarness.update_data_stream()

        while not self._should_stop():
            try:
                self._process_stream()
            except StopIteration:
                self._extend_stream()

        self.logger.info(
            f"==== {self.label} finished: {self.batch_count} batches, "
            f"{self.check_count} drift checks, "
            f"{self.drift_event_count} drift event(s) ===="
        )

    def _should_stop(self) -> bool:
        """True once the stream-update budget is spent."""
        return self.stream_update_count >= self.max_stream_updates

    def _extend_stream(self) -> None:
        """Ask the harness for more data and count the extension."""
        self.modelHarness.update_data_stream()
        self.stream_update_count += 1
        self.logger.info(
            f"\tStream update {self.stream_update_count}/{self.max_stream_updates}",
            level=1,
        )

    def _process_stream(self) -> None:
        """Evaluate one window of the stream, checking drift on the interval.

        :raises StopIteration: Always, once the window is exhausted, to hand
            control back to :meth:`run`.
        """
        loader = self.modelHarness.get_stream_dataloader()

        for batch in loader:
            metrics = self._evaluate_batch(batch)
            self.batch_count += 1
            self.metric_buffer.append(metrics)

            if (
                self.detection_interval > 0
                and self.batch_count % self.detection_interval == 0
            ):
                drift_signal = self._check_drift()
                self.check_count += 1
                if drift_signal.drift_detected:
                    self._handle_drift(drift_signal)

        raise StopIteration

    # ----- evaluation -----

    def _evaluate_batch(self, batch: Any) -> List[float]:
        """Evaluate one streamed batch and, past warmup, write it down.

        The first ``warmup_iters + 1`` batches ARE evaluated -- the detector
        needs their metrics -- but are not logged and not profiled, so the cold
        iterations do not contaminate the FLOP measurements.
        """
        if self.batch_count > self.flops_profiler.warmup_iters:
            with self.flops_profiler.measure_flops(tag="infer"):
                metrics = self._forward_metrics(batch)
                self.logger.stage("eval")
                self.logger.log(
                    dict(zip(self.modelHarness.eval_metrics, metrics)),
                )
        else:
            metrics = self._forward_metrics(batch)

        return metrics

    @torch.no_grad()
    def _forward_metrics(self, batch: Any) -> List[float]:
        """Score one batch with every metric the harness declares, in order."""
        model = self.modelHarness.model
        model.eval()

        x, y = self.modelHarness._unpack(batch)
        x = x.to(self.cfg.device)
        y = y.to(self.cfg.device)
        y_hat = model(x)

        return [
            self.modelHarness._to_scalar(metric(y_hat, y))
            for metric in self.modelHarness.eval_metrics.values()
        ]

    # ----- drift -----

    def _aggregate_buffer(self) -> float:
        """Reduce the buffered metric column to the number the detector sees.

        Recognises ``"mean"``, ``"median"`` and ``"last"``; anything else is
        the mean. The buffer is emptied, so the next window starts clean.

        :raises RuntimeError: If no batch has been evaluated since the last
            check.
        """
        if not self.metric_buffer:
            raise RuntimeError(
                "Drift detection requires evaluation metrics, but none have "
                "been buffered since the last check."
            )

        values = [float(row[self.metric_idx]) for row in self.metric_buffer]
        self.metric_buffer = []

        if self.aggregation == "last":
            agg_metric = values[-1]
        elif self.aggregation == "median":
            agg_metric = float(np.median(values))
        else:
            agg_metric = float(np.mean(values))

        self.last_agg_metric = agg_metric
        return agg_metric

    def _check_drift(self) -> DriftSignal:
        """Aggregate the window's metrics, update the detector, record the row."""
        agg_metric = self._aggregate_buffer()

        if self.batch_count > self.flops_profiler.warmup_iters:
            with self.flops_profiler.measure_flops(tag="detector"):
                drift_signal = self.detector.update(agg_metric)
        else:
            drift_signal = self.detector.update(agg_metric)

        self._log_metrics(drift_signal, agg_metric)
        self.logger.info(f"\tDrift check: {drift_signal}", level=2)

        return drift_signal

    def _log_metrics(self, drift_signal: DriftSignal, metric_value: float) -> None:
        """Write the drift row: the detector's verdict and what it saw.

        Profiler measurements are namespaced ``cperf_`` so wall-clock fields
        stay distinguishable from model metrics downstream.
        """
        # A real detection is always included so no true drift events are
        # dropped; non-events are sampled at roughly one in ten.
        log_detected = drift_signal.drift_detected or (np.random.random() <= 0.1)

        metrics: dict[str, Any] = {
            f"metric_{self.metric_idx}": metric_value,
            "score": drift_signal.drift_score,
            "regime": drift_signal.regime.value,
            "confidence": drift_signal.confidence,
            **({"detected": int(drift_signal.drift_detected)} if log_detected else {}),
            **{
                f"cperf_{key}": value
                for key, value in self.flops_profiler.get_performance().items()
            },
        }

        timerange = getattr(self.modelHarness, "current_window_timerange", None)
        if timerange:
            metrics["data_time_start"] = timerange[0]
            metrics["data_time_end"] = timerange[1]

        self.logger.stage("drift")
        self.logger.log(metrics)

    def _handle_drift(self, drift_signal: DriftSignal) -> None:
        """Count a detection, adapt to it, and optionally forget it happened."""
        self.drift_event_count += 1

        timerange = getattr(self.modelHarness, "current_window_timerange", None)
        window = f" over {timerange[0]} .. {timerange[1]}" if timerange else ""
        self.logger.info(
            f"Drift event {self.drift_event_count}{window}: "
            f"regime={drift_signal.regime.value}, "
            f"score={drift_signal.drift_score:.4f}"
        )

        self._adapt(self.drift_event_count)

        if self.cfg.drift_detection.reset_after_learning:
            self.detector.reset()
            self.logger.info("\tDetector reset after learning", level=1)

    def _adapt(self, event_id: int) -> None:
        """Pause monitoring, run the CL loop, checkpoint, and resume.

        The seam a subclass adapting on a schedule calls instead of waiting
        for a detection, so the checkpoint is written here rather than by the
        caller: the file is named for ``event_id``, which is the schedule's
        trigger count on that path and the drift-event count on this one.
        """
        self.trainer.outer_cl_training_loop(drift_event_id=event_id)

        if self.modelHarness.ckpts_enabled:
            ckptpath = self.modelHarness.save_ckpt(event=event_id)
            self.logger.info(f"\tSaved checkpoint: {ckptpath}", level=1)
