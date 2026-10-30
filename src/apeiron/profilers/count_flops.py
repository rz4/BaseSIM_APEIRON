"""FLOPs / throughput profiling for training and inference blocks.

The profiler wraps two different measurement strategies behind one store:

* :meth:`FLOPSProfiler.measure_flops` uses ``torch.utils.flop_counter``'s
  ``FlopCounterMode``, which sees matmul/convolution style operations and
  counts them with the usual ``2 * M * N * K`` convention.
* :meth:`FLOPSProfiler.measure_flops_optimizer` covers the optimizer step,
  which ``FlopCounterMode`` cannot see at all.  It runs a ``torch.profiler``
  session, maps every observed ATen operation through
  :data:`apeiron.profilers.aten_flops_map.ATEN_FLOPS_PER_ELEMENT` to get a
  FLOPs-per-parameter-element figure, and scales it by the number of
  trainable parameters.

Both strategies append to ``profiles[tag]`` as ``{"flop": [...],
"time": [...]}``, and :meth:`FLOPSProfiler.get_performance` reduces that
store to a flat ``{tag}_flop`` / ``{tag}_time`` / ``{tag}_flops`` dict whose
keys become metric-CSV columns.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional

import torch

from apeiron.profilers.aten_flops_map import ATEN_FLOPS_PER_ELEMENT

__all__ = ["FLOPSProfiler"]


class FLOPSProfiler:
    """Collects FLOP counts and wall-clock durations for tagged blocks.

    Args:
        warmup_iters: Number of warm-up iterations a caller is expected to
            run before the figures are considered representative.  Kept as
            state so callers and reports can agree on it; measurements are
            always recorded.
    """

    def __init__(self, warmup_iters: int = 1) -> None:
        self.warmup_iters = warmup_iters
        self.profiles: Dict[str, Dict[str, List[Any]]] = {}
        self.tag: Optional[str] = None
        self.start_time: Optional[float] = None

    # ------------------------------------------------------------------
    # store
    # ------------------------------------------------------------------
    def _record(self, tag: str, flop: Any, elapsed: float) -> None:
        record = self.profiles.setdefault(tag, {"flop": [], "time": []})
        record["flop"].append(flop)
        record["time"].append(elapsed)

    # ------------------------------------------------------------------
    # measurement
    # ------------------------------------------------------------------
    @contextmanager
    def measure_flops(self, tag: str = "default") -> Iterator[None]:
        """Measure the FLOPs and duration of the wrapped block.

        Args:
            tag: Name the measurement is recorded under.  Repeated use of
                the same tag accumulates measurements, which
                :meth:`get_performance` averages.
        """
        from torch.utils.flop_counter import FlopCounterMode

        self.tag = tag
        self.start_time = time.perf_counter()
        counter = FlopCounterMode(display=False)
        try:
            with counter:
                yield
            # Recorded only on a clean exit: a block that RAISED used to be
            # appended as a zero-FLOP measurement, indistinguishable from a
            # block that genuinely did no work, pulling every average over
            # the tag toward zero with no trace of the failure
            # (claims/LOG.md A50). The exception itself is the record now.
            elapsed = time.perf_counter() - self.start_time
            self._record(tag, int(counter.get_total_flops()), elapsed)
        finally:
            self.tag = None
            self.start_time = None

    @contextmanager
    def measure_flops_optimizer(
        self,
        model: Any,
        device: Any,
        tag: str = "optimizer",
    ) -> Iterator[None]:
        """Measure an optimizer step, which ``FlopCounterMode`` cannot see.

        Total FLOPs are estimated as (FLOPs per parameter element) x
        (number of trainable parameters), where the per-element figure comes
        from mapping the observed ATen operations through
        ``ATEN_FLOPS_PER_ELEMENT``.

        Args:
            model: Module whose trainable parameters the estimate scales by.
            device: Device the step runs on; selects the profiler activities.
            tag: Name the measurement is recorded under.
        """
        activities = [torch.profiler.ProfilerActivity.CPU]
        if "cuda" in str(device).lower() and torch.cuda.is_available():
            activities.append(torch.profiler.ProfilerActivity.CUDA)

        self.tag = tag
        self.start_time = time.perf_counter()
        flop = 0
        try:
            try:
                with torch.profiler.profile(
                    activities=activities, record_shapes=True, with_flops=True
                ) as prof:
                    yield
                per_element = self._estimate_flops_per_elem(prof)
                n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
                flop = int(per_element) * int(n_params)
            finally:
                elapsed = time.perf_counter() - (
                    self.start_time
                    if self.start_time is not None
                    else time.perf_counter()
                )
                self._record(tag, flop, elapsed)
        finally:
            self.tag = None
            self.start_time = None

    # ------------------------------------------------------------------
    # ATen based estimate
    # ------------------------------------------------------------------
    def _estimate_flops_per_elem(self, prof: Any, debug: bool = False) -> int:
        """FLOPs per parameter element implied by a profiler session.

        Every ATen operation the session observed is looked up in
        ``ATEN_FLOPS_PER_ELEMENT``; known entries contribute
        ``flops_per_element * call_count``.  Operations that are absent, or
        whose cost needs dimension-specific formulas (``None`` in the table),
        contribute nothing.

        Args:
            prof: Anything exposing ``key_averages()``.
            debug: Print the per-operation breakdown.

        Returns:
            Estimated FLOPs per parameter element.
        """
        rows = []
        total = 0
        for event in prof.key_averages():
            key = getattr(event, "key", None)
            count = int(getattr(event, "count", 0) or 0)
            per_element = (
                ATEN_FLOPS_PER_ELEMENT.get(key) if isinstance(key, str) else None
            )
            contribution = 0
            if per_element is not None:
                contribution = int(per_element) * count
                total += contribution
            rows.append((key, count, per_element, contribution))

        if debug:
            print("=" * 80)
            print("All Function Calls")
            print("=" * 80)
            print(f"{'operation':<40} {'calls':>7} {'flop/elem':>10} {'subtotal':>12}")
            print("-" * 80)
            for key, count, per_element, contribution in rows:
                shown = "n/a" if per_element is None else str(per_element)
                print(f"{str(key):<40} {count:>7} {shown:>10} {contribution:>12}")
            print("-" * 80)
            print(f"Estimated FLOPs per param element: {total}")
            print("=" * 80)

        return total

    # ------------------------------------------------------------------
    # reporting
    # ------------------------------------------------------------------
    def get_performance(self) -> Dict[str, float]:
        """Average FLOPs, duration and throughput per tag.

        Returns:
            Flat dict with ``{tag}_flop`` (count), ``{tag}_time`` (seconds)
            and ``{tag}_flops`` (throughput) for every recorded tag.  All
            values are floats, since each one becomes a metric-CSV cell.
        """
        performance: Dict[str, float] = {}
        for tag, record in self.profiles.items():
            flops = record["flop"]
            times = record["time"]
            if not flops or not times:
                continue
            avg_flop = float(sum(flops)) / len(flops)
            avg_time = float(sum(times)) / len(times)
            performance[f"{tag}_flop"] = avg_flop
            performance[f"{tag}_time"] = avg_time
            performance[f"{tag}_flops"] = avg_flop / avg_time if avg_time > 0 else 0.0
        return performance

    def print_performance(self, logger: Any = None, level: int = 0) -> None:
        """Print a human-readable summary of the collected measurements.

        Args:
            logger: Optional logger instance. If provided, uses logger.info().
                If None, falls back to print() for backward compatibility.
            level: Verbosity level for logging (default: 0). Only used when
                logger is provided.
        """
        output_fn = (lambda msg: logger.info(msg, level=level)) if logger else print

        performance = self.get_performance()
        if not performance:
            output_fn("No performance data collected.")
            return

        total_flop = 0.0
        total_time = 0.0

        output_fn("=" * 72)
        output_fn("FLOPS PROFILER PERFORMANCE")
        output_fn("=" * 72)
        output_fn(f"{'tag':<24} {'FLOPs':>14} {'time':>12} {'throughput':>18}")
        output_fn("-" * 72)
        for tag in self.profiles:
            if f"{tag}_flop" not in performance:
                continue
            flop = performance[f"{tag}_flop"]
            elapsed = performance[f"{tag}_time"]
            throughput = performance[f"{tag}_flops"]
            total_flop += flop
            total_time += elapsed
            output_fn(
                f"{tag:<24} {self._format_flops(flop):>14} "
                f"{self._format_time(elapsed):>12} "
                f"{self._format_throughput(throughput):>18}"
            )
        output_fn("-" * 72)
        total_throughput = total_flop / total_time if total_time > 0 else 0.0
        output_fn(
            f"{'TOTAL':<24} {self._format_flops(total_flop):>14} "
            f"{self._format_time(total_time):>12} "
            f"{self._format_throughput(total_throughput):>18}"
        )
        output_fn("=" * 72)

    # ------------------------------------------------------------------
    # formatting helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _format_flops(flops: float) -> str:
        """Format a FLOP count with an SI-style unit."""
        flops = float(flops)
        if abs(flops) >= 1e12:
            return f"{flops / 1e12:.2f} TFLOPs"
        if abs(flops) >= 1e9:
            return f"{flops / 1e9:.2f} GFLOPs"
        if abs(flops) >= 1e6:
            return f"{flops / 1e6:.2f} MFLOPs"
        if abs(flops) >= 1e3:
            return f"{flops / 1e3:.2f} KFLOPs"
        return f"{flops:.2f} FLOPs"

    @staticmethod
    def _format_time(seconds: float) -> str:
        """Format a duration with an SI-style unit."""
        seconds = float(seconds)
        if abs(seconds) >= 1.0:
            return f"{seconds:.2f} s"
        if abs(seconds) >= 1e-3:
            return f"{seconds * 1e3:.2f} ms"
        if abs(seconds) >= 1e-6:
            return f"{seconds * 1e6:.2f} μs"
        return f"{seconds * 1e9:.2f} ns"

    @staticmethod
    def _format_throughput(flops_per_second: float) -> str:
        """Format a throughput with an SI-style unit."""
        rate = float(flops_per_second)
        if abs(rate) >= 1e12:
            return f"{rate / 1e12:.2f} TFLOP/s"
        if abs(rate) >= 1e9:
            return f"{rate / 1e9:.2f} GFLOP/s"
        if abs(rate) >= 1e6:
            return f"{rate / 1e6:.2f} MFLOP/s"
        if abs(rate) >= 1e3:
            return f"{rate / 1e3:.2f} KFLOP/s"
        return f"{rate:.2f} FLOP/s"
