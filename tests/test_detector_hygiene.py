"""Detector behaviours the ledger flagged, now asserted (A21, A39, A50, A79).

Each of these was found by comparing a blind reconstruction against the
original: the reconstruction's answer was the better one, and this branch
ships it. The tests keep the better answers from regressing.
"""

from __future__ import annotations

import pytest

from apeiron.drift_detection.detectors.base import DriftSignal, LearningRegime
from apeiron.drift_detection.detectors.model_performance_detector import (
    EnsembleDetector,
    ModelEvalDetector,
)
from apeiron.profilers.count_flops import FLOPSProfiler


class _Fires:
    """A stub sub-detector with a fixed verdict."""

    def __init__(self, regime, detected=True):
        self._signal = DriftSignal(
            regime=regime, drift_detected=detected, drift_score=1.0,
            confidence=0.5, metadata={},
        )
        self.name = f"stub-{regime.value}"

    def update(self, value, **kwargs):
        return self._signal

    def reset(self):
        pass


class TestEnsembleTieIsDeterministic:
    """A39: the original reported a nondeterministic regime on a tied vote
    (set iteration order). The regime of a tie is now the most severe of
    the firing detectors, every time."""

    def test_a_tie_resolves_to_the_more_severe_regime(self):
        pair = [_Fires(LearningRegime.CONTINUAL_LEARNING),
                _Fires(LearningRegime.RETRAIN)]
        for _ in range(20):
            signal = EnsembleDetector(detectors=list(pair), voting="any").update(1.0)
            assert signal.regime is LearningRegime.RETRAIN

    def test_order_does_not_matter(self):
        a = EnsembleDetector(
            detectors=[_Fires(LearningRegime.FINE_TUNING),
                       _Fires(LearningRegime.CONTINUAL_LEARNING)],
            voting="any").update(1.0)
        b = EnsembleDetector(
            detectors=[_Fires(LearningRegime.CONTINUAL_LEARNING),
                       _Fires(LearningRegime.FINE_TUNING)],
            voting="any").update(1.0)
        assert a.regime is b.regime


class TestEvalDetectorRefusesInsteadOfAsserting:
    """A21: a mismatched metric count is a caller's configuration error.
    An assert vanishes under `python -O` and the zip silently truncates."""

    class _Harness:
        def eval(self):
            return [90.0, 1.0]

    def test_mismatched_reference_is_a_value_error(self):
        with pytest.raises(ValueError, match="reference"):
            ModelEvalDetector().update(
                0.0, modelHarness=self._Harness(),
                reference_validation_metrics=[90.0],
                higher_is_better={"accuracy": True, "loss": False},
            )

    def test_mismatched_directions_is_a_value_error(self):
        with pytest.raises(ValueError, match="higher_is_better"):
            ModelEvalDetector().update(
                0.0, modelHarness=self._Harness(),
                reference_validation_metrics=[90.0, 1.0],
                higher_is_better={"accuracy": True},
            )


class TestACrashedMeasurementLeavesNoNumber:
    """A50: the original recorded a block that RAISED as a completed
    zero-FLOP measurement, pulling every average over that column toward
    zero with no trace that anything failed."""

    def test_a_raising_block_records_nothing(self):
        profiler = FLOPSProfiler(warmup_iters=0)
        with pytest.raises(RuntimeError, match="boom"):
            with profiler.measure_flops("doomed"):
                raise RuntimeError("boom")
        assert "doomed_flop" not in profiler.get_performance(), (
            "a crashed measurement was reported as a real zero"
        )
        assert "doomed" not in profiler.profiles

    def test_a_working_block_still_records(self):
        import torch

        profiler = FLOPSProfiler(warmup_iters=0)
        with profiler.measure_flops("fine"):
            torch.randn(4, 4) @ torch.randn(4, 4)
        assert "fine_flop" in profiler.get_performance()
