"""Statistical drift detectors, each a thin wrapper around a river detector.

The wrapper's job is to turn river's per-step boolean into this layer's
:class:`DriftSignal`: it keeps a drift history, averages it over a trailing
window to get ``drift_score``, and maps that score onto the regime ladder.

The averaging window differs per detector -- 100 samples for ADWIN, 50 for
KSWIN and Page-Hinkley -- which is why a single detection scores 0.01 on the
first and 0.02 on the other two.
"""

from __future__ import annotations

from typing import Any

from river import drift

from apeiron.drift_detection.detectors.base import BaseDriftDetector
from apeiron.drift_detection.detectors.base import DriftSignal
from apeiron.drift_detection.detectors.base import LearningRegime


def _regime_for(
    drift_detected: bool,
    drift_score: float,
    minor_threshold: float,
    moderate_threshold: float,
) -> LearningRegime:
    """The ladder, as a plain if/elif chain.

    ``minor_threshold`` is tested FIRST whatever its relation to
    ``moderate_threshold``, and both comparisons are strict, so a score equal
    to a threshold is not below it. An inverted pair therefore yields the
    lightest response for a score under ``minor``, not the heaviest.
    """
    if not drift_detected:
        return LearningRegime.STABLE
    if drift_score < minor_threshold:
        return LearningRegime.CONTINUAL_LEARNING
    if drift_score < moderate_threshold:
        return LearningRegime.FINE_TUNING
    return LearningRegime.RETRAIN


class _RiverDriftDetector(BaseDriftDetector):
    """Shared machinery for the river-backed detectors.

    Subclasses provide ``_SCORE_WINDOW``, build ``self.detector`` in
    :meth:`_make_detector`, and describe themselves in :meth:`_metadata`.
    """

    #: How many recent steps ``drift_score`` averages over.
    _SCORE_WINDOW: int = 50

    _STATE_ATTRS = ("detector", "_drift_history", "_value_history")

    def __init__(
        self,
        name: str,
        minor_threshold: float = 0.3,
        moderate_threshold: float = 0.6,
    ):
        super().__init__(name)
        self.minor_threshold = minor_threshold
        self.moderate_threshold = moderate_threshold
        self.detector = self._make_detector()
        self._is_initialized = True

    def _make_detector(self) -> Any:
        raise NotImplementedError(f"Method not implemented for {self.name}")

    def _metadata(self, drift_score: float) -> dict[str, Any]:
        return {}

    def _confidence(self) -> float | None:
        return None

    def update(self, value: float, **kwargs) -> DriftSignal:
        """Feed one metric value through the river detector."""
        self._value_history.append(value)
        self.detector.update(value)
        drift_detected = bool(self.detector.drift_detected)
        self._drift_history.append(1 if drift_detected else 0)

        recent = self._drift_history[-self._SCORE_WINDOW :]
        drift_score = sum(recent) / len(recent) if recent else 0.0

        return DriftSignal(
            regime=_regime_for(
                drift_detected,
                drift_score,
                self.minor_threshold,
                self.moderate_threshold,
            ),
            drift_detected=drift_detected,
            drift_score=drift_score,
            confidence=self._confidence(),
            metadata=self._metadata(drift_score),
        )

    def reset(self) -> None:
        """Rebuild the river detector and drop both histories."""
        self.detector = self._make_detector()
        self._drift_history = []
        self._value_history = []


class ADWINDetector(_RiverDriftDetector):
    """ADWIN: adaptive windowing on the metric's mean."""

    _SCORE_WINDOW = 100

    def __init__(
        self,
        delta: float = 0.002,
        minor_threshold: float = 0.3,
        moderate_threshold: float = 0.6,
    ):
        self.delta = delta
        super().__init__(
            name="ADWIN",
            minor_threshold=minor_threshold,
            moderate_threshold=moderate_threshold,
        )

    def _make_detector(self) -> Any:
        return drift.ADWIN(delta=self.delta)

    def _confidence(self) -> float | None:
        return 1.0 - self.delta

    def _metadata(self, drift_score: float) -> dict[str, Any]:
        return {
            "window_size": getattr(self.detector, "width", None),
            "n_detections": sum(self._drift_history),
            "recent_drift_rate": drift_score,
        }


class KSWINDetector(_RiverDriftDetector):
    """KSWIN: a Kolmogorov-Smirnov test between a reference and a recent window.

    River draws the reference window at random, so ``seed`` is what makes a run
    reproducible from one process to the next. It defaults to ``None``, which
    does not.
    """

    _SCORE_WINDOW = 50

    def __init__(
        self,
        alpha: float = 0.005,
        window_size: int = 100,
        stat_size: int = 30,
        seed: int | None = None,
        minor_threshold: float = 0.3,
        moderate_threshold: float = 0.6,
    ):
        self.alpha = alpha
        self.window_size = window_size
        self.stat_size = stat_size
        self.seed = seed
        super().__init__(
            name="KSWIN",
            minor_threshold=minor_threshold,
            moderate_threshold=moderate_threshold,
        )

    def _make_detector(self) -> Any:
        return drift.KSWIN(
            alpha=self.alpha,
            window_size=self.window_size,
            stat_size=self.stat_size,
            seed=self.seed,
        )

    def _confidence(self) -> float | None:
        return 1.0 - self.alpha

    def _metadata(self, drift_score: float) -> dict[str, Any]:
        return {
            "window_size": self.window_size,
            "stat_size": self.stat_size,
            "n_detections": sum(self._drift_history),
            "recent_drift_rate": drift_score,
        }


class PageHinkleyDetector(_RiverDriftDetector):
    """Page-Hinkley: a cumulative-sum test on the metric's mean."""

    _SCORE_WINDOW = 50

    def __init__(
        self,
        min_instances: int = 30,
        delta: float = 0.005,
        threshold: float = 50.0,
        alpha: float = 0.9999,
        minor_threshold: float = 0.3,
        moderate_threshold: float = 0.6,
    ):
        self.min_instances = min_instances
        self.delta = delta
        self.threshold = threshold
        self.alpha = alpha
        super().__init__(
            name="PageHinkley",
            minor_threshold=minor_threshold,
            moderate_threshold=moderate_threshold,
        )

    def _make_detector(self) -> Any:
        return drift.PageHinkley(
            min_instances=self.min_instances,
            delta=self.delta,
            threshold=self.threshold,
            alpha=self.alpha,
        )

    def _metadata(self, drift_score: float) -> dict[str, Any]:
        return {
            "threshold": self.threshold,
            "min_instances": self.min_instances,
            "n_detections": sum(self._drift_history),
            "recent_drift_rate": drift_score,
        }
