"""Evidently-backed detectors and the voting ensemble.

:class:`ModelPerformanceDetector` compares a current data frame against a
stored reference with an evidently data-drift report. :class:`ModelEvalDetector`
re-evaluates the model instead and compares validation metrics against their
reference. :class:`EnsembleDetector` combines any number of detectors by vote.
"""

from __future__ import annotations

from typing import Any

import evidently
from evidently.presets import DataDriftPreset

from apeiron.drift_detection.detectors.base import REGIME_SEVERITY
from apeiron.drift_detection.detectors.base import BaseDriftDetector
from apeiron.drift_detection.detectors.base import DriftSignal
from apeiron.drift_detection.detectors.base import LearningRegime

#: The metric inside the data-drift preset that carries the drifted-column
#: tally. Evidently decorates the name with its parameters, so this is a prefix.
DRIFTED_COLUMNS_METRIC = "DriftedColumnsCount"

#: How many checks ``drift_score`` averages over on the evidently path.
SCORE_WINDOW = 10

#: The fallback path's upper boundary, above which it asks for a full retrain.
RETRAIN_SCORE = 0.8

#: Canonical voting strategies, and the aliases accepted for them.
VOTING_ALIASES = {
    "majority": "majority",
    "any": "any",
    "or": "any",
    "unanimous": "unanimous",
    "all": "unanimous",
    "and": "unanimous",
}


class ModelPerformanceDetector(BaseDriftDetector):
    """Data drift between a reference frame and each incoming frame.

    The evidently path reports the share of drifted columns; ``drift_detected``
    is that share strictly above ``drift_share_threshold``, and the regime
    comes from the mean share over the last :data:`SCORE_WINDOW` checks. A
    caller with no frame may pass a bare ``value`` instead, which takes a
    simpler ladder.
    """

    _STATE_ATTRS = ("_drift_history", "_value_history")

    def __init__(
        self,
        reference_data: Any = None,
        reference_predictions: Any = None,
        reference_targets: Any = None,
        drift_share_threshold: float = 0.5,
        minor_threshold: float = 0.3,
        moderate_threshold: float = 0.6,
        name: str = "ModelPerformance",
    ):
        super().__init__(name)
        self.reference_data = (
            reference_data.copy() if reference_data is not None else None
        )
        self.reference_predictions = reference_predictions
        self.reference_targets = reference_targets
        self.drift_share_threshold = drift_share_threshold
        self.minor_threshold = minor_threshold
        self.moderate_threshold = moderate_threshold
        self._is_initialized = self.reference_data is not None

    # -- reference lifecycle ----------------------------------------------
    def set_reference(
        self,
        reference_data: Any,
        predictions: Any = None,
        targets: Any = None,
    ) -> None:
        """Install (or replace) the frame every later check compares against.

        The frame is copied, so a caller that mutates its own frame afterwards
        cannot change what the detector compares against.
        """
        self.reference_data = reference_data.copy()
        self.reference_predictions = predictions
        self.reference_targets = targets
        self._is_initialized = True

    def reset(self) -> None:
        """Drop the drift history; the reference survives."""
        self._drift_history = []
        self._value_history = []

    # -- the check ---------------------------------------------------------
    def update(self, value: float | None = None, **kwargs) -> DriftSignal:
        """Run one drift check, from a data frame or from a bare value."""
        if not self._is_initialized:
            raise ValueError(
                f"{type(self).__name__} is not initialized; call "
                f"set_reference() before the first check"
            )

        data = kwargs.get("data")
        if data is not None:
            return self._update_from_data(
                data,
                predictions=kwargs.get("predictions"),
                targets=kwargs.get("targets"),
            )
        if value is not None:
            return self._update_from_value(float(value))
        raise ValueError(
            "Either 'data' DataFrame or 'value' must be provided to "
            f"{type(self).__name__}.update()"
        )

    def _compared_frames(self, data, predictions, targets):
        """The reference/current pair, with predictions and targets joined on."""
        reference = self.reference_data.copy()
        current = data.copy()
        if predictions is not None and self.reference_predictions is not None:
            reference["prediction"] = self.reference_predictions
            current["prediction"] = predictions
        if targets is not None and self.reference_targets is not None:
            reference["target"] = self.reference_targets
            current["target"] = targets
        return reference, current

    def _update_from_data(self, data, predictions=None, targets=None) -> DriftSignal:
        reference, current = self._compared_frames(data, predictions, targets)

        report = evidently.Report(metrics=[DataDriftPreset()])
        result = report.run(reference_data=reference, current_data=current)
        count, share = self._parse(result.dict())

        self._drift_history.append(share)
        recent = self._drift_history[-SCORE_WINDOW:]
        drift_score = sum(recent) / len(recent) if recent else 0.0
        drift_detected = share > self.drift_share_threshold

        if drift_score >= self.moderate_threshold:
            regime = LearningRegime.RETRAIN
        elif drift_score >= self.minor_threshold:
            regime = LearningRegime.FINE_TUNING
        elif drift_detected:
            regime = LearningRegime.CONTINUAL_LEARNING
        else:
            regime = LearningRegime.STABLE

        return DriftSignal(
            regime=regime,
            drift_detected=drift_detected,
            drift_score=drift_score,
            confidence=0.95,
            metadata={
                "drift_share": share,
                "n_drifted_columns": int(count),
                "n_columns": len(current.columns),
            },
        )

    @staticmethod
    def _parse(result: dict) -> tuple[float, float]:
        """Pull the drifted-column count and share out of an evidently result."""
        metrics = result.get("metrics")
        if not metrics:
            raise KeyError("No metrics found in the evidently result")

        for metric in metrics:
            name = str(metric.get("metric_name", ""))
            if not name.startswith(DRIFTED_COLUMNS_METRIC):
                continue
            payload = metric.get("value")
            if not isinstance(payload, dict):
                raise ValueError(
                    f"Expected dict value for {DRIFTED_COLUMNS_METRIC}, got "
                    f"{type(payload).__name__}"
                )
            return payload.get("count", 0.0), payload.get("share", 0.0)

        raise KeyError(
            f"{DRIFTED_COLUMNS_METRIC} metric not found in the evidently result"
        )

    def _update_from_value(self, value: float) -> DriftSignal:
        """The no-frame fallback: a bare score, on its own ladder."""
        self._drift_history.append(value)
        recent = self._drift_history[-SCORE_WINDOW:]
        drift_score = sum(recent) / len(recent) if recent else 0.0

        if drift_score < self.minor_threshold:
            regime = LearningRegime.STABLE
        elif drift_score < self.moderate_threshold:
            regime = LearningRegime.CONTINUAL_LEARNING
        elif drift_score < RETRAIN_SCORE:
            regime = LearningRegime.FINE_TUNING
        else:
            regime = LearningRegime.RETRAIN

        return DriftSignal(
            regime=regime,
            drift_detected=drift_score > self.drift_share_threshold,
            drift_score=drift_score,
            confidence=0.95,
        )


class ModelEvalDetector(BaseDriftDetector):
    """Drift as a drop in the model's own validation metrics.

    Stateless: it re-evaluates through the harness the caller passes and
    compares each metric against its reference, in the direction
    ``higher_is_better`` declares.
    """

    _STATE_ATTRS = ()

    def __init__(self, name: str = "ModelEval"):
        super().__init__(name)
        self._is_initialized = True

    def update(self, value: float, **kwargs) -> DriftSignal:
        """Re-evaluate the model and compare against the reference metrics."""
        harness = kwargs.get("modelHarness")
        if harness is None:
            raise ValueError(
                f"modelHarness must be provided to {type(self).__name__}.update()"
            )
        reference = kwargs.get("reference_validation_metrics") or []
        higher_is_better = kwargs.get("higher_is_better") or {}

        current = list(harness.eval())
        reference = list(reference)
        directions = list(higher_is_better.values())
        # ValueError, not assert: a mismatched metric count is a caller's
        # configuration error, and an assert vanishes under `python -O`,
        # leaving the zip below to silently truncate (claims/LOG.md A21).
        if len(current) != len(reference):
            raise ValueError(
                f"{type(self).__name__} got {len(current)} current metrics and "
                f"{len(reference)} reference metrics"
            )
        if len(current) != len(directions):
            raise ValueError(
                f"{type(self).__name__} got {len(current)} metrics and "
                f"{len(directions)} entries in `higher_is_better`"
            )

        regressed = [
            i
            for i, (now, before, higher) in enumerate(
                zip(current, reference, directions)
            )
            if (now < before if higher else now > before)
        ]
        drift_detected = bool(regressed)

        return DriftSignal(
            regime=LearningRegime.CONTINUAL_LEARNING
            if drift_detected
            else LearningRegime.STABLE,
            drift_detected=drift_detected,
            drift_score=len(regressed) / len(current) if current else 0.0,
            metadata={
                "current_metrics": current,
                "reference_metrics": reference,
                "regressed_metrics": regressed,
            },
        )

    def reset(self) -> None:
        """Nothing is kept between checks, so there is nothing to clear."""
        return None


class EnsembleDetector(BaseDriftDetector):
    """Several detectors, one verdict, decided by vote.

    ``voting`` is ``"majority"`` (strictly more than half), ``"any"`` (aliased
    ``"or"``) or ``"unanimous"`` (aliased ``"all"`` and ``"and"``).
    """

    _STATE_ATTRS = ()

    def __init__(
        self,
        detectors: list[BaseDriftDetector],
        voting: str = "majority",
        name: str = "Ensemble",
    ):
        super().__init__(name)
        detectors = list(detectors)
        if not detectors:
            raise ValueError(
                f"{type(self).__name__} needs at least one sub-detector"
            )
        canonical = VOTING_ALIASES.get(str(voting).lower())
        if canonical is None:
            raise ValueError(
                f"unknown voting strategy {voting!r}; expected one of "
                f"{sorted(set(VOTING_ALIASES))}"
            )
        self.detectors = detectors
        self.voting = canonical
        self._is_initialized = True

    def update(self, value: float, **kwargs) -> DriftSignal:
        """Poll every sub-detector and combine their verdicts."""
        signals = [detector.update(value, **kwargs) for detector in self.detectors]
        votes = [signal.drift_detected for signal in signals]
        n_votes = sum(1 for vote in votes if vote)
        n_detectors = len(signals)

        if self.voting == "majority":
            drift_detected = n_votes > n_detectors / 2
        elif self.voting == "any":
            drift_detected = n_votes > 0
        elif self.voting == "unanimous":
            drift_detected = n_votes == n_detectors
        else:
            raise ValueError(f"unknown voting strategy {self.voting!r}")

        if drift_detected:
            firing = [s.regime for s in signals if s.drift_detected] or [
                s.regime for s in signals
            ]
            regime = max(firing, key=REGIME_SEVERITY.index)
        else:
            regime = LearningRegime.STABLE

        confidences = [s.confidence for s in signals if s.confidence is not None]
        scores = [s.drift_score for s in signals]

        return DriftSignal(
            regime=regime,
            drift_detected=drift_detected,
            drift_score=sum(scores) / len(scores) if scores else 0.0,
            confidence=(
                sum(confidences) / len(confidences) if confidences else None
            ),
            metadata={
                "n_detectors": n_detectors,
                "n_votes": n_votes,
                "voting": self.voting,
                "detector_names": [d.name for d in self.detectors],
            },
        )

    def reset(self) -> None:
        """Reset every sub-detector."""
        for detector in self.detectors:
            detector.reset()

    # -- checkpointing -----------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        """The ensemble's state is its sub-detectors', positionally aligned."""
        return {
            "detector_class": type(self).__name__,
            "sub_detectors": [d.state_dict() for d in self.detectors],
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore sub-detector state, position by position."""
        saved = state.get("detector_class")
        if saved != type(self).__name__:
            raise ValueError(
                f"checkpoint was written by {saved!r}, not "
                f"{type(self).__name__!r}; refusing to load it"
            )
        if "sub_detectors" not in state:
            raise KeyError(
                f"checkpoint for {type(self).__name__} is missing 'sub_detectors'"
            )
        sub_states = state["sub_detectors"]
        if len(sub_states) != len(self.detectors):
            raise ValueError(
                f"checkpoint holds {len(sub_states)} sub-detectors, but this "
                f"ensemble has {len(self.detectors)}"
            )
        for detector, sub_state in zip(self.detectors, sub_states):
            detector.load_state_dict(sub_state)
