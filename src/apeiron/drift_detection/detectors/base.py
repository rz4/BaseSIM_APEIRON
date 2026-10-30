"""Base types for the drift-detection layer.

Everything a detector is obliged to provide lives here: the regime ladder's
vocabulary (:class:`LearningRegime`), the record a detector hands back on every
metric value (:class:`DriftSignal`), and the abstract detector itself
(:class:`BaseDriftDetector`), including the checkpoint protocol the training
loop uses to resume a run mid-stream.
"""

from __future__ import annotations

import copy
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from dataclasses import field
from enum import Enum
from typing import Any


class LearningRegime(Enum):
    """What the detector asks the rest of the framework to do.

    The members are ordered from least to most adaptation; the monitor acts on
    the value, so these strings are part of the layer's contract.
    """

    STABLE = "stable"
    CONTINUAL_LEARNING = "continual_learning"
    FINE_TUNING = "fine_tuning"
    RETRAIN = "retrain"


#: Least to most adaptation. Used when several verdicts must be combined.
REGIME_SEVERITY: tuple[LearningRegime, ...] = (
    LearningRegime.STABLE,
    LearningRegime.CONTINUAL_LEARNING,
    LearningRegime.FINE_TUNING,
    LearningRegime.RETRAIN,
)


@dataclass(repr=False)
class DriftSignal:
    """One detector's verdict on one metric value.

    Parameters
    ----------
    regime : LearningRegime
        The adaptation the detector recommends.
    drift_detected : bool
        Whether this step fired.
    drift_score : float
        The rolling drift rate over the detector's recent window.
    confidence : float | None
        The detector's confidence, where it has one.
    metadata : dict | None
        Detector-specific extras. ``None`` becomes an empty dict, so callers
        can always subscript it.
    """

    regime: LearningRegime
    drift_detected: bool
    drift_score: float
    confidence: float | None = None
    metadata: dict[str, Any] | None = field(default=None)

    def __post_init__(self) -> None:
        if self.metadata is None:
            self.metadata = {}

    def __repr__(self) -> str:
        return (
            f"DriftSignal(regime={self.regime.value}, "
            f"detected={self.drift_detected}, "
            f"score={self.drift_score:.4f}, "
            f"confidence={self.confidence})"
        )


class BaseDriftDetector(ABC):
    """The interface every detector in this layer implements.

    Subclasses must provide :meth:`update` and :meth:`reset`. Both are declared
    abstract -- a detector that forgets one cannot be constructed -- and both
    have bodies that refuse, because a subclass may legitimately delegate
    upward (``EnsembleDetector`` wraps members; a stateless detector has
    nothing to reset) and the refusal has to name which detector it came from.
    """

    #: Attribute names this detector saves in :meth:`state_dict`. ``None``
    #: means the subclass has not opted in, which is an error rather than an
    #: empty checkpoint; declare ``()`` for a genuinely stateless detector.
    _STATE_ATTRS: tuple[str, ...] | None = None

    def __init__(self, name: str = "BaseDriftDetector"):
        self.name = name
        self._is_initialized = False
        self._drift_history: list[float] = []
        self._value_history: list[float] = []

    # -- the hooks ---------------------------------------------------------
    @abstractmethod
    def update(self, value: float, **kwargs) -> DriftSignal:
        """Feed one metric value and return the resulting signal."""
        raise NotImplementedError(f"Method not implemented for {self.name}")

    @abstractmethod
    def reset(self) -> None:
        """Forget everything learned so far."""
        raise NotImplementedError(f"Method not implemented for {self.name}")

    # -- checkpointing -----------------------------------------------------
    def state_dict(self) -> dict[str, Any]:
        """Snapshot this detector, deeply copied so it cannot be aliased."""
        attrs = type(self)._STATE_ATTRS
        if attrs is None:
            raise NotImplementedError(
                f"{type(self).__name__} does not declare `_STATE_ATTRS`, so it "
                f"cannot be checkpointed; declare the attribute names it needs "
                f"to resume (or `()` if it is stateless)"
            )
        state: dict[str, Any] = {"detector_class": type(self).__name__}
        for attr in attrs:
            state[attr] = copy.deepcopy(getattr(self, attr))
        return state

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore a snapshot taken by :meth:`state_dict` of the same class."""
        attrs = type(self)._STATE_ATTRS
        if attrs is None:
            raise NotImplementedError(
                f"{type(self).__name__} does not declare `_STATE_ATTRS`, so it "
                f"cannot be restored from a checkpoint"
            )
        saved = state.get("detector_class")
        if saved != type(self).__name__:
            raise ValueError(
                f"checkpoint was written by {saved!r}, not "
                f"{type(self).__name__!r}; refusing to load it"
            )
        missing = [attr for attr in attrs if attr not in state]
        if missing:
            raise KeyError(
                f"checkpoint for {type(self).__name__} is missing "
                f"{', '.join(missing)}"
            )
        for attr in attrs:
            setattr(self, attr, copy.deepcopy(state[attr]))

    # -- presentation ------------------------------------------------------
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"
