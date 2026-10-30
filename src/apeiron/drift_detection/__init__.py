"""Drift detection: detectors, their shared base, and the config-driven factory.

Re-exported here because sibling layers (`apeiron.monitoring`, the training
loop) import these names from the package rather than from the modules.
"""

from apeiron.drift_detection.detectors.base import BaseDriftDetector
from apeiron.drift_detection.detectors.base import DriftSignal
from apeiron.drift_detection.detectors.base import LearningRegime
from apeiron.drift_detection.detectors.model_performance_detector import (
    EnsembleDetector,
)
from apeiron.drift_detection.detectors.model_performance_detector import (
    ModelEvalDetector,
)
from apeiron.drift_detection.detectors.model_performance_detector import (
    ModelPerformanceDetector,
)
from apeiron.drift_detection.detectors.statistical_detectors import ADWINDetector
from apeiron.drift_detection.detectors.statistical_detectors import KSWINDetector
from apeiron.drift_detection.detectors.statistical_detectors import (
    PageHinkleyDetector,
)

__all__ = [
    "ADWINDetector",
    "BaseDriftDetector",
    "DriftSignal",
    "EnsembleDetector",
    "KSWINDetector",
    "LearningRegime",
    "ModelEvalDetector",
    "ModelPerformanceDetector",
    "PageHinkleyDetector",
]
