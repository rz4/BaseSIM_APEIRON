"""Apeiron: a framework for drift-driven continual learning.

The names re-exported here are the framework's public surface: an external
harness author only ever needs ``import apeiron``.
"""

from apeiron.config.configuration import (
    Config,
    ContinualLearningCfg,
    DataCfg,
    DriftDetectionCfg,
    LoggingCfg,
    ModelCfg,
    TrainCfg,
    build_config,
)
from apeiron.drift_detection.detectors.base import (
    BaseDriftDetector,
    DriftSignal,
    LearningRegime,
)
from apeiron.drift_detection.detectors.model_performance_detector import (
    EnsembleDetector,
    ModelEvalDetector,
    ModelPerformanceDetector,
)
from apeiron.drift_detection.detectors.statistical_detectors import (
    ADWINDetector,
    KSWINDetector,
    PageHinkleyDetector,
)
from apeiron.driver.continuous_monitor import ContinuousMonitor
from apeiron.logger.logger import Logger, get_logger
from apeiron.model.torch_model_harness import BaseModelHarness
from apeiron.training.continuous_trainer import ContinuousTrainer
from apeiron.training.updater.base import BaseUpdater

__all__ = [
    # configuration
    "Config",
    "ModelCfg",
    "DataCfg",
    "TrainCfg",
    "ContinualLearningCfg",
    "DriftDetectionCfg",
    "LoggingCfg",
    "build_config",
    # extension points
    "BaseModelHarness",
    "BaseUpdater",
    "BaseDriftDetector",
    # drift detection
    "DriftSignal",
    "LearningRegime",
    "ADWINDetector",
    "KSWINDetector",
    "PageHinkleyDetector",
    "ModelPerformanceDetector",
    "ModelEvalDetector",
    "EnsembleDetector",
    # drivers
    "ContinuousMonitor",
    "ContinuousTrainer",
    # logging
    "Logger",
    "get_logger",
]
