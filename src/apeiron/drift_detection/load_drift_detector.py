"""The factory that turns ``[drift_detection]`` config into a detector.

One entry point, :func:`load_drift_detector`, so the rest of the framework
never names a detector class: the config does.
"""

from __future__ import annotations

from apeiron.config.configuration import Config
from apeiron.config.configuration import DriftDetectionCfg
from apeiron.drift_detection.detectors.base import BaseDriftDetector
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

#: The config name of the ensemble, which cannot appear among its own members.
ENSEMBLE_NAME = "EnsembleDetector"


def _build_simple(name: str, cfg: DriftDetectionCfg) -> BaseDriftDetector:
    """Build one non-ensemble detector from the drift-detection config."""
    if name == "ADWINDetector":
        return ADWINDetector(
            delta=cfg.adwin_delta,
            minor_threshold=cfg.adwin_minor_threshold,
            moderate_threshold=cfg.adwin_moderate_threshold,
        )
    if name == "KSWINDetector":
        return KSWINDetector(
            alpha=cfg.kswin_alpha,
            window_size=cfg.kswin_window_size,
            stat_size=cfg.kswin_stat_size,
            seed=cfg.kswin_seed,
        )
    if name == "PageHinkleyDetector":
        return PageHinkleyDetector(
            min_instances=cfg.ph_min_instances,
            delta=cfg.ph_delta,
            threshold=cfg.ph_threshold,
            alpha=cfg.ph_alpha,
        )
    if name == "ModelPerformanceDetector":
        return ModelPerformanceDetector()
    if name == "EvalDetector":
        return ModelEvalDetector()
    raise ValueError(f"Unknown drift detector: {name}")


def load_drift_detector(cfg: Config) -> BaseDriftDetector:
    """Build the detector named by ``cfg.drift_detection.detector_name``.

    Parameters
    ----------
    cfg : Config
        The resolved configuration; only its ``drift_detection`` section is read.

    Returns
    -------
    BaseDriftDetector
        The detector, ready for its first :meth:`~BaseDriftDetector.update`.

    Raises
    ------
    ValueError
        If the name is not one this factory knows, or if an ensemble is
        requested with no members or with itself among them.
    """
    dd = cfg.drift_detection
    name = dd.detector_name

    if name != ENSEMBLE_NAME:
        return _build_simple(name, dd)

    members = tuple(dd.ensemble_detectors)
    if not members:
        raise ValueError(
            f"{ENSEMBLE_NAME} requires `ensemble_detectors` to name at least "
            f"one sub-detector"
        )
    if ENSEMBLE_NAME in members:
        raise ValueError(f"{ENSEMBLE_NAME} cannot be nested inside itself")

    return EnsembleDetector(
        [_build_simple(member, dd) for member in members],
        voting=dd.ensemble_voting,
    )
