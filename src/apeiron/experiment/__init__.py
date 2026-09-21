"""Bounded run directories and the per-run event log.

Entirely opt-in: without an ``[experiment]`` section in the config, nothing in
this module is constructed and apeiron behaves exactly as it did before.
"""

from apeiron.experiment import memmap
from apeiron.experiment.datasets import DatasetStore, Derived, EnsureResult
from apeiron.experiment.determinism import rng_state, seed_everything, set_rng_state
from apeiron.experiment.journal import Event, Journal
from apeiron.experiment.model_info import describe_model, shapes_hash
from apeiron.experiment.restart import RestartStore, RunInterrupted
from apeiron.experiment.run import Run, behavior_hash

__all__ = [
    "DatasetStore",
    "Derived",
    "EnsureResult",
    "Event",
    "Journal",
    "RestartStore",
    "Run",
    "RunInterrupted",
    "behavior_hash",
    "rng_state",
    "seed_everything",
    "set_rng_state",
    "describe_model",
    "memmap",
    "shapes_hash",
]
