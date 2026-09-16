"""Bounded run directories and the per-run event log.

Entirely opt-in: without an ``[experiment]`` section in the config, nothing in
this module is constructed and apeiron behaves exactly as it did before.
"""

from apeiron.experiment.journal import Event, Journal
from apeiron.experiment.run import Run, behavior_hash

__all__ = ["Event", "Journal", "Run", "behavior_hash"]
