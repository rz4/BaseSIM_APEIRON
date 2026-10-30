"""The null update: do nothing at all.

Selected by ``update_mode = "none"``. Used to run the drift-detection and
monitoring machinery without adapting the model, so the loss it reports is the
sentinel ``-1.0`` rather than a measured number -- there was no pass to
measure.
"""

from __future__ import annotations

from typing import Optional

from apeiron.training.updater.base import Batch, BaseUpdater

#: What :meth:`NoUpdater.fwd_bwd` reports in place of a training loss.
NO_LOSS = -1.0


class NoUpdater(BaseUpdater):
    """Runs no forward/backward pass and leaves the gradients untouched."""

    def fwd_bwd(self, batch: Batch, hist_batch: Optional[Batch] = None) -> float:
        return NO_LOSS
