"""The plain continual-learning update: one forward/backward over the batch.

Every other updater is a subclass, so the contract lives here:

``fwd_bwd(batch, hist_batch=None)``
    Runs the forward/backward pass and returns the scalar training loss as a
    plain ``float``. This is the only call the trainer wraps in its FLOP
    counter, so an updater that needs extra second-order arithmetic does it in
    :meth:`update_post_fwd_bwd` instead -- the measured number is meant to be
    the cost of the model's own pass.

``update_post_fwd_bwd()``
    Applies whatever regularisation term the mode has and returns it as a
    ``float``. The trainer logs it as the *forgetting loss* and adds it to the
    generation loss, so it is always a number and it is ``0.0`` when no
    historical information has entered the update yet.

The remaining hooks (:meth:`cl_preprocessing`, :meth:`update_pre_fwd_bwd`,
:meth:`update_post_optimizer_call`, :meth:`cl_postprocessing`) bracket a
continual-learning application and do nothing in the base mode.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional, Tuple

import torch

from apeiron.config.configuration import Config

if TYPE_CHECKING:  # pragma: no cover - typing only
    from apeiron.model.torch_model_harness import BaseModelHarness

Batch = Tuple[torch.Tensor, torch.Tensor]


class BaseUpdater:
    """A vanilla update step, optionally replaying a historical batch."""

    def __init__(self, cfg: Config, modelHarness: "BaseModelHarness") -> None:
        self.cfg = cfg
        self.modelHarness = modelHarness
        self.model = modelHarness.model
        self.criterion = modelHarness.get_criterion()
        self.device = torch.device(cfg.device)
        self.grad_accumulation_steps = max(1, int(cfg.train.grad_accumulation_steps))
        self.mix_historic_data = bool(cfg.continual_learning.mix_historic_data)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _unpack(self, batch: Batch) -> Batch:
        """Harness-aware unpack, moved onto the configured device."""
        x, y = self.modelHarness._unpack(batch)
        return x.to(self.device), y.to(self.device)

    def _trainable(self) -> List[torch.Tensor]:
        return [p for p in self.model.parameters() if p.requires_grad]

    @staticmethod
    def _mix(
        x: torch.Tensor,
        y: torch.Tensor,
        xh: torch.Tensor,
        yh: torch.Tensor,
    ) -> Batch:
        """Half the current batch followed by half as many historical samples.

        The mixed batch keeps the current batch's size: ``n_total // 2`` samples
        come from the history (or all of it, if it is shorter) and the rest from
        the head of the current batch.
        """
        n_total = x.shape[0]
        n_hist = min(n_total // 2, xh.shape[0])
        if n_hist == 0:
            return x, y
        n_cur = n_total - n_hist
        return (
            torch.cat([x[:n_cur], xh[:n_hist]], dim=0),
            torch.cat([y[:n_cur], yh[:n_hist]], dim=0),
        )

    # ------------------------------------------------------------------
    # the measured pass
    # ------------------------------------------------------------------
    def fwd_bwd(self, batch: Batch, hist_batch: Optional[Batch] = None) -> float:
        """One forward/backward, accumulating into ``param.grad``."""
        x, y = self._unpack(batch)
        if self.mix_historic_data and hist_batch is not None:
            xh, yh = self._unpack(hist_batch)
            x, y = self._mix(x, y, xh, yh)

        self.model.train()
        loss = self.criterion(self.model(x), y) / self.grad_accumulation_steps
        loss.backward()
        return float(loss.item())

    # ------------------------------------------------------------------
    # lifecycle hooks -- no-ops for the base mode
    # ------------------------------------------------------------------
    def cl_preprocessing(self) -> None:
        """Called once before a continual-learning application starts."""

    def update_pre_fwd_bwd(self) -> None:
        """Called before every :meth:`fwd_bwd`."""

    def update_post_fwd_bwd(self) -> float:
        """Apply the mode's regularisation; return the logged forgetting loss."""
        return 0.0

    def update_post_optimizer_call(self) -> None:
        """Called after every ``optimizer.step()``."""

    def cl_postprocessing(self) -> None:
        """Called once after a continual-learning application finishes."""
