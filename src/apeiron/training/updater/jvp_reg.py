"""A SAM-style robust update over the current and historical streams.

Despite the module name the update is first-order: no Jacobian-vector product
is formed. One step, given a current batch ``(x, y)`` and a historical batch
``(xh, yh)``:

1. The gradient of the loss on the **current batch only** is taken. The
   historical batch does not enter this direction.
2. The parameters are moved along the **negative**, normalised current
   gradient -- a descent step of length ``jvp_rho_theta``, not an ascent one::

       theta' = theta + jvp_rho_theta * (-g_cur / (||g_cur|| + 1e-12))

   with ``||g_cur||`` the single norm over all parameters at once.
3. The historical inputs are shifted along the normalised, signed drift
   direction between the two batches' means::

       d  = jvp_data_sign * (mean(x) - mean(xh));  d <- d / (||d|| + 1e-12)
       xh <- xh + jvp_rho_x * d

   ``jvp_rho_x = 0`` skips the shift.
4. The backward pass is taken **at the perturbed point** over the half/half
   mixed batch (the same mixing :class:`BaseUpdater` uses), and the parameters
   are then restored exactly -- only ``.grad`` survives the call.

With no historical batch there is nothing to be robust to, so the step is a
plain current-only forward/backward and the reported forgetting loss is
``0.0``. With one, ``update_post_fwd_bwd`` reports the mixed loss accumulated
since the last report and resets it.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from apeiron.config.configuration import Config
from apeiron.training.updater.base import Batch, BaseUpdater

#: Guards both normalisations against a vanishing direction.
EPS = 1e-12


class JVPRegUpdater(BaseUpdater):
    """First-order robust update: perturb, then differentiate the mixture."""

    def __init__(self, cfg: Config, modelHarness) -> None:
        super().__init__(cfg, modelHarness)
        cl = cfg.continual_learning
        self.rho_theta = float(cl.jvp_rho_theta)
        self.rho_x = float(cl.jvp_rho_x)
        self.data_sign = float(cl.jvp_data_sign)

        #: The mixed loss accumulated since the last report.
        self.loss_mem = 0.0

    # ------------------------------------------------------------------
    def update_pre_fwd_bwd(self) -> None:
        self.loss_mem = 0.0

    def fwd_bwd(self, batch: Batch, hist_batch: Optional[Batch] = None) -> float:
        x, y = self._unpack(batch)
        self.model.train()

        if hist_batch is None:
            loss = self.criterion(self.model(x), y) / self.grad_accumulation_steps
            loss.backward()
            return float(loss.item())

        xh, yh = self._unpack(hist_batch)
        params = self._trainable()

        # 1. the current-only descent direction
        loss_cur = self.criterion(self.model(x), y) / self.grad_accumulation_steps
        g_cur = torch.autograd.grad(loss_cur, params)
        g_norm = torch.sqrt(torch.stack([(g**2).sum() for g in g_cur]).sum()) + EPS

        # 2. move there, remembering where we came from
        original = [p.detach().clone() for p in params]
        with torch.no_grad():
            for param, grad in zip(params, g_cur):
                param.add_(self.rho_theta * (-grad / g_norm))

        try:
            # 3. + 4. differentiate the mixture at the perturbed point
            x_mix, y_mix = self._perturbed_mix(x, y, xh, yh)
            loss_mix = (
                self.criterion(self.model(x_mix), y_mix) / self.grad_accumulation_steps
            )
            loss_mix.backward()
            value = float(loss_mix.item())
        finally:
            with torch.no_grad():
                for param, saved in zip(params, original):
                    param.copy_(saved)

        self.loss_mem += value
        return value

    def update_post_fwd_bwd(self) -> float:
        reported = float(self.loss_mem)
        self.loss_mem = 0.0
        return reported

    # ------------------------------------------------------------------
    def _perturbed_mix(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        xh: torch.Tensor,
        yh: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The half/half batch, with the historical half shifted by ``rho_x``."""
        n_total = x.shape[0]
        n_hist = min(n_total // 2, xh.shape[0])
        n_cur = n_total - n_hist

        x_hist = xh[:n_hist]
        if self.rho_x != 0.0 and n_hist > 0:
            direction = self.data_sign * (
                x.reshape(x.shape[0], -1).mean(0)
                - xh.reshape(xh.shape[0], -1).mean(0)
            )
            direction = direction / (direction.norm() + EPS)
            x_hist = x_hist + self.rho_x * direction.reshape(xh.shape[1:])

        return (
            torch.cat([x[:n_cur], x_hist], dim=0),
            torch.cat([y[:n_cur], yh[:n_hist]], dim=0),
        )
