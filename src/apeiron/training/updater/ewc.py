"""Online Elastic Weight Consolidation.

The mode keeps one diagonal curvature estimate for the whole run rather than
one per task:

* During a continual-learning application the squared gradients of every step
  are accumulated (``update_post_fwd_bwd``, from ``param.grad`` *before* the
  penalty term is added -- the penalty is not part of the likelihood whose
  curvature is being estimated).
* At the end of the application the running estimate is folded in with an
  exponential decay and the anchor is moved to the current weights::

      F     <- ewc_ema_decay * F + mean_over_steps(grad ** 2)
      theta* <- theta

* Every step pays the quadratic penalty around the anchor::

      penalty = 0.5 * ewc_lambda * sum(F * (theta - theta*) ** 2)
      grad    += ewc_lambda * F * (theta - theta*)

The penalty is exactly ``0.0`` until the first commit: ``F`` starts at zero, so
there is nothing to forget and nothing to report.

All of the arithmetic is element-wise on ``.grad``, which is why this mode adds
no matmuls on either side of the measured pass.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

from apeiron.config.configuration import Config
from apeiron.training.updater.base import BaseUpdater


class OnlineEWCUpdater(BaseUpdater):
    """EWC with a single, exponentially-decayed diagonal Fisher estimate."""

    def __init__(self, cfg: Config, modelHarness) -> None:
        super().__init__(cfg, modelHarness)
        cl = cfg.continual_learning
        self.ewc_lambda = float(cl.ewc_lambda)
        self.ewc_ema_decay = float(cl.ewc_ema_decay)

        #: The anchor the penalty pulls towards, and its curvature.
        self.theta_star: Dict[str, torch.Tensor] = {}
        self.fisher: Dict[str, torch.Tensor] = {}
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            self.theta_star[name] = param.detach().clone()
            self.fisher[name] = torch.zeros_like(param)

        #: Per-application state, live only between the cl_* hooks.
        self._cl_fisher_accum: Optional[Dict[str, torch.Tensor]] = None
        self._cl_steps = 0

    # ------------------------------------------------------------------
    def cl_preprocessing(self) -> None:
        self._cl_fisher_accum = {
            name: torch.zeros_like(f) for name, f in self.fisher.items()
        }
        self._cl_steps = 0

    def update_post_fwd_bwd(self) -> float:
        with torch.no_grad():
            self._accumulate_fisher()
            return self._apply_penalty()

    def update_post_optimizer_call(self) -> None:
        self._cl_steps += 1

    def cl_postprocessing(self) -> None:
        """Fold the application's estimate into the prior and re-anchor.

        A step that was never prepared for -- ``update_post_optimizer_call``
        without a preceding ``cl_preprocessing`` -- commits nothing: there is no
        accumulator to commit.
        """
        if self._cl_fisher_accum is not None and self._cl_steps > 0:
            with torch.no_grad():
                for name in self.fisher:
                    self.fisher[name] = (
                        self.ewc_ema_decay * self.fisher[name]
                        + self._cl_fisher_accum[name] / self._cl_steps
                    )
                for name, param in self.model.named_parameters():
                    if name in self.theta_star:
                        self.theta_star[name] = param.detach().clone()
        self._cl_fisher_accum = None
        self._cl_steps = 0

    # ------------------------------------------------------------------
    def _accumulate_fisher(self) -> None:
        if self._cl_fisher_accum is None:
            return
        for name, param in self.model.named_parameters():
            if name not in self._cl_fisher_accum or param.grad is None:
                continue
            self._cl_fisher_accum[name] += param.grad.detach().pow(2)

    def _apply_penalty(self) -> float:
        penalty = 0.0
        for name, param in self.model.named_parameters():
            if name not in self.fisher:
                continue
            fisher = self.fisher[name]
            delta = param.detach() - self.theta_star[name]
            penalty += float((fisher * delta.pow(2)).sum().item())
            if param.grad is None:
                param.grad = torch.zeros_like(param)
            param.grad += self.ewc_lambda * fisher * delta
        return 0.5 * self.ewc_lambda * penalty
