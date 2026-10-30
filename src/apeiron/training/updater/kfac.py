"""Online K-FAC: a Kronecker-factored quadratic penalty around an anchor.

For every ``Linear`` / ``Conv2d`` layer two curvature factors are kept, built
from the layer's inputs ``a`` and the gradients of its outputs ``g``::

    A <- kfac_ema_decay * A + mean_over_steps(aT a / rows(a))
    G <- kfac_ema_decay * G + mean_over_steps(gT g / rows(g))

with ``a`` the ``(rows, in_features)`` activation matrix -- for a convolution
the unfolded patch matrix ``(N * L, in_channels * kH * kW)`` -- and ``g`` the
matching ``(rows, out_channels)`` grad-output matrix.

The penalty on the layer's weight ``W`` (flattened to ``(out, -1)``) is::

    penalty = 0.5 * kfac_lambda * sum(dW * (G dW A))
    grad    += kfac_lambda * (G dW A).view_as(W)        with dW = W - W*

WHERE THE ARITHMETIC HAPPENS. The hooks only *store* ``a`` and ``g``; the two
contractions ``aT a`` and ``gT g`` are the most expensive matmuls in the whole
step for a convolution, and they are performed in :meth:`update_post_fwd_bwd`
-- outside the window the trainer wraps in its FLOP counter. Accumulating them
in the backward hook instead is arithmetically equivalent but reports a cost of
a continual-learning update several times larger than the model's own pass.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from apeiron.config.configuration import Config
from apeiron.training.updater.base import BaseUpdater


class OnlineKFACUpdater(BaseUpdater):
    """K-FAC with one exponentially-decayed pair of factors per layer."""

    def __init__(self, cfg: Config, modelHarness) -> None:
        super().__init__(cfg, modelHarness)
        cl = cfg.continual_learning
        self.kfac_lambda = float(cl.kfac_lambda)
        self.kfac_ema_decay = float(cl.kfac_ema_decay)

        #: Tracked layers, by the name they carry in the model.
        self.layers: Dict[str, nn.Module] = {}
        self.theta_star: Dict[str, torch.Tensor] = {}
        self.A: Dict[str, torch.Tensor] = {}
        self.G: Dict[str, torch.Tensor] = {}

        #: Tensors captured by the hooks, consumed by the post-pass hook.
        self._a_cache: Dict[str, torch.Tensor] = {}
        self._g_cache: Dict[str, torch.Tensor] = {}
        self._handles: List[object] = []

        for name, module in self.model.named_modules():
            if not self._supported(module):
                continue
            weight = module.weight
            self.layers[name] = module
            self.theta_star[name] = weight.detach().clone()
            a_dim = self._a_dim(module)
            g_dim = self._g_dim(module)
            self.A[name] = torch.zeros(
                a_dim, a_dim, device=weight.device, dtype=weight.dtype
            )
            self.G[name] = torch.zeros(
                g_dim, g_dim, device=weight.device, dtype=weight.dtype
            )
            self._register_hooks(name, module)

        #: Per-application state, live only between the cl_* hooks.
        self._A_accum: Optional[Dict[str, torch.Tensor]] = None
        self._G_accum: Optional[Dict[str, torch.Tensor]] = None
        self._cl_steps = 0

    # ------------------------------------------------------------------
    # layer introspection
    # ------------------------------------------------------------------
    @staticmethod
    def _supported(module: nn.Module) -> bool:
        """Only layers whose weight has a Kronecker structure are tracked."""
        return isinstance(module, (nn.Linear, nn.Conv2d))

    @staticmethod
    def _a_dim(module: nn.Module) -> int:
        if isinstance(module, nn.Conv2d):
            kh, kw = module.kernel_size
            return module.in_channels * kh * kw
        return module.in_features

    @staticmethod
    def _g_dim(module: nn.Module) -> int:
        if isinstance(module, nn.Conv2d):
            return module.out_channels
        return module.out_features

    @staticmethod
    def _extract_a(module: nn.Module, a: torch.Tensor) -> torch.Tensor:
        """The layer's inputs as a ``(rows, a_dim)`` matrix."""
        if isinstance(module, nn.Conv2d):
            patches = F.unfold(
                a,
                kernel_size=module.kernel_size,
                dilation=module.dilation,
                padding=module.padding,
                stride=module.stride,
            )
            return patches.transpose(1, 2).reshape(-1, patches.shape[1])
        return a.reshape(-1, a.shape[-1])

    @staticmethod
    def _extract_g(module: nn.Module, g: torch.Tensor) -> torch.Tensor:
        """The gradients of the layer's outputs as a ``(rows, g_dim)`` matrix."""
        if isinstance(module, nn.Conv2d):
            return g.permute(0, 2, 3, 1).reshape(-1, g.shape[1])
        return g.reshape(-1, g.shape[-1])

    def _register_hooks(self, name: str, module: nn.Module) -> None:
        """Store the pass's activations and grad-outputs -- nothing more."""

        def forward_hook(_module, inputs, _output, _name=name):
            self._a_cache[_name] = inputs[0].detach()

        def backward_hook(_module, _grad_input, grad_output, _name=name):
            self._g_cache[_name] = grad_output[0].detach()

        self._handles.append(module.register_forward_hook(forward_hook))
        self._handles.append(module.register_full_backward_hook(backward_hook))

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def cl_preprocessing(self) -> None:
        self._A_accum = {name: torch.zeros_like(a) for name, a in self.A.items()}
        self._G_accum = {name: torch.zeros_like(g) for name, g in self.G.items()}
        self._cl_steps = 0

    def update_post_fwd_bwd(self) -> float:
        with torch.no_grad():
            self._accumulate_factors()
            return self._apply_penalty()

    def update_post_optimizer_call(self) -> None:
        self._cl_steps += 1

    def cl_postprocessing(self) -> None:
        """Fold the application's factors into the prior and re-anchor.

        A step taken without a preceding ``cl_preprocessing`` commits nothing.
        """
        if self._A_accum is not None and self._cl_steps > 0:
            with torch.no_grad():
                for name in self.A:
                    self.A[name] = (
                        self.kfac_ema_decay * self.A[name]
                        + self._A_accum[name] / self._cl_steps
                    )
                    self.G[name] = (
                        self.kfac_ema_decay * self.G[name]
                        + self._G_accum[name] / self._cl_steps
                    )
                for name, module in self.layers.items():
                    self.theta_star[name] = module.weight.detach().clone()
        self._A_accum = None
        self._G_accum = None
        self._cl_steps = 0

    # ------------------------------------------------------------------
    def _accumulate_factors(self) -> None:
        """The two contractions, deliberately after the measured pass."""
        if self._A_accum is None or self._G_accum is None:
            return
        for name, module in self.layers.items():
            a = self._a_cache.get(name)
            g = self._g_cache.get(name)
            if a is None or g is None:
                continue
            a_mat = self._extract_a(module, a)
            g_mat = self._extract_g(module, g)
            self._A_accum[name] += a_mat.T @ a_mat / a_mat.shape[0]
            self._G_accum[name] += g_mat.T @ g_mat / g_mat.shape[0]

    def _apply_penalty(self) -> float:
        penalty = 0.0
        for name, module in self.layers.items():
            weight = module.weight
            diff = (weight.detach() - self.theta_star[name]).reshape(
                weight.shape[0], -1
            )
            structured = self.G[name] @ diff @ self.A[name]
            penalty += float((diff * structured).sum().item())
            if weight.grad is None:
                weight.grad = torch.zeros_like(weight)
            weight.grad += self.kfac_lambda * structured.view_as(weight)
        return 0.5 * self.kfac_lambda * penalty
