"""A small Fourier Neural Operator, enough to learn a Well regime.

The operator learns in frequency space: each block takes the spatial Fourier
transform, keeps the lowest modes, mixes channels there with a learned complex
weight, and transforms back. Because the parameters live on modes rather than
pixels, the same weights apply at any resolution -- which is what makes an
operator rather than a plain convolutional net.

Deliberately small and self-contained. The Well publishes pretrained FNO
baselines that are larger and better; swapping one in is a change to
``WELL_FNO._build_model`` alone, and the run's ``model.json`` records which was
used, so two runs can never be quietly confused.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class SpectralConv2d(nn.Module):
    """Channel mixing on the lowest Fourier modes of a 2D field."""

    def __init__(self, in_channels: int, out_channels: int, modes_y: int, modes_x: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes_y = modes_y
        self.modes_x = modes_x

        scale = 1.0 / (in_channels * out_channels)
        shape = (in_channels, out_channels, modes_y, modes_x)
        # Two blocks: the lowest positive and lowest negative y frequencies.
        # rfft2 keeps only non-negative x frequencies, so x needs one block.
        self.low = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))
        self.high = nn.Parameter(scale * torch.randn(*shape, dtype=torch.cfloat))

    def forward(self, x: Tensor) -> Tensor:
        batch, _, height, width = x.shape
        spectrum = torch.fft.rfft2(x)

        my = min(self.modes_y, height // 2)
        mx = min(self.modes_x, width // 2 + 1)

        out = torch.zeros(
            batch,
            self.out_channels,
            height,
            width // 2 + 1,
            dtype=torch.cfloat,
            device=x.device,
        )
        out[:, :, :my, :mx] = torch.einsum(
            "bixy,ioxy->boxy", spectrum[:, :, :my, :mx], self.low[:, :, :my, :mx]
        )
        out[:, :, -my:, :mx] = torch.einsum(
            "bixy,ioxy->boxy", spectrum[:, :, -my:, :mx], self.high[:, :, :my, :mx]
        )
        return torch.fft.irfft2(out, s=(height, width))


class FNO2d(nn.Module):
    """Lift to a wider channel space, mix in frequency, project back."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        width: int = 16,
        modes: int = 8,
        depth: int = 2,
    ):
        super().__init__()
        self.lift = nn.Conv2d(in_channels, width, kernel_size=1)
        self.spectral = nn.ModuleList(
            SpectralConv2d(width, width, modes, modes) for _ in range(depth)
        )
        # The pointwise path carries what the truncated modes drop.
        self.pointwise = nn.ModuleList(
            nn.Conv2d(width, width, kernel_size=1) for _ in range(depth)
        )
        self.project = nn.Sequential(
            nn.Conv2d(width, 2 * width, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(2 * width, out_channels, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.lift(x)
        for spectral, pointwise in zip(self.spectral, self.pointwise):
            x = F.gelu(spectral(x) + pointwise(x))
        return self.project(x)
