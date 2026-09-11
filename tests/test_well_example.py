"""Offline unit tests for the Well example (no network, no downloads)."""

import pytest
import torch

from examples.well.model import vrmse, well_collate


def _sample(t_in: int = 4, t_out: int = 1, h: int = 8, w: int = 12, c: int = 4):
    return {
        "input_fields": torch.randn(t_in, h, w, c),
        "output_fields": torch.randn(t_out, h, w, c),
    }


class TestWellCollate:
    def test_shapes_channels_first(self):
        batch = well_collate([_sample() for _ in range(3)])
        x, y = batch
        assert x.shape == (3, 16, 8, 12)  # (b, t_in*c, h, w)
        assert y.shape == (3, 4, 8, 12)  # (b, t_out*c, h, w)

    def test_constant_fields_appended(self):
        samples = [_sample() for _ in range(2)]
        for s in samples:
            s["constant_fields"] = torch.randn(8, 12, 2)
        x, _ = well_collate(samples)
        assert x.shape == (2, 18, 8, 12)  # 16 + 2 constant channels

    def test_nan_scrubbed(self):
        s = _sample()
        s["input_fields"][0, 0, 0, 0] = float("nan")
        x, y = well_collate([s])
        assert torch.isfinite(x).all() and torch.isfinite(y).all()

    def test_time_stacking_order(self):
        # channel block t of x must equal input_fields[t] moved channels-first
        s = _sample()
        x, _ = well_collate([s])
        expected = s["input_fields"][2].permute(2, 0, 1)  # t=2 -> (c, h, w)
        assert torch.equal(x[0, 2 * 4 : 3 * 4], expected)


class TestVRMSE:
    def test_perfect_prediction_is_zero(self):
        y = torch.randn(2, 4, 8, 12)
        assert vrmse(y.clone(), y).item() == pytest.approx(0.0, abs=1e-6)

    def test_mean_prediction_is_one(self):
        # Predicting the target's spatial mean gives MSE == Var -> VRMSE ~= 1
        y = torch.randn(2, 4, 8, 12)
        y_hat = (
            y.flatten(2).mean(-1, keepdim=True).expand_as(y.flatten(2)).reshape_as(y)
        )
        assert vrmse(y_hat, y).item() == pytest.approx(1.0, rel=0.02)

    def test_scale_invariance(self):
        y = torch.randn(2, 4, 8, 12)
        y_hat = y + 0.1 * torch.randn_like(y)
        a, b = vrmse(y_hat, y), vrmse(10 * y_hat, 10 * y)
        assert a.item() == pytest.approx(b.item(), rel=1e-4)


def test_factory_dispatch_prefix():
    """get_example routes well:* names to the Well harness (import-level check)."""
    import inspect

    from examples import utils

    src = inspect.getsource(utils.get_example)
    assert 'startswith("well:")' in src
