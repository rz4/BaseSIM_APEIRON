"""Tests for the Well example (examples/well).

Uses a miniature file with the real dataset's structure, so nothing here
downloads or depends on the 650 MB regime files.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
import yaml

from apeiron.config.configuration import (
    Config,
    ContinualLearningCfg,
    DataCfg,
    DriftDetectionCfg,
    ExperimentCfg,
    ModelCfg,
    TrainCfg,
)
from examples.well.fno import FNO2d
from examples.well.model import (
    FIELDS,
    N_CHANNELS,
    N_STEPS_INPUT,
    N_STEPS_OUTPUT,
    WELL_FNO,
    RegimeDataset,
    RegimeFrames,
    convert_to_array,
    regime_parameter,
    vrmse,
)

N_TRAJ, N_TIME, HEIGHT, WIDTH = 2, 9, 8, 12


def write_regime(path: Path, seed: int) -> None:
    """A miniature file shaped like a Well 2D regime."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as handle:
        for group, field, _ in FIELDS:
            key = f"{group}/{field}"
            if key in handle:
                continue
            shape: tuple[int, ...] = (N_TRAJ, N_TIME, HEIGHT, WIDTH)
            if field == "velocity":
                shape = shape + (2,)
            handle.create_dataset(key, data=rng.normal(size=shape).astype("float32"))
        handle.attrs["dataset_name"] = "test_dataset"


def write_dataset(root: Path, name: str, regimes: dict[str, int]) -> Path:
    """A whole miniature Well dataset: regimes plus the stats file."""
    base = root / name
    for regime, seed in regimes.items():
        for split in ("train", "valid"):
            write_regime(base / "data" / split / f"{regime}.hdf5", seed)
    stats = {
        "mean": {"density": 0.0, "pressure": 0.0, "velocity": [0.0, 0.0]},
        "std": {"density": 1.0, "pressure": 1.0, "velocity": [1.0, 1.0]},
    }
    (base / "stats.yaml").write_text(yaml.safe_dump(stats))
    return base


def well_cfg(tmp_path: Path, data_root: Path, memmap: bool = True) -> Config:
    return Config(
        model=ModelCfg(name="fno", max_ckpts=1),
        data=DataCfg(
            name="well:mini", path=str(data_root), batch_size=2, memmap=memmap
        ),
        train=TrainCfg(batch_size=2, num_workers=0, init_lr=1e-3, max_iter=1),
        continual_learning=ContinualLearningCfg(update_mode="base"),
        drift_detection=DriftDetectionCfg(detection_interval=2, max_stream_updates=1),
        experiment=ExperimentCfg(path=str(tmp_path / "out"), name="mini"),
        seed=3,
        device="cpu",
    )


@pytest.fixture()
def mini_dataset(tmp_path) -> Path:
    write_dataset(
        tmp_path / "well", "mini", {"mini_tcool_0.30": 1, "mini_tcool_1.00": 2}
    )
    return tmp_path / "well"


# ---------------------------------------------------------------------------
# metric and ordering
# ---------------------------------------------------------------------------
class TestVrmse:
    def test_zero_for_a_perfect_prediction(self):
        y = torch.randn(3, 4, 8, 8)
        assert float(vrmse(y, y)) == pytest.approx(0.0, abs=1e-5)

    def test_scale_invariant_per_channel(self):
        """A field a thousand times larger must not dominate the score."""
        y = torch.randn(2, 2, 8, 8)
        y_hat = y + 0.1 * torch.randn_like(y)
        scaled = y.clone()
        scaled[:, 0] *= 1000
        scaled_hat = y_hat.clone()
        scaled_hat[:, 0] *= 1000
        assert float(vrmse(y_hat, y)) == pytest.approx(
            float(vrmse(scaled_hat, scaled)), rel=1e-4
        )

    def test_grows_with_error(self):
        y = torch.randn(2, 3, 8, 8)
        small = vrmse(y + 0.01 * torch.randn_like(y), y)
        large = vrmse(y + 1.00 * torch.randn_like(y), y)
        assert float(small) < float(large)


class TestRegimeOrdering:
    def test_sorts_by_the_swept_parameter(self):
        names = ["x_tcool_0.3", "x_tcool_10", "x_tcool_2", "x_tcool_0.06"]
        assert sorted(names, key=regime_parameter) == [
            "x_tcool_0.06",
            "x_tcool_0.3",
            "x_tcool_2",
            "x_tcool_10",
        ]

    def test_unparseable_names_sort_last(self):
        assert regime_parameter("no_number_here") == float("inf")


# ---------------------------------------------------------------------------
# conversion and reading
# ---------------------------------------------------------------------------
class TestConversion:
    def test_shape_and_channel_order(self, tmp_path):
        source = tmp_path / "r.hdf5"
        write_regime(source, seed=5)
        dest = tmp_path / "r.npy"
        convert_to_array(source, dest)

        array = np.load(dest, mmap_mode="r")
        assert array.shape == (N_TRAJ, N_TIME, N_CHANNELS, HEIGHT, WIDTH)

        with h5py.File(source) as handle:
            assert np.allclose(array[1, 3, 0], handle["t0_fields/density"][1, 3])
            assert np.allclose(array[1, 3, 1], handle["t0_fields/pressure"][1, 3])
            assert np.allclose(
                array[1, 3, 2], handle["t1_fields/velocity"][1, 3, ..., 0]
            )
            assert np.allclose(
                array[1, 3, 3], handle["t1_fields/velocity"][1, 3, ..., 1]
            )

    def test_mapped_and_direct_reads_agree(self, tmp_path):
        source = tmp_path / "r.hdf5"
        write_regime(source, seed=7)
        dest = tmp_path / "r.npy"
        convert_to_array(source, dest)

        mapped = RegimeFrames(dest, mapped=True)
        direct = RegimeFrames(source, mapped=False)
        assert mapped.shape == direct.shape
        assert np.allclose(mapped.steps(1, 2, 3), direct.steps(1, 2, 3))

    def test_mapped_frames_are_a_view(self, tmp_path):
        source = tmp_path / "r.hdf5"
        write_regime(source, seed=9)
        dest = tmp_path / "r.npy"
        convert_to_array(source, dest)
        frames = RegimeFrames(dest, mapped=True)
        assert isinstance(frames._array, np.memmap)


class TestRegimeDataset:
    def _frames(self, tmp_path) -> RegimeFrames:
        source = tmp_path / "r.hdf5"
        write_regime(source, seed=11)
        dest = tmp_path / "r.npy"
        convert_to_array(source, dest)
        return RegimeFrames(dest, mapped=True)

    def test_sample_shapes(self, tmp_path):
        frames = self._frames(tmp_path)
        ds = RegimeDataset(
            frames, np.zeros(N_CHANNELS, "float32"), np.ones(N_CHANNELS, "float32")
        )
        x, y = ds[0]
        assert x.shape == (N_STEPS_INPUT * N_CHANNELS, HEIGHT, WIDTH)
        assert y.shape == (N_STEPS_OUTPUT * N_CHANNELS, HEIGHT, WIDTH)

    def test_length_counts_every_sliding_window(self, tmp_path):
        frames = self._frames(tmp_path)
        ds = RegimeDataset(
            frames, np.zeros(N_CHANNELS, "float32"), np.ones(N_CHANNELS, "float32")
        )
        per_trajectory = N_TIME - N_STEPS_INPUT - N_STEPS_OUTPUT + 1
        assert len(ds) == N_TRAJ * per_trajectory

    def test_target_follows_the_inputs_in_time(self, tmp_path):
        frames = self._frames(tmp_path)
        ds = RegimeDataset(
            frames, np.zeros(N_CHANNELS, "float32"), np.ones(N_CHANNELS, "float32")
        )
        x, y = ds[0]
        # last input frame is step 3, target is step 4
        assert np.allclose(x[3 * N_CHANNELS : 4 * N_CHANNELS], frames.steps(0, 3, 1)[0])
        assert np.allclose(y, frames.steps(0, N_STEPS_INPUT, 1)[0])

    def test_normalisation_is_applied(self, tmp_path):
        frames = self._frames(tmp_path)
        mean = np.full(N_CHANNELS, 2.0, "float32")
        std = np.full(N_CHANNELS, 4.0, "float32")
        plain = RegimeDataset(
            frames, np.zeros(N_CHANNELS, "float32"), np.ones(N_CHANNELS, "float32")
        )[0][0]
        scaled = RegimeDataset(frames, mean, std)[0][0]
        assert torch.allclose(scaled, (plain - 2.0) / 4.0, atol=1e-5)

    def test_too_short_a_regime_is_reported(self, tmp_path):
        source = tmp_path / "short.hdf5"
        path = tmp_path / "short.npy"
        write_regime(source, seed=13)
        convert_to_array(source, path)
        frames = RegimeFrames(path, mapped=True)
        frames.shape = (N_TRAJ, 2, N_CHANNELS, HEIGHT, WIDTH)
        with pytest.raises(ValueError, match="too few time steps"):
            RegimeDataset(
                frames, np.zeros(N_CHANNELS, "float32"), np.ones(N_CHANNELS, "float32")
            )


# ---------------------------------------------------------------------------
# the harness
# ---------------------------------------------------------------------------
class TestHarness:
    def test_discovers_regimes_in_order(self, tmp_path, mini_dataset):
        harness = WELL_FNO(well_cfg(tmp_path, mini_dataset))
        assert harness.regimes == ["mini_tcool_0.30", "mini_tcool_1.00"]

    def test_requires_experiment_mode(self, tmp_path, mini_dataset):
        cfg = replace(well_cfg(tmp_path, mini_dataset), experiment=None)
        with pytest.raises(RuntimeError, match=r"\[experiment\]"):
            WELL_FNO(cfg)

    def test_window_declares_sources_and_conversions(self, tmp_path, mini_dataset):
        from apeiron.experiment import Derived

        harness = WELL_FNO(well_cfg(tmp_path, mini_dataset))
        declared = harness.window_inputs(0)
        assert set(declared) == {
            "stats.yaml",
            "train/mini_tcool_0.30.hdf5",
            "valid/mini_tcool_0.30.hdf5",
            "train/mini_tcool_0.30.npy",
            "valid/mini_tcool_0.30.npy",
        }
        assert isinstance(declared["train/mini_tcool_0.30.npy"], Derived)
        assert declared["stats.yaml"].endswith("mini/stats.yaml")

    def test_no_conversions_declared_without_memmap(self, tmp_path, mini_dataset):
        harness = WELL_FNO(well_cfg(tmp_path, mini_dataset, memmap=False))
        assert not any(k.endswith(".npy") for k in harness.window_inputs(0))

    def test_model_config_is_recorded(self, tmp_path, mini_dataset):
        harness = WELL_FNO(well_cfg(tmp_path, mini_dataset))
        recorded = harness.model_config()
        assert recorded["architecture"] == "FNO2d"
        assert recorded["in_channels"] == N_STEPS_INPUT * N_CHANNELS
        assert recorded["fields"] == [
            "density",
            "pressure",
            "velocity[0]",
            "velocity[1]",
        ]

    @pytest.mark.parametrize("memmap", [True, False])
    def test_a_window_end_to_end(self, tmp_path, mini_dataset, memmap):
        harness = WELL_FNO(well_cfg(tmp_path, mini_dataset, memmap=memmap))
        stats = harness.ensure_window_inputs(0)
        assert stats is not None and stats.wanted == (5 if memmap else 3)

        harness.update_data_stream()
        x, y = next(iter(harness.get_stream_dataloader()))
        assert x.shape[1] == N_STEPS_INPUT * N_CHANNELS
        assert y.shape[1] == N_STEPS_OUTPUT * N_CHANNELS

        with torch.no_grad():
            prediction = harness.model(x)
        assert prediction.shape == y.shape
        assert float(vrmse(prediction, y)) > 0

    def test_history_appears_only_after_the_first_window(self, tmp_path, mini_dataset):
        harness = WELL_FNO(well_cfg(tmp_path, mini_dataset))
        harness.ensure_window_inputs(0)
        harness.update_data_stream()
        assert harness.get_hist_dataloaders() == (None, None)

        harness.ensure_window_inputs(1)
        harness.update_data_stream()
        hist_train, hist_valid = harness.get_hist_dataloaders()
        assert hist_train is not None and hist_valid is not None
        assert len(hist_train.dataset) == len(
            harness.get_train_dataloaders()[0].dataset
        )


class TestFno:
    def test_shape_is_preserved(self):
        model = FNO2d(in_channels=16, out_channels=4, width=8, modes=4, depth=1)
        out = model(torch.randn(2, 16, 16, 24))
        assert out.shape == (2, 4, 16, 24)

    def test_resolution_independent(self):
        """Parameters live on modes, so the same weights apply at any size."""
        model = FNO2d(in_channels=4, out_channels=4, width=8, modes=4, depth=1)
        assert model(torch.randn(1, 4, 16, 16)).shape == (1, 4, 16, 16)
        assert model(torch.randn(1, 4, 32, 48)).shape == (1, 4, 32, 48)

    def test_gradients_reach_the_spectral_weights(self):
        model = FNO2d(in_channels=4, out_channels=4, width=8, modes=4, depth=1)
        model(torch.randn(1, 4, 16, 16)).sum().backward()
        assert model.spectral[0].low.grad is not None
        assert torch.any(model.spectral[0].low.grad != 0)
