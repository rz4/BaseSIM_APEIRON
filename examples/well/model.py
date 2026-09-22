"""Apeiron harness for PolymathicAI's "The Well" physics simulation datasets.

One window per regime. A Well dataset is a parameter sweep -- the turbulent
radiative layer varies a cooling time from ``tcool=0.03`` to ``tcool=3.16``
across nine files -- so walking the windows in parameter order walks the model
through real physical regime change. Drift here is not simulated by perturbing
inputs; it is what happens when the physics moves.

The run declares what each window needs and the framework puts it there, so a
run that touches three regimes moves three regimes' worth of bytes rather than
the dataset. Each regime's HDF5 is converted once into a memory-mappable array
and read back as a view, which is what makes windows larger than memory work.

Config::

    [data]
    name = "well:turbulent_radiative_layer_2D"
    path = "hf://datasets/polymathic-ai"   # or a directory holding the dataset
    memmap = true

Requires an ``[experiment]`` section: the data comes through the dataset store.
"""

from __future__ import annotations

import json
import urllib.request
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import torch
import yaml
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from apeiron.config.configuration import Config
from apeiron.experiment import Derived, memmap
from apeiron.model.torch_model_harness import BaseModelHarness

from examples.well.fno import FNO2d

# The Well's 2D fields, in the channel order used throughout this harness.
FIELDS: Tuple[Tuple[str, str, Optional[int]], ...] = (
    ("t0_fields", "density", None),
    ("t0_fields", "pressure", None),
    ("t1_fields", "velocity", 0),
    ("t1_fields", "velocity", 1),
)
N_CHANNELS = len(FIELDS)

# Steps of history in, steps predicted out.
N_STEPS_INPUT = 4
N_STEPS_OUTPUT = 1

# One place, so model_config() and _build_model() cannot drift apart.
FNO_SHAPE: Dict[str, int] = {"width": 16, "modes": 8, "depth": 2}

# Bump when the conversion below changes in any way that alters the bytes;
# files carrying an older recipe are rebuilt rather than trusted.
CONVERSION_RECIPE = "well-hdf5->f32[traj,time,c,y,x]:v1"

_EPS = 1e-7


def vrmse(y_hat: Tensor, y: Tensor) -> Tensor:
    """Variance-scaled RMSE, The Well's headline metric. Lower is better.

    Per sample and channel, the spatial error is divided by the target's own
    spatial variance, so fields of wildly different magnitude -- density in the
    tens, pressure near one -- contribute comparably.
    """
    error = (y_hat - y).flatten(2).pow(2).mean(-1)
    variance = y.flatten(2).var(-1)
    return torch.sqrt((error / (variance + _EPS)).mean())


def mse(y_hat: Tensor, y: Tensor) -> Tensor:
    return torch.nn.functional.mse_loss(y_hat, y)


def regime_parameter(name: str) -> float:
    """The swept physical parameter encoded in a Well filename.

    ``turbulent_radiative_layer_tcool_0.06`` -> ``0.06``. Sorting on this
    rather than on the string keeps the sweep in physical order for datasets
    that do not zero-pad.
    """
    try:
        return float(name.rsplit("_", 1)[-1])
    except ValueError:
        return float("inf")


def convert_to_array(source: Path, dest: Path) -> None:
    """Rewrite one regime's HDF5 as a mappable array.

    Shape ``[trajectory, time, channel, y, x]``, float32. Written a trajectory
    at a time, so a file larger than memory converts without being held.
    """
    with h5py.File(source, "r") as handle:
        first = handle[f"{FIELDS[0][0]}/{FIELDS[0][1]}"]
        n_traj, n_time = first.shape[0], first.shape[1]
        spatial = tuple(first.shape[2:])

        shape = (n_traj, n_time, N_CHANNELS, *spatial)
        with memmap.writing(dest, shape, "float32") as out:
            for trajectory in range(n_traj):
                for channel, (group, field, component) in enumerate(FIELDS):
                    block = handle[f"{group}/{field}"][trajectory]
                    if component is not None:
                        block = block[..., component]
                    out[trajectory, :, channel] = block


class RegimeFrames:
    """``[trajectory, time, channel, y, x]`` for one regime.

    Backed by a mapped array when the conversion is enabled, or read straight
    out of HDF5 when it is not. Either way nothing is held in memory that a
    sample does not ask for.
    """

    def __init__(self, path: Path, mapped: bool):
        self.path = path
        self.mapped = mapped
        if mapped:
            self._array = memmap.read(path)
            self.shape = tuple(self._array.shape)
        else:
            self._file = h5py.File(path, "r")
            first = self._file[f"{FIELDS[0][0]}/{FIELDS[0][1]}"]
            self.shape = (
                first.shape[0],
                first.shape[1],
                N_CHANNELS,
                *first.shape[2:],
            )

    def steps(self, trajectory: int, start: int, count: int) -> np.ndarray:
        """``[count, channel, y, x]`` starting at a time index."""
        if self.mapped:
            return np.asarray(self._array[trajectory, start : start + count])

        stop = start + count
        out = np.empty((count, N_CHANNELS, *self.shape[3:]), dtype="float32")
        for channel, (group, field, component) in enumerate(FIELDS):
            block = self._file[f"{group}/{field}"][trajectory, start:stop]
            out[:, channel] = block[..., component] if component is not None else block
        return out


class RegimeDataset(Dataset):
    """Sliding windows of time within one regime.

    A sample is ``N_STEPS_INPUT`` consecutive frames stacked into channels,
    predicting the next ``N_STEPS_OUTPUT``.
    """

    def __init__(self, frames: RegimeFrames, mean: np.ndarray, std: np.ndarray):
        self.frames = frames
        self.mean = mean.reshape(1, -1, 1, 1)
        self.std = std.reshape(1, -1, 1, 1)
        n_traj, n_time = frames.shape[0], frames.shape[1]
        self.per_trajectory = n_time - N_STEPS_INPUT - N_STEPS_OUTPUT + 1
        if self.per_trajectory <= 0:
            raise ValueError(f"{frames.path}: too few time steps")
        self.length = n_traj * self.per_trajectory

    def __len__(self) -> int:
        return self.length

    def _normalised(self, trajectory: int, start: int, count: int) -> Tensor:
        block = self.frames.steps(trajectory, start, count)
        block = (block - self.mean) / self.std
        # Time folds into channels: [t, c, y, x] -> [t*c, y, x].
        flat = block.reshape(-1, *block.shape[2:])
        return torch.from_numpy(np.nan_to_num(flat, copy=True))

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        trajectory, start = divmod(index, self.per_trajectory)
        x = self._normalised(trajectory, start, N_STEPS_INPUT)
        y = self._normalised(trajectory, start + N_STEPS_INPUT, N_STEPS_OUTPUT)
        return x, y


class WELL_FNO(BaseModelHarness):
    """One Well regime per window, an FNO stepping the fields forward."""

    def __init__(self, cfg: Config, model: Optional[nn.Module] = None):
        self.dataset_name = cfg.data.name.split(":", 1)[1]
        self.base_path = cfg.data.path.rstrip("/")
        self.use_memmap = cfg.data.memmap

        super().__init__(cfg=cfg, model=model or self._build_model(cfg))

        if self.datasets is None:
            raise RuntimeError(
                "the Well example needs an [experiment] section: its data "
                "comes through the dataset store"
            )

        self.eval_metrics = {"vrmse": vrmse, "mse": mse}
        self.regimes = self._discover_regimes()
        self.window = -1
        self._stats: Optional[Tuple[np.ndarray, np.ndarray]] = None
        self._loaders: Dict[str, DataLoader] = {}

    # -- model --------------------------------------------------------------
    @staticmethod
    def _build_model(cfg: Config) -> nn.Module:
        model = FNO2d(
            in_channels=N_STEPS_INPUT * N_CHANNELS,
            out_channels=N_STEPS_OUTPUT * N_CHANNELS,
            **FNO_SHAPE,
        )
        if cfg.model.pretrained_path:
            state = torch.load(cfg.model.pretrained_path, map_location="cpu")
            model.load_state_dict(state)
        return model

    def model_config(self) -> Dict[str, Any]:
        """Recorded with the run, since the architecture is not in the config."""
        return {
            "architecture": "FNO2d",
            "in_channels": N_STEPS_INPUT * N_CHANNELS,
            "out_channels": N_STEPS_OUTPUT * N_CHANNELS,
            "n_steps_input": N_STEPS_INPUT,
            "n_steps_output": N_STEPS_OUTPUT,
            "fields": [f"{f}{'' if c is None else f'[{c}]'}" for _, f, c in FIELDS],
            **FNO_SHAPE,
        }

    def get_optmizer(self) -> Optimizer:
        return torch.optim.Adam(self.model.parameters(), lr=self.cfg.train.init_lr)

    def get_criterion(self):
        return nn.MSELoss()

    # -- regimes ------------------------------------------------------------
    def _discover_regimes(self) -> List[str]:
        """Regime names in physical order, without downloading anything."""
        listing = f"{self.base_path}/{self.dataset_name}/data/train"
        if self.base_path.startswith("hf://"):
            owner = self.base_path[len("hf://datasets/") :]
            url = (
                f"https://huggingface.co/api/datasets/{owner}/"
                f"{self.dataset_name}/tree/main/data/train"
            )
            with urllib.request.urlopen(url, timeout=60) as response:
                entries = [item["path"] for item in json.load(response)]
        else:
            entries = [str(p) for p in Path(listing).glob("*.hdf5")]

        names = sorted(
            (Path(e).stem for e in entries if e.endswith(".hdf5")),
            key=regime_parameter,
        )
        if not names:
            raise FileNotFoundError(f"no Well HDF5 files under {listing}")
        return names

    def _regime(self, window: int) -> str:
        return self.regimes[min(window, len(self.regimes) - 1)]

    # -- what a window needs ------------------------------------------------
    def window_inputs(self, window: int) -> Dict[str, Any]:
        regime = self._regime(window)
        root = f"{self.base_path}/{self.dataset_name}"
        declared: Dict[str, Any] = {"stats.yaml": f"{root}/stats.yaml"}

        for split in ("train", "valid"):
            source = f"{split}/{regime}.hdf5"
            declared[source] = f"{root}/data/{split}/{regime}.hdf5"
            if self.use_memmap:
                declared[f"{split}/{regime}.npy"] = Derived(
                    build=partial(self._convert, source),
                    recipe=CONVERSION_RECIPE,
                )
        return declared

    def _convert(self, source: str, dest: Path) -> None:
        assert self.datasets is not None
        convert_to_array(self.datasets.path_for(source), dest)

    def _frames(self, split: str, regime: str) -> RegimeFrames:
        assert self.datasets is not None
        suffix = "npy" if self.use_memmap else "hdf5"
        return RegimeFrames(
            self.datasets.path_for(f"{split}/{regime}.{suffix}"), self.use_memmap
        )

    def _normalisation(self) -> Tuple[np.ndarray, np.ndarray]:
        """Dataset-wide per-channel statistics, from the Well's own stats file.

        Fixed across regimes on purpose: normalising per regime would divide
        out the very shift the run is watching for.
        """
        if self._stats is None:
            assert self.datasets is not None
            stats = yaml.safe_load(self.datasets.path_for("stats.yaml").read_text())
            mean: List[float] = []
            std: List[float] = []
            for _, field, component in FIELDS:
                for key, target in (("mean", mean), ("std", std)):
                    value = stats[key][field]
                    target.append(
                        float(value if component is None else value[component])
                    )
            self._stats = (
                np.asarray(mean, dtype="float32"),
                np.asarray(std, dtype="float32"),
            )
        return self._stats

    def _dataset(self, split: str, regime: str) -> RegimeDataset:
        mean, std = self._normalisation()
        return RegimeDataset(self._frames(split, regime), mean, std)

    def _loader(self, dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.train.num_workers,
            drop_last=False,
        )

    # -- stream protocol ----------------------------------------------------
    def update_data_stream(self) -> None:
        self.window += 1
        regime = self._regime(self.window)

        train = self._dataset("train", regime)
        valid = self._dataset("valid", regime)
        self._loaders = {
            "stream": self._loader(valid, self.cfg.data.batch_size, shuffle=True),
            "train": self._loader(train, self.cfg.train.batch_size, shuffle=True),
            "valid": self._loader(valid, self.cfg.train.batch_size, shuffle=False),
        }

    def get_stream_dataloader(self) -> DataLoader:
        return self._loaders["stream"]

    def get_train_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        return self._loaders["train"], self._loaders["valid"]

    def get_hist_dataloaders(
        self,
    ) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
        """Every regime seen before this one -- the replay anchor."""
        prior = [self._regime(w) for w in range(self.window)]
        if not prior:
            return None, None
        train: Dataset = ConcatDataset([self._dataset("train", r) for r in prior])
        valid: Dataset = ConcatDataset([self._dataset("valid", r) for r in prior])
        return (
            self._loader(train, self.cfg.train.batch_size, shuffle=True),
            self._loader(valid, self.cfg.train.batch_size, shuffle=False),
        )
