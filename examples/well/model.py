# examples/well/model.py
"""Apeiron harness for PolymathicAI's "The Well" physics simulation datasets.

Streams a Well dataset as a sequence of regime windows -- one window per HDF5
file, ordered lexicographically by filename. Well datasets encode a physical
parameter sweep in their filenames (e.g. ``turbulent_radiative_layer_tcool_0.03``
... ``tcool_3.16``), so the window sequence walks the model across the
parameter continuum and drift corresponds to real regime change.

Data flows through ``the_well.data.WellDataset`` (local download or Hugging
Face streaming via an ``hf://`` base path). The model is The Well's FNO
baseline, loaded from a pretrained Hugging Face checkpoint when
``model.pretrained_path`` is a HF repo id (e.g.
``polymathic-ai/FNO-turbulent_radiative_layer_2D``).

Batch convention: samples are collated to plain ``[x, y]`` tensor pairs in
channels-first layout (``x: [B, T_in*C(+const), H, W]``, ``y: [B, T_out*C, H, W]``)
so the rest of apeiron (monitor, trainer, updaters) sees ordinary tuples.
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
from einops import rearrange
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader, Dataset

from apeiron.config.configuration import Config
from apeiron.model.torch_model_harness import BaseModelHarness

# The pretrained Well FNO baselines consume 4 input steps and predict 1 step
# (checkpoint config: dim_in = 4 steps x n_fields, dim_out = n_fields).
N_STEPS_INPUT = 4
N_STEPS_OUTPUT = 1

_EPS = 1e-7


def well_collate(samples: List[Dict[str, Tensor]]) -> List[Tensor]:
    """Collate WellDataset dict samples into a channels-first ``[x, y]`` pair.

    ``input_fields``/``output_fields`` arrive as ``[T, *spatial, C]``; time is
    stacked into channels (matching The Well's ``DefaultChannelsFirstFormatter``)
    and any constant fields are appended as extra input channels. NaNs are
    zeroed, mirroring the reference formatter.
    """
    x = torch.stack([s["input_fields"] for s in samples])
    y = torch.stack([s["output_fields"] for s in samples])
    x = rearrange(x, "b t ... c -> b (t c) ...")
    y = rearrange(y, "b t ... c -> b (t c) ...")
    if "constant_fields" in samples[0]:
        const = torch.stack([s["constant_fields"] for s in samples])
        x = torch.cat([x, rearrange(const, "b ... c -> b c ...")], dim=1)
    return [torch.nan_to_num(x), torch.nan_to_num(y)]


def vrmse(y_hat: Tensor, y: Tensor) -> Tensor:
    """Variance-scaled RMSE (The Well's headline metric; lower is better).

    Per (batch, channel): spatial MSE normalized by the target's spatial
    variance, averaged, then rooted -- matching ``the_well``'s VRMSE
    (NMSE with ``norm_mode="std"``) up to metadata bookkeeping.
    """
    d = (y_hat - y).flatten(2)
    mse = d.pow(2).mean(-1)
    var = y.flatten(2).var(-1)
    return torch.sqrt((mse / (var + _EPS)).mean())


class WELL_FNO(BaseModelHarness):
    """
    Pattern (mirrors MNIST_CNN):
      - One regime window per HDF5 file of the chosen Well dataset, ordered
        by filename (= ordered by the physical parameter sweep).
      - update_data_stream(): advance to the next regime; build train/val/stream
        loaders restricted to that file via WellDataset include_filters.
      - get_hist_dataloaders(): loaders over ALL prior regimes (replay anchor),
        or (None, None) on the first window.
      - Config: data.name = "well:<well_dataset_name>", data.path = well base
        path (local download dir or "hf://datasets/polymathic-ai/").
    """

    def __init__(self, cfg: Config, model: Optional[nn.Module] = None):
        if model is None:
            model = self._load_model(cfg)
        super().__init__(cfg=cfg, model=model)

        self.eval_metrics = {"vrmse": vrmse, "loss": self._mse_metric}
        self.higher_is_better = {"vrmse": False, "loss": False}

        self.dataset_name = cfg.data.name.split(":", 1)[1]
        self.base_path = cfg.data.path

        self.regimes = self._discover_regimes()
        print(f"Well regimes ({len(self.regimes)}): {self.regimes}")

        self.task_counter = 0
        self._cur_train_loader: Optional[DataLoader] = None
        self._cur_val_loader: Optional[DataLoader] = None
        self._cur_stream_loader: Optional[DataLoader] = None

    # ----- model -----

    @staticmethod
    def _load_model(cfg: Config) -> nn.Module:
        from the_well.benchmark.models import FNO

        pretrained = cfg.model.pretrained_path
        try:
            model = FNO.from_pretrained(pretrained)
            print(f"Loaded pretrained Well FNO from {pretrained}")
            return model
        except Exception as e:  # noqa: BLE001 - fall back to random init
            print(f"Warning: could not load pretrained FNO ({e}); random init")
            return FNO(
                dim_in=16,
                dim_out=4,
                n_spatial_dims=2,
                spatial_resolution=(128, 384),
                modes1=16,
                modes2=16,
            )

    def get_optmizer(self) -> Optimizer:
        return torch.optim.Adam(self.model.parameters(), lr=self.cfg.train.init_lr)

    def get_criterion(self):
        return nn.MSELoss()

    @staticmethod
    @torch.no_grad()
    def _mse_metric(y_hat: Tensor, y: Tensor) -> Tensor:
        return nn.functional.mse_loss(y_hat, y)

    # ----- regimes -----

    def _discover_regimes(self) -> List[str]:
        """Sorted basenames of the train split's HDF5 files; one regime each."""
        import fsspec

        data_path = f"{self.base_path.rstrip('/')}/{self.dataset_name}/data/train"
        fs, _ = fsspec.url_to_fs(data_path)
        files = fs.glob(data_path + "/*.h5") + fs.glob(data_path + "/*.hdf5")
        if not files:
            raise FileNotFoundError(f"No Well HDF5 files under {data_path}")
        names = sorted(f.rsplit("/", 1)[-1].rsplit(".", 1)[0] for f in files)
        return names

    def _make_dataset(self, split: str, include: List[str]) -> Dataset:
        from the_well.data.datasets import WellDataset
        from the_well.data.normalization import ZScoreNormalization

        return WellDataset(
            well_base_path=self.base_path,
            well_dataset_name=self.dataset_name,
            well_split_name=split,
            include_filters=include,
            use_normalization=True,
            normalization_type=ZScoreNormalization,
            n_steps_input=N_STEPS_INPUT,
            n_steps_output=N_STEPS_OUTPUT,
        )

    def _make_loader(self, ds: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=self.cfg.train.num_workers,
            collate_fn=well_collate,
            drop_last=False,
        )

    # ----- stream protocol -----

    def update_data_stream(self) -> None:
        regime_idx = min(self.task_counter, len(self.regimes) - 1)
        regime = self.regimes[regime_idx]
        print(f"Well stream -> regime {regime_idx}: {regime}")

        ds_train = self._make_dataset("train", include=[regime])
        ds_val = self._make_dataset("valid", include=[regime])

        self._cur_train_loader = self._make_loader(
            ds_train, self.cfg.train.batch_size, shuffle=True
        )
        self._cur_val_loader = self._make_loader(
            ds_val, self.cfg.train.batch_size, shuffle=False
        )
        self._cur_stream_loader = self._make_loader(
            ds_val, self.cfg.data.batch_size, shuffle=True
        )
        self.task_counter += 1

    def get_stream_dataloader(self) -> DataLoader:
        assert self._cur_stream_loader is not None, "call update_data_stream() first"
        return self._cur_stream_loader

    def get_train_dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        assert self._cur_train_loader is not None and self._cur_val_loader is not None
        return self._cur_train_loader, self._cur_val_loader

    def get_hist_dataloaders(
        self,
    ) -> Tuple[Optional[DataLoader], Optional[DataLoader]]:
        if self.task_counter <= 1:
            return None, None
        prior = [
            self.regimes[i]
            for i in range(min(self.task_counter - 1, len(self.regimes)))
        ]
        hist_train = self._make_dataset("train", include=prior)
        hist_val = self._make_dataset("valid", include=prior)
        return (
            self._make_loader(hist_train, self.cfg.train.batch_size, shuffle=True),
            self._make_loader(hist_val, self.cfg.train.batch_size, shuffle=False),
        )

    # ----- misc -----

    def _unpack(self, batch: Any) -> Tuple[Tensor, Tensor]:
        x, y = batch
        return x, y
