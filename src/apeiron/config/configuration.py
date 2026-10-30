"""Immutable configuration sections and layered configuration resolution."""
from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
import json
import os
from pathlib import Path
import subprocess
import tomllib


@dataclass(frozen=True)
class ModelCfg:
    name: str
    pretrained_path: str = ""
    max_ckpts: int = 0
    ckpts_path: str = ""


@dataclass(frozen=True)
class DataCfg:
    name: str
    path: str
    batch_size: int = 1


@dataclass(frozen=True)
class TrainCfg:
    batch_size: int
    num_workers: int
    init_lr: float
    grad_accumulation_steps: int = 1
    max_iter: int = 600

    def __post_init__(self):
        # The loss is scaled by this count; at 0 the division is by zero and,
        # being a tensor division, raises nothing -- the loss becomes inf,
        # backward() carries it into the optimizer, and every weight is NaN
        # while the run keeps going (claims/LOG.md A59). Refuse at the source.
        if self.grad_accumulation_steps < 1:
            raise ValueError(
                f"train.grad_accumulation_steps must be >= 1, got "
                f"{self.grad_accumulation_steps}: the loss is divided by this "
                f"count, and 0 silently turns every weight NaN"
            )


@dataclass(frozen=True)
class ContinualLearningCfg:
    update_mode: str = "base"
    mix_historic_data: bool = False
    jvp_rho_theta: float = 0.05
    jvp_rho_x: float = 1.0
    jvp_data_sign: float = 1.0
    ewc_lambda: float = 1000.0
    ewc_ema_decay: float = 0.95
    kfac_lambda: float = 0.01
    kfac_ema_decay: float = 0.95


@dataclass(frozen=True)
class DriftDetectionCfg:
    detector_name: str = "ADWINDetector"
    detection_interval: int = 10
    aggregation: str = "mean"
    metric_index: int = 0
    reset_after_learning: bool = False
    max_stream_updates: int = 20
    adwin_delta: float = 0.002
    adwin_minor_threshold: float = 0.3
    adwin_moderate_threshold: float = 0.6
    kswin_alpha: float = 0.005
    kswin_window_size: int = 100
    kswin_stat_size: int = 30
    kswin_seed: int | None = None
    ph_min_instances: int = 30
    ph_delta: float = 0.005
    ph_threshold: float = 50
    ph_alpha: float = 0.9999
    ensemble_detectors: tuple[str, ...] = ()
    ensemble_voting: str = "majority"

    def __post_init__(self):
        names = self.ensemble_detectors
        object.__setattr__(self, "ensemble_detectors",
                           (names,) if isinstance(names, str) else tuple(names))


@dataclass(frozen=True)
class LoggingCfg:
    backend: str = "none"
    experiment_name: str | None = None
    mlflow_tracking_uri: str | None = None
    metrics_output_path: str | None = None


@dataclass(frozen=True)
class Config:
    model: ModelCfg
    data: DataCfg
    train: TrainCfg
    continual_learning: ContinualLearningCfg
    drift_detection: DriftDetectionCfg
    seed: int
    device: str
    multi_gpu: bool = False
    verbosity: str = "INFO"
    logging: LoggingCfg | None = None


def deep_update(x, y):
    for key, value in y.items():
        if isinstance(value, Mapping) and isinstance(x.get(key), Mapping):
            x[key] = deep_update(dict(x[key]), value)
        else:
            x[key] = value
    return x


def kv_to_nested(items):
    result = {}
    for item in items:
        key, value = item.split("=", 1)
        value = value.strip()
        try:
            value = json.loads(value)
        except ValueError:
            pass
        parts = key.split(".")
        current = result
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value
    return result


def env_overrides(prefix="APP_"):
    return kv_to_nested([f"{key[len(prefix):].lower()}={value}"
                         for key, value in os.environ.items()
                         if key.startswith(prefix)])


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--set", action="append", default=[])
    parser.add_argument("--device")
    parser.add_argument("--multi-gpu", action="store_true", default=None)
    return parser.parse_args(argv)


def load_toml(path):
    with open(path, "rb") as source:
        return tomllib.load(source)


def _select_best_gpu():
    try:
        output = subprocess.check_output([
            "nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"
        ])
        free_memory = [int(line.strip()) for line in output.splitlines()]
        return max(range(len(free_memory)), key=free_memory.__getitem__)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None


def get_available_device(multi_gpu=False):
    # Torch supplies the device object required by callers; all configuration
    # parsing itself uses only the standard library.
    import torch

    if not multi_gpu and "CUDA_VISIBLE_DEVICES" not in os.environ:
        best = _select_best_gpu()
        if best is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(best)
    if torch.cuda.is_available():
        return torch.device("cuda" if multi_gpu else "cuda:0")
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
    except Exception:
        pass
    return torch.device("cpu")


def build_config(argv=None):
    args = parse_args(argv)
    values = load_toml(args.config)
    deep_update(values, env_overrides())
    deep_update(values, kv_to_nested(args.set))
    if args.device is not None:
        values["device"] = args.device
    if args.multi_gpu is not None:
        values["multi_gpu"] = args.multi_gpu
    for key, cls in (("model", ModelCfg), ("data", DataCfg), ("train", TrainCfg),
                     ("continual_learning", ContinualLearningCfg),
                     ("drift_detection", DriftDetectionCfg)):
        values[key] = cls(**values.get(key, {}))
    if values.get("logging") is not None:
        values["logging"] = LoggingCfg(**values["logging"])
    if values.get("device", "auto") == "auto":
        values["device"] = str(get_available_device(values.get("multi_gpu", False)))
    return Config(**values)
