"""What model a run used.

The config records a model *name* and a path to weights. That is not enough to
reload a checkpoint later, or to tell whether two runs built the same network:
the architecture lives in the harness source, which the config cannot see.

This module derives what it can from the ``nn.Module`` itself -- class, tensor
shapes, parameter counts -- and asks the harness for the rest through the
optional :meth:`BaseModelHarness.model_config` hook.
"""

from __future__ import annotations

import hashlib
import os
from typing import TYPE_CHECKING, Any

from torch import nn

if TYPE_CHECKING:
    from apeiron.model.torch_model_harness import BaseModelHarness

_WRAPPERS = (nn.DataParallel, nn.parallel.DistributedDataParallel)


def unwrap(model: nn.Module) -> nn.Module:
    """Strip DataParallel/DDP so the recorded class is the real one."""
    while isinstance(model, _WRAPPERS):
        model = model.module
    return model


def tensor_table(model: nn.Module) -> list[dict[str, Any]]:
    """Name, shape and dtype of every tensor in the state dict.

    This is exactly what ``load_state_dict`` checks, so it is the table that
    explains why a checkpoint does or does not fit a model.
    """
    return [
        {"name": name, "shape": list(t.shape), "dtype": str(t.dtype)}
        for name, t in model.state_dict().items()
    ]


def shapes_hash(table: list[dict[str, Any]]) -> str:
    """One string per architecture. Equal hashes load each other's weights."""
    h = hashlib.sha256()
    for entry in table:
        h.update(f"{entry['name']}|{entry['shape']}|{entry['dtype']}\n".encode())
    return h.hexdigest()


def _pretrained(path: str) -> dict[str, Any] | None:
    """Identity of the pretrained weights file, as far as it is free to get."""
    if not path:
        return None
    info: dict[str, Any] = {"path": path, "exists": os.path.exists(path)}
    if info["exists"]:
        st = os.stat(path)
        info["size"] = st.st_size
        info["mtime"] = st.st_mtime
    return info


def describe_model(harness: BaseModelHarness) -> dict[str, Any]:
    """Everything recorded about the model. ``tensors`` is the bulky part."""
    model = unwrap(harness.model)
    table = tensor_table(model)
    params = list(model.parameters())

    return {
        "name": harness.cfg.model.name,
        "class": type(model).__name__,
        "module": type(model).__module__,
        "harness_class": type(harness).__name__,
        "harness_module": type(harness).__module__,
        "pretrained": _pretrained(harness.cfg.model.pretrained_path),
        "parameters": {
            "total": sum(p.numel() for p in params),
            "trainable": sum(p.numel() for p in params if p.requires_grad),
        },
        "shapes_sha256": shapes_hash(table),
        "config": dict(harness.model_config()),
        "tensors": table,
    }


def summarize(description: dict[str, Any]) -> dict[str, Any]:
    """The description as it goes into the event log.

    Drops the per-tensor table, which is bulky and already in ``model.json``,
    and the weights file's mtime, which moves when a file is re-copied without
    its contents changing. Path and size remain: they are the only identity we
    have for the weights, and pointing at different weights should compare as a
    different run.
    """
    out = {k: v for k, v in description.items() if k != "tensors"}
    pretrained = out.get("pretrained")
    if isinstance(pretrained, dict):
        out["pretrained"] = {k: v for k, v in pretrained.items() if k != "mtime"}
    return out
