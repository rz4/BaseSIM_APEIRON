"""Data-source adapters: how remote reservoir objects are listed and fetched.

A *source* fronts one kind of reservoir (hop 2 of the input story) or local
supply (hop 1). It does exactly two things: enumerate the objects under a
base URI as :class:`RemoteObject` records (with sizes and, when the
reservoir declares them, content hashes), and fetch one object to a local
path. Everything about placement -- where objects land, when they are
evicted, what is pinned -- belongs to the residency layer, not here.

Built-in sources:

- ``hf://datasets/<org>/<name>/...`` -- Hugging Face dataset repos
  (sizes + LFS sha256 from the tree API; fetch via ``hf_hub_download``).
- local filesystem paths -- hop-1 ingest by copy.
"""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

_HF_PREFIX = "hf://datasets/"


@dataclass(frozen=True)
class RemoteObject:
    """One reservoir object: canonical URI, store-relative path, identity."""

    uri: str
    relpath: str  # path under the artifact store's store/ root
    size: Optional[int] = None
    sha256: Optional[str] = None  # reservoir-declared content hash, if any


class Source(ABC):
    """Adapter for one reservoir kind."""

    @abstractmethod
    def list(self, base_uri: str) -> List[RemoteObject]:
        """Enumerate all objects under ``base_uri``."""
        raise NotImplementedError

    @abstractmethod
    def fetch(self, obj: RemoteObject, dest: Path) -> None:
        """Materialize one object at ``dest`` (parent dirs exist)."""
        raise NotImplementedError


class HuggingFaceDatasetSource(Source):
    """``hf://datasets/<org>/<name>[/subpath]`` over the Hugging Face Hub."""

    @staticmethod
    def _parse(uri: str) -> tuple[str, str]:
        assert uri.startswith(_HF_PREFIX), f"not an hf dataset uri: {uri}"
        rest = uri[len(_HF_PREFIX) :].strip("/")
        parts = rest.split("/", 2)
        assert len(parts) >= 2, f"expected hf://datasets/<org>/<name>/...: {uri}"
        repo_id = f"{parts[0]}/{parts[1]}"
        subpath = parts[2] if len(parts) == 3 else ""
        return repo_id, subpath

    def list(self, base_uri: str) -> List[RemoteObject]:
        from huggingface_hub import HfApi

        repo_id, subpath = self._parse(base_uri)
        entries = HfApi().list_repo_tree(
            repo_id, subpath or None, repo_type="dataset", recursive=True
        )
        objects = []
        for e in entries:
            if not hasattr(e, "size") or e.size is None:  # folders
                continue
            lfs = getattr(e, "lfs", None)
            objects.append(
                RemoteObject(
                    uri=f"{_HF_PREFIX}{repo_id}/{e.path}",
                    relpath=f"datasets/{repo_id}/{e.path}",
                    size=e.size,
                    sha256=lfs.sha256 if lfs else None,
                )
            )
        return objects

    def fetch(self, obj: RemoteObject, dest: Path) -> None:
        from huggingface_hub import hf_hub_download

        repo_id, filename = self._parse(obj.uri)
        tmp_root = dest.parent / f".fetch-{dest.name}"
        got = hf_hub_download(
            repo_id=repo_id,
            repo_type="dataset",
            filename=filename,
            local_dir=tmp_root,
        )
        shutil.move(got, dest)
        shutil.rmtree(tmp_root, ignore_errors=True)


class LocalSource(Source):
    """Hop-1 ingest: user-supplied files already on disk, copied into the store."""

    def list(self, base_uri: str) -> List[RemoteObject]:
        base = Path(base_uri)
        assert base.exists(), f"local source path does not exist: {base}"
        files = (
            [base]
            if base.is_file()
            else sorted(p for p in base.rglob("*") if p.is_file())
        )
        root = base.parent if base.is_file() else base
        return [
            RemoteObject(
                uri=str(p.resolve()),
                relpath=f"local/{base.name}/{p.relative_to(root)}"
                if base.is_dir()
                else f"local/{p.name}",
                size=p.stat().st_size,
            )
            for p in files
        ]

    def fetch(self, obj: RemoteObject, dest: Path) -> None:
        shutil.copy2(obj.uri, dest)


def get_source(uri: str) -> Source:
    """Pick the source adapter for a URI."""
    if uri.startswith(_HF_PREFIX):
        return HuggingFaceDatasetSource()
    return LocalSource()
