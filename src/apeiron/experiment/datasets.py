"""Where a run's data lives on this machine.

Data reaches a run two ways. An example can ship raw files next to its config,
in which case they are copied in. A large public dataset is fetched a window at
a time as the run reaches it -- the Well is fifteen terabytes, and a run that
touches three regimes should move three regimes' worth of bytes.

Both routes end in the same place with the same layout, so nothing downstream
can tell which one filled it, and a file put there by hand is indistinguishable
from a downloaded one. On a machine where compute nodes should not be reaching
out to the network, staging the directory ahead of time is the whole story::

    output/well_drift/datasets/
      train/file_0.hdf5
      train/file_1.hdf5.part.48213     <- in progress; .part.* is always garbage

Files are written to a neighbouring ``.part`` name and renamed into place, so a
file that exists is a file that finished. Nothing removes anything: the copy is
kept and reused, which is what makes the second run fast.

A third kind of file is *derived*: produced from the ones that were fetched,
the way a memory-mappable array has to be converted out of HDF5. Those are
built once on the same terms, and carry a ``.origin`` marker naming the recipe
that made them, so a conversion that has since changed is rebuilt rather than
silently reused.
"""

from __future__ import annotations

import os
import shutil
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Union

from apeiron.config.configuration import Config

HF_HOST = "https://huggingface.co"
TIMEOUT_S = 60.0
CHUNK = 1 << 20


# Marks a derived file with the recipe that produced it, so a conversion that
# has since changed is rebuilt instead of being used silently.
ORIGIN_SUFFIX = ".origin"


@dataclass(frozen=True)
class Derived:
    """A file the run produces rather than fetches.

    ``build`` writes the artifact to the path it is given. ``recipe`` is a
    string identifying how it was made: anything that would change the bytes --
    a format version, a dtype, which fields were kept -- belongs in it. A file
    whose recorded recipe does not match is rebuilt, which is the difference
    between a cache and a trap.

    Derived files are built after everything fetched, so a conversion can read
    the sources declared alongside it.
    """

    build: Callable[[Path], None]
    recipe: str


Input = Union[str, Derived]


@dataclass(frozen=True)
class EnsureResult:
    """What one call to :meth:`DatasetStore.ensure` had to do."""

    wanted: int = 0
    present: int = 0  # already on disk and current, nothing moved
    fetched: int = 0
    built: int = 0
    bytes_fetched: int = 0
    seconds: float = 0.0

    def as_payload(self) -> dict[str, float | int]:
        return {
            "wanted": self.wanted,
            "present": self.present,
            "fetched": self.fetched,
            "built": self.built,
            "bytes_fetched": self.bytes_fetched,
            "seconds": round(self.seconds, 3),
        }


def hf_url(uri: str) -> str:
    """``hf://datasets/<owner>/<repo>/<path>`` -> a plain download URL."""
    rest = uri[len("hf://") :]
    if not rest.startswith("datasets/"):
        raise ValueError(f"only hf:// dataset URIs are supported, got {uri!r}")
    parts = rest[len("datasets/") :].split("/")
    if len(parts) < 3:
        raise ValueError(f"hf:// URI needs owner/repo/path, got {uri!r}")
    owner, repo, subpath = parts[0], parts[1], "/".join(parts[2:])
    return f"{HF_HOST}/datasets/{owner}/{repo}/resolve/main/{subpath}"


def _download(url: str, dest: Path) -> int:
    request = urllib.request.Request(url)
    # Gated or private repositories need a token; public ones ignore it.
    if token := os.environ.get("HF_TOKEN"):
        if url.startswith(HF_HOST):
            request.add_header("Authorization", f"Bearer {token}")

    written = 0
    with urllib.request.urlopen(request, timeout=TIMEOUT_S) as response:
        declared = response.headers.get("Content-Length")
        with dest.open("wb") as out:
            while chunk := response.read(CHUNK):
                out.write(chunk)
                written += len(chunk)

    if declared is not None and int(declared) != written:
        raise OSError(f"{url}: expected {declared} bytes, got {written}")
    return written


def fetch(source: str, dest: Path) -> int:
    """Put ``source`` at ``dest`` and return the byte count."""
    if source.startswith("hf://"):
        return _download(hf_url(source), dest)
    if source.startswith(("http://", "https://")):
        return _download(source, dest)

    origin = Path(source[len("file://") :] if source.startswith("file://") else source)
    if not origin.is_file():
        raise FileNotFoundError(f"no such file to copy: {origin}")
    shutil.copyfile(origin, dest)
    size = dest.stat().st_size
    if size != origin.stat().st_size:
        raise OSError(f"{origin}: copy is {size} bytes, source is not")
    return size


class DatasetStore:
    """A directory of dataset files, shared by every run of an experiment."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser()

    @classmethod
    def for_config(cls, cfg: Config) -> DatasetStore | None:
        """The store this config asks for, or None outside experiment mode.

        Defaults to a ``datasets`` directory beside the experiment's runs, so
        the data travels with the experiment. Point ``datasets_path`` at shared
        scratch to have several experiments share one copy.
        """
        if cfg.experiment is None:
            return None
        if cfg.experiment.datasets_path:
            return cls(cfg.experiment.datasets_path)

        root = Path(cfg.experiment.path).expanduser()
        if cfg.experiment.name:
            root = root / cfg.experiment.name
        return cls(root / "datasets")

    def path_for(self, name: str) -> Path:
        """Where a declared name lands. Names are relative and cannot escape."""
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"dataset name must be a relative path: {name!r}")
        return self.root / relative

    def _partial(self, target: Path) -> Path:
        """A name beside the target, so replace() is a rename and not a copy.

        The pid keeps two processes working on the same file out of each
        other's way; both land a complete file and the last one wins.
        """
        return target.with_name(f"{target.name}.part.{os.getpid()}")

    def is_current(self, name: str, recipe: str) -> bool:
        """Whether a derived file exists and was made the way we now ask for."""
        target = self.path_for(name)
        origin = self.path_for(name + ORIGIN_SUFFIX)
        if not target.exists() or not origin.exists():
            return False
        return origin.read_text().strip() == recipe

    def derive(self, name: str, spec: Derived) -> int:
        """Build a derived file unless a current one is already there.

        The recipe marker is renamed into place before the artifact, so a file
        that exists always has a matching marker beside it. A build killed
        halfway leaves a stale marker and no artifact, and the next run rebuilds.
        """
        if self.is_current(name, spec.recipe):
            return 0

        target = self.path_for(name)
        origin = self.path_for(name + ORIGIN_SUFFIX)
        target.parent.mkdir(parents=True, exist_ok=True)

        partial_target = self._partial(target)
        partial_origin = self._partial(origin)
        try:
            partial_origin.write_text(spec.recipe + "\n")
            spec.build(partial_target)
            if not partial_target.exists():
                raise OSError(f"build for {name!r} produced no file")
            partial_origin.replace(origin)
            partial_target.replace(target)
        finally:
            partial_target.unlink(missing_ok=True)
            partial_origin.unlink(missing_ok=True)
        return 1

    def ensure(self, inputs: Mapping[str, Input]) -> EnsureResult:
        """Make every declared file present: fetch what is missing, then build.

        Fetching comes first so that a derived file can read the sources
        declared beside it.
        """
        started = time.monotonic()
        present = fetched = built = moved = 0

        sources = {n: s for n, s in inputs.items() if not isinstance(s, Derived)}
        derived = {n: s for n, s in inputs.items() if isinstance(s, Derived)}

        for name, source in sorted(sources.items()):
            target = self.path_for(name)
            if target.exists():
                present += 1
                continue

            target.parent.mkdir(parents=True, exist_ok=True)
            partial = self._partial(target)
            try:
                moved += fetch(str(source), partial)
                partial.replace(target)
                fetched += 1
            finally:
                partial.unlink(missing_ok=True)

        for name, spec in sorted(derived.items()):
            if self.derive(name, spec):
                built += 1
            else:
                present += 1

        return EnsureResult(
            wanted=len(inputs),
            present=present,
            fetched=fetched,
            built=built,
            bytes_fetched=moved,
            seconds=time.monotonic() - started,
        )
