"""Residency arbitration: what data exists on local disk at any moment.

The reservoir (a :class:`~apeiron.experiment.sources.Source`) is the truth;
the :class:`~apeiron.experiment.artifacts.ArtifactStore` is a cache of it.
This manager applies the policy at that boundary, the same way memmap
arbitrates disk<->memory: objects **materialize** on demand, an LRU
**eviction** pass keeps the store inside a byte budget, **pins** (held per
run) mark what must survive -- a replay reservoir's history, for example --
and **prefetch** hides the network hop behind compute for a window sequence
that determinism makes predictable.

The budget is soft: pinned artifacts are never evicted, so if pins alone
exceed the budget the store runs over (with a warning) rather than failing
the run.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, List, Optional

from apeiron.experiment.artifacts import ArtifactStore
from apeiron.experiment.sources import RemoteObject, WindowSpec, get_source
from apeiron.logger import get_logger

if TYPE_CHECKING:
    from apeiron.config.configuration import Config
    from apeiron.experiment.journal import Journal
    from apeiron.experiment.run import Run


@dataclass
class EnsureStats:
    """What one ensure() pass did: transfer and cache accounting."""

    fetched_count: int = 0
    fetched_bytes: int = 0
    hit_count: int = 0
    hit_bytes: int = 0
    evicted_bytes: int = 0
    seconds: float = field(default=0.0)


class ResidencyManager:
    """Materialize / evict / pin / prefetch over one artifact store."""

    def __init__(self, store: ArtifactStore, budget_bytes: int = 0):
        """
        :param store: the experiment's artifact store.
        :param budget_bytes: soft capacity of the disk tier; 0 = unlimited.
        """
        self.store = store
        self.budget_bytes = budget_bytes
        self._fetch_lock = threading.Lock()

    # ----- the core verb -----

    def ensure(
        self, objects: Iterable[RemoteObject], owner: Optional[str] = None
    ) -> EnsureStats:
        """Make every object resident; pin for ``owner``; return accounting.

        Idempotent: already-resident objects are touched (LRU) and pinned,
        not refetched. Missing ones are fetched via their source adapter,
        after an eviction pass makes room under the budget.
        """
        logger = get_logger()
        stats = EnsureStats()
        t0 = time.perf_counter()
        for obj in objects:
            if self.store.is_materialized(obj.uri):
                self.store.touch(obj.uri)
                stats.hit_count += 1
                stats.hit_bytes += obj.size or 0
            else:
                stats.evicted_bytes += self._make_room(obj.size or 0)
                dest = self.store.path_for(obj.relpath)
                dest.parent.mkdir(parents=True, exist_ok=True)
                logger.info(
                    f"\t[residency] materializing {obj.uri}"
                    f" ({(obj.size or 0) / 1e6:.0f} MB)",
                    level=1,
                )
                with self._fetch_lock:  # one network fetch at a time
                    if self.store.is_materialized(obj.uri):  # prefetch race
                        stats.hit_count += 1
                        stats.hit_bytes += obj.size or 0
                    else:
                        get_source(obj.uri).fetch(obj, dest)
                        actual = dest.stat().st_size
                        if obj.size is not None and actual != obj.size:
                            dest.unlink(missing_ok=True)
                            raise IOError(
                                f"size mismatch for {obj.uri}: "
                                f"expected {obj.size}, got {actual}"
                            )
                        self.store.record_materialized(
                            obj.uri, obj.relpath, actual, obj.sha256
                        )
                        stats.fetched_count += 1
                        stats.fetched_bytes += actual
            if owner is not None:
                self.store.pin(obj.uri, owner)
        stats.seconds = time.perf_counter() - t0
        return stats

    # ----- eviction -----

    def _make_room(self, incoming_bytes: int) -> int:
        """Evict LRU unpinned artifacts to fit; returns bytes evicted."""
        if self.budget_bytes <= 0:
            return 0
        logger = get_logger()
        evicted = 0
        target = self.budget_bytes - incoming_bytes
        for uri, size in self.store.evictable_lru():
            if self.store.total_bytes() <= target:
                return evicted
            logger.info(f"\t[residency] evicting {uri} ({size / 1e6:.0f} MB)", level=1)
            self.store.remove(uri)
            evicted += size
        if self.store.total_bytes() + incoming_bytes > self.budget_bytes:
            logger.warning(
                "[residency] pinned artifacts exceed the byte budget "
                f"({self.store.total_bytes() + incoming_bytes} > "
                f"{self.budget_bytes}); budget is soft, continuing over it"
            )
        return evicted

    # ----- prefetch -----

    def prefetch(self, objects: List[RemoteObject]) -> threading.Thread:
        """Materialize objects on a daemon thread (unpinned; errors logged).

        With a deterministic window sequence the next window is known, so
        its fetch can overlap the current window's compute.
        """

        def _run() -> None:
            try:
                self.ensure(objects, owner=None)
            except Exception as e:  # noqa: BLE001 - prefetch must never kill a run
                get_logger().warning(f"[residency] prefetch failed: {e}")

        t = threading.Thread(target=_run, name="residency-prefetch", daemon=True)
        t.start()
        return t

    # ----- lifecycle -----

    def release(self, owner: str) -> int:
        """Drop all pins held by ``owner`` (end of run)."""
        return self.store.release_owner(owner)


class DataResolver:
    """The framework-owned face of residency, handed to harnesses.

    A harness that declares its windows (``describe_window`` returning a
    :class:`~apeiron.experiment.sources.WindowSpec`) never touches stores,
    arbiters, pins, or prefetch threads: the monitor calls
    :meth:`materialize` before the harness builds loaders, and the harness
    only needs :attr:`store_root` to locate the materialized files.

    ``materialize`` journals a ``window_materialized`` event with transfer
    accounting. Prefetch never journals (it runs on a thread; journal event
    order must stay deterministic).
    """

    def __init__(
        self,
        residency: ResidencyManager,
        pin_owner: str,
        journal: "Journal | None" = None,
    ):
        self.residency = residency
        self.pin_owner = pin_owner
        self.journal = journal

    @classmethod
    def for_run(cls, cfg: "Config", run: "Run") -> "DataResolver":
        """Build the resolver for a run from its experiment config."""
        assert cfg.experiment is not None
        store = ArtifactStore(Path(cfg.experiment.path) / "artifacts")
        residency = ResidencyManager(
            store, budget_bytes=cfg.experiment.artifact_budget_bytes
        )
        return cls(residency, pin_owner=run.run_dir.name, journal=run.journal)

    @property
    def store_root(self) -> Path:
        return self.residency.store.store_root

    def materialize(self, spec: WindowSpec) -> Path:
        """Ensure a window's objects are resident and pinned; journal stats."""
        stats = self.residency.ensure(spec.objects, owner=self.pin_owner)
        if self.journal is not None:
            self.journal.record(
                "window_materialized",
                label=spec.label,
                fingerprint=spec.fingerprint,
                fetched_count=stats.fetched_count,
                fetched_bytes=stats.fetched_bytes,
                hit_count=stats.hit_count,
                hit_bytes=stats.hit_bytes,
                evicted_bytes=stats.evicted_bytes,
                seconds=round(stats.seconds, 3),
            )
        return self.store_root

    def prefetch(self, spec: Optional[WindowSpec]) -> None:
        """Overlap the next window's fetch with the current window's compute."""
        if spec is not None and spec.objects:
            self.residency.prefetch(list(spec.objects))
