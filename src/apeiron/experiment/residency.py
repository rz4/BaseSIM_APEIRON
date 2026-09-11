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
from pathlib import Path
from typing import Iterable, List, Optional

from apeiron.experiment.artifacts import ArtifactStore
from apeiron.experiment.sources import RemoteObject, get_source
from apeiron.logger import get_logger


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
    ) -> Path:
        """Make every object resident; pin for ``owner``; return the store root.

        Idempotent: already-resident objects are touched (LRU) and pinned,
        not refetched. Missing ones are fetched via their source adapter,
        after an eviction pass makes room under the budget.
        """
        logger = get_logger()
        for obj in objects:
            if self.store.is_materialized(obj.uri):
                self.store.touch(obj.uri)
            else:
                self._make_room(obj.size or 0)
                dest = self.store.path_for(obj.relpath)
                dest.parent.mkdir(parents=True, exist_ok=True)
                logger.info(
                    f"\t[residency] materializing {obj.uri}"
                    f" ({(obj.size or 0) / 1e6:.0f} MB)",
                    level=1,
                )
                with self._fetch_lock:  # one network fetch at a time
                    if not self.store.is_materialized(obj.uri):  # prefetch race
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
            if owner is not None:
                self.store.pin(obj.uri, owner)
        return self.store.store_root

    # ----- eviction -----

    def _make_room(self, incoming_bytes: int) -> None:
        if self.budget_bytes <= 0:
            return
        logger = get_logger()
        target = self.budget_bytes - incoming_bytes
        for uri, size in self.store.evictable_lru():
            if self.store.total_bytes() <= target:
                return
            logger.info(f"\t[residency] evicting {uri} ({size / 1e6:.0f} MB)", level=1)
            self.store.remove(uri)
        if self.store.total_bytes() + incoming_bytes > self.budget_bytes:
            logger.warning(
                "[residency] pinned artifacts exceed the byte budget "
                f"({self.store.total_bytes() + incoming_bytes} > "
                f"{self.budget_bytes}); budget is soft, continuing over it"
            )

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
