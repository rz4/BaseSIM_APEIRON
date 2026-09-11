"""Experiment-level artifact store: the disk tier of the residency hierarchy.

One store per experiment workspace, shared by all its runs::

    <experiment>/artifacts/
        index.sqlite    # what is resident, its identity, usage, and pins
        store/<...>     # the materialized files (source-mirroring layout)

The store only tracks state -- what exists locally, how big it is, who has
pinned it, when it was last used. Policy (when to materialize, what to
evict) lives in :class:`apeiron.experiment.residency.ResidencyManager`.

Pins carry an owner (the run name); a pinned artifact is never evicted.
SQLite access is serialized behind a lock so the prefetch thread can share
the store.
"""

from __future__ import annotations

import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS artifacts (
    uri TEXT PRIMARY KEY,
    relpath TEXT NOT NULL,
    size INTEGER,
    sha256 TEXT,
    materialized_ts TEXT NOT NULL,
    last_used_ts TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS pins (
    uri TEXT NOT NULL,
    owner TEXT NOT NULL,
    UNIQUE (uri, owner)
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


class ArtifactStore:
    """State of the local disk tier: resident artifacts, usage, pins."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.store_root = self.root / "store"
        self.store_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self.root / "index.sqlite"), check_same_thread=False
        )
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # ----- paths -----

    def path_for(self, relpath: str) -> Path:
        return self.store_root / relpath

    # ----- records -----

    def is_materialized(self, uri: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT relpath FROM artifacts WHERE uri = ?", (uri,)
            ).fetchone()
        return row is not None and self.path_for(row[0]).exists()

    def record_materialized(
        self, uri: str, relpath: str, size: Optional[int], sha256: Optional[str]
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO artifacts "
                "(uri, relpath, size, sha256, materialized_ts, last_used_ts) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (uri, relpath, size, sha256, _now(), _now()),
            )
            self._conn.commit()

    def touch(self, uri: str) -> None:
        with self._lock:
            self._conn.execute(
                "UPDATE artifacts SET last_used_ts = ? WHERE uri = ?", (_now(), uri)
            )
            self._conn.commit()

    def remove(self, uri: str) -> None:
        """Delete an artifact's file and record (caller checks pins)."""
        with self._lock:
            row = self._conn.execute(
                "SELECT relpath FROM artifacts WHERE uri = ?", (uri,)
            ).fetchone()
            if row is not None:
                self.path_for(row[0]).unlink(missing_ok=True)
                self._conn.execute("DELETE FROM artifacts WHERE uri = ?", (uri,))
                self._conn.commit()

    # ----- accounting -----

    def total_bytes(self) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COALESCE(SUM(size), 0) FROM artifacts"
            ).fetchone()
        return int(row[0])

    def evictable_lru(self) -> List[tuple[str, int]]:
        """Unpinned artifacts as (uri, size), least recently used first."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT a.uri, COALESCE(a.size, 0) FROM artifacts a "
                "WHERE NOT EXISTS (SELECT 1 FROM pins p WHERE p.uri = a.uri) "
                "ORDER BY a.last_used_ts ASC, a.rowid ASC"
            ).fetchall()
        return [(r[0], int(r[1])) for r in rows]

    # ----- pins -----

    def pin(self, uri: str, owner: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO pins (uri, owner) VALUES (?, ?)", (uri, owner)
            )
            self._conn.commit()

    def release_owner(self, owner: str) -> int:
        """Drop every pin held by ``owner``; returns how many were dropped."""
        with self._lock:
            cur = self._conn.execute("DELETE FROM pins WHERE owner = ?", (owner,))
            self._conn.commit()
        return cur.rowcount

    def pinned_uris(self) -> set[str]:
        with self._lock:
            rows = self._conn.execute("SELECT DISTINCT uri FROM pins").fetchall()
        return {r[0] for r in rows}

    def close(self) -> None:
        self._conn.close()
