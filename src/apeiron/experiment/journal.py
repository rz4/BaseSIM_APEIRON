"""Append-only event journal for a run, backed by SQLite.

The journal is the run's complete record of what happened and when: window
advances, every drift check (unsampled, unlike the metrics CSV), drift
events, continual-learning rounds with their transfer metrics, and
checkpoints. One row per event; structured columns for what queries filter
on (kind, batch_count, timestamp), JSON for the rest.

Query with any SQLite client, e.g.::

    sqlite3 <run_dir>/journal.sqlite \\
        "SELECT ts, kind, payload FROM events WHERE kind = 'drift_detected'"
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

_SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,
    kind TEXT NOT NULL,
    batch_count INTEGER,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_kind ON events (kind);
"""


class Journal:
    """Append-only SQLite event log; one instance per run."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def record(
        self, kind: str, batch_count: Optional[int] = None, **payload: Any
    ) -> None:
        """Append one event; committed immediately so the journal survives crashes."""
        self._conn.execute(
            "INSERT INTO events (ts, kind, batch_count, payload) VALUES (?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                kind,
                batch_count,
                json.dumps(payload, default=str),
            ),
        )
        self._conn.commit()

    def events(self, kind: Optional[str] = None) -> list[dict[str, Any]]:
        """Read events back (oldest first), optionally filtered by kind."""
        sql = "SELECT id, ts, kind, batch_count, payload FROM events"
        args: tuple[Any, ...] = ()
        if kind is not None:
            sql += " WHERE kind = ?"
            args = (kind,)
        rows = self._conn.execute(sql + " ORDER BY id", args).fetchall()
        return [
            {
                "id": r[0],
                "ts": r[1],
                "kind": r[2],
                "batch_count": r[3],
                **json.loads(r[4]),
            }
            for r in rows
        ]

    # Payload keys that legitimately differ between reruns of the same
    # experiment (allocation-dependent, not behavior-dependent).
    VOLATILE_KEYS = frozenset({"run_name", "original_config", "path"})
    # Event kinds that record cache/transfer/resilience state rather than
    # behavior: cold vs warm reruns differ in materialization, and an
    # interrupted+resumed run differs from an uninterrupted one only in
    # snapshot/interrupt/continue bookkeeping (crash-equivalence).
    VOLATILE_KINDS = frozenset(
        {"window_materialized", "snapshot_saved", "run_interrupted", "run_continued"}
    )

    def signature(self) -> str:
        """Content hash of the run's behavior, for rerun comparison.

        Two runs of the same config on the same data should produce equal
        signatures (the determinism contract). Wall-clock timestamps, row
        ids, allocation-dependent payload keys (run/checkpoint paths), and
        cache-state event kinds (residency accounting) are excluded;
        everything else -- event order, kinds, batch counts, metrics, drift
        decisions -- is included.
        """
        h = hashlib.sha256()
        for e in self.events():
            if e["kind"] in self.VOLATILE_KINDS:
                continue
            payload = {
                k: e[k]
                for k in sorted(e)
                if k not in ("id", "ts") and k not in self.VOLATILE_KEYS
            }
            h.update(json.dumps(payload, sort_keys=True, default=str).encode())
            h.update(b"\x1e")
        return h.hexdigest()

    def resume_state(self) -> dict[str, int]:
        """Counters needed to continue a run from its journal tail.

        Returns ``stream_update_count`` (last window started),
        ``batch_count`` (last recorded), and ``drift_event_count`` (last
        drift event id, 0 if none).
        """
        windows = self.events(kind="window_started")
        drifts = self.events(kind="drift_detected")
        batch_counts = [
            e["batch_count"] for e in self.events() if e.get("batch_count") is not None
        ]
        return {
            "stream_update_count": (
                windows[-1]["stream_update_count"] if windows else 0
            ),
            "batch_count": max(batch_counts) if batch_counts else 0,
            "drift_event_count": (drifts[-1]["drift_event_id"] if drifts else 0),
        }

    def last_id(self) -> int:
        """Id of the newest event (0 if empty); recorded in snapshots."""
        row = self._conn.execute("SELECT COALESCE(MAX(id), 0) FROM events").fetchone()
        return int(row[0])

    def truncate_after(self, event_id: int) -> int:
        """Checkpoint-recovery: drop events newer than ``event_id``.

        A crash can leave events journaled AFTER the last snapshot (the
        journal commits per event, snapshots every N updates). Restoring
        rolls the journal back to the snapshot's position; deterministic
        replay then re-creates the discarded tail identically. Returns the
        number of events dropped.
        """
        cur = self._conn.execute("DELETE FROM events WHERE id > ?", (event_id,))
        self._conn.commit()
        return cur.rowcount

    def close(self) -> None:
        self._conn.close()
