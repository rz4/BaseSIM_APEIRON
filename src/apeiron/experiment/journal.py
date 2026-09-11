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

    def close(self) -> None:
        self._conn.close()
