"""Append-only event log for a single run.

One SQLite file per run, one row per event, committed as it is written so the
log survives a killed process. The log answers two questions the metrics CSV
cannot: *what did the run decide, and when*, and *did two runs do the same
thing* (see :meth:`Journal.signature`).
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id      INTEGER PRIMARY KEY AUTOINCREMENT,
    ts      REAL NOT NULL,
    kind    TEXT NOT NULL,
    payload TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS events_kind ON events (kind);
"""

# Event kinds that describe the process rather than the computation: their
# payloads are wall-clock times, host names and absolute paths, all of which
# differ between two runs that did exactly the same work.
VOLATILE_KINDS = frozenset({"run_started", "run_finished"})

# Payload keys with the same problem, on otherwise comparable events.
VOLATILE_KEYS = frozenset(
    {"elapsed_s", "hostname", "path", "pid", "run_dir", "timestamp"}
)


@dataclass(frozen=True)
class Event:
    """One row of the log."""

    id: int
    ts: float
    kind: str
    payload: dict[str, Any]


class Journal:
    """Append-only event log backed by SQLite."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        # isolation_level=None: every INSERT is its own committed transaction,
        # so an abrupt kill loses at most the event being written.
        self._db = sqlite3.connect(
            self.path, isolation_level=None, check_same_thread=False
        )
        self._db.executescript(SCHEMA)

    def record(self, kind: str, **payload: Any) -> int:
        """Append one event and return its id."""
        blob = json.dumps(payload, sort_keys=True, default=str)
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO events (ts, kind, payload) VALUES (?, ?, ?)",
                (time.time(), kind, blob),
            )
        return int(cur.lastrowid or 0)

    def events(self, kind: str | None = None) -> list[Event]:
        """Return events in the order they were written."""
        if kind is None:
            rows = self._db.execute(
                "SELECT id, ts, kind, payload FROM events ORDER BY id"
            )
        else:
            rows = self._db.execute(
                "SELECT id, ts, kind, payload FROM events WHERE kind = ? ORDER BY id",
                (kind,),
            )
        return [Event(r[0], r[1], r[2], json.loads(r[3])) for r in rows]

    def count(self, kind: str | None = None) -> int:
        """Number of events, optionally of one kind."""
        if kind is None:
            row = self._db.execute("SELECT COUNT(*) FROM events").fetchone()
        else:
            row = self._db.execute(
                "SELECT COUNT(*) FROM events WHERE kind = ?", (kind,)
            ).fetchone()
        return int(row[0])

    def last(self, kind: str) -> Event | None:
        """Most recent event of a kind, or None."""
        row = self._db.execute(
            "SELECT id, ts, kind, payload FROM events WHERE kind = ? "
            "ORDER BY id DESC LIMIT 1",
            (kind,),
        ).fetchone()
        return (
            None if row is None else Event(row[0], row[1], row[2], json.loads(row[3]))
        )

    def signature(self) -> str:
        """Content hash of what the run did.

        Two runs of the same config should produce the same signature. Volatile
        kinds and keys are left out so that timing, host and path differences do
        not register as behavioural differences.
        """
        h = hashlib.sha256()
        for ev in self.events():
            if ev.kind in VOLATILE_KINDS:
                continue
            stable = {k: v for k, v in ev.payload.items() if k not in VOLATILE_KEYS}
            h.update(ev.kind.encode())
            h.update(b"\0")
            h.update(json.dumps(stable, sort_keys=True, default=str).encode())
            h.update(b"\n")
        return h.hexdigest()

    def close(self) -> None:
        with self._lock:
            self._db.close()
