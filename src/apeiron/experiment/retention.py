"""Rule-based checkpoint retention.

Which post-CL checkpoints survive when there are more than ``max_ckpts``?
The policy (``[model] ckpt_retention``) ranks them:

- ``latest`` — newest N (the legacy FIFO behavior).
- ``best_current`` — best score on the window that triggered the event
  (``cl_finished.post_cur_metrics[0]``).
- ``best_hist`` — best score on the historical validation data
  (``cl_finished.post_hist_metrics[0]``).

Two invariants regardless of policy:

- The NEWEST checkpoint always survives (``--continue-from`` needs weights
  matching the stream position), taking one of the N slots.
- The ``latest`` pointer file always names the newest survivor.

Metric-based policies read scores from the run's journal; without a
journal (legacy mode), they fall back to ``latest`` with a warning.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional

from apeiron.logger import get_logger

if TYPE_CHECKING:
    from apeiron.experiment.journal import Journal

_CKPT_RE = re.compile(r"drift_adaptation_(\d+)\.pt$")

POLICIES = ("latest", "best_current", "best_hist")


def _event_scores(journal: "Journal", key: str) -> Dict[int, float]:
    """First-metric score per drift event id from cl_finished events."""
    scores: Dict[int, float] = {}
    for e in journal.events(kind="cl_finished"):
        metrics = e.get(key)
        if metrics:
            scores[int(e["drift_event_id"])] = float(metrics[0])
    return scores


def apply_retention(
    ckpts_dir: str | Path,
    max_ckpts: int,
    policy: str = "latest",
    journal: Optional["Journal"] = None,
    higher_is_better: bool = True,
) -> List[str]:
    """Keep the newest checkpoint plus the policy's top picks; delete the rest.

    Returns the deleted filenames. No-op while at or under ``max_ckpts``.
    """
    if policy not in POLICIES:
        raise ValueError(f"Unknown ckpt_retention policy: {policy!r} (of {POLICIES})")

    ckpts_dir = Path(ckpts_dir)
    found = {
        int(m.group(1)): p
        for p in ckpts_dir.glob("drift_adaptation_*.pt")
        if (m := _CKPT_RE.search(p.name))
    }
    if len(found) <= max_ckpts:
        return []

    effective = policy
    if policy != "latest" and journal is None:
        get_logger().warning(
            f"[retention] policy {policy!r} needs a run journal for scores; "
            "falling back to 'latest'"
        )
        effective = "latest"

    scores: Dict[int, float] = {}
    if effective != "latest":
        assert journal is not None
        key = "post_cur_metrics" if effective == "best_current" else "post_hist_metrics"
        scores = _event_scores(journal, key)
        if not scores:
            get_logger().warning(
                f"[retention] no {key} scores in the journal; falling back to 'latest'"
            )
            effective = "latest"

    newest = max(found)
    if effective == "latest":
        ranked = sorted((e for e in found if e != newest), reverse=True)
    else:
        sign = 1.0 if higher_is_better else -1.0
        worst = float("-inf")
        ranked = sorted(
            (e for e in found if e != newest),
            key=lambda e: sign * scores.get(e, worst),
            reverse=True,
        )

    survivors = {newest, *ranked[: max(0, max_ckpts - 1)]}
    deleted = []
    for event, path in sorted(found.items()):
        if event not in survivors:
            path.unlink()
            deleted.append(path.name)

    (ckpts_dir / "latest").write_text(found[newest].name)
    return deleted
