"""Tests for checkpoint retention policies and the experiment workspace (M6)."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest

from apeiron.config.configuration import ExperimentCfg
from apeiron.experiment import ArtifactStore, Experiment, Journal, Run, apply_retention


@pytest.fixture(autouse=True)
def _patch_retention_logger():
    with patch("apeiron.experiment.retention.get_logger", return_value=MagicMock()):
        yield


def _ckpts(tmp_path, events):
    d = tmp_path / "checkpoints"
    d.mkdir(exist_ok=True)
    for e in events:
        (d / f"drift_adaptation_{e}.pt").write_bytes(b"w")
    (d / "latest").write_text(f"drift_adaptation_{max(events)}.pt")
    return d


def _journal_with_scores(tmp_path, cur=None, hist=None):
    j = Journal(tmp_path / "j.sqlite")
    for event in sorted((cur or {}) | (hist or {})):
        j.record(
            "cl_finished",
            drift_event_id=event,
            post_cur_metrics=[cur[event]] if cur and event in cur else None,
            post_hist_metrics=[hist[event]] if hist and event in hist else None,
        )
    return j


def _surviving(d):
    return sorted(p.name for p in d.glob("drift_adaptation_*.pt"))


# ---------------------------------------------------------------------------
# retention
# ---------------------------------------------------------------------------
class TestRetention:
    def test_noop_under_cap(self, tmp_path):
        d = _ckpts(tmp_path, [1, 2])
        assert apply_retention(d, max_ckpts=2, policy="latest") == []
        assert len(_surviving(d)) == 2

    def test_latest_keeps_newest_n(self, tmp_path):
        d = _ckpts(tmp_path, [1, 2, 3, 4])
        deleted = apply_retention(d, max_ckpts=2, policy="latest")
        assert deleted == ["drift_adaptation_1.pt", "drift_adaptation_2.pt"]
        assert _surviving(d) == ["drift_adaptation_3.pt", "drift_adaptation_4.pt"]
        assert (d / "latest").read_text() == "drift_adaptation_4.pt"

    def test_best_current_keeps_newest_plus_best(self, tmp_path):
        # lower-is-better metric (e.g. VRMSE): event 1 is the best scorer
        d = _ckpts(tmp_path, [1, 2, 3, 4])
        j = _journal_with_scores(tmp_path, cur={1: 0.9, 2: 1.5, 3: 1.4, 4: 1.6})
        deleted = apply_retention(
            d, max_ckpts=2, policy="best_current", journal=j, higher_is_better=False
        )
        assert deleted == ["drift_adaptation_2.pt", "drift_adaptation_3.pt"]
        # newest (4) always survives even though it scored worst
        assert _surviving(d) == ["drift_adaptation_1.pt", "drift_adaptation_4.pt"]
        assert (d / "latest").read_text() == "drift_adaptation_4.pt"
        j.close()

    def test_best_hist_higher_is_better(self, tmp_path):
        d = _ckpts(tmp_path, [1, 2, 3])
        j = _journal_with_scores(tmp_path, hist={1: 60.0, 2: 90.0, 3: 70.0})
        apply_retention(
            d, max_ckpts=2, policy="best_hist", journal=j, higher_is_better=True
        )
        assert _surviving(d) == ["drift_adaptation_2.pt", "drift_adaptation_3.pt"]
        j.close()

    def test_unscored_events_rank_worst(self, tmp_path):
        d = _ckpts(tmp_path, [1, 2, 3])
        j = _journal_with_scores(tmp_path, cur={2: 5.0})  # 1 has no score
        apply_retention(
            d, max_ckpts=2, policy="best_current", journal=j, higher_is_better=True
        )
        assert _surviving(d) == ["drift_adaptation_2.pt", "drift_adaptation_3.pt"]
        j.close()

    def test_metric_policy_without_journal_falls_back(self, tmp_path):
        d = _ckpts(tmp_path, [1, 2, 3])
        apply_retention(d, max_ckpts=2, policy="best_current", journal=None)
        assert _surviving(d) == ["drift_adaptation_2.pt", "drift_adaptation_3.pt"]

    def test_unknown_policy_raises(self, tmp_path):
        d = _ckpts(tmp_path, [1, 2])
        with pytest.raises(ValueError, match="Unknown ckpt_retention"):
            apply_retention(d, max_ckpts=1, policy="best_vibes")


# ---------------------------------------------------------------------------
# workspace
# ---------------------------------------------------------------------------
def _make_run(default_cfg, tmp_path, name, finished=True, drift_events=0):
    cfg = replace(
        default_cfg, experiment=ExperimentCfg(path=str(tmp_path), run_name=name)
    )
    run = Run.create(cfg)
    run.journal.record("window_started", batch_count=0, stream_update_count=0)
    for i in range(drift_events):
        run.journal.record("drift_detected", drift_event_id=i + 1)
        run.journal.record(
            "cl_finished",
            drift_event_id=i + 1,
            post_cur_metrics=[1.0],
            fwt=-0.1 * (i + 1),
            bwt=None,
        )
    if finished:
        run.finish()
    else:
        run.journal.close()
    return run


class TestWorkspace:
    def test_runs_listing(self, default_cfg, tmp_path):
        _make_run(default_cfg, tmp_path, "a", finished=True, drift_events=2)
        _make_run(default_cfg, tmp_path, "b", finished=False)
        exp = Experiment(tmp_path)
        infos = {r.name: r for r in exp.runs()}
        assert infos["a"].status == "finished"
        assert infos["a"].drift_events == 2
        assert infos["a"].last_fwt == pytest.approx(-0.2)
        assert infos["b"].status == "running-or-crashed"

    def test_report_renders(self, default_cfg, tmp_path):
        _make_run(default_cfg, tmp_path, "a")
        text = Experiment(tmp_path).report()
        assert "a" in text and "finished" in text and "signature" in text

    def test_gc_pins_releases_finished_and_vanished(self, default_cfg, tmp_path):
        _make_run(default_cfg, tmp_path, "done", finished=True)
        _make_run(default_cfg, tmp_path, "live", finished=False)
        store = ArtifactStore(tmp_path / "artifacts")
        store.path_for("a").write_bytes(b"1")
        store.record_materialized("uri:a", "a", 1, None)
        for owner in ("done", "live", "vanished_run"):
            store.pin("uri:a", owner)
        store.close()

        released = Experiment(tmp_path).gc_pins()
        assert released == 2  # done + vanished; live survives

        check = ArtifactStore(tmp_path / "artifacts")
        assert check.pinned_uris() == {"uri:a"}
        assert check.release_owner("live") == 1
        check.close()

    def test_store_stats(self, default_cfg, tmp_path):
        _make_run(default_cfg, tmp_path, "a")
        assert Experiment(tmp_path).store_stats() is None  # no store yet
        store = ArtifactStore(tmp_path / "artifacts")
        store.path_for("x").write_bytes(b"123")
        store.record_materialized("uri:x", "x", 3, None)
        store.close()
        stats = Experiment(tmp_path).store_stats()
        assert stats == {"artifacts": 1, "total_bytes": 3, "pinned": 0}
