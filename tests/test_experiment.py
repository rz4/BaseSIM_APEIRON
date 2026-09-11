"""Tests for apeiron.experiment (run directories + event journal)."""

from __future__ import annotations

import json
from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest

from apeiron.config.configuration import ExperimentCfg, build_config
from apeiron.drift_detection.detectors.base import DriftSignal, LearningRegime
from apeiron.driver.continuous_monitor import ContinuousMonitor
from apeiron.experiment import Journal, Run


@pytest.fixture(autouse=True)
def _patch_logger():
    mock_logger = MagicMock()
    mock_logger.step = 0
    with patch(
        "apeiron.driver.continuous_monitor.get_logger", return_value=mock_logger
    ):
        with patch(
            "apeiron.training.continuous_trainer.get_logger", return_value=mock_logger
        ):
            yield mock_logger


# ---------------------------------------------------------------------------
# Journal
# ---------------------------------------------------------------------------
class TestJournal:
    def test_record_and_read_back(self, tmp_path):
        j = Journal(tmp_path / "journal.sqlite")
        j.record("drift_check", batch_count=10, metric=0.5, detected=False)
        j.record("drift_detected", batch_count=12, drift_event_id=1)
        events = j.events()
        assert [e["kind"] for e in events] == ["drift_check", "drift_detected"]
        assert events[0]["batch_count"] == 10
        assert events[0]["metric"] == 0.5
        assert events[0]["detected"] is False
        j.close()

    def test_kind_filter(self, tmp_path):
        j = Journal(tmp_path / "journal.sqlite")
        j.record("a", x=1)
        j.record("b", x=2)
        j.record("a", x=3)
        assert [e["x"] for e in j.events(kind="a")] == [1, 3]
        j.close()

    def test_survives_reopen(self, tmp_path):
        path = tmp_path / "journal.sqlite"
        j = Journal(path)
        j.record("run_started", run_name="run_0001")
        j.close()
        j2 = Journal(path)
        assert j2.events(kind="run_started")[0]["run_name"] == "run_0001"
        j2.close()


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def _exp_cfg(default_cfg, tmp_path, **kwargs):
    return replace(default_cfg, experiment=ExperimentCfg(path=str(tmp_path), **kwargs))


class TestRun:
    def test_create_layout(self, default_cfg, tmp_path):
        toml = tmp_path / "my.toml"
        toml.write_text("# original config\nseed = 1\n")
        run = Run.create(_exp_cfg(default_cfg, tmp_path), original_config=toml)
        assert run.run_dir == tmp_path / "runs" / "run_0001"
        assert (run.run_dir / "config" / "resolved.json").exists()
        assert (
            (run.run_dir / "config" / "original" / "my.toml")
            .read_text()
            .startswith("# original config")
        )
        assert (run.run_dir / "checkpoints").is_dir()
        assert run.journal.events(kind="run_started")[0]["config_sha256"]
        run.finish()
        assert (run.run_dir / "journal.sqlite").exists()

    def test_auto_numbering(self, default_cfg, tmp_path):
        cfg = _exp_cfg(default_cfg, tmp_path)
        r1, r2 = Run.create(cfg), Run.create(cfg)
        assert r1.run_dir.name == "run_0001"
        assert r2.run_dir.name == "run_0002"
        r1.finish(), r2.finish()

    def test_explicit_run_name(self, default_cfg, tmp_path):
        run = Run.create(_exp_cfg(default_cfg, tmp_path, run_name="baseline"))
        assert run.run_dir.name == "baseline"
        run.finish()

    def test_bind_redirects_outputs(self, default_cfg, tmp_path):
        run = Run.create(_exp_cfg(default_cfg, tmp_path))
        bound = run.bind(_exp_cfg(default_cfg, tmp_path))
        assert bound.logging is not None
        assert bound.logging.metrics_output_path == str(run.run_dir / "metrics.csv")
        assert bound.model.ckpts_path == str(run.run_dir / "checkpoints")
        # everything else untouched
        assert bound.seed == default_cfg.seed
        assert bound.train == default_cfg.train
        run.finish()

    def test_resolved_json_is_valid(self, default_cfg, tmp_path):
        run = Run.create(_exp_cfg(default_cfg, tmp_path))
        resolved = json.loads((run.run_dir / "config" / "resolved.json").read_text())
        assert resolved["seed"] == default_cfg.seed
        assert resolved["experiment"]["path"] == str(tmp_path)
        run.finish()


# ---------------------------------------------------------------------------
# Config parsing + CWD hygiene
# ---------------------------------------------------------------------------
class TestExperimentConfig:
    _BASE = """
seed = 7
device = "cpu"
[model]
name = "tiny"
pretrained_path = ""
[data]
name = "test"
path = ""
[train]
batch_size = 4
num_workers = 0
init_lr = 0.01
[drift_detection]
detector_name = "ADWINDetector"
"""

    def test_section_parsed(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        toml = tmp_path / "c.toml"
        toml.write_text(self._BASE + '[experiment]\npath = "exps/demo"\n')
        cfg = build_config(["--config", str(toml)])
        assert cfg.experiment == ExperimentCfg(path="exps/demo")

    def test_experiment_mode_keeps_cwd_clean(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        toml = tmp_path / "c.toml"
        toml.write_text(self._BASE + '[experiment]\npath = "exps/demo"\n')
        build_config(["--config", str(toml)])
        assert not (tmp_path / "resolved_config.json").exists()

    def test_legacy_mode_writes_cwd_resolved_config(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        toml = tmp_path / "c.toml"
        toml.write_text(self._BASE)
        cfg = build_config(["--config", str(toml)])
        assert cfg.experiment is None
        assert (tmp_path / "resolved_config.json").exists()


# ---------------------------------------------------------------------------
# Monitor / trainer journal hooks
# ---------------------------------------------------------------------------
class TestJournalHooks:
    def _signal(self, detected: bool) -> DriftSignal:
        return DriftSignal(
            regime=LearningRegime.CONTINUAL_LEARNING,
            drift_detected=detected,
            drift_score=0.5,
            confidence=0.9,
        )

    def test_monitor_without_journal_unchanged(self, default_cfg, dummy_harness):
        mon = ContinuousMonitor(cfg=default_cfg, modelHarness=dummy_harness)
        assert mon.journal is None
        assert mon.trainer.journal is None

    def test_drift_check_journaled(self, default_cfg, dummy_harness, tmp_path):
        j = Journal(tmp_path / "j.sqlite")
        mon = ContinuousMonitor(cfg=default_cfg, modelHarness=dummy_harness, journal=j)
        mon.metric_buffer = [[1.0], [3.0]]
        with patch.object(mon.detector, "update", return_value=self._signal(False)):
            mon._check_drift()
        ev = j.events(kind="drift_check")
        assert len(ev) == 1
        assert ev[0]["metric"] == 2.0
        assert ev[0]["detected"] is False
        j.close()

    def test_drift_event_journaled(self, default_cfg, dummy_harness, tmp_path):
        j = Journal(tmp_path / "j.sqlite")
        mon = ContinuousMonitor(cfg=default_cfg, modelHarness=dummy_harness, journal=j)
        with patch.object(mon.trainer, "outer_cl_training_loop", return_value=0):
            mon._handle_drift(self._signal(True))
        ev = j.events(kind="drift_detected")
        assert len(ev) == 1
        assert ev[0]["drift_event_id"] == 1
        assert ev[0]["regime"] == "continual_learning"
        j.close()

    def test_window_journaled_on_extend(self, default_cfg, dummy_harness, tmp_path):
        j = Journal(tmp_path / "j.sqlite")
        mon = ContinuousMonitor(cfg=default_cfg, modelHarness=dummy_harness, journal=j)
        mon._extend_stream()
        ev = j.events(kind="window_started")
        assert len(ev) == 1
        assert ev[0]["stream_update_count"] == 1
        j.close()

    def test_cl_round_journaled(
        self, default_cfg, dummy_harness_with_history, tmp_path
    ):
        j = Journal(tmp_path / "j.sqlite")
        mon = ContinuousMonitor(
            cfg=default_cfg, modelHarness=dummy_harness_with_history, journal=j
        )
        mon.trainer.outer_cl_training_loop(drift_event_id=3)
        started = j.events(kind="cl_started")
        finished = j.events(kind="cl_finished")
        assert len(started) == 1 and len(finished) == 1
        assert started[0]["drift_event_id"] == 3
        assert finished[0]["iterations"] == default_cfg.train.max_iter
        assert isinstance(finished[0]["fwt"], float)
        j.close()
