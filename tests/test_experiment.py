"""Tests for the run directory and the event log (src/apeiron/experiment)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from apeiron.config.configuration import Config, ExperimentCfg, build_config
from apeiron.drift_detection.detectors.base import DriftSignal, LearningRegime
from apeiron.driver.continuous_monitor import ContinuousMonitor
from apeiron.experiment import Journal, Run, behavior_hash


def _exp_cfg(cfg: Config, tmp_path: Path, **kw) -> Config:
    return replace(cfg, experiment=ExperimentCfg(path=str(tmp_path), **kw))


# ---------------------------------------------------------------------------
# journal
# ---------------------------------------------------------------------------
class TestJournal:
    def test_round_trip(self, tmp_path):
        j = Journal(tmp_path / "journal.sqlite")
        j.record("window", index=0)
        j.record("drift_check", batch=10, value=1.5, detected=False)
        events = j.events()
        assert [e.kind for e in events] == ["window", "drift_check"]
        assert events[0].payload == {"index": 0}
        assert events[1].payload["value"] == 1.5
        assert events[0].id < events[1].id
        j.close()

    def test_filtering_and_counting(self, tmp_path):
        j = Journal(tmp_path / "journal.sqlite")
        for i in range(3):
            j.record("window", index=i)
        j.record("drift", event=1)
        assert j.count() == 4
        assert j.count("window") == 3
        assert [e.payload["index"] for e in j.events("window")] == [0, 1, 2]
        last = j.last("window")
        assert last is not None and last.payload["index"] == 2
        assert j.last("nothing") is None
        j.close()

    def test_creates_parent_directory(self, tmp_path):
        j = Journal(tmp_path / "a" / "b" / "journal.sqlite")
        j.record("window", index=0)
        assert (tmp_path / "a" / "b" / "journal.sqlite").exists()
        j.close()

    def test_survives_reopen(self, tmp_path):
        path = tmp_path / "journal.sqlite"
        j = Journal(path)
        j.record("window", index=0)
        sig = j.signature()
        j.close()
        again = Journal(path)
        assert again.count() == 1
        assert again.signature() == sig
        again.close()


class TestSignature:
    def _journal(self, tmp_path, name, events):
        j = Journal(tmp_path / f"{name}.sqlite")
        for kind, payload in events:
            j.record(kind, **payload)
        return j

    def test_same_events_same_signature(self, tmp_path):
        events = [
            ("window", {"index": 0}),
            ("drift_check", {"batch": 10, "value": 2.0}),
        ]
        a = self._journal(tmp_path, "a", events)
        b = self._journal(tmp_path, "b", events)
        assert a.signature() == b.signature()
        a.close()
        b.close()

    def test_different_values_differ(self, tmp_path):
        a = self._journal(tmp_path, "a", [("drift_check", {"value": 2.0})])
        b = self._journal(tmp_path, "b", [("drift_check", {"value": 2.5})])
        assert a.signature() != b.signature()
        a.close()
        b.close()

    def test_order_matters(self, tmp_path):
        a = self._journal(
            tmp_path, "a", [("window", {"index": 0}), ("drift", {"event": 1})]
        )
        b = self._journal(
            tmp_path, "b", [("drift", {"event": 1}), ("window", {"index": 0})]
        )
        assert a.signature() != b.signature()
        a.close()
        b.close()

    def test_volatile_kinds_excluded(self, tmp_path):
        a = self._journal(tmp_path, "a", [("window", {"index": 0})])
        b = self._journal(
            tmp_path,
            "b",
            [
                ("run_started", {"run_dir": "/somewhere", "pid": 1}),
                ("window", {"index": 0}),
                ("run_finished", {"status": "finished", "elapsed_s": 12.3}),
            ],
        )
        assert a.signature() == b.signature()
        a.close()
        b.close()

    def test_volatile_keys_excluded(self, tmp_path):
        a = self._journal(
            tmp_path, "a", [("checkpoint", {"name": "c.pt", "path": "/run_0001/c.pt"})]
        )
        b = self._journal(
            tmp_path, "b", [("checkpoint", {"name": "c.pt", "path": "/run_0002/c.pt"})]
        )
        assert a.signature() == b.signature()
        c = self._journal(
            tmp_path,
            "c",
            [("checkpoint", {"name": "other.pt", "path": "/run_0001/c.pt"})],
        )
        assert a.signature() != c.signature()
        a.close()
        b.close()
        c.close()


# ---------------------------------------------------------------------------
# run directory
# ---------------------------------------------------------------------------
class TestRunLayout:
    def test_allocates_sequential_directories(self, default_cfg, tmp_path):
        cfg = _exp_cfg(default_cfg, tmp_path)
        first = Run.create(cfg)
        second = Run.create(cfg)
        assert first.name == "run_0001"
        assert second.name == "run_0002"
        assert first.run_dir.parent == tmp_path / "runs"
        first.finish()
        second.finish()

    def test_skips_existing_numbers(self, default_cfg, tmp_path):
        (tmp_path / "runs" / "run_0007_old").mkdir(parents=True)
        run = Run.create(_exp_cfg(default_cfg, tmp_path))
        assert run.name == "run_0008"
        run.finish()

    def test_run_name_suffix_is_sanitised(self, default_cfg, tmp_path):
        run = Run.create(_exp_cfg(default_cfg, tmp_path, run_name="smoke test/1"))
        assert run.name == "run_0001_smoke-test-1"
        run.finish()

    def test_copies_config_file(self, default_cfg, tmp_path):
        src = tmp_path / "given.toml"
        src.write_text('[model]\nname = "tiny"\n')
        run = Run.create(_exp_cfg(default_cfg, tmp_path), config_path=src)
        assert (run.run_dir / "config.toml").read_text() == src.read_text()
        run.finish()

    def test_requires_experiment_section(self, default_cfg):
        with pytest.raises(ValueError, match=r"\[experiment\]"):
            Run.create(default_cfg)

    def test_records_run_started(self, default_cfg, tmp_path):
        cfg = _exp_cfg(default_cfg, tmp_path)
        run = Run.create(cfg)
        started = run.journal.last("run_started")
        assert started is not None
        assert started.payload["config_sha256"] == behavior_hash(cfg)
        run.finish()


class TestBind:
    def test_redirects_outputs_into_the_run(self, default_cfg, tmp_path):
        from apeiron.config.configuration import LoggingCfg

        cfg = replace(
            _exp_cfg(default_cfg, tmp_path),
            logging=LoggingCfg(backend="none", metrics_output_path="elsewhere.csv"),
        )
        run = Run.create(cfg)
        bound = run.bind(cfg)

        assert bound.model.ckpts_path == str(run.run_dir / "checkpoints")
        assert bound.logging is not None
        assert bound.logging.metrics_output_path == str(run.run_dir / "metrics.csv")
        assert bound.experiment is not None
        assert bound.experiment.run_name == run.name
        # untouched elsewhere
        assert bound.data == cfg.data
        assert bound.seed == cfg.seed
        run.finish()

    def test_leaves_absent_logging_absent(self, default_cfg, tmp_path):
        cfg = _exp_cfg(default_cfg, tmp_path)
        run = Run.create(cfg)
        assert run.bind(cfg).logging is None
        run.finish()

    def test_writes_resolved_config(self, default_cfg, tmp_path):
        cfg = _exp_cfg(default_cfg, tmp_path)
        run = Run.create(cfg)
        run.bind(cfg)
        resolved = json.loads((run.run_dir / "config.resolved.json").read_text())
        assert resolved["model"]["ckpts_path"] == str(run.run_dir / "checkpoints")
        assert resolved["experiment"]["run_name"] == run.name
        run.finish()

    def test_behavior_hash_ignores_experiment_section(self, default_cfg, tmp_path):
        a = _exp_cfg(default_cfg, tmp_path, run_name="a")
        b = _exp_cfg(default_cfg, tmp_path / "other", run_name="b")
        assert behavior_hash(a) == behavior_hash(b)
        assert behavior_hash(a) != behavior_hash(replace(a, seed=a.seed + 1))


class TestFinish:
    def test_writes_signature_file(self, default_cfg, tmp_path):
        run = Run.create(_exp_cfg(default_cfg, tmp_path))
        run.record("window", index=0)
        signature = run.finish()
        assert run.signature_path.read_text().strip() == signature
        assert len(signature) == 64

    def test_two_identical_runs_agree(self, default_cfg, tmp_path):
        cfg = _exp_cfg(default_cfg, tmp_path)
        sigs = []
        for _ in range(2):
            run = Run.create(cfg)
            run.record("window", index=0)
            run.record("drift", event=1, score=0.5)
            sigs.append(run.finish())
        assert sigs[0] == sigs[1]

    def test_records_status(self, default_cfg, tmp_path):
        run = Run.create(_exp_cfg(default_cfg, tmp_path))
        run.finish(status="failed")
        reopened = Journal(run.run_dir / "journal.sqlite")
        last = reopened.last("run_finished")
        assert last is not None and last.payload["status"] == "failed"
        reopened.close()


# ---------------------------------------------------------------------------
# config parsing
# ---------------------------------------------------------------------------
MINIMAL_TOML = """
seed = 7
device = "cpu"

[model]
name = "tiny"

[data]
name = "test"
path = "/tmp"

[train]
batch_size = 4
num_workers = 0
init_lr = 0.01

[drift_detection]
detector_name = "ADWINDetector"
"""


class TestConfigSection:
    def test_absent_section_is_none(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        toml = tmp_path / "c.toml"
        toml.write_text(MINIMAL_TOML)
        cfg = build_config(["--config", str(toml)])
        assert cfg.experiment is None
        # legacy: resolved config still lands in the working directory
        assert (tmp_path / "resolved_config.json").exists()

    def test_present_section_parsed(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        toml = tmp_path / "c.toml"
        toml.write_text(
            MINIMAL_TOML
            + f'\n[experiment]\npath = "{tmp_path / "exp"}"\nrun_name = "x"\n'
        )
        cfg = build_config(["--config", str(toml)])
        assert cfg.experiment is not None
        assert cfg.experiment.run_name == "x"
        # the working directory is left alone in experiment mode
        assert not (tmp_path / "resolved_config.json").exists()


# ---------------------------------------------------------------------------
# monitor wiring
# ---------------------------------------------------------------------------
@pytest.fixture()
def _patched_logger():
    mock_logger = MagicMock()
    mock_logger.step = 0
    with patch(
        "apeiron.driver.continuous_monitor.get_logger", return_value=mock_logger
    ):
        with patch(
            "apeiron.training.continuous_trainer.get_logger", return_value=mock_logger
        ):
            yield mock_logger


class TestMonitorRecords:
    def test_no_run_is_a_no_op(self, default_cfg, dummy_harness, _patched_logger):
        mon = ContinuousMonitor(cfg=default_cfg, modelHarness=dummy_harness)
        assert mon._run is None
        mon._record("window", index=0)  # must not raise

    def test_window_recorded_on_extend(
        self, default_cfg, dummy_harness, tmp_path, _patched_logger
    ):
        run = Run.create(_exp_cfg(default_cfg, tmp_path))
        mon = ContinuousMonitor(cfg=default_cfg, modelHarness=dummy_harness, run=run)
        mon._extend_stream()
        assert [e.payload["index"] for e in run.journal.events("window")] == [1]
        run.finish()

    def test_drift_check_recorded_every_time(
        self, default_cfg, dummy_harness, tmp_path, _patched_logger
    ):
        run = Run.create(_exp_cfg(default_cfg, tmp_path))
        mon = ContinuousMonitor(cfg=default_cfg, modelHarness=dummy_harness, run=run)
        signal = DriftSignal(
            regime=LearningRegime.STABLE, drift_detected=False, drift_score=0.25
        )
        with patch.object(mon.detector, "update", return_value=signal):
            for value in (1.0, 2.0):
                mon.metric_buffer = [[value]]
                mon.batch_count += 10
                mon._check_drift()

        checks = run.journal.events("drift_check")
        assert [c.payload["value"] for c in checks] == [1.0, 2.0]
        assert [c.payload["detected"] for c in checks] == [False, False]
        assert [c.payload["batch"] for c in checks] == [10, 20]
        run.finish()

    def test_drift_and_checkpoint_recorded(
        self, default_cfg, dummy_harness, tmp_path, _patched_logger
    ):
        run = Run.create(_exp_cfg(default_cfg, tmp_path))
        mon = ContinuousMonitor(cfg=default_cfg, modelHarness=dummy_harness, run=run)
        signal = DriftSignal(
            regime=LearningRegime.FINE_TUNING, drift_detected=True, drift_score=0.9
        )
        with patch.object(mon.trainer, "outer_cl_training_loop", return_value=0):
            with patch.object(
                type(dummy_harness), "ckpts_enabled", property(lambda self: True)
            ):
                with patch.object(
                    dummy_harness,
                    "save_ckpt",
                    return_value=str(tmp_path / "ck" / "drift_adaptation_1.pt"),
                ):
                    mon._handle_drift(signal)

        drift = run.journal.last("drift")
        assert drift is not None
        assert drift.payload["event"] == 1
        assert drift.payload["score"] == 0.9
        assert drift.payload["regime"] == LearningRegime.FINE_TUNING.value

        ckpt = run.journal.last("checkpoint")
        assert ckpt is not None
        assert ckpt.payload["name"] == "drift_adaptation_1.pt"
        run.finish()
