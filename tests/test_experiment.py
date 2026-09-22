"""Tests for the run directory and the event log (src/apeiron/experiment)."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from torch import nn

from apeiron.config.configuration import Config, ExperimentCfg, build_config
from apeiron.drift_detection.detectors.base import DriftSignal, LearningRegime
from apeiron.driver.continuous_monitor import ContinuousMonitor
from apeiron.experiment import Journal, Run, behavior_hash, describe_model


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
        assert first.run_dir.parent == tmp_path
        first.finish()
        second.finish()

    def test_skips_existing_numbers(self, default_cfg, tmp_path):
        (tmp_path / "run_0007_old").mkdir(parents=True)
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

    def test_name_groups_runs(self, default_cfg, tmp_path):
        cfg = _exp_cfg(default_cfg, tmp_path, name="well_drift")
        first = Run.create(cfg)
        second = Run.create(cfg)
        assert first.run_dir == tmp_path / "well_drift" / "run_0001"
        assert second.run_dir == tmp_path / "well_drift" / "run_0002"
        # a different experiment numbers from one again, alongside it
        other = Run.create(_exp_cfg(default_cfg, tmp_path, name="mnist_smoke"))
        assert other.run_dir == tmp_path / "mnist_smoke" / "run_0001"
        for run in (first, second, other):
            run.finish()

    def test_name_is_sanitised(self, default_cfg, tmp_path):
        run = Run.create(_exp_cfg(default_cfg, tmp_path, name="../escape"))
        assert run.run_dir == tmp_path / "escape" / "run_0001"
        run.finish()

    def test_ignores_unrelated_contents(self, default_cfg, tmp_path):
        (tmp_path / "mnist.csv").write_text("x")
        (tmp_path / "some_dir").mkdir()
        run = Run.create(_exp_cfg(default_cfg, tmp_path))
        assert run.name == "run_0001"
        assert (tmp_path / "mnist.csv").exists()
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

    def test_empty_section_is_enough(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        toml = tmp_path / "c.toml"
        toml.write_text(MINIMAL_TOML + "\n[experiment]\n")
        cfg = build_config(["--config", str(toml)])
        assert cfg.experiment is not None
        assert cfg.experiment.path == "output"
        assert cfg.experiment.name == ""
        assert cfg.experiment.run_name == ""

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


# ---------------------------------------------------------------------------
# model description
# ---------------------------------------------------------------------------
class TestDescribeModel:
    def test_default_hook_is_empty(self, dummy_harness):
        assert dummy_harness.model_config() == {}

    def test_derived_fields(self, dummy_harness):
        d = describe_model(dummy_harness)
        assert d["class"] == "TinyModel"
        assert d["harness_class"] == "DummyHarness"
        assert d["name"] == dummy_harness.cfg.model.name
        # TinyModel is Linear(4, 3): 12 weights + 3 biases
        assert d["parameters"] == {"total": 15, "trainable": 15}
        assert [t["name"] for t in d["tensors"]] == ["fc.weight", "fc.bias"]
        assert d["tensors"][0]["shape"] == [3, 4]
        assert len(d["shapes_sha256"]) == 64

    def test_hook_value_is_carried(self, default_cfg, make_harness):
        harness = make_harness(default_cfg)
        with patch.object(
            type(harness), "model_config", lambda self: {"width": 4, "depth": 1}
        ):
            assert describe_model(harness)["config"] == {"width": 4, "depth": 1}

    def test_same_architecture_same_hash(self, default_cfg, make_harness):
        # Two harnesses with independently initialised weights: same shapes,
        # so the same hash. The hash is about structure, not values.
        a = describe_model(make_harness(default_cfg))
        b = describe_model(make_harness(default_cfg))
        assert a["shapes_sha256"] == b["shapes_sha256"]

    def test_different_architecture_different_hash(self, default_cfg, make_harness):
        a = describe_model(make_harness(default_cfg))
        b = describe_model(make_harness(default_cfg, model=nn.Linear(8, 3)))
        assert a["shapes_sha256"] != b["shapes_sha256"]

    def test_wrapped_model_is_unwrapped(self, default_cfg, make_harness):
        from apeiron.experiment.model_info import unwrap

        harness = make_harness(default_cfg)
        plain = describe_model(harness)
        harness.model = nn.DataParallel(harness.model)
        assert unwrap(harness.model) is not harness.model
        wrapped = describe_model(harness)
        assert wrapped["class"] == plain["class"] == "TinyModel"
        assert wrapped["shapes_sha256"] == plain["shapes_sha256"]

    def test_pretrained_absent_and_present(self, default_cfg, make_harness, tmp_path):
        assert describe_model(make_harness(default_cfg))["pretrained"] is None

        weights = tmp_path / "w.pth"
        weights.write_bytes(b"0123456789")
        cfg = replace(
            default_cfg, model=replace(default_cfg.model, pretrained_path=str(weights))
        )
        d = describe_model(make_harness(cfg))
        assert d["pretrained"]["exists"] is True
        assert d["pretrained"]["size"] == 10

        missing = replace(
            default_cfg, model=replace(default_cfg.model, pretrained_path="/nope.pth")
        )
        d = describe_model(make_harness(missing))
        assert d["pretrained"] == {"path": "/nope.pth", "exists": False}


class TestRecordModel:
    def test_writes_file_and_event(self, default_cfg, dummy_harness, tmp_path):
        run = Run.create(_exp_cfg(default_cfg, tmp_path))
        run.record_model(dummy_harness)

        on_disk = json.loads((run.run_dir / "model.json").read_text())
        assert on_disk["class"] == "TinyModel"
        assert len(on_disk["tensors"]) == 2

        event = run.journal.last("model")
        assert event is not None
        assert event.payload["shapes_sha256"] == on_disk["shapes_sha256"]
        # the bulky table stays out of the log
        assert "tensors" not in event.payload
        run.finish()

    def test_mtime_left_out_of_the_event(self, default_cfg, make_harness, tmp_path):
        weights = tmp_path / "w.pth"
        weights.write_bytes(b"0123456789")
        cfg = _exp_cfg(
            replace(
                default_cfg,
                model=replace(default_cfg.model, pretrained_path=str(weights)),
            ),
            tmp_path,
        )
        run = Run.create(cfg)
        run.record_model(make_harness(cfg))
        event = run.journal.last("model")
        assert event is not None
        assert "mtime" not in event.payload["pretrained"]
        assert event.payload["pretrained"]["size"] == 10
        assert (
            "mtime"
            in json.loads((run.run_dir / "model.json").read_text())["pretrained"]
        )
        run.finish()

    def test_architecture_change_changes_the_signature(
        self, default_cfg, make_harness, tmp_path
    ):
        cfg = _exp_cfg(default_cfg, tmp_path)
        sigs = []
        for model in (nn.Linear(4, 3), nn.Linear(8, 3)):
            run = Run.create(cfg)
            run.record_model(make_harness(cfg, model=model))
            run.record("window", index=0)
            sigs.append(run.finish())
        assert sigs[0] != sigs[1]

    def test_same_architecture_same_signature(
        self, default_cfg, make_harness, tmp_path
    ):
        cfg = _exp_cfg(default_cfg, tmp_path)
        sigs = []
        for _ in range(2):
            run = Run.create(cfg)
            run.record_model(make_harness(cfg, model=nn.Linear(4, 3)))
            run.record("window", index=0)
            sigs.append(run.finish())
        assert sigs[0] == sigs[1]


# ---------------------------------------------------------------------------
# provenance and naming
# ---------------------------------------------------------------------------
class TestEnvironment:
    def test_records_what_software_ran(self, default_cfg, tmp_path):
        run = Run.create(_exp_cfg(default_cfg, tmp_path))
        started = run.journal.last("run_started")
        assert started is not None
        for key in ("python", "torch", "platform", "hostname", "pid"):
            assert key in started.payload
        run.finish()

    def test_git_revision_when_in_a_repository(self):
        from apeiron.experiment.run import environment

        facts = environment()
        # this repository is one, so the fields should be there and sane
        assert set(facts["git"]) == {"commit", "dirty"}
        assert len(facts["git"]["commit"]) == 40

    def test_git_absent_outside_a_repository(self, monkeypatch):
        from apeiron.experiment import run as run_module

        monkeypatch.setattr(run_module.subprocess, "run", _failing_git)
        assert "git" not in run_module.environment()

    def test_provenance_stays_out_of_the_signature(self, default_cfg, tmp_path):
        """A torch upgrade is worth knowing about; it is not new behaviour."""
        cfg = _exp_cfg(default_cfg, tmp_path)
        first = Run.create(cfg)
        first.record("window", index=0)
        a = first.finish()

        second = Run.create(cfg)
        second.record("run_started", python="9.9", torch="99.0", git={"dirty": True})
        second.record("window", index=0)
        assert second.finish() == a


def _failing_git(*args, **kwargs):
    raise OSError("git not found")


class TestExperimentNameFallback:
    def _toml(self, tmp_path, body: str) -> Path:
        path = tmp_path / "c.toml"
        path.write_text(MINIMAL_TOML + body)
        return path

    def test_falls_back_to_the_logging_project(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        toml = self._toml(
            tmp_path,
            '\n[logging]\nbackend = "none"\nexperiment_name = "mnist-drift"\n'
            "\n[experiment]\n",
        )
        cfg = build_config(["--config", str(toml)])
        assert cfg.experiment is not None
        assert cfg.experiment.name == "mnist-drift"

    def test_an_explicit_name_wins(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        toml = self._toml(
            tmp_path,
            '\n[logging]\nbackend = "none"\nexperiment_name = "tracked-as"\n'
            '\n[experiment]\nname = "stored-as"\n',
        )
        cfg = build_config(["--config", str(toml)])
        assert cfg.experiment is not None
        assert cfg.experiment.name == "stored-as"

    def test_neither_leaves_it_empty(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        toml = self._toml(tmp_path, "\n[experiment]\n")
        cfg = build_config(["--config", str(toml)])
        assert cfg.experiment is not None
        assert cfg.experiment.name == ""
