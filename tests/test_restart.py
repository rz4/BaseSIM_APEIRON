"""Tests for restart state and exact resume (src/apeiron/experiment/restart.py).

The load-bearing test is crash equivalence: a run interrupted partway and
resumed must produce the same journal signature as one that ran through.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

from apeiron.config.configuration import (
    Config,
    ContinualLearningCfg,
    DataCfg,
    DriftDetectionCfg,
    ExperimentCfg,
    LoggingCfg,
    ModelCfg,
    TrainCfg,
    config_from_dict,
    parse_args,
)
from apeiron.driver.continuous_monitor import ContinuousMonitor
from apeiron.evaluation.metrics import accuracy
from apeiron.experiment import Journal, Run, RunInterrupted, seed_everything
from apeiron.experiment.determinism import rng_state, set_rng_state
from apeiron.experiment.restart import KEEP, RestartStore
from apeiron.model.torch_model_harness import BaseModelHarness


# ---------------------------------------------------------------------------
# a harness the real monitoring loop can actually drive
# ---------------------------------------------------------------------------
class Net(nn.Module):
    def __init__(self, in_features: int = 4):
        super().__init__()
        self.fc = nn.Linear(in_features, 3)

    def forward(self, x):
        return self.fc(x)


class DriftingHarness(BaseModelHarness):
    """Each window shifts the inputs, so a detector watching the output mean
    fires. Loaders shuffle, which is what makes the resume path non-trivial."""

    def __init__(self, cfg: Config):
        super().__init__(cfg, Net())
        self.eval_metrics = {
            "accuracy": accuracy,
            "signal": lambda y_hat, y: y_hat.mean(),
        }
        self.window = -1
        self.update_data_stream()

    def _dataset(self, n: int, shift: int) -> TensorDataset:
        g = torch.Generator().manual_seed(500 + shift)
        return TensorDataset(
            torch.randn(n, 4, generator=g) + shift,
            torch.randint(0, 3, (n,), generator=g),
        )

    def update_data_stream(self) -> None:
        self.window += 1
        self._stream = self._dataset(256, self.window * 30)
        self._train = self._dataset(32, self.window * 30)

    def get_stream_dataloader(self) -> DataLoader:
        return DataLoader(self._stream, batch_size=8, shuffle=True)

    def get_train_dataloaders(self):
        loader = DataLoader(self._train, batch_size=8, shuffle=True)
        return loader, DataLoader(self._train, batch_size=8)

    def get_hist_dataloaders(self):
        return (None, None)

    def get_criterion(self):
        return nn.CrossEntropyLoss()

    def get_optmizer(self):
        return SGD(self.model.parameters(), lr=self.cfg.train.init_lr)


class StagedHarness(DriftingHarness):
    """Declares a file per window, so the resume path re-materialises them."""

    shipped: Path

    def window_inputs(self, window: int) -> dict[str, str]:
        return {f"w{window}.bin": str(self.shipped / f"w{window}.bin")}


def _cfg(tmp_path, **experiment) -> Config:
    return Config(
        model=ModelCfg(name="tiny", max_ckpts=2, ckpts_path="ignored"),
        data=DataCfg(name="synthetic", path="", batch_size=8),
        train=TrainCfg(batch_size=8, num_workers=0, init_lr=0.05, max_iter=3),
        continual_learning=ContinualLearningCfg(update_mode="ewc_online"),
        drift_detection=DriftDetectionCfg(
            detector_name="ADWINDetector",
            detection_interval=2,
            max_stream_updates=3,
            metric_index=1,
        ),
        logging=LoggingCfg(backend="none", metrics_output_path="ignored.csv"),
        experiment=ExperimentCfg(path=str(tmp_path), **experiment),
        seed=11,
        device="cpu",
    )


@pytest.fixture(autouse=True)
def _quiet_logger():
    mock = MagicMock()
    mock.step = 0
    with patch("apeiron.driver.continuous_monitor.get_logger", return_value=mock):
        with patch("apeiron.training.continuous_trainer.get_logger", return_value=mock):
            yield mock


def _start(cfg: Config, harness_cls=DriftingHarness) -> tuple[Run, ContinuousMonitor]:
    """What main() does: allocate, bind, seed, build, monitor."""
    run = Run.create(cfg)
    bound = run.bind(cfg)
    seed_everything(bound.seed)
    harness = harness_cls(bound)
    run.record_model(harness)
    return run, ContinuousMonitor(cfg=bound, modelHarness=harness, run=run)


def _reopen(run_dir, harness_cls=DriftingHarness) -> tuple[Run, ContinuousMonitor]:
    """What main() does with --continue-from."""
    run = Run.open(run_dir)
    bound = run.bind(run.resolved_config())
    seed_everything(bound.seed)
    harness = harness_cls(bound)
    monitor = ContinuousMonitor(cfg=bound, modelHarness=harness, run=run)
    state = run.load_restart()
    assert state is not None
    monitor.restore_state(state)
    run.record("run_continued", batch=state["batch_count"])
    return run, monitor


def _interrupt_at(monitor: ContinuousMonitor, batch: int) -> None:
    original = monitor._evaluate_batch

    def evaluate(b):
        if monitor.batch_count >= batch:
            monitor.request_interrupt()
        return original(b)

    monitor._evaluate_batch = evaluate  # type: ignore[method-assign]


# ---------------------------------------------------------------------------
# store mechanics
# ---------------------------------------------------------------------------
class TestRestartStore:
    def test_save_and_load_newest(self, tmp_path):
        store = RestartStore(tmp_path / "restart")
        for n in (10, 30, 20):
            store.save({"batch_count": n, "payload": n})
        assert store.latest().name == "state_000030.pt"
        assert store.load_latest()["payload"] == 30

    def test_keeps_only_the_last_few(self, tmp_path):
        store = RestartStore(tmp_path / "restart")
        for n in range(1, 8):
            store.save({"batch_count": n})
        assert len(store.files()) == KEEP
        assert [p.name for p in store.files()] == [
            "state_000006.pt",
            "state_000007.pt",
        ]

    def test_empty_store(self, tmp_path):
        store = RestartStore(tmp_path / "restart")
        assert store.latest() is None
        assert store.load_latest() is None
        assert store.files() == []

    def test_no_partial_files_left_behind(self, tmp_path):
        store = RestartStore(tmp_path / "restart")
        store.save({"batch_count": 1})
        assert [p.name for p in store.directory.iterdir()] == ["state_000001.pt"]

    def test_numeric_not_lexicographic_ordering(self, tmp_path):
        store = RestartStore(tmp_path / "restart")
        for n in (9, 100):
            store.save({"batch_count": n})
        assert store.latest().name == "state_000100.pt"


class TestRngState:
    def test_round_trip(self):
        seed_everything(3)
        state = rng_state()
        first = torch.randn(4)
        set_rng_state(state)
        assert torch.equal(torch.randn(4), first)

    def test_seed_everything_is_reproducible(self):
        seed_everything(7)
        a = (torch.randn(3), torch.rand(1))
        seed_everything(7)
        b = (torch.randn(3), torch.rand(1))
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


class TestJournalTruncation:
    def test_drops_only_the_tail(self, tmp_path):
        j = Journal(tmp_path / "journal.sqlite")
        ids = [j.record("window", index=i) for i in range(5)]
        assert j.truncate_after(ids[2]) == 2
        assert [e.payload["index"] for e in j.events()] == [0, 1, 2]
        assert j.last_id() == ids[2]
        j.close()

    def test_truncating_past_the_end_is_a_no_op(self, tmp_path):
        j = Journal(tmp_path / "journal.sqlite")
        j.record("window", index=0)
        assert j.truncate_after(999) == 0
        assert j.count() == 1
        j.close()


# ---------------------------------------------------------------------------
# updater memory
# ---------------------------------------------------------------------------
class TestUpdaterState:
    def test_base_updater_carries_nothing(self, default_cfg, dummy_harness):
        from apeiron.training.updater.base import BaseUpdater

        u = BaseUpdater(default_cfg, dummy_harness)
        assert u.state_dict() == {}
        u.load_state_dict({"anything": 1})  # must not raise

    def test_ewc_prior_round_trips(self, default_cfg, make_harness):
        from apeiron.training.updater.ewc import OnlineEWCUpdater

        cfg = replace(
            default_cfg,
            continual_learning=replace(
                default_cfg.continual_learning, update_mode="ewc_online"
            ),
        )
        source = OnlineEWCUpdater(cfg, make_harness(cfg))
        for name in source.fisher:
            source.fisher[name] += 0.5

        target = OnlineEWCUpdater(cfg, make_harness(cfg))
        assert not torch.allclose(
            target.fisher["fc.weight"], source.fisher["fc.weight"]
        )
        target.load_state_dict(source.state_dict())
        assert torch.allclose(target.fisher["fc.weight"], source.fisher["fc.weight"])
        assert torch.allclose(
            target.theta_star["fc.weight"], source.theta_star["fc.weight"]
        )

    def test_kfac_prior_round_trips(self, default_cfg, tiny_cnn, make_harness):
        from apeiron.training.updater.kfac import OnlineKFACUpdater

        cfg = replace(
            default_cfg,
            continual_learning=replace(
                default_cfg.continual_learning, update_mode="kfac_online"
            ),
        )
        source = OnlineKFACUpdater(cfg, make_harness(cfg, model=tiny_cnn))
        for name in source.A:
            source.A[name] += 0.25

        target = OnlineKFACUpdater(cfg, make_harness(cfg, model=tiny_cnn))
        target.load_state_dict(source.state_dict())
        for name in source.A:
            assert torch.allclose(target.A[name], source.A[name])
            assert torch.allclose(target.G[name], source.G[name])


# ---------------------------------------------------------------------------
# config round trip and CLI
# ---------------------------------------------------------------------------
class TestResumeConfig:
    def test_config_survives_a_round_trip(self, tmp_path):
        from dataclasses import asdict

        cfg = _cfg(tmp_path, name="exp", restart_interval=5)
        assert config_from_dict(asdict(cfg)) == cfg

    def test_run_reopens_with_its_config(self, tmp_path):
        cfg = _cfg(tmp_path, name="exp")
        run = Run.create(cfg)
        bound = run.bind(cfg)
        run.finish()
        assert Run.open(run.run_dir).resolved_config() == bound

    def test_open_rejects_a_non_run_directory(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not a run directory"):
            Run.open(tmp_path)

    def test_cli_sources_are_mutually_exclusive(self):
        assert parse_args(["--continue-from", "r"]).continue_from is not None
        assert parse_args(["--config", "c.toml"]).continue_from is None
        with pytest.raises(SystemExit):
            parse_args(["--config", "c.toml", "--continue-from", "r"])
        with pytest.raises(SystemExit):
            parse_args([])


# ---------------------------------------------------------------------------
# guards
# ---------------------------------------------------------------------------
class TestRestoreGuards:
    def test_rejects_a_different_architecture(self, tmp_path):
        cfg = _cfg(tmp_path, name="a", restart_interval=2)
        run, monitor = _start(cfg)
        state = monitor.capture_state()
        run.finish()

        other_run, other = _start(_cfg(tmp_path, name="b"))
        other.modelHarness.model = Net(in_features=8)
        with pytest.raises(ValueError, match="differently shaped model"):
            other.restore_state(state)
        other_run.finish()

    def test_rejects_a_different_config(self, tmp_path):
        run, monitor = _start(_cfg(tmp_path, name="a"))
        state = monitor.capture_state()
        run.finish()

        changed = replace(_cfg(tmp_path, name="b"), seed=999)
        other_run, other = _start(changed)
        with pytest.raises(ValueError, match="different config"):
            other.restore_state(state)
        other_run.finish()

    def test_rejects_an_old_schema(self, tmp_path):
        run, monitor = _start(_cfg(tmp_path, name="a"))
        state = monitor.capture_state()
        state["schema"] = 0
        with pytest.raises(ValueError, match="schema"):
            monitor.restore_state(state)
        run.finish()


# ---------------------------------------------------------------------------
# the point of the whole thing
# ---------------------------------------------------------------------------
class TestCrashEquivalence:
    def _reference(self, tmp_path) -> str:
        run, monitor = _start(_cfg(tmp_path, name="reference"))
        monitor.run()
        return run.finish()

    @pytest.mark.parametrize("interrupt_at", [5, 17, 40])
    def test_interrupted_and_resumed_matches_reference(self, tmp_path, interrupt_at):
        reference = self._reference(tmp_path)

        run, monitor = _start(_cfg(tmp_path, name="crashed", restart_interval=4))
        _interrupt_at(monitor, interrupt_at)
        with pytest.raises(RunInterrupted):
            monitor.run()
        run_dir = run.run_dir
        run.finish(status="interrupted")

        resumed_run, resumed = _reopen(run_dir)
        resumed.run()
        assert resumed_run.finish() == reference

    def test_resume_rolls_back_the_journal_tail(self, tmp_path):
        run, monitor = _start(_cfg(tmp_path, name="rollback", restart_interval=8))
        _interrupt_at(monitor, 21)
        with pytest.raises(RunInterrupted):
            monitor.run()
        state = run.load_restart()
        assert state is not None
        run_dir, saved_id = run.run_dir, state["journal_last_id"]
        run.finish(status="interrupted")

        before = Journal(run_dir / "journal.sqlite")
        assert [e for e in before.events() if e.id > saved_id]  # tail to roll back
        before.close()

        _reopen(run_dir)

        after = Journal(run_dir / "journal.sqlite")
        tail = [e for e in after.events() if e.id > saved_id]
        # Ids are never reused, so the survivor count is what to check: the
        # rolled-back events are gone and only the resume marker is new.
        assert [e.kind for e in tail] == ["run_continued"]
        after.close()

    def test_declared_data_does_not_double_count_on_resume(self, tmp_path):
        """Resuming re-materialises windows already passed. That must not
        re-record them: the events survived the journal rollback."""
        shipped = tmp_path / "shipped"
        shipped.mkdir()
        for w in range(5):
            (shipped / f"w{w}.bin").write_bytes(b"data" * (w + 1))
        StagedHarness.shipped = shipped

        run, monitor = _start(_cfg(tmp_path, name="staged_ref"), StagedHarness)
        monitor.run()
        reference = run.finish()

        run, monitor = _start(
            _cfg(tmp_path, name="staged", restart_interval=4), StagedHarness
        )
        _interrupt_at(monitor, 40)
        with pytest.raises(RunInterrupted):
            monitor.run()
        run_dir = run.run_dir
        run.finish(status="interrupted")

        resumed_run, resumed = _reopen(run_dir, StagedHarness)
        resumed.run()
        assert resumed_run.finish() == reference

        # the behavioural part -- which files -- is on the window events
        log = Journal(run_dir / "journal.sqlite")
        windows = log.events("window")
        assert [w.payload["inputs"] for w in windows] == [
            [f"w{i}.bin"] for i in range(len(windows))
        ]
        log.close()

    def test_two_interruptions_still_match(self, tmp_path):
        reference = self._reference(tmp_path)

        run, monitor = _start(_cfg(tmp_path, name="twice", restart_interval=3))
        _interrupt_at(monitor, 9)
        with pytest.raises(RunInterrupted):
            monitor.run()
        run_dir = run.run_dir
        run.finish(status="interrupted")

        run_b, monitor_b = _reopen(run_dir)
        _interrupt_at(monitor_b, 31)
        with pytest.raises(RunInterrupted):
            monitor_b.run()
        run_b.finish(status="interrupted")

        run_c, monitor_c = _reopen(run_dir)
        monitor_c.run()
        assert run_c.finish() == reference


class TestRestartLifecycle:
    def test_finished_runs_drop_their_restart_state(self, tmp_path):
        run, monitor = _start(_cfg(tmp_path, name="clean", restart_interval=4))
        monitor.run()
        assert run.restart.latest() is not None
        run.finish()
        assert not run.restart_dir.exists()

    def test_interrupted_runs_keep_it(self, tmp_path):
        run, monitor = _start(_cfg(tmp_path, name="kept", restart_interval=4))
        _interrupt_at(monitor, 6)
        with pytest.raises(RunInterrupted):
            monitor.run()
        run.finish(status="interrupted")
        assert run.restart_dir.is_dir()
        assert RestartStore(run.restart_dir).latest() is not None

    def test_interval_zero_saves_only_on_interrupt(self, tmp_path):
        run, monitor = _start(_cfg(tmp_path, name="signal_only"))
        _interrupt_at(monitor, 7)
        with pytest.raises(RunInterrupted):
            monitor.run()
        assert len(run.restart.files()) == 1
        run.finish(status="interrupted")

    def test_restart_events_do_not_change_the_signature(self, tmp_path):
        plain, plain_monitor = _start(_cfg(tmp_path, name="plain"))
        plain_monitor.run()
        a = plain.finish()

        saving, saving_monitor = _start(
            _cfg(tmp_path, name="saving", restart_interval=2)
        )
        saving_monitor.run()
        assert saving.journal.count("restart_saved") > 0
        assert saving.finish() == a
