"""Tests for resilience snapshots and exact resume (M7).

The headline tests are the crash-equivalence ones: a run interrupted at an
arbitrary update and resumed from its snapshot must produce a journal
signature-equal to an uninterrupted run of the same config.
"""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from apeiron.config.configuration import (
    Config,
    ContinualLearningCfg,
    DataCfg,
    DriftDetectionCfg,
    ExperimentCfg,
    ModelCfg,
    TrainCfg,
)
from apeiron.drift_detection.detectors.base import (
    BaseDriftDetector,
    DriftSignal,
    LearningRegime,
)
from apeiron.driver.continuous_monitor import ContinuousMonitor
from apeiron.experiment import Journal
from apeiron.experiment.determinism import seed_everything
from apeiron.experiment.snapshot import (
    SnapshotManager,
    restore_rng_states,
    rng_states,
)
from conftest import DummyHarness


@pytest.fixture(autouse=True)
def _patch_loggers():
    mock_logger = MagicMock()
    mock_logger.step = 0
    with (
        patch("apeiron.driver.continuous_monitor.get_logger", return_value=mock_logger),
        patch(
            "apeiron.training.continuous_trainer.get_logger", return_value=mock_logger
        ),
    ):
        yield mock_logger


# ---------------------------------------------------------------------------
# snapshot manager + rng
# ---------------------------------------------------------------------------
class TestSnapshotManager:
    def test_save_load_roundtrip_and_pointer(self, tmp_path):
        sm = SnapshotManager(tmp_path)
        sm.save({"x": torch.tensor([1.0])}, step=5)
        sm.save({"x": torch.tensor([2.0])}, step=9)
        state = sm.load_latest()
        assert state["step"] == 9 and float(state["x"][0]) == 2.0
        assert not list(tmp_path.glob(".tmp-*"))  # atomic: no temp residue

    def test_keep_last_two(self, tmp_path):
        sm = SnapshotManager(tmp_path, keep=2)
        for step in (1, 2, 3):
            sm.save({}, step=step)
        names = sorted(p.name for p in tmp_path.glob("snapshot_*.pt"))
        assert names == ["snapshot_000000002.pt", "snapshot_000000003.pt"]

    def test_empty_dir_loads_none(self, tmp_path):
        assert SnapshotManager(tmp_path).load_latest() is None

    def test_rng_roundtrip(self):
        seed_everything(3)
        saved = rng_states()
        a = torch.randn(4)
        restore_rng_states(saved)
        b = torch.randn(4)
        assert torch.equal(a, b)


# ---------------------------------------------------------------------------
# updater state
# ---------------------------------------------------------------------------
class TestUpdaterState:
    def _cfg(self, mode):
        return Config(
            model=ModelCfg(name="tiny", pretrained_path=""),
            data=DataCfg(name="test", path=""),
            train=TrainCfg(batch_size=8, num_workers=0, init_lr=0.01, max_iter=2),
            continual_learning=ContinualLearningCfg(update_mode=mode),
            drift_detection=DriftDetectionCfg(),
            seed=1,
            device="cpu",
        )

    def test_ewc_roundtrip(self, tmp_path):
        from apeiron.training.updater.ewc import OnlineEWCUpdater

        cfg = self._cfg("ewc_online")
        h1, h2 = DummyHarness(cfg), DummyHarness(cfg)
        u1 = OnlineEWCUpdater(cfg, h1)
        for t in u1.fisher.values():
            t.add_(torch.rand_like(t))
        u1.cl_preprocessing()
        u1._cl_steps = 3
        u2 = OnlineEWCUpdater(cfg, h2)
        u2.load_state_dict(u1.state_dict())
        for k in u1.fisher:
            assert torch.equal(u1.fisher[k], u2.fisher[k])
            assert torch.equal(u1.theta_star[k], u2.theta_star[k])
        assert u2._cl_steps == 3 and u2._cl_fisher_accum is not None

    def test_kfac_roundtrip(self, tmp_path, tiny_cnn):
        from apeiron.training.updater.kfac import OnlineKFACUpdater

        cfg = self._cfg("kfac_online")
        u1 = OnlineKFACUpdater(cfg, DummyHarness(cfg, model=tiny_cnn))
        for t in u1.A.values():
            t.add_(torch.rand_like(t))
        import copy

        u2 = OnlineKFACUpdater(cfg, DummyHarness(cfg, model=copy.deepcopy(tiny_cnn)))
        u2.load_state_dict(u1.state_dict())
        for k in u1.A:
            assert torch.equal(u1.A[k], u2.A[k])

    def test_base_updater_stateless(self, default_cfg, dummy_harness):
        from apeiron.training.updater.base import BaseUpdater

        u = BaseUpdater(default_cfg, dummy_harness)
        assert u.state_dict() == {}
        u.load_state_dict({})  # no-op


# ---------------------------------------------------------------------------
# declarative task registry
# ---------------------------------------------------------------------------
class DeclarativeHarness(DummyHarness):
    """DummyHarness with rebuildable per-window eval sets."""

    def build_window_eval_loader(self, window: int):
        g = torch.Generator().manual_seed(1000 + window)
        ds = TensorDataset(
            torch.randn(16, 4, generator=g),
            torch.randint(0, 3, (16,), generator=g),
        )
        return DataLoader(ds, batch_size=8)


class TestDeclarativeRegistry:
    def test_refs_stored_and_rebuilt(self, default_cfg):
        h = DeclarativeHarness(default_cfg)
        h.register_task([0.5], window=0)
        h.register_task([0.7], window=1)
        assert h.task_records_refs() == [(0, [0.5]), (1, [0.7])]
        rows = h.eval_past_tasks()
        assert len(rows) == 2 and all(len(r) == 1 for r in rows)

    def test_rebuild_is_deterministic(self, default_cfg):
        h = DeclarativeHarness(default_cfg)
        h.register_task([0.5], window=0)
        assert h.eval_past_tasks() == h.eval_past_tasks()

    def test_restore_roundtrip(self, default_cfg, tiny_model):
        h1 = DeclarativeHarness(default_cfg, model=tiny_model)
        h1.register_task([0.5], window=0)
        h2 = DeclarativeHarness(default_cfg, model=tiny_model)
        h2.restore_task_records(h1.task_records_refs())
        assert h2.task_diagonals == [[0.5]]
        assert h2.eval_past_tasks() == h1.eval_past_tasks()

    def test_legacy_harness_returns_none(self, default_cfg, dummy_harness):
        dummy_harness.register_task([0.5])
        assert dummy_harness.task_records_refs() is None
        assert len(dummy_harness.eval_past_tasks()) == 1


# ---------------------------------------------------------------------------
# crash-equivalence
# ---------------------------------------------------------------------------
class StreamHarness(DummyHarness):
    """Deterministic stream harness usable by the real monitor loop.

    Uses a persistent SHUFFLED train loader (EpochSeededSampler) so the
    crash-equivalence tests cover sampler epoch state -- the hidden state
    class that plain generators leak across CL events.
    """

    def __init__(self, cfg):
        super().__init__(cfg)
        from apeiron.experiment.determinism import EpochSeededSampler

        self._train_sampler = EpochSeededSampler(
            len(self._train_ds), cfg.seed, 0, "train"
        )
        self._train_loader = DataLoader(
            self._train_ds,
            batch_size=cfg.train.batch_size,
            sampler=self._train_sampler,
        )

    def get_stream_dataloader(self):
        return DataLoader(self._val_ds, batch_size=self.cfg.data.batch_size)

    def get_train_dataloaders(self):
        return (
            self._train_loader,
            DataLoader(self._val_ds, batch_size=self.cfg.train.batch_size),
        )

    def rng_state_dict(self):
        return {"train": self._train_sampler.epochs_started}

    def load_rng_state_dict(self, state, in_cl: bool):
        epochs = state.get("train", 0)
        self._train_sampler.epochs_started = max(0, epochs - 1) if in_cl else epochs


class FireAtDetector(BaseDriftDetector):
    """Deterministic detector: fires at exact check indices (picklable)."""

    def __init__(self, fire_at=(3,)):
        super().__init__("FireAt")
        self.fire_at = set(fire_at)
        self.checks = 0

    def update(self, value: float, **kwargs) -> DriftSignal:
        self.checks += 1
        return DriftSignal(
            regime=LearningRegime.CONTINUAL_LEARNING,
            drift_detected=self.checks in self.fire_at,
            drift_score=float(value),
        )

    def reset(self):
        self.checks = 0


def _crash_cfg(tmp_path, name):
    return Config(
        model=ModelCfg(
            name="tiny", pretrained_path="", ckpts_path=str(tmp_path / name / "ckpts")
        ),
        data=DataCfg(name="test", path="", batch_size=8),
        train=TrainCfg(batch_size=8, num_workers=0, init_lr=0.01, max_iter=3),
        continual_learning=ContinualLearningCfg(update_mode="base"),
        drift_detection=DriftDetectionCfg(detection_interval=2, max_stream_updates=2),
        experiment=ExperimentCfg(path=str(tmp_path / name), snapshot_interval=1),
        seed=11,
        device="cpu",
    )


def _build_monitor(tmp_path, name, journal, fire_at):
    seed_everything(11)
    cfg = _crash_cfg(tmp_path, name)
    harness = StreamHarness(cfg)
    mon = ContinuousMonitor(cfg=cfg, modelHarness=harness, journal=journal)
    mon.detector = FireAtDetector(fire_at=fire_at)
    return mon


class TestCrashEquivalence:
    @pytest.mark.parametrize("interrupt_at", [3, 7, 11])
    def test_interrupted_resumed_equals_uninterrupted(self, tmp_path, interrupt_at):
        fire_at = (2, 5)  # two drift events over the run

        # Reference: uninterrupted
        ref_journal = Journal(tmp_path / "ref.sqlite")
        _build_monitor(tmp_path, "ref", ref_journal, fire_at).run()

        # Interrupted: request interrupt at update N -> SystemExit
        crash_journal_path = tmp_path / "crash.sqlite"
        crash_journal = Journal(crash_journal_path)
        mon = _build_monitor(tmp_path, "crash", crash_journal, fire_at)
        original_tick = mon._tick_snapshot

        def tick_with_interrupt(phase):
            if mon._update_counter + 1 >= interrupt_at:
                mon.interrupt_requested = True
            original_tick(phase)

        with patch.object(mon, "_tick_snapshot", side_effect=tick_with_interrupt):
            with pytest.raises(SystemExit):
                mon.run()

        # Resume from the snapshot and finish
        resumed_journal = Journal(crash_journal_path)
        mon2 = _build_monitor(tmp_path, "crash", resumed_journal, fire_at)
        snap = SnapshotManager(
            tmp_path / "crash" / "ckpts" / "resilience"
        ).load_latest()
        assert snap is not None
        mon2.restore_snapshot(snap)
        mon2.run()

        a = Journal(tmp_path / "ref.sqlite")
        b = Journal(crash_journal_path)
        assert a.signature() == b.signature(), (
            f"crash-equivalence failed for interrupt_at={interrupt_at}"
        )
        a.close(), b.close()

    def test_abrupt_kill_rolls_journal_back_to_snapshot(self, tmp_path):
        """SIGKILL-style death: journal committed past the last snapshot.

        The resume must roll the journal back to the snapshot's position;
        deterministic replay then re-creates the tail, keeping the final
        journal signature-equal to an uninterrupted run.
        """
        fire_at = (2, 5)
        ref_journal = Journal(tmp_path / "ref.sqlite")
        _build_monitor(tmp_path, "ref", ref_journal, fire_at).run()

        crash_path = tmp_path / "crash.sqlite"
        crash_journal = Journal(crash_path)
        mon = _build_monitor(tmp_path, "crash", crash_journal, fire_at)
        mon.snapshot_interval = 4  # journal runs ahead of snapshots

        class Boom(Exception):
            pass

        original_tick = mon._tick_snapshot

        def tick_then_die(phase):
            original_tick(phase)
            if mon._update_counter >= 6:  # not a snapshot boundary
                raise Boom()

        with patch.object(mon, "_tick_snapshot", side_effect=tick_then_die):
            with pytest.raises(Boom):
                mon.run()

        events_before = len(Journal(crash_path).events())
        resumed_journal = Journal(crash_path)
        mon2 = _build_monitor(tmp_path, "crash", resumed_journal, fire_at)
        mon2.snapshot_interval = 4
        snap = SnapshotManager(
            tmp_path / "crash" / "ckpts" / "resilience"
        ).load_latest()
        mon2.restore_snapshot(snap)
        mon2.run()

        a, b = Journal(tmp_path / "ref.sqlite"), Journal(crash_path)
        assert a.signature() == b.signature()
        assert events_before > 0  # sanity: the crash had journaled something
        a.close(), b.close()

    def test_interrupt_journals_and_exits_cleanly(self, tmp_path):
        journal = Journal(tmp_path / "j.sqlite")
        mon = _build_monitor(tmp_path, "x", journal, fire_at=())
        mon.interrupt_requested = True
        with pytest.raises(SystemExit):
            mon.run()
        reread = Journal(tmp_path / "j.sqlite")
        assert len(reread.events(kind="run_interrupted")) == 1
        assert len(reread.events(kind="snapshot_saved")) >= 1
        reread.close()


# ---------------------------------------------------------------------------
# monitor restore mechanics
# ---------------------------------------------------------------------------
class TestRestore:
    def test_counters_and_detector_restored(self, tmp_path):
        journal = Journal(tmp_path / "j.sqlite")
        mon = _build_monitor(tmp_path, "a", journal, fire_at=())
        mon.batch_count = 7
        mon.stream_update_count = 1
        mon.drift_event_count = 2
        mon.batches_into_window = 3
        mon.metric_buffer = [[0.5]]
        mon.detector.checks = 4
        mon._update_counter = 9
        mon._save_snapshot("monitor")

        mon2 = _build_monitor(tmp_path, "a", journal, fire_at=())
        mon2.restore_snapshot(mon.snapshots.load_latest())
        assert mon2.batch_count == 7
        assert mon2.stream_update_count == 1
        assert mon2.drift_event_count == 2
        assert mon2._resume_skip_batches == 3
        assert mon2.metric_buffer == [[0.5]]
        assert mon2.detector.checks == 4
        assert mon2._update_counter == 9
        assert mon2._resumed_from_snapshot
        journal.close()

    def test_cl_phase_restore_sets_resume(self, tmp_path):
        journal = Journal(tmp_path / "j.sqlite")
        mon = _build_monitor(tmp_path, "b", journal, fire_at=())
        mon._cl_ctx = {
            "drift_event_id": 2,
            "iter_count": 1,
            "pre_cur_metrics": [0.4],
            "pre_hist_metrics": None,
        }
        mon._save_snapshot("cl")
        mon2 = _build_monitor(tmp_path, "b", journal, fire_at=())
        mon2.restore_snapshot(mon.snapshots.load_latest())
        assert mon2._resume_cl == {
            "drift_event_id": 2,
            "start_iter": 2,
            "pre_cur_metrics": [0.4],
            "pre_hist_metrics": None,
        }
        journal.close()

    def test_changed_config_warns(self, tmp_path):
        journal = Journal(tmp_path / "j.sqlite")
        mon = _build_monitor(tmp_path, "c", journal, fire_at=())
        mon._save_snapshot("monitor")
        snap = mon.snapshots.load_latest()
        mon2 = _build_monitor(tmp_path, "c", journal, fire_at=())
        mon2.cfg = replace(mon2.cfg, seed=999)
        mon2.restore_snapshot(snap)  # warns, does not raise
        journal.close()
