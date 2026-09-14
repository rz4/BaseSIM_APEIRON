"""Tests for the determinism contract (M3): seeding, signatures, continue."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest
import torch

from apeiron.config.configuration import ExperimentCfg, config_from_dict
from apeiron.driver.continuous_monitor import ContinuousMonitor
from apeiron.experiment import Journal, Run
from apeiron.experiment.determinism import (
    seed_everything,
    stable_seed,
    window_generator,
)


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
# determinism helpers
# ---------------------------------------------------------------------------
class TestStableSeed:
    def test_stable_across_calls(self):
        assert stable_seed(42, 3, "train") == stable_seed(42, 3, "train")

    def test_distinct_roles_and_windows(self):
        seeds = {
            stable_seed(42, w, r) for w in range(4) for r in ("train", "stream", "hist")
        }
        assert len(seeds) == 12

    def test_no_arithmetic_collisions(self):
        # seed+1/window-1 style collisions must not happen under hashing
        assert stable_seed(40, 2) != stable_seed(41, 1)

    def test_fits_torch_seed(self):
        assert 0 <= stable_seed("anything", 10**9) < 2**63


class TestWindowGenerator:
    def test_same_triple_same_stream(self):
        a = torch.randperm(100, generator=window_generator(7, 2, "train"))
        b = torch.randperm(100, generator=window_generator(7, 2, "train"))
        assert torch.equal(a, b)

    def test_different_window_different_stream(self):
        a = torch.randperm(100, generator=window_generator(7, 2, "train"))
        b = torch.randperm(100, generator=window_generator(7, 3, "train"))
        assert not torch.equal(a, b)

    def test_order_independent(self):
        # Consuming global RNG state must not affect window generators
        seed_everything(0)
        torch.randn(1000)
        a = torch.randperm(100, generator=window_generator(7, 2, "train"))
        seed_everything(0)
        b = torch.randperm(100, generator=window_generator(7, 2, "train"))
        assert torch.equal(a, b)


class TestSeedEverything:
    def test_torch_numpy_random_reproducible(self):
        import random

        import numpy as np

        seed_everything(123)
        t1, n1, r1 = torch.randn(4), np.random.rand(4), random.random()
        seed_everything(123)
        t2, n2, r2 = torch.randn(4), np.random.rand(4), random.random()
        assert torch.equal(t1, t2) and (n1 == n2).all() and r1 == r2


# ---------------------------------------------------------------------------
# journal signature
# ---------------------------------------------------------------------------
class TestJournalSignature:
    def _journal_with(self, tmp_path, name, events):
        j = Journal(tmp_path / f"{name}.sqlite")
        for kind, payload in events:
            j.record(kind, **payload)
        return j

    def test_identical_events_identical_signature(self, tmp_path):
        events = [
            ("run_started", {"run_name": "run_0001", "config_sha256": "abc"}),
            ("drift_check", {"metric": 0.5, "detected": False}),
        ]
        j1 = self._journal_with(tmp_path, "a", events)
        # Different run_name (volatile) must not change the signature
        j2 = self._journal_with(
            tmp_path,
            "b",
            [
                ("run_started", {"run_name": "run_0002", "config_sha256": "abc"}),
                ("drift_check", {"metric": 0.5, "detected": False}),
            ],
        )
        assert j1.signature() == j2.signature()
        j1.close(), j2.close()

    def test_metric_difference_changes_signature(self, tmp_path):
        j1 = self._journal_with(
            tmp_path, "a", [("drift_check", {"metric": 0.5, "detected": False})]
        )
        j2 = self._journal_with(
            tmp_path, "b", [("drift_check", {"metric": 0.6, "detected": False})]
        )
        assert j1.signature() != j2.signature()
        j1.close(), j2.close()

    def test_event_order_matters(self, tmp_path):
        j1 = self._journal_with(tmp_path, "a", [("a", {}), ("b", {})])
        j2 = self._journal_with(tmp_path, "b", [("b", {}), ("a", {})])
        assert j1.signature() != j2.signature()
        j1.close(), j2.close()


# ---------------------------------------------------------------------------
# resume state + continue plumbing
# ---------------------------------------------------------------------------
class TestResumeState:
    def test_empty_journal(self, tmp_path):
        j = Journal(tmp_path / "j.sqlite")
        assert j.resume_state() == {
            "stream_update_count": 0,
            "batch_count": 0,
            "drift_event_count": 0,
        }
        j.close()

    def test_counters_extracted(self, tmp_path):
        j = Journal(tmp_path / "j.sqlite")
        j.record("window_started", batch_count=0, stream_update_count=0)
        j.record("drift_check", batch_count=5, metric=1.0, detected=False)
        j.record("drift_detected", batch_count=10, drift_event_id=1)
        j.record("window_started", batch_count=25, stream_update_count=1)
        j.record("drift_detected", batch_count=30, drift_event_id=2)
        assert j.resume_state() == {
            "stream_update_count": 1,
            "batch_count": 30,
            "drift_event_count": 2,
        }
        j.close()


class TestConfigFromDict:
    def test_round_trip(self, default_cfg, tmp_path):
        from dataclasses import asdict

        cfg = replace(default_cfg, experiment=ExperimentCfg(path=str(tmp_path)))
        rebuilt = config_from_dict(asdict(cfg))
        assert rebuilt == cfg

    def test_tuple_fields_survive_json(self, default_cfg):
        import json
        from dataclasses import asdict

        # asdict -> json -> dict turns tuples into lists; __post_init__ coerces
        raw = json.loads(json.dumps(asdict(default_cfg)))
        rebuilt = config_from_dict(raw)
        assert rebuilt.drift_detection.ensemble_detectors == ()


class TestBehaviorConfigHash:
    def test_run_name_does_not_change_hash(self, default_cfg, tmp_path):
        a = replace(
            default_cfg, experiment=ExperimentCfg(path=str(tmp_path), run_name="a")
        )
        b = replace(
            default_cfg,
            experiment=ExperimentCfg(path=str(tmp_path / "other"), run_name="b"),
        )
        assert Run._behavior_config_hash(a) == Run._behavior_config_hash(b)

    def test_behavior_change_changes_hash(self, default_cfg, tmp_path):
        a = replace(default_cfg, experiment=ExperimentCfg(path=str(tmp_path)))
        b = replace(a, seed=a.seed + 1)
        assert Run._behavior_config_hash(a) != Run._behavior_config_hash(b)

    def test_rerun_signatures_equal(self, default_cfg, tmp_path):
        """End-to-end: two runs differing only in run_name have equal signatures."""
        for name in ("a", "b"):
            cfg = replace(
                default_cfg,
                experiment=ExperimentCfg(path=str(tmp_path), run_name=name),
            )
            run = Run.create(cfg)
            run.journal.record("drift_check", batch_count=5, metric=1.5, detected=False)
            run.finish()
        sig = lambda n: Journal(  # noqa: E731
            tmp_path / "runs" / n / "journal.sqlite"
        ).signature()
        assert sig("a") == sig("b")


class TestRunReopen:
    def test_open_and_latest_checkpoint(self, default_cfg, tmp_path):
        cfg = replace(default_cfg, experiment=ExperimentCfg(path=str(tmp_path)))
        run = Run.create(cfg)
        analysis = run.run_dir / "checkpoints" / "analysis"
        analysis.mkdir()
        (analysis / "drift_adaptation_1.pt").write_bytes(b"x")
        (analysis / "latest").write_text("drift_adaptation_1.pt")
        run.journal.record("window_started", batch_count=3, stream_update_count=1)
        run.finish()

        reopened = Run.open(run.run_dir)
        assert reopened.latest_checkpoint is not None
        assert reopened.latest_checkpoint.name == "drift_adaptation_1.pt"
        assert reopened.resolved_config()["seed"] == default_cfg.seed
        assert reopened.journal.resume_state()["stream_update_count"] == 1
        reopened.journal.close()

    def test_latest_checkpoint_none_without_saves(self, default_cfg, tmp_path):
        cfg = replace(default_cfg, experiment=ExperimentCfg(path=str(tmp_path)))
        run = Run.create(cfg)
        assert run.latest_checkpoint is None
        run.finish()


class TestMonitorFastForward:
    def test_fresh_run_single_stream_update(self, default_cfg, dummy_harness):
        mon = ContinuousMonitor(cfg=default_cfg, modelHarness=dummy_harness)
        mon.max_stream_updates = 0  # exit immediately after setup
        with patch.object(dummy_harness, "update_data_stream") as mock_update:
            with patch.object(mon, "_should_stop", return_value=True):
                mon.run()
        assert mock_update.call_count == 1

    def test_continued_run_fast_forwards(self, default_cfg, dummy_harness):
        mon = ContinuousMonitor(cfg=default_cfg, modelHarness=dummy_harness)
        mon.stream_update_count = 2  # restored from a journal
        with patch.object(dummy_harness, "update_data_stream") as mock_update:
            with patch.object(mon, "_should_stop", return_value=True):
                mon.run()
        assert mock_update.call_count == 3  # windows 0, 1, 2 replayed
