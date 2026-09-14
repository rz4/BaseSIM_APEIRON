"""Tests for the declarative harness protocol (M5)."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from apeiron.config.configuration import ExperimentCfg
from apeiron.driver.continuous_monitor import ContinuousMonitor
from apeiron.experiment import ArtifactStore, Journal, ResidencyManager, Run
from apeiron.experiment.residency import DataResolver
from apeiron.experiment.sources import LocalSource, WindowSpec


@pytest.fixture(autouse=True)
def _patch_loggers():
    mock_logger = MagicMock()
    mock_logger.step = 0
    with (
        patch("apeiron.driver.continuous_monitor.get_logger", return_value=mock_logger),
        patch(
            "apeiron.training.continuous_trainer.get_logger", return_value=mock_logger
        ),
        patch("apeiron.experiment.residency.get_logger", return_value=mock_logger),
    ):
        yield mock_logger


# ---------------------------------------------------------------------------
# harness defaults
# ---------------------------------------------------------------------------
class TestHarnessDefaults:
    def test_describe_window_default_none(self, dummy_harness):
        assert dummy_harness.describe_window(0) is None
        assert dummy_harness.data_resolver is None

    def test_batch_to_device_tuple(self, dummy_harness):
        batch = (torch.randn(4, 2), torch.randint(0, 3, (4,)))
        moved = dummy_harness.batch_to_device(batch, "cpu")
        assert isinstance(moved, list) and len(moved) == 2
        assert torch.equal(moved[0], batch[0])

    def test_batch_to_device_dict(self, dummy_harness):
        batch = {"x": torch.randn(4, 2), "meta": "tag"}
        moved = dummy_harness.batch_to_device(batch, "cpu")
        assert isinstance(moved, dict)
        assert torch.equal(moved["x"], batch["x"])
        assert moved["meta"] == "tag"  # non-tensors pass through

    def test_batch_size_of(self, dummy_harness):
        assert dummy_harness.batch_size_of((torch.zeros(3, 2), torch.zeros(3))) == 3
        assert dummy_harness.batch_size_of({"x": torch.zeros(5, 2)}) == 5
        assert dummy_harness.batch_size_of(("just", "strings")) is None


class _DictDataset(Dataset):
    def __len__(self):
        return 12

    def __getitem__(self, i):
        return {"x": torch.randn(2), "y": torch.tensor(i % 3)}


class TestTrainerDictBatches:
    def test_safe_next_accepts_dict_batches(self, default_cfg, dummy_harness):
        mon = ContinuousMonitor(cfg=default_cfg, modelHarness=dummy_harness)
        loader = DataLoader(_DictDataset(), batch_size=4)
        it, batch = mon.trainer._safe_next(iter(loader), loader, min_batch=4)
        assert isinstance(batch, dict)
        assert batch["x"].shape[0] == 4

    def test_safe_next_min_batch_still_enforced(self, default_cfg, dummy_harness):
        mon = ContinuousMonitor(cfg=default_cfg, modelHarness=dummy_harness)
        # 12 samples / batch 5 -> sizes 5,5,2; min_batch=5 skips the tail batch
        loader = DataLoader(_DictDataset(), batch_size=5)
        it = iter(loader)
        for _ in range(3):
            it, batch = mon.trainer._safe_next(it, loader, min_batch=5)
            assert batch["x"].shape[0] == 5


# ---------------------------------------------------------------------------
# DataResolver
# ---------------------------------------------------------------------------
def _spec_from_local(tmp_path, name="w0", size=64):
    src = tmp_path / "reservoir"
    src.mkdir(exist_ok=True)
    (src / f"{name}.bin").write_bytes(b"z" * size)
    objs = LocalSource().list(str(src / f"{name}.bin"))
    return WindowSpec(objects=tuple(objs), fingerprint=f"fp-{name}", label=name)


class TestDataResolver:
    def test_materialize_pins_and_journals(self, tmp_path):
        spec = _spec_from_local(tmp_path)
        journal = Journal(tmp_path / "j.sqlite")
        rm = ResidencyManager(ArtifactStore(tmp_path / "artifacts"))
        resolver = DataResolver(rm, pin_owner="run_0001", journal=journal)
        root = resolver.materialize(spec)
        assert (root / spec.objects[0].relpath).exists()
        assert rm.store.pinned_uris() == {spec.objects[0].uri}
        ev = journal.events(kind="window_materialized")
        assert len(ev) == 1
        assert ev[0]["label"] == "w0" and ev[0]["fingerprint"] == "fp-w0"
        assert ev[0]["fetched_count"] == 1 and ev[0]["fetched_bytes"] == 64
        # warm second pass: hits, no fetches
        resolver.materialize(spec)
        ev2 = journal.events(kind="window_materialized")[1]
        assert ev2["hit_count"] == 1 and ev2["fetched_count"] == 0
        journal.close()
        rm.store.close()

    def test_prefetch_none_is_noop(self, tmp_path):
        rm = MagicMock()
        DataResolver(rm, pin_owner="r").prefetch(None)
        rm.prefetch.assert_not_called()

    def test_for_run_wiring(self, default_cfg, tmp_path):
        cfg = replace(
            default_cfg,
            experiment=ExperimentCfg(path=str(tmp_path), artifact_budget_bytes=12345),
        )
        run = Run.create(cfg)
        resolver = DataResolver.for_run(run.bind(cfg), run)
        assert resolver.pin_owner == run.run_dir.name
        assert resolver.journal is run.journal
        assert resolver.residency.budget_bytes == 12345
        assert resolver.store_root == tmp_path / "artifacts" / "store"
        resolver.residency.store.close()
        run.finish()

    def test_window_materialized_excluded_from_signature(self, tmp_path):
        a = Journal(tmp_path / "a.sqlite")
        b = Journal(tmp_path / "b.sqlite")
        for j, fetched in ((a, 5), (b, 0)):  # cold vs warm run
            j.record("window_started", stream_update_count=0, data_fingerprint="fp")
            j.record("window_materialized", label="w0", fetched_count=fetched)
        assert a.signature() == b.signature()
        a.close(), b.close()


# ---------------------------------------------------------------------------
# monitor drives the protocol
# ---------------------------------------------------------------------------
class TestMonitorAdvance:
    def _specs(self, tmp_path, n=3):
        return [_spec_from_local(tmp_path, name=f"w{i}") for i in range(n)]

    def test_materialize_before_update_then_prefetch(
        self, default_cfg, dummy_harness, tmp_path
    ):
        specs = self._specs(tmp_path)
        resolver = MagicMock()
        calls = []
        dummy_harness.data_resolver = resolver
        resolver.materialize.side_effect = lambda s: calls.append(("mat", s.label))
        with (
            patch.object(
                dummy_harness,
                "describe_window",
                side_effect=lambda w: specs[w] if w < len(specs) else None,
            ),
            patch.object(
                dummy_harness,
                "update_data_stream",
                side_effect=lambda: calls.append(("update", None)),
            ),
        ):
            mon = ContinuousMonitor(cfg=default_cfg, modelHarness=dummy_harness)
            mon._advance_stream(0)
        assert calls == [("mat", "w0"), ("update", None)]
        resolver.prefetch.assert_called_once_with(specs[1])

    def test_prefetch_skipped_past_end(self, default_cfg, dummy_harness, tmp_path):
        specs = self._specs(tmp_path, n=1)
        resolver = MagicMock()
        dummy_harness.data_resolver = resolver
        with patch.object(
            dummy_harness,
            "describe_window",
            side_effect=lambda w: specs[w] if w < len(specs) else None,
        ):
            mon = ContinuousMonitor(cfg=default_cfg, modelHarness=dummy_harness)
            mon._advance_stream(0)
        resolver.prefetch.assert_called_once_with(None)

    def test_undeclared_harness_is_legacy(self, default_cfg, dummy_harness):
        mon = ContinuousMonitor(cfg=default_cfg, modelHarness=dummy_harness)
        with patch.object(dummy_harness, "update_data_stream") as mock_update:
            mon._advance_stream(0)
        mock_update.assert_called_once()
        assert mon._last_window_spec is None

    def test_window_journal_uses_spec(self, default_cfg, dummy_harness, tmp_path):
        spec = _spec_from_local(tmp_path, name="regime_a")
        j = Journal(tmp_path / "j.sqlite")
        resolver = MagicMock()
        dummy_harness.data_resolver = resolver
        with patch.object(dummy_harness, "describe_window", return_value=spec):
            mon = ContinuousMonitor(
                cfg=default_cfg, modelHarness=dummy_harness, journal=j
            )
            mon._advance_stream(0, prefetch_next=False)
            mon._journal_window()
        ev = j.events(kind="window_started")[0]
        assert ev["label"] == "regime_a"
        assert ev["data_fingerprint"] == "fp-regime_a"
        j.close()
