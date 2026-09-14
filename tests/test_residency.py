"""Tests for the artifact store, residency arbiter, and source adapters (M4)."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest

from apeiron.config.configuration import ExperimentCfg
from apeiron.experiment import ArtifactStore, ResidencyManager, Run
from apeiron.experiment.sources import (
    HuggingFaceDatasetSource,
    LocalSource,
    get_source,
)


@pytest.fixture(autouse=True)
def _patch_residency_logger():
    with patch(
        "apeiron.experiment.residency.get_logger", return_value=MagicMock()
    ) as m:
        yield m


def _local_objects(tmp_path, sizes: dict[str, int]):
    """Create local files and return their LocalSource objects."""
    src_dir = tmp_path / "reservoir"
    src_dir.mkdir(exist_ok=True)
    for name, size in sizes.items():
        (src_dir / name).write_bytes(b"x" * size)
    return LocalSource().list(str(src_dir))


# ---------------------------------------------------------------------------
# sources
# ---------------------------------------------------------------------------
class TestSources:
    def test_hf_uri_parsing(self):
        repo, sub = HuggingFaceDatasetSource._parse(
            "hf://datasets/polymathic-ai/turbulent_radiative_layer_2D/data/train/f.hdf5"
        )
        assert repo == "polymathic-ai/turbulent_radiative_layer_2D"
        assert sub == "data/train/f.hdf5"

    def test_get_source_dispatch(self, tmp_path):
        assert isinstance(get_source("hf://datasets/a/b"), HuggingFaceDatasetSource)
        assert isinstance(get_source(str(tmp_path)), LocalSource)

    def test_local_list_and_fetch(self, tmp_path):
        objs = _local_objects(tmp_path, {"a.bin": 10, "b.bin": 20})
        assert [o.relpath.rsplit("/", 1)[-1] for o in objs] == ["a.bin", "b.bin"]
        assert [o.size for o in objs] == [10, 20]
        dest = tmp_path / "out" / "a.bin"
        dest.parent.mkdir()
        LocalSource().fetch(objs[0], dest)
        assert dest.read_bytes() == b"x" * 10


# ---------------------------------------------------------------------------
# artifact store
# ---------------------------------------------------------------------------
class TestArtifactStore:
    def test_record_and_accounting(self, tmp_path):
        s = ArtifactStore(tmp_path / "artifacts")
        s.path_for("x/a").parent.mkdir(parents=True)
        s.path_for("x/a").write_bytes(b"12345")
        s.record_materialized("uri:a", "x/a", 5, None)
        assert s.is_materialized("uri:a")
        assert not s.is_materialized("uri:b")
        assert s.total_bytes() == 5
        s.close()

    def test_missing_file_not_materialized(self, tmp_path):
        s = ArtifactStore(tmp_path / "artifacts")
        s.record_materialized("uri:a", "x/a", 5, None)  # file never written
        assert not s.is_materialized("uri:a")
        s.close()

    def test_lru_order_and_pins(self, tmp_path):
        s = ArtifactStore(tmp_path / "artifacts")
        for name in ("a", "b", "c"):
            s.path_for(name).write_bytes(b"1")
            s.record_materialized(f"uri:{name}", name, 1, None)
        s.touch("uri:a")  # a becomes most recent
        assert [u for u, _ in s.evictable_lru()] == ["uri:b", "uri:c", "uri:a"]
        s.pin("uri:b", "run_x")
        assert [u for u, _ in s.evictable_lru()] == ["uri:c", "uri:a"]
        assert s.release_owner("run_x") == 1
        assert len(s.evictable_lru()) == 3
        s.close()

    def test_remove_deletes_file_and_record(self, tmp_path):
        s = ArtifactStore(tmp_path / "artifacts")
        s.path_for("a").write_bytes(b"1")
        s.record_materialized("uri:a", "a", 1, None)
        s.remove("uri:a")
        assert not s.is_materialized("uri:a")
        assert not s.path_for("a").exists()
        s.close()


# ---------------------------------------------------------------------------
# residency manager
# ---------------------------------------------------------------------------
class TestResidencyManager:
    def test_materialize_and_idempotence(self, tmp_path):
        objs = _local_objects(tmp_path, {"a.bin": 100})
        rm = ResidencyManager(ArtifactStore(tmp_path / "artifacts"))
        stats = rm.ensure(objs, owner="run_1")
        local = rm.store.store_root / objs[0].relpath
        assert local.exists() and local.stat().st_size == 100
        assert stats.fetched_count == 1 and stats.fetched_bytes == 100
        assert stats.hit_count == 0
        with patch("apeiron.experiment.residency.get_source") as mock_src:
            stats2 = rm.ensure(objs, owner="run_1")  # second call: no fetch
        mock_src.assert_not_called()
        assert stats2.hit_count == 1 and stats2.hit_bytes == 100
        assert stats2.fetched_count == 0
        rm.store.close()

    def test_size_mismatch_rejected(self, tmp_path):
        objs = _local_objects(tmp_path, {"a.bin": 100})
        bad = [replace(objs[0], size=999)]
        rm = ResidencyManager(ArtifactStore(tmp_path / "artifacts"))
        with pytest.raises(IOError, match="size mismatch"):
            rm.ensure(bad, owner="run_1")
        assert not rm.store.is_materialized(bad[0].uri)
        rm.store.close()

    def test_budget_evicts_lru_unpinned(self, tmp_path):
        objs = _local_objects(tmp_path, {"a.bin": 60, "b.bin": 60, "c.bin": 60})
        rm = ResidencyManager(ArtifactStore(tmp_path / "artifacts"), budget_bytes=150)
        rm.ensure([objs[0]], owner=None)  # a: 60, unpinned
        rm.ensure([objs[1]], owner="run_1")  # b: pinned
        rm.ensure([objs[2]], owner="run_1")  # c: needs room -> evict a
        assert not rm.store.is_materialized(objs[0].uri)  # a evicted (LRU, unpinned)
        assert rm.store.is_materialized(objs[1].uri)  # b pinned, survives
        assert rm.store.is_materialized(objs[2].uri)
        assert rm.store.total_bytes() == 120
        rm.store.close()

    def test_pins_exceed_budget_softly(self, tmp_path):
        objs = _local_objects(tmp_path, {"a.bin": 60, "b.bin": 60})
        rm = ResidencyManager(ArtifactStore(tmp_path / "artifacts"), budget_bytes=100)
        rm.ensure([objs[0]], owner="run_1")
        rm.ensure([objs[1]], owner="run_1")  # nothing evictable; soft overflow
        assert rm.store.is_materialized(objs[0].uri)
        assert rm.store.is_materialized(objs[1].uri)
        assert rm.store.total_bytes() == 120  # over budget, run not failed
        rm.store.close()

    def test_prefetch_thread_materializes_unpinned(self, tmp_path):
        objs = _local_objects(tmp_path, {"a.bin": 10})
        rm = ResidencyManager(ArtifactStore(tmp_path / "artifacts"))
        t = rm.prefetch(objs)
        t.join(timeout=10)
        assert rm.store.is_materialized(objs[0].uri)
        assert rm.store.pinned_uris() == set()
        rm.store.close()

    def test_release(self, tmp_path):
        objs = _local_objects(tmp_path, {"a.bin": 10})
        rm = ResidencyManager(ArtifactStore(tmp_path / "artifacts"))
        rm.ensure(objs, owner="run_1")
        assert rm.release("run_1") == 1
        assert rm.store.pinned_uris() == set()
        rm.store.close()


# ---------------------------------------------------------------------------
# run lifecycle integration
# ---------------------------------------------------------------------------
class TestRunPinRelease:
    def test_finish_releases_this_runs_pins(self, default_cfg, tmp_path):
        cfg = replace(default_cfg, experiment=ExperimentCfg(path=str(tmp_path)))
        run = Run.create(cfg)
        store = ArtifactStore(tmp_path / "artifacts")
        store.path_for("a").write_bytes(b"1")
        store.record_materialized("uri:a", "a", 1, None)
        store.pin("uri:a", run.run_dir.name)
        store.pin("uri:a", "some_other_run")
        store.close()
        run.finish()
        check = ArtifactStore(tmp_path / "artifacts")
        assert check.pinned_uris() == {"uri:a"}  # other run's pin survives
        assert check.release_owner(run.run_dir.name) == 0  # ours already gone
        check.close()

    def test_bind_injects_allocated_run_name(self, default_cfg, tmp_path):
        cfg = replace(default_cfg, experiment=ExperimentCfg(path=str(tmp_path)))
        run = Run.create(cfg)
        bound = run.bind(cfg)
        assert bound.experiment is not None
        assert bound.experiment.run_name == run.run_dir.name
        run.finish()


class TestInflightRobustness:
    """A hung fetch (dead socket after sleep) must never block ensure()."""

    def test_hung_inflight_fetch_does_not_block_ensure(self, tmp_path, monkeypatch):
        import threading
        import time

        from apeiron.experiment.sources import LocalSource as RealLocal

        objs = _local_objects(tmp_path, {"a.bin": 32})
        rm = ResidencyManager(ArtifactStore(tmp_path / "artifacts"))
        monkeypatch.setattr(ResidencyManager, "INFLIGHT_STALL_S", 0.2)
        monkeypatch.setattr(ResidencyManager, "_INFLIGHT_POLL_S", 0.05)

        hang_forever = threading.Event()

        class HangingSource:
            def fetch(self, obj, dest):
                hang_forever.wait()  # dead-socket stand-in: never returns

        sources = iter([HangingSource(), RealLocal()])
        with patch(
            "apeiron.experiment.residency.get_source",
            side_effect=lambda uri: next(sources),
        ):
            rm.prefetch(list(objs))  # grabs the in-flight slot, hangs
            time.sleep(0.1)
            stats = rm.ensure(objs, owner="run_1")  # stalls out, self-fetches

        assert rm.store.is_materialized(objs[0].uri)
        assert stats.fetched_count == 1
        hang_forever.set()
        rm.store.close()

    def test_healthy_inflight_fetch_is_waited_on(self, tmp_path, monkeypatch):
        objs = _local_objects(tmp_path, {"a.bin": 40})
        rm = ResidencyManager(ArtifactStore(tmp_path / "artifacts"))
        monkeypatch.setattr(ResidencyManager, "_INFLIGHT_POLL_S", 0.05)

        t = rm.prefetch(list(objs))
        t.join(timeout=10)
        with patch("apeiron.experiment.residency.get_source") as mock_src:
            stats = rm.ensure(objs, owner="run_1")  # already done: pure hit
        mock_src.assert_not_called()
        assert stats.hit_count == 1
        rm.store.close()
