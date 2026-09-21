"""Tests for the dataset store (src/apeiron/experiment/datasets.py)."""

from __future__ import annotations

from dataclasses import replace
from unittest.mock import MagicMock, patch

import pytest

from apeiron.config.configuration import ExperimentCfg
from apeiron.driver.continuous_monitor import ContinuousMonitor
from apeiron.experiment import DatasetStore, Run
from apeiron.experiment.datasets import EnsureResult, hf_url


def _source_files(tmp_path, **contents) -> dict[str, str]:
    """Write raw files as an example would ship them, next to its config."""
    shipped = tmp_path / "shipped"
    shipped.mkdir(exist_ok=True)
    declared = {}
    for name, body in contents.items():
        path = shipped / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        declared[name] = str(path)
    return declared


# ---------------------------------------------------------------------------
# uri handling
# ---------------------------------------------------------------------------
class TestHuggingFaceUris:
    def test_translates_to_a_download_url(self):
        assert hf_url(
            "hf://datasets/polymathic-ai/turbulent_radiative_layer_2D/data/train/f.hdf5"
        ) == (
            "https://huggingface.co/datasets/polymathic-ai/"
            "turbulent_radiative_layer_2D/resolve/main/data/train/f.hdf5"
        )

    def test_rejects_non_dataset_uris(self):
        with pytest.raises(ValueError, match="dataset URIs"):
            hf_url("hf://models/owner/repo/file.bin")

    def test_rejects_incomplete_uris(self):
        with pytest.raises(ValueError, match="owner/repo/path"):
            hf_url("hf://datasets/owner/repo")


# ---------------------------------------------------------------------------
# the copy route
# ---------------------------------------------------------------------------
class TestCopyRoute:
    def test_copies_shipped_files_in(self, tmp_path):
        declared = _source_files(tmp_path, **{"train/a.bin": b"x" * 40})
        store = DatasetStore(tmp_path / "datasets")
        result = store.ensure(declared)

        landed = store.path_for("train/a.bin")
        assert landed.read_bytes() == b"x" * 40
        assert (result.wanted, result.fetched, result.present) == (1, 1, 0)
        assert result.bytes_fetched == 40

    def test_second_call_moves_nothing(self, tmp_path):
        declared = _source_files(tmp_path, **{"a.bin": b"y" * 10})
        store = DatasetStore(tmp_path / "datasets")
        store.ensure(declared)

        with patch("apeiron.experiment.datasets.fetch") as never:
            again = store.ensure(declared)
        never.assert_not_called()
        assert (again.present, again.fetched, again.bytes_fetched) == (1, 0, 0)

    def test_file_uri_scheme(self, tmp_path):
        source = tmp_path / "raw.bin"
        source.write_bytes(b"z" * 5)
        store = DatasetStore(tmp_path / "datasets")
        store.ensure({"raw.bin": f"file://{source}"})
        assert store.path_for("raw.bin").read_bytes() == b"z" * 5

    def test_missing_source_is_reported(self, tmp_path):
        store = DatasetStore(tmp_path / "datasets")
        with pytest.raises(FileNotFoundError, match="no such file"):
            store.ensure({"a.bin": str(tmp_path / "nope.bin")})


# ---------------------------------------------------------------------------
# the fetch route
# ---------------------------------------------------------------------------
class TestFetchRoute:
    def _response(self, body: bytes, declared: str | None = None):
        chunks = [body, b""]
        response = MagicMock()
        response.read.side_effect = chunks
        response.headers.get.return_value = declared
        response.__enter__ = lambda self: self
        response.__exit__ = lambda *a: False
        return response

    def test_downloads_and_lands_the_file(self, tmp_path):
        store = DatasetStore(tmp_path / "datasets")
        with patch(
            "urllib.request.urlopen", return_value=self._response(b"w" * 30, "30")
        ):
            result = store.ensure({"data/f.hdf5": "https://example.org/f.hdf5"})
        assert store.path_for("data/f.hdf5").read_bytes() == b"w" * 30
        assert result.fetched == 1 and result.bytes_fetched == 30

    def test_short_read_is_rejected(self, tmp_path):
        store = DatasetStore(tmp_path / "datasets")
        with patch(
            "urllib.request.urlopen", return_value=self._response(b"w" * 10, "999")
        ):
            with pytest.raises(OSError, match="expected 999 bytes"):
                store.ensure({"f.bin": "https://example.org/f.bin"})
        assert not store.path_for("f.bin").exists()

    def test_hf_token_is_sent_only_to_huggingface(self, tmp_path, monkeypatch):
        monkeypatch.setenv("HF_TOKEN", "secret")
        store = DatasetStore(tmp_path / "datasets")
        with patch(
            "urllib.request.urlopen",
            side_effect=lambda *a, **k: self._response(b"a", None),
        ) as opened:
            store.ensure({"a": "hf://datasets/owner/repo/a"})
            store.ensure({"b": "https://elsewhere.example/b"})
        first, second = (call.args[0] for call in opened.call_args_list)
        assert first.get_header("Authorization") == "Bearer secret"
        assert second.get_header("Authorization") is None


# ---------------------------------------------------------------------------
# on-disk discipline
# ---------------------------------------------------------------------------
class TestStoreDiscipline:
    def test_a_failed_fetch_leaves_nothing_behind(self, tmp_path):
        store = DatasetStore(tmp_path / "datasets")
        with patch(
            "apeiron.experiment.datasets.fetch", side_effect=OSError("network gone")
        ):
            with pytest.raises(OSError):
                store.ensure({"a.bin": "https://example.org/a"})
        assert list(store.root.iterdir()) == []

    def test_partial_file_is_never_the_target(self, tmp_path):
        seen = {}

        def slow_fetch(source, dest):
            # What another process would see while this one is writing.
            seen["target_exists"] = (store.root / "a.bin").exists()
            dest.write_bytes(b"q" * 4)
            return 4

        store = DatasetStore(tmp_path / "datasets")
        with patch("apeiron.experiment.datasets.fetch", side_effect=slow_fetch):
            store.ensure({"a.bin": "https://example.org/a"})
        assert seen["target_exists"] is False
        assert store.path_for("a.bin").read_bytes() == b"q" * 4

    def test_hand_staged_files_are_used_as_is(self, tmp_path):
        store = DatasetStore(tmp_path / "datasets")
        staged = store.path_for("train/f.hdf5")
        staged.parent.mkdir(parents=True)
        staged.write_bytes(b"staged by hand")

        with patch("apeiron.experiment.datasets.fetch") as never:
            result = store.ensure({"train/f.hdf5": "hf://datasets/o/r/train/f.hdf5"})
        never.assert_not_called()
        assert result.present == 1
        assert staged.read_bytes() == b"staged by hand"

    def test_names_cannot_escape_the_store(self, tmp_path):
        store = DatasetStore(tmp_path / "datasets")
        for bad in ("../outside.bin", "a/../../outside.bin", "/etc/passwd"):
            with pytest.raises(ValueError, match="relative path"):
                store.path_for(bad)


# ---------------------------------------------------------------------------
# where the store lives
# ---------------------------------------------------------------------------
class TestStoreLocation:
    def test_none_without_experiment_mode(self, default_cfg):
        assert DatasetStore.for_config(default_cfg) is None

    def test_beside_the_experiment_runs_by_default(self, default_cfg, tmp_path):
        cfg = replace(
            default_cfg,
            experiment=ExperimentCfg(path=str(tmp_path), name="well_drift"),
        )
        store = DatasetStore.for_config(cfg)
        assert store is not None
        assert store.root == tmp_path / "well_drift" / "datasets"

    def test_explicit_path_wins(self, default_cfg, tmp_path):
        cfg = replace(
            default_cfg,
            experiment=ExperimentCfg(
                path=str(tmp_path),
                name="well_drift",
                datasets_path=str(tmp_path / "scratch" / "shared"),
            ),
        )
        store = DatasetStore.for_config(cfg)
        assert store is not None
        assert store.root == tmp_path / "scratch" / "shared"

    def test_shared_between_runs_of_one_experiment(self, default_cfg, tmp_path):
        cfg = replace(
            default_cfg, experiment=ExperimentCfg(path=str(tmp_path), name="shared")
        )
        first, second = Run.create(cfg), Run.create(cfg)
        store = DatasetStore.for_config(cfg)
        assert store is not None
        assert first.run_dir.parent == second.run_dir.parent == store.root.parent
        first.finish()
        second.finish()


# ---------------------------------------------------------------------------
# harness hook and the loop
# ---------------------------------------------------------------------------
class TestHarnessHook:
    def test_default_declares_nothing(self, dummy_harness):
        assert dummy_harness.window_inputs(0) == {}
        assert dummy_harness.ensure_window_inputs(0) is None

    def test_no_store_means_nothing_happens(self, dummy_harness, tmp_path):
        with patch.object(
            type(dummy_harness), "window_inputs", lambda self, w: {"a": "b"}
        ):
            assert dummy_harness.ensure_window_inputs(0) is None

    def test_declared_inputs_are_materialised(self, dummy_harness, tmp_path):
        declared = _source_files(tmp_path, **{"a.bin": b"m" * 7})
        dummy_harness.datasets = DatasetStore(tmp_path / "datasets")
        with patch.object(
            type(dummy_harness), "window_inputs", lambda self, w: declared
        ):
            stats = dummy_harness.ensure_window_inputs(0)
        assert isinstance(stats, EnsureResult)
        assert stats.fetched == 1
        assert dummy_harness.datasets.path_for("a.bin").exists()


class TestMonitorMaterialises:
    @pytest.fixture(autouse=True)
    def _quiet_logger(self):
        mock = MagicMock()
        mock.step = 0
        with patch("apeiron.driver.continuous_monitor.get_logger", return_value=mock):
            with patch(
                "apeiron.training.continuous_trainer.get_logger", return_value=mock
            ):
                yield mock

    def test_data_is_present_before_the_harness_opens_it(
        self, default_cfg, dummy_harness, tmp_path
    ):
        declared = _source_files(tmp_path, **{"w.bin": b"d" * 3})
        store = DatasetStore(tmp_path / "datasets")
        dummy_harness.datasets = store
        order = []

        def window_inputs(self, window):
            return declared

        def update_data_stream(self):
            order.append(("open", store.path_for("w.bin").exists()))

        cfg = replace(
            default_cfg, experiment=ExperimentCfg(path=str(tmp_path), name="e")
        )
        run = Run.create(cfg)
        with patch.object(type(dummy_harness), "window_inputs", window_inputs):
            with patch.object(
                type(dummy_harness), "update_data_stream", update_data_stream
            ):
                monitor = ContinuousMonitor(
                    cfg=cfg, modelHarness=dummy_harness, run=run
                )
                monitor._extend_stream()

        assert order == [("open", True)]
        event = run.journal.last("dataset")
        assert event is not None
        assert event.payload["fetched"] == 1
        assert event.payload["window"] == 1
        run.finish()

    def test_nothing_recorded_when_nothing_declared(
        self, default_cfg, dummy_harness, tmp_path
    ):
        cfg = replace(
            default_cfg, experiment=ExperimentCfg(path=str(tmp_path), name="e")
        )
        run = Run.create(cfg)
        monitor = ContinuousMonitor(cfg=cfg, modelHarness=dummy_harness, run=run)
        monitor._extend_stream()
        assert run.journal.count("dataset") == 0
        run.finish()
