"""Tests for the experiment report (src/apeiron/experiment/workspace.py)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from apeiron.config.configuration import ExperimentCfg
from apeiron.experiment import Experiment, Run
from apeiron.experiment.__main__ import main as report_main
from apeiron.experiment.workspace import read_run


def _run(cfg, **events) -> Run:
    run = Run.create(cfg)
    for kind, count in events.items():
        for i in range(count):
            run.record(kind, index=i)
    return run


def _cfg(default_cfg, tmp_path, name="exp"):
    return replace(default_cfg, experiment=ExperimentCfg(path=str(tmp_path), name=name))


class TestReadRun:
    def test_counts_what_happened(self, default_cfg, tmp_path):
        cfg = _cfg(default_cfg, tmp_path)
        run = _run(cfg, window=3, drift=2, checkpoint=2)
        run.finish()

        info = read_run(run.run_dir)
        assert (info.windows, info.drifts, info.checkpoints) == (3, 2, 2)
        assert info.status == "finished"
        assert info.signature == run.signature_path.read_text().strip()

    def test_reports_transfer_from_the_last_round(self, default_cfg, tmp_path):
        run = Run.create(_cfg(default_cfg, tmp_path))
        run.record("cl_finished", drift_event_id=1, fwt=1.5, bwt=None)
        run.record("cl_finished", drift_event_id=2, fwt=2.5, bwt=-0.5)
        run.finish()
        info = read_run(run.run_dir)
        assert (info.fwt, info.bwt) == (2.5, -0.5)

    def test_no_rounds_means_no_transfer(self, default_cfg, tmp_path):
        run = Run.create(_cfg(default_cfg, tmp_path))
        run.finish()
        info = read_run(run.run_dir)
        assert info.fwt is None and info.bwt is None

    @pytest.mark.parametrize("status", ["finished", "interrupted", "failed"])
    def test_status_comes_from_the_runs_last_word(self, default_cfg, tmp_path, status):
        run = Run.create(_cfg(default_cfg, tmp_path))
        run.finish(status=status)
        assert read_run(run.run_dir).status == status

    def test_a_run_with_no_last_word(self, default_cfg, tmp_path):
        run = Run.create(_cfg(default_cfg, tmp_path))
        run.record("window", index=0)
        assert read_run(run.run_dir).status == "running or crashed"

    def test_resumable_is_distinguished(self, default_cfg, tmp_path):
        run = Run.create(_cfg(default_cfg, tmp_path))
        run.restart_dir.mkdir(parents=True)
        assert "resumable" in read_run(run.run_dir).status

    def test_signature_computed_for_an_unfinished_run(self, default_cfg, tmp_path):
        """So a crashed run can still be compared with a finished one."""
        run = Run.create(_cfg(default_cfg, tmp_path))
        run.record("window", index=0)
        info = read_run(run.run_dir)
        assert len(info.signature) == 64
        assert not run.signature_path.exists()


class TestExperiment:
    def test_lists_runs_in_order(self, default_cfg, tmp_path):
        cfg = _cfg(default_cfg, tmp_path)
        for _ in range(3):
            Run.create(cfg).finish()
        names = [r.name for r in Experiment(tmp_path / "exp").runs()]
        assert names == ["run_0001", "run_0002", "run_0003"]

    def test_ignores_directories_that_are_not_runs(self, default_cfg, tmp_path):
        cfg = _cfg(default_cfg, tmp_path)
        Run.create(cfg).finish()
        (tmp_path / "exp" / "notes").mkdir()
        (tmp_path / "exp" / "run_9999_incomplete").mkdir()
        assert len(Experiment(tmp_path / "exp").runs()) == 1

    def test_missing_directory_is_not_an_error(self, tmp_path):
        assert Experiment(tmp_path / "nothing").runs() == []
        assert "no runs" in Experiment(tmp_path / "nothing").report()

    def test_flags_runs_that_agree(self, default_cfg, tmp_path):
        cfg = _cfg(default_cfg, tmp_path)
        for _ in range(2):
            run = Run.create(cfg)
            run.record("window", index=0)
            run.finish()
        odd = Run.create(cfg)
        odd.record("window", index=99)
        odd.finish()

        report = Experiment(tmp_path / "exp").report()
        assert "same behaviour: run_0001, run_0002" in report
        assert "run_0003" not in report.split("same behaviour")[1]

    def test_reports_dataset_size(self, default_cfg, tmp_path):
        cfg = _cfg(default_cfg, tmp_path)
        Run.create(cfg).finish()
        store = tmp_path / "exp" / "datasets" / "a"
        store.mkdir(parents=True)
        (store / "f.bin").write_bytes(b"x" * 4096)
        assert "datasets: 4.0 KB" in Experiment(tmp_path / "exp").report()

    def test_a_directory_of_experiments(self, default_cfg, tmp_path):
        for name in ("alpha", "beta"):
            Run.create(_cfg(default_cfg, tmp_path, name=name)).finish()
        report = Experiment(tmp_path).report()
        assert "holds 2 experiment(s)" in report
        assert "alpha  (1 run(s))" in report and "beta  (1 run(s))" in report


class TestCli:
    def test_prints_the_report(self, default_cfg, tmp_path, capsys):
        cfg = _cfg(default_cfg, tmp_path)
        run = Run.create(cfg)
        run.record("drift", event=1)
        run.finish()

        assert report_main([str(tmp_path / "exp")]) == 0
        printed = capsys.readouterr().out
        assert "run_0001" in printed and "finished" in printed

    def test_a_path_with_nothing_in_it(self, tmp_path, capsys):
        assert report_main([str(tmp_path / "empty")]) == 0
        assert "no runs" in capsys.readouterr().out
