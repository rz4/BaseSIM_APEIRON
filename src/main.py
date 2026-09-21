import signal
import sys

from apeiron.logger import get_logger, configure_backend
from apeiron.config.configuration import build_config, parse_args, Config
from apeiron.experiment import Run, RunInterrupted, seed_everything

from examples.utils import get_example

from apeiron.driver.continuous_monitor import ContinuousMonitor


def _install_interrupt_handlers(monitor: ContinuousMonitor) -> None:
    """Save and stop cleanly when the scheduler warns us, or on SIGTERM.

    Schedulers send a signal before a walltime kill (sbatch --signal=USR1@300).
    The handler only raises a flag; the save happens at the next batch
    boundary, where the state is consistent.
    """
    names = [n for n in ("SIGUSR1", "SIGTERM") if hasattr(signal, n)]
    for name in names:
        signal.signal(getattr(signal, name), lambda *_: monitor.request_interrupt())


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg: Config = build_config(argv)

    # Bounded run directory, when the config asks for one. bind() rewrites the
    # output paths so the rest of apeiron writes inside the run without
    # knowing the run exists.
    run: Run | None = None
    resuming = args.continue_from is not None
    if resuming:
        run = Run.open(args.continue_from)
    elif cfg.experiment is not None:
        run = Run.create(cfg, config_path=args.config)
    if run is not None:
        cfg = run.bind(cfg)
        # Only in experiment mode: legacy runs have never applied cfg.seed and
        # changing that would change their results.
        seed_everything(cfg.seed)

    # Must precede get_example(): get_logger() ignores its arguments once an
    # instance exists, so a harness that logs from __init__ would pin the config.
    backend = configure_backend(cfg)
    logger = get_logger(
        verbosity=cfg.verbosity,
        backend=backend,
        csv_path=cfg.logging.metrics_output_path if cfg.logging else None,
    )
    if run is not None:
        logger.add_log_file(run.log_path)
        logger.info(f"==== Run directory: {run.run_dir} ====", level=0)

    modelHarness = get_example(cfg=cfg)

    if run is not None and not resuming:
        run.record_model(modelHarness)

    # Determine project/experiment name
    project_name = "basesim-framework"
    if cfg.logging and cfg.logging.experiment_name:
        project_name = cfg.logging.experiment_name

    logger.init(cfg, project=project_name)

    # Create continuous monitor - replaces fixed loop and detector instantiation
    monitor = ContinuousMonitor(
        cfg=cfg,
        modelHarness=modelHarness,
        run=run,
    )

    if resuming:
        assert run is not None
        state = run.load_restart()
        if state is None:
            raise SystemExit(
                f"{run.run_dir} has no restart state to continue from. "
                "Set [experiment] restart_interval to save one periodically."
            )
        monitor.restore_state(state)
        run.record("run_continued", batch=state["batch_count"])

    if run is not None:
        _install_interrupt_handlers(monitor)

    # Run continuous monitoring
    try:
        monitor.run()
    except RunInterrupted:
        # Stopped on request with state saved; --continue-from picks it up.
        logger.finish()
        assert run is not None
        run.finish(status="interrupted")
        return 0
    except BaseException:
        if run is not None:
            run.finish(status="failed")
        raise

    # TODO: Save a model checkpoint

    logger.finish()

    if run is not None:
        signature = run.finish()
        logger.info(f"==== Signature: {signature} ====", level=0)

    return 0


if __name__ == "__main__":
    sys.exit(main())
