import sys

from apeiron.logger import get_logger, configure_backend
from apeiron.config.configuration import build_config, parse_args, Config
from apeiron.experiment import Run

from examples.utils import get_example

from apeiron.driver.continuous_monitor import ContinuousMonitor


def main(argv: list[str] | None = None) -> int:
    cfg: Config = build_config(argv)

    # Bounded run directory, when the config asks for one. bind() rewrites the
    # output paths so the rest of apeiron writes inside the run without
    # knowing the run exists.
    run: Run | None = None
    if cfg.experiment is not None:
        run = Run.create(cfg, config_path=parse_args(argv).config)
        cfg = run.bind(cfg)

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

    if run is not None:
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

    # Run continuous monitoring
    try:
        monitor.run()
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
