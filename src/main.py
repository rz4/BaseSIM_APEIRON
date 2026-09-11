import sys

from apeiron.logger import get_logger, configure_backend
from apeiron.config.configuration import build_config, parse_args, Config
from apeiron.experiment import Run

from examples.utils import get_example

from apeiron.driver.continuous_monitor import ContinuousMonitor


def main(argv: list[str] | None = None) -> int:
    cfg: Config = build_config(argv)

    # Experiment mode: allocate the bounded run directory and rebind all
    # output paths into it BEFORE the logger is constructed (the CSV path is
    # fixed at logger creation).
    run: Run | None = None
    if cfg.experiment is not None:
        run = Run.create(cfg, original_config=parse_args(argv).config)
        cfg = run.bind(cfg)

    # Must precede get_example(): get_logger() ignores its arguments once an
    # instance exists, so a harness that logs from __init__ would pin the config.
    backend = configure_backend(cfg)
    logger = get_logger(
        verbosity=cfg.verbosity,
        backend=backend,
        csv_path=cfg.logging.metrics_output_path if cfg.logging else None,
    )

    modelHarness = get_example(cfg=cfg)

    # Determine project/experiment name
    project_name = "basesim-framework"
    if cfg.logging and cfg.logging.experiment_name:
        project_name = cfg.logging.experiment_name

    logger.init(cfg, project=project_name)

    # Create continuous monitor - replaces fixed loop and detector instantiation
    monitor = ContinuousMonitor(
        cfg=cfg,
        modelHarness=modelHarness,
        journal=run.journal if run else None,
    )

    # Run continuous monitoring
    monitor.run()

    # TODO: Save a model checkpoint

    if run is not None:
        run.finish()

    logger.finish()

    return 0


if __name__ == "__main__":
    sys.exit(main())
