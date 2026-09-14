import signal
import sys

import torch

from apeiron.logger import get_logger, configure_backend
from apeiron.config.configuration import (
    build_config,
    config_from_dict,
    deep_update,
    kv_to_nested,
    parse_args,
    Config,
)
from apeiron.experiment import Run
from apeiron.experiment.determinism import seed_everything
from apeiron.experiment.residency import DataResolver

from examples.utils import get_example

from apeiron.driver.continuous_monitor import ContinuousMonitor


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run: Run | None = None
    resume: dict | None = None

    if args.continue_from is not None:
        # Continue an existing run: reuse its resolved config (with --set
        # overrides on top), append to its journal, restore counters, and
        # pick the stream up at the last started window.
        run = Run.open(args.continue_from)
        raw = deep_update(run.resolved_config(), kv_to_nested(args.set))
        cfg: Config = config_from_dict(raw)
        resume = run.journal.resume_state()
        run.journal.record("run_continued", **resume)
        cfg = run.bind(cfg)
    else:
        cfg = build_config(argv)

        # Experiment mode: allocate the bounded run directory and rebind all
        # output paths into it BEFORE the logger is constructed (the CSV path
        # is fixed at logger creation).
        if cfg.experiment is not None:
            run = Run.create(cfg, original_config=args.config)
            cfg = run.bind(cfg)

    # One seed in, reproducible run out (see docs/experiment.md).
    seed_everything(cfg.seed)

    # Must precede get_example(): get_logger() ignores its arguments once an
    # instance exists, so a harness that logs from __init__ would pin the config.
    backend = configure_backend(cfg)
    logger = get_logger(
        verbosity=cfg.verbosity,
        backend=backend,
        csv_path=cfg.logging.metrics_output_path if cfg.logging else None,
    )

    modelHarness = get_example(cfg=cfg)

    # Experiment mode: hand the harness the framework-owned data resolver.
    # Harnesses that declare their windows (describe_window) get managed
    # residency -- materialization, pinning, prefetch -- for free.
    if run is not None:
        modelHarness.data_resolver = DataResolver.for_run(cfg, run)

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

    if resume is not None and run is not None:
        snapshot = run.latest_snapshot(map_location=cfg.device)
        if snapshot is not None:
            # Full-state resilience snapshot: exact resume (weights, optimizer,
            # RNG, detector, counters, mid-window/mid-CL position).
            monitor.restore_snapshot(snapshot)
        else:
            # No snapshot: window-boundary resume from journal counters plus
            # the newest analysis checkpoint, if any.
            monitor.stream_update_count = resume["stream_update_count"]
            monitor.batch_count = resume["batch_count"]
            monitor.drift_event_count = resume["drift_event_count"]
            ckpt = run.latest_checkpoint
            if ckpt is not None:
                # Default checkpoint payload is model.state_dict(); harnesses
                # overriding build_checkpoint_payload() load their own format.
                state = torch.load(ckpt, map_location=cfg.device, weights_only=False)
                modelHarness.model.load_state_dict(state)
                logger.info(f"Continuing from checkpoint: {ckpt}", level=0)
            else:
                logger.warning(
                    "Continuing WITHOUT any checkpoint: no resilience snapshot "
                    "and no analysis checkpoint found -- the model restarts "
                    "from its pretrained weights, which is NOT a faithful "
                    "continuation. Enable [experiment] snapshot_interval or "
                    "model.max_ckpts to avoid this."
                )

    # Walltime resilience: Slurm-style pre-kill warnings (--signal=USR1@300)
    # and polite terminations request a snapshot + clean exit at the next
    # update boundary. Test locally with: kill -USR1 <pid>
    def _request_interrupt(signum, frame):  # noqa: ARG001
        monitor.interrupt_requested = True

    signal.signal(signal.SIGUSR1, _request_interrupt)
    signal.signal(signal.SIGTERM, _request_interrupt)

    # Run continuous monitoring
    try:
        monitor.run()
    except SystemExit:
        # Interrupt path: snapshot + run_interrupted already recorded; flush
        # the metrics CSV and exit without run_finished (the run is not done).
        logger.finish()
        raise

    # TODO: Save a model checkpoint

    if run is not None:
        run.finish()

    logger.finish()

    return 0


if __name__ == "__main__":
    sys.exit(main())
