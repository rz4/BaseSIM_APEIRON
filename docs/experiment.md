# Experiment runs

Adding an `[experiment]` section to a config bounds every output of a run
inside a single directory. Without the section, apeiron behaves exactly as
before (outputs written to the working directory).

## Configuration

```toml
[experiment]
path = "experiments/my_experiment"  # experiment workspace directory
run_name = ""                       # optional; auto-numbered when empty
```

Every invocation of `src.main` with this section allocates a new run
directory under `<path>/runs/` (`run_0001`, `run_0002`, ... unless
`run_name` is set) and redirects all outputs into it:

```
experiments/my_experiment/
    runs/
        run_0001/
            config/
                original/<name>.toml   # the config file as given, verbatim
                resolved.json          # full resolved config (after overrides)
            journal.sqlite             # event journal (see below)
            metrics.csv                # metrics CSV (always written)
            checkpoints/               # post-CL checkpoints, if enabled
```

Notes:

- `logging.metrics_output_path` and `model.ckpts_path` are overridden to
  point inside the run directory; other logging settings (backend, names)
  are unchanged. A config with no `[logging]` section still gets a
  `metrics.csv`.
- Checkpointing is enabled the same way as before: set `model.max_ckpts > 0`
  (`ckpts_path` no longer needs to be set).
- `resolved_config.json` is no longer written to the working directory in
  experiment mode; it lives at `config/resolved.json` instead.

## The event journal

`journal.sqlite` is an append-only SQLite log of everything that happened in
the run. One table, `events`, with columns `id`, `ts` (UTC ISO-8601),
`kind`, `batch_count`, and `payload` (JSON).

Event kinds and their payload fields:

| kind | payload |
|---|---|
| `run_started` | `run_name`, `config_sha256`, `original_config` |
| `window_started` | `stream_update_count`, `data_time_start/end` (if the harness exposes them) |
| `drift_check` | `metric`, `score`, `detected`, `regime` — every check, unsampled |
| `drift_detected` | `drift_event_id`, `score`, `regime`, `confidence`, `stream_update_count` |
| `cl_started` | `drift_event_id`, `update_mode`, `pre_cur_metrics`, `pre_hist_metrics` |
| `cl_finished` | `drift_event_id`, `iterations`, `post_cur_metrics`, `post_hist_metrics`, `fwt`, `bwt` |
| `checkpoint_saved` | `drift_event_id`, `path` |
| `run_finished` | `exit_code` |

Unlike the metrics CSV (which subsamples non-detection drift rows), the
journal records every drift check.

### Querying

Any SQLite client works:

```bash
sqlite3 experiments/my_experiment/runs/run_0001/journal.sqlite \
  "SELECT ts, batch_count, payload FROM events WHERE kind='drift_detected'"
```

Or from Python:

```python
from apeiron.experiment import Journal

j = Journal("experiments/my_experiment/runs/run_0001/journal.sqlite")
for event in j.events(kind="cl_finished"):
    print(event["drift_event_id"], event["fwt"], event["bwt"])
```

## Determinism and reruns

`src.main` seeds torch, numpy, and the stdlib RNG from the top-level `seed`
config at startup. Harnesses that additionally derive their per-window
shuffle order from `apeiron.experiment.determinism.window_generator(seed,
window, role)` (as the Well example does) make batch order a pure function
of the config — independent of how much RNG state was consumed earlier in
the run.

Two runs of the same config on the same data then behave identically. To
verify, compare journal signatures — a content hash of the run's behavior
(event order, metrics, drift decisions) that excludes wall-clock
timestamps and allocation-dependent fields like run names:

```python
from apeiron.experiment import Journal

a = Journal("experiments/e/runs/run_0001/journal.sqlite").signature()
b = Journal("experiments/e/runs/run_0002/journal.sqlite").signature()
assert a == b
```

Each `window_started` event also records a `data_fingerprint` when the
harness exposes one (file names + sizes for the Well example), so a rerun
on silently-changed data fails the signature comparison rather than
producing an unexplained divergence.

## Continuing a run

```bash
poetry run python -m src.main --continue-from experiments/e/runs/run_0001
```

`--continue-from` (instead of `--config`) reuses the run's resolved config
(`--set` overrides still apply on top — e.g. raise
`drift_detection.max_stream_updates` to extend a completed run), appends to
its journal, restores the stream/batch/drift-event counters from the
journal tail, loads the newest checkpoint if one was saved, and resumes
monitoring at the last started window.

Limitations (by design, for now):

- Resume granularity is the window: a run that stopped mid-window replays
  that window from its start (the counters keep the detection cadence
  continuous).
- Checkpoints are only written after CL events when `model.max_ckpts > 0`;
  without them, a continue restarts from the pretrained weights.
- Detector state and the transfer-metric task registry are rebuilt fresh,
  not restored: the detector re-warms, and BWT after a continue only spans
  tasks registered since.
- The metrics CSV is rewritten by the continuing process (it reflects the
  latest segment); the journal is the durable, append-only record.
