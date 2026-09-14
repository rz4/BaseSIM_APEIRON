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

## Managed data residency (artifact store)

Each experiment workspace can hold a shared, disk-tier cache of remote
data, used by all its runs:

```
<experiment path>/
    artifacts/
        index.sqlite    # what is resident: identity, size, sha256, usage, pins
        store/<...>     # the materialized files
    runs/...
```

Harnesses that support it (the Well example does) materialize each stream
window's files from the remote reservoir on demand, **pin** them for the
duration of the run (pinned files are never evicted; pins are released
when the run finishes), and **prefetch** the next window's files in the
background while the current window is being processed.

```toml
[experiment]
path = "experiments/my_experiment"
artifact_budget_bytes = 20_000_000_000  # soft 20 GB cap; 0 = unlimited
```

When materializing would exceed the budget, least-recently-used unpinned
artifacts are evicted first. The budget is soft: pinned artifacts are
never evicted, so if a run's pins alone exceed it the store runs over
with a warning rather than failing the run.

For the Well example: with an `[experiment]` section and
`data.path = "hf://datasets/polymathic-ai/"`, no manual download is
needed — regime files land in the store on first use and later runs of
the same experiment reuse them. Without an `[experiment]` section the
hf:// path streams remotely and local paths are read directly, as before.

### Declaring windows from a harness

A harness opts into managed residency by declaring each stream window's
data needs instead of fetching data itself:

```python
from apeiron.experiment.sources import WindowSpec

class MyHarness(BaseModelHarness):
    def describe_window(self, window: int) -> WindowSpec | None:
        if window >= self.n_windows:
            return None  # past the end; also stops prefetch
        return WindowSpec(
            objects=tuple(self._window_objects(window)),  # RemoteObjects
            fingerprint=self._window_fingerprint(window),
            label=self._window_name(window),
        )
```

Before each `update_data_stream()`, the monitor materializes the declared
objects (pinned for the run) and afterwards prefetches the next window's
objects in the background. The framework injects `self.data_resolver`;
the harness reads the materialized files under
`self.data_resolver.store_root` and never touches stores, pins, budgets,
or threads. Harnesses that return `None` (the default) keep fetching
their own data — nothing changes for them.

Each materialization is journaled as a `window_materialized` event with
transfer accounting (`fetched_bytes`, `hit_bytes`, `evicted_bytes`,
`seconds`). These events record cache state, not behavior, so they are
excluded from journal signatures (a cold and a warm rerun still compare
equal).

Harnesses whose loaders yield dict batches (rather than `(x, y)` tuples)
should also override `batch_to_device` / `batch_size_of` if the defaults
don't fit; the trainer routes all batch handling through these hooks.

## Checkpoint retention policies

With checkpointing enabled (`model.max_ckpts > 0`), the retention rule
decides which post-CL snapshots survive the cap:

```toml
[model]
max_ckpts = 3
ckpt_retention = "best_hist"  # "latest" (default) | "best_current" | "best_hist"
```

- `latest` — newest N (the historical FIFO behavior).
- `best_current` — best first-metric score on the window that triggered
  each event (`cl_finished.post_cur_metrics[0]` in the journal).
- `best_hist` — best first-metric score on the historical validation data.

Regardless of policy, the newest checkpoint always survives (it is what
`--continue-from` needs to match the stream position) and the `latest`
pointer names it. Metric direction follows the harness's
`higher_is_better` for its first metric. Metric-based policies read
scores from the run journal, so they need experiment mode; without it
they fall back to `latest` with a warning. Evictions are journaled as
`checkpoints_evicted` events.

## The experiment workspace

Summarize an experiment's runs (and its artifact store) from their
journals:

```bash
python -m apeiron.experiment experiments/my_experiment            # report
python -m apeiron.experiment experiments/my_experiment --gc-pins  # + stale-pin GC
```

`--gc-pins` releases artifact pins held by runs that already finished or
whose directory is gone (a crashed run never reaches its pin-release
step, and stale pins block eviction). The same view is available in
Python via `apeiron.experiment.Experiment`.

Control-arm runs (`src.cl_only`) accept the same `[experiment]` section
and land in the same workspace as the detector runs they are compared
against — same determinism contract, same managed data residency (their
declared windows are materialized up front, since a schedule run visits
every window). Their journals carry a final `schedule_summary` event.

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
