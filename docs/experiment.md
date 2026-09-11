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
