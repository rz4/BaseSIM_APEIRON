# Experiment runs

Adding an `[experiment]` section to a config puts everything a run reads and
writes into one directory. Without the section, apeiron writes where it always
did and nothing in this page applies.

```toml
[experiment]
path = "experiments/mnist_drift"   # runs are allocated under <path>/runs/
run_name = ""                      # optional label appended to the directory name
```

## Run directory

```
experiments/mnist_drift/
  runs/
    run_0001/
      config.toml            copy of the config file passed to --config
      config.resolved.json   the config that actually ran, after --set and APP_ overrides
      journal.sqlite         event log
      metrics.csv            the [logging] metrics CSV
      log.txt                console output
      signature.txt          hash of the journal, written when the run ends
      checkpoints/           model checkpoints (max_ckpts / ckpts_path)
    run_0002/
      ...
```

Directories are allocated in order and never reused. Allocation uses an
exclusive `mkdir`, so several jobs starting at the same moment each get their
own number rather than colliding.

`[logging] metrics_output_path` and `[model] ckpts_path` are rewritten at
startup to point inside the run directory; their values in the config file are
ignored in experiment mode. `metrics.csv` appears only when the config has a
`[logging]` section.

## Event log

`journal.sqlite` holds one table:

| column | meaning |
|---|---|
| `id` | insertion order |
| `ts` | unix timestamp |
| `kind` | event kind (below) |
| `payload` | JSON object |

Each event is committed as it is written, so the log of a killed run is intact
up to the moment it died.

| kind | payload | when |
|---|---|---|
| `run_started` | `run_dir`, `config_sha256`, `hostname`, `pid` | run directory allocated |
| `window` | `index` | after each `update_data_stream()` |
| `drift_check` | `window`, `batch`, `value`, `detected`, `score` | every drift check, including negative ones |
| `drift` | `event`, `window`, `batch`, `score`, `regime` | drift detected, before training starts |
| `checkpoint` | `event`, `name`, `path` | checkpoint written |
| `run_finished` | `status`, `elapsed_s` | run ends, including on failure |

`config_sha256` is a hash of the whole config except the `[experiment]`
section: two runs with the same value should do the same work and differ only
in where their output lands.

Query it with any SQLite client:

```sql
-- how the monitored metric moved
SELECT json_extract(payload, '$.batch'), json_extract(payload, '$.value')
FROM events WHERE kind = 'drift_check' ORDER BY id;

-- drift events and the checkpoints they produced
SELECT kind, payload FROM events WHERE kind IN ('drift', 'checkpoint') ORDER BY id;
```

or from Python:

```python
from apeiron.experiment import Journal

j = Journal("experiments/mnist_drift/runs/run_0001/journal.sqlite")
print(j.count("drift_check"), "checks,", j.count("drift"), "drift events")
```

## Comparing runs

`signature.txt` is a hash of the event log with the volatile parts removed:
timestamps, host names, process ids and absolute paths are excluded, as are
the `run_started` and `run_finished` events. What remains is what the run did.

Two runs of the same config should produce the same signature, so a rerun can
be checked without reading any metrics:

```bash
diff experiments/mnist_drift/runs/run_0001/signature.txt \
     experiments/mnist_drift/runs/run_0002/signature.txt
```

A difference means the two runs behaved differently. Note that this is a
property of the config and the machine: it holds for a rerun on the same host,
not necessarily across machines with different floating-point behavior.
