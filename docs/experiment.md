# Experiment runs

Adding an `[experiment]` section to a config puts everything a run reads and
writes into one directory. Without the section, apeiron writes where it always
did and nothing in this page applies.

```toml
[experiment]
```

The section on its own is enough; every key has a default.

```toml
[experiment]
path = "output"          # where experiments live
name = "mnist_drift"     # this experiment; runs land in <path>/<name>/
run_name = ""            # optional label appended to a run's directory name
restart_interval = 0     # save restart state every N batches; 0 = only on signal
```

`path` is relative to the working directory, so on a shared machine set it to
scratch space rather than relying on the default. `name` groups the runs of one
experiment together; leaving it empty puts runs directly under `path`.

## Run directory

```
output/
  mnist_drift/
    run_0001/
      config.toml            copy of the config file passed to --config
      config.resolved.json   the config that actually ran, after --set and APP_ overrides
      model.json             what model was used
      journal.sqlite         event log
      metrics.csv            the [logging] metrics CSV
      log.txt                console output
      signature.txt          hash of the journal, written when the run ends
      checkpoints/           model checkpoints (max_ckpts / ckpts_path)
      restart/               state for resuming; removed when the run completes
    run_0002/
      ...
  cifar_drift/
    run_0001/
      ...
```

Runs are numbered in order within an experiment and never reused; a new
experiment starts again at `run_0001`. Anything already in the directory that is
not named `run_NNNN` is ignored, so pointing at a directory with other contents
is fine. Allocation uses an exclusive `mkdir`, so several jobs starting at the
same moment each get their own number rather than colliding.

`[logging] metrics_output_path` and `[model] ckpts_path` are rewritten at
startup to point inside the run directory; their values in the config file are
ignored in experiment mode. `metrics.csv` appears only when the config has a
`[logging]` section.

## Model record

`model.json` records what the config cannot: the architecture lives in the
harness source, so `[model] name = "dummy"` and a path to weights is not enough
to reload a checkpoint later or to tell two runs apart.

Most of it is derived from the model with no cooperation from the harness:

| field | meaning |
|---|---|
| `class`, `module` | the model class, with `DataParallel`/DDP unwrapped |
| `harness_class`, `harness_module` | the harness that built it |
| `parameters` | total and trainable parameter counts |
| `tensors` | every state-dict entry: name, shape, dtype |
| `shapes_sha256` | hash of that table -- equal hashes load each other's weights |
| `pretrained` | path, size and mtime of the weights file, if any |
| `config` | whatever the harness returns from `model_config()` |

The framework can see a network's shape but not the choices behind it. A
harness that constructs its model from arguments should return them:

```python
class WELL_FNO(BaseModelHarness):
    def model_config(self) -> dict:
        return {"modes": 16, "width": 64, "n_steps_input": 4}
```

The default returns `{}`, so existing harnesses need no change.

Everything except the `tensors` table is also written to the event log, which
means the architecture takes part in the run signature: a change to harness
code that alters the network compares as a different run even though the config
file is untouched.

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
| `model` | the model record, without the `tensors` table | harness built |
| `restart_saved` | `reason`, `batch`, `name` | restart state written |
| `run_continued` | `batch` | resumed with `--continue-from` |
| `run_interrupted` | `batch` | stopped on signal with state saved |
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

j = Journal("output/mnist_drift/run_0001/journal.sqlite")
print(j.count("drift_check"), "checks,", j.count("drift"), "drift events")
```

## Continuing a killed run

Jobs get killed: walltime limits, node failures, a laptop closing. A run saves
everything needed to carry on into `restart/`, and

```bash
poetry run python -m src.main --continue-from output/mnist_drift/run_0001
```

picks up from the newest saved state. `--continue-from` reads that run's
`config.resolved.json`, so no other arguments are needed -- and `--set`
overrides are deliberately not applied, since changing the config mid-run is
not a resume.

### When state is saved

At batch boundaries in the monitoring loop, either every `restart_interval`
batches or when the process is signalled. **Never inside a training round**: a
round is treated as one unit, so a run killed during training replays that
round from its start. The cost is bounded by one round; the benefit is that
training needs no resume logic at all.

With `restart_interval = 0` state is saved only on a signal, which survives a
walltime kill but not an abrupt one. Set an interval to survive both.

### Stopping on a signal

`SIGUSR1` and `SIGTERM` ask the run to save at the next batch boundary and exit
0. Under Slurm that is:

```bash
#SBATCH --signal=USR1@300
```

which delivers the signal five minutes before the walltime kill, leaving time
to write state and exit cleanly. The run records `run_interrupted` and finishes
with status `interrupted`.

### What is saved

Model weights, optimizer state, every random generator, the drift detector, the
updater's memory across drift events (EWC's Fisher and anchor, KFAC's factors),
the loop's counters including the partially filled metric buffer, and the
position in the event log.

Per-round accumulators are not saved, because the round replays.

### Correctness

Two things are checked before a state is loaded, because both fail later and
less clearly otherwise: the config hash, and the model's shape hash. Resuming
into a differently shaped model -- the harness source changed underneath -- is
refused with a message saying so.

The event log is committed per event, so it runs ahead of state written every N
batches. On resume the events past the saved point are dropped and the replayed
batches re-create them. Without this, a resumed run would double-count.

A completed run has nothing to resume from, so `restart/` is deleted when it
finishes. A failed or interrupted run keeps it.

### The bar

A run interrupted and resumed produces **the same signature** as a run that was
never interrupted. That is what the tests assert, and it is checkable by hand:

```bash
diff output/mnist_drift/run_0001/signature.txt \
     output/mnist_drift/run_0002/signature.txt
```

## Comparing runs

`signature.txt` is a hash of the event log with the volatile parts removed:
timestamps, host names, process ids and absolute paths are excluded, as are
the `run_started` and `run_finished` events. What remains is what the run did.

Two runs of the same config should produce the same signature -- the seed is
applied for real in experiment mode, which it is not in legacy mode -- so a
rerun can be checked without reading any metrics:

```bash
diff output/mnist_drift/run_0001/signature.txt \
     output/mnist_drift/run_0002/signature.txt
```

A difference means the two runs behaved differently. Note that this is a
property of the config and the machine: it holds for a rerun on the same host,
not necessarily across machines with different floating-point behavior.
