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
datasets_path = ""       # where data is kept; default <path>/<name>/datasets
```

`path` is relative to the working directory, so on a shared machine set it to
scratch space rather than relying on the default. `name` groups the runs of one
experiment together; leaving it empty puts runs directly under `path`.

## Run directory

```
output/
  mnist_drift/
    datasets/              data this experiment uses, shared by every run
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
| `window` | `index`, `inputs` | after each `update_data_stream()` |
| `dataset` | `wanted`, `present`, `fetched`, `built`, `bytes_fetched`, `seconds` | window data made ready |
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

## Data

A harness can leave its data alone and apeiron will not interfere. A harness
that declares what each window needs gets it put in place first:

```python
class WELL_FNO(BaseModelHarness):
    def window_inputs(self, window: int) -> dict[str, str]:
        """name in the store -> where to get it"""
        f = self.files[window]
        return {f"train/{f}": f"hf://datasets/polymathic-ai/{self.dataset}/data/train/{f}"}

    def update_data_stream(self) -> None:
        # everything window_inputs declared is on disk by the time this runs
        path = self.datasets.path_for(f"train/{self.files[self.window]}")
```

The default returns `{}`, so existing harnesses need no change.

Declaring per window rather than per run is the point. The Well is fifteen
terabytes; a run that touches three regimes should move three regimes' worth of
bytes, not the dataset.

### Two routes in

| source | what happens |
|---|---|
| a local path, or `file://...` | copied in -- an example can ship raw data beside its config |
| `hf://datasets/owner/repo/path` | downloaded when a window first needs it |
| `https://...` | the same |
| a `Derived(...)` | produced from the others, once (see below) |

Both end in the same place with the same layout, so nothing downstream can tell
which route filled it. `HF_TOKEN` is sent to huggingface.co when set, and to
nowhere else.

### The store

```
output/mnist_drift/datasets/
  train/file_0.hdf5
  train/file_0.npy                 <- derived, memory-mappable
  train/file_0.npy.origin          <- the recipe that produced it
  train/file_1.hdf5.part.48213     <- in progress; .part.* is always garbage
```

Files are written to a neighbouring `.part` name and renamed into place, so a
file that exists is a file that finished -- there is no index to consult and no
partial file to mistake for a whole one. Two processes fetching the same file
use different `.part` names and both land a complete file.

Nothing is ever removed. The copy is kept and reused: the second run of an
experiment does no I/O at all, and the `dataset` event records the difference.

Because the layout is derived from the names the harness declares and not from
where the bytes came from, **staging by hand works**. Copy files into the store
with whatever transfer tool you like and the run finds them already there --
which is what you want on a machine whose compute nodes should not be reaching
out to the network.

`datasets_path` defaults to a `datasets` directory beside the experiment's
runs, so data travels with the experiment. Point it at shared scratch to have
several experiments share one copy.

### Memory-mapped data

A window that does not fit in memory should be paged in by the operating
system rather than read through the process. That needs a file whose bytes
*are* the array, and HDF5 generally is not one -- chunked, often compressed,
reached through a library. So the mappable file is derived: converted once out
of what was fetched, then reused forever, on the same terms as a download.

Declare it alongside the source it is built from:

```python
from apeiron.experiment import Derived, memmap

def window_inputs(self, window: int) -> dict:
    source = f"train/file_{window}.hdf5"
    return {
        source: f"hf://datasets/polymathic-ai/{self.dataset}/data/{source}",
        f"train/file_{window}.npy": Derived(
            build=lambda dest: self._convert(source, dest),
            recipe="hdf5->f32:chw:v1",
        ),
    }

def _convert(self, source: str, dest: Path) -> None:
    with h5py.File(self.datasets.path_for(source)) as h5:
        frames = h5["t0_fields/density"]
        with memmap.writing(dest, frames.shape, "float32") as out:
            for i in range(frames.shape[0]):
                out[i] = frames[i]          # goes to disk, not to RAM
```

and read it back as a view:

```python
frames = memmap.read(self.datasets.path_for(f"train/file_{w}.npy"))
batch = torch.from_numpy(np.ascontiguousarray(frames[a:b]))
```

Nothing is read until a page is touched, and only the slice is copied.

Fetched files always land before derived ones are built, so a conversion can
read the sources declared beside it.

The format is `.npy`, which is not a choice worth making twice: the header is
self-describing, the array is contiguous after it, numpy maps it with one call,
and it is a single file, so it lands with the same atomic rename as everything
else.

**The `recipe` string is the part that matters.** A converted file is only
valid for the conversion that produced it -- change the dtype, the layout, or
which fields you keep, and the old file is wrong but still sitting there. The
recipe is written to a `.origin` marker beside the artifact and compared before
the file is used; a mismatch rebuilds. Put anything that would change the bytes
into it. This is the same guard the restart path applies with the model's shape
hash, for the same reason: silently using the wrong bytes is worse than an
error.

An artifact with no marker is not trusted either, so a file dropped into the
store by hand is rebuilt rather than assumed current. That is the one place
hand-staging does not apply -- stage the sources, not the conversions.

### What is recorded

The `window` event carries the names the window declared, so the log says what
data each window used. The `dataset` event carries the accounting -- how many
files were already present, how many were fetched, how many bytes, how long --
and is left out of the signature, because whether a file had to be downloaded
is a fact about the machine rather than about what the run computed. A cold run
and a warm run of the same config compare equal.

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
