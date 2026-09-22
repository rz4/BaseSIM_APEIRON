# The Well: continual learning under real regime change

[The Well](https://polymathic-ai.org/the_well/) is a collection of physics
simulation datasets from PolymathicAI. Each one is a parameter sweep, which is
what makes it useful here: the drift is not simulated by perturbing inputs, it
is what happens when the physics moves.

This example uses `turbulent_radiative_layer_2D` — nine simulations of a shear
layer between hot and cold gas, varying the cooling time `tcool` from 0.03 to
3.16. One window per regime, walked in parameter order, so the model is carried
across a physical continuum and asked to keep up.

```bash
poetry run python -m src.main --config examples/well/well_trl2d.toml
```

## What a run does

| | |
|---|---|
| window | one regime: 8 training trajectories, 101 time steps, 128×384 |
| sample | 4 consecutive frames in, the next frame out |
| channels | density, pressure, velocity x, velocity y |
| model | a small FNO (`fno.py`), ~67k parameters |
| monitored metric | VRMSE, The Well's headline metric — lower is better |

Each regime is a ~650 MB file, fetched when the run reaches it and converted
once into a memory-mappable array. Budget roughly **1.3 GB of disk per
regime**: the source and the converted copy. The conversion does not save
space — these files are already uncompressed — it buys random access into a
window without reading it through.

Nothing is downloaded until a window needs it, so a run that stops after three
regimes has moved three regimes' worth of bytes. The file listing comes from
the HuggingFace API, so the nine regimes are known before anything is fetched.

## Trying it without 6 GB of downloads

Stop after the first window:

```bash
poetry run python -m src.main --config examples/well/well_trl2d.toml \
  --set drift_detection.max_stream_updates=1
```

Or point at a copy you already have:

```toml
[data]
path = "data/the_well"     # holds <path>/turbulent_radiative_layer_2D/...
```

Local files are copied into the store rather than read in place, so a run never
depends on a directory outside itself.

You can also fill the store by hand. It is laid out by what the files are, not
where they came from:

```
output/well_trl2d/datasets/
  stats.yaml
  train/turbulent_radiative_layer_tcool_0.03.hdf5
  train/turbulent_radiative_layer_tcool_0.03.npy        converted
  train/turbulent_radiative_layer_tcool_0.03.npy.origin the recipe that made it
  valid/turbulent_radiative_layer_tcool_0.03.hdf5
```

Stage the `.hdf5` files with any transfer tool and the run finds them already
there — useful where compute nodes should not be reaching out to the network.
Stage the sources, not the conversions: an artifact without a matching recipe
marker is rebuilt rather than trusted.

Set `[data] memmap = false` to skip the conversion and read HDF5 directly. The
harness produces identical samples either way.

## Being interrupted

Nine regimes take hours. Set a save interval and a killed run continues:

```toml
[experiment]
restart_interval = 200      # batches between saves
```

```bash
poetry run python -m src.main --continue-from output/well_trl2d/run_0001
```

Under a scheduler, `#SBATCH --signal=USR1@300` makes the job save and exit
cleanly before the walltime kill. An interrupted run resumed produces the same
journal signature as one that ran straight through; that is checked on this
example, not only in tests.

## Normalisation

Per-channel statistics come from the dataset's own `stats.yaml` and are held
fixed across regimes. Normalising per regime would divide out the very shift
the run is watching for.

## The model

`fno.py` is a small self-contained Fourier Neural Operator: lift to a wider
channel space, mix the lowest Fourier modes with a learned complex weight,
project back. Its parameters live on modes rather than pixels, so the same
weights apply at any resolution.

It is deliberately modest. The Well publishes pretrained FNO baselines that are
larger and better trained; swapping one in is a change to `WELL_FNO._build_model`
alone. The run's `model.json` records the architecture and a hash of its tensor
shapes, so a run using one model can never be quietly compared with a run using
another.

## Other Well datasets

Any dataset whose regimes are separate files under `data/train` works by
changing one line:

```toml
[data]
name = "well:active_matter"
```

Two-dimensional datasets with the same field layout work as-is. Others need the
`FIELDS` tuple in `model.py` adjusted to name their fields, and 3D datasets need
an FNO with a third spectral dimension.
