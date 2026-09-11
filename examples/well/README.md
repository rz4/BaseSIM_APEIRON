# The Well example

Runs apeiron's drift-detection + continual-learning loop over a dataset from
[The Well](https://polymathic-ai.org/the_well/) (PolymathicAI), using The
Well's pretrained FNO baseline as the monitored model.

## How it maps onto apeiron

- **Stream**: one window per HDF5 file of the chosen dataset, ordered by
  filename. Well filenames encode a physical parameter sweep (for
  `turbulent_radiative_layer_2D`: cooling time `tcool` 0.03 → 3.16, 9 files),
  so the stream walks the parameter continuum and drift is real regime change.
- **Model**: `the_well` FNO baseline, loaded from the Hugging Face checkpoint
  named by `[model] pretrained_path`. Task: predict the next timestep's fields
  from the previous 4 (channels-first, time stacked into channels).
- **Metrics**: `vrmse` (The Well's headline metric, lower is better) and MSE
  `loss`. `[drift_detection] metric_index = 0` monitors VRMSE.
- **Replay**: `get_hist_dataloaders()` covers all prior regimes;
  `mix_historic_data = true` blends them into each CL step.

## Requirements

`the_well` (installed with the repo's dependencies). Data access is either:

- **Streaming (no download)** — default config:
  `path = "hf://datasets/polymathic-ai/"`. Fine for smoke tests; slow for
  real runs (every batch is fetched from Hugging Face).
- **Local download** (recommended for real runs):

  ```bash
  the-well-download --base-path ./data/the_well --dataset turbulent_radiative_layer_2D
  ```

  then set `path = "./data/the_well"` in the TOML.

## Running

Full run (9 regimes):

```bash
poetry run python -m src.main --config examples/well/well_trl2d.toml
```

Quick smoke test (2 regimes, tiny CL loop, streaming):

```bash
poetry run python -m src.main --config examples/well/well_trl2d.toml \
  --set drift_detection.max_stream_updates=2 \
  --set drift_detection.detection_interval=3 \
  --set train.max_iter=2
```

Metrics are written to `[logging] metrics_output_path`
(default `output/well_trl2d.csv`).

## Notes

- `device = "cpu"` in the shipped config: the FNO uses complex FFTs, which
  Apple MPS does not support. Set `cuda` on NVIDIA machines.
- The pretrained baselines were trained across the full parameter sweep, so
  early windows are in-distribution; VRMSE still varies strongly by regime,
  which is the signal the detector monitors.
- Other Well datasets: change `data.name` to `well:<dataset>` and
  `pretrained_path` to the matching `polymathic-ai/FNO-<dataset>` checkpoint.
  Datasets whose checkpoints use different input depths would need
  `N_STEPS_INPUT`/`N_STEPS_OUTPUT` adjusted in `examples/well/model.py`.
