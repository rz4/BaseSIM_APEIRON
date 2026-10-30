# The regrown package, and what this branch changes on purpose

This branch replaces `src/apeiron/` with an implementation written **blind**
from the executable criteria in `claims/` — the producer never saw the
original — and verified against the original on three surfaces before any
deliberate change was made:

* the pinned MNIST training run: all 31 metric series by digest over their
  `(step, value)` sequences, the trained weights by SHA-256, the drift
  event index, the final accuracy;
* the repository's own test suite (590 of 591, the one failure being an
  exception's wording);
* the shipped examples, cross-compared run-for-run: MNIST and CIFAR
  divergence equal to the original's own run-to-run wall-clock noise, The
  Well identical except one genuine finding (A79, fixed here).

Provenance for every claim below is `claims/LOG.md`, by identifier. The
criteria themselves are merged into `tests/` (prefixed by layer), so the
contracts that forced the equivalence now guard it.

## Deliberate behaviour changes

Everything the reconstruction campaign showed to be wrong in the original,
fixed with the test that keeps it fixed. **The metrics file's schema
changes**: runs from this branch are not row-compatible with runs from
`main` (see A66/A31 below).

| id | before | after |
|---|---|---|
| A19 | `evaluation.test()` summed per-batch mean losses and divided by the sample count: the reported loss was low by exactly the batch count (6x for 12 samples in 2 batches) and scaled with batch size. Empty loader returned a bare `0.0`. | Loss is the mean over samples under either criterion reduction; an empty loader raises. |
| A59 | `grad_accumulation_steps = 0` divided the loss by zero — a tensor division, so no error: `inf` loss, NaN weights, run continues. | Refused at config construction. |
| A18 | A config with no `[logging]` section talked to wandb. | Defaults to `none`; telemetry is opt-in. |
| A20 | A misspelled `aggregation` silently became the mean. | Raises, naming the three options. |
| A52 | `verbosity = "WARN"` (or any typo) silently fell back to INFO — more output than asked, on runs trying to be quiet. | Still falls back (a logging setting should not kill a run), but warns. |
| A21 | `ModelEvalDetector` asserted on mismatched metric counts; asserts vanish under `python -O` and the zip beneath truncated silently. | `ValueError`. (The regrowth had already removed the original's `print` and fabricated drift score.) |
| A50 | A measured block that raised was recorded as a completed zero-FLOP measurement, pulling every average over that tag toward zero with no trace of the failure. | A crashed block records nothing; the exception is the record. |
| A66 | The logging call sat inside the profiler's post-warmup branch, so `warmup_iters` silently decided how many evaluations reached the metrics file — raising it to profile more carefully deleted data points. | Every evaluation is logged; only profiling waits out warmup. The measured window holds the forward pass alone. **Schema change: the first `warmup_iters + 1` eval rows now exist.** |
| A31 | The row carrying `fwt`/`bwt` had no event id and could not be joined to the adaptation it scores. | Carries `drift_event_id`. **Schema change.** |
| #129 | `MNIST_CNN`'s default argument built `Cnn()` at import time, consuming the global RNG before any seed and invalidating the stored first-drift reference. | Built lazily; losses proven identical under both import orders; reference regenerated from values both implementations agree on. |

## Improvements inherited from the reconstruction (no change on this branch)

The blind rebuild made these choices where the original is wrong, and the
pinned run proved they change nothing the framework computes:

* abstract hooks on `BaseModelHarness`/`BaseDriftDetector` **refuse** when
  delegated to instead of returning `None` that fails far away (A75);
* detectors print as `ADWINDetector(name='ADWIN')`, not a memory address
  that differs between processes (A75);
* a `None` confidence reaches the CSV as an empty cell (clean NaN), not
  the literal string `"N/A"` in a numeric column (A79);
* the ensemble's tied vote resolves to the most severe firing regime,
  deterministically (A39); a non-string `ensemble_voting` and degenerate
  evaluation inputs get named `ValueError`s (A55, A57);
* unknown TOML sections are rejected; an unknown top-level key names the
  key rather than blaming `TrainCfg.__init__` (A57);
* the monitor's three subclass seams (`_log_metrics` arity, the
  `StopIteration` window protocol, `_adapt` writing the checkpoint keyed
  on its argument) are honoured and tested with a real subclass (A74, A77).

## Deliberately NOT changed

* **A69's column renames** (`jvp_reg_total_loss` is written whatever
  `update_mode` is). Renaming breaks the join key of every existing run;
  that trade is the team's, not this branch's.
* **A78's exception wording** upstream test: the assertion now matches the
  condition, not the sentence.
* **A22/A61** (regime-vs-verdict authority, ladder order): behaviour
  questions, recorded, not answered unilaterally.

## Housekeeping found by the tooling

* The trainer called `get_logger(__name__)`, passing a module name as the
  *backend* positional — harmless only because a logger always exists
  first. mypy caught it once the package was annotated; it now fetches the
  singleton with no arguments. The package is `ruff` and `mypy` clean.
* Every test now gets a fresh logger singleton; several criteria replace
  methods on the `get_logger()` instance, and the crippled singleton used
  to leak into whichever test ran next.

## What this branch does not claim

Equivalence beyond the verified surfaces. The resume path is unprobed
(issue #136 names a real defect there that no criterion here would catch),
the final assembly is one producer family, and off-surface behaviour is
measured to differ in general (A46). The verified surfaces are exactly:
the pinned run, this suite, and the three examples above.

## Verified against the examples

Each shipped example was run on this branch and on the original package
with identical settings and seeds, and every series classified. The
branch's entire deviation from `main` behaviour, across all three, is
exactly the changes declared above — nothing unexplained:

| example | identical | A66 (+2 warmup rows) | A31 (event ids) | A79 (`N/A` → empty) | unexplained |
|---|---|---|---|---|---|
| MNIST (2 seeds) | 28 series, values to the digit | `eval/accuracy`, `eval/loss` | `eval/drift_event_id` | n/a (ADWIN carries one) | **0** |
| CIFAR-10 VGG-11 | 8 series | same two | — | n/a | **0** |
| The Well (PR #140) | 6 series, incl. every VRMSE and drift score | `eval/mse`, `eval/vrmse` | — | `drift/confidence`: 19 string `N/A`s become empty cells | **0** |

(Wall-clock `_time`/`_flops` series differ between any two runs of either
implementation and are excluded. `eval/step`, a position-valued column,
shifts by the two warmup rows around each validation annotation — the
same annotation at the same batch, two rows later in the stage.)

Drift fires at the same checks, final accuracies match to the digit, and
every training value — losses, scores, accuracies — is byte-identical to
the original implementation's. The fixes removed defects; they did not
move the model.
