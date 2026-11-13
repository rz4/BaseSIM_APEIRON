# Upstream candidates: what to raise and what the change is

A sign-off checklist. Nothing here has been sent; nothing goes out
without explicit approval, per the standing rule. Ledger ids (`A*`) are
for our cross-reference only — issue text, when drafted, justifies each
item from the code with a runnable demonstration and does not mention
the reconstruction methodology.

"Fix on ret/2" means the exact change already exists on
`ret/2-apeiron-regrown` and could accompany the issue as a PR.

---

## A. Straight bug reports — wrong output, one-line-able fix

1. **The reported evaluation loss is wrong by the batch count** (A19).
   `evaluation.test()` sums per-batch *means* and divides by the
   *sample* count: 12 samples in 2 batches reports 0.169 where the true
   mean loss is 1.017 — exactly 6× low, and the error scales with batch
   size, which a loss must not. Second defect in the same function: an
   empty loader returns a bare `0.0` instead of an `(accuracy, loss)`
   pair, so any caller unpacking it crashes.
   **Change:** weight each batch mean by its size (or honor sum
   reduction), and raise on an empty loader. Fix on ret/2.

2. **`grad_accumulation_steps = 0` silently turns every weight NaN**
   (A59). The loss is tensor-divided by the count, so zero produces
   `inf` with no exception; one backward pass later every weight is NaN
   and the run keeps going, filling the metrics file with NaN.
   **Change:** validate the key at config construction
   (`>= 1`, with a message saying why). Fix on ret/2.

3. **A run with no `[logging]` section talks to wandb** (A18). The
   backend default is `"wandb"`, so the first run of an unfamiliar
   config performs network egress nobody asked for.
   **Change:** default to `backend = "none"`. Fix on ret/2.

4. **A misspelled `aggregation` silently becomes the mean** (A20).
   `_aggregate_buffer` recognises mean/median/last and falls back to
   mean for anything else, with no message.
   **Change:** raise `ValueError` on an unrecognised name. Fix on
   ret/2. (Adjacent to issue #145's silent-config theme.)

5. **An unrecognised `verbosity` silently becomes INFO** (A52). A user
   who writes the ordinary abbreviation `"WARN"` to quieten a long run
   gets the *full* info stream — the failure direction is the unhelpful
   one.
   **Change:** warn on (or refuse) an unknown verbosity name. Fix on
   ret/2.

6. **The profiler's warmup silently deletes evaluation rows** (A66,
   located by A65). The `stage("eval")`/`log(...)` pair sits *inside*
   the post-warmup branch of `_evaluate_batch`, so `warmup_iters`
   decides how many evaluations reach the metrics file. Raising it to
   profile more carefully deletes data points, with nothing saying so.
   **Change:** log every evaluation; keep only the profiling inside the
   warmup branch. Fix on ret/2.

7. **A transfer row cannot be joined to the drift event that produced
   it** (A31). The forgetting/transfer numbers a CL paper reports are
   the ones hardest to attribute.
   **Change:** carry the drift event id on the transfer row. Fix on
   ret/2.

8. **A measured block that raises is recorded as a completed zero-FLOP
   measurement** (A50). The CSV then carries `0.0` indistinguishable
   from a block that genuinely cost nothing, and every average over the
   column is pulled toward zero by failures with no trace.
   **Change:** record nothing on exception (the error already
   propagates). Fix on ret/2.

9. **`ModelEvalDetector` prints, asserts, and fabricates its score**
   (A21). `assert` disappears under `python -O`, printing bypasses the
   logger, and the fabricated drift score is logged indistinguishably
   from a measured one.
   **Change:** `ValueError` instead of assert; log through the logger;
   never fabricate. Fix on ret/2. (Adjacent to issue #56.)

10. **Ensemble ties are nondeterministic and a non-string voting name
    leaks an `AttributeError`** (A39, A55). A tied vote reports
    whichever regime iteration order produces;
    `ensemble_voting = 1` dies with
    `'int' object has no attribute 'strip'`.
    **Change:** deterministic tie-break; validate the voting value with
    a clean `ValueError`. Tie-break fix on ret/2.

11. **A missing confidence is logged as the string `"N/A"` in a numeric
    CSV column** (A79) — and a real `0.0` would be ambiguous with it.
    **Change:** leave the cell empty (or add a flag column). Fix on
    ret/2.

12. **`--trigger-at` without `--schedule fixed` dies obscurely** (A16).
    The error names neither flag.
    **Change:** validate the flag combination in the parser, naming
    both.

13. **A dict batch is unpacked into its keys** (A57 set, sharpest
    item). `x, y = batch` on a dict yields two *strings*, and the
    failure surfaces later as `'str' object has no attribute 'to'` — an
    error about strings from a line about devices.
    **Change:** validate the batch shape where it is unpacked and raise
    an error that names the actual problem.

14. **`get_logger(__name__)` passes a module name as the backend
    argument** (found by mypy; latent because the positional is
    currently swallowed). **Change:** `get_logger()`. Fix on ret/2.

15. **Blanket `*.toml` / `*.json` in `.gitignore`** (A4) hides real
    configs from git. **Change:** scope the patterns. Fix on ret/2.

16. **Dependency hygiene** (A6, A7, A8). `EnsembleDetector` hard-imports
    evidently; the evidently guard in the drift tests is leaky; and
    `requires-python >= 3.13` while river 0.23.0 ships no 3.14 wheel,
    so the project is effectively `== 3.13` in a way the metadata does
    not admit.
    **Change:** lazy import with a clear error, a sound test guard, and
    an honest python pin (or a river upgrade path).

17. **Every reported FLOPs figure rests on an unverified lookup table**
    (A25). **Change:** verify the table against counted operations (the
    verification exists on ret/2).

## B. The code is right and nothing protects it — offer tests

18. **The metrics file's schema is entirely unasserted** (A64). A run
    can train, complete, and write a file no downstream analysis can
    read, with every number in it correct. The single highest-value
    test in this list. (Instance of issue #51.)

19. **Nothing constrains which side of the profiler's window
    second-order work happens on** (A73). A reimplementation reported
    8.5× the true K-FAC cost while every other number matched.
    **Change:** a test asserting each `update_mode`'s measured pass
    costs what a vanilla pass costs, stated relative to `base` so it is
    model-independent. Test on ret/2.

20. **The trainer's log-stage naming and the logger's `step` columns
    are unasserted** (A67, A68). `stage` is sticky, so an
    implementation that never sets it writes every training row into
    the drift namespace; the `step`/`{stage}/step` contribution is what
    makes the CSV joinable. Tests only; the original is correct.
    (Instance of #51.)

21. **The `detected` column exists on a random 10% sample of rows**
    (A70), so the file's *schema* varies run to run unless numpy is
    seeded. **Change:** an always-present column carrying the flag —
    small code change plus test. (Instance of #51.)

22. **The monitor's override seams are load-bearing and asserted
    nowhere** (A74, A77, A48): `_log_metrics` arity, the
    `StopIteration` stream-end window, and `_adapt`'s obligation to
    write the checkpoint keyed on the event id it was given — scheduled
    CL never persists otherwise. Both side runners read these seams.
    **Change:** tests that subclass for real. Tests on ret/2.
    (Instance of issue #111.)

23. **The ABC hooks' refusing bodies are unasserted** (A75). The
    original deliberately gives each hook a body that raises (reachable
    by `super()` delegation), but nothing asserts it —
    `get_train_dataloaders` proves abstractness and refusal are
    independent properties. Also: `BaseDriftDetector` has no `__repr__`,
    so a detector prints as an address. Tests (plus one `__repr__`) on
    ret/2.

24. **Profiler contract edges** (A76): warmup must not be measured,
    one block per optimizer step, and an uninspectable batch is
    accepted silently. Tests on ret/2.

## C. Questions for the maintainers — decisions, not defects

25. **The CL loss columns are hard-coded `jvp_reg_*` whatever the
    update mode** (A69). A `kfac_online` run writes its losses into
    columns named for a different updater. Renaming breaks every
    existing run's tooling, so this is a decision, not a fix.

26. **A detector's regime can outvote its own verdict, and the regime
    ladder's evaluation order is unpinned** (A22, A61). With inverted
    thresholds the current code gives the *least* adaptation; a
    different reading gives the most. Decide which is authoritative and
    state the ladder's order. (Adjacent to issue #86.)

27. **`pytest.raises(..., match="Empty loader")` promotes an
    exception's wording to interface** (A78). Either the message is
    load-bearing and belongs in a named constant, or the test should
    match the type and condition.

28. **The package `__init__` welds every layer together** (A1), and the
    two side runners duplicate the main loop (A28) — which is what made
    the override seams load-bearing in two places. Structural; worth an
    issue only if the team wants the discussion.

## D. Comments on existing threads, not new issues

29. **Issue #129 (import-time RNG):** the fix exists on ret/2 — build
    the example model lazily instead of `model = Cnn()` as a default
    argument. Could be offered as a closing PR.
30. **PR #140 (The Well):** `fetch()` writes downloads in place, so a
    truncated transfer poisons the cache for every later run (A80).
    One review comment: download to a temp name, rename on success.
31. **PR #144 (harness fixture):** two further fixture improvements
    beyond the one it fixes — a fixture that never drifts and never
    advances its stream cannot reach the guarded paths (R27).
32. **Issue #51 / #111:** items 18, 20, 21, 22 above are concrete
    instances; could be posted as scoped comments instead of new
    issues if the team prefers fewer threads.

## E. Hold — do not raise

- **Unknown TOML section vs unknown top-level key** (A57 config part):
  an earlier PR asserting the unknown-key behaviour was declined in
  favour of documenting it. Raising either answer reopens a settled
  question. Recorded here, not pushed.
- **Seeds / run reproducibility** (A41): PR #142 already covers it,
  more thoroughly than the finding asks. Nothing to add.
- **`accuracy_topk` untested** (A3, that part): covered by PR #146.
- **Logger delegation freedom** (A27): low value as an upstream
  conversation; internal note only.
