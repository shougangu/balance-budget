# Parallel Post-Training Workers

## Problem

The unified early pipeline runs post-training (DPO/GRPO) checkpoints sequentially. SFT
produces N checkpoints, and each is processed one at a time. With `--parallel K`, we want
K checkpoints training concurrently, each in its own Slurm GPU allocation.

## Design

### Metadata State

Checkpoint rows in the JSONL metadata file gain a `"claimed"` field:

| `claimed` | `completed` | Meaning |
|-----------|-------------|---------|
| false     | false       | Available for pickup |
| true      | false       | A worker is training on this |
| true      | true        | Done |

Existing metadata files without `"claimed"` are treated as `claimed: false` via
`.get("claimed", False)`.

### New function: `claim_next_checkpoint(metadata_file)`

Reads the JSONL metadata file. Finds the first row where both `claimed` and `completed`
are false. Writes `"claimed": true` back to the file. Returns the row, or `None` if
nothing is available.

No file locking. The race window (two workers claiming the same checkpoint) requires
near-simultaneous Slurm job starts and overlapping file I/O — negligible in practice.
Worst case: one checkpoint trained twice. Locking can be added later if needed.

### Worker changes (`run_dpo` / `run_grpo`)

Replace `next_checkpoint()` with `claim_next_checkpoint()`. When no checkpoint is
available to claim, exit with code 42 (special "no work" code, distinct from errors).

### Post-training loop

The outer subprocess loop changes from:

```python
# Before
while next_checkpoint(metadata_file) is not None:
    subprocess.run(pt_cmd)
```

To:

```python
# After
while True:
    result = subprocess.run(pt_cmd)
    if result.returncode == 42:
        break
    if result.returncode != 0:
        sys.exit(f"Worker failed with return code {result.returncode}")
```

Each subprocess claims its own checkpoint. GPU memory freed between iterations via
process exit.

### New CLI arg: `--parallel N`

Default: 1 (current sequential behavior).

### Orchestrator flow (`--parallel N > 1`)

1. Run SFT (unchanged — single subprocess, produces metadata files).
2. Submit `N - 1` sbatch worker jobs. Each worker runs the same shell wrapper
   (`unified_early_pipeline.sh`) with args:
   `--run-dpo --run-all --metadata-file X [Y Z ...]`
   This enters orchestrator mode, skips SFT (because `--run-dpo` is set), and runs the
   post-training loop with checkpoint claiming.
3. Orchestrator itself runs the post-training loop as the Nth worker (uses its own GPU).
4. Orchestrator exits when its loop finishes (no more unclaimed checkpoints). It does
   not wait for other workers — fire and forget.
5. The shell wrapper runs the sweetspot table on exit. Each worker runs this too. The
   last worker to finish produces the complete sweetspot table from W&B. Earlier runs
   may be incomplete; that is acceptable.

### Sbatch worker submission

Reuses the existing `tuning/slurm/unified_early_pipeline.sh` script — no new shell
script. The orchestrator submits workers from Python:

```python
result = subprocess.run(
    ["sbatch", "tuning/slurm/unified_early_pipeline.sh"] + worker_args,
    capture_output=True, text=True,
)
# stdout: "Submitted batch job 12345"
```

Worker args include `--run-dpo --run-all --metadata-file ...` plus whatever flags the
orchestrator was invoked with (minus `--parallel`, which is stripped to prevent workers
from spawning more workers).

### Error handling

- **Worker crash**: Checkpoint stays `claimed: true, completed: false`. Manual reset
  needed (edit the metadata JSONL). Stale-claim detection can be added later.
- **Worker failure**: Independent — orchestrator doesn't track worker exit codes. Failed
  sbatch jobs show up in `squeue`/`sacct` as usual.
- **`--parallel 1`**: Unchanged behavior. No sbatch submission, sequential loop.

### GPU utilization

The orchestrator's GPU allocation is used for SFT, then for post-training (as the Nth
worker). Zero idle GPU time.

### Files changed

| File | Change |
|------|--------|
| `tuning/training/unified_early_pipeline.py` | Add `claim_next_checkpoint()`, `--parallel` arg, sbatch dispatch, update `run_dpo`/`run_grpo`/post-training loop |
